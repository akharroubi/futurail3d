"""Release acceptance test of the built Rail 3D ZIP in fresh QGIS 3 and 4 profiles.

    python tests/e2e_qgis_process.py [--zip dist/Rail_3D_v1.0.0.zip] [--input <LAS/LAZ>]
        [--work <scratch parent>] [--qgis3 BAT] [--qgis4 BAT]

Both runtimes are required. Each run gets a new private directory below
--work; existing files and profiles are never removed. The ZIP is checked
(one top folder Rail3D, the weights inside it with the registry's SHA-256,
no tests or point clouds), then installed into a fresh QGIS profile and run
through qgis_process on CUDA: default codes in memory on QGIS 3, custom
codes (OUTPUT_CODES_JSON) in streaming mode on QGIS 4, both through
rail3d:classify_point_cloud (no classifier or model argument exists). The
weights ship inside the ZIP, so no download may happen: the profile's model
URL is pointed at a closed local port, the log must not mention a download
and the profile's Rail3D/models folder must stay without files. The output
files are checked for integrity (point count, attributes, header, VLRs,
provenance VLR user_id Rail3D) and the class IDs of both runtimes compared.

The plugin environment comes from RAIL3D_CACHE_DIR (an existing install,
else ~/.qgis_rail3d). Reports and logs remain in the printed work directory.
Any missing prerequisite, integrity failure or missing run returns exit 1.
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess  # nosec B404 - developer tool, not shipped in the plugin ZIP
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import laspy
import numpy as np

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parent
NAME = "Rail3D"
MODEL_ID = "rail3d_5cm"
WEIGHTS = "rail3d_5cm_fp16.pth"
ALGORITHM = "rail3d:classify_point_cloud"
DEFAULT_CODES = {1: 2, 2: 10, 3: 15, 4: 14, 5: 5, 6: 64, 7: 1}
CUSTOM_CODES = {**DEFAULT_CODES, 2: 200, 6: 17, 7: 255}
MIN_AGREEMENT = 0.9995
# A closed port: any download attempt fails at once instead of reaching the
# public fallback URL, so a run that needs the network cannot pass.
NO_NETWORK_URL = "http://127.0.0.1:9/no-download-expected.pth"
VALIDATION = Path(os.environ.get("RAIL3D_VALIDATION")
                  or r"C:\Users\user\Desktop\QGIS_Plugin\v1.3\validation")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_zip():
    candidates = sorted((PLUGIN / "dist").glob("Rail_3D_v*.zip"))
    return str(candidates[-1]) if candidates else str(PLUGIN / "dist" / "Rail_3D_v1.0.0.zip")


def registry_sha256(source):
    """SHA-256 of the railway model's weights as the ZIP's registry declares it."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if any(isinstance(target, ast.Name) and target.id == "RAIL3D_MODEL" for target in node.targets):
            fields = {item.arg: item.value for item in node.value.keywords}
            if ast.literal_eval(fields["id"]) != MODEL_ID:
                raise ValueError("The ZIP's registry has another model id")
            if ast.literal_eval(fields["weights_filename"]) != WEIGHTS:
                raise ValueError("The ZIP specifies a different weights file")
            value = ast.literal_eval(fields["weights_sha256"])
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid SHA-256 in the ZIP")
            return value
    raise ValueError("RAIL3D_MODEL not found in the ZIP registry")


def check_zip(zip_path):
    """The ZIP is self-contained: returns (weights SHA-256, member count)."""
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if (not parts or parts[0] != NAME or ".." in parts
                    or "\\" in member.filename or ":" in member.filename):
                raise ValueError(f"Unsafe plugin ZIP entry: {member.filename}")
            if len(parts) > 1 and parts[1] in ("tests", "dist", "__pycache__"):
                raise ValueError(f"The ZIP ships {member.filename}")
            if member.filename.lower().endswith((".las", ".laz", ".pyc")):
                raise ValueError(f"The ZIP ships {member.filename}")
        names = {member.filename for member in members}
        for required in (f"{NAME}/metadata.txt", f"{NAME}/__init__.py", f"{NAME}/core/registry.py",
                         f"{NAME}/core/{MODEL_ID}.json", f"{NAME}/weights/{WEIGHTS}"):
            if required not in names:
                raise ValueError(f"The ZIP lacks {required}")
        expected = registry_sha256(archive.read(f"{NAME}/core/registry.py").decode("utf-8"))
        digest = hashlib.sha256()
        with archive.open(f"{NAME}/weights/{WEIGHTS}") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("The bundled weights do not match the registry's SHA-256")
    return expected, len(members)


def make_profile(root, zip_path, ini_name):
    # Refuse reuse. Only a newly created private run directory is writable.
    root.mkdir(parents=False, exist_ok=False)
    plugins = root / "profiles" / "default" / "python" / "plugins"
    plugins.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(plugins)
    ini = root / "profiles" / "default" / "QGIS" / ini_name
    ini.parent.mkdir(parents=True)
    ini.write_text(
        "[PythonPlugins]\n"
        f"{NAME}=true\n\n"
        f"[{NAME}]\nmodel_url\\{MODEL_ID}={NO_NETWORK_URL}\n",
        encoding="utf-8")
    return plugins / NAME / "weights" / WEIGHTS


def run_qgis_process(launcher, profile, cache, algorithm, args, log):
    env = dict(os.environ, QGIS_CUSTOM_CONFIG_PATH=str(profile),
               RAIL3D_CACHE_DIR=str(cache))
    command = ["cmd", "/c", str(launcher), "--json", "run",
               algorithm, "--", *args]
    start = time.monotonic()
    with open(log, "w", encoding="utf-8", errors="replace") as stream:
        code = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,  # nosec B603
                              check=False, timeout=1800).returncode
    return code, time.monotonic() - start


def profile_model_files(profile):
    """Files below the fresh profile's Rail3D/models folder (expected: none)."""
    folder = profile / "profiles" / "default" / NAME / "models"
    if not folder.exists():
        return []
    return [str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()]


def record_signature(records):
    # laspy may reorder the Extra Bytes VLR during a format upgrade. Compare
    # every record's raw payload and metadata, retaining duplicate counts.
    return sorted((record.user_id, record.record_id, record.description,
                   bytes(record.record_data_bytes()))
                  for record in (records or []) if record.user_id != NAME)


def check(source_path, out_path, codes, expected_sha256):
    source, output = laspy.read(source_path), laspy.read(out_path)
    problems = []
    if len(output.points) != len(source.points):
        problems.append(f"{len(output.points)} points written for {len(source.points)}")
    for attr in ("scales", "offsets", "mins", "maxs"):
        if not np.array_equal(getattr(source.header, attr), getattr(output.header, attr)):
            problems.append(f"header {attr} changed")
    if source.header.parse_crs() != output.header.parse_crs():
        problems.append("CRS changed")
    for attr in ("vlrs", "evlrs"):
        if record_signature(getattr(source.header, attr)) != record_signature(getattr(output.header, attr)):
            problems.append(f"non-plugin {attr.upper()} changed")
    source_extras = list(source.point_format.extra_dimension_names)
    output_extras = list(output.point_format.extra_dimension_names)
    if output_extras != source_extras:
        problems.append(f"extra dimensions changed: {source_extras} -> {output_extras}")
    # Both writers upgrade only when a code actually written exceeds 31 (a
    # class the model did not predict on this file does not count).
    written_codes = np.asarray(output.classification)
    upgrade = source.point_format.id < 6 and written_codes.size and int(written_codes.max()) > 31
    expected_format = ({0: 6, 1: 6, 2: 7, 3: 7, 4: 9, 5: 10}[source.point_format.id]
                       if upgrade else source.point_format.id)
    expected_version = "1.4" if upgrade else str(source.header.version)
    if output.point_format.id != expected_format or str(output.header.version) != expected_version:
        problems.append("unexpected LAS version or point-format conversion")
    output_names = set(output.point_format.dimension_names)
    for name in source.point_format.dimension_names:
        if name == "classification":
            continue
        if name == "scan_angle_rank" and upgrade:
            if "scan_angle" not in output_names:
                problems.append("converted scan angle missing")
            else:
                original = np.asarray(source[name], dtype=np.float64)
                converted = np.asarray(output.scan_angle, dtype=np.float64) * 0.006
                if not np.all(np.abs(original - converted) <= 0.003000001):
                    problems.append("legacy scan angle changed by more than 0.003 degrees")
            continue
        if name not in output_names:
            problems.append(f"attribute {name} missing")
            continue
        if name in source_extras:
            left, right = source.points.array[name], output.points.array[name]
            same = left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        else:
            left, right = np.asarray(source[name]), np.asarray(output[name])
            same = left.shape == right.shape and left.tobytes() == right.tobytes()
        if not same:
            problems.append(f"attribute {name} changed")
    labels = np.asarray(output.classification).copy()
    present = set(np.unique(labels).tolist())
    if not present <= set(codes.values()):
        problems.append(f"unexpected codes {sorted(present - set(codes.values()))}")
    if any(record.user_id == "AerialLiDAR" for record in output.header.vlrs):
        problems.append("a VLR of the original plugin was written")
    records = [json.loads(bytes(record.record_data_bytes())) for record in output.header.vlrs
               if record.user_id == NAME and record.record_id == 1]
    mappings = [record for record in records if record.get("field") == "classification"]
    if len(mappings) != 1 or mappings[0].get("model") != MODEL_ID:
        problems.append("classification provenance VLR missing or ambiguous")
    else:
        if mappings[0].get("weights_sha256") != expected_sha256:
            problems.append("provenance VLR names other weights")
        written = {int(k): value["asprs"] for k, value in mappings[0]["classes"].items()}
        if written != codes:
            problems.append(f"VLR mapping {written} != {codes}")
    return problems, labels


def class_ids(labels, codes):
    lookup = np.full(256, -1, dtype=np.int16)
    for class_id, code in codes.items():
        lookup[code] = class_id
    result = lookup[labels]
    if np.any(result < 0):
        raise ValueError("Cannot compare output containing unmapped codes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", default=default_zip())
    parser.add_argument("--input", default=str(VALIDATION / "rail_e2e_input.laz"))
    parser.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "rail3d_e2e"))
    parser.add_argument("--cache", default=os.environ.get("RAIL3D_CACHE_DIR",
                                                          str(Path.home() / ".qgis_rail3d")))
    parser.add_argument("--qgis3", default=r"C:\Program Files\QGIS 3.44.10\bin\qgis_process-qgis-ltr.bat")
    parser.add_argument("--qgis4", default=r"C:\Users\user\qgis4_test\app\bin\qgis_process-qgis.bat")
    args = parser.parse_args()
    work_parent = Path(args.work).resolve()
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="qgis_e2e_", dir=work_parent)).resolve()
    if work.parent != work_parent:
        raise RuntimeError("Private work directory escaped the requested parent")
    report_path = work / "e2e_report.json"
    report = {"work_dir": str(work), "report": str(report_path), "input": str(Path(args.input).resolve()),
              "zip": str(Path(args.zip).resolve()), "model": MODEL_ID, "algorithm": ALGORITHM,
              "cache": str(Path(args.cache).resolve()), "passed": False}
    print(f"Work directory: {work}", flush=True)
    source_hash = None
    labels = {}
    failed = False
    try:
        launchers = {"qgis3": Path(args.qgis3).resolve(), "qgis4": Path(args.qgis4).resolve()}
        missing = [str(path) for path in launchers.values() if not path.is_file()]
        if missing:
            raise ValueError("Both QGIS launchers are required; missing: " + "; ".join(missing))
        if not Path(args.cache).is_dir():
            raise ValueError(f"The plugin environment is missing: {args.cache} (set RAIL3D_CACHE_DIR)")
        source_path = Path(args.input).resolve(strict=True)
        zip_path = Path(args.zip).resolve(strict=True)
        expected_hash, member_count = check_zip(zip_path)
        source_hash = sha256(source_path)
        report.update(input_sha256=source_hash, zip_sha256=sha256(zip_path), zip_members=member_count,
                      weights_sha256=expected_hash)
        with laspy.open(source_path) as reader:
            if not reader.header.point_count:
                raise ValueError("A non-empty reference LAS is required")
            # A single streaming tile preserves the same inference context.
            span = float(np.max(reader.header.maxs[:2] - reader.header.mins[:2]))
            if not np.isfinite(span):
                raise ValueError("Reference LAS has non-finite bounds")
        tile_size = max(1.0, span * 2.0 + 1.0)
        runs = (
            ("qgis3", "QGIS3.ini", DEFAULT_CODES, ["TILE_ENABLED=false"]),
            ("qgis4", "QGIS4.ini", CUSTOM_CODES,
             ["TILE_ENABLED=true", "TILE_STREAMING=true", f"TILE_SIZE_M={tile_size}", "TILE_BUFFER_M=0"]),
        )
        for label, ini, codes, extra in runs:
            profile, out_dir = work / f"profile_{label}", work / f"out_{label}"
            entry = {"launcher": str(launchers[label]), "log": str(work / f"{label}.log"),
                     "algorithm": ALGORITHM}
            try:
                bundled = make_profile(profile, zip_path, ini)
                if sha256(bundled) != expected_hash:
                    raise ValueError("The weights extracted from the ZIP do not match the registry")
                params = [f"INPUT={source_path}", f"OUTPUT_FOLDER={out_dir}", "DEVICE=1",
                          "LOAD_AS_LAYER=false", "UNITS=0", *extra]
                if codes != DEFAULT_CODES:
                    params.append("OUTPUT_CODES_JSON=" + json.dumps({str(k): v for k, v in codes.items()},
                                                                      separators=(",", ":")))
                entry["codes"] = "custom" if codes != DEFAULT_CODES else "default"
                code, seconds = run_qgis_process(launchers[label], profile, Path(args.cache).resolve(),
                                                 ALGORITHM, params, work / f"{label}.log")
                entry.update(exit_code=code, seconds=round(seconds, 1))
                if code:
                    raise ValueError(f"qgis_process exited with status {code}; see {entry['log']}")
                log_text = Path(entry["log"]).read_text(encoding="utf-8", errors="replace")
                if "Downloading" in log_text or NO_NETWORK_URL in log_text:
                    raise ValueError(f"The run tried to download the weights; see {entry['log']}")
                if "Classifier: Railway classifier" not in log_text:
                    raise ValueError(f"The run did not report the Railway classifier; see {entry['log']}")
                cached = profile_model_files(profile)
                if cached:
                    raise ValueError(f"The fresh profile received model files: {cached}")
                entry.update(weights_source="bundled", profile_model_files=cached)
                outputs = sorted(out_dir.glob("*_classified.la*")) if out_dir.exists() else []
                if len(outputs) != 1:
                    raise ValueError(f"Expected one classified output, found {len(outputs)}")
                problems, labels[label] = check(source_path, outputs[0], codes, expected_hash)
                entry.update(output=str(outputs[0]), points=len(labels[label]), problems=problems)
                if problems:
                    failed = True
                if sha256(source_path) != source_hash:
                    raise ValueError("The source input file was modified")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failed = True
            report[label] = entry
            print(label, json.dumps(entry), flush=True)
        if set(labels) != {"qgis3", "qgis4"}:
            raise ValueError("Both QGIS versions must produce validated outputs")
        ids3, ids4 = class_ids(labels["qgis3"], DEFAULT_CODES), class_ids(labels["qgis4"], CUSTOM_CODES)
        if ids3.shape != ids4.shape:
            raise ValueError("Cross-runtime output lengths differ")
        agreement = float(np.mean(ids3 == ids4))
        report.update(qgis3_vs_qgis4_label_agreement=agreement, minimum_agreement=MIN_AGREEMENT)
        if agreement < MIN_AGREEMENT:
            raise ValueError(f"Cross-runtime class agreement {agreement:.6f} is below {MIN_AGREEMENT}")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    finally:
        if source_hash is not None:
            try:
                report["input_sha256_after"] = sha256(Path(args.input).resolve())
                if report["input_sha256_after"] != source_hash:
                    report["source_changed"] = True
                    failed = True
            except OSError as exc:
                report["source_check_error"] = str(exc)
                failed = True
    report["passed"] = not failed
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    print("E2E FAILED" if failed else "E2E PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
