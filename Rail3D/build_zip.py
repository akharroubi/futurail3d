"""Build the installable ZIP of Rail 3D from this folder.

    python build_zip.py            -> dist/Rail_3D_v<version>.zip

The archive holds one top-level folder, ``Rail3D``: the plugin's package
name (QGIS imports the folder as a Python package, so the name must not
change between versions or an update installs side by side).

An explicit list of runtime files is packed, plus the model weights in
``weights/`` (the ZIP is self-contained: nothing is downloaded on first
use apart from the Python packages). No tests, caches or stray files.
Timestamps are fixed and text files use LF, so the same source gives the
same files.

The build refuses to run with any Bandit finding, Flake8 syntax-class
errors or unlisted long hex strings (the policies of the author's public
plugin, kept so that both code bases stay interchangeable).
"""
import argparse
import re
import sys
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = "Rail3D"
DIRS = ("core", "dialogs", "gui", "processing", "utils", "widgets",
        "workers", "assets", "i18n", "weights")
FILES = ("__init__.py", "plugin.py", "config.py", "metadata.txt", "icon.png",
         "LICENSE", "README.md", "CHANGELOG.md", "THIRD_PARTY_NOTICES.md")
SKIP_SUFFIXES = (".pyc", ".pyo", ".log", ".zip", ".bak")
TEXT_SUFFIXES = (".py", ".md", ".txt", ".json", ".svg", ".ts", ".ui", ".cfg")
FIXED_TIME = (2026, 11, 5, 0, 0, 0)
FORBIDDEN_SUFFIXES = (".exe", ".dll", ".so", ".dylib", ".pyd", ".sh", ".bat", ".cmd",
                      ".las", ".laz")


def read_version() -> str:
    for line in (ROOT / "metadata.txt").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("metadata.txt has no version=")


def collect():
    files = [ROOT / name for name in FILES]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")
    for directory in DIRS:
        files.extend(
            p for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
            and p.suffix.lower() not in SKIP_SUFFIXES
        )
    return sorted(files)


_HEX_RUN = re.compile(rb"[0-9a-fA-F]{32,}")


def check_weights() -> None:
    """Every model's bundled weights must be present and match the registry."""
    weights = sorted((ROOT / "weights").glob("*.pth"))
    if not weights:
        raise SystemExit("weights/ holds no .pth file: copy the model weights there first")
    registry = (ROOT / "core" / "registry.py").read_text(encoding="utf-8")
    for path in weights:
        if path.name not in registry:
            raise SystemExit(f"{path.name} is not declared in core/registry.py")
        expected = re.search(
            rf'weights_filename="{re.escape(path.name)}".*?weights_sha256="([0-9a-f]{{64}})"',
            registry, re.S)
        if not expected:
            raise SystemExit(f"No SHA-256 found for {path.name} in core/registry.py")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected.group(1):
            raise SystemExit(f"{path.name}: SHA-256 {digest[:12]}... does not match the registry")
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, SHA-256 verified")


def check_secrets_scanner(files) -> None:
    """Long hex runs only on lines marked ``pragma: allowlist secret``."""
    problems = []
    for path in files:
        if path.suffix.lower() in (".png", ".jpg", ".ico", ".pth", ".pt"):
            continue
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            if _HEX_RUN.search(line) and b"pragma: allowlist secret" not in line:
                problems.append(f"{path.relative_to(ROOT)}:{number}")
    if problems:
        raise SystemExit(
            "Long hex strings (shorten them or add "
            "'# pragma: allowlist secret' on the line):\n  " + "\n  ".join(problems))


def check_bandit() -> None:
    """Refuse to build unless Bandit reports nothing, with no config file."""
    import json
    import subprocess  # nosec B404 - build tool only, never shipped
    targets = [str(ROOT / name) for name in DIRS if (ROOT / name).is_dir()]
    targets += [str(ROOT / name) for name in FILES if name.endswith(".py")]
    try:
        result = subprocess.run(  # nosec B603 - fixed argument list
            [sys.executable, "-m", "bandit", "-q", "-f", "json", "-r", *targets],
            capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SystemExit(f"Could not run Bandit: {exc}")
    if "No module named bandit" in result.stderr:
        raise SystemExit("Bandit is required to build: python -m pip install bandit")
    if result.returncode not in (0, 1):
        raise SystemExit(f"Bandit failed to run: {result.stderr}")
    scan = json.loads(result.stdout or "{}")
    if "results" not in scan or scan.get("errors"):
        raise SystemExit("Bandit did not complete a full source scan: " + result.stderr)
    findings = scan["results"]
    if findings:
        listed = [f"{f['test_id']} {Path(f['filename']).relative_to(ROOT)}:{f['line_number']}"
                  for f in findings]
        raise SystemExit("Bandit findings:\n  " + "\n  ".join(listed))
    if (ROOT / ".bandit").exists():
        raise SystemExit("Remove .bandit: the scan must pass without a config file.")


def check_no_binaries(files) -> None:
    """Executables and data never go into the plugin ZIP."""
    found = [str(p.relative_to(ROOT)) for p in files if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    if found:
        raise SystemExit("Executable or data files must not be packed:\n  " + "\n  ".join(found))


def check_metadata_parses() -> None:
    """QGIS reads metadata.txt with configparser: parse it the same way."""
    import configparser
    parser = configparser.ConfigParser()
    try:
        parser.read(ROOT / "metadata.txt", encoding="utf-8")
        for key in parser["general"]:
            parser["general"][key]
    except (configparser.Error, KeyError) as exc:
        raise SystemExit(f"metadata.txt does not parse: {exc}. Write 'percent' instead of '%'.")


def check_flake8_blockers() -> None:
    """Refuse Flake8's syntax-class errors (E9, F821, F823, F831)."""
    import subprocess  # nosec B404 - build tool only, never shipped
    targets = [str(ROOT / name) for name in DIRS if (ROOT / name).is_dir()]
    targets += [str(ROOT / name) for name in FILES if name.endswith(".py")]
    result = subprocess.run(  # nosec B603 - fixed argument list
        [sys.executable, "-m", "flake8", "--isolated", "--select=E9,F821,F823,F831", *targets],
        capture_output=True, text=True, check=False)
    if "No module named flake8" in result.stderr:
        raise SystemExit("Flake8 is required to build: python -m pip install flake8")
    if result.returncode != 0:
        raise SystemExit("Flake8 blocking errors:\n" + (result.stdout or result.stderr))


def build(destination: Path) -> str:
    files = collect()
    check_metadata_parses()
    check_weights()
    check_no_binaries(files)
    check_secrets_scanner(files)
    check_bandit()
    check_flake8_blockers()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = f"{PACKAGE}/{path.relative_to(ROOT).as_posix()}"
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            data = path.read_bytes()
            if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("LICENSE"):
                data = data.replace(b"\r\n", b"\n")
            archive.writestr(info, data)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_name(destination.name + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8")
    size_mb = destination.stat().st_size / 1e6
    print(f"{destination}\n{len(files)} files, {size_mb:.1f} MB\nSHA-256 {digest}")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=None)
    args = parser.parse_args()
    default = ROOT / "dist" / f"Rail_3D_v{read_version()}.zip"
    build(Path(args.output) if args.output else default)
