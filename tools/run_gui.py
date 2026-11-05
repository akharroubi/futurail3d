"""Install the built ZIP into a fresh QGIS profile and drive the real window.

    python tools/run_gui.py {qgis3|qgis4} <input.laz> [--zip dist.zip] [--keep]

Extracts the ZIP into <tools>/gui_<key>/profiles/default/python/plugins
(what "Install from ZIP" does), then launches QGIS twice with
tools/gui_run.py: once to enable the plugin, once to drive it. The
screenshots and report.json land in <tools>/gui_<key>/out. The plugin's
Python environment is the one of RAIL3D_CACHE_DIR (default ~/.qgis_rail3d),
which must be installed already (tools/fresh_install.py).
"""
import argparse
import json
import os
import shutil
import subprocess  # nosec B404 - developer tool, not shipped
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
LAUNCHERS = {
    "qgis3": r"C:\Program Files\QGIS 3.44.10\bin\qgis-ltr.bat",
    "qgis4": r"C:\Users\user\qgis4_test\app\bin\qgis.bat",
}


def install_zip(zip_path, profile_root):
    plugins = profile_root / "profiles" / "default" / "python" / "plugins"
    plugins.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != "Rail3D" or ".." in parts:
                raise ValueError(f"Unsafe plugin ZIP entry: {member.filename}")
        archive.extractall(plugins)


def launch(key, profile_root, out, source, timeout):
    env = dict(os.environ, RAIL3D_GUI_OUT=str(out), RAIL3D_GUI_INPUT=str(source))
    command = ["cmd", "/c", LAUNCHERS[key], "--profiles-path", str(profile_root),
               "--noversioncheck", "--nologo", "--code", str(TOOLS / "gui_run.py")]
    report = out / "report.json"
    report.unlink(missing_ok=True)
    print("launch:", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=False)  # nosec B603 - fixed launcher list
    start = time.monotonic()
    while not report.exists():
        if time.monotonic() - start > timeout:
            raise SystemExit(f"no report after {timeout} s")
        time.sleep(3)
    time.sleep(4)  # let QGIS exit
    return json.loads(report.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("key", choices=sorted(LAUNCHERS))
    parser.add_argument("source")
    parser.add_argument("--zip", default=None)
    parser.add_argument("--keep", action="store_true", help="reuse the profile if it exists")
    args = parser.parse_args()
    zip_path = Path(args.zip) if args.zip else next((REPO / "Rail3D" / "dist").glob("Rail_3D_v*.zip"))
    profile_root = TOOLS / f"gui_{args.key}"
    out = profile_root / "out"
    if profile_root.exists() and not args.keep:
        shutil.rmtree(profile_root)
    if not profile_root.exists():
        install_zip(zip_path, profile_root)
        print("plugin extracted from", zip_path, flush=True)
        first = launch(args.key, profile_root, out, args.source, 300)
        print("first launch:", first.get("stage"), flush=True)
    second = launch(args.key, profile_root, out, args.source, 2700)
    print(json.dumps({k: v for k, v in second.items() if k != "log_tail"}, indent=2, default=str))
    print("log tail:\n" + str(second.get("log_tail", "")))
    return 0 if "error" not in second and second.get("outputs") else 1


if __name__ == "__main__":
    sys.exit(main())
