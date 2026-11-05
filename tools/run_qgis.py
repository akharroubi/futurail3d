"""Run a Python script with a QGIS launcher (.bat) and tee its output.

    python tools/run_qgis.py {qgis3|qgis4|qgis3-gui|qgis4-gui|process3|process4} <script or args...>

The OSGeo4W launchers are batch files: they are started through cmd.exe
with the arguments passed verbatim (Git Bash mangles them). Output goes to
the console and to tools/logs/<name>.log. Exit status is the script's.
"""
import os
import subprocess  # nosec B404 - developer tool, not shipped
import sys
import time
from pathlib import Path

LAUNCHERS = {
    "qgis3": r"C:\Program Files\QGIS 3.44.10\bin\python-qgis-ltr.bat",
    "qgis4": r"C:\Users\user\qgis4_test\app\bin\python-qgis.bat",
    "qgis3-gui": r"C:\Program Files\QGIS 3.44.10\bin\qgis-ltr.bat",
    "qgis4-gui": r"C:\Users\user\qgis4_test\app\bin\qgis.bat",
    "process3": r"C:\Program Files\QGIS 3.44.10\bin\qgis_process-qgis-ltr.bat",
    "process4": r"C:\Users\user\qgis4_test\app\bin\qgis_process-qgis.bat",
}


def main():
    key, *rest = sys.argv[1:]
    launcher = LAUNCHERS[key]
    if not Path(launcher).is_file():
        raise SystemExit(f"launcher not found: {launcher}")
    logs = Path(__file__).resolve().parent / "logs"
    logs.mkdir(exist_ok=True)
    name = Path(rest[0]).stem if rest else key
    log = logs / f"{key}_{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    command = ["cmd", "/c", launcher, *rest]
    print("running:", " ".join(command), flush=True)
    print("log:", log, flush=True)
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        process = subprocess.Popen(  # nosec B603 - fixed launcher list
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", env=dict(os.environ))
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stream.write(line)
        code = process.wait()
    print(f"exit {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
