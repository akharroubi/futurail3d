"""Run the plugin's one-click setup headlessly into a fresh cache folder.

    cmd /c "C:\\Program Files\\QGIS 3.44.10\\bin\\python-qgis-ltr.bat" tools\\fresh_install.py

What the CEO's first click does, minus the dock: a portable Python, uv,
PyTorch (CUDA build chosen from the driver), the LiDAR packages and
spconv are installed into RAIL3D_CACHE_DIR (default ~/.qgis_rail3d).
Prints every progress message with a timestamp and exits non-zero on
failure.
"""
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from qgis.core import QgsApplication  # noqa: E402

app = QgsApplication([], False)
app.initQgis()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib  # noqa: E402

venv = importlib.import_module("Rail3D.utils.venv_manager")
print("cache dir:", venv.CACHE_DIR, flush=True)
has_gpu, info = venv.detect_nvidia_gpu()
print("gpu:", has_gpu, info, flush=True)
start = time.monotonic()


def progress(percent, message):
    print(f"[{time.monotonic() - start:7.1f}s] {percent:3d}% {message}", flush=True)


ok, message = venv.create_venv_and_install(
    progress_callback=progress, cancel_check=lambda: False, cuda_enabled=has_gpu)
print("result:", ok, message, flush=True)
ready, status = venv.get_venv_status()
print("status:", ready, status, flush=True)
raise SystemExit(0 if ok and ready else 1)
