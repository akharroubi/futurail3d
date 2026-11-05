"""Check the weights download that Setup performs after the packages install.

    python tools/run_qgis.py qgis3 tools/setup_download_check.py

Uses a temporary QGIS profile (QGIS_CUSTOM_CONFIG_PATH) so nothing is
written to the user's profile; needs the plugin environment of
RAIL3D_CACHE_DIR (default ~/.qgis_rail3d) and internet access. Calls the
worker's own ``_download_weights`` and verifies the file it fetched.
"""
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

profile = Path(tempfile.mkdtemp(prefix="rail3d_setup_"))
os.environ["QGIS_CUSTOM_CONFIG_PATH"] = str(profile)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from qgis.core import QgsApplication  # noqa: E402

app = QgsApplication([], False)
app.initQgis()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib  # noqa: E402

registry = importlib.import_module("Rail3D.core.registry")
manager_module = importlib.import_module("Rail3D.utils.model_manager")
worker_module = importlib.import_module("Rail3D.workers.deps_install_worker")

spec = registry.RAIL3D_MODEL
manager = manager_module.ModelManager(spec)
print("profile:", profile, flush=True)
print("available before:", manager.is_model_available(), flush=True)
worker = worker_module.DepsInstallWorker(cuda_enabled=True)
start = time.monotonic()
message = worker._download_weights("Virtual environment ready")
print(f"message after {time.monotonic() - start:.0f} s: {message}", flush=True)
path = manager.get_model_path()
print("path:", path, path.exists() and path.stat().st_size, flush=True)
digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
ok = manager.is_model_available() and digest == spec.weights_sha256 and str(profile) in str(path)
print("sha ok:", digest == spec.weights_sha256, "available:", manager.is_model_available(), flush=True)
raise SystemExit(0 if ok else 1)
