# Model weights

The plugin ZIP published on the Releases page contains the model weights
in this folder (`rail3d_5cm_fp16.pth`, 172 MB). They are not stored in the
Git repository.

To build the ZIP from a clone, copy the weights file here first; the build
(`python build_zip.py`) verifies its SHA-256 against `core/registry.py`.
