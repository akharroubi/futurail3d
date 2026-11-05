# Third-party notices and licences

The plugin's original source is GPL-3.0-or-later; see LICENSE.

## Railway model weights

The weights (`weights/rail3d_5cm_fp16.pth`) are © 2026 Abderrazzaq
Kharroubi, GeoScITY Lab, University of Liège. They are provided to the
recipient for evaluation, as is, without warranty, and are not to be
redistributed. SHA-256: starts with `bd3356898bcf` (full value in
`core/registry.py`).

## Network implementation

The point transformer implementation vendored in `core/litept/` is derived
from an MIT-licensed project of the Photogrammetry and Remote Sensing Lab,
ETH Zurich. The upstream MIT notice is retained, as the licence requires,
in `core/litept/LICENSE.upstream`.

## Runtime dependencies

Python, PyTorch, torchvision, laspy, lazrs, scipy, NumPy, timm,
numpy_indexed, spconv and their dependencies retain their own licences.
They are installed separately into a per-user environment; they are not
relicensed by this file.
