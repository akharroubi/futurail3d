"""
Configuration constants, data classes and mappings for Rail 3D.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

_LOG = logging.getLogger(__name__)


# =============================================================================
# Plugin Info
# =============================================================================

PLUGIN_NAME = "Rail 3D"
SETTINGS_PREFIX = "Rail3D"



def _read_plugin_version() -> str:
    """Read ``version=`` from metadata.txt.

    metadata.txt is what the QGIS plugin manager and plugins.qgis.org
    show, so it is the single source of truth. v1.0.2 shipped with a
    separate constant still saying 1.0.1, which the About dialog, the
    Processing provider name and the install marker then reported.
    """
    try:
        metadata = Path(__file__).resolve().parent / "metadata.txt"
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version="):
                return line.split("=", 1)[1].strip()
    except Exception:
        _LOG.debug("Ignored non-fatal error", exc_info=True)
    return "unknown"


PLUGIN_VERSION = _read_plugin_version()


# =============================================================================
# Tiling defaults
# =============================================================================

# Target number of points per tile (auto-size mode). The model runs one
# fixed-size voxel block at a time, so this bounds the host RAM a tile
# needs (its point arrays and block index lists), not GPU memory.
TILE_AUTO_TARGET_POINTS = 10_000_000

# Lateral buffer in metres around each tile, to provide spatial context for
# points near tile edges. 0 (the default since 1.3.3) means automatic: the
# loaded model's own need (core.tiling.resolve_buffer), with which tiled
# labels equal untiled ones. A larger user value is used as given.
TILE_BUFFER_AUTO = 0.0
TILE_DEFAULT_BUFFER_M = TILE_BUFFER_AUTO
# Buffer for a backend that does not state its need (none of the shipped
# ones); the default of 1.0 to 1.3.2.
TILE_FALLBACK_BUFFER_M = 50.0

# Point count above which the UI suggests turning tiling on.
TILE_RECOMMEND_ABOVE = 20_000_000


# =============================================================================
# Models
# =============================================================================
# Everything model-specific (weights URLs, SHA-256, bundled configuration,
# class mapping, supported devices, licence) is declared in
# core/registry.py, one ModelSpec per model.


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClassInfo:
    """Represents a classification class with model ID, name, ASPRS code, and color."""
    model_id: int
    name: str
    asprs_code: int
    color: str


@dataclass
class DataItem:
    """Represents a file or folder in the data manager."""
    path: Path
    is_folder: bool
    tile_count: int = 1
    tile_paths: List[Path] = None
    point_count: int = 0
    size_mb: float = 0.0

    def __post_init__(self):
        if self.tile_paths is None:
            self.tile_paths = []

    @property
    def display_name(self) -> str:
        if self.is_folder:
            return f"{self.path.name} ({self.tile_count} tiles)"
        return self.path.name


# =============================================================================
# Class names
# =============================================================================

# Standard ASPRS class names for dropdowns
ASPRS_CLASSES = {
    0: "Unclassified",
    1: "Unassigned",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire - Guard",
    14: "Wire - Conductor",
    15: "Transmission Tower",
    16: "Wire - Connector",
    17: "Bridge Deck",
    18: "High Noise",
    19: "Overhead Structure",
    20: "Ignored Ground",
    21: "Snow",
    22: "Temporal Exclusion"}
