"""The model the plugin runs, declared once.

The entry says where its weights come from (download URLs,
SHA-256), which configuration file describes it, how its class ids map to
LAS codes, which LAS dimensions and compute devices it needs, and how it is
licensed. The dock, the Processing algorithm, the model manager and the
installer all read this table; nothing else in the plugin hard-codes a model.

Rail 3D ships one classifier (Railway) with one engine (the railway model).
The classifier / engine structure is kept so that a later model can be added
without touching the callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from ..config import ClassInfo

_CORE_DIR = Path(__file__).resolve().parent
# Weights published by the plugin author, fetched by Setup or by the first
# run and SHA-256 verified; a verified copy in the plugin's weights/ folder
# is used instead when one is there (offline installs).
_GITHUB_RELEASES = "https://github.com/akharroubi/AerialLidarClassifier/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    family: str                       # "litept": picks the backend
    short_name: str
    description: str
    resolution: str                   # human-readable voxel size
    weights_filename: str
    weights_urls: Tuple[str, ...]     # tried in order; SHA-256 verified
    weights_sha256: str
    weights_size_mb: float
    config_filename: str              # bundled next to this file
    class_mapping: Dict[int, ClassInfo]
    class_summary: str                # one line for the UI
    supported_devices: Tuple[str, ...]
    required_dims: Tuple[str, ...] = ("X", "Y", "Z")
    extra_packages: Tuple[str, ...] = ()   # logical names resolved by the installer
    licence: str = ""
    attribution: str = ""
    homepage: str = ""
    # The user can choose each class's LAS code (dock dialog, Processing
    # OUTPUT_CODES_JSON). Such models write the standard classification
    # field only.
    editable_codes: bool = False

    @property
    def config_path(self) -> Path:
        return _CORE_DIR / self.config_filename

    @property
    def bundled_weights_path(self) -> Path:
        """A weights file placed in the plugin folder (offline installs)."""
        return _CORE_DIR.parent / "weights" / self.weights_filename

    def supports_device(self, device: str) -> bool:
        return str(device).lower().split(":")[0] in self.supported_devices

    def device_requirement_text(self) -> str:
        if self.supported_devices == ("cuda",):
            return "NVIDIA CUDA GPU required"
        return "runs on GPU or CPU"


# Railway logits 0..6 become plugin model IDs 1..7. Default codes follow the
# ASPRS LAS 1.4 table where a class exists: ground 2, rail 10, wire 14
# (conductor), mast/pole 15 (transmission tower, the structure that carries
# the conductors), vegetation 5. Fences have no ASPRS class and take 64, the
# first user-definable code; other objects are 1 (unclassified). All codes
# are editable.
_RAIL_MAPPING: Dict[int, ClassInfo] = {
    1: ClassInfo(1, "Ground", 2, "#A87E55"),
    2: ClassInfo(2, "Rail", 10, "#E31A1C"),
    3: ClassInfo(3, "Pole", 15, "#FF00FF"),
    4: ClassInfo(4, "Wire", 14, "#FFA500"),
    5: ClassInfo(5, "Vegetation", 5, "#228B22"),
    6: ClassInfo(6, "Fence", 64, "#00CED1"),
    7: ClassInfo(7, "Other", 1, "#C0C0C0"),
}

RAIL3D_MODEL = ModelSpec(
    id="rail3d_5cm",
    display_name="Rail 3D railway model (5 cm)",
    family="litept",
    short_name="Railway model",
    description=(
        "Deep-learning model for railway corridor LiDAR at 5 cm, using XYZ only. "
        "Seven classes: ground (with track bed and platforms), rails, poles "
        "(catenary masts, portals and their supports), overhead wires, "
        "vegetation, fences and barriers, and other objects. Output codes are editable."
    ),
    resolution="5 cm",
    weights_filename="rail3d_5cm_fp16.pth",
    weights_urls=(
        f"{_GITHUB_RELEASES}/v1.3/litept_l_rail_5cm_fp16.pth",
    ),
    weights_sha256="bd3356898bcf79b04996493a89f45393f8c1868a0fd2e712792179d298f27237",  # noqa: E501  # pragma: allowlist secret
    weights_size_mb=171.8,
    config_filename="rail3d_5cm.json",
    class_mapping=_RAIL_MAPPING,
    class_summary=(
        "Ground 2, Rail 10, Pole 15, Wire 14, Vegetation 5, Fence 64, Other 1; "
        "every code is editable (64 is user-defined)"
    ),
    supported_devices=("cuda",),
    extra_packages=("spconv", "scipy"),
    licence="Model weights © 2026 Abderrazzaq Kharroubi; evaluation copy, not for redistribution",
    attribution="Model weights © 2026 Abderrazzaq Kharroubi, GeoScITY Lab, University of Liège.",
    homepage="https://github.com/akharroubi/futurail3d",
    editable_codes=True,
)

# Every model (engine). Append only.
MODELS: Tuple[ModelSpec, ...] = (RAIL3D_MODEL,)
DEFAULT_MODEL_ID = RAIL3D_MODEL.id


def get_model(model_id: Optional[str]) -> ModelSpec:
    for spec in MODELS:
        if spec.id == model_id:
            return spec
    raise KeyError(f"Unknown model id: {model_id!r}")


# ---------------------------------------------------------------------------
# Classifiers: what the user chooses (one per acquisition type)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Classifier:
    id: str
    display_name: str
    description: str
    # Preference order: the first engine that can run on the computer is used.
    engines: Tuple[ModelSpec, ...] = ()

    @property
    def primary(self) -> Optional[ModelSpec]:
        return self.engines[0] if self.engines else None

    @property
    def list_label(self) -> str:
        """The name shown in selectors."""
        return self.display_name


RAILWAY = Classifier(
    id="railway",
    display_name="Railway classifier",
    description="Railway corridor LiDAR (mobile or airborne). 5 cm model, NVIDIA CUDA GPU.",
    engines=(RAIL3D_MODEL,),
)

# Processing saves CLASSIFIER as an index: append only.
CLASSIFIERS: Tuple[Classifier, ...] = (RAILWAY,)
DEFAULT_CLASSIFIER_ID = RAILWAY.id


def get_classifier(classifier_id: Optional[str]) -> Classifier:
    for classifier in CLASSIFIERS:
        if classifier.id == classifier_id:
            return classifier
    raise KeyError(f"Unknown classifier id: {classifier_id!r}")


def classifier_for_model(model_id: Optional[str]) -> Classifier:
    """The classifier an engine belongs to."""
    for classifier in CLASSIFIERS:
        if any(spec.id == model_id for spec in classifier.engines):
            return classifier
    raise KeyError(f"Unknown model id: {model_id!r}")


def select_engine(classifier: Classifier, device: str,
                  runs_here: Callable[[ModelSpec, str], bool]) -> Optional[ModelSpec]:
    """First engine of ``classifier`` that ``runs_here`` on ``device``.

    When none can run, the primary engine is returned so its device or
    dependency gate can explain what is missing.
    """
    for spec in classifier.engines:
        if runs_here(spec, device):
            return spec
    return classifier.primary
