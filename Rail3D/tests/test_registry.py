"""Consistency checks for core/registry.py (no QGIS, no torch needed)."""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import PLUGIN_ROOT, load_plugin_module  # noqa: E402

registry = load_plugin_module("core.registry")
RAIL = registry.RAIL3D_MODEL


def test_every_model_has_its_bundled_config():
    for spec in registry.MODELS:
        assert spec.config_path.is_file(), f"{spec.id}: {spec.config_path} missing"


def test_hashes_and_urls_look_right():
    for spec in registry.MODELS:
        assert re.fullmatch(r"[0-9a-f]{64}", spec.weights_sha256), spec.id
        # The weights are fetched at first use: every model has a public URL.
        assert spec.weights_urls and all(u.startswith("https://") for u in spec.weights_urls)
        assert 0 < spec.weights_size_mb < 400


def test_weights_are_not_shipped_and_an_offline_copy_has_a_fixed_place():
    """The plugin folder carries no weights (they download at first use).

    A verified copy may be placed by hand in weights/ for offline installs;
    when one is there it must be the released file.
    """
    for spec in registry.MODELS:
        path = spec.bundled_weights_path
        assert path == PLUGIN_ROOT / "weights" / spec.weights_filename
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == spec.weights_sha256, f"{path} is not the released file"
        assert round(path.stat().st_size / 1e6, 1) == spec.weights_size_mb
    # No weights file anywhere in the repository tree that the ZIP could pack.
    shipped = [p for p in PLUGIN_ROOT.rglob("*") if p.suffix.lower() in (".pth", ".pt", ".ckpt")
               and "weights" not in p.parts]
    assert not shipped, shipped


def test_class_mappings_are_asprs_codes():
    for spec in registry.MODELS:
        ids = sorted(spec.class_mapping)
        assert ids == list(range(1, len(ids) + 1)), f"{spec.id}: ids must be 1..n"
        for info in spec.class_mapping.values():
            assert 0 <= info.asprs_code <= 255
            assert info.model_id in spec.class_mapping


def test_ids_and_families_are_unique_and_known():
    ids = [spec.id for spec in registry.MODELS]
    assert len(set(ids)) == len(ids)
    assert {spec.family for spec in registry.MODELS} <= {"litept"}
    assert registry.get_model(registry.DEFAULT_MODEL_ID) is RAIL


def test_railway_card_codes_and_names():
    assert registry.MODELS == (RAIL,)
    assert RAIL.id == "rail3d_5cm"
    card = json.loads(RAIL.config_path.read_text(encoding="utf-8"))
    assert card["id"] == RAIL.id
    assert card["display_name"] == RAIL.display_name
    assert card["num_classes"] == len(RAIL.class_mapping) == 7
    assert card["grid_size"] == .05
    assert card["weights_file"] == RAIL.weights_filename
    assert round(card["weights_bytes"] / 1e6, 1) == RAIL.weights_size_mb
    assert "weights_sha256" not in card
    assert card["class_names"] == [c.name.lower() for c in RAIL.class_mapping.values()]
    # ASPRS where a class exists (Ground 2, Rail 10, Wire 14, Tower 15, High
    # vegetation 5, Unclassified 1); fences take the first user-definable code.
    assert {i: c.asprs_code for i, c in RAIL.class_mapping.items()} == {
        1: 2, 2: 10, 3: 15, 4: 14, 5: 5, 6: 64, 7: 1}
    assert RAIL.editable_codes
    assert RAIL.supported_devices == ("cuda",)
    assert RAIL.extra_packages == ("spconv", "scipy")
    assert RAIL.device_requirement_text() == "NVIDIA CUDA GPU required"
    # No internal model or architecture name reaches the user.
    for text in (RAIL.display_name, RAIL.short_name, RAIL.description, RAIL.licence,
                 RAIL.attribution, RAIL.class_summary, card["display_name"]):
        assert "LitePT" not in text and "—" not in text, text


def test_user_defined_default_codes_start_at_64_without_gaps():
    for spec in registry.MODELS:
        user = sorted({c.asprs_code for c in spec.class_mapping.values() if c.asprs_code >= 64})
        assert user == list(range(64, 64 + len(user))), (spec.id, user)


def test_classifiers_cover_every_model_once():
    assert [c.id for c in registry.CLASSIFIERS] == ["railway"]
    assert registry.DEFAULT_CLASSIFIER_ID == "railway"
    engines = [spec.id for c in registry.CLASSIFIERS for spec in c.engines]
    assert sorted(engines) == sorted(spec.id for spec in registry.MODELS)
    assert registry.RAILWAY.primary is RAIL
    assert registry.RAILWAY.list_label == "Railway classifier"
    assert registry.classifier_for_model(RAIL.id) is registry.RAILWAY
    assert registry.get_classifier("railway") is registry.RAILWAY
    for bad in (None, "", "unknown"):
        for lookup in (registry.get_classifier, registry.classifier_for_model):
            try:
                lookup(bad)
            except KeyError:
                pass
            else:
                raise AssertionError(f"{lookup.__name__}({bad!r}) did not raise")


def test_select_engine_always_returns_the_railway_model():
    def runs(ready):
        return lambda spec, device: spec.supports_device(device) and ready

    # The primary engine comes back even where it cannot run, so its gate
    # can explain what is missing.
    for device, ready in (("cuda", True), ("cuda", False), ("cpu", True), ("mps", False)):
        assert registry.select_engine(registry.RAILWAY, device, runs(ready)) is RAIL


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
