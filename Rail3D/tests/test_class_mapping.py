"""Output-code overrides must stay lossless, explicit and isolated from defaults."""

import json
import unittest

from _load import load_plugin_module

mapping = load_plugin_module("utils.class_mapping")
registry = load_plugin_module("core.registry")


class RailwayOutputCodes(unittest.TestCase):
    def setUp(self):
        self.spec = registry.get_model(mapping.RAILWAY_MODEL_ID)

    def test_partial_overrides_preserve_defaults_and_snapshot_values(self):
        codes = {6: 255, "7": 0}
        selected = mapping.output_model_spec(self.spec, codes)
        codes[6] = 12
        self.assertEqual(selected.class_mapping[6].asprs_code, 255)
        self.assertEqual(selected.class_mapping[7].asprs_code, 0)
        self.assertEqual(self.spec.class_mapping[6].asprs_code, 64)
        self.assertEqual(selected.class_mapping[1].asprs_code, 2)
        selected.class_mapping[1].asprs_code = 99
        self.assertEqual(self.spec.class_mapping[1].asprs_code, 2)

    def test_blank_defaults_and_json_roundtrip(self):
        for blank in (None, "", "  ", {}):
            self.assertEqual(mapping.output_model_spec(self.spec, blank), self.spec)
        selected = mapping.output_model_spec(self.spec, '{"6": 200, "7": 200}')
        saved = mapping.output_codes_json(selected)
        self.assertEqual(len(json.loads(saved)), 7)
        self.assertEqual(mapping.output_model_spec(self.spec, saved), selected)

    def test_defaults_overrides_and_roundtrip(self):
        self.assertTrue(mapping.supports_output_codes(self.spec))
        self.assertEqual(mapping.output_model_spec(self.spec), self.spec)
        selected = mapping.output_model_spec(self.spec, '{"6": 200, "7": 0}')
        self.assertEqual(selected.class_mapping[6].asprs_code, 200)
        self.assertEqual(selected.class_mapping[7].asprs_code, 0)
        self.assertEqual(selected.class_mapping[2].asprs_code, 10)
        self.assertEqual(self.spec.class_mapping[6].asprs_code, 64)
        saved = mapping.output_codes_json(selected)
        self.assertEqual(json.loads(saved), {"1": 2, "2": 10, "3": 15, "4": 14, "5": 5, "6": 200, "7": 0})
        self.assertEqual(mapping.output_model_spec(self.spec, saved), selected)
        # Merging classes is allowed.
        merged = mapping.output_model_spec(self.spec, {3: 14, 4: 14})
        self.assertEqual(merged.class_mapping[3].asprs_code, 14)

    def test_invalid_ids_codes_shapes_and_duplicate_ids_are_rejected(self):
        invalid = [
            '[]', 'null', '"text"', '{broken', '{"5": 1, "5": 2}',
            {0: 2}, {8: 2}, {9: 2}, {10: 2}, {"05": 2}, {True: 2}, {1.0: 2}, {"1.0": 2},
            {5: True}, {5: 2.0}, {5: "2"}, {5: -1}, {5: 256}, {5: None},
            {5: 2, "5": 3}, [2, 3],
        ]
        for codes in invalid:
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                mapping.output_model_spec(self.spec, codes)

    def test_ids_of_other_models_are_rejected(self):
        for codes in ({8: 1}, {9: 1}, {0: 2}, '{"8": 3}'):
            with self.subTest(codes=codes), self.assertRaisesRegex(ValueError, "valid IDs"):
                mapping.output_model_spec(self.spec, codes)

    def test_classification_field_only(self):
        for field in (None, "", "classification", " Classification "):
            self.assertEqual(mapping.validate_model_output_field(self.spec, field), "classification")
        with self.assertRaisesRegex(ValueError, "Railway model writes to the existing"):
            mapping.validate_model_output_field(self.spec, "rail_ids")

    def test_non_editable_model_rejects_overrides_and_names_the_editable_one(self):
        from dataclasses import replace
        fixed = replace(self.spec, id="fixed", editable_codes=False)
        self.assertFalse(mapping.supports_output_codes(fixed))
        self.assertEqual(mapping.output_model_spec(fixed), fixed)
        with self.assertRaises(ValueError) as caught:
            mapping.output_model_spec(fixed, {1: 255})
        self.assertIn("Railway model", str(caught.exception))
        self.assertEqual(mapping.validate_model_output_field(fixed, "raw_ids"), "raw_ids")


class AsprsCodeNames(unittest.TestCase):
    def test_standard_reserved_and_user_codes(self):
        self.assertEqual(mapping.asprs_code_name(10), "Rail")
        self.assertEqual(mapping.asprs_code_name(15), "Transmission tower")
        self.assertEqual(mapping.asprs_code_name(14), "Wire conductor (phase)")
        self.assertEqual(mapping.asprs_code_name(22), "Temporal exclusion")
        self.assertEqual(mapping.asprs_code_name(23), "Reserved")
        self.assertEqual(mapping.asprs_code_name(63), "Reserved")
        self.assertEqual(mapping.asprs_code_name(64), "User definable")
        self.assertEqual(mapping.asprs_code_name(255), "User definable")
        for bad in (-1, 256):
            with self.assertRaises(ValueError):
                mapping.asprs_code_name(bad)
        # Every default code of every model has a meaning; none is "Reserved".
        for spec in registry.MODELS:
            for info in spec.class_mapping.values():
                self.assertNotEqual(mapping.asprs_code_name(info.asprs_code), "Reserved", info)


if __name__ == "__main__":
    unittest.main()
