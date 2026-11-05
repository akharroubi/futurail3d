"""Integration checks of the Rail 3D plugin with real QGIS/Qt and LAS I/O.

Run with QGIS's python-qgis-ltr.bat (QGIS 3) or python-qgis.bat (QGIS 4).
Inference is stubbed (real CUDA inference: the backend unit test and
qgis_rail_gpu.py). No network (the weights download is served from a local
socket or stubbed), no GPU, installed plugin or user settings needed.

The plugin environment is read from RAIL3D_CACHE_DIR when set, else
~/.qgis_rail3d (only laspy and numpy are imported from it here).
"""
import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# QGIS must load its SSL/Qt DLLs before unittest.mock imports Python's SSL.
from qgis.core import (  # noqa: E402
    QgsApplication, QgsPointCloudLayer, QgsProcessingContext, QgsProcessingException,
    QgsProcessingFeedback,
)
app = QgsApplication([], False)
app.initQgis()

import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from dataclasses import replace  # noqa: E402
from unittest.mock import PropertyMock, patch  # noqa: E402

from qgis.PyQt.QtCore import QTimer  # noqa: E402

# Windows Qt's offscreen platform has an empty font database. Give optional
# visual evidence the same readable font the normal Windows application uses.
if os.environ.get('ALC_UI_CAPTURE_DIR') and Path('C:/Windows/Fonts/segoeui.ttf').is_file():
    from qgis.PyQt.QtGui import QFont, QFontDatabase
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    app.setFont(QFont('Segoe UI', 9))

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get('RAIL3D_CACHE_DIR') or (Path.home() / '.qgis_rail3d'))
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(CACHE_DIR / 'venv_py3.12/Lib/site-packages'))
import laspy  # noqa: E402
import numpy as np  # noqa: E402


def module(name):
    return importlib.import_module(ROOT.name + '.' + name)


registry = module('core.registry')
RAIL = registry.get_model('rail3d_5cm')
tasks = module('workers.classifier_task')
manager_module = module('utils.model_manager')
mapping = module('utils.class_mapping')
processing = module('processing.classify_algorithm')
venv = module('utils.venv_manager')
backends = module('core.backends')
config = module('config')
DEFAULT_CODES = [2, 10, 15, 14, 5, 64, 1]
PARAMETER_NAMES = ['INPUT', 'OUTPUT_FOLDER', 'SUFFIX', 'DEVICE', 'FIELD_NAME', 'LOAD_AS_LAYER',
                   'OUTPUT_CODES_JSON', 'TILE_ENABLED', 'TILE_SIZE_M', 'TILE_BUFFER_M',
                   'TILE_STREAMING', 'UNITS']


def make_source(path, extras=True):
    """Legacy RGB LAS with distinct attributes and scaled 64-bit extra data."""
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = [.01, .01, .01]
    header.offsets = [500000., 5600000., 0.]
    if extras:
        header.add_extra_dim(laspy.ExtraBytesParams(
            name='precise', type='uint64', scales=[.001], offsets=[100.]))
    header.vlrs.append(laspy.VLR(user_id='RAIL-test', record_id=42, record_data=b'keep-metadata'))
    cloud = laspy.LasData(header)
    cloud.points = laspy.ScaleAwarePointRecord.zeros(27, header=header)
    index = np.arange(27)
    cloud.X = index * 100
    cloud.Y = index % 3 * 100
    cloud.Z = index % 7 * 25
    cloud.intensity = index + 1000
    cloud.classification = np.full(27, 9, dtype=np.uint8)
    cloud.return_number = index % 3 + 1
    cloud.number_of_returns = np.full(27, 3, dtype=np.uint8)
    cloud.synthetic = index % 2
    cloud.key_point = index % 3 == 0
    cloud.withheld = index % 5 == 0
    cloud.scan_direction_flag = index % 2
    cloud.edge_of_flight_line = index % 7 == 0
    cloud.scan_angle_rank = index - 13
    cloud.user_data = index + 20
    cloud.point_source_id = index + 300
    cloud.gps_time = index + 100000.25
    cloud.red = index + 100
    cloud.green = index + 200
    cloud.blue = index + 300
    if extras:
        cloud.points.array['precise'] = np.uint64(2**63) + index.astype(np.uint64)
    cloud.write(path)
    return laspy.read(path)


def rail_predictions(xyz, progress=None, cancel=None):
    if progress is not None:
        progress(80.)
    return ((np.rint(xyz[:, 0]).astype(np.int64) - 500000) % 7 + 1).astype(np.int32)


class RecordingFeedback(QgsProcessingFeedback):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.warnings = []

    def pushInfo(self, info):  # noqa: N802
        self.messages.append(info)
        super().pushInfo(info)

    def pushWarning(self, warning):  # noqa: N802
        self.warnings.append(warning)
        super().pushWarning(warning)


class RailStubBackend:
    def __init__(self):
        self.loaded = []
        self.unloaded = 0

    def load(self, device):
        self.loaded.append(device)

    def predict(self, xyz, progress=None, cancel=None):
        return rail_predictions(xyz, progress, cancel)

    def unload(self):
        self.unloaded += 1


def expected_codes(overrides=None):
    codes = list(DEFAULT_CODES)
    for class_id, code in (overrides or {}).items():
        codes[class_id - 1] = code
    # make_source writes 27 points with x = 0..26 m.
    return np.array([codes[i % 7] for i in range(27)], dtype=np.uint8)


class FakeSettings:
    """QgsSettings stand-in backed by a dict: the user's profile is never touched."""
    store = {}

    def value(self, key, default=None, **kwargs):
        return self.store.get(key, default)

    def setValue(self, key, value):  # noqa: N802
        self.store[key] = value


class RailChecks(unittest.TestCase):
    def assert_preserved(self, before, after, classification=False):
        for name in ('X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                     'synthetic', 'key_point', 'withheld', 'scan_direction_flag',
                     'edge_of_flight_line', 'scan_angle_rank', 'user_data',
                     'point_source_id', 'gps_time', 'red', 'green', 'blue'):
            if name == 'scan_angle_rank' and name not in after.point_format.dimension_names:
                np.testing.assert_allclose(before[name], np.asarray(after.scan_angle) * .006, atol=.003, rtol=0)
            else:
                np.testing.assert_array_equal(before[name], after[name], err_msg=name)
        if 'precise' in before.point_format.dimension_names:
            np.testing.assert_array_equal(before.points.array['precise'], after.points.array['precise'])
        np.testing.assert_array_equal(before.header.scales, after.header.scales)
        np.testing.assert_array_equal(before.header.offsets, after.header.offsets)
        if classification:
            np.testing.assert_array_equal(before.classification, after.classification)
        self.assertTrue(any(v.user_id == 'RAIL-test' and v.record_data_bytes() == b'keep-metadata'
                            for v in after.vlrs))

    def assert_metadata(self, cloud, field, codes):
        records = [v for v in cloud.vlrs if v.user_id == 'Rail3D' and v.record_id == 1]
        self.assertEqual(len(records), 1)
        self.assertFalse([v for v in cloud.vlrs if v.user_id == 'AerialLiDAR'])
        data = json.loads(records[0].record_data_bytes().decode('utf8'))
        self.assertEqual(data['model'], RAIL.id)
        self.assertEqual(data['weights_sha256'], RAIL.weights_sha256)
        self.assertEqual(data['field'], field)
        self.assertEqual(data['encoding'], 'ASPRS')
        self.assertEqual(sorted(data['classes']), [str(i) for i in range(1, 8)])
        for class_id, code in codes.items():
            self.assertEqual(data['classes'][str(class_id)]['asprs'], code)
        self.assertEqual(data['classes']['2']['name'], 'Rail')

    def processing_params(self, algorithm, source, folder, stream=False, codes='{"2":200,"6":17}'):
        return {
            algorithm.INPUT: str(source),
            algorithm.OUTPUT_FOLDER: str(folder), algorithm.SUFFIX: '_processed',
            algorithm.DEVICE: 1, algorithm.FIELD_NAME: 'classification',
            algorithm.LOAD_AS_LAYER: False, algorithm.TILE_ENABLED: stream,
            algorithm.TILE_STREAMING: stream, algorithm.TILE_SIZE_M: 10.,
            algorithm.TILE_BUFFER_M: 0., algorithm.UNITS: 1,
            algorithm.OUTPUT_CODES_JSON: codes,
        }

    def processing_patches(self, stack, backend, cuda=True):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))
        stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch}))
        stack.enter_context(patch.object(venv, 'get_venv_status', return_value=(True, 'ready')))
        stack.enter_context(patch.object(venv, 'ensure_venv_packages_available'))
        stack.enter_context(patch.object(manager_module.ModelManager, 'is_model_available', return_value=True))
        stack.enter_context(patch.object(manager_module.ModelManager, 'get_model_path', return_value=Path('unused.pt')))
        return stack.enter_context(patch.object(backends, 'create_backend', return_value=backend))

    def run_processing(self, parameters, cuda=True, libraries_ready=True, context=None):
        """Processing run with stubbed inference; returns (engine id, warnings, infos)."""
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        readiness = module('utils.backend_readiness')
        with ExitStack() as stack:
            backend = RailStubBackend()
            factory = self.processing_patches(stack, backend, cuda=cuda)
            stack.enter_context(patch.object(
                readiness, 'litept_dependency_status',
                return_value=(libraries_ready, '' if libraries_ready else 'spconv missing')))
            feedback = RecordingFeedback()
            algorithm.processAlgorithm(parameters, context or QgsProcessingContext(), feedback)
            return factory.call_args.args[0].id, feedback.warnings, feedback.messages

    # ------------------------------------------------------------------
    # Registry, Processing interface, model card
    # ------------------------------------------------------------------

    def test_registry_lists_the_single_railway_model_and_classifier(self):
        self.assertEqual(registry.MODELS, (RAIL,))
        self.assertEqual(registry.CLASSIFIERS, (registry.RAILWAY,))
        self.assertEqual(registry.RAILWAY.id, 'railway')
        self.assertEqual(registry.RAILWAY.display_name, 'Railway classifier')
        self.assertEqual(registry.RAILWAY.engines, (RAIL,))
        self.assertIs(registry.RAILWAY.primary, RAIL)
        self.assertEqual(registry.DEFAULT_MODEL_ID, RAIL.id)
        self.assertEqual(registry.DEFAULT_CLASSIFIER_ID, 'railway')
        self.assertEqual(registry.get_classifier('railway'), registry.RAILWAY)
        self.assertEqual(registry.classifier_for_model(RAIL.id), registry.RAILWAY)
        self.assertEqual(RAIL.display_name, 'Rail 3D railway model (5 cm)')
        self.assertEqual(RAIL.short_name, 'Railway model')
        self.assertEqual(RAIL.supported_devices, ('cuda',))
        self.assertTrue(RAIL.editable_codes)
        self.assertTrue(mapping.supports_output_codes(RAIL))
        self.assertEqual(mapping.editable_model_names(), 'Railway model')
        # The former classifiers, engines and lock are gone with their names.
        for name in ('AIRBORNE', 'MOBILE_MAPPING', 'POWER_LINES', 'FALLBACK_MODEL_ID',
                     'default_model_for_device', 'model_locked'):
            self.assertFalse(hasattr(registry, name), name)
        for name in ('access', 'locked', 'access_message'):
            self.assertFalse(hasattr(registry.RAILWAY, name), name)
        with self.assertRaises(KeyError):
            registry.get_model('another_model')
        with self.assertRaises(KeyError):
            registry.get_classifier('airborne')
        # The engine selector returns the primary engine even where it cannot run.
        self.assertIs(registry.select_engine(registry.RAILWAY, 'cpu', lambda s, d: False), RAIL)
        self.assertIs(registry.select_engine(registry.RAILWAY, 'cuda', lambda s, d: True), RAIL)

    def test_processing_algorithm_has_no_classifier_or_model_parameter(self):
        self.assertFalse(hasattr(processing, 'ClassifyLidarAlgorithm'))
        self.assertFalse(hasattr(processing, '_ALG_DEPRECATED'))
        self.assertFalse(hasattr(processing, '_enum_index'))
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        self.assertEqual(algorithm.name(), 'classify_point_cloud')
        self.assertEqual(algorithm.displayName(), 'Classify railway point cloud')
        self.assertIsNone(algorithm.parameterDefinition('CLASSIFIER'))
        self.assertIsNone(algorithm.parameterDefinition('MODEL'))
        self.assertEqual([d.name() for d in algorithm.parameterDefinitions()], PARAMETER_NAMES)
        self.assertEqual(algorithm.parameterDefinition('DEVICE').options(),
                         ['Auto (GPU if available)', 'GPU (CUDA)', 'CPU', 'GPU (Apple MPS)'])
        self.assertEqual(algorithm.parameterDefinition('FIELD_NAME').defaultValue(), 'classification')
        self.assertTrue(algorithm.parameterDefinition('OUTPUT_CODES_JSON').flags() & processing._FLAG_ADVANCED)
        self.assertEqual([o.name() for o in algorithm.outputDefinitions()], ['OUTPUT_FOLDER', 'OUTPUT_FILE'])
        # Whatever the parameters, the railway classifier and its engine run.
        for parameters in ({}, {'CLASSIFIER': 0, 'MODEL': 3}):
            self.assertEqual(algorithm._requested_engines(parameters, QgsProcessingContext(),
                                                          QgsProcessingFeedback()),
                             (registry.RAILWAY, (RAIL,)))
        help_text = algorithm.shortHelpString()
        self.assertIn('Rail 3D', help_text)
        self.assertIn('Railway classifier', help_text)
        self.assertIn('Rail 3D railway model (5 cm)', help_text)
        self.assertIn('Rail 10', help_text)
        self.assertIn('NVIDIA CUDA GPU required', help_text)
        for absent in ('members only', 'maven.com', 'SegFormer', 'Airborne', 'Mobile mapping',
                       'Power lines', 'classify_lidar', 'MODEL index'):
            self.assertNotIn(absent, help_text, absent)
        provider = module('processing.provider').Rail3DProvider()
        self.assertEqual(provider.id(), 'rail3d')
        self.assertEqual(provider.name(), 'Rail 3D')
        registry_ = QgsApplication.processingRegistry()
        if registry_.providerById(provider.id()) is None:
            registry_.addProvider(provider)
        registered = registry_.algorithmById('rail3d:classify_point_cloud')
        self.assertIsNotNone(registered)
        self.assertIsNotNone(registry_.providerById('rail3d'))
        self.assertIsNone(registry_.algorithmById('rail3d:classify_lidar'))
        self.assertEqual(len(registry_.providerById('rail3d').algorithms()), 1)

    def test_default_codes_are_asprs_where_a_class_exists(self):
        codes = {i: c.asprs_code for i, c in RAIL.class_mapping.items()}
        self.assertEqual(codes, {1: 2, 2: 10, 3: 15, 4: 14, 5: 5, 6: 64, 7: 1})
        meanings = [mapping.asprs_code_name(c) for c in codes.values()]
        self.assertEqual(meanings, ['Ground', 'Rail', 'Transmission tower', 'Wire conductor (phase)',
                                    'High vegetation', 'User definable', 'Unclassified'])
        # The only user-defined default is the lowest user-definable code.
        self.assertEqual([c for c in codes.values() if c >= 64], [64])
        self.assertEqual([c.name for c in RAIL.class_mapping.values()],
                         ['Ground', 'Rail', 'Pole', 'Wire', 'Vegetation', 'Fence', 'Other'])

    def test_model_card_matches_the_registry(self):
        self.assertEqual(RAIL.weights_filename, 'rail3d_5cm_fp16.pth')
        self.assertEqual(RAIL.config_filename, 'rail3d_5cm.json')
        self.assertEqual(RAIL.config_path, ROOT / 'core' / 'rail3d_5cm.json')
        self.assertTrue(RAIL.config_path.is_file())
        card = json.loads(RAIL.config_path.read_text(encoding='utf-8'))
        self.assertEqual(card['id'], RAIL.id)
        self.assertEqual(card['display_name'], RAIL.display_name)
        self.assertEqual(card['family'], RAIL.family)
        self.assertEqual(card['weights_file'], RAIL.weights_filename)
        self.assertEqual(card['weights_bytes'], 171822701)
        self.assertAlmostEqual(card['weights_bytes'] / 1e6, RAIL.weights_size_mb, delta=0.1)
        self.assertNotIn('weights_sha256', card)
        # The one public URL the first use downloads from (SHA-256 verified).
        self.assertEqual(len(RAIL.weights_urls), 1)
        self.assertTrue(RAIL.weights_urls[0].startswith('https://github.com/akharroubi/'))
        self.assertEqual(RAIL.bundled_weights_path, ROOT / 'weights' / RAIL.weights_filename)

    def fresh_profile(self, stack, folder):
        """Point the model manager at an empty QGIS profile below ``folder``."""
        profile = Path(folder) / 'profile'
        profile.mkdir()
        stack.enter_context(patch.object(
            manager_module, 'QgsApplication',
            types.SimpleNamespace(qgisSettingsDirPath=lambda: str(profile) + os.sep)))
        FakeSettings.store = {}
        stack.enter_context(patch.object(manager_module, 'QgsSettings', FakeSettings))
        return profile

    def test_fresh_profile_has_no_weights_and_the_first_run_downloads_them(self):
        """The plugin folder ships no weights: a new profile starts without
        them, the first run fetches them from the registry URL (the download
        itself is stubbed here, no network), and the dock says so."""
        self.assertEqual(manager_module.ModelManager.PLUGIN_FOLDER_NAME, 'Rail3D')
        self.assertFalse(RAIL.bundled_weights_path.exists(),
                         f'{RAIL.bundled_weights_path}: the plugin no longer ships weights')
        self.assertEqual(config.SETTINGS_PREFIX, 'Rail3D')
        self.assertEqual(config.PLUGIN_NAME, 'Rail 3D')
        self.assertEqual(Path(venv.CACHE_DIR), CACHE_DIR)
        readiness = module('utils.backend_readiness')
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            profile = self.fresh_profile(stack, folder)
            manager = manager_module.ModelManager(RAIL)
            self.assertIsNone(manager.bundled_path())
            self.assertFalse(manager.is_model_available())
            cache = profile / 'Rail3D' / 'models' / RAIL.id / RAIL.weights_filename
            self.assertEqual(manager.get_model_path(), cache)
            self.assertEqual(manager.get_model_size_mb(), 0.0)
            self.assertEqual(manager.get_candidate_urls(), [RAIL.weights_urls[0]])
            self.assertEqual(manager.get_model_url(), RAIL.weights_urls[0])
            # ensure_available fetches: the transfer is replaced by a stub
            # that fails, so the message chain is exercised without network.
            logged = []
            stack.enter_context(patch.object(manager_module, 'log_info', logged.append))
            stack.enter_context(patch.object(manager_module, 'log_warning', logged.append))
            with patch.object(manager_module.ModelManager, '_download_one',
                              return_value=(False, 'Network error: offline')) as transfer:
                ok, message = manager.ensure_available(lambda r, t: None, lambda: False)
                transfer.assert_called_once()
                self.assertEqual(transfer.call_args.args[0], RAIL.weights_urls[0])
            self.assertFalse(ok)
            self.assertIn('Could not get the Rail 3D railway model (5 cm) weights', message)
            self.assertIn('Network error: offline', message)
            self.assertIn('Check the internet connection', message)
            self.assertIn('import the weights file with the folder icon', message)
            self.assertIn('Rail 3D railway model (5 cm): weights not found locally; downloading '
                          'them now (about 172 MB, once).', logged)
            self.assertIn('Downloading the Railway model weights (source 1/1)...', logged)
            # The log never shows the URL, and no file is left behind.
            self.assertFalse([m for m in logged if 'http' in m], logged)
            self.assertEqual([p.name for p in cache.parent.iterdir()], [])
            # A successful transfer is promoted into the profile cache.
            payload = b'released railway weights'
            spec = replace(RAIL, weights_sha256=hashlib.sha256(payload).hexdigest())
            manager = manager_module.ModelManager(spec)

            def deliver(url, tmp_path, progress_callback=None, cancel_callback=None):
                tmp_path.write_bytes(payload)
                return True, ''

            with patch.object(manager_module.ModelManager, '_download_one', side_effect=deliver):
                self.assertEqual(manager.ensure_available(), (True, ''))
            self.assertTrue(manager.is_model_available())
            self.assertEqual(manager.get_model_path(), cache)
            self.assertEqual(cache.read_bytes(), payload)
            self.assertEqual([p.name for p in cache.parent.iterdir()], [cache.name])
            with patch.object(manager, 'download_model') as download:
                self.assertEqual(manager.ensure_available(), (True, ''))
                download.assert_not_called()
            manager.delete_model()
            self.assertFalse(cache.exists())
            # The dock in the same fresh profile, with a CUDA GPU and the
            # libraries ready: not an error, the run downloads first.
            panel_module = module('gui.main_panel')
            stack.enter_context(patch.object(panel_module, 'QgsSettings', FakeSettings))
            stack.enter_context(patch.object(panel_module, 'get_gpu_info', return_value={
                'available': True, 'backend': 'cuda', 'name': 'Test GPU'}))
            stack.enter_context(patch.object(readiness, 'litept_dependency_status', return_value=(True, '')))
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertFalse(panel.model_ready)
                self.assertTrue(panel.model_gate_ok)
                self.assertIn('Weights download on first run (172 MB)', panel.model_label.text())
                self.assertIn('download automatically on first use', panel.import_btn.toolTip())
                self.assertFalse(panel.run_btn.isEnabled())
                panel.files = [Path(folder) / 'corridor.las']
                panel.output_widget.setFilePath(folder)
                panel._update_run_button()
                self.assertTrue(panel.run_btn.isEnabled())
                self.assertEqual(panel.run_btn.toolTip(), '')
                hint = panel.current_file_label.text()
                self.assertIn('Ready: 1 file(s), Railway classifier on GPU (CUDA).', hint)
                self.assertIn('Its weights (172 MB) download automatically first.', hint)
            finally:
                panel.close()
            self.assertFalse((profile / 'Rail3D' / 'models' / RAIL.id / RAIL.weights_filename).exists())

    def test_verified_offline_copy_in_the_plugin_weights_folder_is_used(self):
        """A verified file placed in <plugin>/weights/ (offline installs) is
        preferred to the profile cache; a wrong file there is ignored."""
        payload = b'released railway weights'
        spec = replace(RAIL, weights_sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            profile = self.fresh_profile(stack, folder)
            offline = Path(folder) / 'weights' / spec.weights_filename
            offline.parent.mkdir()
            stack.enter_context(patch.object(type(spec), 'bundled_weights_path',
                                             new_callable=PropertyMock, return_value=offline))
            manager = manager_module.ModelManager(spec)
            cache = profile / 'Rail3D' / 'models' / spec.id / spec.weights_filename
            self.assertEqual(spec.bundled_weights_path, offline)
            self.assertIsNone(manager.bundled_path())
            self.assertEqual(manager.get_model_path(), cache)
            # A wrong file in weights/ is not trusted.
            offline.write_bytes(b'tampered weights')
            self.assertIsNone(manager.bundled_path())
            self.assertFalse(manager.is_model_available())
            self.assertEqual(manager.get_model_path(), cache)
            # The released file is used in place, with no download.
            offline.write_bytes(payload)
            self.assertEqual(manager.bundled_path(), offline)
            self.assertEqual(manager.get_model_path(), offline)
            self.assertTrue(manager.is_model_available())
            self.assertAlmostEqual(manager.get_model_size_mb(), len(payload) / 1e6)
            with patch.object(manager, 'download_model') as download, \
                    patch.object(manager, 'get_candidate_urls', return_value=[]):
                self.assertEqual(manager.ensure_available(), (True, ''))
                download.assert_not_called()
            self.assertFalse(cache.exists())
            # Deleting the cache leaves the offline copy alone.
            manager.delete_model()
            self.assertEqual(offline.read_bytes(), payload)
            self.assertTrue(manager.is_model_available())
            # The dock reports the copy as ready.
            panel_module = module('gui.main_panel')
            stack.enter_context(patch.object(panel_module, 'QgsSettings', FakeSettings))
            stack.enter_context(patch.object(panel_module, 'get_gpu_info', return_value={'available': False}))
            classifier = replace(registry.RAILWAY, engines=(spec,))
            stack.enter_context(patch.object(panel_module, 'get_classifier', lambda _id: classifier))
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertIs(panel._current_spec(), spec)
                self.assertTrue(panel.model_ready)
                self.assertNotIn('download', panel.model_label.text())
                self.assertIn('download automatically on first use', panel.import_btn.toolTip())
            finally:
                panel.close()

    def test_extra_dimension_descriptions_fit_utf8_byte_limit(self):
        safety = module('utils.output_safety')
        for text in ('AI classification (' + RAIL.display_name + ')',
                     'a' * 31 + '\u2014 tail', '\u00e9' * 20, '\U0001f30d' * 9):
            description = safety.extra_dim_description(text)
            self.assertLessEqual(len(description.encode('utf8')), 32)
            self.assertTrue(text.startswith(description))
            header = laspy.LasHeader(point_format=6, version='1.4')
            header.add_extra_dim(laspy.ExtraBytesParams(name='rail_ids', type='int32', description=description))

    # ------------------------------------------------------------------
    # Classification task (dock path)
    # ------------------------------------------------------------------

    def test_task_modes_default_and_custom_codes_preserve_attributes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = make_source(source)
            source_bytes = source.read_bytes()
            for overrides in (None, {2: 200, 6: 17, 7: 0}):
                for mode in ('memory', 'tiled', 'streaming'):
                    with self.subTest(mode=mode, overrides=overrides):
                        output = folder / f"{mode}_{'custom' if overrides else 'default'}.laz"
                        task = tasks.ClassificationTask(
                            [source], folder, '_out', RAIL.id, 'cuda', 'classification',
                            tile_enabled=mode != 'memory', tile_auto=False, tile_size_m=10,
                            tile_buffer_m=1, tile_streaming=mode == 'streaming',
                            units_override='metre', output_codes=overrides)
                        task._process_one(source, output, rail_predictions, laspy, 0, 100)
                        result = laspy.read(output)
                        # 64 and 200 exceed the 5-bit legacy range: LAS 1.4 upgrade.
                        self.assertEqual(result.point_format.id, 7)
                        self.assertEqual(str(result.header.version), '1.4')
                        np.testing.assert_array_equal(result.classification, expected_codes(overrides))
                        self.assertEqual(list(result.point_format.extra_dimension_names), ['precise'])
                        self.assert_preserved(original, result)
                        written = {i + 1: c for i, c in enumerate(DEFAULT_CODES)}
                        written.update(overrides or {})
                        self.assert_metadata(result, 'classification', written)
                        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual(RAIL.class_mapping[6].asprs_code, 64)

    def test_legacy_file_stays_legacy_when_codes_fit_five_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = make_source(source, extras=False)
            output = folder / 'small_codes.las'
            task = tasks.ClassificationTask([source], folder, '_out', RAIL.id, 'cuda',
                                            'classification', units_override='metre',
                                            output_codes={6: 20})
            task._process_one(source, output, rail_predictions, laspy, 0, 100)
            result = laspy.read(output)
            self.assertEqual(result.point_format.id, 3)
            self.assertEqual(str(result.header.version), '1.2')
            np.testing.assert_array_equal(result.classification, expected_codes({6: 20}))
            self.assert_preserved(original, result)
            self.assert_metadata(result, 'classification', {6: 20})

    def test_rail_refuses_extra_class_field_without_writing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            make_source(source)
            original_bytes = source.read_bytes()
            for stream in (False, True):
                with self.assertRaisesRegex(ValueError, 'Railway model.*classification'):
                    tasks.ClassificationTask([source], folder, '_raw', RAIL.id, 'cuda', 'rail_ids',
                        tile_enabled=stream, tile_auto=False, tile_size_m=10, tile_buffer_m=0,
                        tile_streaming=stream, units_override='metre')
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(list(folder.iterdir()), [source])

    def test_task_refuses_cpu_before_loading_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.las'
            make_source(source)
            task = tasks.ClassificationTask([source], folder, '_cpu', RAIL.id, 'cpu', 'classification')
            with patch.object(tasks, '_get_torch'), patch.object(tasks, 'create_backend') as task_factory, \
                    patch.object(manager_module.ModelManager, 'ensure_available') as download:
                self.assertFalse(task.run())
                task_factory.assert_not_called()
                download.assert_not_called()
                self.assertIn("Rail 3D railway model (5 cm) does not run on 'cpu'", task.error_message)
                self.assertIn('NVIDIA CUDA GPU required', task.error_message)
            self.assertFalse((Path(folder) / 'input_cpu.las').exists())

    def test_outputs_open_in_qgis_without_new_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = make_source(source, extras=False)
            for stream in (False, True):
                for extension in ('.las', '.laz'):
                    with self.subTest(stream=stream, extension=extension):
                        output = folder / (('stream' if stream else 'memory') + extension)
                        task = tasks.ClassificationTask([source], folder, '_out', RAIL.id, 'cuda',
                            'classification', tile_enabled=stream, tile_auto=False, tile_size_m=10,
                            tile_buffer_m=0, tile_streaming=stream, units_override='metre')
                        task._process_one(source, output, rail_predictions, laspy, 0, 100)
                        written = laspy.read(output)
                        self.assertEqual(list(written.point_format.extra_dimension_names), [])
                        self.assert_preserved(original, written)
                        options = QgsPointCloudLayer.LayerOptions()
                        options.skipIndexGeneration = True
                        options.skipStatisticsCalculation = True
                        options.skipCrsValidation = True
                        layer = QgsPointCloudLayer(str(output), output.stem, 'pdal', options)
                        self.assertTrue(layer.isValid(), layer.error().message())
                        self.assertEqual(layer.dataProvider().pointCount(), 27)

    def test_classified_style_lists_every_rail_class(self):
        helpers = module('utils.helpers')
        categories = helpers._point_cloud_categories(RAIL.class_mapping)
        self.assertEqual([c.value() for c in categories], [1, 2, 5, 10, 14, 15, 64])
        labels = {c.value(): c.label() for c in categories}
        self.assertEqual(labels[10], 'Rail')
        self.assertEqual(labels[64], 'Fence')
        self.assertEqual(labels[2], 'Ground')
        colours = {c.value(): c.color().name().upper() for c in categories}
        self.assertEqual(colours[10], '#E31A1C')
        merged = mapping.output_model_spec(RAIL, {3: 14})
        labels = {c.value(): c.label() for c in helpers._point_cloud_categories(merged.class_mapping)}
        self.assertEqual(labels[14], 'Pole / Wire')

    # ------------------------------------------------------------------
    # Processing algorithm
    # ------------------------------------------------------------------

    def test_processing_parses_codes_and_writes_classification(self):
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        for stream, codes, overrides in ((False, '{"2":200,"6":17}', {2: 200, 6: 17}),
                                         (True, '', None)):
            with self.subTest(stream=stream), tempfile.TemporaryDirectory() as folder, \
                    ExitStack() as stack:
                folder = Path(folder)
                source = folder / 'input.las'
                original = make_source(source)
                backend = RailStubBackend()
                factory = self.processing_patches(stack, backend)
                feedback = RecordingFeedback()
                result = algorithm.processAlgorithm(
                    self.processing_params(algorithm, source, folder, stream, codes),
                    QgsProcessingContext(), feedback)
                written = laspy.read(result[algorithm.OUTPUT_FILE])
                self.assertEqual(Path(result[algorithm.OUTPUT_FILE]), folder / 'input_processed.las')
                self.assertEqual(factory.call_args.args[0].id, RAIL.id)
                self.assertEqual(backend.loaded, ['cuda'])
                self.assertEqual(backend.unloaded, 1)
                np.testing.assert_array_equal(written.classification, expected_codes(overrides))
                self.assert_preserved(original, written)
                expected = {i + 1: c for i, c in enumerate(DEFAULT_CODES)}
                expected.update(overrides or {})
                self.assert_metadata(written, 'classification', expected)
                self.assertTrue(any(m.startswith('Railway model output codes:')
                                    for m in feedback.messages), feedback.messages)
                self.assertIn('Classifier: Railway classifier', feedback.messages)
                self.assertIn('Model: Rail 3D railway model (5 cm)', feedback.messages)
                self.assertFalse([m for m in feedback.messages if 'Downloading' in m], feedback.messages)

    def test_processing_ignores_legacy_classifier_and_model_keys(self):
        """Scripts written for the multi-model plugin keep working: the keys
        are not parameters any more and the railway model runs."""
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.las'
            make_source(source)
            params = self.processing_params(algorithm, source, folder, codes='')
            for extra in ({}, {'CLASSIFIER': 0}, {'CLASSIFIER': 2, 'MODEL': 3}, {'MODEL': 1}):
                with self.subTest(extra=extra):
                    engine, warnings, infos = self.run_processing(dict(params, **extra))
                    self.assertEqual(engine, RAIL.id)
                    self.assertFalse([w for w in warnings if 'engine' in w], warnings)
            # Dependencies missing or CPU only: there is no other engine to fall
            # back to; the device gate names the model.
            with self.assertRaisesRegex(QgsProcessingException, "Rail 3D railway model.*does not run on 'cpu'"):
                self.run_processing(dict(params, **{algorithm.DEVICE: 2}))
            with self.assertRaisesRegex(QgsProcessingException, "does not run on 'cpu'"):
                self.run_processing(dict(params, **{algorithm.DEVICE: 0}), cuda=False)
            with self.assertRaisesRegex(QgsProcessingException, 'CUDA was requested'):
                self.run_processing(dict(params, **{algorithm.DEVICE: 1}), cuda=False)

    def test_processing_rejects_bad_codes_field_and_cpu_before_inference(self):
        algorithm = processing.ClassifyPointCloudAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            source = Path(folder) / 'input.las'
            make_source(source)
            factory = self.processing_patches(stack, RailStubBackend())
            parameters = self.processing_params(algorithm, source, folder)
            # ID 8 existed for another model of the original plugin, not here.
            for invalid in ('{"8":3}', '{"9":1}', '{"6":256}', '{"0":3}', '{"6":true}', '[2,3]', '{broken}'):
                with self.subTest(value=invalid), self.assertRaises(QgsProcessingException):
                    parameters[algorithm.OUTPUT_CODES_JSON] = invalid
                    algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            parameters[algorithm.OUTPUT_CODES_JSON] = ''
            parameters[algorithm.FIELD_NAME] = 'rail_extra'
            with self.assertRaisesRegex(QgsProcessingException, 'Railway model.*classification'):
                algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            factory.assert_not_called()
            self.assertFalse((Path(folder) / 'input_processed.las').exists())
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            source = Path(folder) / 'input.las'
            make_source(source)
            factory = self.processing_patches(stack, RailStubBackend(), cuda=False)
            parameters = self.processing_params(algorithm, source, folder)
            parameters[algorithm.DEVICE] = 0
            with self.assertRaisesRegex(QgsProcessingException, "railway model.*does not run on 'cpu'"):
                algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            factory.assert_not_called()
            self.assertFalse((Path(folder) / 'input_processed.las').exists())

    def test_graphical_modeler_editor_keeps_the_step_parameters(self):
        """Opening and accepting a saved model step keeps its values, with no
        classifier or model choice to lose.

        Uses QGIS's own ModelerParametersDialog, as a user editing a step does.
        """
        from qgis.core import (QgsProcessingModelAlgorithm,
                               QgsProcessingModelChildAlgorithm,
                               QgsProcessingModelChildParameterSource)
        plugins_dir = Path(QgsApplication.pkgDataPath()) / 'python' / 'plugins'
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        from processing.core.Processing import Processing
        Processing.initialize()
        from processing.modeler.ModelerParametersDialog import ModelerParametersDialog
        provider = module('processing.provider').Rail3DProvider()
        registry_ = QgsApplication.processingRegistry()
        if registry_.providerById(provider.id()) is None:
            registry_.addProvider(provider)
        algorithm = registry_.algorithmById('rail3d:classify_point_cloud')
        self.assertIsNotNone(algorithm)
        saved = {'INPUT': 'input.las', 'OUTPUT_FOLDER': '.', 'DEVICE': 1, 'FIELD_NAME': 'classification',
                 'OUTPUT_CODES_JSON': '{"6": 70}', 'TILE_ENABLED': True, 'TILE_STREAMING': True,
                 'TILE_SIZE_M': 250.0, 'UNITS': 1}
        model = QgsProcessingModelAlgorithm()
        child = QgsProcessingModelChildAlgorithm(algorithm.id())
        child.setChildId('step')
        for key, value in saved.items():
            child.addParameterSources(key, [QgsProcessingModelChildParameterSource.fromStaticValue(value)])
        model.addChildAlgorithm(child)
        dialog = ModelerParametersDialog(algorithm, model, 'step')
        try:
            edited = dialog.widget.widget.createAlgorithm()
        finally:
            dialog.close()
        values = {key: sources[0].staticValue()
                  for key, sources in edited.parameterSources().items() if sources}
        self.assertNotIn('CLASSIFIER', values)
        self.assertNotIn('MODEL', values)
        self.assertEqual(int(values['DEVICE']), 1, values)
        self.assertEqual(int(values['UNITS']), 1, values)
        self.assertEqual(values['OUTPUT_CODES_JSON'], '{"6": 70}', values)
        self.assertEqual(values['FIELD_NAME'], 'classification', values)
        self.assertTrue(bool(values['TILE_ENABLED']), values)
        self.assertTrue(bool(values['TILE_STREAMING']), values)
        self.assertEqual(float(values['TILE_SIZE_M']), 250.0, values)
        resolved = algorithm._requested_engines(values, QgsProcessingContext(), QgsProcessingFeedback())
        self.assertEqual(resolved, (registry.RAILWAY, (RAIL,)))

    # ------------------------------------------------------------------
    # Weights download (first use) and import
    # ------------------------------------------------------------------

    def test_download_ignores_queued_progress_after_feedback_destruction(self):
        from qgis.PyQt import sip
        from qgis.PyQt.QtCore import QByteArray

        class Signal:
            def connect(self, callback):
                self.queued_callback = callback
                self.disconnected = False

            def disconnect(self, callback):
                self.disconnected = callback is self.queued_callback

        class Request:
            ErrorCode = types.SimpleNamespace(NoError=0)

            def __init__(self, error):
                self.downloadProgress = Signal()
                self.error = error

            def get(self, *args):
                self.downloadProgress.queued_callback(1, 2)
                return self.error

            def errorMessage(self):
                return 'test network failure'

            def reply(self):
                return types.SimpleNamespace(content=lambda: QByteArray(b'weights'))

        for outcome in ('success', 'network_error', 'cancelled'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as folder:
                caller_feedback = QgsProcessingFeedback()
                if outcome == 'cancelled':
                    caller_feedback.cancel()
                request = Request(1 if outcome == 'network_error' else 0)

                # Keep a queued callback even after disconnect, as Qt can do.
                class RequestFactory:
                    ErrorCode = Request.ErrorCode

                    def __new__(cls):
                        return request

                with patch.object(manager_module, 'QgsBlockingNetworkRequest', RequestFactory):
                    ok, message = manager_module.ModelManager._download_one(
                        'http://localhost/weights', Path(folder) / 'weights',
                        lambda r, t: caller_feedback.setProgress(100 * r / t),
                        caller_feedback.isCanceled,
                    )
                self.assertEqual(ok, outcome == 'success', message)
                self.assertTrue(request.downloadProgress.disconnected)
                sip.delete(caller_feedback)
                # A stale slot must never call the deleted Processing object.
                request.downloadProgress.queued_callback(2, 2)

    def test_weights_fallback_download_verify_import_and_refuse_bad_files(self):
        """First use: real QGIS network download from a local server into
        the profile cache, SHA-256 verified, as the registry URL would be
        used."""
        import http.server
        import threading
        payload = b'released railway weights'
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            served = folder / 'served'
            served.mkdir()
            (served / RAIL.weights_filename).write_bytes(payload)
            (served / 'tampered.pth').write_bytes(b'tampered weights')

            class Quiet(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(served), **kwargs)

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                spec = replace(RAIL, weights_sha256=hashlib.sha256(payload).hexdigest(),
                               weights_urls=(f'{base}/missing.pth', f'{base}/tampered.pth',
                                             f'{base}/{RAIL.weights_filename}'))
                manager = manager_module.ModelManager(spec)
                cache = folder / 'cache' / spec.weights_filename
                cache.parent.mkdir()
                progress = []
                with patch.object(manager, 'bundled_path', return_value=None), \
                     patch.object(manager, '_cache_path', return_value=cache), \
                     patch.object(manager, 'models_root', return_value=folder / 'cache'), \
                     patch.object(manager, 'get_model_url', return_value=''):
                    self.assertFalse(manager.is_model_available())
                    self.assertEqual(manager.get_model_path(), cache)
                    # A 404 and a wrong-SHA file are skipped; the third URL wins.
                    ok, message = manager.ensure_available(lambda r, t: progress.append((r, t)))
                    self.assertTrue(ok, message)
                    self.assertEqual(cache.read_bytes(), payload)
                    self.assertTrue(manager.is_model_available())
                    self.assertTrue(progress and progress[-1][0] == len(payload))
                    self.assertEqual(sorted(p.name for p in cache.parent.iterdir()), [cache.name])
                    # Already present: no second download.
                    with patch.object(manager, 'download_model') as download:
                        self.assertEqual(manager.ensure_available(), (True, ''))
                        download.assert_not_called()
                    # Damaged cache is not trusted, then repaired by a download.
                    cache.write_bytes(b'damaged cached weights')
                    self.assertFalse(manager.is_model_available())
                    self.assertTrue(manager.ensure_available()[0])
                    self.assertEqual(cache.read_bytes(), payload)
                    # Only tampered or missing files: refused, nothing promoted.
                    manager.delete_model()
                    bad = replace(spec, weights_urls=(f'{base}/missing.pth', f'{base}/tampered.pth'))
                    manager.spec = bad
                    ok, message = manager.ensure_available()
                    self.assertFalse(ok)
                    self.assertIn('SHA-256 mismatch', message)
                    self.assertFalse(cache.exists())
                    self.assertEqual(list(cache.parent.iterdir()), [])
                    # Offline import: verified copy, wrong file refused.
                    ok, message = manager.import_file(str(served / 'tampered.pth'))
                    self.assertFalse(ok)
                    self.assertFalse(cache.exists())
                    ok, message = manager.import_file(str(served / RAIL.weights_filename))
                    self.assertTrue(ok, message)
                    self.assertEqual(cache.read_bytes(), payload)
                    # No URL at all: an import instruction, never a silent failure.
                    manager.delete_model()
                    manager.spec = replace(spec, weights_urls=())
                    with patch.object(manager, 'download_model') as download:
                        ok, message = manager.ensure_available()
                        download.assert_not_called()
                    self.assertFalse(ok)
                    self.assertIn('Import', message)
            finally:
                server.shutdown()
                server.server_close()

    # ------------------------------------------------------------------
    # Dock
    # ------------------------------------------------------------------

    def dock_patches(self, stack, settings, gpu=None):
        panel_module = module('gui.main_panel')
        FakeSettings.store = settings
        stack.enter_context(patch.object(panel_module, 'QgsSettings', FakeSettings))
        stack.enter_context(patch.object(panel_module, 'get_gpu_info', return_value=gpu or {'available': False}))
        stack.enter_context(patch.object(manager_module.ModelManager, 'is_model_available', return_value=True))
        return panel_module

    def test_dock_opens_on_the_railway_model_with_editable_codes(self):
        prefix = config.SETTINGS_PREFIX
        # Settings of the original plugin's profile: another classifier and a
        # raw-ID field name. Neither applies to Rail 3D.
        settings = {f'{prefix}/settings_version': 3, f'{prefix}/classifier_id': 'airborne',
                    f'{prefix}/model_id': 'another_model',
                    f'{prefix}/field_name': 'my_raw_ids',
                    f'{prefix}/airborne_field_name': 'my_raw_ids'}
        with ExitStack() as stack:
            panel_module = self.dock_patches(stack, settings)
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(panel.windowTitle(), 'Rail 3D')
                self.assertEqual(panel.objectName(), 'Rail3DDock')
                for gone in ('model_combo', 'access_card', 'cohort_card', '_airborne_field_name',
                             '_select_initial_model', '_on_model_changed'):
                    self.assertFalse(hasattr(panel, gone), gone)
                self.assertIs(panel._current_classifier(), registry.RAILWAY)
                self.assertIs(panel_module.ClassifierDockWidget._current_classifier(), registry.RAILWAY)
                self.assertEqual(panel.model_name_label.text(), 'Railway classifier')
                self.assertIs(panel._current_spec(), RAIL)
                self.assertFalse(panel.mms_codes_row.isHidden())
                self.assertEqual(panel.field_edit.text(), 'classification')
                self.assertFalse(panel.field_edit.isEnabled())
                self.assertTrue(panel.import_btn.isEnabled())
                self.assertTrue(panel.model_ready)
                # No GPU on the test machine: the device gate explains, run is off.
                self.assertFalse(panel.model_gate_ok)
                self.assertIn('NVIDIA CUDA GPUs only', panel.model_gate_message)
                self.assertIn('Needs an NVIDIA GPU', panel.model_label.text())
                self.assertFalse(panel.run_btn.isEnabled())
                self.assertIn('Add LAS / LAZ / COPC files', panel.run_btn.toolTip())
                panel.files = [Path('corridor.las')]
                panel._update_run_button()
                self.assertEqual(panel.run_btn.toolTip(), 'Choose an output folder.')
                panel.output_widget.setFilePath(str(ROOT))
                panel._update_run_button()
                self.assertFalse(panel.run_btn.isEnabled())
                self.assertEqual(panel.run_btn.toolTip(), panel.model_gate_message)
                self.assertNotIn('members only', panel.run_btn.toolTip())
                panel.files = []
                panel.output_widget.setFilePath('')
                panel._update_run_button()
                tip = panel.model_name_label.toolTip()
                self.assertIn('<b>Railway classifier</b>', tip)
                self.assertIn('Rail 3D railway model (5 cm), 7 classes', tip)
                self.assertIn("Rail</td><td align='right'>&nbsp;&nbsp;10</td><td>&nbsp;&nbsp;Rail<", tip)
                self.assertIn("Fence</td><td align='right'>&nbsp;&nbsp;64</td><td>&nbsp;&nbsp;User definable", tip)
                self.assertIn('#E31A1C', tip)
                self.assertIn('Codes are yours to change', tip)
                self.assertNotIn('<i>', tip)
                self.assertTrue(panel.model_label.toolTip().endswith(tip))
                self.assertEqual(panel._output_codes_by_model, {RAIL.id: dict(enumerate(DEFAULT_CODES, start=1))})
                self.assertIn('Fence: 64', panel.mms_codes_btn.toolTip())
                panel._save_settings()
                self.assertEqual(settings[f'{prefix}/field_name'], 'classification')
                self.assertEqual(settings[f'{prefix}/airborne_field_name'], 'my_raw_ids')
            finally:
                panel.close()
        self.assertFalse([k for k in settings if not k.startswith('Rail3D/')], settings)

    def test_dock_gate_opens_with_a_cuda_gpu_and_the_model_libraries(self):
        readiness = module('utils.backend_readiness')
        with ExitStack() as stack:
            panel_module = self.dock_patches(stack, {}, gpu={'available': True, 'backend': 'cuda',
                                                            'name': 'Test GPU'})
            stack.enter_context(patch.object(readiness, 'litept_dependency_status', return_value=(True, '')))
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertTrue(panel.gpu_check.isChecked())
                self.assertTrue(panel.model_gate_ok)
                self.assertEqual(panel.model_gate_message, '')
                self.assertIs(panel._current_spec(), RAIL)
                # Files are still missing, so the run button waits for them.
                self.assertFalse(panel.run_btn.isEnabled())
                self.assertNotEqual(panel.run_btn.toolTip(), '')
                # GPU use switched off: the gate says what to enable.
                panel.gpu_check.setChecked(False)
                self.assertFalse(panel.model_gate_ok)
                self.assertIn("Enable 'Use GPU'", panel.model_gate_message)
                self.assertIn('Test GPU', panel.model_gate_message)
                panel.gpu_check.setChecked(True)
                self.assertTrue(panel.model_gate_ok)
            finally:
                panel.close()
            stack.enter_context(patch.object(readiness, 'litept_dependency_status',
                                             return_value=(False, 'spconv missing')))
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertFalse(panel.model_gate_ok)
                self.assertEqual(panel.model_gate_message, 'spconv missing')
                self.assertIn('Model libraries missing', panel.model_label.text())
            finally:
                panel.close()

    def test_dock_editor_accept_reset_and_persistence(self):
        dialog_module = module('dialogs.class_mapping_dialog')
        prefix = config.SETTINGS_PREFIX
        settings = {f'{prefix}/settings_version': 3,
                    f'{prefix}/output_codes/{RAIL.id}': json.dumps({'6': 65})}
        with ExitStack() as stack:
            panel_module = self.dock_patches(stack, settings)
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(panel._output_codes_by_model[RAIL.id][6], 65)
                self.assertIn('Fence: 65', panel.mms_codes_btn.toolTip())
                self.assertIn("Fence</td><td align='right'>&nbsp;&nbsp;65</td><td>&nbsp;&nbsp;User definable",
                              panel.model_name_label.toolTip())
                editor_error = []

                def edit_and_accept():
                    dialog = app.activeModalWidget()
                    try:
                        self.assertIsInstance(dialog, dialog_module.ClassMappingDialog)
                        self.assertEqual(dialog.windowTitle(), 'Railway model output class codes')
                        self.assertEqual(len(dialog._spins), 7)
                        self.assertEqual(dialog._spins[6].maximum(), 255)
                        self.assertEqual(dialog._spins[6].value(), 65)
                        self.assertEqual(dialog.table.item(1, 1).text(), 'Rail')
                        self.assertEqual(dialog.table.item(1, 4).text(), 'Rail')
                        self.assertEqual(dialog.table.item(5, 4).text(), 'User definable')
                        dialog._spins[2].setValue(30)
                        self.assertEqual(dialog.table.item(1, 4).text(), 'Reserved')
                        dialog._spins[2].setValue(200)
                        dialog._spins[7].setValue(0)
                        dialog.accept()
                    except BaseException as error:
                        editor_error.append(error)
                        if dialog is not None:
                            dialog.reject()

                QTimer.singleShot(0, edit_and_accept)
                panel._edit_output_codes()
                if editor_error:
                    raise editor_error[0]
                self.assertEqual(panel._output_codes_by_model[RAIL.id],
                                 {1: 2, 2: 200, 3: 15, 4: 14, 5: 5, 6: 65, 7: 0})
                saved = json.loads(settings[f'{prefix}/output_codes/{RAIL.id}'])
                self.assertEqual(saved, {'1': 2, '2': 200, '3': 15, '4': 14, '5': 5, '6': 65, '7': 0})
                self.assertIn('Rail: 200', panel.mms_codes_btn.toolTip())
                self.assertIn("Rail</td><td align='right'>&nbsp;&nbsp;200</td><td>&nbsp;&nbsp;User definable",
                              panel.model_name_label.toolTip())
                self.assertFalse(panel.mms_codes_row.isHidden())
                self.assertFalse(panel.field_edit.isEnabled())
            finally:
                panel.close()
            restored = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(restored._output_codes_by_model[RAIL.id][2], 200)
                self.assertEqual(restored._output_codes_by_model[RAIL.id][6], 65)
                dialog = dialog_module.ClassMappingDialog(RAIL, restored._output_codes_by_model[RAIL.id])
                capture_dir = os.environ.get('ALC_UI_CAPTURE_DIR')
                if capture_dir:
                    capture_dir = Path(capture_dir)
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    dialog.show()
                    app.processEvents()
                    self.assertTrue(dialog.grab().save(str(capture_dir / 'rail_output_codes.png')))
                self.assertEqual(dialog.output_codes()[2], 200)
                dialog._reset_defaults()
                self.assertEqual(dialog.output_codes(), dict(enumerate(DEFAULT_CODES, start=1)))
                self.assertEqual(dialog.table.item(5, 4).text(), 'User definable')
                dialog.close()
                # Invalid saved codes fall back to the defaults instead of breaking the dock.
                settings[f'{prefix}/output_codes/{RAIL.id}'] = '{"9": 1}'
            finally:
                restored.close()
            fallback = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(fallback._output_codes_by_model[RAIL.id], dict(enumerate(DEFAULT_CODES, start=1)))
            finally:
                fallback.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
