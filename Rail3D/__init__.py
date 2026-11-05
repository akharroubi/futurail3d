"""Rail 3D - QGIS plugin.

Deep-learning classification of railway corridor LiDAR point clouds
(ground, rails, poles, wires, vegetation, fences, other). See
core/registry.py for the model and its licence.
"""

import os

from qgis.PyQt.QtCore import QCoreApplication, QTranslator, QLocale, QSettings


PLUGIN_DIR = os.path.dirname(__file__)
_translator = None


def _install_translator() -> None:
    """Load the .qm translation matching the QGIS locale, if available."""
    global _translator

    locale = QSettings().value("locale/userLocale", QLocale().name()) or "en"
    locale = str(locale)[:2]

    qm_path = os.path.join(
        PLUGIN_DIR, "i18n", f"rail3d_{locale}.qm"
    )
    if not os.path.exists(qm_path):
        return

    translator = QTranslator()
    if translator.load(qm_path):
        QCoreApplication.installTranslator(translator)
        _translator = translator


def classFactory(iface):  # noqa: N802 - required by QGIS API
    _install_translator()
    from .plugin import Rail3DPlugin
    return Rail3DPlugin(iface)


def tr(message: str, context: str = "Rail3D") -> str:
    """Translate a string via Qt's translation system."""
    return QCoreApplication.translate(context, message)
