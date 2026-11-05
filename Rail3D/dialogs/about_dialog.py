"""About dialog for the Rail 3D plugin."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..config import PLUGIN_NAME, PLUGIN_VERSION


REPO_URL = "https://github.com/akharroubi/futurail3d"


class AboutDialog(QDialog):
    """About / credits dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {PLUGIN_NAME}")
        self.setMinimumSize(520, 440)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        body = QWidget()
        scroll.setWidget(body)

        layout = QVBoxLayout(body)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel(PLUGIN_NAME)
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version = QLabel(f"Version {PLUGIN_VERSION}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        layout.addSpacing(8)

        # Description
        desc = QLabel(
            "Classification of railway corridor LiDAR point clouds "
            "(LAS / LAZ / COPC) with a deep-learning model, written to the "
            "LAS classification field.\n\n"
            "Seven classes: ground, rail, pole, wire, vegetation, fence and "
            "other. Every output code is editable. NVIDIA GPU required."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        layout.addSpacing(12)

        # Model credit blocks, from the registry so they never go stale.
        from ..core.registry import MODELS
        for spec in MODELS:
            model_block = QLabel(
                "<div style='text-align:center;'>"
                f"<b>{spec.display_name}</b><br>"
                f"{spec.attribution}<br>"
                f"<small>{spec.licence}</small>"
                "</div>"
            )
            model_block.setWordWrap(True)
            model_block.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction
            )
            layout.addWidget(model_block)
            layout.addSpacing(6)

        # Plugin credit
        plugin_block = QLabel(
            "<div style='text-align:center;'>"
            "<b>QGIS plugin</b><br>"
            "Abderrazzaq Kharroubi<br>"
            "GeoScITY Lab - University of Liege<br>"
            f'<a href="{REPO_URL}">{REPO_URL}</a><br>'
            "<small>Plugin code: GPL-3.0. Third-party notices: "
            "THIRD_PARTY_NOTICES.md in the plugin folder.</small>"
            "</div>"
        )
        plugin_block.setWordWrap(True)
        plugin_block.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plugin_block.setOpenExternalLinks(True)
        layout.addWidget(plugin_block)

        layout.addStretch()

        # Close button (lives outside the scroll area)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        outer.addWidget(close)
