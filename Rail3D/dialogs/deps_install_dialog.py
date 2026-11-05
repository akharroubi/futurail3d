"""Dependency installation dock widget for the Rail 3D QGIS plugin.

Provides a user-facing panel for installing AI dependencies on first use.
Shows GPU detection info, progress bars, and status messages during
the installation process.
"""

import os
import sys

from ..config import PLUGIN_NAME

from qgis.PyQt.QtCore import QThread, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDockWidget,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _GpuProbe(QThread):
    """detect_nvidia_gpu() off the GUI thread: nvidia-smi can take up to
    20 s to answer (a laptop GPU waking up). Its result is cached, so the
    install that follows does not ask again."""

    detected = pyqtSignal(bool, object)

    def run(self):
        try:
            from ..utils.venv_manager import detect_nvidia_gpu
            has_gpu, gpu_info = detect_nvidia_gpu()
        except Exception:
            has_gpu, gpu_info = False, {}
        self.detected.emit(bool(has_gpu), dict(gpu_info or {}))


class DepsInstallDockWidget(QDockWidget):
    """Dock widget for managing dependency installation.

    Signals:
        install_requested: Emitted when user clicks Install Dependencies.
        cancel_requested: Emitted when user clicks Cancel.
    """

    install_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the dock widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(f"{PLUGIN_NAME} - Setup", parent)
        self.setObjectName("Rail3DDepsInstallDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Welcome section
        welcome_group = QGroupBox("Welcome")
        welcome_layout = QVBoxLayout(welcome_group)

        welcome_text = QLabel(
            f"{PLUGIN_NAME} needs to install its AI "
            "dependencies before you can run it.\n\n"
            "This is a one-time setup, with nothing to configure. It will:\n"
            "  \u2022 Download a Python runtime (~50 MB)\n"
            "  \u2022 Download a fast package installer (~15 MB)\n"
            "  \u2022 Install PyTorch + LiDAR packages (about 3 GB)\n"
            "  \u2022 Download the model (about 170 MB)\n\n"
            "QGIS stays usable while it runs (usually 5 to 15 minutes)."
        )
        welcome_text.setWordWrap(True)
        welcome_layout.addWidget(welcome_text)

        from ..utils.venv_manager import CACHE_DIR

        _home = os.path.expanduser("~")
        _display = (
            ("~" + CACHE_DIR[len(_home):])
            if CACHE_DIR.startswith(_home)
            else CACHE_DIR
        )
        location_label = QLabel(
            f"<small>Installation location: <code>{_display}</code></small>"
        )
        location_label.setWordWrap(True)
        welcome_layout.addWidget(location_label)

        layout.addWidget(welcome_group)

        # GPU info section
        self.gpu_group = QGroupBox("GPU Detection")
        gpu_layout = QVBoxLayout(self.gpu_group)

        self.gpu_label = QLabel("Detecting GPU...")
        self.gpu_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_label)

        layout.addWidget(self.gpu_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.install_button = QPushButton("Install Dependencies")
        self.install_button.setMinimumHeight(36)
        self.install_button.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 13px; }"
        )
        self.install_button.clicked.connect(self.install_requested.emit)
        button_layout.addWidget(self.install_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.hide()
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Progress section
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)

        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)

        # Status section
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Reinstall button (shown after errors)
        self.reinstall_button = QPushButton("Reinstall Dependencies")
        self.reinstall_button.setMinimumHeight(36)
        self.reinstall_button.clicked.connect(self._on_reinstall_clicked)
        self.reinstall_button.hide()
        layout.addWidget(self.reinstall_button)

        # Help links
        help_label = QLabel(
            "<small>"
            "If you encounter issues, please check the "
            '<a href="https://github.com/akharroubi/futurail3d">'
            "plugin documentation</a> or report a bug on "
            '<a href="https://github.com/akharroubi/futurail3d/issues">'
            "GitHub</a>."
            "</small>")
        help_label.setWordWrap(True)
        help_label.setOpenExternalLinks(True)
        layout.addWidget(help_label)

        layout.addStretch()
        self.setWidget(container)

        # Detect GPU on creation
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect the GPU in a background thread and update the GPU info
        label when it answers."""
        from ..utils.thread_lifetime import retain_thread

        self._gpu_detected = False
        self.gpu_label.setText("Detecting the GPU...")
        self._gpu_probe = retain_thread(_GpuProbe())
        self._gpu_probe.detected.connect(self._show_gpu)
        self._gpu_probe.start()

    def _show_gpu(self, has_gpu, gpu_info):
        """Update the GPU info label from the detection result."""
        self._gpu_detected = False
        try:
            from ..utils.venv_manager import (
                _select_cuda_index, _spconv_extra_for, _spconv_is_fallback)

            self._gpu_detected = bool(has_gpu)
            if has_gpu:
                name = gpu_info.get("name", "Unknown GPU")
                memory_mb = gpu_info.get("memory_mb")
                memory_str = f" ({memory_mb} MB)" if memory_mb else ""
                cuda_index = _select_cuda_index(gpu_info)
                if cuda_index and _spconv_is_fallback(cuda_index):
                    # RTX 50 series (CUDA 12.8 torch): spconv's CUDA 12.6
                    # build, compiled for the GPU by the driver on first use.
                    models = (
                        "The GPU (CUDA) version will be installed. On this GPU "
                        "generation the model uses spconv's CUDA 12.6 build; "
                        "the NVIDIA driver compiles it for the GPU the first "
                        "time the model runs (10 to 30 s)."
                    )
                elif cuda_index and not _spconv_extra_for(cuda_index):
                    # A CUDA toolkit spconv has no build for at all.
                    models = (
                        "The GPU (CUDA) version will be installed. The model "
                        "needs spconv, which has no build for this GPU "
                        "generation yet: the model cannot run on it."
                    )
                else:
                    models = "The GPU (CUDA) version will be installed."
                self.gpu_label.setText(f"GPU detected: {name}{memory_str}\n{models}")
                self.gpu_group.setStyleSheet(
                    "QGroupBox { border: 1px solid #4CAF50; "
                    "border-radius: 4px; padding: 8px; margin-top: 8px; }"
                )
            else:
                if sys.platform == "darwin":
                    msg = (
                        "No NVIDIA GPU detected (CUDA is not available on macOS).\n"
                        f"{PLUGIN_NAME} needs an NVIDIA GPU: the model cannot run "
                        "on this computer. The CPU packages can still be installed.")
                else:
                    msg = (
                        "No NVIDIA GPU detected.\n"
                        f"{PLUGIN_NAME} needs an NVIDIA GPU with a current driver: "
                        "the model cannot run on this computer. If the computer has "
                        "an NVIDIA GPU, install its driver from nvidia.com, restart "
                        "and open the plugin again. The CPU packages can still be "
                        "installed.")
                self.gpu_label.setText(msg)
                self.gpu_group.setStyleSheet(
                    "QGroupBox { border: 1px solid #c62828; "
                    "border-radius: 4px; padding: 8px; margin-top: 8px; }"
                )
        except Exception:
            self.gpu_label.setText(
                "Could not detect the GPU.\nThe CPU packages will be installed; "
                f"{PLUGIN_NAME} needs an NVIDIA GPU to run the model.")

    def _on_reinstall_clicked(self):
        """Cleanup runs in the background worker, never on the GUI thread."""
        self.reinstall_button.hide()
        self.install_requested.emit()

    def show_install_ui(self):
        """Show the install button, hide progress and status."""
        self._completion_mode = False
        self.install_button.show()
        self.cancel_button.hide()
        self.progress_frame.hide()
        self.status_label.hide()
        self.reinstall_button.hide()

    def show_completion_ui(self, message: str):
        """The environment is current and only lacks packages this release
        adds (spconv's build for RTX 50 series): offer to add them in place
        instead of a reinstall."""
        self.show_install_ui()
        self._completion_mode = True
        self.install_button.setText("Add missing packages")
        self.status_label.setText(message)
        self.status_label.setStyleSheet("")
        self.status_label.show()

    def show_progress_ui(self):
        """Show progress bar and cancel button, hide install button."""
        self.install_button.hide()
        self.cancel_button.show()
        self.progress_frame.show()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting installation...")
        self.status_label.hide()
        self.reinstall_button.hide()

    def set_progress(self, percent: int, message: str):
        """Update the progress bar and label.

        Args:
            percent: Progress percentage (0-100).
            message: Status message to display.
        """
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def set_status(self, ok: bool, message: str):
        """Set the status display.

        Args:
            ok: Whether the status is a success.
            message: The status message to display.
        """
        self.status_label.setText(message)
        self.status_label.show()
        if ok:
            self.status_label.setStyleSheet(
                "QLabel { color: #4CAF50; font-weight: bold; }"
            )
        else:
            self.status_label.setStyleSheet(
                "QLabel { color: #F44336; font-weight: bold; }"
            )

    def show_complete_ui(self, success: bool, message: str):
        """Show the completion status.

        Args:
            success: Whether installation was successful.
            message: Completion message.
        """
        self.cancel_button.hide()

        if success:
            self.progress_bar.setValue(100)
            self.progress_label.setText("Installation complete!")
            self.set_status(True, message)
            self.install_button.hide()
            self.reinstall_button.hide()
        else:
            self.progress_frame.hide()
            self.set_status(False, message)
            self.install_button.hide()
            self.reinstall_button.show()
