"""Real QGIS window run of Rail 3D, with the screenshots of the user guide.

Launched twice with the same --profiles-path (see tools/gui_run.py --help
in run_gui.py): the first launch enables the plugin and quits; the second
drives the plugin like a user: Setup panel (screenshot), plugin manager
"Install from ZIP" page (screenshot), main panel (screenshot), output codes
editor (screenshot), add the railway file, Run, wait, screenshots of the
finished panel and of the map. Writes report.json to RAIL3D_GUI_OUT.
"""
import json
import os
import time
import traceback
from pathlib import Path

from qgis.core import QgsProject, QgsSettings
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtWidgets import QApplication, QListWidget
from qgis.utils import iface, isPluginLoaded, plugins

NAME = "Rail3D"
OUT = os.environ["RAIL3D_GUI_OUT"]
RAIL_INPUT = os.environ["RAIL3D_GUI_INPUT"]
os.makedirs(OUT, exist_ok=True)
report = {"t0": time.time(), "steps": []}


def step(text):
    report["steps"].append(f"{round(time.time() - report['t0'], 1)}s {text}")


def shot(widget, name):
    widget.grab().save(os.path.join(OUT, name))
    step(f"screenshot {name}")


def finish():
    report["elapsed"] = round(time.time() - report["t0"], 1)
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, default=str)
    QgsProject.instance().setDirty(False)
    QTimer.singleShot(1000, iface.actionExit().trigger)


def guarded(function):
    def wrapper(*args):
        try:
            function(*args)
        except Exception:
            report["error"] = traceback.format_exc()
            finish()
    return wrapper


def dock():
    return plugins[NAME].dock


@guarded
def show_setup():
    plugin = plugins[NAME]
    plugin._show_deps_dock()
    QTimer.singleShot(6000, after_setup)


@guarded
def after_setup():
    plugin = plugins[NAME]
    if plugin._deps_dock is not None:
        report["setup_gpu_text"] = plugin._deps_dock.gpu_label.text()
        shot(plugin._deps_dock, "setup_panel.png")
        plugin._close_deps_dock()
    QTimer.singleShot(500, show_plugin_manager)


@guarded
def show_plugin_manager():
    QTimer.singleShot(5000, grab_plugin_manager)
    iface.actionManagePlugins().trigger()


@guarded
def grab_plugin_manager():
    dialog = QApplication.activeModalWidget()
    if dialog is None:
        report["plugin_manager"] = "no modal dialog"
    else:
        try:
            options = dialog.findChild(QListWidget, "mOptionsListWidget")
            for row in range(options.count()):
                if "ZIP" in options.item(row).text():
                    options.setCurrentRow(row)
                    break
            QApplication.processEvents()
            time.sleep(0.5)
            QApplication.processEvents()
            shot(dialog, "plugin_manager_zip.png")
            report["plugin_manager"] = dialog.windowTitle()
        finally:
            dialog.reject()
    QTimer.singleShot(1000, open_dock)


@guarded
def open_dock():
    plugins[NAME].toggle_action.trigger()
    QTimer.singleShot(10000, after_open)


@guarded
def after_open():
    d = dock()
    if d is None:
        report["setup_dock_shown_instead"] = True
        plugin = plugins[NAME]
        if plugin._deps_dock is not None:
            report["setup_status"] = plugin._deps_dock.status_label.text()
        finish()
        return
    from qgis.utils import pluginMetadata
    report["plugin_version"] = pluginMetadata(NAME, "version")
    report["model_name"] = d.model_name_label.text()
    report["initial_label"] = d.model_label.text()
    report["device"] = d.device_label.text()
    report["gpu_checked"] = d.gpu_check.isChecked()
    report["gate_ok"] = d.model_gate_ok
    report["weights_ready"] = d.model_ready
    report["hint"] = d.current_file_label.text()
    report["field_enabled"] = d.field_edit.isEnabled()
    report["codes_row_visible"] = d.mms_codes_row.isVisibleTo(d)
    shot(d, "panel_empty.png")
    shot(iface.mainWindow(), "qgis_window_panel.png")
    QTimer.singleShot(1500, grab_codes_dialog)
    d.mms_codes_btn.click()          # modal: returns after the timer closed it
    QTimer.singleShot(500, add_file)


@guarded
def grab_codes_dialog():
    dialog = QApplication.activeModalWidget()
    if dialog is None:
        raise RuntimeError("codes dialog did not open")
    report["codes_dialog"] = {
        "title": dialog.windowTitle(), "rows": dialog.table.rowCount(),
        "defaults": [[dialog.table.item(r, c).text() for c in (0, 1, 2, 4)]
                     for r in range(dialog.table.rowCount())]}
    shot(dialog, "codes_dialog.png")
    dialog.reject()


@guarded
def add_file():
    d = dock()
    d._clear_files()
    d._on_files_dropped([Path(RAIL_INPUT)])
    folder = os.path.join(OUT, "classified")
    os.makedirs(folder, exist_ok=True)
    d.output_widget.setFilePath(folder)
    QTimer.singleShot(4000, lambda: start(folder))


@guarded
def start(folder):
    d = dock()
    report["run_enabled"] = d.run_btn.isEnabled()
    report["hint_before_run"] = d.current_file_label.text()
    shot(d, "panel_ready.png")
    d.run_btn.click()
    report["started_at"] = round(time.time() - report["t0"], 1)
    step("run started")
    QTimer.singleShot(5000, lambda: poll(folder, False))


@guarded
def poll(folder, shot_taken):
    d = dock()
    if d._running and time.time() - report["t0"] < 2400:
        texts = report.setdefault("progress_texts", [])
        text = d.current_file_label.text()
        if not texts or texts[-1] != text:
            texts.append(text)
        if not shot_taken and d.progress_bar.value() > 5:
            shot(d, "panel_running.png")
            shot_taken = True
        QTimer.singleShot(2000, lambda: poll(folder, shot_taken))
        return
    if not report.get("settled"):
        report["settled"] = True
        report["seconds"] = round(time.time() - report["t0"] - report["started_at"], 1)
        QTimer.singleShot(8000, lambda: poll(folder, True))
        return
    report["status"] = d.current_file_label.text()
    report["outputs"] = sorted(os.listdir(folder))
    report["log_tail"] = d.log_text.toPlainText()[-3000:]
    layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
              if os.path.normcase(folder) in os.path.normcase(lyr.source())]
    report["layers"] = [{"name": lyr.name(), "valid": lyr.isValid(),
                         "points": lyr.dataProvider().pointCount(),
                         "renderer": lyr.renderer().type() if lyr.renderer() else None}
                        for lyr in layers]
    canvas = iface.mapCanvas()
    if layers:
        canvas.setExtent(layers[0].extent())
    canvas.refresh()
    step("run finished")
    QTimer.singleShot(8000, shoot_result)


@guarded
def shoot_result():
    shot(dock(), "panel_done.png")
    shot(iface.mapCanvas(), "map_result.png")
    shot(iface.mainWindow(), "qgis_window_result.png")
    finish()


def run():
    settings = QgsSettings()
    if not settings.value(f"PythonPlugins/{NAME}", False, type=bool):
        settings.setValue(f"PythonPlugins/{NAME}", True)
        settings.sync()
        report["stage"] = "enabled"
        finish()
        return
    report["stage"] = "drive"
    report["plugin_loaded"] = bool(isPluginLoaded(NAME))
    report["qgis"] = __import__("qgis.core", fromlist=["Qgis"]).Qgis.version()
    if not report["plugin_loaded"]:
        finish()
        return
    show_setup()


QTimer.singleShot(4000, guarded(run))
