"""
Interactions module for EDView.
Contains the user actions and calculations of the tabs, and the application
controller.
"""

import os
import sys
import json
import threading
import traceback
from pathlib import Path
import numpy as np
import h5py
from PIL import Image

from PySide6.QtCore import Qt, QTimer, QObject
from PySide6.QtGui import QIcon, QCursor
from PySide6.QtWidgets import (QApplication, QMessageBox, QFileDialog,
                               QTreeWidgetItem, QMenu)

import algodif.io.loadersaver as losa

import edview.lib.rendering as ren
import edview.lib.utilities as util


class BaseLogic:
    """Worker threads and remembered folders, shared by the tabs."""

    def folder_key(self, purpose):
        return f"{type(self).__name__}.{purpose}"

    def last_folder(self, purpose):
        return util.get_last_folder(self.folder_key(purpose))

    def remember_folder(self, purpose, path):
        util.set_last_folder(self.folder_key(purpose), path)

    def is_busy(self):
        return self.worker is not None and self.worker.is_alive()

    def run_worker(self, task, on_done, title, message):
        """Run task(cancel_event) in a thread behind a modal dialog."""
        if self.is_busy():
            raise ValueError("A calculation is running!")
        self.worker_result = None
        self.worker_error = None
        self.progress = message
        self.cancel_event = threading.Event()
        self.dialog = ren.WorkerDialog(self, title, message,
                                       self.cancel_worker)
        self.dialog.show()

        def target():
            try:
                self.worker_result = task(self.cancel_event)
            except Exception as error:
                traceback.print_exc()
                self.worker_error = error

        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()
        QTimer.singleShot(util.WORKER_POLL_MS,
                          lambda: self.poll_worker(on_done))

    def set_progress(self, text):
        self.progress = text

    def cancel_worker(self):
        if not self.is_busy():
            return
        self.cancel_event.set()
        if self.dialog is not None:
            self.dialog.btn_cancel.setEnabled(False)
        self.progress = "Cancelling... The current step must finish first."

    def poll_worker(self, on_done):
        if self.worker.is_alive():
            if self.dialog is not None:
                self.dialog.set_message(self.progress)
            self.set_status(self.progress)
            QTimer.singleShot(util.WORKER_POLL_MS,
                              lambda: self.poll_worker(on_done))
            return
        if self.dialog is not None:
            self.dialog.finished_task = True
            self.dialog.accept()
            self.dialog.deleteLater()
            self.dialog = None
        if self.closing:
            return
        if self.worker_error is not None:
            self.set_status(f"Error: {self.worker_error}")
            QMessageBox.critical(self, "Error", str(self.worker_error))
            return
        try:
            on_done(self.worker_result)
        except Exception as error:
            traceback.print_exc()
            self.set_status(f"Error: {error}")
            QMessageBox.critical(self, "Error", str(error))


# ---------------------------------------------------------------------------
# Data browser
# ---------------------------------------------------------------------------

class BrowserTab(BaseLogic, ren.BrowserTabUI):
    """Browse folders, preview files and send them to the other tabs."""

    def __init__(self, base_folder=None):
        super().__init__()
        self.open_callback = None
        self.selected_folder = ""
        self.preview_file = ""
        config = util.load_config() or {}
        base_folder = base_folder or config.get("base_folder") or \
            os.path.expanduser("~")
        if not os.path.isdir(base_folder):
            base_folder = os.path.expanduser("~")
        self.set_base_folder(base_folder)

    def set_base_folder(self, folder):
        self.base_folder = os.path.abspath(folder)
        self.lbl_base_folder.setText(self.base_folder)
        self.selected_folder = self.base_folder
        self.refresh_tree()

    def select_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select base folder",
                                                  self.base_folder)
        if folder:
            self.set_base_folder(folder)
            util.update_config(base_folder=self.base_folder)

    def refresh_tree(self):
        self.tree.clear()
        item = QTreeWidgetItem(self.tree)
        item.setText(0, self.base_folder)
        item.setData(0, Qt.UserRole, self.base_folder)
        QTreeWidgetItem(item).setText(0, "")
        item.setExpanded(True)
        self.list_files.clear()
        self.show_preview_page("info")

    def on_tree_expand(self, item):
        path = item.data(0, Qt.UserRole)
        if not path or not os.path.isdir(path):
            return
        item.takeChildren()
        try:
            subfolders = sorted((d for d in os.listdir(path)
                                 if os.path.isdir(os.path.join(path, d))),
                                key=util.natural_key)
        except OSError as error:
            self.set_status(f"Error reading folder: {error}")
            return
        for sub in subfolders:
            child = QTreeWidgetItem(item)
            child.setText(0, sub)
            child_path = os.path.join(path, sub)
            child.setData(0, Qt.UserRole, child_path)
            try:
                has_kids = any(e.is_dir() for e in os.scandir(child_path))
            except OSError:
                has_kids = False
            if has_kids:
                QTreeWidgetItem(child).setText(0, "")

    def on_folder_select(self):
        items = self.tree.selectedItems()
        if not items:
            return
        path = items[0].data(0, Qt.UserRole)
        if not path or not os.path.isdir(path):
            return
        self.selected_folder = path
        self.set_status(path)
        self.list_files.clear()
        try:
            files = sorted((f.name for f in os.scandir(path) if f.is_file()),
                           key=util.natural_key)
        except OSError as error:
            self.set_status(f"Error reading folder: {error}")
            return
        self.list_files.addItems(files)
        self.show_preview_page("info")
        self.lbl_preview_info.setText(f"{len(files)} file(s) in\n{path}")

    def selected_file(self):
        items = self.list_files.selectedItems()
        if not items:
            return ""
        return os.path.join(self.selected_folder, items[0].text())

    def on_file_select_change(self):
        QTimer.singleShot(util.FILE_SELECT_DEBOUNCE_MS, self.preview_selected)

    def preview_selected(self):
        path = self.selected_file()
        if not path or path == self.preview_file:
            return
        self.preview_file = path
        self.set_status(path)
        ext = Path(path).suffix.lower()
        try:
            if ext in util.IMAGE_EXT:
                mat = np.asarray(Image.open(path), dtype=np.float32)
                if mat.ndim == 3:
                    mat = np.mean(mat, axis=2)
                self.set_hdf_controls_visible(False)
                self.show_preview_page("image")
                self.show_image(self.p_preview, mat, log=False,
                                auto_range=True)
                self.set_title(self.p_preview, f"{Path(path).name}  "
                                               f"(height, width) = "
                                               f"{mat.shape}")
            elif ext in util.HDF_EXT:
                self.preview_hdf(path)
            elif ext in util.TEXT_EXT or util.is_text_file(path):
                self.preview_text(path)
            else:
                self.show_preview_page("info")
                size = util.format_bytes(os.path.getsize(path))
                self.lbl_preview_info.setText(f"No preview for this file "
                                              f"type.\n{path}\n{size}")
        except Exception as error:
            self.show_preview_page("info")
            self.lbl_preview_info.setText(f"Can't preview the file:\n"
                                          f"{error}")

    def preview_text(self, path):
        max_bytes = 5 * 1024 * 1024
        if os.path.getsize(path) > max_bytes:
            raise ValueError("The file is larger than 5 MB.")
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()
        if path.lower().endswith(".json"):
            try:
                text = json.dumps(json.loads(text), indent=4)
            except ValueError:
                pass
        self.text_preview.setPlainText(text)
        self.show_preview_page("text")

    def preview_hdf(self, path):
        self.text_hdf.setPlainText("\n".join(util.hdf_tree_lines(path)))
        keys = [key for key, shape in util.hdf_array_keys(path, 1)]
        self.combo_preview_key.blockSignals(True)
        self.combo_preview_key.clear()
        self.combo_preview_key.addItems(keys)
        self.combo_preview_key.blockSignals(False)
        self.set_hdf_controls_visible(True)
        self.show_preview_page("image")
        self.on_preview_key_changed(self.combo_preview_key.currentText())

    def on_preview_key_changed(self, key):
        path = self.preview_file
        if not key or not path.lower().endswith(util.HDF_EXT):
            return
        try:
            with h5py.File(path, "r") as file:
                dataset = file[key]
                shape = dataset.shape
                if dataset.ndim == 1:
                    mat = np.asarray(dataset[()], dtype=np.float32)[None]
                else:
                    index = (0,) * (dataset.ndim - 2)
                    mat = np.asarray(dataset[index], dtype=np.float32)
            self.show_image(self.p_preview, mat, log=False, auto_range=True)
            first = " (first image)" if len(shape) > 2 else ""
            self.set_title(self.p_preview, f"{key}: {shape}{first}")
        except Exception as error:
            self.set_status(f"Can't preview '{key}': {error}")

    def open_selected(self):
        path = self.selected_file() or self.selected_folder
        if not path:
            return
        if self.open_callback is not None:
            self.open_callback(self.combo_target.currentText(), path)

    def on_file_context_menu(self, pos):
        item = self.list_files.itemAt(pos)
        menu = QMenu(self)
        path = os.path.join(self.selected_folder, item.text()) if item \
            else self.selected_folder
        act_copy = menu.addAction("Copy full path")
        act_folder = menu.addAction("Copy folder path")
        chosen = menu.exec(QCursor.pos())
        if chosen == act_copy:
            QApplication.clipboard().setText(os.path.normpath(path))
        elif chosen == act_folder:
            QApplication.clipboard().setText(
                os.path.normpath(self.selected_folder))


# ---------------------------------------------------------------------------
# Data correction and conversion
# ---------------------------------------------------------------------------

def scan_name_of(path):
    """Name of a scan from a folder or file path ('tifs' folders skipped)."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if path.is_file() and path.suffix.lower() in util.HDF_EXT:
        return path.stem
    if folder.name.lower() == "tifs":
        folder = folder.parent
    return folder.name


class CorrectionTab(BaseLogic, ren.CorrectionTabUI):
    """Correct raw scans with calibration coefficients; convert data."""

    def __init__(self):
        super().__init__()
        self.scans = []
        self.plot_correction()

    # Input
    def update_hdf_keys(self, combo, path, min_ndim=2):
        try:
            keys = [k for k, _ in util.hdf_array_keys(path, min_ndim)]
        except OSError:
            return
        current = combo.currentText()
        combo.clear()
        combo.addItems(keys)
        if current in keys:
            combo.setCurrentText(current)

    @ren.safe_call
    def browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select raw data file(s)", self.last_folder("input"),
            "Data (*.tif *.tiff *.hdf *.h5 *.hdf5 *.nxs *.dat);;"
            "All files (*.*)")
        if not paths:
            return
        self.remember_folder("input", paths[0])
        self.set_input(";".join(paths))

    @ren.safe_call
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select a scan folder", self.last_folder("input"))
        if folder:
            self.remember_folder("input", folder)
            self.set_input(folder)

    def set_input(self, text):
        self.edit_input.setText(text)
        first = text.split(";")[0].strip()
        if first.lower().endswith(util.HDF_EXT) and os.path.isfile(first):
            self.update_hdf_keys(self.combo_hdf_key, first)

    def get_scans(self):
        """[(name, items, metadata path), ...] of the input scans."""
        text = self.edit_input.text().strip()
        if not text:
            raise ValueError("Please select the input data!")
        hdf_key = self.combo_hdf_key.currentText().strip()
        axis = self.get_value(self.edit_axis, "Axis", int)
        scans_text = self.edit_scans.text().strip()
        if scans_text:
            if not os.path.isdir(text):
                raise ValueError("With 'Scans', the input must be the base "
                                 "folder of the scan folders!")
            folders = util.scan_folders(text, scans_text)
        else:
            folders = [text]
        scans = []
        for folder in folders:
            items = util.raw_scan_items(folder, hdf_key, axis)
            first = folder.split(";")[0].strip()
            scans.append((scan_name_of(first), items,
                          util.find_metadata_file(first)))
        return scans

    @ren.safe_call
    def load_input(self):
        self.scans = self.get_scans()
        name, items, _ = self.scans[0]
        hdf_key = self.combo_hdf_key.currentText().strip()
        axis = self.get_value(self.edit_axis, "Axis", int)
        self.raw_image = util.load_raw_image(items[0][0], hdf_key,
                                             items[0][1], axis)
        self.strip = min(self.strip, self.raw_image.shape[0] - 1)
        num = sum(len(scan[1]) for scan in self.scans)
        self.lbl_input_info.setText(
            f"{len(self.scans)} scan(s), {num} image(s). First: {name}, "
            f"{len(items)} image(s) of shape {self.raw_image.shape}.")
        self.edit_start.setText("0")
        self.edit_stop.setText("")
        self.update_preview()

    @ren.safe_call
    def load_coefficients(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select coefficients file",
            self.last_folder("coefficients"),
            "CSV (*.csv);;All files (*.*)")
        if not path:
            return
        self.remember_folder("coefficients", path)
        coefs = np.asarray(losa.load_csv(path), dtype=np.float64)
        if coefs.ndim == 1:
            coefs = coefs[np.newaxis, :]
        if coefs.ndim != 2 or np.any(np.isnan(coefs)):
            raise ValueError("The file must contain a numeric matrix: one "
                             "row of coefficients (a0, a1, ...) per strip.")
        self.coefs = coefs
        self.edit_coef.setText(path)
        if "energy" in Path(path).name.lower():
            self.combo_kind.setCurrentText("energy")
        self.set_status(f"Loaded coefficients of shape {coefs.shape}.")
        self.update_preview()

    def get_step(self, width):
        if self.coefs is None:
            raise ValueError("Please load the calibration coefficients!")
        method = self.combo_step.currentText()
        value = self.get_value(self.edit_step, "Value", allow_none=True)
        return util.get_output_step(self.coefs, width, method, value,
                                    self.combo_kind.currentText())

    @ren.safe_call
    def update_preview(self):
        self.corrected = self.corrected_raw = self.step = None
        if self.raw_image is not None and self.coefs is not None:
            width = self.raw_image.shape[1]
            step, source = self.get_step(width)
            self.corrected = util.correct_image(
                self.raw_image, self.coefs, step,
                self.combo_kind.currentText())
            self.corrected_raw = self.corrected
            self.step = step
            self.set_status(f"Preview updated. Output step: {step:.6g} "
                            f"({source}); x-max: {step * (width - 1):.4f}.")
        self.plot_correction()

    def get_background_para(self):
        if not self.check_bck.isChecked():
            return None
        method = self.combo_bck.currentText()
        specs = util.BCK_METHODS[method][1]
        edits = self.bck_edits[method]
        kwargs = {key: self.get_value(edits[key], label, dtype)
                  for key, label, _, dtype, _ in specs}
        return method, kwargs

    @ren.safe_call
    def apply_background(self):
        if self.corrected_raw is None:
            raise ValueError("No corrected preview. Load the input and the "
                             "coefficients first!")
        method, kwargs = self.get_background_para()
        image = self.corrected_raw

        def task(cancel_event):
            return util.remove_background(image, method, kwargs,
                                          cancel_event, self.set_progress)

        def done(result):
            if result is None:
                self.set_status("Background removal cancelled.")
                return
            if image is self.corrected_raw:
                self.corrected = result
                self.plot_correction()
            self.set_status(f"Background removed ({method}).")

        self.run_worker(task, done, "Background removal",
                        f"Removing background ({method})...")

    def on_click(self, plot, point):
        if self.raw_image is None:
            return
        self.strip = int(np.clip(round(point.y()), 0,
                                 self.raw_image.shape[0] - 1))
        self.plot_correction()

    def plot_correction(self):
        if self.raw_image is None:
            for plot in (self.p_raw, self.p_corr):
                plot.image_item.clear()
                plot.image_item.raw_data = None
            self.p_strip.clear()
            self.raw_line.hide()
            self.corr_line.hide()
            return
        self.show_image(self.p_raw, self.raw_image)
        self.raw_line.setValue(self.strip)
        self.raw_line.show()
        self.p_strip.clear()
        label, unit, _ = util.X_KINDS[self.combo_kind.currentText()]
        if self.corrected is not None:
            height, width = self.corrected.shape
            self.show_image(self.p_corr, self.corrected,
                            (-0.5 * self.step, -0.5, width * self.step,
                             height))
            self.set_label(self.p_corr, "bottom", f"{label} ({unit})")
            self.corr_line.setValue(self.strip)
            self.corr_line.show()
            self.p_strip.plot(np.arange(width) * self.step,
                              self.corrected[self.strip],
                              pen=ren.pg.mkPen(util.COLORS["blue"]))
            self.set_label(self.p_strip, "bottom", f"{label} ({unit})")
            self.set_title(self.p_strip, f"Corrected strip {self.strip}")
        else:
            self.p_corr.image_item.clear()
            self.p_corr.image_item.raw_data = None
            self.corr_line.hide()
            self.p_strip.plot(self.raw_image[self.strip],
                              pen=ren.pg.mkPen(util.COLORS["blue"]))
            self.set_label(self.p_strip, "bottom", "Pixel index")
            self.set_title(self.p_strip, f"Raw strip {self.strip} (load the "
                                         f"coefficients to correct)")

    # Output
    @ren.safe_call
    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", self.last_folder("output"))
        if folder:
            self.remember_folder("output", folder)
            self.edit_output.setText(folder)

    @ren.safe_call
    def make_output_folder(self):
        folder = self.edit_output.text().strip()
        if not folder:
            raise ValueError("Please type the path of the folder to create!")
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.set_status(f"Folder ready: {folder}")

    def get_output_para(self, width):
        fmt = self.combo_format.currentText()
        step, _ = self.get_step(width)
        para = {"x_kind": self.combo_kind.currentText(), "step": step,
                "hdf_key": self.combo_hdf_key.currentText().strip(),
                "axis": self.get_value(self.edit_axis, "Axis", int),
                "format": fmt, "bck": self.get_background_para(),
                "xy_type": self.combo_xy_type.currentText(),
                "zip": self.combo_zip.currentText() == "zip",
                "raw": self.combo_xy_data.currentText() == "raw",
                "xy_folder": self.edit_xy_folder.text().strip() or "xy"}
        return para

    @ren.safe_call
    def run_correction(self):
        if not self.scans:
            self.load_input()
            if not self.scans:
                return
        folder = self.edit_output.text().strip()
        if not folder:
            raise ValueError("Please select the output folder!")
        if not os.path.isdir(folder):
            raise ValueError(f"Output folder does not exist: {folder}. "
                             f"Click 'Make folder' to create it.")
        para = self.get_output_para(self.raw_image.shape[1])
        start = self.get_value(self.edit_start, "Start index", int)
        stop = self.get_value(self.edit_stop, "Stop index", int,
                              allow_none=True)
        jobs = []
        for name, items, meta_path in self.scans:
            last = len(items) - 1 if stop is None else min(stop,
                                                           len(items) - 1)
            if not 0 <= start <= last:
                raise ValueError(f"Invalid image range [{start}, {stop}] "
                                 f"for {name} ({len(items)} images).")
            jobs.append((name, items[start:last + 1], meta_path))
        coefs = self.coefs
        num_images = sum(len(job[1]) for job in jobs)

        def task(cancel_event):
            done_images = 0
            for name, items, meta_path in jobs:
                metadata = losa.load_json(meta_path) if meta_path else {}
                num, cancelled = util.correct_scan(
                    items, Path(folder) / name, coefs, para, cancel_event,
                    self.set_progress, metadata)
                done_images += num
                if cancelled:
                    return done_images, True
            return done_images, False

        def done(result):
            num, cancelled = result
            prefix = "Cancelled. " if cancelled else ""
            self.set_status(f"{prefix}Saved {num}/{num_images} image(s) of "
                            f"{len(jobs)} scan(s) ({para['format']}) to "
                            f"{folder}. Output step: {para['step']:.6g}.")

        self.run_worker(task, done, "Correction in progress",
                        f"Correcting {num_images} image(s) of {len(jobs)} "
                        f"scan(s)...")

    # Conversion
    @ren.safe_call
    def browse_conv_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source file", self.last_folder("conversion"),
            "HDF (*.hdf *.h5 *.hdf5 *.nxs);;All files (*.*)")
        if path:
            self.remember_folder("conversion", path)
            self.set_conversion_source(path)

    @ren.safe_call
    def browse_conv_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select source folder", self.last_folder("conversion"))
        if folder:
            self.remember_folder("conversion", folder)
            self.set_conversion_source(folder)

    def set_conversion_source(self, path):
        if os.path.isfile(path) and not path.lower().endswith(util.HDF_EXT):
            path = os.path.dirname(path)
        self.edit_conv_source.setText(path)
        if path.lower().endswith(util.HDF_EXT):
            self.update_hdf_keys(self.combo_conv_key, path)

    @ren.safe_call
    def browse_conv_output(self):
        mode = util.CONVERSIONS[self.combo_conversion.currentText()]
        if mode in ("tif_to_hdf", "tif2d_to_hdf4d"):
            path, _ = QFileDialog.getSaveFileName(
                self, "Output HDF file", self.last_folder("conv_output"),
                "HDF (*.hdf *.h5)")
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Output folder", self.last_folder("conv_output"))
        if path:
            self.remember_folder("conv_output", path)
            self.edit_conv_output.setText(path)

    @ren.safe_call
    def run_conversion(self):
        mode = util.CONVERSIONS[self.combo_conversion.currentText()]
        source = self.edit_conv_source.text().strip()
        output = self.edit_conv_output.text().strip()
        hdf_key = self.combo_conv_key.currentText().strip()
        x_step = self.get_value(self.edit_conv_step, "X step",
                                allow_none=True)
        kind = self.combo_conv_kind.currentText()
        xy_type = self.combo_conv_xy_type.currentText()
        zipped = self.combo_conv_zip.currentText() == "zip"

        def task(cancel_event):
            return util.convert_data(mode, source, output, hdf_key, x_step,
                                     kind, xy_type, zipped, cancel_event,
                                     self.set_progress)

        def done(result):
            message, _ = result
            self.lbl_conv_info.setText(message)
            self.set_status(message)

        self.run_worker(task, done, "Conversion in progress",
                        f"Converting {source}...")


# ---------------------------------------------------------------------------
# Viewers
# ---------------------------------------------------------------------------

class ViewerLogic(BaseLogic):
    """Input and saving actions shared by the viewers."""

    hdf_min_ndim = 3

    def update_hdf_keys(self, path):
        try:
            keys = [k for k, _ in util.hdf_array_keys(path,
                                                      self.hdf_min_ndim)]
        except OSError as error:
            self.set_status(f"Can't read {path}: {error}")
            return
        current = self.combo_hdf_key.currentText()
        self.combo_hdf_key.clear()
        self.combo_hdf_key.addItems(keys)
        if current in keys:
            self.combo_hdf_key.setCurrentText(current)

    def set_path(self, path, append=False):
        if os.path.isfile(path) and not path.lower().endswith(util.HDF_EXT):
            path = os.path.dirname(path)
        if append and self.edit_path.text().strip():
            paths = [p.strip() for p in self.edit_path.text().split(";")
                     if p.strip()]
            if path not in paths:
                paths.append(path)
            self.edit_path.setText("; ".join(paths))
        else:
            self.edit_path.setText(path)
        if path.lower().endswith(util.HDF_EXT):
            self.update_hdf_keys(path)

    @ren.safe_call
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select a folder of processed tif files",
            self.last_folder("data"))
        if folder:
            self.remember_folder("data", folder)
            self.set_path(folder, append=self.multi_input)

    @ren.safe_call
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a data file", self.last_folder("data"),
            "Data (*.hdf *.h5 *.hdf5 *.nxs *.tif *.tiff);;All files (*.*)")
        if path:
            self.remember_folder("data", path)
            self.set_path(path, append=self.multi_input)

    @ren.safe_call
    def browse_metadata(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata", self.last_folder("metadata"),
            "JSON (*.json);;All files (*.*)")
        if path:
            self.remember_folder("metadata", path)
            self.edit_metadata.setText(path)

    def input_options(self):
        return {"hdf_key": self.combo_hdf_key.currentText().strip(),
                "metadata_path": self.edit_metadata.text().strip(),
                "x_step": self.get_value(self.edit_step, "X step",
                                         allow_none=True),
                "x_kind": self.combo_kind.currentText()}

    def radius(self):
        return int(self.combo_radius.currentText())

    def show_scan_image(self, plot, mat, x_step):
        height, width = mat.shape
        self.show_image(plot, mat, (-0.5 * x_step, -0.5, width * x_step,
                                    height),
                        self.combo_cmap.currentText(),
                        self.check_log.isChecked())
        self.set_label(plot, "bottom", self.x_label())

    def show_qslice(self, mat, y_label):
        self.show_image(self.p_slice, mat, None, self.combo_cmap.currentText(),
                        self.check_log.isChecked())
        self.set_label(self.p_slice, "left", f"{y_label} (index)")

    def set_span(self, x_index, x_step):
        radius = self.radius()
        self.img_span.setRegion(((x_index - radius - 0.5) * x_step,
                                 (x_index + radius + 0.5) * x_step))
        self.img_span.show()
        self.prof_line.setValue(x_index * x_step)
        self.prof_line.show()

    def set_crosshairs(self, scan_index):
        if self.row is None:
            for line in (self.img_row_line, self.slice_hline,
                         self.slice_vline):
                line.hide()
            return
        self.img_row_line.setValue(self.row)
        self.slice_hline.setValue(scan_index)
        self.slice_vline.setValue(self.row)
        for line in (self.img_row_line, self.slice_hline, self.slice_vline):
            line.show()

    def plot_waterfall(self, curves, current, ylabel):
        """Curves (x, y, label) offset vertically; current one in red."""
        self.p_prof.clear()
        self.p_prof.addItem(self.prof_line)
        peak = max((float(np.max(y)) for _, y, _ in curves if len(y)),
                   default=1.0)
        spacing = peak * 0.3 if peak > 0 else 1.0
        for i, (x, y, _) in enumerate(curves):
            color = util.COLORS["red"] if i == current \
                else util.COLORS["blue"]
            width = 1.5 if i == current else 0.8
            self.p_prof.plot(x, np.asarray(y) + i * spacing,
                             pen=ren.pg.mkPen(color, width=width))
        self.set_label(self.p_prof, "left", ylabel)
        self.set_label(self.p_prof, "bottom", self.x_label())
        self.pattern_curves = curves

    def plot_single(self, x, y, title):
        self.p_prof.clear()
        self.p_prof.addItem(self.prof_line)
        self.p_prof.plot(x, y, pen=ren.pg.mkPen(util.COLORS["blue"],
                                                width=1.0))
        self.set_title(self.p_prof, title)
        self.set_label(self.p_prof, "left", "Intensity")
        self.set_label(self.p_prof, "bottom", self.x_label())
        self.pattern_curves = [(x, y, "intensity")]

    def refresh_all(self):
        self.update_image()
        self.update_qslice()

    def clear_profile(self):
        self.p_prof.clear()
        self.p_prof.addItem(self.prof_line)
        self.prof_line.hide()
        self.pattern_curves = None

    @ren.safe_call
    def save_pattern(self):
        curves = getattr(self, "pattern_curves", None)
        if not curves:
            raise ValueError("Click on the image to select a row first!")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pattern(s)", os.path.join(self.last_folder("save"),
                                                  "pattern.csv"),
            "CSV (*.csv)")
        if not path:
            return
        self.remember_folder("save", path)
        lengths = {len(x) for x, _, _ in curves}
        if len(lengths) == 1:
            columns = [curves[0][0]] + [y for _, y, _ in curves]
            header = "x," + ",".join(label for _, _, label in curves)
        else:
            columns, labels = [], []
            for x, y, label in curves:
                columns += [x, y]
                labels += [f"x_{label}", label]
            width = max(lengths)
            columns = [np.pad(np.asarray(c, float), (0, width - len(c)),
                              constant_values=np.nan) for c in columns]
            header = ",".join(labels)
        np.savetxt(path, np.column_stack(columns), delimiter=",",
                   header=header, comments="")
        self.set_status(f"Saved: {path}")

    @ren.safe_call
    def save_qslice(self):
        mat = getattr(self, "qslice_raw", None)
        if mat is None:
            raise ValueError("No q-slice image! Load data first.")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save q-slice", os.path.join(self.last_folder("save"),
                                               "q_slice.tif"),
            "TIFF (*.tif)")
        if not path:
            return
        self.remember_folder("save", path)
        path = losa.save_image(path, np.float32(mat))
        self.set_status(f"Saved: {path}")


class EddtViewer(ViewerLogic, ren.EddtViewerUI):
    """Viewer of a 1D scan; MultiLinesViewer shows a waterfall of all
    images."""

    @ren.safe_call
    def load_data(self):
        self.dataset = util.load_scan(self.edit_path.text(),
                                      **self.input_options())
        data = self.dataset
        self.row = None
        self.clear_profile()
        self.slider_img.set_range(data.depth - 1)
        self.slider_x.set_range(data.width - 1)
        self.slider_img.set_value(0, notify=False)
        self.slider_x.set_value(data.width // 2, notify=False)
        self.refresh_all()
        for plot in (self.p_img, self.p_slice):
            plot.enableAutoRange()
        self.set_status(f"Loaded {data.name}: {data.depth} image(s) of "
                        f"shape ({data.height}, {data.width}). X-step: "
                        f"{data.x_step:.6g}. Click on the image to select a "
                        f"row.")

    def update_image(self):
        data = self.dataset
        if data is None:
            return
        idx = self.slider_img.value()
        self.show_scan_image(self.p_img, data.data[idx], data.x_step)
        self.set_title(self.p_img, f"{data.motor_name}: "
                                   f"{data.positions[idx]:.3f} | Image "
                                   f"{idx}")
        self.set_crosshairs(idx)
        self.update_profile()

    def update_qslice(self):
        data = self.dataset
        if data is None:
            return
        x_idx = self.slider_x.value()
        self.qslice_raw = data.q_slice(x_idx, self.radius())
        self.show_qslice(self.qslice_raw, data.motor_name)
        label = util.X_KINDS[data.x_kind][0]
        self.set_title(self.p_slice, f"Mean of slices around {label}="
                                     f"{x_idx * data.x_step:.3f}")
        self.set_span(x_idx, data.x_step)
        self.set_crosshairs(self.slider_img.value())

    def update_profile(self):
        data = self.dataset
        if data is None or self.row is None:
            return
        idx = self.slider_img.value()
        if self.stack_mode:
            curves = [(data.x_list, data.data[i, self.row], f"image_{i}")
                      for i in range(data.depth)]
            self.plot_waterfall(curves, idx, "Image index (offset)")
            self.set_title(self.p_prof, f"Patterns of all images at row "
                                        f"{self.row}")
        else:
            self.plot_single(data.x_list, data.data[idx, self.row],
                             f"Pattern at row {self.row} | Image {idx}")

    def on_click(self, plot, point):
        data = self.dataset
        if data is None:
            return
        if plot is self.p_img:
            self.row = int(np.clip(round(point.y()), 0, data.height - 1))
            self.slider_x.set_value(data.x_index(point.x()), notify=False)
            self.update_qslice()
            self.update_profile()
        elif plot is self.p_slice:
            self.row = int(np.clip(round(point.x()), 0, data.height - 1))
            self.slider_img.set_value(int(np.clip(round(point.y()), 0,
                                                  data.depth - 1)))
        elif plot is self.p_prof:
            self.slider_x.set_value(data.x_index(point.x()))


class MultiLinesViewer(EddtViewer, ren.MultiLinesViewerUI):
    stack_mode = True


class MultiScansViewer(ViewerLogic, ren.MultiScansViewerUI):
    """Viewer of several 1D scans."""

    @ren.safe_call
    def load_data(self):
        paths = [p.strip() for p in self.edit_path.text().split(";")
                 if p.strip()]
        if not paths:
            raise ValueError("Please select the folders/files of the scans!")
        options = self.input_options()
        if len(paths) > 1:
            options["metadata_path"] = ""
        self.datasets = [util.load_scan(path, **options) for path in paths]
        self.row = None
        self.clear_profile()
        depth = max(d.depth for d in self.datasets)
        self.slider_img.set_range(depth - 1)
        self.slider_scan.set_range(len(self.datasets) - 1)
        self.slider_x.set_range(max(d.width for d in self.datasets) - 1)
        self.slider_img.set_value(0, notify=False)
        self.slider_scan.set_value(0, notify=False)
        self.slider_x.set_value(self.datasets[0].width // 2, notify=False)
        self.refresh_all()
        for plot in (self.p_img, self.p_slice):
            plot.enableAutoRange()
        self.set_status(f"Loaded {len(self.datasets)} scan(s): "
                        f"{', '.join(d.name for d in self.datasets)}.")

    def current(self):
        data = self.datasets[self.slider_scan.value()]
        return data, min(self.slider_img.value(), data.depth - 1)

    def update_image(self):
        if not self.datasets:
            return
        data, idx = self.current()
        self.show_scan_image(self.p_img, data.data[idx], data.x_step)
        self.set_title(self.p_img, f"Scan: {data.name} | {data.motor_name}: "
                                   f"{data.positions[idx]:.3f} | Image "
                                   f"{idx}")
        self.update_qslice()
        self.update_profile()

    def update_qslice(self):
        if not self.datasets:
            return
        data, idx = self.current()
        x_idx = min(self.slider_x.value(), data.width - 1)
        self.qslice_raw = data.q_slice(x_idx, self.radius())
        self.show_qslice(self.qslice_raw, data.motor_name)
        label = util.X_KINDS[data.x_kind][0]
        self.set_title(self.p_slice, f"Scan: {data.name} | Mean around "
                                     f"{label}={x_idx * data.x_step:.3f}")
        self.set_span(x_idx, data.x_step)
        self.set_crosshairs(idx)

    def update_profile(self):
        if not self.datasets or self.row is None:
            return
        img_idx = self.slider_img.value()
        curves = []
        for data in self.datasets:
            if img_idx < data.depth and self.row < data.height:
                curves.append((data.x_list, data.data[img_idx, self.row],
                               data.name))
            else:
                curves.append((data.x_list, np.zeros(data.width),
                               data.name))
        self.plot_waterfall(curves, self.slider_scan.value(),
                            "Scan index (offset)")
        self.set_title(self.p_prof, f"Patterns at row {self.row} | Image "
                                    f"{img_idx}")

    def on_click(self, plot, point):
        if not self.datasets:
            return
        data, _ = self.current()
        if plot is self.p_img:
            self.row = int(np.clip(round(point.y()), 0, data.height - 1))
            self.slider_x.set_value(data.x_index(point.x()), notify=False)
            self.update_qslice()
            self.update_profile()
        elif plot is self.p_slice:
            self.row = int(np.clip(round(point.x()), 0, data.height - 1))
            self.slider_img.set_value(int(np.clip(round(point.y()), 0,
                                                  data.depth - 1)))
        elif plot is self.p_prof:
            self.slider_x.set_value(data.x_index(point.x()))


class Scan2DViewer(ViewerLogic, ren.Scan2DViewerUI):
    """Viewer of a 2D scan (tif files loaded on demand, or a 4D HDF)."""

    hdf_min_ndim = 4

    @ren.safe_call
    def load_data(self):
        dataset = util.Scan2DDataset(self.edit_path.text(),
                                     **self.input_options())
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = dataset
        self.row = None
        self.qslice_key = None
        self.clear_profile()
        self.slider_row.set_range(dataset.nrow - 1)
        self.slider_col.set_range(dataset.ncol - 1)
        self.slider_x.set_range(dataset.width - 1)
        self.slider_row.set_value(0, notify=False)
        self.slider_col.set_value(0, notify=False)
        self.slider_x.set_value(dataset.width // 2, notify=False)
        for slider, (name, _) in zip((self.slider_row, self.slider_col),
                                     dataset.motors):
            slider.label.setText(f"{name} (index)")
        self.combo_slice_axis.blockSignals(True)
        self.combo_slice_axis.clear()
        self.combo_slice_axis.addItems([f"Axis 0: {dataset.motors[0][0]}",
                                        f"Axis 1: {dataset.motors[1][0]}"])
        self.combo_slice_axis.blockSignals(False)
        self.refresh_all()
        for plot in (self.p_img, self.p_slice):
            plot.enableAutoRange()
        self.set_status(f"Loaded 2D scan {dataset.name}: "
                        f"{dataset.nrow} x {dataset.ncol} images of shape "
                        f"({dataset.height}, {dataset.width}).")

    def indices(self):
        return self.slider_row.value(), self.slider_col.value()

    def update_image(self):
        data = self.dataset
        if data is None:
            return
        row, col = self.indices()
        self.show_scan_image(self.p_img, data.get_image(row, col),
                             data.x_step)
        (name0, pos0), (name1, pos1) = data.motors
        self.set_title(self.p_img, f"{name0}: {pos0[row]:.3f} | {name1}: "
                                   f"{pos1[col]:.3f}")
        self.update_qslice()
        self.update_profile()

    def update_qslice(self):
        data = self.dataset
        if data is None:
            return
        axis = self.combo_slice_axis.currentIndex()
        row, col = self.indices()
        fixed = col if axis == 0 else row
        x_idx = self.slider_x.value()
        radius = self.radius()
        key = (axis, fixed, x_idx, radius)
        if key != self.qslice_key:
            self.set_status(f"Loading q-slices along "
                            f"{data.motors[axis][0]}...")
            QApplication.processEvents()
            self.qslice_raw = data.q_slice(axis, fixed, x_idx, radius)
            self.qslice_key = key
        self.show_qslice(self.qslice_raw, data.motors[axis][0])
        label = util.X_KINDS[data.x_kind][0]
        self.set_title(self.p_slice, f"Mean around {label}="
                                     f"{x_idx * data.x_step:.3f}")
        self.set_span(x_idx, data.x_step)
        self.set_crosshairs(row if axis == 0 else col)

    def update_profile(self):
        data = self.dataset
        if data is None or self.row is None:
            return
        row, col = self.indices()
        self.plot_single(data.x_list, data.get_image(row, col)[self.row],
                         f"Pattern at row {self.row} | Image ({row}, {col})")

    def on_click(self, plot, point):
        data = self.dataset
        if data is None:
            return
        if plot is self.p_img:
            self.row = int(np.clip(round(point.y()), 0, data.height - 1))
            self.slider_x.set_value(data.x_index(point.x()), notify=False)
            self.update_qslice()
            self.update_profile()
        elif plot is self.p_slice:
            self.row = int(np.clip(round(point.x()), 0, data.height - 1))
            slider = self.slider_row if self.combo_slice_axis.currentIndex() \
                == 0 else self.slider_col
            slider.set_value(int(np.clip(round(point.y()), 0,
                                         slider.spin.maximum())))
        elif plot is self.p_prof:
            self.slider_x.set_value(data.x_index(point.x()))


# ---------------------------------------------------------------------------
# Application controller
# ---------------------------------------------------------------------------

class EdviewInteraction(QObject):
    """Primary controller of the EDView application."""

    def __init__(self, base_folder=None):
        super().__init__()
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.app.setApplicationName(util.APP_NAME)
        if os.path.exists(ren.ICON_PATH):
            self.app.setWindowIcon(QIcon(ren.ICON_PATH))
        self.app.setFont(ren.select_ui_font(point_size=util.FONT_SIZE))
        config = util.load_config() or {}
        theme = config.get("theme", ren.DEFAULT_THEME)
        if theme not in ren.THEMES:
            theme = ren.DEFAULT_THEME
        ren.apply_theme(self.app, theme)
        self.browser = BrowserTab(base_folder)
        self.correction = CorrectionTab()
        self.viewers = {"EDDT viewer": EddtViewer(),
                        "Multiple-lines viewer": MultiLinesViewer(),
                        "Multiple-scans viewer": MultiScansViewer(),
                        "2D-scan viewer": Scan2DViewer()}
        self.tabs = [self.browser, self.correction] + \
            list(self.viewers.values())
        pages = [(self.browser, "Data browser"),
                 (self.correction, "Data correction")] + \
            [(tab, name) for name, tab in self.viewers.items()]
        self.main_win = ren.MainWindow(pages)
        self.main_win.combo_theme.blockSignals(True)
        self.main_win.combo_theme.setCurrentText(theme)
        self.main_win.combo_theme.blockSignals(False)
        self.main_win.combo_theme.currentTextChanged.connect(
            self.on_theme_changed)
        self.main_win.close_handler = self.on_close_requested
        self.browser.open_callback = self.open_in_tab

    def open_in_tab(self, target, path):
        tab_widget = self.main_win.tab_widget
        if target in self.viewers:
            viewer = self.viewers[target]
            viewer.set_path(path, append=viewer.multi_input)
            tab_widget.setCurrentWidget(viewer)
            if not viewer.multi_input:
                viewer.load_data()
            else:
                viewer.set_status("Path added. Add more scans, then click "
                                  "Load.")
        elif target.startswith("Data correction"):
            self.correction.side_tabs.setCurrentIndex(0)
            self.correction.set_input(path)
            tab_widget.setCurrentWidget(self.correction)
            self.correction.set_status("Input set. Click Load to preview.")
        else:
            self.correction.side_tabs.setCurrentIndex(1)
            self.correction.set_conversion_source(path)
            tab_widget.setCurrentWidget(self.correction)

    def on_theme_changed(self, theme):
        self.main_win.setUpdatesEnabled(False)
        try:
            ren.apply_theme(self.app, theme)
            for tab in self.tabs:
                tab.refresh_plot_theme()
        finally:
            self.main_win.setUpdatesEnabled(True)
        util.update_config(theme=theme)

    def on_close_requested(self):
        """Stop running calculations. Return False to cancel closing."""
        busy = [tab for tab in self.tabs if tab.is_busy()]
        if busy:
            answer = QMessageBox.question(self.main_win, "Quit",
                                          "A calculation is running. Quit "
                                          "anyway?")
            if answer != QMessageBox.Yes:
                return False
            for tab in busy:
                tab.cancel_event.set()
        for tab in self.tabs:
            tab.closing = True
        scan2d = self.viewers["2D-scan viewer"]
        if scan2d.dataset is not None:
            scan2d.dataset.close()
        return True

    def run(self):
        self.main_win.show()
        self.main_win.center_on_screen()
        return self.app.exec()
