"""
Interactions module for EDView.
Contains the user actions and calculations of the tabs, and the application
controller.
"""

import os
import sys
import json
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, QTimer, QObject, QEvent
from PySide6.QtGui import QIcon, QCursor
from PySide6.QtWidgets import (QApplication, QMessageBox, QFileDialog,
                               QTreeWidgetItem, QMenu, QListView, QTreeView,
                               QAbstractItemView, QListWidgetItem)

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
        self.selected_folder = ""
        self.preview_file = ""
        # "images": slider over the images of the folder; "hdf": over the
        # slices of a dataset
        self.preview_mode = None
        self.folder_images = []
        self.hdf_shapes = {}
        # Set by the controller: callback(paths) adding inputs to the
        # correction tab
        self.add_to_correction = None
        # Set by the controller: min HDF ndim of each viewer by name;
        # view_in(name, path) and metadata_to(name, path) switch to it
        self.viewer_ndims = {}
        self.view_in = None
        self.metadata_to = None
        # Set by the controller: name of the viewer of several scans, and
        # view_scans(paths) showing them in it
        self.scans_viewer = ""
        self.view_scans = None
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

    @staticmethod
    def current_selected(view):
        """The current (last clicked) item if selected, else the first
        selected item, or None."""
        item = view.currentItem()
        if item is not None and item.isSelected():
            return item
        items = view.selectedItems()
        return items[0] if items else None

    def on_folder_select(self):
        item = self.current_selected(self.tree)
        if item is None:
            return
        path = item.data(0, Qt.UserRole)
        if not path or not os.path.isdir(path) or \
                path == self.selected_folder and self.list_files.count():
            return
        self.selected_folder = path
        self.preview_file = ""
        self.preview_mode = None
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
        item = self.current_selected(self.list_files)
        if item is None:
            return ""
        return os.path.join(self.selected_folder, item.text())

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
                self.preview_image(path)
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
            self.preview_mode = None
            self.show_preview_page("info")
            self.lbl_preview_info.setText(f"Can't preview the file:\n"
                                          f"{error}")

    def show_preview(self, mat, title):
        """Show a 2D image (zoom out only if its shape changed) or a 1D
        line."""
        if mat.ndim == 1:
            self.show_plot_page(True)
            plot = self.p_preview_line
            plot.clear()
            plot.plot(np.arange(len(mat)), mat,
                      pen=ren.pg.mkPen(util.COLORS["blue"], width=1.2))
            plot.enableAutoRange()
            self.set_title(plot, title)
            return
        self.show_plot_page(False)
        old = getattr(self.p_preview.image_item, "raw_data", None)
        auto_range = old is None or np.shape(old) != mat.shape
        self.show_image(self.p_preview, mat, auto_range=auto_range)
        self.set_title(self.p_preview, title)

    def preview_image(self, path):
        """Show an image file, with a slider over the images of its
        folder."""
        self.folder_images = [
            (row, os.path.join(self.selected_folder,
                               self.list_files.item(row).text()))
            for row in range(self.list_files.count())
            if self.list_files.item(row).text().lower().endswith(
                util.IMAGE_EXT)]
        paths = [p for _, p in self.folder_images]
        index = paths.index(path) if path in paths else 0
        self.preview_mode = "images"
        self.set_hdf_controls_visible(False)
        self.show_preview_page("image")
        self.setup_slice_row(0, "Image in folder", len(paths), index)
        self.hide_slice_rows(1)
        self.show_folder_image(index)

    def show_folder_image(self, index):
        path = self.folder_images[index][1]
        mat = util.load_preview_image(path)
        self.show_preview(mat, f"{Path(path).name} | Image {index + 1}/"
                               f"{len(self.folder_images)} | (height, "
                               f"width) = {mat.shape}")

    def update_preview_slice(self):
        try:
            if self.preview_mode == "images":
                index = self.slice_rows[0][3].value()
                row, path = self.folder_images[index]
                # Select the file in the list without previewing it again
                self.preview_file = path
                self.list_files.blockSignals(True)
                self.list_files.setCurrentRow(row)
                self.list_files.blockSignals(False)
                self.list_files.scrollToItem(self.list_files.item(row))
                self.set_status(path)
                self.show_folder_image(index)
            elif self.preview_mode == "hdf":
                self.show_hdf_slice()
        except Exception as error:
            traceback.print_exc()
            self.set_status(f"Can't preview: {error}")

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
        self.hdf_shapes = dict(util.hdf_array_keys(path, 1))
        self.preview_mode = "hdf"
        self.combo_preview_key.blockSignals(True)
        self.combo_preview_key.clear()
        for key, shape in self.hdf_shapes.items():
            self.combo_preview_key.addItem(f"{key}    {shape}", key)
        self.combo_preview_key.blockSignals(False)
        self.set_hdf_controls_visible(True)
        self.show_preview_page("image")
        if not self.hdf_shapes:
            self.hide_slice_rows()
            self.show_preview_page("info")
            self.lbl_preview_info.setText(
                f"No numeric array in\n{path}\nDouble-click the file to "
                f"show its content.")
            return
        self.on_preview_key_changed()

    def preview_key(self):
        return self.combo_preview_key.currentData() or ""

    def num_slice_axes(self):
        """Number of sliced axes: 1 for 3D datasets, 2 for 4D or more."""
        shape = self.hdf_shapes.get(self.preview_key(), ())
        return min(max(len(shape) - 2, 0), len(self.slice_rows))

    def on_preview_key_changed(self):
        shape = self.hdf_shapes.get(self.preview_key())
        if self.preview_mode != "hdf" or shape is None:
            return
        num_axes = self.num_slice_axes()
        for num, (_, _, combo, _) in enumerate(self.slice_rows):
            combo.blockSignals(True)
            combo.clear()
            if num < num_axes:
                combo.addItems([str(i) for i in range(len(shape))])
                combo.setCurrentIndex(num)
            combo.blockSignals(False)
        for num in range(num_axes):
            self.setup_axis_slider(num)
        self.hide_slice_rows(num_axes)
        self.try_show_hdf_slice()

    def on_preview_axis_changed(self, num):
        if self.preview_mode != "hdf" or num >= self.num_slice_axes():
            return
        combos = [row[2] for row in self.slice_rows]
        axis = combos[num].currentIndex()
        if self.num_slice_axes() == 2:
            other = combos[1 - num]
            if other.currentIndex() == axis:
                # The two sliced axes must differ
                free = next(i for i in range(other.count()) if i != axis)
                other.blockSignals(True)
                other.setCurrentIndex(free)
                other.blockSignals(False)
                self.setup_axis_slider(1 - num)
        self.setup_axis_slider(num)
        self.try_show_hdf_slice()

    def setup_axis_slider(self, num):
        shape = self.hdf_shapes[self.preview_key()]
        axis = self.slice_rows[num][2].currentIndex()
        self.setup_slice_row(num, "Index", shape[axis], show_axis=True)

    def try_show_hdf_slice(self):
        try:
            self.show_hdf_slice()
        except Exception as error:
            traceback.print_exc()
            self.set_status(f"Can't preview '{self.preview_key()}': "
                            f"{error}")

    def show_hdf_slice(self):
        key = self.preview_key()
        shape = self.hdf_shapes[key]
        rows = self.slice_rows[:self.num_slice_axes()]
        axes = [combo.currentIndex() for _, _, combo, _ in rows]
        indices = [slider.value() for _, _, _, slider in rows]
        mat = util.load_hdf_slice(self.preview_file, key, axes, indices)
        title = f"{key}: {shape}"
        for axis, index in zip(axes, indices):
            title += f" | Axis {axis}: {index}/{shape[axis] - 1}"
        if len(shape) > 4:
            title += " (other leading axes at 0)"
        self.show_preview(mat, title)

    def on_file_double_click(self, item):
        path = os.path.join(self.selected_folder, item.text())
        if path.lower().endswith(util.HDF_EXT):
            self.show_hdf_content(path)

    @ren.safe_call
    def show_hdf_content(self, path=None):
        """Window with the groups, datasets and attributes of an HDF
        file."""
        path = path or self.preview_file
        if not path.lower().endswith(util.HDF_EXT):
            return
        window = ren.HdfContentWindow(self, path, util.hdf_nodes(path))
        window.datasetActivated.connect(self.preview_hdf_dataset)
        window.show()

    def preview_hdf_dataset(self, path, key):
        """Preview a dataset double-clicked in an HDF content window."""
        try:
            if path != self.preview_file or self.preview_mode != "hdf":
                self.preview_file = path
                self.preview_hdf(path)
            index = self.combo_preview_key.findData(key)
            if index < 0:
                self.set_status(f"'{key}' is not a numeric array: no "
                                f"preview.")
                return
            self.combo_preview_key.setCurrentIndex(index)
            self.set_status(f"{path} :: {key}")
        except Exception as error:
            traceback.print_exc()
            self.set_status(f"Can't preview '{key}': {error}")

    @staticmethod
    def is_correction_folder(path):
        """True if a folder (or its 'tifs' sub-folder) has several tif or
        dat files to correct."""
        tifs = os.path.join(path, "tifs")
        folder = tifs if os.path.isdir(tifs) else path
        return len(util.find_files(folder, util.TIF_EXT + (".dat",))) >= 2

    def send_to_correction(self, path):
        if os.path.isdir(path) and not self.is_correction_folder(path):
            QMessageBox.information(
                self, "Can't add the folder",
                f"The folder must contain multiple tif or dat files:"
                f"\n{path}")
            return
        self.add_to_correction([path])
        self.set_status(f"Added to the Data correction input: {path}")

    def multi_selection_menu(self, paths, kind):
        """Menu for several selected folders or files (kind): add them to
        the Data correction input, or view them in the Multiple-scans
        viewer. Unsuitable ones are skipped."""
        if kind == "folder":
            to_correct = [p for p in paths if self.is_correction_folder(p)]
            scans = [p for p in paths if util.find_files(p)]
        else:
            to_correct = [p for p in paths if p.lower().endswith(
                util.TIF_EXT + util.HDF_EXT + (".dat",))]
            scans = [p for p in paths if p.lower().endswith(util.HDF_EXT)
                     and self.scans_viewer in self.viewers_for(p)]
        menu = QMenu(self)
        act_add = act_view = None
        if self.add_to_correction is not None and to_correct:
            act_add = menu.addAction(f"Add {len(to_correct)} {kind}(s) to "
                                     f"Data correction input")
        if self.view_scans is not None and scans:
            act_view = menu.addAction(f"View {len(scans)} scan(s) in "
                                      f"{self.scans_viewer}")
        if menu.isEmpty():
            self.set_status(f"The selected {kind}s can't be added to the "
                            f"Data correction input or viewed as scans.")
            return
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        if chosen == act_add:
            kept, what = to_correct, "the Data correction input"
            if kind == "folder":
                what += " (needs multiple tif or dat files)"
            self.add_to_correction(to_correct)
            self.set_status(f"Added {len(to_correct)} {kind}(s) to the Data "
                            f"correction input.")
        elif chosen == act_view:
            kept, what = scans, f"the {self.scans_viewer}"
            self.view_scans(scans)
        else:
            return
        skipped = [os.path.basename(p) for p in paths if p not in kept]
        if skipped:
            QMessageBox.information(
                self, "Skipped", f"Not suitable for {what}:\n"
                + "\n".join(skipped[:20])
                + ("\n..." if len(skipped) > 20 else ""))

    def viewers_for(self, path, folder_only=False):
        """Names of the viewers able to show a folder of tif files, a tif
        file (its folder) or an HDF file (by its dataset dimensions)."""
        if self.view_in is None:
            return []
        if os.path.isdir(path):
            return self.tif_viewers(path) if util.find_files(path) else []
        if folder_only:
            return []
        lower = path.lower()
        if lower.endswith(util.TIF_EXT):
            return self.tif_viewers(os.path.dirname(path))
        if lower.endswith(util.HDF_EXT):
            try:
                return util.viewers_for_hdf(path, self.viewer_ndims)
            except OSError:
                return []
        return []

    @staticmethod
    def tif_viewers(folder):
        """Viewers of a folder of tif files; the 2D-scan viewer only for
        files named '..._<row>_<col>.tif'."""
        is_2d = util.is_2d_scan_folder(folder)
        return [name for name in util.TIF_VIEWERS
                if is_2d or name not in util.TIF_2D_VIEWERS]

    def metadata_viewers(self, path):
        if self.metadata_to is None or not os.path.isfile(path) or \
                not path.lower().endswith((".json",) + util.HDF_EXT):
            return []
        return list(self.viewer_ndims)

    @staticmethod
    def add_viewer_menu(menu, title, names):
        """Submenu of viewer names; {action: name}."""
        if not names:
            return {}
        sub = menu.addMenu(title)
        return {sub.addAction(name): name for name in names}

    def on_tree_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        selected = self.tree.selectedItems()
        if item is not None and item.isSelected() and len(selected) > 1:
            paths = [i.data(0, Qt.UserRole) for i in selected]
            paths = sorted((p for p in paths if p and os.path.isdir(p)),
                           key=util.natural_key)
            self.multi_selection_menu(paths, "folder")
            return
        path = item.data(0, Qt.UserRole) if item else None
        if not path or not os.path.isdir(path):
            return
        try:
            with os.scandir(path) as entries:
                empty = not any(True for _ in entries)
        except OSError:
            empty = True
        menu = QMenu(self)
        act_add = act_delete = None
        if self.add_to_correction is not None and not empty:
            act_add = menu.addAction("Add folder to Data correction input")
        views = self.add_viewer_menu(menu, "View in",
                                     self.viewers_for(path, True))
        if os.path.abspath(path) != self.base_folder:
            if not menu.isEmpty():
                menu.addSeparator()
            act_delete = menu.addAction("Delete folder")
        if menu.isEmpty():
            return
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        if chosen == act_add:
            self.send_to_correction(path)
        elif chosen in views:
            self.view_in(views[chosen], path)
        elif chosen == act_delete:
            self.delete_folder(item, path)

    def delete_folder(self, item, path):
        try:
            num = sum(len(files) for _, _, files in os.walk(path))
        except OSError:
            num = 0
        answer = QMessageBox.warning(
            self, "Delete folder",
            f"Permanently delete this folder and its content ({num} "
            f"file(s))?\n{path}\n\nThis can't be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(path)
        except OSError as error:
            QMessageBox.critical(self, "Error", f"Can't delete the folder:"
                                                f"\n{error}")
            return
        deleted = os.path.abspath(path)
        parent = item.parent()
        if parent is not None:
            parent.removeChild(item)
        selected = os.path.abspath(self.selected_folder)
        if selected == deleted or selected.startswith(deleted + os.sep):
            self.list_files.clear()
            self.preview_file = ""
            self.preview_mode = None
            self.show_preview_page("info")
            self.lbl_preview_info.setText("Folder deleted.")
            if parent is not None:
                self.tree.setCurrentItem(parent)
        self.set_status(f"Deleted: {path}")

    def delete_file(self, item, path):
        answer = QMessageBox.warning(
            self, "Delete file",
            f"Permanently delete this file?\n{path}\n\nThis can't be "
            f"undone.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except OSError as error:
            QMessageBox.critical(self, "Error", f"Can't delete the file:"
                                                f"\n{error}")
            return
        self.list_files.takeItem(self.list_files.row(item))
        if path == self.preview_file:
            self.preview_file = ""
            self.preview_mode = None
            self.show_preview_page("info")
            self.lbl_preview_info.setText("File deleted.")
        self.set_status(f"Deleted: {path}")

    def on_file_context_menu(self, pos):
        item = self.list_files.itemAt(pos)
        selected = self.list_files.selectedItems()
        if item is not None and item.isSelected() and len(selected) > 1:
            paths = sorted((os.path.join(self.selected_folder, i.text())
                            for i in selected), key=util.natural_key)
            self.multi_selection_menu(paths, "file")
            return
        menu = QMenu(self)
        path = os.path.join(self.selected_folder, item.text()) if item \
            else self.selected_folder
        if item:
            try:
                stat = os.stat(path)
                # Creation time when the OS records it, else modification
                created = getattr(stat, "st_birthtime", None)
                label = "Created" if created else "Modified"
                date = datetime.fromtimestamp(created or stat.st_mtime)
                info = [f"Size: {util.format_bytes(stat.st_size)}",
                        f"{label}: {date:%Y-%m-%d %H:%M:%S}"]
            except OSError as error:
                info = [f"Can't read file info: {error}"]
            for text in info:
                menu.addAction(text).setEnabled(False)
            menu.addSeparator()
        act_add = None
        if self.add_to_correction is not None:
            if item and path.lower().endswith(util.TIF_EXT + util.HDF_EXT
                                              + (".dat",)):
                act_add = menu.addAction("Add file to Data correction "
                                         "input")
            elif not item and self.list_files.count():
                act_add = menu.addAction("Add folder to Data correction "
                                         "input")
        views = self.add_viewer_menu(menu, "View in",
                                     self.viewers_for(path))
        metas = self.add_viewer_menu(menu, "Use as metadata in",
                                     self.metadata_viewers(path)
                                     if item else [])
        if act_add is not None or views or metas:
            menu.addSeparator()
        act_copy = menu.addAction("Copy full path")
        act_folder = menu.addAction("Copy folder path")
        act_delete = None
        if item:
            menu.addSeparator()
            act_delete = menu.addAction("Delete file")
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        if chosen == act_add:
            self.send_to_correction(path)
        elif chosen in views:
            self.view_in(views[chosen], path)
        elif chosen in metas:
            self.metadata_to(metas[chosen], path)
        elif chosen == act_delete:
            self.delete_file(item, path)
        elif chosen == act_copy:
            QApplication.clipboard().setText(os.path.normpath(path))
        elif chosen == act_folder:
            QApplication.clipboard().setText(
                os.path.normpath(self.selected_folder))


# ---------------------------------------------------------------------------
# Data correction and conversion
# ---------------------------------------------------------------------------

def select_folders(parent, title, start):
    """Folder dialog allowing several folders (Ctrl/Shift-click)."""
    dialog = QFileDialog(parent, title, start)
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    # Native dialogs can't select several folders
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    for view in (dialog.findChild(QListView, "listView"),
                 dialog.findChild(QTreeView, "treeView")):
        if view is not None:
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    if not dialog.exec():
        return []
    return [f for f in dialog.selectedFiles() if os.path.isdir(f)]


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
        self.add_inputs(paths)

    @ren.safe_call
    def browse_folders(self):
        folders = select_folders(self, "Select scan folder(s)",
                                 self.last_folder("input"))
        if folders:
            # Remember the parent, to select sibling scans next time
            self.remember_folder("input", os.path.dirname(folders[0]))
            self.add_inputs(folders)

    def input_entries(self):
        return [self.list_input.item(i).data(Qt.UserRole)
                for i in range(self.list_input.count())]

    @ren.safe_call
    def add_inputs(self, paths):
        """Add rows: each folder or HDF file is a scan; other files are
        grouped by folder into one scan."""
        entries, groups = [], {}
        for path in paths:
            path = os.path.normpath(path)
            if os.path.isdir(path) or path.lower().endswith(util.HDF_EXT):
                entries.append(path)
            elif os.path.isfile(path):
                groups.setdefault(os.path.dirname(path), []).append(path)
        for files in groups.values():
            entries.append(";".join(sorted(files, key=util.natural_key)))
        existing = set(self.input_entries())
        for entry in entries:
            if entry in existing:
                continue
            existing.add(entry)
            files = entry.split(";")
            if len(files) > 1:
                label = f"{Path(files[0]).parent.name}: {len(files)} files"
            else:
                label = Path(entry).name
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, entry)
            item.setToolTip("\n".join(files[:20]) +
                            ("\n..." if len(files) > 20 else ""))
            self.list_input.addItem(item)
        hdf = next((e for e in self.input_entries()
                    if e.lower().endswith(util.HDF_EXT)), None)
        if hdf and self.combo_hdf_key.count() == 0:
            self.update_hdf_keys(self.combo_hdf_key, hdf)
        self.on_inputs_changed()

    def remove_inputs(self):
        for item in self.list_input.selectedItems():
            self.list_input.takeItem(self.list_input.row(item))
        self.on_inputs_changed()

    def clear_inputs(self):
        self.list_input.clear()
        self.on_inputs_changed()

    def on_inputs_changed(self):
        self.scans = []
        num = self.list_input.count()
        self.lbl_input_info.setText(
            f"{num} scan(s) in the list. Click 'Load'." if num
            else "No input loaded.")

    def get_scans(self):
        """[(name, items, metadata path, input entry), ...] of the input
        scans."""
        entries = self.input_entries()
        if not entries:
            raise ValueError("Please add input folders or files!")
        hdf_key = self.combo_hdf_key.currentText().strip()
        axis = self.get_value(self.edit_axis, "Axis", int)
        scans = []
        for entry in entries:
            items = util.raw_scan_items(entry, hdf_key, axis)
            first = entry.split(";")[0]
            scans.append((scan_name_of(first), items,
                          util.find_metadata_file(first), entry))
        return scans

    @ren.safe_call
    def load_input(self):
        self.scans = self.get_scans()
        name, items = self.scans[0][:2]
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
        self.set_status(f"Loaded coefficients of shape {coefs.shape}.")
        self.update_preview()

    def get_step(self, width):
        if self.coefs is None:
            raise ValueError("Please load the calibration coefficients!")
        method = self.combo_step.currentText()
        value = self.get_value(self.edit_step, "Value", allow_none=True)
        return util.get_output_step(self.coefs, width, method, value)

    @ren.safe_call
    def update_preview(self):
        self.corrected = self.corrected_raw = self.step = None
        if self.raw_image is not None and self.coefs is not None:
            width = self.raw_image.shape[1]
            step, source = self.get_step(width)
            self.corrected = util.correct_image(self.raw_image, self.coefs,
                                                step)
            self.corrected_raw = self.corrected
            self.step = step
            self.set_status(f"Preview updated. Output q-step: {step:.6g} "
                            f"({source}); q-max: {step * (width - 1):.4f}.")
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
        label = util.Q_LABEL
        if self.corrected is not None:
            height, width = self.corrected.shape
            self.show_image(self.p_corr, self.corrected,
                            (-0.5 * self.step, -0.5, width * self.step,
                             height))
            self.set_label(self.p_corr, "bottom", label)
            self.corr_line.setValue(self.strip)
            self.corr_line.show()
            self.p_strip.plot(np.arange(width) * self.step,
                              self.corrected[self.strip],
                              pen=ren.pg.mkPen(util.COLORS["blue"]))
            self.set_label(self.p_strip, "bottom", label)
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
        step, source = self.get_step(width)
        para = {"step": step, "step_source": source,
                "coef_file": self.edit_coef.text().strip(),
                "hdf_key": self.combo_hdf_key.currentText().strip(),
                "axis": self.get_value(self.edit_axis, "Axis", int),
                "format": fmt, "bck": self.get_background_para(),
                "xy_type": self.combo_xy_type.currentText(),
                "zip": self.combo_zip.currentText() == "zip",
                "raw": self.combo_xy_data.currentText() == "raw",
                "xy_folder": self.edit_xy_folder.text().strip() or "xy",
                "xy_x": self.combo_xy_x.currentText(),
                "max_two_theta": self.get_max_two_theta(
                    self.combo_xy_x, self.edit_max_tth),
                "out_key": util.check_output_key(self.edit_out_key.text())}
        return para

    def get_max_two_theta(self, combo, edit):
        if combo.currentText() != "two-theta":
            return util.DEFAULT_MAX_TWO_THETA
        value = self.get_value(edit, "Max 2theta")
        if not 0 < value < 180:
            raise ValueError("Max 2theta must be between 0 and 180 "
                             "degrees!")
        return value

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
        out_base = Path(folder).resolve()
        for name, _, _, entry in self.scans:
            in_base = util.input_base_folder(entry).resolve()
            if out_base == in_base:
                raise ValueError(
                    f"The output base folder must be different from the "
                    f"input base folder:\n{in_base}\nOtherwise '{name}' "
                    f"would be overwritten.")
        para = self.get_output_para(self.raw_image.shape[1])
        start = self.get_value(self.edit_start, "Start index", int)
        stop = self.get_value(self.edit_stop, "Stop index", int,
                              allow_none=True)
        jobs = []
        for name, items, meta_path, entry in self.scans:
            last = len(items) - 1 if stop is None else min(stop,
                                                           len(items) - 1)
            if not 0 <= start <= last:
                raise ValueError(f"Invalid image range [{start}, {stop}] "
                                 f"for {name} ({len(items)} images).")
            jobs.append((name, items[start:last + 1], meta_path, entry))
        existing = [name for name, *_ in jobs
                    if (Path(folder) / name).is_dir()
                    and any((Path(folder) / name).iterdir())]
        if existing:
            shown = ", ".join(existing[:10]) + \
                (", ..." if len(existing) > 10 else "")
            bck_note = ("Background removal is ON: use another output base "
                        "folder to keep the data with background.\n\n"
                        if para["bck"] is not None else "")
            answer = QMessageBox.warning(
                self, "Data may be overwritten",
                f"{len(existing)} scan folder(s) already exist in\n{folder}"
                f"\n{shown}\n\nTheir data may be overwritten.\n\n"
                f"{bck_note}Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                self.set_status("Correction cancelled: choose another "
                                "output base folder.")
                return
        coefs = self.coefs
        num_images = sum(len(job[1]) for job in jobs)

        def task(cancel_event):
            done_images = 0
            for name, items, meta_path, entry in jobs:
                metadata = losa.load_json(meta_path) if meta_path else {}
                num, cancelled = util.correct_scan(
                    items, Path(folder) / name, coefs, para, cancel_event,
                    self.set_progress, metadata, entry)
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
        path = QFileDialog.getExistingDirectory(
            self, "Output folder", self.last_folder("conv_output"))
        if path:
            self.remember_folder("conv_output", path)
            self.edit_conv_output.setText(path)

    @ren.safe_call
    def make_conv_output_folder(self):
        folder = self.edit_conv_output.text().strip()
        if not folder:
            raise ValueError("Please type the path of the folder to create!")
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.set_status(f"Folder ready: {folder}")

    def conversion_output(self, mode, source):
        """Output folder, or the HDF file in it for conversions to HDF."""
        folder = self.edit_conv_output.text().strip()
        if not folder:
            raise ValueError("Please select the output folder!")
        if mode not in ("tif_to_hdf", "tif2d_to_hdf4d"):
            return folder
        if not os.path.isdir(folder):
            raise ValueError(f"Output folder does not exist: {folder}. "
                             f"Click 'Make folder' to create it.")
        name = self.edit_conv_out_name.text().strip() or \
            Path(source.rstrip("/\\")).name
        if not name:
            raise ValueError("Please input the output file name!")
        if not name.lower().endswith(util.HDF_EXT):
            name += ".hdf"
        return os.path.join(folder, name)

    @ren.safe_call
    def run_conversion(self):
        mode = util.CONVERSIONS[self.combo_conversion.currentText()]
        source = self.edit_conv_source.text().strip()
        output = self.conversion_output(mode, source)
        hdf_key = self.combo_conv_key.currentText().strip()
        x_step = self.get_value(self.edit_conv_step, "Q step",
                                allow_none=True)
        xy_type = self.combo_conv_xy_type.currentText()
        zipped = self.combo_conv_zip.currentText() == "zip"
        x_axis = self.combo_conv_xy_x.currentText()
        max_tth = self.get_max_two_theta(self.combo_conv_xy_x,
                                         self.edit_conv_max_tth)
        out_key = util.check_output_key(self.edit_conv_out_key.text())
        nrow = ncol = chunks = None
        scan_order = self.combo_conv_order.currentText()
        if mode == "tif2d_to_hdf4d":
            nrow = self.get_value(self.edit_conv_nrow, "Rows", int,
                                  allow_none=True)
            ncol = self.get_value(self.edit_conv_ncol, "Columns", int,
                                  allow_none=True)
            if (nrow is None) != (ncol is None) or \
                    (nrow is not None and min(nrow, ncol) < 1):
                raise ValueError("Please input both rows and columns "
                                 "(positive), or leave both empty!")
            chunks = util.parse_chunks(self.edit_conv_chunks.text())

        def task(cancel_event):
            return util.convert_data(mode, source, output, hdf_key, x_step,
                                     xy_type, zipped, cancel_event,
                                     self.set_progress, x_axis, max_tth,
                                     out_key, nrow, ncol, scan_order, chunks)

        def done(result):
            message, _ = result
            self.lbl_conv_info.setText(message)
            self.set_status(message)

        self.run_worker(task, done, "Conversion in progress",
                        f"Converting {source}...")


# ---------------------------------------------------------------------------
# Viewers
# ---------------------------------------------------------------------------

def lazy_note(datasets):
    """Status note for scans read on demand (too big for memory)."""
    lazy = [d for d in datasets if d.lazy]
    if not lazy:
        return ""
    size = util.format_bytes(sum(d.nbytes for d in lazy))
    return (f" Big data ({size}): read on demand from the file, slower "
            f"q-slices and patterns.")


class ViewerLogic(BaseLogic):
    """Input and saving actions shared by the viewers."""

    # (min, max) number of dimensions of the HDF datasets the viewer shows
    hdf_ndims = (3, 3)
    # Set by the controller: name of the viewer, ranges of all viewers,
    # and open_in(name, path) to show a file in another viewer
    viewer_name = ""
    viewer_ndims = {}
    open_in = None

    def update_hdf_keys(self, path):
        try:
            keys = [k for k, _ in util.hdf_array_keys(path,
                                                      *self.hdf_ndims)]
        except OSError as error:
            self.set_status(f"Can't read {path}: {error}")
            return
        current = self.combo_hdf_key.currentText()
        self.combo_hdf_key.clear()
        self.combo_hdf_key.addItems(keys)
        if current in keys:
            self.combo_hdf_key.setCurrentText(current)

    def offer_other_viewers(self, error):
        """For HDF data this viewer can't show (util.WrongDimensionError):
        ask to open the file in a viewer able to show it."""
        names = []
        if self.open_in is not None:
            try:
                names = [name for name in util.viewers_for_hdf(
                    error.path, self.viewer_ndims)
                    if name != self.viewer_name]
            except OSError:
                pass
        if not names:
            raise error
        box = QMessageBox(QMessageBox.Question, "Open in another viewer",
                          f"{error}\n\nOpen the file in another viewer?",
                          parent=self)
        buttons = {box.addButton(f"Open in {name}", QMessageBox.AcceptRole):
                   name for name in names}
        box.addButton(QMessageBox.Cancel)
        QApplication.changeOverrideCursor(Qt.ArrowCursor)
        box.exec()
        name = buttons.get(box.clickedButton())
        if name is None:
            self.set_status(f"Not loaded: {error}")
            return
        self.open_in(name, error.path)

    def check_scan_input(self, path, hdf_key):
        """Check the dimensions of the HDF data of a 1D scan before
        reading them. False (after offering the other viewers) if this
        viewer can't show them."""
        try:
            util.check_scan_input(path, hdf_key)
        except util.WrongDimensionError as error:
            self.offer_other_viewers(error)
            return False
        return True

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
            self.reset_metadata()
            self.set_path(folder, append=self.multi_input)
            self.load_data()

    def reset_metadata(self):
        """Clear the metadata, q-step and positions of the previous data."""
        for combo in self.position_combos:
            combo.clear()
        self.edit_metadata.clear()
        self.edit_step.clear()

    @ren.safe_call
    def show_input(self, path):
        """Fill the inputs for a folder of tif files or an HDF file, with
        the metadata, HDF key and q-step found."""
        is_hdf = path.lower().endswith(util.HDF_EXT)
        if os.path.isfile(path) and not is_hdf:
            path = os.path.dirname(path)
        for combo in self.position_combos:
            combo.clear()
        self.edit_step.setText("")
        self.edit_metadata.blockSignals(True)
        self.edit_metadata.setText("" if is_hdf
                                   else util.find_metadata_file(path))
        self.edit_metadata.blockSignals(False)
        self.set_path(path)
        key = ""
        if is_hdf:
            keys = [self.combo_hdf_key.itemText(i)
                    for i in range(self.combo_hdf_key.count())]
            if util.HDF_DATA_KEY in keys:
                self.combo_hdf_key.setCurrentText(util.HDF_DATA_KEY)
            key = self.combo_hdf_key.currentText().strip()
        meta = self.edit_metadata.text().strip()
        metadata = losa.load_json(meta) \
            if meta.lower().endswith(".json") else {}
        step = util.get_x_step(metadata, path if is_hdf else None, key,
                               path)
        if step:
            self.edit_step.setText(f"{step:.10g}")
        self.set_status(f"Input: {path}")

    @ren.safe_call
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a data file", self.last_folder("data"),
            "Data (*.hdf *.h5 *.hdf5 *.nxs *.tif *.tiff);;All files (*.*)")
        if path:
            self.remember_folder("data", path)
            self.reset_metadata()
            self.set_path(path, append=self.multi_input)
            self.load_data()

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
                "x_step": self.get_value(self.edit_step, "Q step",
                                         allow_none=True)}

    def radius(self):
        return int(self.combo_radius.currentText())

    def loaded_datasets(self):
        if getattr(self, "datasets", None):
            return list(self.datasets)
        data = getattr(self, "dataset", None)
        return [] if data is None else [data]

    def x_calibrated(self):
        return all(d.calibrated for d in self.loaded_datasets())

    def x_label(self):
        return util.Q_LABEL if self.x_calibrated() else "Pixel index"

    def x_name(self):
        return "Q" if self.x_calibrated() else "Pixel"

    def check_x_step(self):
        """Show the q-step found for the loaded data, or ask for it."""
        datasets = self.loaded_datasets()
        if datasets and self.x_calibrated():
            if not self.edit_step.text().strip():
                self.edit_step.setText(f"{datasets[0].x_step:.10g}")
            return
        if datasets:
            QMessageBox.information(
                self, "Q-step not found",
                "The q-step wasn't found in the map_info, the metadata or "
                "the HDF file. The x-axis is shown in pixels.\n\nType the "
                "q-step in 'Q step' and press Enter to show q.")

    @ren.safe_call
    def on_step_entered(self):
        """Apply the typed q-step to the loaded data (empty: pixels)."""
        datasets = self.loaded_datasets()
        if not datasets:
            return
        step = self.get_value(self.edit_step, "Q step", allow_none=True)
        if step is not None and step <= 0:
            raise ValueError("The q-step must be positive!")
        for data in datasets:
            data.set_x_step(step)
        self.refresh_all()
        for plot in self.plots:
            plot.enableAutoRange()
        self.set_status(f"X-axis: {self.x_label()}.")

    def show_scan_image(self, plot, mat, x_step):
        height, width = mat.shape
        self.show_image(plot, mat, (-0.5 * x_step, -0.5, width * x_step,
                                    height))
        self.set_label(plot, "bottom", self.x_label())

    def show_qslice(self, mat, y_label):
        self.show_image(self.p_slice, mat)
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


class PositionsMixin:
    """Metadata (JSON or nxs/HDF) and choice of the positions of the
    images by their key, shared by the 1D-scan viewers. With several scans,
    the keys are listed from the first scan and each scan reads its own
    metadata."""

    def __init__(self):
        super().__init__()
        self.edit_path.editingFinished.connect(self.update_positions)
        self.combo_hdf_key.currentTextChanged.connect(
            lambda _t: self.update_positions())
        self.edit_metadata.textChanged.connect(
            lambda _t: self.update_positions())

    def first_path(self):
        return self.edit_path.text().split(";")[0].strip()

    def set_path(self, path, append=False):
        super().set_path(path, append)
        self.update_positions()

    @ren.safe_call
    def browse_metadata(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata", self.last_folder("metadata"),
            "Metadata (*.json *.nxs *.hdf *.h5 *.hdf5);;All files (*.*)")
        if path:
            self.remember_folder("metadata", path)
            self.edit_metadata.setText(path)

    def update_positions(self):
        """List the 1D arrays of the metadata (JSON lists, or 1D datasets
        of a nxs/HDF file) with one value per image."""
        items, num = [], None
        try:
            items, num = util.list_positions(
                self.first_path(), self.combo_hdf_key.currentText().strip(),
                self.edit_metadata.text())
        except (OSError, ValueError) as error:
            self.set_status(f"Can't list the positions: {error}")
        current = self.combo_motor.currentData()
        self.combo_motor.clear()
        for key, size in items:
            self.combo_motor.addItem(f"{key}  ({size})", key)
        index = self.combo_motor.findData(current)
        if index >= 0:
            self.combo_motor.setCurrentIndex(index)
        self.combo_motor.setEnabled(bool(items))
        if num is not None:
            self.combo_motor.setToolTip(
                f"Positions of the {num} images: a 1D array with {num} "
                f"values. Its name labels the axes.")

    def input_options(self):
        options = super().input_options()
        options["motor_key"] = self.combo_motor.currentData() or ""
        return options

    def metadata_for_all(self):
        """Metadata file of the positions of all scans: the one given, or
        the one of the first scan."""
        return self.edit_metadata.text().strip() or \
            util.positions_source(self.first_path())[1]

    def warn_positions(self, notes):
        if notes:
            QMessageBox.warning(self, "Positions not used",
                                "\n".join(notes))

    @ren.safe_call
    def load_metadata(self):
        """Use the selected positions for the loaded data, without loading
        the images again."""
        self.update_positions()
        datasets = self.loaded_datasets()
        if not datasets:
            self.set_status("Metadata listed. Load the data to use the "
                            "positions.")
            return
        metadata = self.metadata_for_all()
        key = self.combo_motor.currentData() or ""
        notes = []
        for data in datasets:
            name, positions, note = util.resolve_positions(
                data.source, metadata, key, data.depth)
            data.set_positions(name, positions)
            data.position_note = note
            if note:
                notes.append(f"{data.name}: {note}")
        self.refresh_all()
        data = datasets[0]
        self.set_status(" ".join(notes) or (
            "Positions: image index." if data.motor_name == "Image"
            else f"Positions: {data.motor_name}."))
        self.warn_positions(notes)

    @staticmethod
    def position_text(data, idx):
        """Motor name and position of an image, for plot titles."""
        if data.motor_name == "Image":
            return f"Image {idx}"
        return f"{data.motor_name} = {data.positions[idx]:.6g} | Image " \
               f"{idx}"


class Positions2DMixin(PositionsMixin):
    """Metadata and positions of the two axes of a 2D scan: a key (1D
    array) for the rows and one for the columns."""

    def position_keys(self):
        return tuple(combo.currentData() or ""
                     for combo in self.position_combos)

    def scan_grid(self):
        """(nrow, ncol, scan_order) of a 3D HDF dataset; None if not
        used."""
        return None

    def update_positions(self):
        """List the 1D arrays of the metadata with one value per row (axis
        0) or per column (axis 1)."""
        shape, arrays = None, []
        try:
            shape = util.scan2d_shape(
                self.first_path(), self.combo_hdf_key.currentText().strip(),
                self.scan_grid())
            arrays = util.list_position_arrays(self.first_path(),
                                               self.edit_metadata.text())
        except (OSError, ValueError) as error:
            self.set_status(f"Can't list the positions: {error}")
        units = ("row", "column")
        for axis, combo in enumerate(self.position_combos):
            num = shape[axis] if shape else None
            current = combo.currentData()
            items = [(key, size) for key, size in arrays if size == num]
            combo.clear()
            for key, size in items:
                combo.addItem(f"{key}  ({size})", key)
            index = combo.findData(current)
            if index < 0 and axis == 0 and shape and shape[0] == shape[1] \
                    and len(items) > 1:
                # Same number of rows and columns: the columns are listed
                # first, the rows second
                index = 1
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.setEnabled(bool(items))
            if num is not None:
                combo.setToolTip(f"Positions of the {num} {units[axis]}s: "
                                 f"a 1D array with {num} values. Its name "
                                 f"labels the plots.")

    def input_options(self):
        # The 2D scan takes one key per axis, not the key of PositionsMixin
        options = super(PositionsMixin, self).input_options()
        options["motor_keys"] = self.position_keys()
        return options

    def positions_status(self):
        names = [name for name, _ in self.dataset.motors]
        return f"Positions: {names[0]} (rows), {names[1]} (columns)."

    @ren.safe_call
    def load_metadata(self):
        """Use the selected positions for the loaded data, without loading
        the images again."""
        self.update_positions()
        data = self.dataset
        if data is None:
            self.set_status("Metadata listed. Load the data to use the "
                            "positions.")
            return
        data.apply_positions(self.position_keys(), self.metadata_for_all())
        self.apply_motor_labels()
        self.refresh_all()
        self.set_status(" ".join(data.position_notes)
                        or self.positions_status())
        self.warn_positions(data.position_notes)


class EddtViewer(PositionsMixin, ViewerLogic, ren.EddtViewerUI):
    """Viewer of a 1D scan; MultiLinesViewer shows a waterfall of all
    images."""

    hdf_ndims = (2, 3)

    @ren.safe_call
    def load_data(self):
        path = self.edit_path.text()
        options = self.input_options()
        if not self.check_scan_input(path, options["hdf_key"]):
            return

        def task(cancel_event):
            return util.load_scan(path, cancel_event=cancel_event,
                                  progress=self.set_progress, **options)

        self.run_worker(task, self.on_data_loaded, "Loading data",
                        f"Loading {path}...")

    def on_data_loaded(self, dataset):
        if dataset is None:
            self.set_status("Loading cancelled.")
            return
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = dataset
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
        positions = data.position_note or (
            "Positions: image index." if data.motor_name == "Image"
            else f"Positions: {data.motor_name}.")
        step = f"Q-step: {data.x_step:.6g}." if data.calibrated \
            else "Q-step not found: x-axis in pixels."
        self.set_status(f"Loaded {data.name}: {data.depth} image(s) of "
                        f"shape ({data.height}, {data.width}). {step} "
                        f"{positions}{lazy_note([data])} Click on the "
                        f"image to select a row.")
        self.check_x_step()
        self.warn_positions([data.position_note] if data.position_note
                            else [])

    def update_image(self):
        data = self.dataset
        if data is None:
            return
        idx = self.slider_img.value()
        self.show_scan_image(self.p_img, data.image(idx), data.x_step)
        self.set_title(self.p_img, self.position_text(data, idx))
        self.set_crosshairs(idx)
        self.update_profile()

    def update_qslice(self):
        data = self.dataset
        if data is None:
            return
        x_idx = self.slider_x.value()
        self.qslice_raw = data.q_slice(x_idx, self.radius())
        self.show_qslice(self.qslice_raw, data.motor_name)
        self.set_title(self.p_slice, f"Mean of slices around "
                                     f"{self.x_name()}="
                                     f"{x_idx * data.x_step:.4g}")
        self.set_span(x_idx, data.x_step)
        self.set_crosshairs(self.slider_img.value())

    def update_profile(self):
        data = self.dataset
        if data is None or self.row is None:
            return
        idx = self.slider_img.value()
        if self.stack_mode:
            patterns = data.row_patterns(self.row)
            curves = [(data.x_list, patterns[i], f"image_{i}")
                      for i in range(data.depth)]
            self.plot_waterfall(curves, idx, f"{data.motor_name} "
                                             f"(image index, offset)")
            self.set_title(self.p_prof, f"Patterns of all images at row "
                                        f"{self.row} | Red: "
                                        f"{self.position_text(data, idx)}")
        else:
            self.plot_single(data.x_list, data.pattern(idx, self.row),
                             f"Pattern at row {self.row} | "
                             f"{self.position_text(data, idx)}")

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


class GLWaterfallMixin:
    """Patterns drawn as lines in a 3D view (OpenGL): x = q, y = image or
    scan, z = intensity, with a hover readout. Viewers without the 3D view
    (no PyOpenGL) keep their 2D waterfall."""

    def __init__(self):
        super().__init__()
        self.gl_points = None
        self.gl_marker = None
        if self.gl_view is not None:
            self.gl_view.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.gl_view:
            if event.type() == QEvent.MouseMove and not event.buttons():
                self.on_gl_hover(event.position())
            elif event.type() == QEvent.Leave and self.cursor_func:
                self.cursor_func("")
        return super().eventFilter(obj, event)

    def on_gl_hover(self, pos):
        """Show x, y, z of the line point nearest to the mouse."""
        if self.gl_points is None or self.cursor_func is None:
            return
        view = self.gl_view
        viewport = view.getViewport()
        matrix = view.projectionMatrix(viewport, viewport) * \
            view.viewMatrix()
        mat = np.array(matrix.copyDataTo(), dtype=float).reshape(4, 4)
        pts = np.column_stack([self.gl_points,
                               np.ones(len(self.gl_points))])
        clip = pts @ mat.T
        w_clip = clip[:, 3]
        valid = w_clip > 1e-9
        if not np.any(valid):
            return
        ndc = clip[valid, :2] / w_clip[valid, None]
        px = (ndc[:, 0] + 1) / 2 * view.width()
        py = (1 - ndc[:, 1]) / 2 * view.height()
        dist2 = (px - pos.x()) ** 2 + (py - pos.y()) ** 2
        i = int(np.argmin(dist2))
        if dist2[i] > util.GL_HOVER_PX ** 2:
            self.cursor_func("")
            return
        x_val, line, z_val = self.gl_values[np.flatnonzero(valid)[i]]
        self.cursor_func(f"x: {self.x_name()} = {x_val:.5g}   y: "
                         f"{self.gl_y_name} = {self.gl_y_texts[int(line)]}"
                         f"   z: intensity = {z_val:.5g}")

    def clear_profile(self):
        if self.gl_view is None:
            return super().clear_profile()
        self.gl_view.clear()
        self.gl_marker = None
        self.gl_points = None
        self.pattern_curves = None

    def set_span(self, x_index, x_step):
        if self.gl_view is None:
            return super().set_span(x_index, x_step)
        radius = self.radius()
        self.img_span.setRegion(((x_index - radius - 0.5) * x_step,
                                 (x_index + radius + 0.5) * x_step))
        self.img_span.show()
        self.update_gl_marker()

    @staticmethod
    def gl_color(color, alpha=1.0):
        rgba = ren.pg.mkColor(color).getRgbF()
        return rgba[0], rgba[1], rgba[2], alpha

    def draw_gl_curves(self, curves, current, y_name, y_texts):
        """Draw curves [(x, intensity), ...] scaled into util.GL_BOX, one
        per y step, the current one in red, with a grid, the axis names and
        their ranges. y_texts: label of each curve on the y axis."""
        view = self.gl_view
        view.clear()
        size_x, size_y, size_z = util.GL_BOX
        width = max(len(x) for x, _ in curves)
        step = max(1, int(np.ceil(width / util.GL_MAX_POINTS)))
        curves = [(np.asarray(x[::step], dtype=float),
                   np.asarray(y[::step], dtype=float)) for x, y in curves]
        x_min = min(float(x[0]) for x, _ in curves)
        x_max = max(float(x[-1]) for x, _ in curves)
        z_min = min(min(float(np.min(y)) for _, y in curves), 0.0)
        z_max = max(float(np.max(y)) for _, y in curves)
        x_range = max(x_max - x_min, 1e-12)
        z_range = max(z_max - z_min, 1e-12)
        num = len(curves)
        y_n = np.linspace(0, size_y, num) if num > 1 else np.zeros(1)
        self.gl_scale = (x_min, x_range)
        self.gl_y_name, self.gl_y_texts = y_name, y_texts
        fg = ren.sel_theme["PG_FG"]
        grid_color = ren.pg.mkColor(fg)
        grid_color.setAlphaF(0.25)
        grid = ren.gl.GLGridItem(color=grid_color)
        grid.setSize(size_x, size_y)
        grid.setSpacing(size_x / 10, size_y / 10)
        grid.translate(size_x / 2, size_y / 2, 0)
        view.addItem(grid)
        points, values = [], []
        for i, (x, y) in enumerate(curves):
            pos = np.column_stack([(x - x_min) / x_range * size_x,
                                   np.full(len(x), y_n[i]),
                                   (y - z_min) / z_range * size_z])
            points.append(pos)
            # Point values for hovering: x, curve index, intensity
            values.append(np.column_stack([x, np.full(len(x), i), y]))
            is_current = i == current
            color = util.COLORS["red"] if is_current else util.COLORS["blue"]
            view.addItem(ren.gl.GLLinePlotItem(
                pos=pos, color=self.gl_color(color, 1.0 if is_current
                                             else 0.8),
                width=2.5 if is_current else 1.0, antialias=True,
                mode="line_strip"))
        self.gl_points = np.vstack(points)
        self.gl_values = np.vstack(values)
        labels = [
            ((size_x / 2, -0.12 * size_y, 0), self.x_label()),
            ((0, -0.06 * size_y, 0), f"{x_min:.3g}"),
            ((size_x, -0.06 * size_y, 0), f"{x_max:.3g}"),
            ((size_x * 1.04, size_y / 2, 0), y_name),
            ((size_x * 1.02, 0, 0), y_texts[0]),
            ((size_x * 1.02, size_y, 0), y_texts[-1]),
            ((0, size_y * 1.02, size_z), f"Intensity max {z_max:.3g}")]
        font = ren.QFont(ren.QApplication.font())
        font.setPointSize(max(font.pointSize() - 2, 1))
        for pos, text in labels:
            view.addItem(ren.gl.GLTextItem(pos=pos, text=str(text),
                                           color=ren.pg.mkColor(fg),
                                           font=font))
        self.gl_marker = ren.gl.GLLinePlotItem(
            color=self.gl_color(util.COLORS["yellow"]), width=2.0,
            antialias=True, mode="line_strip")
        view.addItem(self.gl_marker)
        self.update_gl_marker()

    def update_gl_marker(self):
        """Line at the q of the q-slice center, across all curves."""
        x_value = self.gl_marker_x()
        if self.gl_marker is None or x_value is None:
            return
        x_min, x_range = self.gl_scale
        size_x, size_y, _ = util.GL_BOX
        x = (x_value - x_min) / x_range * size_x
        self.gl_marker.setData(pos=np.array([[x, 0, 0], [x, size_y, 0]]))


class MultiLinesViewer(GLWaterfallMixin, EddtViewer,
                       ren.MultiLinesViewerUI):
    """Patterns of all images at a row, in 3D (x = q, y = image, z =
    intensity), or a 2D waterfall without PyOpenGL."""

    stack_mode = True

    def gl_marker_x(self):
        data = self.dataset
        return None if data is None else self.slider_x.value() * data.x_step

    def update_profile(self):
        if self.gl_view is None:
            return super().update_profile()
        data = self.dataset
        if data is None or self.row is None:
            return
        idx = self.slider_img.value()
        patterns = data.row_patterns(self.row)
        self.pattern_curves = [(data.x_list, patterns[i], f"image_{i}")
                               for i in range(data.depth)]
        self.draw_gl_curves([(x, y) for x, y, _ in self.pattern_curves], idx,
                            data.motor_name,
                            [f"{p:.4g}" for p in data.positions])
        self.lbl_gl_title.setText(f"Patterns of all images at row "
                                  f"{self.row} | Red: "
                                  f"{self.position_text(data, idx)}")


class MultiScansViewer(GLWaterfallMixin, PositionsMixin, ViewerLogic,
                       ren.MultiScansViewerUI):
    """Viewer of several 1D scans with the same number of images; the
    patterns of all scans are shown in 3D (x = q, y = scan, z =
    intensity)."""

    def gl_marker_x(self):
        if not self.datasets:
            return None
        data, _ = self.current()
        return self.slider_x.value() * data.x_step

    def scan_paths(self):
        return [p.strip() for p in self.edit_path.text().split(";")
                if p.strip()]

    def num_images(self, path):
        """Number of images of a scan (folder of tif files or HDF file), or
        None if unknown."""
        if os.path.isdir(path):
            return len(util.find_files(path))
        if path.lower().endswith(util.HDF_EXT) and os.path.isfile(path):
            key = self.combo_hdf_key.currentText().strip()
            try:
                return util.hdf_num_images(path, key) if key else None
            except OSError:
                return None
        return None

    def add_scans(self, paths):
        """Add scans to the list, dropping (with a warning) those whose
        number of images differs from the first scan, then load."""
        current = self.scan_paths()
        for path in paths:
            if path.lower().endswith(util.HDF_EXT) and \
                    not self.combo_hdf_key.count():
                self.update_hdf_keys(path)
        new = [p for p in dict.fromkeys(paths) if p not in current]
        counts = {p: self.num_images(p) for p in current + new}
        known = [c for c in counts.values() if c is not None]
        if current and counts.get(current[0]) is not None:
            ref = counts[current[0]]
        elif known:
            # Most common number of images among the selected scans
            ref = max(set(known), key=known.count)
        else:
            ref = None
        kept, dropped = [], []
        for path in current + new:
            count = counts[path]
            if ref is not None and count is not None and count != ref:
                dropped.append(f"{scan_name_of(path)}: {count} images")
            else:
                kept.append(path)
        self.edit_path.setText("; ".join(kept))
        self.update_positions()
        if dropped:
            QMessageBox.warning(
                self, "Scans dropped",
                f"All scans must have the same number of images ({ref}). "
                f"Dropped:\n" + "\n".join(dropped))
        if kept:
            self.load_data()

    @ren.safe_call
    def view_scans(self, paths):
        """Replace the scans with folders / HDF files (e.g. selected in the
        Data browser); the first one gives the HDF key, metadata and
        q-step."""
        self.show_input(paths[0])
        self.add_scans(paths)

    @ren.safe_call
    def browse_folder(self):
        folders = select_folders(self, "Select scan folder(s) of tif "
                                       "files", self.last_folder("data"))
        if folders:
            self.remember_folder("data", os.path.dirname(folders[0]))
            self.add_scans(folders)

    @ren.safe_call
    def browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select data file(s)", self.last_folder("data"),
            "Data (*.hdf *.h5 *.hdf5 *.nxs *.tif *.tiff);;All files (*.*)")
        if not paths:
            return
        self.remember_folder("data", paths[0])
        # A tif file stands for its folder
        scans = [p if p.lower().endswith(util.HDF_EXT)
                 else os.path.dirname(p) for p in paths]
        self.add_scans(list(dict.fromkeys(scans)))

    @ren.safe_call
    def load_data(self):
        paths = self.scan_paths()
        if not paths:
            raise ValueError("Please select the folders/files of the scans!")
        options = self.input_options()
        # The metadata of the first scan gives the positions of all scans
        options["metadata_path"] = self.metadata_for_all()
        for path in paths:
            if not self.check_scan_input(path, options["hdf_key"]):
                return
        # Scans share the memory budget; bigger ones are read on demand
        options["max_memory_mb"] = util.SCAN_IN_MEMORY_MB / len(paths)

        def task(cancel_event):
            datasets = []
            try:
                for k, path in enumerate(paths):
                    def progress(text, k=k):
                        self.set_progress(f"Scan {k + 1}/{len(paths)}: "
                                          f"{text}")

                    data = util.load_scan(path, cancel_event=cancel_event,
                                          progress=progress, **options)
                    if data is None:
                        break
                    datasets.append(data)
            except Exception:
                for data in datasets:
                    data.close()
                raise
            if len(datasets) < len(paths):
                for data in datasets:
                    data.close()
                return None
            return paths, datasets

        self.run_worker(task, self.on_data_loaded, "Loading data",
                        f"Loading {len(paths)} scan(s)...")

    def on_data_loaded(self, result):
        if result is None:
            self.set_status("Loading cancelled.")
            return
        paths, datasets = result
        # Same number of images as the first scan (e.g. HDF scans unchecked
        # when added)
        ref = datasets[0].depth
        dropped = [f"{d.name}: {d.depth} images" for d in datasets
                   if d.depth != ref]
        if dropped:
            for data in datasets:
                if data.depth != ref:
                    data.close()
            pairs = [(p, d) for p, d in zip(paths, datasets)
                     if d.depth == ref]
            paths, datasets = [p for p, _ in pairs], [d for _, d in pairs]
            self.edit_path.setText("; ".join(paths))
            QMessageBox.warning(
                self, "Scans dropped",
                f"All scans must have the same number of images ({ref}). "
                f"Dropped:\n" + "\n".join(dropped))
        for data in self.datasets or []:
            data.close()
        self.datasets = datasets
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
        notes = [f"{d.name}: {d.position_note}" for d in self.datasets
                 if d.position_note]
        first = self.datasets[0]
        positions = " ".join(notes) or (
            "Positions: image index." if first.motor_name == "Image"
            else f"Positions: {first.motor_name}.")
        self.set_status(f"Loaded {len(self.datasets)} scan(s): "
                        f"{', '.join(d.name for d in self.datasets)}. "
                        f"{positions}{lazy_note(self.datasets)}")
        self.check_x_step()
        self.warn_positions(notes)

    def current(self):
        data = self.datasets[self.slider_scan.value()]
        return data, min(self.slider_img.value(), data.depth - 1)

    def update_image(self):
        if not self.datasets:
            return
        data, idx = self.current()
        self.show_scan_image(self.p_img, data.image(idx), data.x_step)
        self.set_title(self.p_img, f"Scan: {data.name} | "
                                   f"{self.position_text(data, idx)}")
        self.update_qslice()
        self.update_profile()

    def update_qslice(self):
        if not self.datasets:
            return
        data, idx = self.current()
        x_idx = min(self.slider_x.value(), data.width - 1)
        self.qslice_raw = data.q_slice(x_idx, self.radius())
        self.show_qslice(self.qslice_raw, data.motor_name)
        self.set_title(self.p_slice, f"Scan: {data.name} | Mean around "
                                     f"{self.x_name()}="
                                     f"{x_idx * data.x_step:.4g}")
        self.set_span(x_idx, data.x_step)
        self.set_crosshairs(idx)

    def update_profile(self):
        if not self.datasets or self.row is None:
            return
        img_idx = self.slider_img.value()
        curves = []
        for data in self.datasets:
            if img_idx < data.depth and self.row < data.height:
                curves.append((data.x_list, data.pattern(img_idx, self.row),
                               data.name))
            else:
                curves.append((data.x_list, np.zeros(data.width),
                               data.name))
        current = self.slider_scan.value()
        if self.gl_view is not None:
            self.pattern_curves = curves
            self.draw_gl_curves([(x, y) for x, y, _ in curves], current,
                                "Scan", [d.name for d in self.datasets])
            data = self.datasets[current]
            self.lbl_gl_title.setText(
                f"Patterns of all scans at row {self.row} | "
                f"{self.position_text(data, min(img_idx, data.depth - 1))}"
                f" | Red: {data.name}")
            return
        self.plot_waterfall(curves, current, "Scan index (offset)")
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


class Scan2DViewer(Positions2DMixin, ViewerLogic, ren.Scan2DViewerUI):
    """Viewer of one 2D scan (tif files loaded on demand, or a 4D HDF),
    with the positions of its rows and columns from the metadata."""

    hdf_ndims = (3, 4)

    def __init__(self):
        super().__init__()
        for edit in (self.edit_scan_nrow, self.edit_scan_ncol):
            edit.editingFinished.connect(self.update_positions)
        self.scan_grid_box.setEnabled(False)
        self.lbl_scan_grid.setEnabled(False)

    def scan_grid(self):
        if not self.scan_grid_box.isEnabled():
            return None
        nrow = self.get_value(self.edit_scan_nrow, "Rows", int,
                              allow_none=True)
        ncol = self.get_value(self.edit_scan_ncol, "Columns", int,
                              allow_none=True)
        return nrow, ncol, self.combo_scan_order.currentText()

    def update_positions(self):
        # The rows, columns and scan order are only for 3D HDF datasets
        path = self.first_path()
        ndim = None
        if os.path.isfile(path) and path.lower().endswith(util.HDF_EXT):
            try:
                ndim = util.hdf_ndim(path,
                                     self.combo_hdf_key.currentText().strip())
            except OSError:
                pass
        self.scan_grid_box.setEnabled(ndim == 3)
        self.lbl_scan_grid.setEnabled(ndim == 3)
        super().update_positions()

    def input_options(self):
        options = super().input_options()
        options["scan_grid"] = self.scan_grid()
        return options

    def apply_motor_labels(self):
        """Names of the scan axes on the sliders and the q-slice choice."""
        data = self.dataset
        for slider, (name, _) in zip((self.slider_row, self.slider_col),
                                     data.motors):
            slider.label.setText(f"{name} (index)")
        index = self.combo_slice_axis.currentIndex()
        self.combo_slice_axis.blockSignals(True)
        self.combo_slice_axis.clear()
        self.combo_slice_axis.addItems([f"Axis 0: {data.motors[0][0]}",
                                        f"Axis 1: {data.motors[1][0]}"])
        self.combo_slice_axis.setCurrentIndex(max(index, 0))
        self.combo_slice_axis.blockSignals(False)

    def position_title(self, row, col):
        (name0, pos0), (name1, pos1) = self.dataset.motors
        return f"{name0} = {pos0[row]:.6g} | {name1} = {pos1[col]:.6g}"

    @ren.safe_call
    def load_data(self):
        try:
            dataset = util.Scan2DDataset(self.edit_path.text(),
                                         **self.input_options())
        except util.WrongDimensionError as error:
            self.offer_other_viewers(error)
            return
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
        self.apply_motor_labels()
        self.refresh_all()
        for plot in (self.p_img, self.p_slice):
            plot.enableAutoRange()
        order = f" Scan order: {dataset.scan_order}." \
            if dataset.scan_order else ""
        self.set_status(f"Loaded 2D scan {dataset.name}: "
                        f"{dataset.nrow} x {dataset.ncol} images of shape "
                        f"({dataset.height}, {dataset.width}).{order} "
                        f"{self.positions_status()}")
        self.check_x_step()
        self.warn_positions(dataset.position_notes)

    def indices(self):
        return self.slider_row.value(), self.slider_col.value()

    def update_image(self):
        data = self.dataset
        if data is None:
            return
        row, col = self.indices()
        self.show_scan_image(self.p_img, data.get_image(row, col),
                             data.x_step)
        self.set_title(self.p_img, self.position_title(row, col))
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
        self.set_span(x_idx, data.x_step)
        if key != self.qslice_key:
            fixed_slider = self.slider_col if axis == 0 else self.slider_row
            needs_read = self.qslice_key is None or \
                self.qslice_key[:2] != (axis, fixed)
            if self.slider_x.is_dragging() or (
                    needs_read and fixed_slider.is_dragging()):
                # Load when the slider is released
                return
            if needs_read:
                self.set_status(f"Loading q-slices along "
                                f"{data.motors[axis][0]}...")
                QApplication.processEvents()
            self.qslice_raw = data.q_slice(axis, fixed, x_idx, radius)
            self.qslice_key = key
            if needs_read:
                self.set_status(f"Loaded q-slices along "
                                f"{data.motors[axis][0]}.")
        self.show_qslice(self.qslice_raw, data.motors[axis][0])
        self.set_title(self.p_slice, f"Mean around {self.x_name()}="
                                     f"{x_idx * data.x_step:.4g}")
        self.set_crosshairs(row if axis == 0 else col)

    def update_profile(self):
        data = self.dataset
        if data is None or self.row is None:
            return
        row, col = self.indices()
        self.plot_single(data.x_list, data.get_image(row, col)[self.row],
                         f"Pattern at row {self.row} | "
                         f"{self.position_title(row, col)}")

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


class Volume4DViewer(Positions2DMixin, ViewerLogic, ren.Volume4DViewerUI):
    """Viewer of a 4D HDF dataset (q, Z, rows, columns): volume at a
    q-range in 3D, q-Z image, 2D slice of the volume and pattern. Data are
    read on demand; big files can be downsampled."""

    hdf_ndims = (4, 4)
    position_title = Scan2DViewer.position_title

    def apply_motor_labels(self):
        """Names of the scan axes on the sliders, slices and 3D view."""
        (name0, _), (name1, _) = self.dataset.motors
        self.slider_row.label.setText(name0)
        self.slider_col.label.setText(name1)
        self.align_slider_labels()
        combo = self.combo_slice_dir
        index = combo.currentIndex()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([f"Fixed {name0}: {name1} vs Z",
                        f"Fixed {name1}: {name0} vs Z",
                        f"Fixed Z: {name0} vs {name1}"])
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)
        if self.gl_view is not None:
            self.draw_gl_frame()
            self.draw_volume()

    @ren.safe_call
    def load_data(self):
        try:
            dataset = util.Volume4DDataset(self.edit_path.text(),
                                           **self.input_options())
        except util.WrongDimensionError as error:
            self.offer_other_viewers(error)
            return
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = dataset
        self.volume = self.volume_key = None
        self.clear_profile()
        for slider, num, value in (
                (self.slider_x, dataset.width, dataset.width // 2),
                (self.slider_row, dataset.nrow, 0),
                (self.slider_col, dataset.ncol, 0),
                (self.slider_z, dataset.height, dataset.height // 2)):
            slider.set_range(num - 1)
            slider.set_value(value, notify=False)
        self.edit_ds_output.setText(
            util.downsample_output_path(dataset.source))
        self.apply_motor_labels()
        self.refresh_all()
        for plot in self.plots:
            plot.enableAutoRange()
        if self.gl_view is not None:
            self.reset_gl_camera()
        nbytes = util.format_bytes(dataset.volume_nbytes(self.radius()))
        self.set_status(f"Loaded 4D dataset {dataset.name}: (q, Z, rows, "
                        f"columns) = {dataset.shape}; {nbytes} read per "
                        f"q-range. {self.positions_status()}")
        self.check_x_step()
        self.warn_positions(dataset.position_notes)

    def indices(self):
        return (self.slider_row.value(), self.slider_col.value(),
                self.slider_z.value())

    def refresh_all(self):
        self.update_image()
        self.update_volume()

    def update_image(self):
        """q-Z image at the scan position, its pattern and the slice."""
        data = self.dataset
        if data is None:
            return
        if self.slider_row.is_dragging() or self.slider_col.is_dragging():
            # The slice comes from the volume in memory; the image waits
            self.update_slice()
            return
        row, col, _ = self.indices()
        self.show_scan_image(self.p_img, data.get_image(row, col),
                             data.x_step)
        self.set_title(self.p_img, self.position_title(row, col))
        self.update_z()

    def update_z(self):
        if self.dataset is None:
            return
        self.update_profile()
        self.update_slice()

    def update_profile(self):
        data = self.dataset
        row, col, z = self.indices()
        self.plot_single(data.x_list, data.get_image(row, col)[z],
                         f"Pattern at Z {z} | {self.position_title(row, col)}")
        self.img_row_line.setValue(z)
        self.img_row_line.show()

    def update_volume(self):
        """Volume at the q-range: read when the Q slider is released."""
        data = self.dataset
        if data is None:
            return
        x_idx, radius = self.slider_x.value(), self.radius()
        self.set_span(x_idx, data.x_step)
        if self.slider_x.is_dragging():
            return
        key = (x_idx, radius)
        if key != self.volume_key:
            loading = not data.has_volume(*key)
            if loading:
                self.set_status(
                    f"Loading the volume at {self.x_name()} index {x_idx} "
                    f"({util.format_bytes(data.volume_nbytes(radius))})...")
                QApplication.processEvents()
            self.volume = data.volume(*key)
            self.volume_key = key
            self.draw_volume()
            if loading:
                self.set_status(f"Volume at {self.x_name()} = "
                                f"{x_idx * data.x_step:.4g} loaded.")
        self.update_slice()

    update_qslice = update_volume

    def update_slice(self):
        """2D slice of the volume, with the crosshair of the indices."""
        data, vol = self.dataset, self.volume
        if data is None or vol is None:
            return
        row, col, z = self.indices()
        (name0, pos0), (name1, pos1) = data.motors
        mode = self.combo_slice_dir.currentIndex()
        z_label = "Z (strip index)"
        if mode == 0:
            mat, xlabel, ylabel = vol[:, row, :].T, z_label, name1
            hval, vval, fixed = col, z, f"{name0} = {pos0[row]:.6g}"
        elif mode == 1:
            mat, xlabel, ylabel = vol[:, :, col].T, z_label, name0
            hval, vval, fixed = row, z, f"{name1} = {pos1[col]:.6g}"
        else:
            mat, xlabel, ylabel = vol[z], f"{name1} (index)", name0
            hval, vval, fixed = row, col, f"Z = {z}"
        self.qslice_raw = mat
        self.show_image(self.p_slice, mat)
        self.set_label(self.p_slice, "bottom", xlabel)
        self.set_label(self.p_slice, "left", f"{ylabel} (index)")
        self.set_title(self.p_slice, f"Mean around {self.x_name()}="
                                     f"{self.volume_key[0] * data.x_step:.4g}"
                                     f" | {fixed}")
        self.slice_hline.setValue(hval)
        self.slice_vline.setValue(vval)
        self.slice_hline.show()
        self.slice_vline.show()
        self.apply_slice_aspect()
        self.update_gl_plane()

    def pixel_size(self):
        sizes = [self.get_value(edit, f"Pixel size {name}")
                 for edit, name in zip(self.edit_pixel, "xyz")]
        if min(sizes) <= 0:
            raise ValueError("Pixel sizes must be positive!")
        return sizes

    def gl_box(self):
        """Size of the 3D box with the real ratio; the longest side is
        the one of util.GL_BOX."""
        data = self.dataset
        if data is None:
            return util.GL_BOX
        px, py, pz = self.pixel_size()
        dims = np.array([data.ncol * px, data.nrow * py, data.height * pz])
        return tuple(dims / dims.max() * max(util.GL_BOX))

    def reset_gl_camera(self):
        size_x, size_y, size_z = self.gl_box()
        self.gl_view.setCameraPosition(
            pos=ren.QVector3D(size_x / 2, size_y / 2, size_z / 2),
            distance=1.6 * max(size_x, size_y, size_z), elevation=25,
            azimuth=-60)

    def apply_slice_aspect(self):
        """Real ratio of the 2D slice with the 'equal' aspect."""
        if ren.display["aspect"] != "equal" or self.volume is None:
            return
        px, py, pz = self.pixel_size()
        mode = self.combo_slice_dir.currentIndex()
        # Image x / y: Z / columns, Z / rows, columns / rows
        ratio = (pz / px, pz / py, px / py)[mode]
        self.p_slice.getViewBox().setAspectLocked(True, ratio)

    @ren.safe_call
    def apply_pixel_size(self):
        self.pixel_size()
        if self.dataset is None or self.gl_view is None:
            self.update_slice()
            return
        self.draw_gl_frame()
        self.draw_volume()
        self.update_slice()
        self.reset_gl_camera()

    def thresholds(self):
        tmin = self.get_value(self.edit_tmin, "Threshold min %")
        tmax = self.get_value(self.edit_tmax, "Threshold max %")
        if not 0 <= tmin < tmax <= 100:
            raise ValueError("Thresholds must satisfy 0 <= Min < Max <= 100!")
        return tmin, tmax

    @ren.safe_call
    def apply_thresholds(self):
        self.draw_volume()

    def apply_display_settings(self, auto_range=False):
        super().apply_display_settings(auto_range)
        self.apply_slice_aspect()
        self.draw_volume()

    def draw_gl_frame(self):
        """Box, axis names and the marker of the slice plane."""
        view = self.gl_view
        view.clear()
        self.gl_volume = None
        size_x, size_y, size_z = self.gl_box()
        fg = ren.pg.mkColor(ren.sel_theme["PG_FG"])
        box = ren.gl.GLBoxItem()
        box.setSize(size_x, size_y, size_z)
        box.setColor(fg)
        view.addItem(box)
        (name0, _), (name1, _) = self.dataset.motors
        font = ren.QFont(ren.QApplication.font())
        font.setPointSize(max(font.pointSize() - 2, 1))
        for pos, text in (((size_x / 2, -0.08 * size_y, 0), name1),
                          ((-0.08 * size_x, size_y / 2, 0), name0),
                          ((0, 0, size_z * 1.04), "Z (strip 0 on top)")):
            view.addItem(ren.gl.GLTextItem(pos=pos, text=str(text),
                                           color=fg, font=font))
        self.gl_plane = ren.gl.GLLinePlotItem(
            color=GLWaterfallMixin.gl_color(util.COLORS["yellow"]),
            width=2.0, antialias=True, mode="line_strip")
        view.addItem(self.gl_plane)

    def draw_volume(self):
        """Volume with the colormap and contrast of the images; alpha grows
        with the intensity, voxels outside the thresholds are hidden."""
        vol = self.volume
        if vol is None:
            return
        if self.p_proj is not None:
            self.show_image(self.p_proj, vol.mean(axis=0))
            return
        step = max(1, int(np.ceil((vol.size / util.GL_MAX_VOXELS) ** (1 / 3))))
        sub = vol[::step, ::step, ::step]
        data = util.log_scale(sub) if ren.display["scale"] == "log" else sub
        tmin, tmax = self.thresholds()
        low, high = np.percentile(data, (tmin, tmax))
        vmin, vmax = util.get_levels(data, ren.display["pmin"],
                                     ren.display["pmax"])
        norm = np.clip((data - vmin) / max(vmax - vmin, 1e-12), 0, 1)
        rgba = ren.get_colormap(ren.display["cmap"]).map(
            norm.ravel(), mode="byte").reshape(norm.shape + (4,))
        alpha = norm * (util.GL_VOLUME_ALPHA * 255)
        alpha[(data < low) | (data > high)] = 0
        rgba[..., 3] = alpha.astype(np.ubyte)
        # GL axes: x = column, y = row, z = strip (strip 0 on top)
        rgba = np.ascontiguousarray(rgba[::-1].transpose(2, 1, 0, 3))
        if self.gl_volume is None:
            self.gl_volume = ren.gl.GLVolumeItem(rgba, sliceDensity=1,
                                                 smooth=True,
                                                 glOptions="translucent")
            self.gl_view.addItem(self.gl_volume)
        else:
            self.gl_volume.setData(rgba)
        size_x, size_y, size_z = self.gl_box()
        self.gl_volume.resetTransform()
        self.gl_volume.scale(size_x / rgba.shape[0], size_y / rgba.shape[1],
                             size_z / rgba.shape[2])
        self.update_gl_plane()

    def update_gl_plane(self):
        if self.gl_plane is None or self.dataset is None:
            return
        data = self.dataset
        row, col, z = self.indices()
        sx, sy, sz = self.gl_box()
        mode = self.combo_slice_dir.currentIndex()
        if mode == 0:
            y = (row + 0.5) / data.nrow * sy
            pts = [[0, y, 0], [sx, y, 0], [sx, y, sz], [0, y, sz]]
        elif mode == 1:
            x = (col + 0.5) / data.ncol * sx
            pts = [[x, 0, 0], [x, sy, 0], [x, sy, sz], [x, 0, sz]]
        else:
            h = (data.height - z - 0.5) / data.height * sz
            pts = [[0, 0, h], [sx, 0, h], [sx, sy, h], [0, sy, h]]
        self.gl_plane.setData(pos=np.array(pts + pts[:1], dtype=float))

    def on_click(self, plot, point):
        data = self.dataset
        if data is None:
            return
        x, y = int(round(point.x())), int(round(point.y()))
        if plot is self.p_img:
            self.slider_z.set_value(y)
            self.slider_x.set_value(data.x_index(point.x()))
        elif plot is self.p_prof:
            self.slider_x.set_value(data.x_index(point.x()))
        elif plot is self.p_slice:
            mode = self.combo_slice_dir.currentIndex()
            if mode == 0:
                self.slider_z.set_value(x, notify=False)
                self.slider_col.set_value(y)
            elif mode == 1:
                self.slider_z.set_value(x, notify=False)
                self.slider_row.set_value(y)
            else:
                self.slider_col.set_value(x, notify=False)
                self.slider_row.set_value(y)

    @ren.safe_call
    def browse_ds_output(self):
        start = self.edit_ds_output.text().strip() or \
            self.last_folder("output")
        path, _ = QFileDialog.getSaveFileName(
            self, "Downsampled output file", start,
            "HDF (*.hdf *.h5 *.hdf5)")
        if path:
            self.remember_folder("output", path)
            self.edit_ds_output.setText(path)

    @ren.safe_call
    def run_downsample(self):
        source = self.edit_path.text().strip()
        key = self.combo_hdf_key.currentText().strip()
        if not (os.path.isfile(source)
                and source.lower().endswith(util.HDF_EXT)) or not key:
            raise ValueError("Select a 4D HDF file and its HDF key as "
                             "input!")
        bins = tuple(spin.value() for spin in self.spin_bins)
        if bins == (1, 1, 1, 1):
            raise ValueError("Set a bin larger than 1!")
        chunks = util.parse_chunks(self.edit_ds_chunks.text())
        output = self.edit_ds_output.text().strip() or \
            util.downsample_output_path(source)
        out_key = util.check_output_key(self.edit_ds_key.text())
        if os.path.exists(output):
            answer = QMessageBox.question(self, "Overwrite",
                                          f"{output} exists. Overwrite?")
            if answer != QMessageBox.Yes:
                return
            data = self.dataset
            if data is not None and Path(data.source).resolve() == \
                    Path(output).resolve():
                data.close()
                self.dataset = None

        def task(cancel_event):
            return util.downsample_4d(source, key, output, out_key, bins,
                                      chunks, report=self.set_progress,
                                      cancelled=cancel_event.is_set)

        self.run_worker(task, lambda result: self.on_downsampled(
            result, output, out_key), "Downsampling", "Starting...")

    def on_downsampled(self, result, output, out_key):
        message, cancelled = result
        self.set_status(message)
        if cancelled or not self.check_ds_load.isChecked():
            return
        # The binned q-step is read from the output
        self.edit_step.clear()
        self.set_path(output)
        self.combo_hdf_key.setCurrentText(out_key)
        self.load_data()


# ---------------------------------------------------------------------------
# Application controller
# ---------------------------------------------------------------------------

class EdviewInteraction(QObject):
    """Primary controller of the EDView application."""

    def __init__(self, base_folder=None):
        super().__init__()
        self.shut_down = False
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.app.setApplicationName(util.APP_NAME)
        if os.path.exists(ren.ICON_PATH):
            self.app.setWindowIcon(QIcon(ren.ICON_PATH))
        self.app.aboutToQuit.connect(self.shutdown)
        self.app.setFont(ren.select_ui_font(point_size=util.FONT_SIZE))
        self.tooltip_wrapper = ren.ToolTipWrapper()
        self.app.installEventFilter(self.tooltip_wrapper)
        config = util.load_config() or {}
        theme = config.get("theme", ren.DEFAULT_THEME)
        if theme not in ren.THEMES:
            theme = ren.DEFAULT_THEME
        ren.apply_theme(self.app, theme)
        self.browser = BrowserTab(base_folder)
        self.correction = CorrectionTab()
        self.browser.add_to_correction = self.correction.add_inputs
        self.viewers = {"EDDT viewer": EddtViewer(),
                        "Multiple-lines viewer": MultiLinesViewer(),
                        "Multiple-scans viewer": MultiScansViewer(),
                        "2D-scan viewer": Scan2DViewer(),
                        "4D viewer": Volume4DViewer()}
        self.tabs = [self.browser, self.correction] + \
            list(self.viewers.values())
        pages = [(self.browser, "Data browser"),
                 (self.correction, "Data correction")] + \
            [(tab, name) for name, tab in self.viewers.items()]
        self.main_win = ren.MainWindow(pages)

        def view_in(name, path):
            viewer = self.viewers[name]
            self.main_win.tab_widget.setCurrentWidget(viewer)
            viewer.show_input(path)
            viewer.load_data()

        def metadata_to(name, path):
            viewer = self.viewers[name]
            self.main_win.tab_widget.setCurrentWidget(viewer)
            viewer.edit_metadata.setText(path)
            viewer.load_metadata()

        def view_scans(paths):
            viewer = self.viewers[scans_viewer]
            self.main_win.tab_widget.setCurrentWidget(viewer)
            viewer.view_scans(paths)

        scans_viewer = "Multiple-scans viewer"
        self.browser.scans_viewer = scans_viewer
        self.browser.view_scans = view_scans
        viewer_ndims = {name: viewer.hdf_ndims for name, viewer in
                        self.viewers.items()}
        self.browser.viewer_ndims = viewer_ndims
        for name, viewer in self.viewers.items():
            viewer.viewer_name = name
            viewer.viewer_ndims = viewer_ndims
            viewer.open_in = view_in
        self.browser.view_in = view_in
        self.browser.metadata_to = metadata_to
        self.main_win.combo_theme.blockSignals(True)
        self.main_win.combo_theme.setCurrentText(theme)
        self.main_win.combo_theme.blockSignals(False)
        self.main_win.combo_theme.currentTextChanged.connect(
            self.on_theme_changed)
        self.main_win.close_handler = self.on_close_requested
        win = self.main_win
        for combo in (win.combo_cmap, win.combo_scale, win.combo_aspect):
            combo.currentTextChanged.connect(
                lambda _t: self.on_display_changed())
        for edit in (win.edit_pmin, win.edit_pmax):
            edit.editingFinished.connect(self.on_contrast_entered)
        win.btn_reset.clicked.connect(self.reset_display)

    def on_display_changed(self):
        """Apply the colormap, scale and aspect to the images of all tabs."""
        win = self.main_win
        aspect = win.combo_aspect.currentText()
        aspect_changed = aspect != ren.display["aspect"]
        ren.display.update(cmap=win.combo_cmap.currentText(),
                           scale=win.combo_scale.currentText(),
                           aspect=aspect)
        self.redraw_images(auto_range=aspect_changed)

    def on_contrast_entered(self):
        win = self.main_win
        try:
            pmin = util.to_number(win.edit_pmin.text(), "Min %")
            pmax = util.to_number(win.edit_pmax.text(), "Max %")
            if not 0 <= pmin < pmax <= 100:
                raise ValueError("Percentiles must satisfy 0 <= Min < Max "
                                 "<= 100!")
        except ValueError as error:
            QMessageBox.critical(win, "Error", str(error))
            win.edit_pmin.setText(f"{ren.display['pmin']:g}")
            win.edit_pmax.setText(f"{ren.display['pmax']:g}")
            return
        if (pmin, pmax) == (ren.display["pmin"], ren.display["pmax"]):
            return
        ren.display.update(pmin=pmin, pmax=pmax)
        self.redraw_images()

    def redraw_images(self, auto_range=False):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for tab in self.tabs:
                try:
                    tab.apply_display_settings(auto_range)
                except Exception as error:
                    traceback.print_exc()
                    tab.set_status(f"Error: {error}")
        finally:
            QApplication.restoreOverrideCursor()

    def reset_display(self):
        """Default contrast, and full view of the plots of the current
        tab."""
        win = self.main_win
        pmin, pmax = (float(v) for v in util.IMAGE_PERCENTILES)
        win.edit_pmin.setText(f"{pmin:g}")
        win.edit_pmax.setText(f"{pmax:g}")
        if (pmin, pmax) != (ren.display["pmin"], ren.display["pmax"]):
            ren.display.update(pmin=pmin, pmax=pmax)
            self.redraw_images()
        for plot in getattr(win.tab_widget.currentWidget(), "plots", []):
            plot.enableAutoRange()

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
        self.shutdown()
        return True

    def shutdown(self):
        """Cancel the running calculations, wait for them to stop (each
        step must finish first) and close the open HDF files. Called on
        closing the window and on quitting (e.g. Ctrl+C)."""
        if self.shut_down:
            return
        self.shut_down = True
        busy = [tab for tab in self.tabs if tab.is_busy()]
        for tab in self.tabs:
            tab.closing = True
        for tab in busy:
            tab.cancel_event.set()
        for tab in busy:
            tab.worker.join(util.EXIT_WAIT_S / len(busy))
            if tab.worker.is_alive():
                print(f"Warning: a calculation of the {type(tab).__name__} "
                      f"didn't stop in time.", file=sys.stderr, flush=True)
        for viewer in self.viewers.values():
            for data in viewer.loaded_datasets():
                try:
                    data.close()
                except Exception:
                    traceback.print_exc()

    def run(self):
        self.main_win.show()
        self.main_win.center_on_screen()
        return self.app.exec()
