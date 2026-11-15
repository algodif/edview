"""
Rendering module for EDView.
Contains themes, widget and plot helpers, and the layouts of the tabs.
"""

import os
import platform
import functools
import traceback
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QRect, QRectF, Signal
from PySide6.QtGui import QFont, QColor, QPalette, QFontDatabase, QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                               QPushButton, QLineEdit, QComboBox, QCheckBox,
                               QSpinBox, QAbstractSpinBox, QGroupBox,
                               QTabWidget, QScrollArea, QFrame, QListWidget,
                               QPlainTextEdit, QStatusBar, QMessageBox,
                               QDialog, QSizePolicy, QStackedWidget,
                               QTreeWidget, QSplitter, QSlider)

import edview.lib.utilities as util

pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "assets", "edview_icon.png")

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
DARK = {
    "BG": "#1a1d23", "SURFACE": "#504d4d", "SURFACE2": "#2a2f3a",
    "BORDER": "#3a3f4e", "BORDER_HI": "#4d9fff", "ACCENT": "#4d9fff",
    "ACCENT_HOV": "#6ab0ff", "ACCENT_PRE": "#3a7fd4", "TEXT": "#d0d5e0",
    "TEXT_DIM": "#7a8099", "TEXT_HEAD": "#ffffff", "SEL_BG": "#2d4d7a",
    "SEL_FG": "#ffffff", "BTN_BG": "#2e3340", "BTN_HOV": "#374058",
    "BTN_PRE": "#1e2a40", "SCROLL": "#3a3f4e", "SCROLL_HOV": "#4d9fff",
    "STATUS_BG": "#181b20", "ALT_ROW": "#1e2230", "PG_BG": "#1e2128",
    "PG_FG": "#c8cdd6",
}
LIGHT = {
    "BG": "#f4f6f9", "SURFACE": "#c2c2c2", "SURFACE2": "#eef1f6",
    "BORDER": "#cdd2dc", "BORDER_HI": "#2563eb", "ACCENT": "#2563eb",
    "ACCENT_HOV": "#3b82f6", "ACCENT_PRE": "#1d4ed8", "TEXT": "#1e2333",
    "TEXT_DIM": "#6b7280", "TEXT_HEAD": "#0f172a", "SEL_BG": "#dbeafe",
    "SEL_FG": "#1e3a5f", "BTN_BG": "#ffffff", "BTN_HOV": "#eff6ff",
    "BTN_PRE": "#dbeafe", "SCROLL": "#c5cbd8", "SCROLL_HOV": "#2563eb",
    "STATUS_BG": "#e8eaf0", "ALT_ROW": "#f8fafc", "PG_BG": "w",
    "PG_FG": "k",
}
CLASSIC = {
    "BG": "#f3f3f3", "SURFACE": "#e0e0e0", "SURFACE2": "#ffffff",
    "BORDER": "#d1d1d1", "BORDER_HI": "#005fb8", "ACCENT": "#005fb8",
    "ACCENT_HOV": "#1975c5", "ACCENT_PRE": "#004c93", "TEXT": "#1a1a1a",
    "TEXT_DIM": "#5d5d5d", "TEXT_HEAD": "#000000", "SEL_BG": "#cce4f7",
    "SEL_FG": "#000000", "BTN_BG": "#fbfbfb", "BTN_HOV": "#f0f0f0",
    "BTN_PRE": "#e5e5e5", "SCROLL": "#8a8a8a", "SCROLL_HOV": "#5d5d5d",
    "STATUS_BG": "#eeeeee", "ALT_ROW": "#f9f9f9", "PG_BG": "w",
    "PG_FG": "k",
}
THEMES = {"Light": LIGHT, "Classic": CLASSIC, "Dark": DARK}
DEFAULT_THEME = "Classic"
sel_theme = dict(THEMES[DEFAULT_THEME])


def build_qss(c: dict) -> str:
    """Generate the application QSS from a colour-palette dict."""
    fs = util.FONT_SIZE
    return f"""
    QWidget {{
        font-size: {fs}px;
        color: {c['TEXT']};
        background-color: {c['BG']};
    }}
    QGroupBox {{
        border: 1px solid {c['BORDER']};
        border-radius: 3px;
        margin-top: 8px;
        padding: 0px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 5px;
        color: {c['TEXT_DIM']};
        font-size: {fs - 1}px;
        font-weight: 500;
    }}
    QFrame, QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QPushButton {{
        background-color: {c['BTN_BG']};
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        padding: 4px 6px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {c['BTN_HOV']};
        border: 1px solid {c['BORDER_HI']};
        color: {c['TEXT_HEAD']};
    }}
    QPushButton:pressed {{
        background-color: {c['BTN_PRE']};
    }}
    QPushButton:disabled {{
        background-color: {c['SURFACE']};
        color: {c['TEXT_DIM']};
    }}
    QPushButton#AccentOutlineBtn {{
        color: {c['ACCENT']};
        border: 1px solid {c['ACCENT']};
        font-weight: 600;
    }}
    QLineEdit, QSpinBox, QPlainTextEdit {{
        background-color: {c['SURFACE2']};
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        padding: 3px 6px;
        selection-background-color: {c['SEL_BG']};
        selection-color: {c['SEL_FG']};
    }}
    QLineEdit:focus, QSpinBox:focus {{
        border: 1px solid {c['ACCENT']};
    }}
    QLineEdit:disabled, QSpinBox:disabled {{
        background-color: {c['SURFACE']};
        color: {c['TEXT_DIM']};
    }}
    QComboBox {{
        background-color: {c['SURFACE2']};
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        padding: 3px 5px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['SURFACE2']};
        selection-background-color: {c['SEL_BG']};
        selection-color: {c['SEL_FG']};
    }}
    QListWidget, QTreeWidget {{
        background-color: {c['SURFACE2']};
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        outline: none;
    }}
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {c['SEL_BG']};
        color: {c['SEL_FG']};
    }}
    QListWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {c['BTN_HOV']};
    }}
    QCheckBox {{
        spacing: 6px;
        background-color: transparent;
    }}
    QLabel {{
        background-color: transparent;
    }}
    QLabel#HeaderPathLabel {{
        color: {c['ACCENT']};
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: 1px solid {c['BORDER']};
        top: -1px;
    }}
    QTabBar::tab {{
        background: {c['SURFACE2']};
        color: {c['TEXT_DIM']};
        border: 1px solid {c['BORDER']};
        padding: 5px 14px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {c['BG']};
        color: {c['ACCENT']};
        font-weight: 600;
        border-bottom-color: {c['BG']};
    }}
    QTabBar::tab:hover {{
        background: {c['BTN_HOV']};
    }}
    QSplitter::handle {{
        background-color: {c['BORDER']};
    }}
    QStatusBar {{
        background-color: {c['STATUS_BG']};
        border-top: 1px solid {c['BORDER']};
    }}
    QStatusBar::item {{
        border: none;
    }}
    QToolTip {{
        background-color: {c['SURFACE2']};
        color: {c['TEXT']};
        border: 1px solid {c['BORDER_HI']};
        padding: 1px 4px;
    }}
"""


def select_ui_font(point_size=13, weight=QFont.Normal):
    """Choose a UI font based on OS, using a priority list."""
    sysname = platform.system()
    families = set(QFontDatabase.families())
    if sysname == "Linux":
        priority = ["Liberation Sans", "Cantarell", "Ubuntu", "Noto Sans",
                    "DejaVu Sans"]
    elif sysname == "Windows":
        priority = ["Segoe UI", "Calibri", "Arial", "Tahoma"]
    elif sysname == "Darwin":
        priority = [".SF NS Text", "Helvetica Neue", "Helvetica", "Arial"]
    else:
        priority = ["Noto Sans", "DejaVu Sans", "Arial"]
    chosen = next((name for name in priority if name in families), None)
    font = QFont(chosen) if chosen else QFont()
    font.setPointSize(point_size)
    font.setWeight(weight)
    return font


def calculate_geometry(ratio=0.8):
    """Window size and position centered on the primary screen."""
    screen = QApplication.primaryScreen().availableGeometry()
    width = min(int(screen.width() * ratio), util.MAX_WIN_W)
    height = min(int(screen.height() * ratio), util.MAX_WIN_H)
    return QRect(screen.center().x() - width // 2,
                 screen.center().y() - height // 2, width, height)


def apply_theme(app, theme_name="Classic"):
    """Apply a named theme to the application."""
    global sel_theme
    sel_theme = dict(THEMES.get(theme_name, CLASSIC))
    pg.setConfigOptions(background=sel_theme["PG_BG"],
                        foreground=sel_theme["PG_FG"])
    if app.style().name().lower() != "fusion":
        app.setStyle("Fusion")
    pal = QPalette()
    for role, key in ((QPalette.Window, "BG"), (QPalette.WindowText, "TEXT"),
                      (QPalette.Base, "SURFACE2"),
                      (QPalette.AlternateBase, "ALT_ROW"),
                      (QPalette.Text, "TEXT"), (QPalette.Button, "BTN_BG"),
                      (QPalette.ButtonText, "TEXT"),
                      (QPalette.Highlight, "SEL_BG"),
                      (QPalette.HighlightedText, "SEL_FG"),
                      (QPalette.ToolTipBase, "SURFACE2"),
                      (QPalette.ToolTipText, "TEXT")):
        pal.setColor(role, QColor(sel_theme[key]))
    for role in (QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(sel_theme["TEXT_DIM"]))
    app.setPalette(pal)
    app.setStyleSheet(build_qss(sel_theme))


def get_colormap(name):
    if name == "gray":
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    return pg.colormap.get(name)


def safe_call(func):
    """Show errors in a message box and display a busy cursor."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return func(self, *args, **kwargs)
        except Exception as error:
            traceback.print_exc()
            self.set_status(f"Error: {error}")
            QApplication.changeOverrideCursor(Qt.ArrowCursor)
            QMessageBox.critical(self, "Error", str(error))
        finally:
            QApplication.restoreOverrideCursor()
    return wrapper


class WorkerDialog(QDialog):
    """Modal dialog shown while a calculation runs in a thread."""

    def __init__(self, parent, title, message, on_cancel):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.on_cancel = on_cancel
        layout = QVBoxLayout(self)
        layout.setContentsMargins(util.UI_MARGIN_L, util.UI_MARGIN_L,
                                  util.UI_MARGIN_L, util.UI_MARGIN_L)
        head = QLabel(f"{title}. Please wait...")
        font = QFont(head.font())
        font.setBold(True)
        head.setFont(font)
        self.lbl_message = QLabel(message)
        self.lbl_message.setWordWrap(True)
        self.lbl_message.setMinimumWidth(380)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(util.BTN_H)
        self.btn_cancel.clicked.connect(lambda: self.on_cancel())
        layout.addWidget(head)
        layout.addWidget(self.lbl_message)
        layout.addWidget(self.btn_cancel, 0, Qt.AlignHCenter)
        self.finished_task = False

    def set_message(self, text):
        if text != self.lbl_message.text():
            self.lbl_message.setText(text)

    def reject(self):
        self.on_cancel()

    def closeEvent(self, event):
        if self.finished_task:
            event.accept()
            return
        event.ignore()
        self.on_cancel()


class SliderRow(QWidget):
    """Label, slider and spin box kept in sync; emits valueChanged."""

    valueChanged = Signal(int)

    def __init__(self, label, tip=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(util.UI_SPACING_L)
        self.label = QLabel(label)
        self.label.setMinimumWidth(util.FORM_LABEL_W + 40)
        self.slider = QSlider(Qt.Horizontal)
        self.spin = QSpinBox()
        self.spin.setFixedWidth(util.SMALL_ENTRY_W + 20)
        self.spin.setKeyboardTracking(False)
        for widget in (self.label, self.slider, self.spin):
            if tip:
                widget.setToolTip(tip)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.set_range(0)

    def set_range(self, maximum):
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setRange(0, max(0, maximum))
            widget.blockSignals(False)
        self.setEnabled(maximum > 0)

    def value(self):
        return self.spin.value()

    def set_value(self, value, notify=True):
        value = int(np.clip(value, 0, self.spin.maximum()))
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        if notify:
            self.valueChanged.emit(value)

    def _on_slider(self, value):
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _on_spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)


# ---------------------------------------------------------------------------
# Base tab: widget and plot helpers
# ---------------------------------------------------------------------------

class BaseTab(QWidget):
    """Widget and plot helpers shared by the tabs."""

    def __init__(self):
        super().__init__()
        self.status_func = None
        self.cursor_func = None
        self.glws = []
        self.plots = []
        self.image_items = []
        self.worker = None
        self.dialog = None
        self.closing = False

    def set_status(self, text):
        if self.status_func is not None:
            self.status_func(" ".join(str(text).split()))

    @staticmethod
    def make_label(text, tip=None):
        label = QLabel(text)
        if tip:
            label.setToolTip(tip)
        return label

    @staticmethod
    def make_button(text, callback, tip=None):
        button = QPushButton(text)
        button.setFixedHeight(util.BTN_H)
        button.clicked.connect(lambda: callback())
        if tip:
            button.setToolTip(tip)
        return button

    @staticmethod
    def make_line_edit(default="", width=None, tip=None):
        edit = QLineEdit(str(default))
        if width is not None:
            edit.setFixedWidth(width)
        if tip:
            edit.setToolTip(tip)
        return edit

    @staticmethod
    def make_combo(items, current=None, tip=None, editable=False):
        combo = QComboBox()
        combo.addItems([str(v) for v in items])
        combo.setEditable(editable)
        if current is not None:
            combo.setCurrentText(str(current))
        if tip:
            combo.setToolTip(tip)
        return combo

    @staticmethod
    def make_grid(parent=None):
        grid = QGridLayout(parent) if parent is not None else QGridLayout()
        grid.setContentsMargins(util.UI_MARGIN_M, util.UI_MARGIN_L,
                                util.UI_MARGIN_M, util.UI_MARGIN_M)
        grid.setHorizontalSpacing(util.UI_SPACING_L)
        grid.setVerticalSpacing(util.UI_SPACING_ML)
        return grid

    def make_group(self, panel, title):
        group = QGroupBox(title)
        grid = self.make_grid(group)
        grid.setColumnMinimumWidth(0, util.FORM_LABEL_W)
        grid.setColumnStretch(1, 1)
        panel.addWidget(group)
        return grid

    def add_entry(self, grid, row, label, default, tip=None):
        grid.addWidget(self.make_label(label, tip), row, 0)
        edit = self.make_line_edit(default, tip=tip)
        grid.addWidget(edit, row, 1)
        return edit

    def add_combo(self, grid, row, label, items, current=None, tip=None,
                  editable=False):
        grid.addWidget(self.make_label(label, tip), row, 0)
        combo = self.make_combo(items, current, tip, editable)
        grid.addWidget(combo, row, 1)
        return combo

    @staticmethod
    def add_check(grid, row, label, default, tip=None):
        check = QCheckBox(label)
        check.setChecked(bool(default))
        if tip:
            check.setToolTip(tip)
        grid.addWidget(check, row, 0, 1, 2)
        return check

    def add_buttons(self, grid, row, buttons):
        """A row of equally sized buttons spanning the two columns."""
        widget = QWidget()
        hbox = QHBoxLayout(widget)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(util.UI_SPACING_ML)
        result = []
        for text, callback in buttons:
            button = self.make_button(text, callback)
            hbox.addWidget(button, 1)
            result.append(button)
        grid.addWidget(widget, row, 0, 1, 2)
        return result

    @staticmethod
    def make_page(stack):
        """Page of a stacked widget with a two-column grid, rows at the
        top."""
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(util.UI_SPACING_L)
        grid.setVerticalSpacing(util.UI_SPACING_ML)
        grid.setColumnMinimumWidth(0, util.FORM_LABEL_W)
        grid.setColumnStretch(1, 1)
        grid.setAlignment(Qt.AlignTop)
        stack.addWidget(page)
        return grid

    def make_left_panel(self, layout, width=util.LEFT_PANEL_W):
        """Scrollable panel of parameter groups on the left side."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(width + 24)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        panel = QVBoxLayout(content)
        panel.setContentsMargins(0, 0, util.UI_MARGIN_M, 0)
        panel.setSpacing(util.UI_SPACING_L)
        scroll.setWidget(content)
        layout.addWidget(scroll, 0)
        return panel

    @staticmethod
    def get_value(widget, name, dtype=float, allow_none=False):
        text = widget.currentText() if isinstance(widget, QComboBox) \
            else widget.text()
        return util.to_number(text, name, dtype, allow_none)

    def unify_widget_heights(self):
        for cls in (QLineEdit, QComboBox, QSpinBox, QCheckBox):
            for widget in self.findChildren(cls):
                if isinstance(widget.parent(), (QComboBox,
                                                QAbstractSpinBox)):
                    continue
                widget.setFixedHeight(util.BTN_H)

    # Plot helpers
    def make_plot_widget(self):
        glw = pg.GraphicsLayoutWidget()
        glw.setBackground(sel_theme["PG_BG"])
        glw.ci.setSpacing(util.UI_SPACING_L)
        glw.scene().sigMouseMoved.connect(
            lambda pos, scene=glw.scene(): self.on_plot_hover(scene, pos))
        self.glws.append(glw)
        return glw

    def add_plot(self, glw, row, col, title="", xlabel="", ylabel="",
                 image=False, colspan=1):
        plot = glw.addPlot(row=row, col=col, colspan=colspan)
        for column in range(col, col + colspan):
            glw.ci.layout.setColumnStretchFactor(column, 1)
        plot.setPreferredWidth(1)
        font = QFont(QApplication.font())
        font.setPointSize(util.PLOT_FONT_SIZE)
        for name in ("left", "bottom"):
            plot.getAxis(name).setTickFont(font)
        if image:
            plot.invertY(True)
            item = pg.ImageItem()
            plot.addItem(item)
            self.image_items.append(item)
            plot.image_item = item
        else:
            plot.showGrid(x=True, y=True, alpha=0.3)
        self.set_title(plot, title)
        if xlabel:
            self.set_label(plot, "bottom", xlabel)
        if ylabel:
            self.set_label(plot, "left", ylabel)
        self.plots.append(plot)
        return plot

    @staticmethod
    def set_title(plot, text):
        plot.setTitle(text, size=f"{util.PLOT_TITLE_SIZE}pt",
                      color=sel_theme["PG_FG"])

    @staticmethod
    def set_label(plot, axis, text):
        plot.setLabel(axis, text, color=sel_theme["PG_FG"],
                      **{"font-size": f"{util.PLOT_FONT_SIZE}pt"})

    def show_image(self, plot, mat, rect=None, cmap="inferno", log=True,
                   auto_range=False):
        """Show a 2D array; rect = (x0, y0, width, height) in data units."""
        data = util.log_scale(mat) if log else np.asarray(mat, np.float32)
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        item = plot.image_item
        item.setImage(data, levels=(vmin, vmax), autoLevels=False)
        item.setLookupTable(get_colormap(cmap).getLookupTable(nPts=256))
        height, width = data.shape
        if rect is None:
            rect = (-0.5, -0.5, width, height)
        item.setRect(QRectF(*rect))
        item.raw_data = mat
        if auto_range:
            plot.enableAutoRange()

    @staticmethod
    def make_line(plot, angle, color, width=1.0):
        line = pg.InfiniteLine(angle=angle, movable=False,
                               pen=pg.mkPen(color, width=width))
        line.setZValue(10)
        plot.addItem(line)
        line.hide()
        return line

    @staticmethod
    def make_span(plot, color=util.COLORS["yellow"]):
        qcolor = QColor(color)
        qcolor.setAlpha(70)
        span = pg.LinearRegionItem(movable=False, brush=qcolor,
                                   pen=pg.mkPen(None))
        span.setZValue(9)
        plot.addItem(span)
        span.hide()
        return span

    def connect_click(self, glw, plots, callback):
        """Call callback(plot, point, event) on a left click (not drag)."""
        def on_click(event):
            if event.button() != Qt.LeftButton:
                return
            pos = event.scenePos()
            for plot in plots:
                if plot.vb.sceneBoundingRect().contains(pos):
                    try:
                        callback(plot, plot.vb.mapSceneToView(pos))
                    except Exception as error:
                        traceback.print_exc()
                        self.set_status(f"Error: {error}")
                    return

        glw.scene().sigMouseClicked.connect(on_click)

    def on_plot_hover(self, scene, pos):
        if self.cursor_func is None:
            return
        for plot in self.plots:
            if plot.scene() is not scene \
                    or not plot.vb.sceneBoundingRect().contains(pos):
                continue
            point = plot.vb.mapSceneToView(pos)
            text = f"x={point.x():.5g}  y={point.y():.5g}"
            item = getattr(plot, "image_item", None)
            mat = getattr(item, "raw_data", None) if item else None
            if mat is not None:
                pixel = item.mapFromScene(pos)
                row, col = int(np.floor(pixel.y())), int(np.floor(pixel.x()))
                if 0 <= row < mat.shape[0] and 0 <= col < mat.shape[1]:
                    text += f"  value={float(mat[row, col]):.5g}"
            self.cursor_func(text)
            return
        self.cursor_func("")

    def refresh_plot_theme(self):
        fg = sel_theme["PG_FG"]
        for glw in self.glws:
            glw.setBackground(sel_theme["PG_BG"])
        for plot in self.plots:
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(fg)
                axis.setTextPen(fg)
                if axis.labelText:
                    self.set_label(plot, name, axis.labelText)
            self.set_title(plot, plot.titleLabel.text)


# ---------------------------------------------------------------------------
# Data browser
# ---------------------------------------------------------------------------

class BrowserTabUI(BaseTab):
    """Folder tree, file list and preview of the selected file."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(util.UI_MARGIN_M, util.UI_MARGIN_M,
                                  util.UI_MARGIN_M, util.UI_MARGIN_M)
        layout.setSpacing(util.UI_SPACING_L)

        header = QHBoxLayout()
        title = QLabel("Base folder:")
        self.lbl_base_folder = QLabel("")
        self.lbl_base_folder.setObjectName("HeaderPathLabel")
        self.lbl_base_folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        btn_select = self.make_button("Select folder",
                                      self.select_base_folder)
        btn_select.setObjectName("AccentOutlineBtn")
        btn_select.setFixedWidth(130)
        btn_refresh = self.make_button("Refresh", self.refresh_tree,
                                       "Reload the folder tree.")
        btn_refresh.setFixedWidth(90)
        header.addWidget(title)
        header.addWidget(self.lbl_base_folder, 1)
        header.addWidget(btn_select)
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(util.TREE_MIN_W)
        self.tree.itemExpanded.connect(self.on_tree_expand)
        self.tree.itemSelectionChanged.connect(self.on_folder_select)
        splitter.addWidget(self.tree)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(util.UI_SPACING_M)
        self.list_files = QListWidget()
        self.list_files.itemSelectionChanged.connect(
            self.on_file_select_change)
        self.list_files.itemDoubleClicked.connect(
            lambda _item: self.open_selected())
        self.list_files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_files.customContextMenuRequested.connect(
            self.on_file_context_menu)
        middle_layout.addWidget(self.list_files, 1)
        actions = QHBoxLayout()
        target_tip = ("Tab to send the selected file, or the selected folder "
                      "if no file is selected. Double-click a file to do the "
                      "same.")
        actions.addWidget(self.make_label("Open in", target_tip))
        self.combo_target = self.make_combo(
            list(util.VIEWER_TABS) + ["Data correction (input)",
                                      "Data conversion (source)"],
            tip=target_tip)
        actions.addWidget(self.combo_target, 1)
        btn_open = self.make_button("Open", self.open_selected, target_tip)
        btn_open.setObjectName("AccentOutlineBtn")
        btn_open.setFixedWidth(80)
        actions.addWidget(btn_open)
        middle_layout.addLayout(actions)
        splitter.addWidget(middle)

        # Preview: info, text, image, HDF
        self.preview = QStackedWidget()
        self.lbl_preview_info = QLabel("Select a file to preview it.")
        self.lbl_preview_info.setAlignment(Qt.AlignCenter)
        self.lbl_preview_info.setWordWrap(True)
        self.preview.addWidget(self.lbl_preview_info)
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.text_preview.setFont(mono)
        self.preview.addWidget(self.text_preview)
        image_page = QWidget()
        image_layout = QVBoxLayout(image_page)
        image_layout.setContentsMargins(0, 0, 0, 0)
        hdf_row = QHBoxLayout()
        self.lbl_hdf_key = QLabel("HDF key")
        self.combo_preview_key = QComboBox()
        self.combo_preview_key.currentTextChanged.connect(
            self.on_preview_key_changed)
        hdf_row.addWidget(self.lbl_hdf_key)
        hdf_row.addWidget(self.combo_preview_key, 1)
        image_layout.addLayout(hdf_row)
        self.text_hdf = QPlainTextEdit()
        self.text_hdf.setReadOnly(True)
        self.text_hdf.setFont(mono)
        self.text_hdf.setMaximumHeight(160)
        image_layout.addWidget(self.text_hdf)
        glw = self.make_plot_widget()
        self.p_preview = self.add_plot(glw, 0, 0, image=True)
        image_layout.addWidget(glw, 1)
        self.preview.addWidget(image_page)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 3)
        layout.addWidget(splitter, 1)
        self.unify_widget_heights()

    def show_preview_page(self, name):
        pages = {"info": 0, "text": 1, "image": 2}
        self.preview.setCurrentIndex(pages[name])

    def set_hdf_controls_visible(self, visible):
        for widget in (self.lbl_hdf_key, self.combo_preview_key,
                       self.text_hdf):
            widget.setVisible(visible)


# ---------------------------------------------------------------------------
# Data correction and conversion
# ---------------------------------------------------------------------------

class CorrectionTabUI(BaseTab):
    """Correct raw data with calibration coefficients; convert data."""

    def __init__(self):
        super().__init__()
        self.coefs = None
        self.items = []
        self.raw_image = None
        self.corrected = None
        self.corrected_raw = None
        self.step = None
        self.strip = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(util.UI_MARGIN_M, util.UI_MARGIN_M,
                                  util.UI_MARGIN_M, util.UI_MARGIN_M)
        tabs = QTabWidget()
        tabs.setFixedWidth(util.LEFT_PANEL_W + 40)
        self.side_tabs = tabs
        correction = QWidget()
        correction_layout = QHBoxLayout(correction)
        correction_layout.setContentsMargins(0, util.UI_MARGIN_S, 0, 0)
        panel = self.make_left_panel(correction_layout)
        self.build_input_group(panel)
        self.build_calibration_group(panel)
        self.build_background_group(panel)
        self.build_output_group(panel)
        panel.addStretch(1)
        tabs.addTab(correction, "Correction")
        conversion = QWidget()
        conversion_layout = QHBoxLayout(conversion)
        conversion_layout.setContentsMargins(0, util.UI_MARGIN_S, 0, 0)
        panel = self.make_left_panel(conversion_layout)
        self.build_conversion_group(panel)
        panel.addStretch(1)
        tabs.addTab(conversion, "Conversion")
        layout.addWidget(tabs, 0)

        glw = self.make_plot_widget()
        layout.addWidget(glw, 1)
        self.p_raw = self.add_plot(glw, 0, 0, "Raw image", "Pixel index",
                                   "Strip index", image=True)
        self.p_corr = self.add_plot(glw, 0, 1, "Corrected image", "",
                                    "Strip index", image=True)
        self.p_strip = self.add_plot(glw, 1, 0, "", "", "Intensity",
                                     colspan=2)
        self.raw_line = self.make_line(self.p_raw, 0, util.COLORS["lime"])
        self.corr_line = self.make_line(self.p_corr, 0, util.COLORS["lime"])
        self.connect_click(glw, [self.p_raw, self.p_corr], self.on_click)
        self.unify_widget_heights()

    def build_input_group(self, panel):
        grid = self.make_group(panel, "Input data")
        self.edit_input = self.add_entry(
            grid, 0, "Input", "",
            "A scan folder (with a 'tifs' sub-folder), a folder of tif "
            "files, files separated by ';', or an HDF file. With 'Scans', a "
            "base folder of 'scan_xxxxx' folders.")
        self.add_buttons(grid, 1, [("Browse files...", self.browse_files),
                                   ("Browse folder...", self.browse_folder)])
        self.combo_hdf_key = self.add_combo(
            grid, 2, "HDF key", [], tip="Dataset of HDF input files.",
            editable=True)
        self.edit_axis = self.add_entry(
            grid, 3, "Axis (3D HDF)", 0,
            "Axis along which images are stacked in 3D HDF datasets.")
        self.edit_scans = self.add_entry(
            grid, 4, "Scans", "",
            "Optional. Scan numbers in the base folder given as Input, "
            "e.g. '1-5, 8' for scan_00001 to scan_00005 and scan_00008. "
            "Empty: the Input is a single scan.")
        self.add_buttons(grid, 5, [("Load", self.load_input)])
        self.lbl_input_info = QLabel("No input loaded.")
        self.lbl_input_info.setWordWrap(True)
        grid.addWidget(self.lbl_input_info, 6, 0, 1, 2)

    def build_calibration_group(self, panel):
        grid = self.make_group(panel, "Calibration")
        self.edit_coef = QLineEdit("None")
        self.edit_coef.setReadOnly(True)
        grid.addWidget(self.edit_coef, 0, 0, 1, 2)
        self.add_buttons(grid, 1, [("Load coefficients...",
                                    self.load_coefficients)])
        self.combo_kind = self.add_combo(
            grid, 2, "Type", list(util.X_KINDS), "q",
            "Calibration type of the coefficients: q (1/\u00c5) or energy "
            "(keV).")
        step_tip = ("Step of the uniform grid of the corrected data:\n"
                    "- mean/median/max: of the pixel spacing of the map "
                    "computed from the coefficients.\n"
                    "- x-max: step = x-max / (width - 1).\n"
                    "- x-step: the given step.")
        self.combo_step = self.add_combo(grid, 3, "Output step",
                                         util.OUTPUT_STEP_OPTIONS, "mean",
                                         step_tip)
        self.edit_step = self.add_entry(grid, 4, "Value", "", step_tip)
        self.add_buttons(grid, 5, [("Update preview", self.update_preview)])

        def update_state(_text=None):
            self.edit_step.setEnabled(self.combo_step.currentText()
                                      in ("x-max", "x-step"))

        self.combo_step.currentTextChanged.connect(update_state)
        update_state()

    def build_background_group(self, panel):
        grid = self.make_group(panel, "Background removal")
        self.check_bck = self.add_check(
            grid, 0, "Remove background", True,
            "Remove the background of each strip of the corrected images. "
            "Images with and without background are saved.")
        self.combo_bck = self.add_combo(grid, 1, "Method",
                                        list(util.BCK_METHODS),
                                        util.DEFAULT_BCK_METHOD)
        self.stack_bck = QStackedWidget()
        self.bck_edits = {}
        for method, (_, specs) in util.BCK_METHODS.items():
            page_grid = self.make_page(self.stack_bck)
            self.bck_edits[method] = {
                key: self.add_entry(page_grid, row, label, default, tip)
                for row, (key, label, default, _, tip) in enumerate(specs)}
        grid.addWidget(self.stack_bck, 2, 0, 1, 2)
        self.btn_bck_apply = self.add_buttons(
            grid, 3, [("Apply to preview", self.apply_background)])[0]

        def update_state(_value=None):
            enabled = self.check_bck.isChecked()
            for widget in (self.combo_bck, self.stack_bck,
                           self.btn_bck_apply):
                widget.setEnabled(enabled)
            self.stack_bck.setCurrentIndex(self.combo_bck.currentIndex())

        self.check_bck.toggled.connect(update_state)
        self.combo_bck.currentIndexChanged.connect(update_state)
        update_state()

    def build_output_group(self, panel):
        grid = self.make_group(panel, "Output")
        self.edit_output = self.add_entry(
            grid, 0, "Folder", "",
            "Output base folder. Each scan is saved in a sub-folder named "
            "after the scan, with metadata.json (incl. the output step), "
            "the coefficients and the map of the raw pixels.")
        self.add_buttons(grid, 1, [("Browse...", self.browse_output),
                                   ("Make folder", self.make_output_folder)])
        format_tip = ("- tif: 'tifs_with_bgr' and 'tifs_without_bgr' "
                      "folders.\n"
                      "- hdf: '<scan>.hdf' with datasets "
                      f"'{util.HDF_DATA_KEY}' and "
                      f"'{util.HDF_DATA_NO_BGR_KEY}'.\n"
                      "- xy/csv: one file per strip.")
        self.combo_format = self.add_combo(grid, 2, "Format",
                                           util.OUTPUT_FORMATS, "tif",
                                           format_tip)
        self.stack_format = QStackedWidget()
        page = self.make_page(self.stack_format)
        page.addWidget(QLabel("Saved in 'tifs_with_bgr' and "
                              "'tifs_without_bgr'."), 0, 0, 1, 2)
        page = self.make_page(self.stack_format)
        page.addWidget(QLabel("Saved in '<scan>.hdf'."), 0, 0, 1, 2)
        page = self.make_page(self.stack_format)
        data_tip = ("- raw: measured intensities with the calibrated x-value "
                    "of each pixel (non-uniform grid). No background "
                    "removal.\n"
                    "- corrected: intensities on the uniform grid, without "
                    "background if selected.")
        self.combo_xy_data = self.add_combo(page, 0, "Data",
                                            ["corrected", "raw"], tip=data_tip)
        self.combo_xy_type = self.add_combo(page, 1, "File type",
                                            ["xy", "csv"])
        self.combo_zip = self.add_combo(
            page, 2, "Packing", ["zip", "unzip"],
            tip="One zip file or one sub-folder per image.")
        self.edit_xy_folder = self.add_entry(page, 3, "Sub-folder", "xy",
                                             "Sub-folder of the scan output.")
        grid.addWidget(self.stack_format, 3, 0, 1, 2)
        self.combo_format.currentIndexChanged.connect(
            self.stack_format.setCurrentIndex)
        self.edit_start = self.add_entry(
            grid, 4, "Start index", 0,
            "Index of the first image of each scan to correct.")
        self.edit_stop = self.add_entry(
            grid, 5, "Stop index", "",
            "Index of the last image of each scan to correct (inclusive). "
            "Empty: the last image.")
        button = self.add_buttons(grid, 6, [("Correct and save",
                                             self.run_correction)])[0]
        button.setObjectName("AccentOutlineBtn")

    def build_conversion_group(self, panel):
        grid = self.make_group(panel, "Data conversion")
        self.combo_conversion = self.add_combo(grid, 0, "Conversion",
                                               list(util.CONVERSIONS))
        self.edit_conv_source = self.add_entry(
            grid, 1, "Source", "", "Folder of tif files or HDF file.")
        self.add_buttons(grid, 2, [("Browse file...",
                                    self.browse_conv_file),
                                   ("Browse folder...",
                                    self.browse_conv_folder)])
        self.combo_conv_key = self.add_combo(grid, 3, "HDF key", [],
                                             tip="Dataset of the HDF source.",
                                             editable=True)
        self.combo_conv_kind = self.add_combo(grid, 4, "X type",
                                              list(util.X_KINDS), "q")
        self.edit_conv_step = self.add_entry(
            grid, 5, "X step", "",
            "Step of the x-values (q or energy). Empty: read from the "
            "metadata or the HDF file.")
        self.edit_conv_output = self.add_entry(
            grid, 6, "Output", "",
            "Output HDF file (to HDF) or folder (to tif or xy/csv).")
        self.add_buttons(grid, 7, [("Browse output...",
                                    self.browse_conv_output)])
        self.combo_conv_xy_type = self.add_combo(grid, 8, "File type",
                                                 ["xy", "csv"])
        self.combo_conv_zip = self.add_combo(grid, 9, "Packing",
                                             ["zip", "unzip"])
        button = self.add_buttons(grid, 10, [("Convert",
                                              self.run_conversion)])[0]
        button.setObjectName("AccentOutlineBtn")
        self.lbl_conv_info = QLabel("")
        self.lbl_conv_info.setWordWrap(True)
        grid.addWidget(self.lbl_conv_info, 11, 0, 1, 2)

        def update_state(_text=None):
            is_xy = util.CONVERSIONS[
                self.combo_conversion.currentText()] == "to_xy"
            self.combo_conv_xy_type.setEnabled(is_xy)
            self.combo_conv_zip.setEnabled(is_xy)

        self.combo_conversion.currentTextChanged.connect(update_state)
        update_state()


# ---------------------------------------------------------------------------
# Viewers
# ---------------------------------------------------------------------------

class BaseViewerUI(BaseTab):
    """Input group, plot area and display controls of the viewers."""

    multi_input = False

    def __init__(self):
        super().__init__()
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(util.UI_MARGIN_M,
                                            util.UI_MARGIN_M,
                                            util.UI_MARGIN_M,
                                            util.UI_MARGIN_M)
        self.layout_main.setSpacing(util.UI_SPACING_M)
        self.build_input_group()

    def build_input_group(self):
        group = QGroupBox("Data")
        grid = self.make_grid(group)
        path_tip = ("Folder of processed tif files (e.g. 'tifs_with_bgr') or "
                    "an HDF file.")
        if self.multi_input:
            path_tip += " Several scans separated by ';'."
        grid.addWidget(self.make_label("Path", path_tip), 0, 0)
        self.edit_path = self.make_line_edit(tip=path_tip)
        grid.addWidget(self.edit_path, 0, 1, 1, 5)
        buttons = [("Folder...", self.browse_folder),
                   ("File...", self.browse_file)]
        if self.multi_input:
            buttons.append(("Clear", lambda: self.edit_path.setText("")))
        for i, (text, callback) in enumerate(buttons):
            button = self.make_button(text, callback)
            button.setFixedWidth(90)
            grid.addWidget(button, 0, 6 + i)
        key_tip = "Dataset of the HDF file."
        grid.addWidget(self.make_label("HDF key", key_tip), 1, 0)
        self.combo_hdf_key = self.make_combo([], tip=key_tip, editable=True)
        grid.addWidget(self.combo_hdf_key, 1, 1)
        kind_tip = "Unit of the x-axis."
        grid.addWidget(self.make_label("X type", kind_tip), 1, 2)
        self.combo_kind = self.make_combo(list(util.X_KINDS), "q", kind_tip)
        grid.addWidget(self.combo_kind, 1, 3)
        step_tip = ("Step of the x-values (q or energy) of the pixels. "
                    "Empty: read from the metadata or the HDF file.")
        grid.addWidget(self.make_label("X step", step_tip), 1, 4)
        self.edit_step = self.make_line_edit("", util.SMALL_ENTRY_W + 30,
                                             step_tip)
        grid.addWidget(self.edit_step, 1, 5)
        meta_tip = ("JSON metadata with the motor positions and the x-step. "
                    "Empty: the first JSON file in the data folder or its "
                    "parent.")
        grid.addWidget(self.make_label("Metadata", meta_tip), 2, 0)
        self.edit_metadata = self.make_line_edit(tip=meta_tip)
        grid.addWidget(self.edit_metadata, 2, 1, 1, 5)
        button = self.make_button("JSON...", self.browse_metadata)
        button.setFixedWidth(90)
        grid.addWidget(button, 2, 6)
        load = self.make_button("Load", self.load_data)
        load.setObjectName("AccentOutlineBtn")
        load.setFixedWidth(90)
        grid.addWidget(load, 1, 6)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(5, 1)
        self.layout_main.addWidget(group, 0)

    def build_display_controls(self, slider_rows):
        """Slider rows on the left, display options on the right."""
        controls = QHBoxLayout()
        sliders = QVBoxLayout()
        for row in slider_rows:
            sliders.addWidget(row)
        controls.addLayout(sliders, 1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(util.UI_SPACING_L)
        radius_tip = ("Half-width (pixels) of the x-range averaged for the "
                      "q-slice image.")
        grid.addWidget(self.make_label("Radius", radius_tip), 0, 0)
        self.combo_radius = self.make_combo(util.RADII, util.DEFAULT_RADIUS,
                                            radius_tip)
        self.combo_radius.currentTextChanged.connect(
            lambda _t: self.update_qslice())
        grid.addWidget(self.combo_radius, 0, 1)
        grid.addWidget(self.make_label("Colormap"), 0, 2)
        self.combo_cmap = self.make_combo(util.IMAGE_CMAPS, "inferno")
        self.combo_cmap.currentTextChanged.connect(
            lambda _t: self.refresh_all())
        grid.addWidget(self.combo_cmap, 0, 3)
        self.check_log = QCheckBox("Log scale")
        self.check_log.setChecked(True)
        self.check_log.toggled.connect(lambda _c: self.refresh_all())
        grid.addWidget(self.check_log, 1, 0, 1, 2)
        btn_pattern = self.make_button(
            "Save pattern", self.save_pattern,
            "Save the displayed diffraction pattern(s) to a CSV file.")
        btn_slice = self.make_button(
            "Save q-slice", self.save_qslice,
            "Save the q-slice image to a TIF file.")
        grid.addWidget(btn_pattern, 1, 2)
        grid.addWidget(btn_slice, 1, 3)
        self.display_grid = grid
        controls.addLayout(grid, 0)
        self.layout_main.addLayout(controls, 0)

    def x_label(self):
        kind = self.combo_kind.currentText()
        label, unit, _ = util.X_KINDS[kind]
        return f"{label} ({unit})"


class EddtViewerUI(BaseViewerUI):
    """1D scan: image, q-slice across the scan, and the pattern of a row.
    In stack mode the patterns of all images are shown as a waterfall."""

    stack_mode = False

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.row = None
        glw = self.make_plot_widget()
        self.layout_main.addWidget(glw, 1)
        self.p_img = self.add_plot(glw, 0, 0, "", "", "Z (strip index)",
                                   image=True)
        self.p_slice = self.add_plot(glw, 0, 1, "", "Z (strip index)", "",
                                     image=True)
        title = "Patterns of all images" if self.stack_mode \
            else "Diffraction pattern"
        self.p_prof = self.add_plot(glw, 0, 2, title, "",
                                    "Image index (offset)" if self.stack_mode
                                    else "Intensity")
        self.img_row_line = self.make_line(self.p_img, 0,
                                           util.COLORS["lime"])
        self.img_span = self.make_span(self.p_img)
        self.slice_hline = self.make_line(self.p_slice, 0,
                                          util.COLORS["lime"])
        self.slice_vline = self.make_line(self.p_slice, 90,
                                          util.COLORS["lime"])
        self.prof_line = self.make_line(self.p_prof, 90,
                                        util.COLORS["yellow"], 2)
        self.connect_click(glw, [self.p_img, self.p_slice, self.p_prof],
                           self.on_click)
        self.slider_img = SliderRow("Image index (motor)")
        self.slider_x = SliderRow("X index",
                                  "Pixel index of the q-slice center.")
        self.slider_img.valueChanged.connect(lambda _v: self.update_image())
        self.slider_x.valueChanged.connect(lambda _v: self.update_qslice())
        self.build_display_controls([self.slider_img, self.slider_x])
        self.unify_widget_heights()


class MultiLinesViewerUI(EddtViewerUI):
    stack_mode = True


class MultiScansViewerUI(BaseViewerUI):
    """Several 1D scans: image of a scan, patterns of all scans at the same
    image index and row, and the q-slice of the displayed scan."""

    multi_input = True

    def __init__(self):
        super().__init__()
        self.datasets = []
        self.row = None
        glw = self.make_plot_widget()
        self.layout_main.addWidget(glw, 1)
        self.p_img = self.add_plot(glw, 0, 0, "", "", "Z (strip index)",
                                   image=True)
        self.p_prof = self.add_plot(glw, 0, 1, "Patterns of all scans", "",
                                    "Scan index (offset)")
        self.p_slice = self.add_plot(glw, 0, 2, "", "Z (strip index)", "",
                                     image=True)
        self.img_row_line = self.make_line(self.p_img, 0,
                                           util.COLORS["lime"])
        self.img_span = self.make_span(self.p_img)
        self.slice_hline = self.make_line(self.p_slice, 0,
                                          util.COLORS["lime"])
        self.slice_vline = self.make_line(self.p_slice, 90,
                                          util.COLORS["lime"])
        self.prof_line = self.make_line(self.p_prof, 90,
                                        util.COLORS["yellow"], 2)
        self.connect_click(glw, [self.p_img, self.p_slice, self.p_prof],
                           self.on_click)
        self.slider_img = SliderRow("Image index (motor)")
        self.slider_scan = SliderRow("Scan index")
        self.slider_x = SliderRow("X index",
                                  "Pixel index of the q-slice center.")
        self.slider_img.valueChanged.connect(lambda _v: self.update_image())
        self.slider_scan.valueChanged.connect(lambda _v: self.update_image())
        self.slider_x.valueChanged.connect(lambda _v: self.update_qslice())
        self.build_display_controls([self.slider_img, self.slider_scan,
                                     self.slider_x])
        self.unify_widget_heights()


class Scan2DViewerUI(BaseViewerUI):
    """2D scan: image at a scan position, its pattern at a row, and the
    q-slice along one scan axis."""

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.row = None
        glw = self.make_plot_widget()
        self.layout_main.addWidget(glw, 1)
        self.p_img = self.add_plot(glw, 0, 0, "", "", "Z (strip index)",
                                   image=True)
        self.p_prof = self.add_plot(glw, 0, 1, "Diffraction pattern", "",
                                    "Intensity")
        self.p_slice = self.add_plot(glw, 0, 2, "", "Z (strip index)", "",
                                     image=True)
        self.img_row_line = self.make_line(self.p_img, 0,
                                           util.COLORS["lime"])
        self.img_span = self.make_span(self.p_img)
        self.slice_hline = self.make_line(self.p_slice, 0,
                                          util.COLORS["lime"])
        self.slice_vline = self.make_line(self.p_slice, 90,
                                          util.COLORS["lime"])
        self.prof_line = self.make_line(self.p_prof, 90,
                                        util.COLORS["yellow"], 2)
        self.connect_click(glw, [self.p_img, self.p_slice, self.p_prof],
                           self.on_click)
        self.slider_row = SliderRow("Axis 0 index (rows)")
        self.slider_col = SliderRow("Axis 1 index (columns)")
        self.slider_x = SliderRow("X index",
                                  "Pixel index of the q-slice center.")
        self.slider_row.valueChanged.connect(lambda _v: self.update_image())
        self.slider_col.valueChanged.connect(lambda _v: self.update_image())
        self.slider_x.valueChanged.connect(lambda _v: self.update_qslice())
        self.build_display_controls([self.slider_row, self.slider_col,
                                     self.slider_x])
        axis_tip = "Scan axis of the q-slice image; the other index is fixed."
        self.combo_slice_axis = self.make_combo(
            ["Axis 0 (rows)", "Axis 1 (columns)"], tip=axis_tip)
        self.combo_slice_axis.currentIndexChanged.connect(
            lambda _i: self.update_qslice())
        grid = self.display_grid
        grid.addWidget(self.make_label("Q-slice along", axis_tip), 2, 0)
        grid.addWidget(self.combo_slice_axis, 2, 1, 1, 3)
        self.unify_widget_heights()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main window: a tab per tool, a shared status bar."""

    def __init__(self, tabs):
        super().__init__()
        self.setWindowTitle(util.APP_NAME)
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        self.setGeometry(calculate_geometry(util.MAIN_WIN_RATIO))
        self.close_handler = None
        self.tab_widget = QTabWidget()
        for widget, title in tabs:
            self.tab_widget.addTab(widget, title)
        self.setCentralWidget(self.tab_widget)
        status_bar = QStatusBar(self)
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_status.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Preferred)
        status_bar.addWidget(self.lbl_status, 1)
        self.lbl_cursor = QLabel("")
        status_bar.addPermanentWidget(self.lbl_cursor)
        status_bar.addPermanentWidget(QLabel("Theme"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(list(THEMES))
        self.combo_theme.setCurrentText(DEFAULT_THEME)
        status_bar.addPermanentWidget(self.combo_theme)
        for widget, _ in tabs:
            widget.status_func = self.lbl_status.setText
            widget.cursor_func = self.lbl_cursor.setText

    def center_on_screen(self):
        screen = self.screen().availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(screen.center())
        self.move(frame.topLeft())

    def closeEvent(self, event):
        if self.close_handler is not None and not self.close_handler():
            event.ignore()
            return
        event.accept()
