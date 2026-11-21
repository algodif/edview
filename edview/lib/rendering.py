"""
Rendering module for EDView.
Contains themes, widget and plot helpers, and the layouts of the tabs.
"""

import os
import platform
import functools
import textwrap
import traceback
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QRect, QRectF, Signal, QTimer, QObject, QEvent
from PySide6.QtGui import (QFont, QColor, QPalette, QFontDatabase, QIcon,
                           QVector3D, QFontMetrics)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                               QPushButton, QLineEdit, QComboBox, QCheckBox,
                               QSpinBox, QAbstractSpinBox, QGroupBox,
                               QTabWidget, QTabBar, QScrollArea, QFrame,
                               QListWidget,
                               QPlainTextEdit, QStatusBar, QMessageBox,
                               QAbstractItemView,
                               QDialog, QSizePolicy, QStackedWidget,
                               QTreeWidget, QTreeWidgetItem, QSplitter,
                               QSlider, QHeaderView)

import edview.lib.utilities as util

try:
    import pyqtgraph.opengl as gl
    GL_AVAILABLE = True
    GL_ERROR = ""
except ImportError as gl_error:
    # Without PyOpenGL the multiple-lines viewer uses a 2D waterfall
    gl = None
    GL_AVAILABLE = False
    GL_ERROR = str(gl_error)

pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")
ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "assets", "algodif_logo.png")

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
DARK = {
    "BG": "#1a1d23", "SURFACE": "#504d4d", "SURFACE2": "#2a2f3a",
    "BORDER": "#3a3f4e", "BORDER_HI": "#4d9fff", "ACCENT": "#4d9fff",
    "ACCENT_HOV": "#6ab0ff", "ACCENT_PRE": "#3a7fd4", "TEXT": "#d0d5e0",
    "TEXT_DIM": "#7a8099", "TEXT_HEAD": "#ffffff", "SEL_BG": "#2d4d7a",
    "SEL_FG": "#ffffff", "BTN_BG": "#363d4d", "BTN_HOV": "#424b60",
    "BTN_PRE": "#2a3448", "SCROLL": "#3a3f4e", "SCROLL_HOV": "#4d9fff",
    "STATUS_BG": "#181b20", "ALT_ROW": "#1e2230", "PG_BG": "#1e2128",
    "PG_FG": "#c8cdd6",
}
LIGHT = {
    "BG": "#f4f6f9", "SURFACE": "#c2c2c2", "SURFACE2": "#eef1f6",
    "BORDER": "#cdd2dc", "BORDER_HI": "#2563eb", "ACCENT": "#2563eb",
    "ACCENT_HOV": "#3b82f6", "ACCENT_PRE": "#1d4ed8", "TEXT": "#1e2333",
    "TEXT_DIM": "#6b7280", "TEXT_HEAD": "#0f172a", "SEL_BG": "#dbeafe",
    "SEL_FG": "#1e3a5f", "BTN_BG": "#dde5f2", "BTN_HOV": "#cbd8ee",
    "BTN_PRE": "#b9cbea", "SCROLL": "#c5cbd8", "SCROLL_HOV": "#2563eb",
    "STATUS_BG": "#e8eaf0", "ALT_ROW": "#f8fafc", "PG_BG": "w",
    "PG_FG": "k",
}
# Light theme with darker text, borders and selection for better contrast
LIGHT_HC = {
    "BG": "#dfe5ee", "SURFACE": "#c3cbd8", "SURFACE2": "#f4f7fb",
    "BORDER": "#4f5b6e", "BORDER_HI": "#0039a6", "ACCENT": "#0039a6",
    "ACCENT_HOV": "#1d4ed8", "ACCENT_PRE": "#002a7a", "TEXT": "#000000",
    "TEXT_DIM": "#2f3a4c", "TEXT_HEAD": "#000000", "SEL_BG": "#0039a6",
    "SEL_FG": "#ffffff", "BTN_BG": "#c9d4e5", "BTN_HOV": "#b5c6e2",
    "BTN_PRE": "#9fb5dc", "SCROLL": "#5f6b7e", "SCROLL_HOV": "#0039a6",
    "STATUS_BG": "#ccd4e0", "ALT_ROW": "#e8edf5", "PG_BG": "w",
    "PG_FG": "k",
}
# Neutral light tones similar to the Windows 11 default theme
CLASSIC = {
    "BG": "#f3f3f3", "SURFACE": "#e0e0e0", "SURFACE2": "#ffffff",
    "BORDER": "#d1d1d1", "BORDER_HI": "#005fb8", "ACCENT": "#005fb8",
    "ACCENT_HOV": "#1975c5", "ACCENT_PRE": "#004c93", "TEXT": "#1a1a1a",
    "TEXT_DIM": "#5d5d5d", "TEXT_HEAD": "#000000", "SEL_BG": "#cce4f7",
    "SEL_FG": "#000000", "BTN_BG": "#e1e4e8", "BTN_HOV": "#d3d8df",
    "BTN_PRE": "#c4cad3", "SCROLL": "#8a8a8a", "SCROLL_HOV": "#5d5d5d",
    "STATUS_BG": "#eeeeee", "ALT_ROW": "#f9f9f9", "PG_BG": "w",
    "PG_FG": "k",
}
THEMES = {"Light": LIGHT, "Light-HC": LIGHT_HC, "Classic": CLASSIC,
          "Dark": DARK}
DEFAULT_THEME = "Classic"
sel_theme = dict(THEMES[DEFAULT_THEME])
# Image display settings shared by all tabs (set from the top frame)
display = {"cmap": util.IMAGE_CMAPS[0], "scale": util.IMAGE_SCALES[0],
           "aspect": util.IMAGE_ASPECTS[0],
           "pmin": float(util.IMAGE_PERCENTILES[0]),
           "pmax": float(util.IMAGE_PERCENTILES[1])}


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
        background-color: {c['BG']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 5px;
        color: {c['TEXT_DIM']};
        background-color: {c['BG']};
        font-size: {fs - 1}px;
        font-weight: 500;
    }}
    QFrame {{
        border-radius: 2px;
        background-color: transparent;
    }}
    QScrollArea {{
        border: none;
    }}
    QPushButton {{
        background-color: {c['BTN_BG']};
        color: {c['TEXT']};
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
        border: 1px solid {c['ACCENT_PRE']};
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
    QPushButton#AccentOutlineBtn:hover {{
        border: 1px solid {c['ACCENT_HOV']};
        color: {c['ACCENT_HOV']};
    }}
    QLineEdit, QSpinBox {{
        background-color: {c['SURFACE2']};
        color: {c['TEXT']};
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
        color: {c['TEXT']};
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        padding: 3px 5px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['SURFACE2']};
        color: {c['TEXT']};
        border: 1px solid {c['BORDER']};
        selection-background-color: {c['SEL_BG']};
        selection-color: {c['SEL_FG']};
    }}
    QListWidget, QTreeWidget {{
        background-color: {c['SURFACE2']};
        color: {c['TEXT']};
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
    QHeaderView::section {{
        background-color: {c['SURFACE']};
        color: {c['TEXT_HEAD']};
        border: none;
        border-bottom: 1px solid {c['BORDER']};
        padding: 4px 8px;
        font-weight: 500;
    }}
    QPlainTextEdit {{
        background-color: {c['SURFACE2']};
        color: {c['TEXT']};
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        padding: 4px;
        selection-background-color: {c['SEL_BG']};
        selection-color: {c['SEL_FG']};
    }}
    QCheckBox {{
        color: {c['TEXT']};
        spacing: 6px;
        background-color: transparent;
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {c['BORDER']};
        border-radius: 2px;
        background-color: {c['SURFACE2']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['ACCENT']};
        border: 1px solid {c['ACCENT']};
    }}
    QLabel {{
        background-color: transparent;
        color: {c['TEXT']};
    }}
    QLabel#HeaderPathLabel {{
        color: {c['ACCENT']};
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: 1px solid {c['BORDER']};
        top: -1px;
    }}
    QTabWidget#MainTabs::tab-bar {{
        alignment: center;
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
    QScrollBar:vertical {{
        background: {c['SURFACE']};
        width: 8px;
        border-radius: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {c['SCROLL']};
        border-radius: 2px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c['SCROLL_HOV']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {c['SURFACE']};
        height: 8px;
        border-radius: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['SCROLL']};
        border-radius: 2px;
        min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c['SCROLL_HOV']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QWidget#HdfContentWindow {{
        background-color: {c['BG']};
    }}
    QSplitter::handle {{
        background-color: {c['BORDER']};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:hover {{
        background-color: {c['ACCENT']};
    }}
    QSlider {{
        background: transparent;
        min-height: 22px;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {c['BORDER']};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {c['ACCENT']};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {c['ACCENT']};
        border: 2px solid {c['SURFACE2']};
        width: 12px;
        height: 12px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {c['ACCENT_HOV']};
    }}
    QSlider::handle:horizontal:pressed {{
        background: {c['ACCENT_PRE']};
    }}
    QSlider::sub-page:horizontal:disabled,
    QSlider::handle:horizontal:disabled {{
        background: {c['TEXT_DIM']};
    }}
    QFrame#DisplayBar {{
        background-color: {c['SURFACE2']};
        border: 1px solid {c['BORDER']};
        border-radius: 3px;
    }}
    QLabel#BarTitle {{
        color: {c['ACCENT']};
        font-weight: 600;
    }}
    QFrame#BarSeparator {{
        background-color: {c['BORDER']};
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{
        border: 1px solid {c['BORDER_HI']};
    }}
    QStatusBar {{
        background-color: {c['STATUS_BG']};
        border-top: 1px solid {c['BORDER']};
        color: {c['TEXT']};
    }}
    QStatusBar::item {{
        border: none;
    }}
    QToolTip {{
        background-color: {c['SURFACE2']};
        color: {c['TEXT']};
        border: 1px solid {c['BORDER_HI']};
        padding: 1px 4px;
        font-size: {fs - 2}px;
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


def wrap_tooltip(text, width=util.TOOLTIP_WIDTH):
    """Wrap each line of a plain-text tooltip to at most width chars."""
    return "\n".join(textwrap.fill(line, width) if line else line
                     for line in text.split("\n"))


class ToolTipWrapper(QObject):
    """Application event filter wrapping long tooltips before they show,
    including tooltips set after the widgets are built."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ToolTip and isinstance(obj, QWidget):
            text = obj.toolTip()
            if text and not text.lstrip().startswith("<") and any(
                    len(line) > util.TOOLTIP_WIDTH
                    for line in text.split("\n")):
                obj.setToolTip(wrap_tooltip(text))
        return False


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


class PathListWidget(QListWidget):
    """List of paths accepting files and folders dropped from a file
    manager."""

    pathsDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()
                 if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.pathsDropped.emit(paths)
        else:
            super().dropEvent(event)


class SliderRow(QWidget):
    """Label, slider and spin box kept in sync; emits valueChanged."""

    valueChanged = Signal(int)
    released = Signal()

    def __init__(self, label, tip=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(util.UI_MARGIN_S, 0, util.UI_MARGIN_S, 0)
        layout.setSpacing(util.UI_SPACING_L + 2)
        self.label = QLabel(label)
        self.label.setMinimumWidth(util.FORM_LABEL_W + 40)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setPageStep(10)
        self.slider.setCursor(Qt.PointingHandCursor)
        self.spin = QSpinBox()
        self.spin.setFixedWidth(util.SMALL_ENTRY_W + 20)
        self.spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin.setKeyboardTracking(False)
        self.lbl_max = QLabel("/ 0")
        self.lbl_max.setMinimumWidth(util.SMALL_ENTRY_W - 10)
        for widget in (self.label, self.slider, self.spin):
            if tip:
                widget.setToolTip(tip)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.lbl_max)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self.released)
        self.spin.valueChanged.connect(self._on_spin)
        self.set_range(0)

    def is_dragging(self):
        return self.slider.isSliderDown()

    def set_range(self, maximum):
        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setRange(0, max(0, maximum))
            widget.blockSignals(False)
        self.lbl_max.setText(f"/ {max(0, maximum)}")
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
    def make_sub_grid():
        """Widget with a two-column grid (rows at the top), aligned with the
        parent group grid."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(util.UI_SPACING_L)
        grid.setVerticalSpacing(util.UI_SPACING_ML)
        grid.setColumnMinimumWidth(0, util.FORM_LABEL_W)
        grid.setColumnStretch(1, 1)
        grid.setAlignment(Qt.AlignTop)
        return widget, grid

    def make_page(self, stack):
        """Page of a stacked widget with a two-column grid."""
        page, grid = self.make_sub_grid()
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
        glw.ci.setContentsMargins(util.UI_SPACING_S, util.UI_SPACING_S,
                                  util.UI_SPACING_S, util.UI_SPACING_S)
        glw.ci.setSpacing(util.UI_SPACING_M)
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
            plot.getViewBox().setDefaultPadding(0)
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
        # An empty title still reserves a 30-px row; None hides it
        plot.setTitle(text or None, size=f"{util.PLOT_TITLE_SIZE}pt",
                      color=sel_theme["PG_FG"])

    @staticmethod
    def set_label(plot, axis, text):
        plot.setLabel(axis, text, color=sel_theme["PG_FG"],
                      **{"font-size": f"{util.PLOT_FONT_SIZE}pt"})

    def show_image(self, plot, mat, rect=None, auto_range=False):
        """Show a 2D array with the shared display settings;
        rect = (x0, y0, width, height) in data units."""
        item = plot.image_item
        item.raw_data = mat
        item.display_rect = rect
        self.render_image(item)
        if auto_range:
            plot.enableAutoRange()

    @staticmethod
    def render_image(item):
        mat = item.raw_data
        data = util.log_scale(mat) if display["scale"] == "log" \
            else np.asarray(mat, np.float32)
        vmin, vmax = util.get_levels(data, display["pmin"], display["pmax"])
        item.setImage(data, levels=(vmin, vmax), autoLevels=False)
        item.setLookupTable(get_colormap(display["cmap"]).getLookupTable(
            nPts=256))
        height, width = data.shape
        rect = item.display_rect or (-0.5, -0.5, width, height)
        item.setRect(QRectF(*rect))
        view_box = item.getViewBox()
        if view_box is not None:
            if display["aspect"] == "equal":
                # Square image pixels, also when the x-axis is scaled
                ratio = (rect[3] / height) / (rect[2] / width)
                view_box.setAspectLocked(True, ratio)
            else:
                view_box.setAspectLocked(False)

    def apply_display_settings(self, auto_range=False):
        """Redraw the shown images with the shared display settings."""
        for plot in self.plots:
            item = getattr(plot, "image_item", None)
            if getattr(item, "raw_data", None) is None:
                continue
            self.render_image(item)
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
        btn_select = self.make_button("Select base",
                                      self.select_base_folder,
                                      "Select the base folder of the tree.")
        btn_refresh = self.make_button("Refresh", self.refresh_tree,
                                       "Reload the folder tree.")
        for btn in (btn_select, btn_refresh):
            btn.setFixedWidth(110)
        header.addWidget(title)
        header.addWidget(self.lbl_base_folder, 1)
        header.addWidget(btn_select)
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(util.TREE_MIN_W)
        # Column as wide as its content: horizontal scroll bar for deep trees
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tree.itemExpanded.connect(self.on_tree_expand)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setToolTip("Ctrl/Shift-click to select several folders "
                             "(right-click: Multiple-scans viewer, Data "
                             "correction input).")
        self.tree.itemSelectionChanged.connect(self.on_folder_select)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(
            self.on_tree_context_menu)
        splitter.addWidget(self.tree)

        self.list_files = QListWidget()
        self.list_files.setMinimumWidth(util.TREE_MIN_W)
        self.list_files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_files.setToolTip("Double-click an HDF file to show its "
                                   "content. Ctrl/Shift-click to select "
                                   "several files.")
        self.list_files.itemSelectionChanged.connect(
            self.on_file_select_change)
        self.list_files.itemDoubleClicked.connect(self.on_file_double_click)
        self.list_files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_files.customContextMenuRequested.connect(
            self.on_file_context_menu)
        splitter.addWidget(self.list_files)

        # Preview: info, text, image, HDF
        self.preview = QStackedWidget()
        self.lbl_preview_info = QLabel("Select a file to preview it.")
        self.lbl_preview_info.setAlignment(Qt.AlignCenter)
        self.lbl_preview_info.setWordWrap(True)
        self.preview.addWidget(self.lbl_preview_info)
        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.preview.addWidget(self.text_preview)
        image_page = QWidget()
        image_layout = QVBoxLayout(image_page)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(util.UI_SPACING_M)
        hdf_row = QHBoxLayout()
        key_tip = "Numeric dataset of the HDF file to preview."
        self.lbl_hdf_key = self.make_label("Dataset", key_tip)
        self.combo_preview_key = self.make_combo([], tip=key_tip)
        self.combo_preview_key.currentIndexChanged.connect(
            lambda _i: self.on_preview_key_changed())
        self.btn_hdf_content = self.make_button(
            "Content...", self.show_hdf_content,
            "Show the full content of the HDF file (or double-click the "
            "file).")
        self.btn_hdf_content.setFixedWidth(util.MEDIUM_ENTRY_W)
        hdf_row.addWidget(self.lbl_hdf_key)
        hdf_row.addWidget(self.combo_preview_key, 1)
        hdf_row.addWidget(self.btn_hdf_content)
        image_layout.addLayout(hdf_row)
        # Image plot for 2D data, line plot for 1D data
        self.plot_stack = QStackedWidget()
        glw = self.make_plot_widget()
        self.p_preview = self.add_plot(glw, 0, 0, image=True)
        self.plot_stack.addWidget(glw)
        glw_line = self.make_plot_widget()
        self.p_preview_line = self.add_plot(glw_line, 0, 0, "", "Index",
                                            "Value")
        self.p_preview_line.setClipToView(True)
        self.p_preview_line.setDownsampling(auto=True, mode="peak")
        self.plot_stack.addWidget(glw_line)
        image_layout.addWidget(self.plot_stack, 1)
        # Rows of (axis, index slider): 1 for 3D datasets and folders of
        # images, 2 for 4D datasets
        axis_tip = "Axis of the dataset to slice along."
        self.slice_rows = []
        for num in range(2):
            row = QWidget()
            hbox = QHBoxLayout(row)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(util.UI_SPACING_L)
            label = self.make_label(f"Axis {num + 1}", axis_tip)
            combo = self.make_combo([], tip=axis_tip)
            combo.setFixedWidth(util.SMALL_ENTRY_W)
            combo.currentIndexChanged.connect(
                lambda _i, n=num: self.on_preview_axis_changed(n))
            slider = SliderRow("Index")
            slider.label.setMinimumWidth(util.SMALL_ENTRY_W)
            slider.valueChanged.connect(lambda _v: self.slice_timer.start())
            hbox.addWidget(label)
            hbox.addWidget(combo)
            hbox.addWidget(slider, 1)
            row.hide()
            image_layout.addWidget(row)
            self.slice_rows.append((row, label, combo, slider))
        self.slice_timer = QTimer(self)
        self.slice_timer.setSingleShot(True)
        self.slice_timer.setInterval(util.SLICE_UPDATE_DELAY_MS)
        self.slice_timer.timeout.connect(self.update_preview_slice)
        self.preview.addWidget(image_page)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 3)
        # Same initial width for the folder tree and the file list
        splitter.setSizes([1000, 1000, 4500])
        layout.addWidget(splitter, 1)
        self.unify_widget_heights()

    def show_preview_page(self, name):
        pages = {"info": 0, "text": 1, "image": 2}
        self.preview.setCurrentIndex(pages[name])

    def set_hdf_controls_visible(self, visible):
        for widget in (self.lbl_hdf_key, self.combo_preview_key,
                       self.btn_hdf_content):
            widget.setVisible(visible)

    def show_plot_page(self, line):
        self.plot_stack.setCurrentIndex(1 if line else 0)

    def setup_slice_row(self, num, label, size, index=0, show_axis=False):
        """Slider of row 'num' over 'size' images. Without the axis combo,
        the row is shown only if there are several images."""
        row, lbl_axis, combo, slider = self.slice_rows[num]
        lbl_axis.setVisible(show_axis)
        combo.setVisible(show_axis)
        slider.label.setText(label)
        slider.set_range(size - 1)
        slider.set_value(index, notify=False)
        row.setVisible(show_axis or size > 1)

    def hide_slice_rows(self, start=0):
        for row, _, _, _ in self.slice_rows[start:]:
            row.hide()


class HdfContentWindow(QWidget):
    """Groups, datasets and attributes of an HDF file. Double-click a
    dataset to preview it in the browser."""

    datasetActivated = Signal(str, str)

    def __init__(self, parent, file_path, nodes):
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        # Without a styled background, a QWidget window is painted black
        self.setObjectName("HdfContentWindow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle(f"HDF content: {os.path.basename(file_path)}")
        main = parent.window()
        width = int(main.width() * util.CHILD_WIN_RATIO)
        height = int(main.height() * util.CHILD_WIN_RATIO)
        center = main.frameGeometry().center()
        self.setGeometry(center.x() - width // 2, center.y() - height // 2,
                         width, height)
        self.file_path = file_path
        layout = QVBoxLayout(self)
        layout.setContentsMargins(util.UI_MARGIN_L, util.UI_MARGIN_L,
                                  util.UI_MARGIN_L, util.UI_MARGIN_L)
        layout.setSpacing(util.UI_SPACING_L)
        header = QLabel(file_path)
        header.setObjectName("HeaderPathLabel")
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(header)
        splitter = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Shape", "Type", "Link"])
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(14)
        self.tree.setToolTip("Double-click a numeric dataset to preview "
                             "it in the Data browser.")
        self.tree.itemSelectionChanged.connect(self.on_select)
        self.tree.itemDoubleClicked.connect(self.on_double_click)
        splitter.addWidget(self.tree)
        self.text_info = QPlainTextEdit()
        self.text_info.setReadOnly(True)
        self.text_info.setLineWrapMode(QPlainTextEdit.NoWrap)
        splitter.addWidget(self.text_info)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        self.populate(nodes)

    def populate(self, nodes):
        root = QTreeWidgetItem(self.tree, ["/", "", "File", ""])
        root.setData(0, Qt.UserRole, "")
        items = {"": root}
        types = {"group": "Group", "broken": "Broken link"}
        for path, kind, shape, dtype, link in nodes:
            parent = items.get(path.rpartition("/")[0], root)
            item = QTreeWidgetItem(parent, [
                path.rpartition("/")[2],
                "" if shape is None else str(shape),
                types.get(kind, dtype), link])
            item.setData(0, Qt.UserRole, path)
            item.setData(1, Qt.UserRole, kind)
            if link:
                item.setToolTip(3, link)
            items[path] = item
        if len(nodes) <= 500:
            self.tree.expandAll()
        else:
            root.setExpanded(True)
        for column in range(4):
            self.tree.resizeColumnToContents(column)
        self.tree.setCurrentItem(root)

    def on_select(self):
        items = self.tree.selectedItems()
        if not items:
            return
        try:
            text = util.hdf_item_info(self.file_path,
                                      items[0].data(0, Qt.UserRole))
        except Exception as error:
            text = f"Can't read the item: {error}"
        self.text_info.setPlainText(text)

    def on_double_click(self, item, _column):
        if item.data(1, Qt.UserRole) == "dataset":
            self.datasetActivated.emit(self.file_path,
                                       item.data(0, Qt.UserRole))


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
        self.build_calibration_group(panel)
        self.build_input_group(panel)
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
        self.list_input = PathListWidget()
        self.list_input.setFixedHeight(util.INPUT_LIST_H)
        self.list_input.setToolTip(
            "Scans to correct, one per row: scan folders (with a 'tifs' "
            "sub-folder), folders of tif files, HDF files, or groups of "
            "image files.\nFiles and folders can also be dropped here.")
        self.list_input.pathsDropped.connect(self.add_inputs)
        grid.addWidget(self.list_input, 0, 0, 1, 2)
        buttons = self.add_buttons(
            grid, 1, [("+ Folders", self.browse_folders),
                      ("+ Files", self.browse_files),
                      ("Remove", self.remove_inputs),
                      ("Clear", self.clear_inputs)])
        tips = ("Add scan folders (Ctrl/Shift-click to select several).",
                "Add data files. Image files selected together form one "
                "scan; each HDF file is a scan.",
                "Remove the selected rows.", "Remove all rows.")
        for button, tip in zip(buttons, tips):
            button.setToolTip(tip)
        self.combo_hdf_key = self.add_combo(
            grid, 2, "HDF key", [], tip="Dataset of HDF input files.",
            editable=True)
        self.edit_axis = self.add_entry(
            grid, 3, "Axis (3D HDF)", 0,
            "Axis along which images are stacked in 3D HDF datasets.")
        self.add_buttons(grid, 4, [("Load", self.load_input)])
        self.lbl_input_info = QLabel("No input loaded.")
        self.lbl_input_info.setWordWrap(True)
        grid.addWidget(self.lbl_input_info, 5, 0, 1, 2)

    def build_calibration_group(self, panel):
        grid = self.make_group(panel, "Calibration")
        self.edit_coef = QLineEdit("None")
        self.edit_coef.setReadOnly(True)
        grid.addWidget(self.edit_coef, 0, 0, 1, 2)
        self.add_buttons(grid, 1, [("Load coefficients...",
                                    self.load_coefficients)])
        step_tip = ("Step of the uniform q-grid of the corrected data:\n"
                    "- mean/median/max: of the pixel spacing of the q-map "
                    "computed from the coefficients.\n"
                    "- q-max: step = q-max / (width - 1).\n"
                    "- q-step: the given step.")
        self.combo_step = self.add_combo(grid, 2, "Output step",
                                         util.OUTPUT_STEP_OPTIONS, "mean",
                                         step_tip)
        self.edit_step = self.add_entry(grid, 3, "Value", "", step_tip)
        self.add_buttons(grid, 4, [("Update preview", self.update_preview)])

        def update_state(_text=None):
            self.edit_step.setEnabled(self.combo_step.currentText()
                                      in ("q-max", "q-step"))

        self.combo_step.currentTextChanged.connect(update_state)
        update_state()

    def build_background_group(self, panel):
        grid = self.make_group(panel, "Background removal")
        self.check_bck = self.add_check(
            grid, 0, "Remove background", False,
            "Remove the background of each strip of the corrected images "
            "before saving. Use a different output base folder than for "
            "data saved with background, to not overwrite it.")
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
            "Output base folder, different from the folder of the input "
            "scans. Each scan is saved in a sub-folder with the name of the "
            "input scan folder (or HDF file), images keep their file "
            f"names. '{util.MAP_INFO_FOLDER}' holds the raw and corrected "
            "q-maps and a JSON with the coefficients file, the input path "
            "and the q-step (stored in the file for HDF output).")
        self.add_buttons(grid, 1, [("Browse...", self.browse_output),
                                   ("Make folder", self.make_output_folder)])
        format_tip = ("- tif: one tif per image, in the scan folder.\n"
                      "- hdf: '<scan>.hdf' with dataset "
                      f"'{util.HDF_DATA_KEY}', the q-maps "
                      f"'{util.HDF_RAW_MAP_KEY}' and "
                      f"'{util.HDF_CORRECTED_MAP_KEY}', and the map info "
                      f"in '{util.HDF_MAP_INFO_KEY}'.\n"
                      "- xy/csv: one file per strip.")
        self.combo_format = self.add_combo(grid, 2, "Format",
                                           util.OUTPUT_FORMATS, "tif",
                                           format_tip)
        # One option box per format; only the selected one is shown so the
        # group height follows its number of rows
        format_boxes = []
        box, page = self.make_sub_grid()
        format_boxes.append(box)
        page.addWidget(QLabel("Saved in '<scan>/<image name>.tif'."), 0, 0,
                       1, 2)
        box, page = self.make_sub_grid()
        format_boxes.append(box)
        self.edit_out_key = self.add_entry(
            page, 0, "Dataset key", util.HDF_DATA_KEY,
            "Dataset of the corrected images in '<scan>.hdf'.")
        box, page = self.make_sub_grid()
        format_boxes.append(box)
        data_tip = ("- raw: measured intensities with the calibrated q-value "
                    "of each pixel (non-uniform grid). No background "
                    "removal.\n"
                    "- corrected: intensities on the uniform grid, without "
                    "background if selected.")
        self.combo_xy_data = self.add_combo(page, 0, "Data",
                                            ["corrected", "raw"], tip=data_tip)
        self.combo_xy_type = self.add_combo(page, 1, "File type",
                                            ["xy", "csv"])
        self.combo_zip = self.add_combo(
            page, 2, "Packing", ["zip", "unzip"], "unzip",
            tip="One zip file or one sub-folder per image.")
        self.combo_xy_x, self.edit_max_tth = self.add_two_theta_rows(page, 3)
        self.edit_xy_folder = self.add_entry(page, 5, "Sub-folder", "xy",
                                             "Sub-folder of the scan output.")
        formats = QWidget()
        vbox = QVBoxLayout(formats)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        for box in format_boxes:
            vbox.addWidget(box)
        grid.addWidget(formats, 3, 0, 1, 2)

        def show_format_box(index):
            for i, box in enumerate(format_boxes):
                box.setVisible(i == index)

        self.combo_format.currentIndexChanged.connect(show_format_box)
        show_format_box(self.combo_format.currentIndex())
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

    def add_two_theta_rows(self, grid, row):
        """X-axis of xy/csv files and the max 2theta of the conversion."""
        x_tip = ("x-values of the xy/csv files:\n- q: q (1/\u00c5).\n"
                 "- two-theta: equivalent 2theta (degrees) from q "
                 "(algodif q_to_two_theta): the largest q maps to Max "
                 "2theta, with one reference energy for all strips.")
        combo = self.add_combo(grid, row, "X-axis", util.XY_X_AXES, "q",
                               x_tip)
        edit = self.add_entry(grid, row + 1, "Max 2theta",
                              util.DEFAULT_MAX_TWO_THETA,
                              "2theta (degrees, 0-180) of the largest q.")

        def update_state(_text=None):
            edit.setEnabled(combo.currentText() == "two-theta")

        combo.currentTextChanged.connect(update_state)
        update_state()
        return combo, edit

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
        # Option boxes shown depending on the conversion
        box_key, sub = self.make_sub_grid()
        self.combo_conv_key = self.add_combo(sub, 0, "HDF key", [],
                                             tip="Dataset of the HDF source.",
                                             editable=True)
        grid.addWidget(box_key, 3, 0, 1, 2)
        box_step, sub = self.make_sub_grid()
        self.edit_conv_step = self.add_entry(
            sub, 0, "Q step", "",
            "Step of the q-values. Empty: read from the map_info, the "
            "metadata or the HDF file.")
        grid.addWidget(box_step, 4, 0, 1, 2)
        self.edit_conv_output = self.add_entry(
            grid, 5, "Output folder", "",
            "Folder of the output HDF file, tif files or xy/csv files.")
        self.add_buttons(grid, 6, [("Browse output folder...",
                                    self.browse_conv_output),
                                   ("Make folder",
                                    self.make_conv_output_folder)])
        box_hdf, sub = self.make_sub_grid()
        self.edit_conv_out_name = self.add_entry(
            sub, 0, "Output file name", "",
            "Name of the output HDF file in the output folder. Empty: the "
            "name of the source folder.")
        self.edit_conv_out_key = self.add_entry(
            sub, 1, "Output hdf key", util.HDF_DATA_KEY,
            "Dataset of the images in the output HDF file.")
        grid.addWidget(box_hdf, 7, 0, 1, 2)
        box_4d, sub = self.make_sub_grid()
        size_tip = ("Number of scan rows / columns. Empty (both): from file "
                    "names ending with '_<row>_<col>'. Otherwise the sorted "
                    "tif files follow the scan order.")
        self.edit_conv_nrow = self.add_entry(sub, 0, "Rows", "", size_tip)
        self.edit_conv_ncol = self.add_entry(sub, 1, "Columns", "", size_tip)
        self.combo_conv_order = self.add_combo(
            sub, 2, "Scan order", util.SCAN_ORDERS, "row",
            "Acquisition order of the sorted tif files (first image at row "
            "0, column 0):\n- row: each row before the next row.\n"
            "- column: each column before the next column.\n"
            "- snake_row / snake_column: alternate rows / columns "
            "reversed.")
        chunk_tip = (
            "Chunk shape of the 4D dataset (width, height, rows, columns), "
            "the data shape used by algodif. Empty: no chunking.\nFew "
            "pixels with many scan positions per chunk make maps over the "
            "scan (q-slices) fast; e.g. 64, 64, 8, 8.")
        self.edit_conv_chunks = self.add_entry(
            sub, 3, "Chunk shape",
            ", ".join(str(v) for v in util.DEFAULT_4D_CHUNKS), chunk_tip)
        self.lbl_conv_chunk = QLabel("")
        sub.addWidget(self.lbl_conv_chunk, 4, 1)

        def update_chunk_size(text):
            try:
                chunks = util.parse_chunks(text)
            except ValueError:
                self.lbl_conv_chunk.setText("Invalid: use 4 integers")
                return
            self.lbl_conv_chunk.setText(
                "No chunking" if chunks is None else
                f"Chunk size: "
                f"{util.format_bytes(util.chunk_nbytes(chunks))} (float32)")

        self.edit_conv_chunks.textChanged.connect(update_chunk_size)
        update_chunk_size(self.edit_conv_chunks.text())
        grid.addWidget(box_4d, 8, 0, 1, 2)
        box_xy, sub = self.make_sub_grid()
        self.combo_conv_xy_type = self.add_combo(sub, 0, "File type",
                                                 ["xy", "csv"])
        self.combo_conv_zip = self.add_combo(sub, 1, "Packing",
                                             ["zip", "unzip"], "unzip")
        self.combo_conv_xy_x, self.edit_conv_max_tth = \
            self.add_two_theta_rows(sub, 2)
        grid.addWidget(box_xy, 9, 0, 1, 2)
        button = self.add_buttons(grid, 10, [("Convert",
                                              self.run_conversion)])[0]
        button.setObjectName("AccentOutlineBtn")
        self.lbl_conv_info = QLabel("")
        self.lbl_conv_info.setWordWrap(True)
        grid.addWidget(self.lbl_conv_info, 11, 0, 1, 2)

        def update_state(_text=None):
            mode = util.CONVERSIONS[self.combo_conversion.currentText()]
            box_key.setVisible(mode in ("hdf_to_tif", "to_xy"))
            box_step.setVisible(mode != "hdf_to_tif")
            box_hdf.setVisible(mode in ("tif_to_hdf", "tif2d_to_hdf4d"))
            box_4d.setVisible(mode == "tif2d_to_hdf4d")
            box_xy.setVisible(mode == "to_xy")

        self.combo_conversion.currentTextChanged.connect(update_state)
        update_state()


# ---------------------------------------------------------------------------
# Viewers
# ---------------------------------------------------------------------------

class BaseViewerUI(BaseTab):
    """Input group, plot area and display controls of the viewers."""

    multi_input = False
    # Choice of the 1D array of positions (one value per image)
    select_positions = False
    # Label and tooltip of the combo of each positions key
    positions_labels = ("Positions",)
    positions_tips = ("Key of the positions of the images in the metadata "
                      "(a list with one value per image). Its name labels "
                      "the plots.",)
    position_combos = ()
    # 3D view of patterns (set by make_gl_panel)
    gl_view = None

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
        """Grid of two rows sharing columns: labels, fields, buttons.
        Row 0: path, HDF key, Load data. Row 1: q-step, metadata,
        positions, Load metadata."""
        group = QGroupBox("Data")
        grid = self.make_grid(group)

        def button(text, callback, tip=None):
            btn = self.make_button(text, callback, tip)
            btn.setFixedWidth(120)
            return btn

        path_tip = "Folder of processed tif files or an HDF file."
        if self.multi_input:
            path_tip += (" Several scans separated by ';'; the metadata of "
                         "the first scan gives the positions of all scans.")
        grid.addWidget(self.make_label("Path", path_tip), 0, 0)
        self.edit_path = self.make_line_edit(tip=path_tip)
        grid.addWidget(self.edit_path, 0, 1, 1, 3)
        grid.addWidget(button("Folder...", self.browse_folder), 0, 4)
        grid.addWidget(button("File...", self.browse_file), 0, 5)
        key_tip = "Dataset of the HDF file."
        grid.addWidget(self.make_label("HDF key", key_tip), 0, 6)
        self.combo_hdf_key = self.make_combo([], tip=key_tip, editable=True)
        self.combo_hdf_key.setMinimumWidth(util.LEFT_PANEL_W - 100)
        grid.addWidget(self.combo_hdf_key, 0, 7)
        grid.addWidget(button("Load data", self.load_data), 0, 8)

        step_tip = ("Step of the q-values of the pixels. Empty: read from "
                    "the map_info of corrected data, the metadata or the "
                    "HDF file; the x-axis is in pixels if not found. Press "
                    "Enter to apply a typed value.")
        grid.addWidget(self.make_label("Q step", step_tip), 1, 0)
        self.edit_step = self.make_line_edit("", 2 * util.MEDIUM_ENTRY_W,
                                             step_tip)
        self.edit_step.returnPressed.connect(self.on_step_entered)
        grid.addWidget(self.edit_step, 1, 1)
        meta_tip = ("Metadata with the positions of the images: a JSON or "
                    "nxs/HDF file. Empty: the JSON file next to the data, or "
                    "the HDF data file.")
        grid.addWidget(self.make_label("Metadata", meta_tip), 1, 2)
        self.edit_metadata = self.make_line_edit(tip=meta_tip)
        self.btn_metadata = button("File...", self.browse_metadata,
                                   "Select a metadata file (JSON or nxs).")
        two_axes = self.select_positions and len(self.positions_labels) > 1
        if two_axes:
            # Shorter metadata field: room for the positions of both axes
            grid.addWidget(self.edit_metadata, 1, 3)
            grid.addWidget(self.btn_metadata, 1, 4)
        else:
            grid.addWidget(self.edit_metadata, 1, 3, 1, 2)
            grid.addWidget(self.btn_metadata, 1, 5)
        if self.select_positions:
            self.position_combos = []
            if two_axes:
                axes = QWidget()
                hbox = QHBoxLayout(axes)
                hbox.setContentsMargins(0, 0, 0, 0)
                hbox.setSpacing(util.UI_SPACING_L)
                grid.addWidget(axes, 1, 5, 1, 3)
            for num, (label, tip) in enumerate(zip(self.positions_labels,
                                                   self.positions_tips)):
                combo = self.make_combo([], tip=tip)
                if two_axes:
                    hbox.addWidget(self.make_label(label, tip))
                    hbox.addWidget(combo, 1)
                else:
                    combo.setMinimumWidth(util.LEFT_PANEL_W - 100)
                    grid.addWidget(self.make_label(label, tip), 1, 6)
                    grid.addWidget(combo, 1, 7)
                self.position_combos.append(combo)
            self.combo_motor = self.position_combos[0]
            grid.addWidget(button(
                "Load metadata", self.load_metadata,
                "Use the selected positions for the loaded data."), 1, 8)
        self.build_extra_inputs(grid)
        grid.setColumnStretch(3, 3)
        grid.setColumnStretch(7, 2)
        self.layout_main.addWidget(group, 0)

    def build_extra_inputs(self, grid):
        """Viewer-specific inputs below the two rows of the data group."""

    def build_display_controls(self, slider_rows, stacked=False,
                               first_row=0):
        """Slider rows on the left, display options on the right from
        first_row; stacked: one save button per row, as wide as the radius
        row."""
        controls = QHBoxLayout()
        sliders = QVBoxLayout()
        for row in slider_rows:
            sliders.addWidget(row)
        controls.addLayout(sliders, 1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(util.UI_SPACING_L)
        radius_tip = ("Half-width (pixels) of the x-range averaged for the "
                      "q-slice image.")
        grid.addWidget(self.make_label("Radius", radius_tip), first_row, 0)
        self.combo_radius = self.make_combo(util.RADII, util.DEFAULT_RADIUS,
                                            radius_tip)
        self.combo_radius.currentTextChanged.connect(
            lambda _t: self.update_qslice())
        grid.addWidget(self.combo_radius, first_row, 1)
        btn_pattern = self.make_button(
            "Save pattern", self.save_pattern,
            "Save the displayed diffraction pattern(s) to a CSV file.")
        btn_slice = self.make_button(
            "Save q-slice", self.save_qslice,
            "Save the q-slice image to a TIF file.")
        row = first_row + 1
        if stacked:
            grid.addWidget(btn_pattern, row, 0, 1, 2)
            grid.addWidget(btn_slice, row + 1, 0, 1, 2)
        else:
            for btn in (btn_pattern, btn_slice, self.combo_radius):
                btn.setFixedWidth(util.SAVE_BTN_W)
            grid.addWidget(btn_pattern, row, 0)
            grid.addWidget(btn_slice, row, 1)
        self.display_grid = grid
        controls.addLayout(grid, 0)
        self.layout_main.addLayout(controls, 0)

    def x_label(self):
        return util.Q_LABEL

    def make_gl_panel(self, title):
        """Title and 3D view (OpenGL) of patterns."""
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(util.UI_SPACING_S)
        self.lbl_gl_title = QLabel(f"{title} (3D). Click on the image to "
                                   f"select a row.")
        self.lbl_gl_title.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.lbl_gl_title)
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor(sel_theme["PG_BG"])
        self.gl_view.setMinimumWidth(300)
        self.gl_view.setMouseTracking(True)
        self.gl_view.setToolTip("Drag: rotate. Wheel: zoom. Ctrl+drag or "
                                "middle-drag: pan.")
        self.reset_gl_camera()
        vbox.addWidget(self.gl_view, 1)
        return panel

    def reset_gl_camera(self):
        width, depth, height = util.GL_BOX
        self.gl_view.setCameraPosition(
            pos=QVector3D(width / 2, depth / 2, height / 2),
            distance=1.4 * max(width, depth), elevation=25, azimuth=-60)

    def refresh_plot_theme(self):
        super().refresh_plot_theme()
        if self.gl_view is not None:
            self.gl_view.setBackgroundColor(sel_theme["PG_BG"])


class EddtViewerUI(BaseViewerUI):
    """1D scan: image, q-slice across the scan, and the pattern of a row.
    In stack mode the patterns of all images are shown as a waterfall."""

    stack_mode = False
    select_positions = True

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.row = None
        self.gl_view = None
        use_gl = self.stack_mode and GL_AVAILABLE
        glw = self.make_plot_widget()
        # With the 3D view, the q-slice gets its own widget so the splitter
        # sets the widths image : q-slice : 3D view = 3 : 2 : 3
        glw_slice = self.make_plot_widget() if use_gl else glw
        self.p_img = self.add_plot(glw, 0, 0, "", "", "Z (strip index)",
                                   image=True)
        self.p_slice = self.add_plot(glw_slice, 0, 0 if use_gl else 1, "",
                                     "Z (strip index)", "", image=True)
        self.img_row_line = self.make_line(self.p_img, 0,
                                           util.COLORS["lime"])
        self.img_span = self.make_span(self.p_img)
        self.slice_hline = self.make_line(self.p_slice, 0,
                                          util.COLORS["lime"])
        self.slice_vline = self.make_line(self.p_slice, 90,
                                          util.COLORS["lime"])
        clickable = [self.p_img, self.p_slice]
        if use_gl:
            splitter = QSplitter(Qt.Horizontal)
            for widget, stretch in ((glw, 3), (glw_slice, 2),
                                    (self.make_gl_panel("Patterns of all "
                                                        "images"), 3)):
                splitter.addWidget(widget)
                splitter.setStretchFactor(splitter.count() - 1, stretch)
            splitter.setSizes([3000, 2000, 3000])
            self.layout_main.addWidget(splitter, 1)
            self.p_prof = self.prof_line = None
            self.connect_click(glw_slice, [self.p_slice], self.on_click)
            clickable = [self.p_img]
        else:
            self.layout_main.addWidget(glw, 1)
            title = "Patterns of all images" if self.stack_mode \
                else "Diffraction pattern"
            self.p_prof = self.add_plot(
                glw, 0, 2, title, "", "Image index (offset)"
                if self.stack_mode else "Intensity")
            self.prof_line = self.make_line(self.p_prof, 90,
                                            util.COLORS["yellow"], 2)
            clickable.append(self.p_prof)
        self.connect_click(glw, clickable, self.on_click)
        self.slider_img = SliderRow("Image index (motor)")
        self.slider_x = SliderRow("Q index",
                                  "Pixel index of the q-slice center.")
        self.slider_img.valueChanged.connect(lambda _v: self.update_image())
        self.slider_x.valueChanged.connect(lambda _v: self.update_qslice())
        self.build_display_controls([self.slider_img, self.slider_x])
        self.unify_widget_heights()


class MultiLinesViewerUI(EddtViewerUI):
    stack_mode = True


class MultiScansViewerUI(BaseViewerUI):
    """Several 1D scans: patterns of all scans at the same image index and
    row (3D view, or 2D waterfall without PyOpenGL), the q-slice and the
    image of the displayed scan."""

    multi_input = True
    select_positions = True

    def __init__(self):
        super().__init__()
        self.datasets = []
        self.row = None
        glw = self.make_plot_widget()
        if GL_AVAILABLE:
            # Widths image : q-slice : 3D view = 3 : 2 : 3
            glw_slice = self.make_plot_widget()
            self.p_img = self.add_plot(glw, 0, 0, "", "", "Z (strip index)",
                                       image=True)
            self.p_slice = self.add_plot(glw_slice, 0, 0, "",
                                         "Z (strip index)", "", image=True)
            splitter = QSplitter(Qt.Horizontal)
            for widget, stretch in ((glw, 3), (glw_slice, 2),
                                    (self.make_gl_panel("Patterns of all "
                                                        "scans"), 3)):
                splitter.addWidget(widget)
                splitter.setStretchFactor(splitter.count() - 1, stretch)
            splitter.setSizes([3000, 2000, 3000])
            self.layout_main.addWidget(splitter, 1)
            self.p_prof = self.prof_line = None
            self.connect_click(glw_slice, [self.p_slice], self.on_click)
            self.connect_click(glw, [self.p_img], self.on_click)
        else:
            self.layout_main.addWidget(glw, 1)
            self.p_img = self.add_plot(glw, 0, 0, "", "", "Z (strip index)",
                                       image=True)
            self.p_prof = self.add_plot(glw, 0, 1, "Patterns of all scans",
                                        "", "Scan index (offset)")
            self.p_slice = self.add_plot(glw, 0, 2, "", "Z (strip index)", "",
                                         image=True)
            self.prof_line = self.make_line(self.p_prof, 90,
                                            util.COLORS["yellow"], 2)
            self.connect_click(glw, [self.p_img, self.p_slice, self.p_prof],
                               self.on_click)
        self.img_row_line = self.make_line(self.p_img, 0,
                                           util.COLORS["lime"])
        self.img_span = self.make_span(self.p_img)
        self.slice_hline = self.make_line(self.p_slice, 0,
                                          util.COLORS["lime"])
        self.slice_vline = self.make_line(self.p_slice, 90,
                                          util.COLORS["lime"])
        self.slider_img = SliderRow("Image index (motor)")
        self.slider_scan = SliderRow("Scan index")
        self.slider_x = SliderRow("Q index",
                                  "Pixel index of the q-slice center.")
        self.slider_img.valueChanged.connect(lambda _v: self.update_image())
        self.slider_scan.valueChanged.connect(lambda _v: self.update_image())
        self.slider_x.valueChanged.connect(lambda _v: self.update_qslice())
        self.build_display_controls([self.slider_scan, self.slider_img,
                                     self.slider_x], stacked=True)
        self.unify_widget_heights()


class Scan2DViewerUI(BaseViewerUI):
    """2D scan: image at a scan position, its pattern at a row, and the
    q-slice along one scan axis."""

    select_positions = True
    positions_labels = ("Axis 0", "Axis 1")
    positions_tips = (
        "Key of the positions of the rows (axis 0 of the scan) in the "
        "metadata: a list with one value per row. Its name labels the "
        "plots.",
        "Key of the positions of the columns (axis 1 of the scan) in the "
        "metadata: a list with one value per column. Its name labels the "
        "plots.")

    def build_extra_inputs(self, grid):
        """Rows, columns and scan order of a 3D HDF dataset (images in
        acquisition order)."""
        tip = ("For a 3D HDF dataset (images, strips, pixels): number of "
               "scan rows / columns. One of them is enough, the other is "
               "computed from the number of images. Not used for tif files "
               "(named '..._<row>_<col>.tif') or 4D datasets.")
        order_tip = ("Acquisition order of the images of a 3D HDF dataset "
                     "(first image at row 0, column 0):\n- row: each row "
                     "before the next row.\n- column: each column before "
                     "the next column.\n- snake_row / snake_column: "
                     "alternate rows / columns reversed.")
        self.lbl_scan_grid = self.make_label("3D scan", tip)
        grid.addWidget(self.lbl_scan_grid, 2, 0)
        box = QWidget()
        hbox = QHBoxLayout(box)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(util.UI_SPACING_L)
        self.edit_scan_nrow = self.make_line_edit("", util.MEDIUM_ENTRY_W,
                                                  tip)
        self.edit_scan_ncol = self.make_line_edit("", util.MEDIUM_ENTRY_W,
                                                  tip)
        self.combo_scan_order = self.make_combo(util.SCAN_ORDERS, "row",
                                                order_tip)
        for label, widget, widget_tip in (
                ("Rows", self.edit_scan_nrow, tip),
                ("Columns", self.edit_scan_ncol, tip),
                ("Scan order", self.combo_scan_order, order_tip)):
            hbox.addWidget(self.make_label(label, widget_tip))
            hbox.addWidget(widget)
        hbox.addStretch(1)
        grid.addWidget(box, 2, 1, 1, 8)
        self.scan_grid_box = box

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
        self.p_prof = self.add_plot(glw, 0, 2, "Diffraction pattern", "",
                                    "Intensity")
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
        self.slider_x = SliderRow("Q index",
                                  "Pixel index of the q-slice center.")
        self.slider_row.valueChanged.connect(lambda _v: self.update_image())
        self.slider_col.valueChanged.connect(lambda _v: self.update_image())
        self.slider_x.valueChanged.connect(lambda _v: self.update_qslice())
        # Heavy q-slice loading waits for the slider release
        for slider in (self.slider_row, self.slider_col, self.slider_x):
            slider.released.connect(self.update_qslice)
        self.build_display_controls([self.slider_row, self.slider_col,
                                     self.slider_x], first_row=1)
        axis_tip = "Scan axis of the q-slice image; the other index is fixed."
        self.combo_slice_axis = self.make_combo(
            ["Axis 0 (rows)", "Axis 1 (columns)"], tip=axis_tip)
        self.combo_slice_axis.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_slice_axis.setFixedWidth(util.SAVE_BTN_W)
        self.combo_slice_axis.currentIndexChanged.connect(
            lambda _i: self.update_qslice())
        grid = self.display_grid
        grid.addWidget(self.make_label("Q-slice along", axis_tip), 0, 0)
        grid.addWidget(self.combo_slice_axis, 0, 1)
        self.unify_widget_heights()


class Volume4DViewerUI(BaseViewerUI):
    """4D dataset (q, Z, rows, columns): 3D view of the volume at a
    q-range, the q-Z image at a scan position, a 2D slice of the volume and
    the pattern at a Z. Big files can be downsampled for viewing."""

    select_positions = True
    positions_labels = Scan2DViewerUI.positions_labels
    positions_tips = Scan2DViewerUI.positions_tips

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.volume = None
        self.volume_key = None
        self.gl_volume = None
        self.gl_plane = None
        self.build_downsample_group()
        glw_img = self.make_plot_widget()
        glw_slice = self.make_plot_widget()
        glw_prof = self.make_plot_widget()
        self.p_img = self.add_plot(glw_img, 0, 0, "", "", "Z (strip index)",
                                   image=True)
        self.p_slice = self.add_plot(glw_slice, 0, 0, "", "Z (strip index)",
                                     "", image=True)
        self.p_prof = self.add_plot(glw_prof, 0, 0, "Diffraction pattern",
                                    "", "Intensity")
        if GL_AVAILABLE:
            top_left = self.make_gl_panel("Volume")
            self.lbl_gl_title.setText("Volume: mean over the q-range (3D). "
                                      "Yellow: plane of the 2D slice.")
            self.p_proj = None
        else:
            top_left = self.make_plot_widget()
            self.p_proj = self.add_plot(top_left, 0, 0, "Volume: mean over "
                                        "Z", "Column index", "Row index",
                                        image=True)
        columns = QSplitter(Qt.Horizontal)
        for top, bottom in ((top_left, glw_slice), (glw_img, glw_prof)):
            column = QSplitter(Qt.Vertical)
            column.addWidget(top)
            column.addWidget(bottom)
            column.setSizes([1000, 1000])
            columns.addWidget(column)
        columns.setSizes([1000, 1000])
        self.layout_main.addWidget(columns, 1)
        self.img_span = self.make_span(self.p_img)
        self.img_row_line = self.make_line(self.p_img, 0,
                                           util.COLORS["lime"])
        self.slice_hline = self.make_line(self.p_slice, 0,
                                          util.COLORS["lime"])
        self.slice_vline = self.make_line(self.p_slice, 90,
                                          util.COLORS["lime"])
        self.prof_line = self.make_line(self.p_prof, 90,
                                        util.COLORS["yellow"], 2)
        for glw, plot in ((glw_img, self.p_img), (glw_slice, self.p_slice),
                          (glw_prof, self.p_prof)):
            self.connect_click(glw, [plot], self.on_click)
        self.slider_x = SliderRow("Q index", "Pixel index of the center of "
                                  "the q-range of the volume.")
        self.slider_row = SliderRow("Row", "Row index (axis 0).")
        self.slider_col = SliderRow("Column", "Column index (axis 1).")
        # Row slider starts with the others; the column label is compact
        self.slider_col.label.setMinimumWidth(0)
        self.slider_z = SliderRow("Z index (strip)")
        self.slider_x.valueChanged.connect(lambda _v: self.update_volume())
        self.slider_row.valueChanged.connect(lambda _v: self.update_image())
        self.slider_col.valueChanged.connect(lambda _v: self.update_image())
        self.slider_z.valueChanged.connect(lambda _v: self.update_z())
        # Data are read from the file when the slider is released
        self.slider_x.released.connect(self.update_volume)
        self.slider_row.released.connect(self.update_image)
        self.slider_col.released.connect(self.update_image)
        self.build_4d_controls()
        self.align_slider_labels()
        self.unify_widget_heights()

    def align_slider_labels(self):
        """Left sliders start together, right after the longest label."""
        labels = [s.label for s in (self.slider_z, self.slider_row,
                                    self.slider_x)]
        for label in labels:
            label.setMinimumWidth(0)
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setFixedWidth(width)
        # "/ max" labels only as wide as the longest one
        maxima = [s.lbl_max for s in (self.slider_z, self.slider_row,
                                      self.slider_col, self.slider_x)]
        width = max(label.sizeHint().width() for label in maxima)
        for label in maxima:
            label.setFixedWidth(width)

    def build_4d_controls(self):
        """Sliders on the left (Z; rows and columns; Q), compact options
        on the right."""
        controls = QHBoxLayout()
        sliders = QVBoxLayout()
        sliders.addWidget(self.slider_z)
        pair = QHBoxLayout()
        pair.addWidget(self.slider_row, 1)
        pair.addWidget(self.slider_col, 1)
        sliders.addLayout(pair)
        sliders.addWidget(self.slider_x)
        controls.addLayout(sliders, 1)
        controls.addSpacing(util.UI_SPACING_M)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        controls.addWidget(separator)
        controls.addSpacing(util.UI_SPACING_M)
        grid = QGridLayout()
        grid.setHorizontalSpacing(util.UI_SPACING_M)
        entry_w = 40

        slice_tip = "Slice of the volume shown below the 3D view."
        self.combo_slice_dir = self.make_combo(util.SLICE_DIRECTIONS,
                                               tip=slice_tip)
        self.combo_slice_dir.currentIndexChanged.connect(
            lambda _i: self.update_slice())
        grid.addWidget(self.make_label("2D slice", slice_tip), 0, 0)
        grid.addWidget(self.combo_slice_dir, 0, 1, 1, 6)

        thres_tip = ("Percentiles of the volume: voxels outside are "
                     "transparent in the 3D view. Press Enter to apply.")
        grid.addWidget(self.make_label("Threshold %", thres_tip), 1, 0)
        self.edit_tmin = self.make_line_edit(0, entry_w, thres_tip)
        self.edit_tmax = self.make_line_edit(100, entry_w, thres_tip)
        for col, edit in ((1, self.edit_tmin), (2, self.edit_tmax)):
            edit.returnPressed.connect(self.apply_thresholds)
            grid.addWidget(edit, 1, col)
        pixel_tip = ("Pixel size along x (columns), y (rows) and z (strips) "
                     "in the same unit, for the real object ratio in the 3D "
                     "view, and in the 2D slice with the 'equal' aspect. "
                     "Press Enter to apply.")
        grid.addWidget(self.make_label("Pixel x, y, z", pixel_tip),
                       1, 3)
        self.edit_pixel = []
        for col in range(4, 7):
            edit = self.make_line_edit(1, entry_w, pixel_tip)
            edit.returnPressed.connect(self.apply_pixel_size)
            grid.addWidget(edit, 1, col)
            self.edit_pixel.append(edit)

        radius_tip = ("Half-width (pixels) of the q-range averaged for the "
                      "volume.")
        grid.addWidget(self.make_label("Radius", radius_tip), 2, 0)
        self.combo_radius = self.make_combo(util.RADII, util.DEFAULT_RADIUS,
                                            radius_tip)
        self.combo_radius.currentTextChanged.connect(
            lambda _t: self.update_qslice())
        grid.addWidget(self.combo_radius, 2, 1, 1, 2)
        buttons = QWidget()
        hbox = QHBoxLayout(buttons)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(util.UI_SPACING_M)
        hbox.addWidget(self.make_button(
            "Save pattern", self.save_pattern,
            "Save the displayed diffraction pattern to a CSV file."), 1)
        hbox.addWidget(self.make_button(
            "Save slice", self.save_qslice,
            "Save the 2D slice of the volume (bottom-left image) to a TIF "
            "file."), 1)
        grid.addWidget(buttons, 2, 3, 1, 4)
        for widget in (self.combo_slice_dir, self.combo_radius):
            widget.setMinimumWidth(0)
            widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(4)
        self.display_grid = grid
        controls.addLayout(grid, 0)
        self.layout_main.addLayout(controls, 0)

    def build_downsample_group(self):
        """Mean binning of the input to a smaller 4D HDF file; two rows
        next to the Data group."""
        group = QGroupBox("Downsample (output)")
        grid = self.make_grid(group)

        def button(text, callback, tip=None):
            btn = self.make_button(text, callback, tip)
            btn.setFixedWidth(120)
            return btn

        bins_tip = ("Number of pixels averaged along q, Z, rows and columns. "
                    "E.g. 2, 2, 1, 1 makes the file 4 times smaller.")
        grid.addWidget(self.make_label("Bins q, Z, r, c", bins_tip), 0, 0)
        bins = QWidget()
        hbox = QHBoxLayout(bins)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(util.UI_SPACING_M)
        self.spin_bins = []
        for value in util.DEFAULT_BINS:
            spin = QSpinBox()
            spin.setRange(1, 64)
            spin.setValue(value)
            spin.setToolTip(bins_tip)
            spin.setFixedWidth(util.SMALL_ENTRY_W - 15)
            hbox.addWidget(spin)
            self.spin_bins.append(spin)
        grid.addWidget(bins, 0, 1)
        chunk_tip = ("Chunk shape (q, Z, rows, columns) of the output. "
                     "Empty: no chunking.")
        grid.addWidget(self.make_label("Chunks", chunk_tip), 0, 2)
        self.edit_ds_chunks = self.make_line_edit(
            ", ".join(str(c) for c in util.DEFAULT_4D_CHUNKS),
            util.MEDIUM_ENTRY_W, chunk_tip)
        grid.addWidget(self.edit_ds_chunks, 0, 3)
        self.check_ds_load = QCheckBox("Load output")
        self.check_ds_load.setChecked(True)
        self.check_ds_load.setToolTip("View the downsampled file when done.")
        grid.addWidget(self.check_ds_load, 0, 4)
        btn_run = button("Downsample", self.run_downsample,
                         "Mean binning of the input dataset, block by "
                         "block.")
        grid.addWidget(btn_run, 0, 5)

        out_tip = (f"Output HDF file. Empty: the input file name with "
                   f"'{util.DOWNSAMPLE_SUFFIX}'.")
        grid.addWidget(self.make_label("Output file", out_tip), 1, 0)
        self.edit_ds_output = self.make_line_edit(tip=out_tip)
        grid.addWidget(self.edit_ds_output, 1, 1)
        key_tip = "Dataset of the output file."
        grid.addWidget(self.make_label("Key", key_tip), 1, 2)
        self.edit_ds_key = self.make_line_edit(util.HDF_DATA_KEY,
                                               tip=key_tip)
        grid.addWidget(self.edit_ds_key, 1, 3, 1, 2)
        grid.addWidget(button("Browse...", self.browse_ds_output), 1, 5)
        grid.setColumnStretch(1, 1)

        # Data and Downsample groups side by side
        data_group = self.layout_main.itemAt(0).widget()
        self.layout_main.removeWidget(data_group)
        # Narrower buttons and fields so both groups fit in one row
        for widget in (data_group, group):
            for btn in widget.findChildren(QPushButton):
                btn.setFixedWidth(util.MEDIUM_ENTRY_W)
                if btn.text() == "Load metadata":
                    btn.setText("Load")
        self.combo_hdf_key.setMinimumWidth(util.MEDIUM_ENTRY_W)
        for combo in self.position_combos:
            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(4)
        self.edit_step.setFixedWidth(util.MEDIUM_ENTRY_W)
        top = QHBoxLayout()
        top.setSpacing(util.UI_SPACING_L)
        top.addWidget(data_group, 3)
        top.addWidget(group, 2)
        self.layout_main.insertLayout(0, top, 0)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class EqualWidthTabBar(QTabBar):
    """Tab bar giving every tab the width of the widest one, whichever tab
    is selected (the selected tab has a bold font)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tab_width = 0

    def bold_extra(self, index):
        """Extra width of a tab label in bold, as when it's selected."""
        if index == self.currentIndex():
            return 0
        bold = QFont(self.font())
        bold.setWeight(QFont.DemiBold)
        text = self.tabText(index)
        return max(QFontMetrics(bold).horizontalAdvance(text)
                   - QFontMetrics(self.font()).horizontalAdvance(text), 0)

    def tabSizeHint(self, index):
        size = super().tabSizeHint(index)
        widths = [super(EqualWidthTabBar, self).tabSizeHint(i).width()
                  + self.bold_extra(i) for i in range(self.count())]
        # Never shrink, so the tabs keep their positions
        self.tab_width = max([self.tab_width] + widths)
        size.setWidth(self.tab_width)
        return size

    def minimumTabSizeHint(self, index):
        return self.tabSizeHint(index)

    def tabInserted(self, index):
        self.tab_width = 0
        super().tabInserted(index)

    def tabRemoved(self, index):
        self.tab_width = 0
        super().tabRemoved(index)

    def changeEvent(self, event):
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self.tab_width = 0
        super().changeEvent(event)


class MainWindow(QMainWindow):
    """Main window: a tab per tool, a shared status bar."""

    def __init__(self, tabs):
        super().__init__()
        self.setWindowTitle(util.APP_NAME)
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        self.setGeometry(calculate_geometry(util.MAIN_WIN_RATIO))
        self.close_handler = None
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(util.UI_MARGIN_M, util.UI_MARGIN_M,
                                  util.UI_MARGIN_M, util.UI_MARGIN_S)
        layout.setSpacing(util.UI_SPACING_ML)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabBar(EqualWidthTabBar())
        for widget, title in tabs:
            self.tab_widget.addTab(widget, title)
        layout.addWidget(self.tab_widget, 1)
        layout.addWidget(self.build_display_bar(), 0)
        self.setCentralWidget(central)
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

    def build_display_bar(self):
        """Colormap, scale, aspect and contrast of the images of all tabs.
        Combo boxes apply on change; percentiles apply on Enter."""
        bar = QFrame()
        bar.setObjectName("DisplayBar")
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(util.UI_MARGIN_L, util.UI_MARGIN_S,
                                util.UI_MARGIN_L, util.UI_MARGIN_S)
        hbox.setSpacing(util.UI_SPACING_L)
        title = QLabel("Image display")
        title.setObjectName("BarTitle")
        hbox.addWidget(title)
        self.combo_cmap = BaseTab.make_combo(util.IMAGE_CMAPS,
                                             display["cmap"],
                                             "Colormap of the images.")
        self.combo_scale = BaseTab.make_combo(
            util.IMAGE_SCALES, display["scale"],
            "Intensity scale of the images.")
        self.combo_aspect = BaseTab.make_combo(
            util.IMAGE_ASPECTS, display["aspect"],
            "auto: stretch the images to fill the plots.\n"
            "equal: square pixels, keep the original image ratio.")
        self.edit_pmin = BaseTab.make_line_edit(
            util.IMAGE_PERCENTILES[0], util.MEDIUM_ENTRY_W,
            "Lower percentile of the intensity for the contrast. Press "
            "Enter or leave the field to apply.")
        self.edit_pmax = BaseTab.make_line_edit(
            util.IMAGE_PERCENTILES[1], util.MEDIUM_ENTRY_W,
            "Upper percentile of the intensity for the contrast. Press "
            "Enter or leave the field to apply.")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setFixedSize(util.MEDIUM_ENTRY_W, util.BTN_H)
        self.btn_reset.setToolTip(
            "Reset the contrast, and zoom out the plots of the current tab.")
        for widget in (self.combo_cmap, self.combo_scale, self.combo_aspect,
                       self.edit_pmin, self.edit_pmax, self.btn_reset):
            widget.setFixedWidth(util.MEDIUM_ENTRY_W)
        groups = [[("Colormap", self.combo_cmap),
                   ("Scale", self.combo_scale),
                   ("Aspect", self.combo_aspect)],
                  [("Contrast  Min %", self.edit_pmin),
                   ("Max %", self.edit_pmax)]]
        for group in groups:
            separator = QFrame()
            separator.setObjectName("BarSeparator")
            separator.setFixedSize(1, util.BTN_H - 6)
            hbox.addSpacing(util.UI_SPACING_M)
            hbox.addWidget(separator)
            hbox.addSpacing(util.UI_SPACING_M)
            for label, widget in group:
                hbox.addWidget(BaseTab.make_label(label, widget.toolTip()))
                widget.setFixedHeight(util.BTN_H)
                hbox.addWidget(widget)
                hbox.addSpacing(util.UI_SPACING_M)
        hbox.addWidget(self.btn_reset)
        hbox.addStretch(1)
        return bar

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
