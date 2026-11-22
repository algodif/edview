"""
Utilities module for EDView.
Contains constants, config handling, data loading, data correction and
conversion methods without GUI dependencies.
"""

import os
import re
import json
import glob
import platform
import warnings
from pathlib import Path
from collections import OrderedDict
import numpy as np
import h5py
from PIL import Image

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.post.postprocessing as post
import algodif.util.utility as autil

APP_NAME = "EDView - EDDT data viewer"
FONT_SIZE = 14
MAIN_WIN_RATIO = 0.9
# Size of child windows relative to the main window
CHILD_WIN_RATIO = 0.8
MAX_WIN_W = 1920
MAX_WIN_H = 1080
PLOT_FONT_SIZE = 11
PLOT_TITLE_SIZE = 11
# Width of the save buttons of the viewers
SAVE_BTN_W = 120
# Max characters per line of tooltips
TOOLTIP_WIDTH = 100

UI_MARGIN_S = 4
UI_MARGIN_M = 6
UI_MARGIN_L = 10
UI_SPACING_S = 2
UI_SPACING_M = 4
UI_SPACING_ML = 6
UI_SPACING_L = 8
BTN_H = 28
LEFT_PANEL_W = 360
FORM_LABEL_W = 110
SMALL_ENTRY_W = 70
MEDIUM_ENTRY_W = 100
TREE_MIN_W = 150
INPUT_LIST_H = 120
FILE_SELECT_DEBOUNCE_MS = 120
SLICE_UPDATE_DELAY_MS = 30
WORKER_POLL_MS = 200
# Max time (s) to wait for running calculations to stop when quitting
EXIT_WAIT_S = 5.0
# Size (x: q, y: images, z: intensity) of the box of the 3D waterfall
GL_BOX = (10.0, 10.0, 5.0)
# Max points per line of the 3D waterfall (lines are decimated above)
GL_MAX_POINTS = 2000
# Max distance (pixels) of the mouse to a point to show its values
GL_HOVER_PX = 12
# Number of full images kept in memory by the 2D-scan viewer
# Viewers of a folder of tif files (Data browser menus)
TIF_VIEWERS = ("EDDT viewer", "Multiple-lines viewer", "2D-scan viewer")
# Viewers needing tif files named '..._<row>_<col>.tif'
TIF_2D_VIEWERS = ("2D-scan viewer",)
IMAGE_CACHE_SIZE = 20
LINE_CACHE_MB = 1024  # images along a q-slice line kept in memory
# 1D-scan viewers: HDF scans up to this size (float32) are loaded in
# memory, bigger ones are read on demand
SCAN_IN_MEMORY_MB = 2048
# Images read per step when loading a scan (the loading can be cancelled
# between steps)
SCAN_READ_MB = 256
# 4D viewer: volumes (q-range means) kept in memory, voxels drawn in 3D
VOLUME_CACHE_SIZE = 4
GL_MAX_VOXELS = 4_000_000
GL_VOLUME_ALPHA = 0.5
SLICE_DIRECTIONS = ("Fixed row: columns vs Z", "Fixed column: rows vs Z",
                    "Fixed Z: rows vs columns")
# Binning of (q, Z, rows, columns) for downsampling
DEFAULT_BINS = (2, 2, 1, 1)
DOWNSAMPLE_SUFFIX = "_dsp"

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
HDF_EXT = (".hdf", ".h5", ".hdf5", ".nxs")
TEXT_EXT = (".json", ".txt", ".csv", ".xy", ".log", ".py", ".yaml", ".yml",
            ".md", ".dat")
TIF_EXT = (".tif", ".tiff")

IMAGE_CMAPS = ("inferno", "viridis", "magma", "plasma", "gray")
IMAGE_SCALES = ("log", "linear")
# auto: fill the plot; equal: square image pixels (keep the image ratio)
IMAGE_ASPECTS = ("auto", "equal")
IMAGE_PERCENTILES = (0, 100)  # Default contrast: full range
RADII = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
DEFAULT_RADIUS = 9
COLORS = {"blue": "#1f77b4", "red": "#d62728", "green": "#2ca02c",
          "lime": "#39ff14", "yellow": "#ffd400"}

Q_LABEL = "Q (1/\u00c5)"
Q_STEP_KEY = "q_step"
# Folder (in each corrected scan) with the q-maps and how they were made
MAP_INFO_FOLDER = "map_info"
MAP_INFO_FILE = "map_info.json"
RAW_Q_MAP_FILE = "raw_q_map.tif"
CORRECTED_Q_MAP_FILE = "correct_q_map.tif"
# Output step of corrected data: statistic of the pixel spacing of the
# q-map computed from the coefficients, or from a q-max or step value
OUTPUT_STEP_OPTIONS = ("mean", "median", "max", "q-max", "q-step")
OUTPUT_FORMATS = ("tif", "hdf", "xy/csv")
# x-axis of xy/csv files: q, or the equivalent 2theta (algodif q_to_two_theta)
XY_X_AXES = ("q", "two-theta")
DEFAULT_MAX_TWO_THETA = 90.0
# 4D datasets of 2D scans, as in algodif convert_tif_to_4D_hdf: shape
# (width, height, nrow, ncol), i.e. image[strip, pixel] at
# data[pixel, strip, row, column]
LAYOUT_ATTR = "axes"
LAYOUT_4D = "width,height,row,column"
DEFAULT_4D_CHUNKS = (64, 64, 8, 8)
# Acquisition order of the images of a 2D scan (first image at (0, 0))
SCAN_ORDERS = ("row", "column", "snake_row", "snake_column")
HDF_DATA_KEY = "entry/data/data"
HDF_RAW_MAP_KEY = "entry/q_map/raw_map"
HDF_CORRECTED_MAP_KEY = "entry/q_map/corrected_map"
HDF_MAP_INFO_KEY = "entry/map_info"
# q-map key of files written by older versions
HDF_LEGACY_MAP_KEY = "entry/map"
HDF_POSITIONS_KEY = "entry/positions"
CONVERSIONS = {
    "TIF folder -> HDF (3D stack)": "tif_to_hdf",
    "HDF (3D stack) -> TIF folder": "hdf_to_tif",
    "TIF folder (2D scan) -> HDF (4D)": "tif2d_to_hdf4d",
    "TIF folder / HDF -> xy/csv": "to_xy",
}

# Background removal of corrected images, applied to each strip (row).
# Method -> (algodif.prep.preprocessing function,
#            [(key, label, default, dtype, tooltip), ...])
BCK_METHODS = {
    "Ranked descent": ("remove_image_background_ranked_descent", [
        ("width", "Min width", 15, int, "Minimum width of a peak (pixel)."),
        ("ratio", "Ratio", 0.1, float,
         "Minimum ratio between the peak height above its baseline and the "
         "mean signal value around the peak."),
        ("sigma", "Sigma", 5, float,
         "Gaussian smoothing used during peak detection.")]),
    "ALS": ("remove_image_background_based_als", [
        ("ld", "Smoothness (ld)", 1e6, float,
         "Smoothing parameter. Higher values yield smoother baselines."),
        ("weight", "Weight", 0.05, float,
         "Asymmetry parameter (0 < weight < 1)."),
        ("num_iter", "Iterations", 10, int, "Number of iterations.")]),
    "arPLS": ("remove_image_background_based_arpls", [
        ("ld", "Smoothness (ld)", 1e6, float,
         "Smoothing parameter. Higher values yield smoother baselines."),
        ("weight", "Weight", 0.05, float,
         "Weight deviation (0 < weight < 1)."),
        ("num_iter", "Iterations", 5, int,
         "Maximum number of iterations.")]),
}
DEFAULT_BCK_METHOD = "Ranked descent"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_config_path():
    """Get path to save a config file depending on the OS system."""
    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        return os.path.join(home, "AppData", "Roaming", "EDView",
                            "edview_config.json")
    if platform.system() == "Darwin":
        return os.path.join(home, "Library", "Application Support", "EDView",
                            "edview_config.json")
    return os.path.join(home, ".edview", "edview_config.json")


def save_config(data):
    """Save data (dictionary) to the config file (json format)."""
    config_path = get_config_path()
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)


def suppress_warnings():
    """Hide harmless warnings of peak fitting and plotting."""
    warnings.filterwarnings("ignore", message="Covariance of the parameters "
                                              "could not be estimated")
    # pyqtgraph computes slopes of curves with repeated x-values
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            module="pyqtgraph")


def load_config():
    """Load the config file. Return None if missing or unreadable."""
    try:
        with open(get_config_path(), "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def update_config(**kwargs):
    """Update some entries of the config file; failures are ignored."""
    config = load_config() or {}
    config.update(kwargs)
    try:
        save_config(config)
    except OSError:
        pass


def get_last_folder(key):
    """Return the last folder saved under key, or "" if not available."""
    config = load_config() or {}
    folder = config.get("last_folders", {}).get(key, "")
    return folder if folder and os.path.isdir(folder) else ""


def set_last_folder(key, path):
    """Save the folder of path (a file or a folder) under key."""
    if not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    config = load_config() or {}
    config.setdefault("last_folders", {})[key] = os.path.abspath(folder)
    try:
        save_config(config)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def to_number(text, name, dtype=float, allow_none=False):
    text = str(text).strip()
    if allow_none and text.lower() in ("", "none"):
        return None
    try:
        return int(float(text)) if dtype is int else dtype(text)
    except ValueError:
        raise ValueError(f"Invalid value for '{name}': '{text}'")


def parse_indices(text, num):
    """Parse e.g. '1-5, 8' into a sorted list of integers (inclusive)."""
    values = set()
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token[1:]:
                start, stop = (int(v) for v in token.split("-", 1))
            else:
                start = stop = int(token)
        except ValueError:
            raise ValueError(f"Invalid range: '{token}'. Use e.g. '1-5, 8'")
        values.update(range(min(start, stop), max(start, stop) + 1))
    values = sorted(values)
    if num is not None and values and (values[0] < 0 or values[-1] >= num):
        raise ValueError(f"Indices must be in [0, {num - 1}]")
    return values


def natural_key(text):
    """Sort key: 'img_2' before 'img_10'."""
    return [int(v) if v.isdigit() else v.lower()
            for v in re.split(r"(\d+)", str(text))]


def find_files(folder, exts=TIF_EXT):
    """Files of the given extensions in a folder, naturally sorted."""
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, name) for name in os.listdir(folder)
             if name.lower().endswith(exts)]
    return sorted(files, key=natural_key)


def is_text_file(file_path, num_bytes=1024):
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(num_bytes)
        return b"\x00" not in chunk
    except OSError:
        return False


def format_bytes(num):
    size = float(max(num, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else \
                f"{size:.1f} {unit}"
        size /= 1024.0


def walk_hdf(group, prefix="", seen=None):
    """Yield (path, obj, link) of all members of an HDF group, following
    soft and external links. obj is None for broken links."""
    if seen is None:
        seen = {(group.file.filename, hash(group.id))}
    for name in group:
        path = f"{prefix}/{name}".strip("/")
        link = group.get(name, getlink=True)
        try:
            obj = group[name]
        except (KeyError, OSError, ValueError):
            yield path, None, link
            continue
        yield path, obj, link
        if isinstance(obj, h5py.Group):
            # Linked groups may point back to a parent; the hash of an
            # object id identifies the object, whatever the path used
            key = (obj.file.filename, hash(obj.id))
            if key not in seen:
                seen.add(key)
                yield from walk_hdf(obj, path, seen)


def describe_link(link):
    """Text of a soft or external HDF link, '' for hard links."""
    if isinstance(link, h5py.ExternalLink):
        return f"{link.filename}::{link.path}"
    if isinstance(link, h5py.SoftLink):
        return link.path
    return ""


def hdf_array_keys(file_path, min_ndim=1, max_ndim=None):
    """Keys and shapes of numeric datasets with min_ndim to max_ndim
    (None: no limit) dims (external links included), sorted by number of
    dimensions (descending)."""
    out = []
    with h5py.File(file_path, "r") as file:
        for path, obj, _ in walk_hdf(file):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= min_ndim \
                    and (max_ndim is None or obj.ndim <= max_ndim) \
                    and np.issubdtype(obj.dtype, np.number):
                out.append((path, obj.shape))
    out.sort(key=lambda item: len(item[1]), reverse=True)
    return out


def viewers_for_hdf(file_path, viewer_ndims):
    """Names of the viewers ({name: (min ndim, max ndim)}) able to show at
    least one dataset of an HDF file."""
    ndims = {len(shape) for _, shape in hdf_array_keys(file_path, 2)}
    return [name for name, (low, high) in viewer_ndims.items()
            if any(low <= num <= high for num in ndims)]


class WrongDimensionError(ValueError):
    """HDF data a viewer can't show (wrong number of dimensions); other
    viewers may show the file at path."""

    def __init__(self, message, path):
        super().__init__(message)
        self.path = path


def load_hdf_slice(file_path, key, axes=(), indices=()):
    """1D/2D data of an HDF dataset: the dataset itself (1D/2D) or the
    slice at 'indices' along 'axes'. Other leading axes are fixed at 0 if
    more than 2 axes remain."""
    with h5py.File(file_path, "r") as file:
        dataset = file[key]
        ndim = dataset.ndim
        if ndim <= 2:
            return np.asarray(dataset[()], dtype=np.float32)
        selection = [slice(None)] * ndim
        for axis, index in zip(axes, indices):
            selection[axis] = int(index)
        free = [i for i in range(ndim) if isinstance(selection[i], slice)]
        for i in free[:-2]:
            selection[i] = 0
        return np.asarray(dataset[tuple(selection)], dtype=np.float32)


def load_preview_image(file_path):
    """2D float image of an image file (color channels averaged)."""
    with Image.open(file_path) as img:
        mat = np.asarray(img, dtype=np.float32)
    if mat.ndim == 3:
        mat = np.mean(mat, axis=2)
    return mat


def get_levels(data, pmin=0, pmax=100):
    """Color limits of an image from intensity percentiles."""
    if pmin <= 0 and pmax >= 100:
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    else:
        vmin, vmax = (float(v) for v in
                      np.nanpercentile(data, (pmin, pmax)))
    if not vmax > vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def hdf_nodes(file_path):
    """[(path, kind, shape, dtype, link), ...] of all groups and datasets
    of an HDF file (soft and external links followed), parents before
    children. kind: 'group', 'dataset' or 'broken' (unresolved link)."""
    nodes = []
    with h5py.File(file_path, "r") as file:
        for path, obj, link in walk_hdf(file):
            if isinstance(obj, h5py.Dataset):
                nodes.append((path, "dataset", obj.shape, str(obj.dtype),
                              describe_link(link)))
            elif isinstance(obj, h5py.Group):
                nodes.append((path, "group", None, "", describe_link(link)))
            else:
                nodes.append((path, "broken", None, "",
                              describe_link(link)))
    return nodes


def format_hdf_value(value):
    """Readable text of an HDF value or attribute (bytes decoded)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "O"):
            items = [v.decode("utf-8", errors="replace")
                     if isinstance(v, bytes) else str(v)
                     for v in value.ravel()[:200]]
            return "[" + ", ".join(items) + "]"
        return np.array2string(value, threshold=200, max_line_width=100)
    return str(value)


def hdf_item_info(file_path, path, max_values=200):
    """Text describing a group or dataset of an HDF file: shape, type,
    attributes, and the value of small datasets."""
    with h5py.File(file_path, "r") as file:
        lines = [f"Path: /{path.strip('/')}"]
        if path:
            parent, _, name = path.rpartition("/")
            link = file[parent or "/"].get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                lines += [f"External link: {link.filename}",
                          f"Target path: {link.path}"]
            elif isinstance(link, h5py.SoftLink):
                lines.append(f"Soft link to: {link.path}")
        try:
            obj = file[path or "/"]
        except (KeyError, OSError, ValueError) as error:
            lines.append(f"Broken link: {error}")
            return "\n".join(lines)
        if isinstance(obj, h5py.Dataset):
            lines += ["Type: Dataset", f"Shape: {obj.shape}",
                      f"Data type: {obj.dtype}"]
            try:
                lines.append(f"Size: {format_bytes(obj.nbytes)}")
                if obj.chunks:
                    lines.append(f"Chunks: {obj.chunks}")
                if obj.compression:
                    lines.append(f"Compression: {obj.compression}")
                if obj.size <= max_values:
                    lines += ["", "Value:", format_hdf_value(obj[()])]
            except (TypeError, ValueError, OSError) as error:
                lines.append(f"Can't read the value: {error}")
        else:
            lines += ["Type: Group", f"Members: {len(obj)}"]
        if len(obj.attrs):
            lines += ["", "Attributes:"]
            for name in obj.attrs:
                try:
                    value = format_hdf_value(obj.attrs[name])
                except (TypeError, ValueError, OSError) as error:
                    value = f"<unreadable: {error}>"
                lines.append(f"    {name}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def find_metadata_file(path):
    """JSON metadata next to the data: in the folder (or the folder of the
    file), then in its parent folder."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    for candidate in (folder, folder.parent):
        files = sorted(candidate.glob("*.json"))
        if files:
            return str(files[0])
    return ""


def find_map_info_folder(path):
    """map_info folder of corrected data (in the folder of the data or its
    parent), or None."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    for candidate in (folder, folder.parent):
        if (candidate / MAP_INFO_FOLDER / MAP_INFO_FILE).is_file():
            return candidate / MAP_INFO_FOLDER
    return None


def find_map_info(path):
    """Content of the map_info JSON of corrected data, or {}."""
    folder = find_map_info_folder(path)
    if folder is None:
        return {}
    try:
        with open(folder / MAP_INFO_FILE, "r") as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def find_value(data, key):
    """First value of a key in a nested dictionary, or None."""
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = find_value(value, key)
            if found is not None:
                return found
    return None


def get_motor_positions(metadata, num=None):
    """Names and positions of the scanned motors (lists of numbers)."""
    if not metadata:
        return []
    keys = losa.find_key_of_list_json(metadata)
    motors = []
    for key in dict.fromkeys(keys):
        # Exact key: get_list_by_key_json matches substrings
        values = find_value(metadata, key)
        if isinstance(values, list):
            try:
                motors.append((key, [float(v) for v in values]))
            except (TypeError, ValueError):
                continue
    if num is not None:
        motors = [m for m in motors if len(m[1]) == num] or motors
    return motors


# ---------------------------------------------------------------------------
# Datasets for the viewers
# ---------------------------------------------------------------------------

class HdfStack:
    """3D HDF dataset (images, strips, pixels) read on demand as float32;
    the file stays open until close()."""

    def __init__(self, path, key):
        self.file = h5py.File(path, "r")
        self.dataset = self.file[key]
        self.shape = self.dataset.shape
        self.ndim = self.dataset.ndim

    def __getitem__(self, index):
        return np.asarray(self.dataset[index], dtype=np.float32)

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class ScanDataset:
    """Stack of processed images (images, strips, pixels) of a scan, with
    the q-values of the pixels (pixel indices if the q-step is unknown) and
    the motor positions. data: array in memory, or HdfStack read on
    demand (lazy)."""

    def __init__(self, data, x_step, motor_name="", positions=None, name="",
                 source="", files=None):
        self.lazy = isinstance(data, HdfStack)
        if self.lazy:
            self.data = data
        else:
            self.data = np.asarray(data, dtype=np.float32)
            if self.data.ndim == 2:
                self.data = self.data[np.newaxis]
        self.depth, self.height, self.width = self.data.shape
        self.nbytes = self.depth * self.height * self.width * 4
        self.set_x_step(x_step)
        self.set_positions(motor_name, positions)
        self.name = name
        self.source = source
        self.files = files or []
        self.position_note = ""
        self._images = OrderedDict()
        self._row_key = self._rows = None
        self._qslice_key = self._qslice = None

    def set_x_step(self, x_step):
        """q-step of the pixels; None or 0: pixel indices."""
        self.calibrated = bool(x_step and x_step > 0)
        self.x_step = float(x_step) if self.calibrated else 1.0
        self.x_list = np.arange(self.width) * self.x_step

    def set_positions(self, motor_name, positions):
        """Positions of the images; the image index if not one per image."""
        self.motor_name = motor_name or "Image"
        if positions is None or len(positions) != self.depth:
            positions = list(range(self.depth))
            self.motor_name = "Image"
        self.positions = list(positions)

    @property
    def x_max(self):
        return float(self.x_list[-1]) if self.width > 1 else self.x_step

    def x_index(self, x_value):
        return int(np.clip(round(x_value / self.x_step), 0, self.width - 1))

    def image(self, index):
        """Image (strips, pixels) at an image index."""
        if not self.lazy:
            return self.data[index]
        img = self._images.get(index)
        if img is None:
            img = self.data[index]
            self._images[index] = img
            if len(self._images) > IMAGE_CACHE_SIZE:
                self._images.popitem(last=False)
        else:
            self._images.move_to_end(index)
        return img

    def row_patterns(self, row):
        """Patterns of all images at a strip: shape (images, pixels)."""
        if self._row_key != row:
            self._rows = self.data[:, row, :]
            self._row_key = row
        return self._rows

    def pattern(self, index, row):
        """Pattern of an image at a strip."""
        if self._row_key == row:
            return self._rows[index]
        if index in self._images:
            return self._images[index][row]
        return self.data[index, row]

    def q_slice(self, index, radius):
        """Mean over the x-columns around index: shape (images, strips)."""
        key = (index, radius)
        if self._qslice_key != key:
            start = max(index - radius, 0)
            stop = min(index + radius + 1, self.width)
            self._qslice = np.mean(self.data[:, :, start:stop], axis=2)
            self._qslice_key = key
        return self._qslice

    def close(self):
        if self.lazy:
            self.data.close()


def step_from_q_map_tif(path):
    """q-step from map_info/correct_q_map.tif in the folder of the data or
    its parent, or None."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    for candidate in (folder, folder.parent):
        file = candidate / MAP_INFO_FOLDER / CORRECTED_Q_MAP_FILE
        if file.is_file():
            try:
                row = np.asarray(losa.load_image(str(file)),
                                 dtype=float)[0]
            except (OSError, ValueError):
                return None
            if row.size > 1 and row[1] > row[0]:
                return float(row[1] - row[0])
    return None


def get_x_step(metadata, hdf_path=None, hdf_key=None, data_path=None):
    """q-step of the pixels from the map_info of corrected data (JSON or
    correct_q_map.tif), the q-map saved in the HDF file or the metadata.
    Return None if not found."""
    if data_path:
        value = find_map_info(data_path).get(Q_STEP_KEY)
        if value:
            return float(value)
        value = step_from_q_map_tif(data_path)
        if value:
            return value
    if hdf_path:
        with h5py.File(hdf_path, "r") as file:
            key = f"{HDF_MAP_INFO_KEY}/{Q_STEP_KEY}"
            if key in file:
                return float(file[key][()])
            for key in (HDF_CORRECTED_MAP_KEY, HDF_LEGACY_MAP_KEY):
                if key in file:
                    row = np.asarray(file[key][0], dtype=float)
                    if row.size > 1:
                        return float(row[1] - row[0])
            if hdf_key and hdf_key in file:
                for name in (Q_STEP_KEY, "x_step"):
                    if name in file[hdf_key].attrs:
                        return float(file[hdf_key].attrs[name])
    for name in (Q_STEP_KEY, "x_step"):
        value = find_value(metadata, name)
        if value is not None:
            return float(value)
    return None


def is_cancelled(cancel_event):
    return cancel_event is not None and cancel_event.is_set()


def check_scan_input(path, hdf_key=""):
    """Check a 1D scan (folder of tif files, or 2D/3D HDF dataset) from
    the shape of its HDF dataset, without reading the data. Raise
    WrongDimensionError if the HDF data have other dimensions (other
    viewers may show them)."""
    path = str(path).strip()
    if not path or not os.path.exists(path):
        raise ValueError(f"No such file or folder: {path}")
    if not (os.path.isfile(path) and path.lower().endswith(HDF_EXT)):
        return
    if not hdf_key:
        if not hdf_array_keys(path, 2, 3):
            raise WrongDimensionError(f"No 2D or 3D dataset (images, "
                                      f"strips, pixels) in {path}", path)
        raise ValueError("Please select the HDF key of the dataset!")
    with h5py.File(path, "r") as file:
        obj = file.get(hdf_key)
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(f"No dataset '{hdf_key}' in {path}")
        if not np.issubdtype(obj.dtype, np.number):
            raise ValueError(f"Dataset '{hdf_key}' isn't numeric "
                             f"({obj.dtype})")
        ndim, shape = obj.ndim, obj.shape
    if ndim not in (2, 3):
        hint = " A 4D dataset is a 2D scan." if ndim == 4 else ""
        raise WrongDimensionError(
            f"Dataset '{hdf_key}' is {ndim}D {shape}: this viewer needs a "
            f"2D or 3D dataset (images, strips, pixels).{hint}", path)


def read_stack(dataset, cancel_event=None, progress=None):
    """2D/3D HDF dataset as a float32 array, read in blocks of images of
    SCAN_READ_MB so it can be cancelled. Return None if cancelled."""
    if dataset.ndim == 2:
        return np.asarray(dataset[()], dtype=np.float32)
    depth = dataset.shape[0]
    image_nbytes = max(int(np.prod(dataset.shape[1:])) * 4, 1)
    step = max(1, SCAN_READ_MB * 2 ** 20 // image_nbytes)
    out = np.empty(dataset.shape, dtype=np.float32)
    for start in range(0, depth, step):
        if is_cancelled(cancel_event):
            return None
        if progress is not None:
            progress(f"Reading images {start}/{depth}...")
        stop = min(start + step, depth)
        out[start:stop] = dataset[start:stop]
    return out


def read_tif_stack(files, cancel_event=None, progress=None):
    """Images of tif files as a float32 array (images, strips, pixels).
    Return None if cancelled."""
    out = None
    for i, file in enumerate(files):
        if is_cancelled(cancel_event):
            return None
        if progress is not None and i % 50 == 0:
            progress(f"Reading tif files {i}/{len(files)}...")
        img = losa.load_image(file)
        if out is None:
            out = np.empty((len(files),) + img.shape, dtype=np.float32)
        elif img.shape != out.shape[1:]:
            raise ValueError(f"All images must have the same shape: "
                             f"{file} is {img.shape}, the first one is "
                             f"{out.shape[1:]}")
        out[i] = img
    return out


def load_scan(path, hdf_key="", metadata_path="", x_step=None,
              motor_key="", max_memory_mb=SCAN_IN_MEMORY_MB,
              cancel_event=None, progress=None):
    """Load a processed scan from a folder of tif files or an HDF file.
    metadata_path: JSON or nxs/HDF file. motor_key: list name (JSON) or key
    of a 1D dataset (nxs/HDF) with the positions; used only if it has one
    value per image (the position_note of the dataset says otherwise).
    A 3D HDF dataset bigger than max_memory_mb (as float32) is read on
    demand (call close() on the dataset). Return None if cancelled."""
    check_scan_input(path, hdf_key)
    path = str(path).strip()
    kind, source = positions_source(path, metadata_path)
    json_path = source if kind == "json" else \
        ("" if metadata_path else find_metadata_file(path))
    metadata = losa.load_json(json_path) if json_path else {}
    files = []
    is_hdf = os.path.isfile(path) and path.lower().endswith(HDF_EXT)
    default_motors = []
    if is_hdf:
        with h5py.File(path, "r") as file:
            dataset = file[hdf_key]
            if HDF_POSITIONS_KEY in file and dataset.ndim == 3:
                group = file[HDF_POSITIONS_KEY]
                names = list(group.keys())
                if names:
                    default_motors = [(names[0],
                                       list(group[names[0]][()]))]
            nbytes = int(np.prod(dataset.shape)) * 4
            lazy = dataset.ndim == 3 and nbytes > max_memory_mb * 2 ** 20
            if not lazy:
                data = read_stack(dataset, cancel_event, progress)
                if data is None:
                    return None
        if lazy:
            data = HdfStack(path, hdf_key)
    else:
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        files = find_files(folder)
        if not files:
            raise ValueError(f"No tif files in: {folder}")
        data = read_tif_stack(files, cancel_event, progress)
        if data is None:
            return None
    try:
        if x_step is None:
            x_step = get_x_step(metadata, path if is_hdf else None,
                                hdf_key, path)
        depth = data.shape[0] if data.ndim == 3 else 1
        motor_name, positions, note = resolve_positions(
            path, metadata_path, motor_key, depth, default_motors)
    except Exception:
        if isinstance(data, HdfStack):
            data.close()
        raise
    name = Path(path).stem if is_hdf else Path(path).name
    if name.lower().startswith("tifs") and not is_hdf:
        name = Path(path).parent.name
    dataset = ScanDataset(data, x_step, motor_name, positions, name, path,
                          files)
    dataset.position_note = note
    return dataset


def resolve_positions(path, metadata_path, motor_key, depth,
                      default_motors=(), unit="images"):
    """(motor name, positions or None, note) of the images of a scan.
    Without motor_key: the first array with one value per image. unit:
    what the values belong to ('images', 'rows' or 'columns')."""
    kind, source = positions_source(path, metadata_path)
    index = f"{unit[:-1]} index"
    if motor_key and not (source and os.path.isfile(source)):
        return "", None, (f"No metadata file for '{motor_key}': the "
                          f"{index} is used.")
    if motor_key:
        if kind == "hdf":
            values = read_hdf_1d(source, motor_key)
        else:
            values = json_positions(source, motor_key)
        if values is None:
            return "", None, (f"No 1D array '{motor_key}' in {source}: "
                               f"the {index} is used.")
        if len(values) != depth:
            return "", None, (f"'{motor_key}' has {len(values)} values "
                               f"for {depth} {unit}: the {index} is "
                               f"used.")
        return position_label(motor_key), list(values), ""
    for name, values in default_motors:
        if len(values) == depth:
            return name, list(values), ""
    if kind == "json" and source and os.path.isfile(source):
        for name, values in json_number_lists(source).items():
            if len(values) == depth:
                return name, values, ""
    return "", None, ""


def resolve_positions_2d(path, metadata_path, keys, nrow, ncol):
    """[(name, positions or None, note), ...] of the rows (axis 0) and the
    columns (axis 1) of a 2D scan. A key selects a 1D array of the
    metadata (see resolve_positions). Without a key, the first list of the
    JSON with nrow / ncol values is used, columns first (the fast axis is
    listed first), never the same list twice."""
    sizes, units = (nrow, ncol), ("rows", "columns")
    out = [resolve_positions(path, metadata_path, key, size, unit=unit)
           if key else None for key, size, unit in zip(keys, sizes, units)]
    kind, source = positions_source(path, metadata_path)
    lists = {}
    if kind == "json" and source and os.path.isfile(source):
        lists = json_number_lists(source)
    taken = {key for key in keys if key}
    for axis in (1, 0):
        if out[axis] is not None:
            continue
        out[axis] = ("", None, "")
        for name, values in lists.items():
            if len(values) == sizes[axis] and name not in taken:
                taken.add(name)
                out[axis] = (name, list(values), "")
                break
    return out


def scan2d_shape(path, hdf_key="", scan_grid=None):
    """(nrow, ncol) of a 2D scan (folder of tif files named
    '..._<row>_<col>.tif', 4D HDF dataset, or 3D HDF dataset with
    scan_grid = (nrow, ncol, scan_order)), or None."""
    path = str(path).strip()
    try:
        if os.path.isfile(path) and path.lower().endswith(HDF_EXT):
            if not hdf_key:
                return None
            with h5py.File(path, "r") as file:
                dataset = file.get(hdf_key)
                if not isinstance(dataset, h5py.Dataset) \
                        or dataset.ndim not in (3, 4):
                    return None
                shape = dataset.shape
                if dataset.ndim == 3:
                    nrow, ncol, _ = scan_grid or (None, None, None)
                    return resolve_scan_grid(shape[0], nrow, ncol)
                return (shape[2], shape[3]) if is_layout_4d(dataset) \
                    else (shape[0], shape[1])
        if path and os.path.exists(path):
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            _, nrow, ncol = structure_2d_scan(find_files(folder))
            return nrow, ncol
    except (OSError, ValueError):
        pass
    return None


def positions_source(path, metadata_path=""):
    """('hdf', file) or ('json', file or '') holding the positions of the
    images: the metadata file if given (JSON or nxs/HDF), else the HDF data
    file, else the JSON found next to the data."""
    metadata_path = str(metadata_path or "").strip()
    if metadata_path:
        kind = "hdf" if metadata_path.lower().endswith(HDF_EXT) else "json"
        return kind, metadata_path
    path = str(path).strip()
    if os.path.isfile(path) and path.lower().endswith(HDF_EXT):
        return "hdf", path
    return "json", find_metadata_file(path) if path else ""


def position_label(key):
    """Motor name of a positions key: its last part, or the parent for
    generic names such as 'entry/sample/x/value'."""
    parts = [p for p in str(key).split("/") if p]
    if not parts:
        return ""
    if len(parts) > 1 and parts[-1].lower() in ("value", "values", "data",
                                                 "position", "positions"):
        return parts[-2]
    return parts[-1]


def read_hdf_1d(file_path, key):
    """Numeric 1D dataset of an HDF file as a float array, or None."""
    with h5py.File(file_path, "r") as file:
        obj = file.get(key)
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1 \
                and np.issubdtype(obj.dtype, np.number):
            return np.asarray(obj[()], dtype=float)
    return None


def json_positions(file_path, key):
    """List of numbers named key in a JSON file (see json_number_lists),
    or None."""
    return json_number_lists(file_path).get(key)


def json_number_lists(file_path):
    """{name: list of floats} of all lists of numbers in a JSON file, at
    any depth of dicts and lists:
    - a list under a dict key is named by the key;
    - a list following a string in a list is named by that string, or by
      its 'name=...' part, e.g. Bluesky plan_args "args":
      ["Motor(name=\"y\")", [-2.5, -2.4, ...]];
    - other lists in lists are named '<parent>[index]'.
    Lists of 1-element lists are flattened. If a name is used twice for
    different values, the full path is used as the name."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    found = OrderedDict()

    def as_numbers(value):
        if not isinstance(value, list) or not value:
            return None
        if all(isinstance(v, list) and len(v) == 1 for v in value):
            value = [v[0] for v in value]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in value):
            return [float(v) for v in value]
        return None

    def add(name, path, values):
        if name in found and found[name] != values:
            name = "/".join(path)
        found.setdefault(name, values)

    def label_of(text):
        match = re.search(r"name\s*=\s*[\"']([^\"']+)", text)
        return match.group(1) if match else text.strip()

    def search(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                values = as_numbers(value)
                if values is not None:
                    add(str(key), path + [str(key)], values)
                elif isinstance(value, (dict, list)):
                    search(value, path + [str(key)])
        elif isinstance(node, list):
            for i, value in enumerate(node):
                values = as_numbers(value)
                item = f"[{i}]"
                if values is not None:
                    prev = node[i - 1] if i > 0 else None
                    if isinstance(prev, str) and prev.strip():
                        add(label_of(prev), path + [item], values)
                    else:
                        parent = path[-1] if path else "list"
                        add(f"{parent}{item}", path + [item], values)
                elif isinstance(value, (dict, list)):
                    search(value, path + [item])

    values = as_numbers(data)
    if values is not None:
        found["values"] = values
    else:
        search(data, [])
    return found


def list_position_arrays(path, metadata_path=""):
    """All 1D arrays of the metadata that can hold positions: [(key, number
    of values), ...]. Lists of numbers of a JSON file, or numeric 1D
    datasets of a nxs/HDF file."""
    kind, source = positions_source(path, metadata_path)
    items = []
    if source and os.path.isfile(source):
        if kind == "hdf":
            skip = ("entry/q_map", HDF_MAP_INFO_KEY)
            with h5py.File(source, "r") as file:
                for key, obj, _ in walk_hdf(file):
                    if isinstance(obj, h5py.Dataset) and obj.ndim == 1 \
                            and np.issubdtype(obj.dtype, np.number) \
                            and not key.startswith(skip):
                        items.append((key, obj.shape[0]))
        else:
            items = [(key, len(values)) for key, values
                     in json_number_lists(source).items()]
    return items


def list_positions(path, hdf_key="", metadata_path=""):
    """1D arrays that can be the positions of the images: ([(key, number
    of values), ...], number of images or None). Arrays with a number of
    values different from the number of images are skipped."""
    path = str(path).strip()
    num = None
    if os.path.isfile(path) and path.lower().endswith(HDF_EXT):
        num = hdf_num_images(path, hdf_key) if hdf_key else None
    elif path and os.path.exists(path):
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        num = len(find_files(folder))
    items = list_position_arrays(path, metadata_path)
    if num is not None:
        items = [item for item in items if item[1] == num]
    return items, num


def hdf_ndim(file_path, key):
    """Number of dimensions of a dataset of an HDF file, or None."""
    with h5py.File(file_path, "r") as file:
        obj = file.get(key) if key else None
        return obj.ndim if isinstance(obj, h5py.Dataset) else None


def hdf_num_images(file_path, key):
    """Number of images of a 3D dataset (axis 0), or None."""
    with h5py.File(file_path, "r") as file:
        if key in file and isinstance(file[key], h5py.Dataset) \
                and file[key].ndim == 3:
            return file[key].shape[0]
    return None


def structure_2d_scan(files):
    """Arrange files named '..._<row>_<col>.tif' in a 2D grid (None for
    missing files). Return (grid, nrow, ncol)."""
    temp = {}
    max_row = max_col = -1
    for path in files:
        parts = Path(path).stem.split("_")
        if len(parts) < 3:
            continue
        try:
            row, col = int(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        temp.setdefault(row, {})[col] = path
        max_row, max_col = max(max_row, row), max(max_col, col)
    if max_row < 0:
        raise ValueError("File names must end with '_<row>_<col>' for a "
                         "2D scan, e.g. 'scan_00013_00002_00024.tif'")
    grid = [[temp.get(r, {}).get(c) for c in range(max_col + 1)]
            for r in range(max_row + 1)]
    return grid, max_row + 1, max_col + 1


def resolve_scan_grid(num, nrow=None, ncol=None):
    """(nrow, ncol) of a 2D scan of num images; a missing size is
    computed from the other."""
    if not nrow and not ncol:
        raise ValueError(f"The 3D dataset has {num} images: please input "
                         f"the rows and/or columns of the 2D scan!")
    if min(nrow or 1, ncol or 1) < 1:
        raise ValueError("Rows and columns must be positive!")
    size = nrow or ncol
    if not (nrow and ncol):
        if num % size:
            raise ValueError(f"{num} images can't be split in rows / "
                             f"columns of {size}!")
        nrow, ncol = (nrow, num // nrow) if nrow else (num // ncol, ncol)
    if nrow * ncol != num:
        raise ValueError(f"{nrow} x {ncol} = {nrow * ncol} scan positions, "
                         f"but the dataset has {num} images!")
    return nrow, ncol


def is_2d_scan_folder(folder):
    """True if a folder has tif files named '..._<row>_<col>.tif'."""
    try:
        structure_2d_scan(find_files(folder))
    except ValueError:
        return False
    return True


class Scan2DDataset:
    """2D-scan data: an image per scan position (row, col), from a folder of
    tif files (loaded on demand), a 4D HDF dataset of shape (width,
    height, nrow, ncol) or (nrow, ncol, height, width), or a 3D HDF
    dataset (images, height, width) with scan_grid = (nrow, ncol,
    scan_order) (a missing size is computed from the other)."""

    layout_4d = False
    hdf_ndims = (3, 4)

    def __init__(self, path, hdf_key="", metadata_path="", x_step=None,
                 cache_size=IMAGE_CACHE_SIZE, motor_keys=("", ""),
                 scan_grid=None):
        path = str(path).strip()
        if not path or not os.path.exists(path):
            raise ValueError(f"No such file or folder: {path}")
        given_metadata = metadata_path
        metadata_path = metadata_path or find_metadata_file(path)
        metadata = losa.load_json(metadata_path) if metadata_path else {}
        self.hdf_file = None
        self.dataset = None
        self.grid = None
        # 3D dataset: image index at each (row, col)
        self.index_map = None
        self.scan_order = None
        is_hdf = os.path.isfile(path) and path.lower().endswith(HDF_EXT)
        dims = " or ".join(f"{n}D" for n in range(self.hdf_ndims[0],
                                                     self.hdf_ndims[1] + 1))
        if is_hdf:
            if not hdf_key:
                if not hdf_array_keys(path, *self.hdf_ndims):
                    raise WrongDimensionError(f"No {dims} dataset (2D "
                                              f"scan) in {path}", path)
                raise ValueError("Please select the HDF key of the "
                                 "dataset!")
            self.hdf_file = h5py.File(path, "r")
            if hdf_key not in self.hdf_file:
                self.close()
                raise ValueError(f"No dataset '{hdf_key}' in {path}")
            self.dataset = self.hdf_file[hdf_key]
            ndim = self.dataset.ndim
            low, high = self.hdf_ndims
            if not low <= ndim <= high:
                shape = self.dataset.shape
                self.close()
                raise WrongDimensionError(f"A 2D scan needs a {dims} "
                                          f"dataset. Got {shape}", path)
            if ndim == 3:
                try:
                    self.set_scan_grid(scan_grid)
                except ValueError:
                    self.close()
                    raise
            else:
                self.layout_4d = is_layout_4d(self.dataset)
                if self.layout_4d:
                    self.width, self.height, self.nrow, self.ncol = \
                        self.dataset.shape
                else:
                    self.nrow, self.ncol, self.height, self.width = \
                        self.dataset.shape
        else:
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            files = find_files(folder)
            if not files:
                raise ValueError(f"No tif files in: {folder}")
            self.grid, self.nrow, self.ncol = structure_2d_scan(files)
            first = next(p for row in self.grid for p in row if p)
            self.height, self.width = losa.load_image(first).shape
        if x_step is None:
            x_step = get_x_step(metadata, path if is_hdf else None, hdf_key,
                                path)
        self.set_x_step(x_step)
        self.name = Path(path).stem if is_hdf else Path(path).name
        self.source = path
        self.apply_positions(motor_keys, given_metadata)
        self.cache_size = cache_size
        self._cache = OrderedDict()
        # Images of the last q-slice line, (axis, fixed): q changes are fast
        self.line_key = None
        self._line = None

    def set_scan_grid(self, scan_grid):
        """Image index of each scan position of a 3D dataset, from
        scan_grid = (nrow, ncol, scan_order)."""
        nrow, ncol, scan_order = scan_grid or (None, None, "row")
        num, self.height, self.width = self.dataset.shape
        self.nrow, self.ncol = resolve_scan_grid(num, nrow, ncol)
        self.index_map = np.zeros((self.nrow, self.ncol), dtype=np.int64)
        for index, (row, col) in enumerate(scan_positions(
                self.nrow, self.ncol, scan_order)):
            self.index_map[row, col] = index
        self.scan_order = scan_order

    def apply_positions(self, motor_keys=("", ""), metadata_path=""):
        """Positions of the rows and columns from the metadata (see
        resolve_positions_2d); the indices where not available. Notes on
        the positions not used are in position_notes."""
        results = resolve_positions_2d(self.source, metadata_path,
                                       motor_keys, self.nrow, self.ncol)
        self.motors, self.position_notes = [], []
        for axis, (name, values, note) in enumerate(results):
            num = (self.nrow, self.ncol)[axis]
            default = ("Row", "Column")[axis]
            if values is None or len(values) != num:
                name, values = default, list(range(num))
            self.motors.append((name or default, list(values)))
            if note:
                self.position_notes.append(note)

    def set_x_step(self, x_step):
        """q-step of the pixels; None or 0: pixel indices."""
        self.calibrated = bool(x_step and x_step > 0)
        self.x_step = float(x_step) if self.calibrated else 1.0
        self.x_list = np.arange(self.width) * self.x_step

    @property
    def x_max(self):
        return float(self.x_list[-1])

    def x_index(self, x_value):
        return int(np.clip(round(x_value / self.x_step), 0, self.width - 1))

    def _read(self, row, col):
        if self.index_map is not None:
            return np.asarray(self.dataset[int(self.index_map[row, col])],
                              dtype=np.float32)
        if self.dataset is not None:
            if self.layout_4d:
                return np.asarray(self.dataset[:, :, row, col],
                                  dtype=np.float32).T
            return np.asarray(self.dataset[row, col], dtype=np.float32)
        try:
            path = self.grid[row][col]
        except IndexError:
            path = None
        if path is None:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return np.asarray(losa.load_image(path), dtype=np.float32)

    def get_image(self, row, col):
        key = (row, col)
        if self._line is not None:
            axis, fixed = self.line_key
            if (col if axis == 0 else row) == fixed:
                return self._line[row if axis == 0 else col]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = self._read(row, col)
        self._cache[key] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    def q_slice(self, axis, fixed, index, radius):
        """Mean over x-columns around index, along one scan axis with the
        other index fixed. axis=0: images (i, fixed); axis=1: (fixed, j).
        Return shape (n, strips)."""
        start = max(index - radius, 0)
        stop = min(index + radius + 1, self.width)
        if self.dataset is not None and self.layout_4d:
            # Read only the pixel range: fast with chunks over scan positions
            if axis == 0:
                block = self.dataset[start:stop, :, :, fixed]
            else:
                block = self.dataset[start:stop, :, fixed, :]
            return np.mean(np.asarray(block, dtype=np.float32), axis=0).T
        if (axis, fixed) != self.line_key:
            self.load_line(axis, fixed)
        if self._line is not None:
            return np.mean(self._line[:, :, start:stop], axis=2)
        num = self.nrow if axis == 0 else self.ncol
        out = np.zeros((num, self.height), dtype=np.float32)
        for k in range(num):
            key = (k, fixed) if axis == 0 else (fixed, k)
            img = self._cache.get(key)
            if img is None:
                img = self._read(*key)
            out[k] = np.mean(img[:, start:stop], axis=1)
        return out

    def load_line(self, axis, fixed):
        """Keep the images along one scan axis in memory if they fit in
        LINE_CACHE_MB, reusing the cached ones."""
        self.line_key = (axis, fixed)
        self._line = None
        num = self.nrow if axis == 0 else self.ncol
        if num * self.height * self.width * 4 > LINE_CACHE_MB * 2 ** 20:
            return
        line = np.empty((num, self.height, self.width), dtype=np.float32)
        for k in range(num):
            key = (k, fixed) if axis == 0 else (fixed, k)
            img = self._cache.get(key)
            line[k] = img if img is not None else self._read(*key)
        self._line = line

    def close(self):
        if self.hdf_file is not None:
            try:
                self.hdf_file.close()
            except Exception:
                pass
            self.hdf_file = None


def take_4d(dataset, layout_4d, q=slice(None), z=slice(None),
            row=slice(None), col=slice(None)):
    """Read a 4D dataset by (q, Z, row, column) indices; the kept axes are
    returned in that order, for both layouts."""
    if layout_4d:
        return np.asarray(dataset[q, z, row, col], dtype=np.float32)
    out = np.asarray(dataset[row, col, z, q], dtype=np.float32)
    kept = [name for name, index in zip("rczq", (row, col, z, q))
            if isinstance(index, slice)]
    return out.transpose([kept.index(name) for name in "qzrc"
                          if name in kept])


class Volume4DDataset(Scan2DDataset):
    """4D HDF dataset (q, Z, rows, columns) with volumes (mean over a
    q-range) cached for the 3D view and the slices."""

    hdf_ndims = (4, 4)

    def __init__(self, path, *args, **kwargs):
        path = str(path).strip()
        if not (os.path.isfile(path) and path.lower().endswith(HDF_EXT)):
            raise ValueError("The 4D viewer needs a 4D HDF dataset. Convert "
                             "the tif files of a 2D scan in 'Data "
                             "correction' first.")
        super().__init__(path, *args, **kwargs)
        self._volumes = OrderedDict()
        attrs = self.dataset.attrs
        if kwargs.get("x_step") is None and "binning" in attrs \
                and Q_STEP_KEY in attrs:
            # Binned q-step, not the one of the map_info of the source
            self.set_x_step(float(attrs[Q_STEP_KEY]))

    @property
    def shape(self):
        return self.width, self.height, self.nrow, self.ncol

    def take(self, **indices):
        return take_4d(self.dataset, self.layout_4d, **indices)

    def has_volume(self, index, radius):
        return (index, radius) in self._volumes

    def volume_nbytes(self, radius):
        return (2 * radius + 1) * self.height * self.nrow * self.ncol * 4

    def volume(self, index, radius):
        """Mean over the q-range [index - radius, index + radius]: shape
        (Z, rows, columns)."""
        key = (index, radius)
        if key in self._volumes:
            self._volumes.move_to_end(key)
            return self._volumes[key]
        start = max(index - radius, 0)
        stop = min(index + radius + 1, self.width)
        vol = np.mean(self.take(q=slice(start, stop)), axis=0)
        self._volumes[key] = vol
        if len(self._volumes) > VOLUME_CACHE_SIZE:
            self._volumes.popitem(last=False)
        return vol


def downsample_output_path(source):
    path = Path(source)
    return str(path.with_name(path.stem + DOWNSAMPLE_SUFFIX + path.suffix))


def bin_array(block, bins):
    """Mean over bins of each axis; sizes must be multiples of bins."""
    shape = []
    for size, num in zip(block.shape, bins):
        shape += [size // num, num]
    return block.reshape(shape).mean(axis=tuple(range(1, 2 * len(bins), 2)))


def downsample_4d(source, hdf_key, output, out_key=HDF_DATA_KEY,
                  bins=DEFAULT_BINS, chunks=DEFAULT_4D_CHUNKS, report=None,
                  cancelled=None):
    """Mean binning of a 4D dataset to a (q, Z, rows, columns) dataset,
    block by block (one output chunk per block). Return (message,
    cancelled)."""
    bins = tuple(int(b) for b in bins)
    if len(bins) != 4 or min(bins) < 1:
        raise ValueError("Binning must be 4 positive integers!")
    out_path = Path(output)
    if out_path.suffix.lower() not in HDF_EXT:
        out_path = out_path.with_suffix(".hdf")
    if out_path.resolve() == Path(source).resolve():
        raise ValueError("The output file must differ from the input file!")
    out_key = check_output_key(out_key)
    with h5py.File(source, "r") as fin:
        if hdf_key not in fin or fin[hdf_key].ndim != 4:
            raise ValueError(f"No 4D dataset '{hdf_key}' in {source}")
        dset = fin[hdf_key]
        layout = is_layout_4d(dset)
        shape = dset.shape if layout else \
            (dset.shape[3], dset.shape[2], dset.shape[0], dset.shape[1])
        out_shape = tuple(n // b for n, b in zip(shape, bins))
        if min(out_shape) < 1:
            raise ValueError(f"Binning {bins} is too large for the shape "
                             f"{shape} (q, Z, rows, columns)!")
        if chunks is not None:
            chunks = tuple(min(c, s) for c, s in zip(chunks, out_shape))
        step = get_x_step({}, source, hdf_key)
        blocks = chunks or out_shape
        starts = [range(0, n, b) for n, b in zip(out_shape, blocks)]
        total = int(np.prod([len(s) for s in starts]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as fout:
            out = fout.create_dataset(out_key, out_shape, dtype="float32",
                                      chunks=chunks)
            out.attrs[LAYOUT_ATTR] = LAYOUT_4D
            out.attrs["binning"] = bins
            out.attrs["source"] = f"{source}:/{hdf_key}"
            if step:
                out.attrs[Q_STEP_KEY] = float(step * bins[0])
            copy_binned_positions(fin, fout, shape[2:], bins[2:])
            done = 0
            for q0 in starts[0]:
                for z0 in starts[1]:
                    for r0 in starts[2]:
                        for c0 in starts[3]:
                            if cancelled is not None and cancelled():
                                return "Cancelled", True
                            ends = [min(s + b, n) for s, b, n in zip(
                                (q0, z0, r0, c0), blocks, out_shape)]
                            index = {name: slice(s * b, e * b) for name, s,
                                     e, b in zip(("q", "z", "row", "col"),
                                                 (q0, z0, r0, c0), ends,
                                                 bins)}
                            block = take_4d(dset, layout, **index)
                            out[q0:ends[0], z0:ends[1], r0:ends[2],
                                c0:ends[3]] = bin_array(block, bins)
                            done += 1
                    if report is not None:
                        report(f"Downsampling: {100 * done / total:.1f}%")
    chunk_text = "no chunking" if chunks is None else \
        f"chunks {chunks} ({format_bytes(chunk_nbytes(chunks))})"
    return f"Downsampled {shape} to {out_shape} (q, Z, rows, columns), " \
           f"{chunk_text}, saved to {out_path}:/{out_key}", False


def copy_binned_positions(fin, fout, scan_shape, bins):
    """Copy the positions of the rows and columns, averaged as the data."""
    group = fin.get(HDF_POSITIONS_KEY)
    if not isinstance(group, h5py.Group):
        return
    for name, item in group.items():
        if not isinstance(item, h5py.Dataset) or item.ndim != 1:
            continue
        values = np.asarray(item[()])
        for size, num in zip(scan_shape, bins):
            if len(values) == size and np.issubdtype(values.dtype,
                                                     np.number):
                values = bin_array(values[:size // num * num], (num,))
                fout.create_dataset(f"{HDF_POSITIONS_KEY}/{name}",
                                    data=values)
                break


def log_scale(mat):
    mat = np.asarray(mat, dtype=np.float32)
    return np.log(mat - np.min(mat) + 1.0)


# ---------------------------------------------------------------------------
# Data correction
# ---------------------------------------------------------------------------

def load_raw_image(path, hdf_key="", index=None, axis=0):
    """Load a 2D image from a tif/dat file or an HDF dataset (a slice of a
    3D dataset if index is given)."""
    ext = Path(path).suffix.lower()
    if ext in HDF_EXT:
        dataset, hdf_object = losa.load_hdf(path, hdf_key,
                                            return_file_obj=True)
        try:
            if index is None:
                mat = np.squeeze(dataset[()])
            else:
                slices = [slice(None)] * 3
                slices[axis] = index
                mat = dataset[tuple(slices)]
        finally:
            hdf_object.close()
    elif ext == ".dat":
        mat = losa.load_dat(path)
    else:
        mat = losa.load_image(path)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Data from '{Path(path).name}' must be 2D. Got "
                         f"shape {mat.shape}")
    return mat


def raw_scan_items(path, hdf_key="", axis=0):
    """Images of a raw scan: [(file, index or None), ...]. path can be a
    scan folder (with a 'tifs' sub-folder), a folder of tif files, files
    separated by ';', or an HDF file (3D dataset: one image per slice)."""
    paths = [p.strip() for p in str(path).split(";") if p.strip()]
    if len(paths) == 1 and os.path.isdir(paths[0]):
        folder = paths[0]
        if os.path.isdir(os.path.join(folder, "tifs")):
            folder = os.path.join(folder, "tifs")
        paths = find_files(folder, TIF_EXT + (".dat",))
        if not paths:
            raise ValueError(f"No tif files in: {folder}")
    items = []
    for item in paths:
        if not os.path.isfile(item):
            raise ValueError(f"No such file: {item}")
        if Path(item).suffix.lower() not in HDF_EXT:
            items.append((item, None))
            continue
        if not hdf_key:
            raise ValueError("Please input the HDF key!")
        with h5py.File(item, "r") as file:
            if hdf_key not in file:
                raise ValueError(f"No dataset '{hdf_key}' in {item}")
            shape = file[hdf_key].shape
        if len(shape) == 3 and 1 not in shape:
            items += [(item, i) for i in range(shape[axis])]
        else:
            items.append((item, None))
    return items


def input_base_folder(entry):
    """Folder holding the input scan (scan folder, HDF file or files), where
    its output folder must not be created."""
    path = Path(str(entry).split(";")[0].strip())
    if path.is_file() and path.suffix.lower() in HDF_EXT:
        return path.parent
    folder = path if path.is_dir() else path.parent
    if folder.name.lower() == "tifs":
        folder = folder.parent
    return folder.parent


def get_q_map(coefs, height, width):
    """q-values of the raw pixels from the coefficients."""
    return np.float32(post.get_q_map(height, width, coefs))


def get_output_step(coefs, width, method, value):
    """Step of the uniform q-grid of corrected data, and its source."""
    if method in ("q-max", "q-step"):
        if value is None or value <= 0:
            raise ValueError(f"The {method} value must be positive!")
        if method == "q-max":
            return value / (width - 1.0), f"from q-max {value:.4f}"
        return float(value), "given step"
    valid = np.any(coefs != 0, axis=1)
    if not np.any(valid):
        raise ValueError("No calibrated strips in the coefficients!")
    q_map = get_q_map(coefs[valid], int(np.sum(valid)), width)
    func = {"mean": np.mean, "median": np.median, "max": np.max}[method]
    return float(func(np.diff(q_map, axis=1))), \
        f"{method} pixel spacing of the q-map"


def correct_image(image, coefs, step):
    """Correct strips with non-zero coefficients to a uniform q-grid;
    others are zeros."""
    if coefs.shape[0] != image.shape[0]:
        raise ValueError(f"Image has {image.shape[0]} strips, but there are "
                         f"{coefs.shape[0]} coefficient rows!")
    valid = np.any(coefs != 0, axis=1)
    out = np.zeros_like(image, dtype=np.float32)
    if np.any(valid):
        out[valid] = post.correct_q_image(image[valid], coefs[valid], step,
                                          ncore=1)
    return out


def remove_background(image, method, kwargs, cancel_event=None,
                      progress=None):
    """Remove the background of each strip, in chunks of strips so it can be
    cancelled. Return None if cancelled."""
    func = getattr(prep, BCK_METHODS[method][0])
    height = image.shape[0]
    chunk = 16
    result = np.zeros_like(image, dtype=np.float32)
    for start in range(0, height, chunk):
        if cancel_event is not None and cancel_event.is_set():
            return None
        if progress is not None:
            progress(f"Background removal: {start}/{height} strips")
        stop = min(start + chunk, height)
        block = image[start:stop]
        # ALS and arPLS can't normalize a block with a zero maximum
        if np.max(block) != 0:
            result[start:stop] = func(block, axis=0, **kwargs)
    return np.clip(result, 0, None)


def item_tag(item):
    path, index = item
    stem = Path(path).stem
    return stem if index is None else f"{stem}_{index:05d}"


def save_xy_image(path, x_map, image, file_type, zipped):
    """Save the strips of an image as xy/csv files, zipped or in a folder."""
    if zipped:
        losa.save_xy(str(path) + ".zip", x_map, image, file_type=file_type,
                     zip_file=True)
    else:
        # save_xy drops the suffix; a dummy one keeps dots in the name
        losa.save_xy(str(path) + ".dir", x_map, image, file_type=file_type,
                     zip_file=False)


def xy_x_map(q_map, x_axis, max_two_theta):
    """x-values of xy/csv files: the q-map, or the equivalent 2theta
    (degrees, one reference energy for the whole map). Return (x_map,
    reference energy in keV or None)."""
    if x_axis != "two-theta":
        return q_map, None
    # Uncalibrated pixels can have negative q, not allowed by q_to_two_theta
    two_theta, energy = autil.q_to_two_theta(np.clip(q_map, 0, None),
                                             max_two_theta,
                                             return_energy=True)
    return np.float32(two_theta), energy


def make_map_info(coefs, step, step_source, height, width, coef_file,
                  input_path, items, bck=None):
    """(info dict, q-map of the raw pixels, q-map of the corrected pixels
    with a uniform step)."""
    raw_map = get_q_map(coefs, height, width)
    q_map = np.tile(np.float32(step * np.arange(width)), (height, 1))
    info = {"coefficients_file": str(coef_file),
            "input_path": str(input_path),
            "input_images": [f"{p}" + ("" if i is None else f"[{i}]")
                             for p, i in items],
            Q_STEP_KEY: float(step),
            "q_step_source": step_source,
            "q_unit": "1/\u00c5",
            "image_shape": [int(height), int(width)],
            "background_removal": None if bck is None else
            {"method": bck[0], **bck[1]}}
    return info, raw_map, q_map


def save_map_info(output_folder, info, raw_map=None, q_map=None):
    """Save map_info/ with the info (JSON) and, if given, the q-maps
    (tif)."""
    folder = Path(output_folder) / MAP_INFO_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    if raw_map is not None and q_map is not None:
        losa.save_image(str(folder / RAW_Q_MAP_FILE), raw_map)
        losa.save_image(str(folder / CORRECTED_Q_MAP_FILE), q_map)
        info = dict(info, raw_q_map=RAW_Q_MAP_FILE,
                    corrected_q_map=CORRECTED_Q_MAP_FILE)
    # Plain JSON: losa.save_json wraps the content in a 'Metadata' entry
    with open(folder / MAP_INFO_FILE, "w") as file:
        json.dump(info, file, indent=2)


def save_map_info_hdf(file, info, raw_map, q_map):
    """Save the q-maps as 2D datasets and each info entry as a dataset of
    the map_info group of an open HDF file."""
    file.create_dataset(HDF_RAW_MAP_KEY, data=raw_map)
    file.create_dataset(HDF_CORRECTED_MAP_KEY, data=q_map)
    info = dict(info, raw_q_map=HDF_RAW_MAP_KEY,
                corrected_q_map=HDF_CORRECTED_MAP_KEY)
    group = file.require_group(HDF_MAP_INFO_KEY)
    str_type = h5py.string_dtype()
    for key, value in info.items():
        if isinstance(value, (dict, type(None))):
            group.create_dataset(key, data=json.dumps(value), dtype=str_type)
        elif isinstance(value, str):
            group.create_dataset(key, data=value, dtype=str_type)
        elif isinstance(value, list) and value and \
                isinstance(value[0], str):
            group.create_dataset(key, data=np.array(value, dtype=object),
                                 dtype=str_type)
        else:
            group.create_dataset(key, data=np.asarray(value))


def is_layout_4d(dataset):
    """True if a 4D dataset has the shape (width, height, nrow, ncol)."""
    layout = dataset.attrs.get(LAYOUT_ATTR)
    if isinstance(layout, bytes):
        layout = layout.decode()
    if layout is not None:
        return str(layout) == LAYOUT_4D
    # Files without the attribute: the pixel axis is the longest one
    shape = dataset.shape
    return shape[0] == max(shape) and shape[0] > shape[3]


def scan_positions(nrow, ncol, scan_order="row"):
    """(row, col) of the images in acquisition order, as in algodif
    convert_tif_to_4D_hdf."""
    if scan_order not in SCAN_ORDERS:
        raise ValueError(f"Scan order must be one of {SCAN_ORDERS}")
    indices = np.arange(nrow * ncol).reshape(nrow, ncol)
    if scan_order == "row":
        order = indices.ravel()
    elif scan_order == "column":
        order = indices.T.ravel()
    elif scan_order == "snake_row":
        indices[1::2] = indices[1::2, ::-1].copy()
        order = indices.ravel()
    else:
        indices[:, 1::2] = indices[::-1, 1::2].copy()
        order = indices.T.ravel()
    return [divmod(int(i), ncol) for i in order]


def parse_chunks(text):
    """Chunk shape (width, height, nrow, ncol) from text like
    '64, 64, 8, 8'. None if empty (no chunking)."""
    text = str(text).strip()
    if not text or text.lower() == "none":
        return None
    try:
        chunks = tuple(int(v) for v in re.split(r"[,\s]+", text) if v)
    except ValueError:
        chunks = ()
    if len(chunks) != 4 or min(chunks) < 1:
        raise ValueError("The chunk shape must be 4 positive integers: "
                         "width, height, rows, columns.")
    return chunks


def chunk_nbytes(chunks, itemsize=4):
    return int(np.prod(chunks)) * itemsize if chunks else 0


def tif_scan_to_hdf4d(source, output, out_key, nrow=None, ncol=None,
                      scan_order="row", chunks=DEFAULT_4D_CHUNKS,
                      x_step=None, metadata=None, report=None,
                      cancelled=None):
    """Convert the tif files of a 2D scan to a 4D dataset of shape (width,
    height, nrow, ncol), as in algodif convert_tif_to_4D_hdf. Without nrow
    and ncol, positions come from file names ending with '_<row>_<col>'.
    Return (message, cancelled)."""
    files = find_files(source)
    if not files:
        raise ValueError(f"No tif files in: {source}")
    if nrow and ncol:
        if len(files) != nrow * ncol:
            raise ValueError(f"Expected {nrow * ncol} tif files "
                             f"({nrow} x {ncol}), found {len(files)}.")
        pos_files = dict(zip(scan_positions(nrow, ncol, scan_order), files))
    else:
        grid, nrow, ncol = structure_2d_scan(files)
        pos_files = {(r, c): grid[r][c] for r in range(nrow)
                     for c in range(ncol) if grid[r][c]}
    height, width = losa.load_image(files[0]).shape
    shape = (width, height, nrow, ncol)
    if chunks is not None:
        chunks = tuple(min(c, s) for c, s in zip(chunks, shape))
    out_path = Path(output)
    if out_path.suffix.lower() not in HDF_EXT:
        out_path = out_path.with_suffix(".hdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as file:
        out = file.create_dataset(out_key, shape, dtype="float32",
                                  chunks=chunks)
        out.attrs[LAYOUT_ATTR] = LAYOUT_4D
        out.attrs["scan_order"] = scan_order
        step = x_step or get_x_step(metadata or {}, data_path=source)
        if step:
            out.attrs[Q_STEP_KEY] = float(step)
            write_hdf_q_maps(file, source, step, height, width)
        # Fill whole chunks along the scan axes at once: a chunk is written
        # once instead of being re-read and re-written for each image
        block_r, block_c = chunks[2:] if chunks else (1, 1)
        done = 0
        for r0 in range(0, nrow, block_r):
            for c0 in range(0, ncol, block_c):
                r1, c1 = min(r0 + block_r, nrow), min(c0 + block_c, ncol)
                block = np.zeros((width, height, r1 - r0, c1 - c0),
                                 dtype=np.float32)
                for row in range(r0, r1):
                    for col in range(c0, c1):
                        if cancelled is not None and cancelled():
                            return "Cancelled", True
                        path = pos_files.get((row, col))
                        if path is None:
                            continue
                        image = losa.load_image(path)
                        if image.shape != (height, width):
                            raise ValueError(
                                f"Image shape mismatch: {path}. Expected "
                                f"{(height, width)}, got {image.shape}.")
                        block[:, :, row - r0, col - c0] = image.T
                        done += 1
                if report is not None:
                    report(f"Image {done}/{len(pos_files)}")
                out[:, :, r0:r1, c0:c1] = block
    chunk_text = "no chunking" if chunks is None else \
        f"chunks {chunks} ({format_bytes(chunk_nbytes(chunks))})"
    return f"Saved a 4D dataset (width, height, rows, columns) = {shape}, " \
           f"{chunk_text}, to {out_path}:/{out_key}", False


def write_hdf_q_maps(file, source, step, height, width):
    """Corrected q-map of a converted dataset, with the raw q-map and the
    info of the map_info folder of the source if available."""
    q_map = np.tile(np.float32(step * np.arange(width)), (height, 1))
    folder = find_map_info_folder(source)
    if folder is not None and (folder / RAW_Q_MAP_FILE).is_file():
        info = find_map_info(source)
        for key in ("raw_q_map", "corrected_q_map"):
            info.pop(key, None)
        info[Q_STEP_KEY] = float(step)
        raw_map = losa.load_image(str(folder / RAW_Q_MAP_FILE))
        save_map_info_hdf(file, info, raw_map, q_map)
    else:
        file.create_dataset(HDF_CORRECTED_MAP_KEY, data=q_map)


def correct_scan(items, output_folder, coefs, para, cancel_event=None,
                 progress=None, metadata=None, input_path=""):
    """Correct the images of a scan and save them, with the map info
    (map_info/ folder, or datasets of the HDF output) and a copy of the
    scan metadata.

    para: dict with keys step, step_source, coef_file, hdf_key, axis,
    format, bck (None or (method, kwargs)), xy_type, zip, raw (xy/csv raw
    data), xy_folder, xy_x ("q" or "two-theta"), max_two_theta, out_key
    (dataset of HDF output).
    Return (number of saved images, cancelled).
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    step, fmt = para["step"], para["format"]
    first = load_raw_image(items[0][0], para["hdf_key"], items[0][1],
                           para["axis"])
    height, width = first.shape
    if coefs.shape[0] != height:
        raise ValueError(f"Images have {height} strips, but there are "
                         f"{coefs.shape[0]} coefficient rows!")
    info, raw_map, q_map = make_map_info(coefs, step, para["step_source"],
                                         height, width, para["coef_file"],
                                         input_path, items, para["bck"])
    xy_map = None
    if fmt == "xy/csv":
        xy_map, energy = xy_x_map(raw_map if para["raw"] else q_map,
                                  para["xy_x"], para["max_two_theta"])
        info["xy_x_axis"] = "q (1/\u00c5)" if energy is None else {
            "x": "two_theta (deg)", "max_two_theta": para["max_two_theta"],
            "reference_energy_keV": energy}
    if fmt == "tif":
        save_map_info(output_folder, info, raw_map, q_map)
    elif fmt == "xy/csv":
        # The x-values are in the xy/csv files: no q-map tif needed
        save_map_info(output_folder, info)
    if metadata:
        with open(output_folder / "metadata.json", "w") as file:
            json.dump(metadata, file, indent=2)
    motors = get_motor_positions(metadata, len(items)) if metadata else []
    hdf_out = None
    if fmt == "hdf":
        hdf_path = output_folder / f"{output_folder.name}.hdf"
        hdf_out = losa.open_hdf_stream(str(hdf_path),
                                       (len(items), height, width),
                                       key_path=check_output_key(
                                           para.get("out_key", "")))
        file = hdf_out.file
        save_map_info_hdf(file, info, raw_map, q_map)
        hdf_out.attrs[Q_STEP_KEY] = float(step)
        hdf_out.attrs["background_removed"] = para["bck"] is not None
        for name, values in motors:
            file.create_dataset(f"{HDF_POSITIONS_KEY}/{name}",
                                data=np.asarray(values))
    try:
        for num, item in enumerate(items):
            if cancel_event is not None and cancel_event.is_set():
                return num, True
            tag = item_tag(item)
            prefix = f"{output_folder.name}: image {num + 1}/{len(items)}. "

            def report(text, prefix=prefix):
                if progress is not None:
                    progress(prefix + text)

            report("Correcting...")
            image = first if num == 0 else load_raw_image(
                item[0], para["hdf_key"], item[1], para["axis"])
            if fmt == "xy/csv" and para["raw"]:
                save_xy_image(output_folder / para["xy_folder"] / tag,
                              xy_map, image, para["xy_type"], para["zip"])
                continue
            data = correct_image(image, coefs, step)
            if para["bck"] is not None:
                method, kwargs = para["bck"]
                data = remove_background(data, method, kwargs, cancel_event,
                                         report)
                if data is None:
                    return num, True
            report("Saving...")
            if fmt == "tif":
                losa.save_image(str(output_folder / f"{tag}.tif"), data)
            elif fmt == "hdf":
                hdf_out[num] = data
            else:
                save_xy_image(output_folder / para["xy_folder"] / tag,
                              xy_map, data, para["xy_type"], para["zip"])
        return len(items), False
    finally:
        if hdf_out is not None:
            hdf_out.file.close()


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def check_output_key(key):
    """Clean dataset key of HDF outputs; it can't be a group used for the
    q-maps, map info or motor positions."""
    key = str(key).strip().strip("/") or HDF_DATA_KEY
    for reserved in ("entry/q_map", HDF_MAP_INFO_KEY, HDF_POSITIONS_KEY,
                     HDF_LEGACY_MAP_KEY):
        if key == reserved or key.startswith(reserved + "/"):
            raise ValueError(f"The output key can't be in '{reserved}'!")
    return key


def convert_data(mode, source, output, hdf_key="", x_step=None,
                 xy_type="xy", zipped=False, cancel_event=None,
                 progress=None, x_axis="q",
                 max_two_theta=DEFAULT_MAX_TWO_THETA, out_key=HDF_DATA_KEY,
                 nrow=None, ncol=None, scan_order="row",
                 chunks=DEFAULT_4D_CHUNKS):
    """Convert processed data between formats. nrow, ncol, scan_order and
    chunks are for 2D scans to 4D HDF. Return (message, cancelled)."""
    source = str(source).strip()
    output = str(output).strip()
    out_key = check_output_key(out_key)
    if not source or not os.path.exists(source):
        raise ValueError(f"No such file or folder: {source}")
    if not output:
        raise ValueError("Please input the output path!")

    def report(text):
        if progress is not None:
            progress(text)

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    metadata_path = find_metadata_file(source)
    metadata = losa.load_json(metadata_path) if metadata_path else {}
    is_hdf = os.path.isfile(source) and source.lower().endswith(HDF_EXT)
    if mode == "tif_to_hdf":
        files = find_files(source)
        if not files:
            raise ValueError(f"No tif files in: {source}")
        first = losa.load_image(files[0])
        out = losa.open_hdf_stream(output, (len(files),) + first.shape,
                                   key_path=out_key)
        try:
            step = x_step or get_x_step(metadata, data_path=source)
            if step:
                out.attrs[Q_STEP_KEY] = float(step)
                write_hdf_q_maps(out.file, source, step, *first.shape)
            for name, values in get_motor_positions(metadata, len(files)):
                out.file.create_dataset(f"{HDF_POSITIONS_KEY}/{name}",
                                        data=np.asarray(values))
            for i, path in enumerate(files):
                if cancelled():
                    return f"Cancelled after {i} images", True
                report(f"Image {i + 1}/{len(files)}")
                out[i] = losa.load_image(path)
            out_path = out.file.filename
        finally:
            out.file.close()
        return f"Saved {len(files)} images to {out_path}:/{out_key}", \
            False
    if mode == "hdf_to_tif":
        if not is_hdf:
            raise ValueError("Please select an HDF file!")
        with h5py.File(source, "r") as file:
            if hdf_key not in file:
                raise ValueError(f"No dataset '{hdf_key}' in {source}")
            dataset = file[hdf_key]
            if dataset.ndim == 2:
                losa.save_image(os.path.join(output, "img_00000.tif"),
                                dataset[()])
                return f"Saved 1 image to {output}", False
            if dataset.ndim != 3:
                raise ValueError(f"Dataset must be 2D or 3D. Got "
                                 f"{dataset.shape}")
            for i in range(dataset.shape[0]):
                if cancelled():
                    return f"Cancelled after {i} images", True
                report(f"Image {i + 1}/{dataset.shape[0]}")
                losa.save_image(os.path.join(output, f"img_{i:05d}.tif"),
                                dataset[i])
            return f"Saved {dataset.shape[0]} images to {output}", False
    if mode == "tif2d_to_hdf4d":
        return tif_scan_to_hdf4d(source, output, out_key, nrow, ncol,
                                 scan_order, chunks, x_step, metadata,
                                 report, cancelled)
    if mode == "to_xy":
        if is_hdf:
            with h5py.File(source, "r") as file:
                if hdf_key not in file:
                    raise ValueError(f"No dataset '{hdf_key}' in {source}")
                shape = file[hdf_key].shape
            if len(shape) not in (2, 3):
                raise ValueError(f"Dataset must be 2D or 3D. Got {shape}")
            num = 1 if len(shape) == 2 else shape[0]
            names = [f"img_{i:05d}" for i in range(num)]
        else:
            files = find_files(source)
            if not files:
                raise ValueError(f"No tif files in: {source}")
            num = len(files)
            names = [Path(f).stem for f in files]
        step = x_step or get_x_step(metadata, source if is_hdf else None,
                                    hdf_key, source)
        if not step:
            raise ValueError("Can't find the q-step in the map_info or the "
                             "metadata. Please input it.")
        for i in range(num):
            if cancelled():
                return f"Cancelled after {i} images", True
            report(f"Image {i + 1}/{num}")
            if is_hdf:
                with h5py.File(source, "r") as file:
                    data = file[hdf_key]
                    image = np.asarray(data[()] if data.ndim == 2
                                       else data[i], dtype=np.float32)
            else:
                image = losa.load_image(files[i])
            x_map = np.tile(np.float32(step * np.arange(image.shape[1])),
                            (image.shape[0], 1))
            x_map, _ = xy_x_map(x_map, x_axis, max_two_theta)
            save_xy_image(Path(output) / names[i], x_map, image, xy_type,
                          zipped)
        return f"Saved {num} image(s) as {xy_type} files to {output}", False
    raise ValueError(f"Unknown conversion: {mode}")
