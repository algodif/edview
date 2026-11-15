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
from pathlib import Path
from collections import OrderedDict
import numpy as np
import h5py

import algodif.io.loadersaver as losa
import algodif.prep.preprocessing as prep
import algodif.post.postprocessing as post

APP_NAME = "EDView - EDXD data viewer"
FONT_SIZE = 14
MAIN_WIN_RATIO = 0.9
MAX_WIN_W = 1920
MAX_WIN_H = 1080
PLOT_FONT_SIZE = 11
PLOT_TITLE_SIZE = 11

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
TREE_MIN_W = 220
FILE_SELECT_DEBOUNCE_MS = 120
WORKER_POLL_MS = 200
# Number of full images kept in memory by the 2D-scan viewer
IMAGE_CACHE_SIZE = 20

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
HDF_EXT = (".hdf", ".h5", ".hdf5", ".nxs")
TEXT_EXT = (".json", ".txt", ".csv", ".xy", ".log", ".py", ".yaml", ".yml",
            ".md", ".dat")
TIF_EXT = (".tif", ".tiff")

IMAGE_CMAPS = ("inferno", "viridis", "magma", "plasma", "gray")
RADII = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
DEFAULT_RADIUS = 9
COLORS = {"blue": "#1f77b4", "red": "#d62728", "green": "#2ca02c",
          "lime": "#39ff14", "yellow": "#ffd400"}

# Kind of calibration -> (axis label, unit, name of the step in metadata)
X_KINDS = {"q": ("Q", "1/\u00c5", "q_step"),
           "energy": ("Energy", "keV", "energy_step")}
# Output step of corrected data: statistic of the pixel spacing of the
# x-map computed from the coefficients, or from an x-max or step value
OUTPUT_STEP_OPTIONS = ("mean", "median", "max", "x-max", "x-step")
OUTPUT_FORMATS = ("tif", "hdf", "xy/csv")
HDF_DATA_KEY = "entry/data/data"
HDF_DATA_NO_BGR_KEY = "entry/data/data_without_bgr"
HDF_MAP_KEY = "entry/map"
HDF_POSITIONS_KEY = "entry/positions"
CONVERSIONS = {
    "TIF folder -> HDF (3D stack)": "tif_to_hdf",
    "HDF (3D stack) -> TIF folder": "hdf_to_tif",
    "TIF folder (2D scan) -> HDF (4D)": "tif2d_to_hdf4d",
    "TIF folder / HDF -> xy/csv": "to_xy",
}
VIEWER_TABS = ("EDDT viewer", "Multiple-lines viewer",
               "Multiple-scans viewer", "2D-scan viewer")

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


def hdf_array_keys(file_path, min_ndim=1):
    """Keys and shapes of numeric datasets with at least min_ndim dims,
    sorted by number of dimensions (descending)."""
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= min_ndim \
                and np.issubdtype(obj.dtype, np.number):
            out.append((name, obj.shape))

    with h5py.File(file_path, "r") as file:
        file.visititems(visit)
    out.sort(key=lambda item: len(item[1]), reverse=True)
    return out


def hdf_tree_lines(file_path):
    """Text lines describing the groups and datasets of an HDF file."""
    lines = []

    def visit(name, obj):
        depth = name.count("/")
        label = name.split("/")[-1]
        if isinstance(obj, h5py.Dataset):
            lines.append(f"{'    ' * depth}{label}  {obj.shape} {obj.dtype}")
        else:
            lines.append(f"{'    ' * depth}{label}/")

    with h5py.File(file_path, "r") as file:
        file.visititems(visit)
    return lines


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
            motors.append((key, [float(v) for v in values]))
    if num is not None:
        motors = [m for m in motors if len(m[1]) == num] or motors
    return motors


# ---------------------------------------------------------------------------
# Datasets for the viewers
# ---------------------------------------------------------------------------

class ScanDataset:
    """Stack of processed images (images, strips, pixels) of a scan, with
    the x-values (q or energy) of the pixels and the motor positions."""

    def __init__(self, data, x_step, x_kind="q", motor_name="",
                 positions=None, name="", source="", files=None):
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim == 2:
            self.data = self.data[np.newaxis]
        self.depth, self.height, self.width = self.data.shape
        self.x_step = float(x_step)
        self.x_kind = x_kind
        self.x_list = np.arange(self.width) * self.x_step
        self.motor_name = motor_name or "Image"
        if positions is None or len(positions) != self.depth:
            positions = list(range(self.depth))
            self.motor_name = "Image"
        self.positions = list(positions)
        self.name = name
        self.source = source
        self.files = files or []

    @property
    def x_max(self):
        return float(self.x_list[-1]) if self.width > 1 else self.x_step

    def x_index(self, x_value):
        return int(np.clip(round(x_value / self.x_step), 0, self.width - 1))

    def q_slice(self, index, radius):
        """Mean over the x-columns around index: shape (images, strips)."""
        start = max(index - radius, 0)
        stop = min(index + radius + 1, self.width)
        return np.mean(self.data[:, :, start:stop], axis=2)


def get_x_step(metadata, x_kind, hdf_path=None, hdf_key=None):
    """Step of the x-values from the metadata or from the x-map saved in the
    HDF file. Return None if not found."""
    if hdf_path:
        with h5py.File(hdf_path, "r") as file:
            if HDF_MAP_KEY in file:
                row = np.asarray(file[HDF_MAP_KEY][0], dtype=float)
                if row.size > 1:
                    return float(row[1] - row[0])
            if hdf_key and hdf_key in file:
                for name in ("x_step", X_KINDS[x_kind][2]):
                    if name in file[hdf_key].attrs:
                        return float(file[hdf_key].attrs[name])
    for name in (X_KINDS[x_kind][2], "x_step", "q_step", "energy_step"):
        value = find_value(metadata, name)
        if value is not None:
            return float(value)
    return None


def load_scan(path, hdf_key="", metadata_path="", x_step=None, x_kind="q",
              motor_index=0):
    """Load a processed scan from a folder of tif files or an HDF file."""
    path = str(path).strip()
    if not path or not os.path.exists(path):
        raise ValueError(f"No such file or folder: {path}")
    metadata_path = metadata_path or find_metadata_file(path)
    metadata = losa.load_json(metadata_path) if metadata_path else {}
    files = []
    is_hdf = os.path.isfile(path) and path.lower().endswith(HDF_EXT)
    if is_hdf:
        if not hdf_key:
            raise ValueError("Please select the HDF key of the dataset!")
        with h5py.File(path, "r") as file:
            if hdf_key not in file:
                raise ValueError(f"No dataset '{hdf_key}' in {path}")
            data = np.asarray(file[hdf_key][()], dtype=np.float32)
            metadata_motors = []
            if HDF_POSITIONS_KEY in file and data.ndim == 3:
                group = file[HDF_POSITIONS_KEY]
                names = list(group.keys())
                if names:
                    name = names[min(motor_index, len(names) - 1)]
                    metadata_motors = [(name, list(group[name][()]))]
        if data.ndim not in (2, 3):
            raise ValueError(f"Dataset must be 2D or 3D. Got {data.shape}")
    else:
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        files = find_files(folder)
        if not files:
            raise ValueError(f"No tif files in: {folder}")
        data = np.asarray([losa.load_image(f) for f in files],
                          dtype=np.float32)
        metadata_motors = []
    if x_step is None:
        x_step = get_x_step(metadata, x_kind, path if is_hdf else None,
                            hdf_key)
    if x_step is None or x_step <= 0:
        raise ValueError("Can't find the x-step (q-step or energy step) in "
                         "the metadata. Please input it.")
    depth = data.shape[0] if data.ndim == 3 else 1
    motors = metadata_motors or get_motor_positions(metadata, depth)
    motor_name, positions = "", None
    if motors:
        motor_name, positions = motors[min(motor_index, len(motors) - 1)]
    name = Path(path).stem if is_hdf else Path(path).name
    if name.lower().startswith("tifs") and not is_hdf:
        name = Path(path).parent.name
    return ScanDataset(data, x_step, x_kind, motor_name, positions, name,
                       path, files)


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


class Scan2DDataset:
    """2D-scan data: an image per scan position (row, col), from a folder of
    tif files (loaded on demand) or a 4D HDF dataset."""

    def __init__(self, path, hdf_key="", metadata_path="", x_step=None,
                 x_kind="q", cache_size=IMAGE_CACHE_SIZE):
        path = str(path).strip()
        if not path or not os.path.exists(path):
            raise ValueError(f"No such file or folder: {path}")
        metadata_path = metadata_path or find_metadata_file(path)
        metadata = losa.load_json(metadata_path) if metadata_path else {}
        self.hdf_file = None
        self.dataset = None
        self.grid = None
        is_hdf = os.path.isfile(path) and path.lower().endswith(HDF_EXT)
        if is_hdf:
            self.hdf_file = h5py.File(path, "r")
            if hdf_key not in self.hdf_file:
                self.close()
                raise ValueError(f"No dataset '{hdf_key}' in {path}")
            self.dataset = self.hdf_file[hdf_key]
            if self.dataset.ndim != 4:
                shape = self.dataset.shape
                self.close()
                raise ValueError(f"A 2D scan needs a 4D dataset (rows, "
                                 f"columns, strips, pixels). Got {shape}")
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
            x_step = get_x_step(metadata, x_kind,
                                path if is_hdf else None, hdf_key)
        if x_step is None or x_step <= 0:
            self.close()
            raise ValueError("Can't find the x-step (q-step or energy step) "
                             "in the metadata. Please input it.")
        self.x_step = float(x_step)
        self.x_kind = x_kind
        self.x_list = np.arange(self.width) * self.x_step
        motors = get_motor_positions(metadata)
        # Columns (fast axis) are listed before rows in the metadata
        self.motors = []
        for idx, num in ((1, self.nrow), (0, self.ncol)):
            name, values = (motors[idx] if len(motors) > idx
                            else ("", []))
            if len(values) != num:
                values = list(range(num))
                name = name or ("Row" if num == self.nrow else "Column")
            self.motors.append((name or "Index", values))
        self.name = Path(path).stem if is_hdf else Path(path).name
        self.source = path
        self.cache_size = cache_size
        self._cache = OrderedDict()

    @property
    def x_max(self):
        return float(self.x_list[-1])

    def x_index(self, x_value):
        return int(np.clip(round(x_value / self.x_step), 0, self.width - 1))

    def _read(self, row, col):
        if self.dataset is not None:
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
        num = self.nrow if axis == 0 else self.ncol
        out = np.zeros((num, self.height), dtype=np.float32)
        for k in range(num):
            key = (k, fixed) if axis == 0 else (fixed, k)
            img = self._cache.get(key)
            if img is None:
                img = self._read(*key)
            out[k] = np.mean(img[:, start:stop], axis=1)
        return out

    def close(self):
        if self.hdf_file is not None:
            try:
                self.hdf_file.close()
            except Exception:
                pass
            self.hdf_file = None


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


def scan_folders(base_folder, scans_text):
    """Scan folders 'scan_<number>' (5 digits) in a base folder."""
    numbers = parse_indices(scans_text, None)
    if not numbers:
        raise ValueError("Please input scan numbers, e.g. '1-5, 8'")
    folders = []
    for number in numbers:
        folder = os.path.join(base_folder, f"scan_{number:05d}")
        if not os.path.isdir(folder):
            raise ValueError(f"No such folder: {folder}")
        folders.append(folder)
    return folders


def get_x_map(coefs, height, width, x_kind):
    """x-values (q or energy) of the raw pixels from the coefficients."""
    if x_kind == "energy":
        return np.float32(post.get_energy_map(height, width, coefs))
    return np.float32(post.get_q_map(height, width, coefs))


def get_output_step(coefs, width, method, value, x_kind):
    """Step of the uniform grid of corrected data, and its source."""
    if method in ("x-max", "x-step"):
        if value is None or value <= 0:
            raise ValueError(f"The {method} value must be positive!")
        if method == "x-max":
            return value / (width - 1.0), f"from x-max {value:.4f}"
        return float(value), "given step"
    valid = np.any(coefs != 0, axis=1)
    if not np.any(valid):
        raise ValueError("No calibrated strips in the coefficients!")
    x_map = get_x_map(coefs[valid], int(np.sum(valid)), width, x_kind)
    func = {"mean": np.mean, "median": np.median, "max": np.max}[method]
    return float(func(np.diff(x_map, axis=1))), \
        f"{method} pixel spacing of the {x_kind}-map"


def correct_image(image, coefs, step, x_kind):
    """Correct strips with non-zero coefficients to a uniform grid; others
    are zeros."""
    if coefs.shape[0] != image.shape[0]:
        raise ValueError(f"Image has {image.shape[0]} strips, but there are "
                         f"{coefs.shape[0]} coefficient rows!")
    valid = np.any(coefs != 0, axis=1)
    out = np.zeros_like(image, dtype=np.float32)
    if np.any(valid):
        if x_kind == "energy":
            out[valid] = post.correct_energy_image(image[valid],
                                                   coefs[valid], step,
                                                   ncore=1)
        else:
            out[valid] = post.correct_q_image(image[valid], coefs[valid],
                                              step, ncore=1)
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


def correct_scan(items, output_folder, coefs, para, cancel_event=None,
                 progress=None, metadata=None):
    """Correct the images of a scan and save them with the metadata.

    para: dict with keys x_kind, step, hdf_key, axis, format, bck (None or
    (method, kwargs)), xy_type, zip, raw (xy/csv raw data).
    Return (number of saved images, cancelled).
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    x_kind, step, fmt = para["x_kind"], para["step"], para["format"]
    first = load_raw_image(items[0][0], para["hdf_key"], items[0][1],
                           para["axis"])
    height, width = first.shape
    if coefs.shape[0] != height:
        raise ValueError(f"Images have {height} strips, but there are "
                         f"{coefs.shape[0]} coefficient rows!")
    raw_map = get_x_map(coefs, height, width, x_kind)
    x_map = np.tile(np.float32(step * np.arange(width)), (height, 1))
    step_name = X_KINDS[x_kind][2]
    losa.save_csv(str(output_folder / f"{x_kind}_calibration_coefficients"
                                      f".csv"), coefs)
    losa.save_image(str(output_folder / f"{x_kind}_map.tif"), raw_map)
    meta = dict(metadata or {})
    meta.update({step_name: float(step), "x_kind": x_kind,
                 "input": [f"{p}" + ("" if i is None else f"[{i}]")
                           for p, i in items]})
    losa.save_json(str(output_folder / "metadata.json"), meta)
    motors = get_motor_positions(metadata, len(items)) if metadata else []
    hdf_out = hdf_out2 = None
    if fmt == "hdf":
        hdf_path = output_folder / f"{output_folder.name}.hdf"
        hdf_out = losa.open_hdf_stream(str(hdf_path),
                                       (len(items), height, width),
                                       key_path=HDF_DATA_KEY)
        file = hdf_out.file
        file.create_dataset(HDF_MAP_KEY, data=x_map)
        hdf_out.attrs[step_name] = float(step)
        hdf_out.attrs["x_step"] = float(step)
        for name, values in motors:
            file.create_dataset(f"{HDF_POSITIONS_KEY}/{name}",
                                data=np.asarray(values))
        if para["bck"] is not None:
            hdf_out2 = file.create_dataset(HDF_DATA_NO_BGR_KEY,
                                           (len(items), height, width),
                                           dtype="float32")
            hdf_out2.attrs[step_name] = float(step)
            hdf_out2.attrs["x_step"] = float(step)
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
                              raw_map, image, para["xy_type"], para["zip"])
                continue
            corrected = correct_image(image, coefs, step, x_kind)
            no_bgr = None
            if para["bck"] is not None:
                method, kwargs = para["bck"]
                no_bgr = remove_background(corrected, method, kwargs,
                                           cancel_event, report)
                if no_bgr is None:
                    return num, True
            report("Saving...")
            if fmt == "tif":
                losa.save_image(str(output_folder / "tifs_with_bgr" /
                                    f"{tag}.tif"), corrected)
                if no_bgr is not None:
                    losa.save_image(str(output_folder / "tifs_without_bgr" /
                                        f"{tag}.tif"), no_bgr)
            elif fmt == "hdf":
                hdf_out[num] = corrected
                if no_bgr is not None:
                    hdf_out2[num] = no_bgr
            else:
                data = no_bgr if no_bgr is not None else corrected
                save_xy_image(output_folder / para["xy_folder"] / tag,
                              x_map, data, para["xy_type"], para["zip"])
        return len(items), False
    finally:
        if hdf_out is not None:
            hdf_out.file.close()


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def convert_data(mode, source, output, hdf_key="", x_step=None,
                 x_kind="q", xy_type="xy", zipped=True, cancel_event=None,
                 progress=None):
    """Convert processed data between formats. Return (message,
    cancelled)."""
    source = str(source).strip()
    output = str(output).strip()
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
                                   key_path=HDF_DATA_KEY)
        try:
            step = x_step or get_x_step(metadata, x_kind)
            if step:
                out.attrs["x_step"] = float(step)
                out.attrs[X_KINDS[x_kind][2]] = float(step)
                out.file.create_dataset(HDF_MAP_KEY, data=np.tile(
                    np.float32(step * np.arange(first.shape[1])),
                    (first.shape[0], 1)))
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
        return f"Saved {len(files)} images to {out_path}:/{HDF_DATA_KEY}", \
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
        grid, nrow, ncol = structure_2d_scan(find_files(source))
        first = next(p for row in grid for p in row if p)
        height, width = losa.load_image(first).shape
        out = losa.open_hdf_stream(output, (nrow, ncol, height, width),
                                   key_path=HDF_DATA_KEY)
        try:
            step = x_step or get_x_step(metadata, x_kind)
            if step:
                out.attrs["x_step"] = float(step)
                out.file.create_dataset(HDF_MAP_KEY, data=np.tile(
                    np.float32(step * np.arange(width)), (height, 1)))
            for row in range(nrow):
                for col in range(ncol):
                    if cancelled():
                        return "Cancelled", True
                    report(f"Image ({row}, {col}) of ({nrow}, {ncol})")
                    if grid[row][col]:
                        out[row, col] = losa.load_image(grid[row][col])
            out_path = out.file.filename
        finally:
            out.file.close()
        return f"Saved a 4D dataset {(nrow, ncol, height, width)} to " \
               f"{out_path}:/{HDF_DATA_KEY}", False
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
        step = x_step or get_x_step(metadata, x_kind,
                                    source if is_hdf else None, hdf_key)
        if not step:
            raise ValueError("Can't find the x-step in the metadata. "
                             "Please input it.")
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
            save_xy_image(Path(output) / names[i], x_map, image, xy_type,
                          zipped)
        return f"Saved {num} image(s) as {xy_type} files to {output}", False
    raise ValueError(f"Unknown conversion: {mode}")
