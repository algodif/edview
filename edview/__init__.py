from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("edview")
except PackageNotFoundError:  # Running from source without installing
    __version__ = "unknown"
