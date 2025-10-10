from importlib import metadata

try:
    from ._version import __version__
except ModuleNotFoundError:
    # Fallback for editable installs where setuptools_scm has not written _version.py yet.
    try:
        __version__ = metadata.version("metriq-gym")
    except metadata.PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = ["__version__"]
