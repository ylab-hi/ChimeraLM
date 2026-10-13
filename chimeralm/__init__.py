"""ChimeraLM: A Deep Learning Model to Detect Artificical Reads."""

from importlib import import_module

from . import data, models, utils

__version__ = "1.0.5"

__all__ = ["data", "models", "ui", "utils"]


def __getattr__(name: str):
    """Lazy import for heavy modules (ui loads model at import time)."""
    if name == "ui":
        # `from . import ui` would call hasattr(package, "ui") -> __getattr__ -> recursion.
        return import_module(".ui", __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
