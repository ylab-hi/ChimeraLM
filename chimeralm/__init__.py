"""ChimeraLM: A Deep Learning Model to Detect Artificical Reads."""

from . import data, models, utils

__version__ = "1.0.5"

__all__ = ["data", "models", "ui", "utils"]


def __getattr__(name: str):
    """Lazy import for heavy modules (ui loads model at import time)."""
    if name == "ui":
        from . import ui

        return ui
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
