"""Surreal Event-based Depth / Segmentation Package.

This package provides models, dataset utilities and real-time viewers for
processing event camera streams. It is being prepared for PyPI
publication; APIs may change.
"""

# Eagerly import commonly used submodules so that attribute completion works:
from . import config  # noqa: F401
from . import utils  # noqa: F401
from . import models  # noqa: F401
from . import stream  # noqa: F401

# Re-export primary model classes / functions for convenience
try:  # pragma: no cover - optional if dependency tree changes
    from .models.ConvLSTM import EConvlstm  # noqa: F401
except Exception:  # pragma: no cover
    EConvlstm = None  # type: ignore

# Optionally expose stream helpers
try:  # pragma: no cover
    from .stream import load_model, run  # noqa: F401
except Exception:  # pragma: no cover
    load_model = None  # type: ignore
    run = None  # type: ignore

__all__ = [
    "config",
    "utils",
    "models",
    "stream",
    "EConvlstm",
    "load_model",
    "run",
]

__version__ = "1.0.0"
