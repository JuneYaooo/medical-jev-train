"""Compatibility alias for jevtrain.core.model."""
import sys
from .core import model as _module
sys.modules[__name__] = _module
