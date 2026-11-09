"""Compatibility alias for jevtrain.core.options."""
import sys
from .core import options as _module
sys.modules[__name__] = _module
