"""Compatibility alias for jevtrain.core.contracts."""
import sys
from .core import contracts as _module
sys.modules[__name__] = _module
