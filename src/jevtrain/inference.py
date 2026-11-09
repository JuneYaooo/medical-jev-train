"""Compatibility alias for jevtrain.core.inference."""
import sys
from .core import inference as _module
sys.modules[__name__] = _module
