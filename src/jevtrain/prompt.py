"""Compatibility alias for jevtrain.core.prompt."""
import sys
from .core import prompt as _module
sys.modules[__name__] = _module
