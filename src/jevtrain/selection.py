"""Checkpoint selection on the development set: accuracy, then macro accuracy, then lower NLL."""
import math
import sys


def selection_key(result):
    # A cohort can be entirely unsupported. Rank absent NLL below every finite
    # NLL without persisting Infinity in checkpoint-selection JSON.
    nll = result['nll'] if result['nll'] is not None else sys.float_info.max
    values = (result['accuracy'], result['macro_accuracy'], -nll)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite selection metrics')
    return values
