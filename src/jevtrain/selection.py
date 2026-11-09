"""Checkpoint selection on the development set: accuracy, then macro accuracy, then lower NLL."""
import math


def selection_key(result):
    values = (result['accuracy'], result['macro_accuracy'], -result['nll'])
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite selection metrics')
    return values
