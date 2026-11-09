"""Temperature scaling of a single-view categorical distribution."""
import math


def scale(probabilities, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1
                                for p in probabilities.values()):
        raise ValueError('invalid probabilities')
    if abs(sum(probabilities.values()) - 1) > 1e-4:
        raise ValueError('probabilities must sum to one')
    if temperature == 1:
        return dict(probabilities)
    logits = {k: math.log(max(p, 1e-30)) / temperature for k, p in probabilities.items()}
    peak = max(logits.values())
    exp = {k: math.exp(v - peak) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def fit_temperature(records):
    """Fit NLL on validation only; callers must bind the held-out cohort and model."""
    if not records or any(r.get('gold') not in r['probabilities'] for r in records):
        raise ValueError('labelled validation records required')
    def nll(log_t):
        return -sum(math.log(max(scale(r['probabilities'], math.exp(log_t))[r['gold']], 1e-30))
                    for r in records) / len(records)
    lo, hi = math.log(0.05), math.log(20)
    for _ in range(80):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    best = (lo + hi) / 2
    return math.exp(best) if nll(best) < nll(0) else 1.0
