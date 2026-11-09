"""Build a checkpoint-bound validation calibration artifact."""
from .core.calibration import fit_temperature, scale
from .inference import sha256
from .metrics import metrics, read_rows


def calibrate(path, checkpoint):
    records = read_rows(path)
    if any(r.get('views', 1) != 1 or r.get('score_type') for r in records):
        raise ValueError('calibration requires single-view, direct-choice validation predictions')
    temperature = fit_temperature(records)
    scaled = [{**r, 'probabilities': scale(r['probabilities'], temperature)} for r in records]
    return {'method': 'temperature_scaling', 'temperature': temperature,
            'checkpoint_sha256': sha256(checkpoint), 'validation_sha256': sha256(path),
            'n': len(records), 'before': metrics(records), 'after': metrics(scaled)}
