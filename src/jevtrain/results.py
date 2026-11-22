"""Common result schema; unsupported inputs remain in the accuracy denominator."""
from collections import defaultdict
from .metrics import metrics


def result_schema(records, checkpoint_sha256, sealed_test=None):
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate evaluation IDs')
    supported = [r for r in records if r.get('probabilities')]
    scored = metrics(supported) if supported else None
    tasks = defaultdict(lambda: {'n': 0, 'supported': 0, 'correct': 0})
    for row in records:
        entry = tasks[row['task']]
        entry['n'] += 1
        entry['supported'] += bool(row.get('probabilities'))
        entry['correct'] += bool(row.get('probabilities')) and bool(row['correct'])
    correct = sum(v['correct'] for v in tasks.values())
    return {'schema_version': 1, 'checkpoint_sha256': checkpoint_sha256,
            'evaluation_role': 'development_benchmark',
            'all': {'n': len(records), 'correct': correct,
                    'accuracy': correct / len(records) if records else None},
            'answerable': scored, 'by_task': dict(sorted(tasks.items())),
            'calibration': {'method': 'uncalibrated', 'ece': scored['ece'] if scored else None,
                            'denominator': len(supported), 'bins': 10},
            'sealed_test': sealed_test or {'evaluated': False, 'run_count': 0}}
