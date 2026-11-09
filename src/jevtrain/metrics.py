"""Data I/O, input fingerprints and grouped evaluation metrics."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .contracts import content_hash, require


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    temporary.replace(path)

def read_rows(path):
    import gzip
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def input_hash(request):
    # The prompt assigns letter positions in insertion order. Canonical JSON
    # alone sorts keys and would incorrectly erase an option permutation.
    return content_hash({'request': request,
                         'option_order': list(request['questions']['decision']['criteria'])})

def metrics(records):
    require(bool(records), 'cannot score empty evaluation')
    require(all(r['correct'] is not None for r in records), 'evaluation needs labels')
    tasks, bins = defaultdict(list), defaultdict(list)
    nll, brier = 0.0, 0.0
    for r in records:
        tasks[r['task']].append(r['correct'])
        p = r['probabilities']
        nll -= math.log(max(p[r['gold']], 1e-30))
        brier += sum((v - (k == r['gold'])) ** 2 for k, v in p.items())
        confidence = max(p.values())
        bins[min(int(confidence * 10), 9)].append((confidence, r['correct']))
    per = {task: {'n': len(values), 'accuracy': sum(values) / len(values)} for task, values in tasks.items()}
    return {'n': len(records), 'groups': len({r['group_id'] for r in records}),
            'accuracy': sum(r['correct'] for r in records) / len(records),
            'macro_accuracy': sum(v['accuracy'] for v in per.values()) / len(per),
            'nll': nll / len(records), 'brier': brier / len(records), 'by_task': per,
            'reliability': [{'lower': i / 10, 'upper': (i + 1) / 10, 'n': len(values),
                             'confidence': sum(p for p, _ in values) / len(values),
                             'accuracy': sum(c for _, c in values) / len(values)}
                            for i, values in sorted(bins.items())],
            'confidence_by_correctness': {str(correct).lower():
                (sum(max(r['probabilities'].values()) for r in records if bool(r['correct']) == correct) /
                 sum(bool(r['correct']) == correct for r in records))
                if any(bool(r['correct']) == correct for r in records) else None
                for correct in (True, False)},
            'ece': sum(abs(sum(p - c for p, c in values)) for values in bins.values()) / len(records)}

def comparison(base, candidate, seed=17):
    require(len({r['id'] for r in base}) == len(base), 'duplicate base prediction')
    indexed = {r['id']: r for r in candidate}
    require(len(indexed) == len(candidate) and set(indexed) == {r['id'] for r in base}, 'evaluation cohorts differ')
    groups = defaultdict(list)
    fixed = broken = wrong = 0
    for b in base:
        c = indexed[b['id']]
        require(all(b[k] == c[k] for k in ('input_sha256', 'group_id', 'task', 'gold')), 'paired evaluation input changed')
        groups[b['group_id']].append(int(c['correct']) - int(b['correct']))
        wrong += not b['correct']
        fixed += not b['correct'] and c['correct']
        broken += b['correct'] and not c['correct']
    require(groups, 'empty paired comparison')
    values = list(groups.values())
    rng = random.Random(seed)
    boot = []
    for _ in range(1000):
        sample = [rng.choice(values) for _ in values]
        boot.append(sum(map(sum, sample)) / sum(map(len, sample)))
    boot.sort()
    return {'accuracy_delta': (fixed - broken) / len(base),
            'group_bootstrap_95ci': [boot[24], boot[974]], 'groups': len(groups),
            'wrong_to_right': fixed, 'right_to_wrong': broken,
            'repair_rate': fixed / wrong if wrong else None}
