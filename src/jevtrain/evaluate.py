"""Multi-view evaluation, per-task metrics and paired comparisons."""
from collections import Counter, defaultdict
import random

from .contracts import require
from .metrics import comparison, input_hash, metrics

from .data import option_keys, views
from .model import UnsupportedRow


def predict(model, rows, view_count=1, batch_size=8, large_options=False, tournament_above=None,
            absent_option=False):
    """One record per supported row; probabilities are averaged over cyclic views by semantic key.

    Rows with more than ``tournament_above`` options use block elimination
    (``large_options`` is the legacy spelling of ``tournament_above=26``), so
    ``tournament_above=52`` scores 27-52 options directly and only wider rows in
    blocks. ``absent_option`` uses the block-training subset format.

    Returns (records, unsupported). Each record keeps per-view predictions so
    position instability can be measured without another pass.
    """
    from .core.tournament import strategy, tournament
    if tournament_above is None and large_options:
        tournament_above = 26
    expanded, owners, unsupported = [], [], []
    partitioned = {}
    for index, row in enumerate(rows):
        if strategy(len(option_keys(row)), tournament_above) == 'tournament':
            try:
                partitioned[index] = tournament(model, row, batch_size=batch_size, absent_option=absent_option)
            except UnsupportedRow as error:
                unsupported.append({'id': row['id'], 'task': row.get('task'), 'reason': str(error)})
            continue
        try:
            model.encode(row)
        except UnsupportedRow as error:
            unsupported.append({'id': row['id'], 'task': row.get('task'), 'reason': str(error)})
            continue
        for v in views(row, view_count):
            try:
                model.encode(v)
            except UnsupportedRow:
                continue  # A reordered view can tokenize slightly longer.
            expanded.append(v)
            owners.append(index)
    probabilities = model.probabilities(expanded, batch_size=batch_size)
    grouped = defaultdict(list)
    for owner, v, p in zip(owners, expanded, probabilities):
        grouped[owner].append((v, p))
    records = []
    for index in sorted(grouped):
        row = rows[index]
        keys = option_keys(row)
        per_view = grouped[index]
        mean = {k: sum(p[k] for _, p in per_view) / len(per_view) for k in keys}
        winner = max(keys, key=mean.__getitem__)
        view_predictions = [max(p, key=p.__getitem__) for _, p in per_view]
        gold = row.get('gold')
        records.append({
            'id': row['id'], 'group_id': row['group_id'], 'task': row['task'],
            'input_sha256': input_hash(row['request']), 'gold': gold, 'option_order': keys,
            'prediction': winner, 'probabilities': mean,
            'correct': (winner == gold) if gold is not None else None,
            'views': len(per_view), 'view_predictions': view_predictions,
            'flipped': len(set(view_predictions)) > 1,
            'wrong_view_fraction': (sum(p != gold for p in view_predictions) / len(view_predictions)
                                    if gold is not None else None)})
    for index, output in partitioned.items():
        row = rows[index]
        records.append({'id': row['id'], 'group_id': row['group_id'], 'task': row['task'],
                        'input_sha256': input_hash(row['request']), 'gold': row.get('gold'),
                        'prediction': output['answer'], 'probabilities': output['scores'],
                        'correct': output['answer'] == row['gold'] if row.get('gold') else None,
                        'views': 1, 'flipped': False, 'score_type': output['score_type'],
                        'absent_option': output['absent_option'], 'tournament': output['tournament']})
    return records, unsupported


def accuracy_ci(records, seed=17, samples=1000):
    groups = defaultdict(list)
    for r in records:
        groups[r['group_id']].append(int(r['correct']))
    values = list(groups.values())
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        sample = [rng.choice(values) for _ in values]
        boot.append(sum(map(sum, sample)) / sum(map(len, sample)))
    boot.sort()
    return [boot[int(samples * 0.025)], boot[int(samples * 0.975) - 1]]


def summary(records, unsupported=()):
    labelled = [r for r in records if r['correct'] is not None]
    require(labelled, 'no labelled supported records to score')
    result = metrics(labelled)
    result['correct'] = sum(r['correct'] for r in labelled)
    result['accuracy_95ci'] = accuracy_ci(labelled)
    result['flip_rate'] = sum(r['flipped'] for r in labelled if r['views'] > 1) / max(
        1, sum(r['views'] > 1 for r in labelled))
    for task, value in result['by_task'].items():
        value['correct'] = sum(r['correct'] for r in labelled if r['task'] == task)
    result['unsupported'] = len(unsupported)
    result['unsupported_by_task'] = dict(Counter(u.get('task') for u in unsupported))
    return result


def paired(base, candidate):
    """Paired delta on the shared ids, with group bootstrap CI."""
    shared = {r['id'] for r in base} & {r['id'] for r in candidate}
    require(shared, 'no shared evaluation ids')
    b = [r for r in base if r['id'] in shared]
    c = [r for r in candidate if r['id'] in shared]
    return comparison(b, c)


def task_table(base, candidate):
    """Per-task accuracy for two record sets on shared ids."""
    shared = {r['id'] for r in base} & {r['id'] for r in candidate}
    rows = defaultdict(lambda: {'n': 0, 'base': 0, 'candidate': 0})
    index = {r['id']: r for r in candidate}
    for r in base:
        if r['id'] in shared:
            entry = rows[r['task']]
            entry['n'] += 1
            entry['base'] += r['correct']
            entry['candidate'] += index[r['id']]['correct']
    return {t: {**v, 'delta': (v['candidate'] - v['base']) / v['n']} for t, v in sorted(rows.items())}
