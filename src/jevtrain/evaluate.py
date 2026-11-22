"""Multi-view evaluation, per-task metrics and paired comparisons."""
from collections import Counter, defaultdict
import math
import random

from .contracts import require
from .metrics import comparison, input_hash, metrics

from .data import option_keys, views
from .model import UnsupportedRow
from .evaluation_cohort import full_records, indexed


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


def _valid_probabilities(record):
    probabilities = record.get('probabilities')
    if not isinstance(probabilities, dict) or record['gold'] not in probabilities:
        return False
    if 'option_order' in record and set(probabilities) != set(record['option_order']):
        return False
    values = list(probabilities.values())
    return (bool(values) and all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                                for p in values)
            and math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5))


def prediction_concentration(records, min_n=5, max_share=0.8, min_excess=0.2):
    """Flag tasks whose answers collapse onto one option far beyond that option's gold share."""
    tasks = defaultdict(list)
    for r in records:
        if r.get('prediction') is not None and r.get('gold') is not None and not r.get('unsupported'):
            tasks[r['task']].append(r)
    flagged = {}
    for task, values in sorted(tasks.items()):
        if len(values) < min_n:
            continue
        answer, count = Counter(r['prediction'] for r in values).most_common(1)[0]
        share = count / len(values)
        gold_share = sum(r['gold'] == answer for r in values) / len(values)
        if share >= max_share and share - gold_share >= min_excess:
            flagged[task] = {'n': len(values), 'answer': answer, 'prediction_share': share, 'gold_share': gold_share}
    return flagged


def summary(records, unsupported=(), *, rows=None):
    """Accuracy over all labelled inputs; probability metrics over valid supported predictions.

    Legacy summary(records) remains valid for complete supported cohorts. Supply
    authoritative rows whenever any inputs are unsupported or missing.
    """
    records, unsupported = list(records), list(unsupported)
    require(rows is not None or not unsupported, 'rows are required to score unsupported inputs')
    outcomes = full_records(rows, records, unsupported) if rows is not None else list(indexed(records, 'prediction').values())
    labelled = [r for r in outcomes if r['correct'] is not None]
    require(labelled, 'no labelled records to score')
    supported = [r for r in labelled if r.get('prediction') is not None and not r.get('unsupported')]
    probabilistic = [r for r in supported if _valid_probabilities(r)]
    result = metrics(probabilistic) if probabilistic else {
        'nll': None, 'brier': None, 'ece': None, 'reliability': [],
        'confidence_by_correctness': {'true': None, 'false': None}}
    result.update(input_n=len(outcomes), n=len(labelled), unlabelled_n=len(outcomes) - len(labelled),
                  groups=len({r['group_id'] for r in labelled}), supported_n=len(supported),
                  coverage=len(supported) / len(labelled),
                  supported_accuracy=(sum(r['correct'] for r in supported) / len(supported) if supported else None),
                  probability_metrics_n=len(probabilistic))
    result['correct'] = sum(r['correct'] for r in labelled)
    result['accuracy'] = result['correct'] / len(labelled)
    result['accuracy_95ci'] = accuracy_ci(labelled)
    result['flip_rate'] = sum(r.get('flipped', False) for r in supported if r.get('views', 1) > 1) / max(
        1, sum(r.get('views', 1) > 1 for r in supported))
    tasks = defaultdict(list)
    for r in labelled:
        tasks[r['task']].append(r['correct'])
    result['by_task'] = {task: {'n': len(values), 'correct': sum(values), 'accuracy': sum(values) / len(values)}
                         for task, values in sorted(tasks.items())}
    result['macro_accuracy'] = sum(v['accuracy'] for v in result['by_task'].values()) / len(tasks)
    rejected = [r for r in outcomes if r.get('unsupported') or r.get('prediction') is None]
    result['unsupported'] = len(rejected)
    result['unsupported_by_task'] = dict(Counter(r.get('task') for r in rejected))
    result['collapsed_tasks'] = prediction_concentration(labelled)
    return result


def _paired_records(base, candidate, rows):
    if rows is not None:
        rows = list(rows)
        base, candidate = full_records(rows, base), full_records(rows, candidate)
    else:
        base, candidate = list(base), list(candidate)
    b, c = indexed(base, 'base prediction'), indexed(candidate, 'candidate prediction')
    require(b.keys() == c.keys(), 'evaluation cohorts differ; provide rows for full-cohort comparison')
    for uid in b:
        require(all(b[uid][key] == c[uid][key] for key in ('group_id', 'task', 'gold', 'input_sha256')),
                'paired evaluation input changed: ' + str(uid))
    return ([r for r in base if r['correct'] is not None],
            [r for r in candidate if r['correct'] is not None])


def paired(base, candidate, *, rows=None):
    """Paired delta on identical cohorts; missing predictions count wrong with rows."""
    b, c = _paired_records(base, candidate, rows)
    return comparison(b, c)


def task_table(base, candidate, *, rows=None):
    """Per-task accuracy on the same full cohort used by paired()."""
    base, candidate = _paired_records(base, candidate, rows)
    rows = defaultdict(lambda: {'n': 0, 'base': 0, 'candidate': 0})
    index = {r['id']: r for r in candidate}
    for r in base:
        entry = rows[r['task']]
        entry['n'] += 1
        entry['base'] += r['correct']
        entry['candidate'] += index[r['id']]['correct']
    return {t: {**v, 'delta': (v['candidate'] - v['base']) / v['n']} for t, v in sorted(rows.items())}
