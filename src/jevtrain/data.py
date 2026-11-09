"""Choice records: option views, training targets and stratified splits."""
from collections import defaultdict
import random

from .contracts import content_hash, require
from .options import positional_reasons
from .metrics import input_hash, read_rows, write_json, write_rows  # noqa: F401  (re-exported for callers)


def options(row):
    return row['request']['questions']['decision']['criteria']


def option_keys(row):
    return list(options(row))


def can_permute(row):
    return not positional_reasons(row['request'])


def view(row, offset):
    """Cyclic option order. Offset 0 keeps the original order; labels stay semantic keys."""
    opts = options(row)
    keys = list(opts)
    offset %= len(keys)
    if offset == 0:
        return row
    order = keys[offset:] + keys[:offset]
    decision = {**row['request']['questions']['decision'], 'criteria': {k: opts[k] for k in order}}
    request = {**row['request'], 'questions': {**row['request']['questions'], 'decision': decision}}
    return {**row, 'request': request}


def view_offsets(row, count):
    """Evenly spaced cyclic offsets; position-dependent questions keep one view."""
    n = len(options(row))
    if count <= 1 or not can_permute(row):
        return [0]
    count = min(count, n)
    return sorted({round(i * n / count) % n for i in range(count)})


def views(row, count):
    return [view(row, offset) for offset in view_offsets(row, count)]


def random_offset(row, rng):
    return rng.randrange(len(options(row))) if can_permute(row) else 0


def target_distribution(row, keys, objective):
    """Target over the displayed key order: hard label with optional label smoothing.

    ``objective``: label_smoothing (eps), hard_tasks (list of task names that
    always use the unsmoothed hard label).
    """
    require(row.get('gold') in keys, 'gold must be one of the displayed options')
    n = len(keys)
    eps = float(objective.get('label_smoothing', 0.0))
    require(0 <= eps < 1, 'invalid label smoothing')
    if row.get('task') in set(objective.get('hard_tasks', [])):
        eps = 0.0
    return [(1 - eps) * (k == row['gold']) + eps / n for k in keys]


def group_hash(seed, group):
    return content_hash([str(seed), group])


def stratified_split(rows, per_task, names=('development', 'test'), seed=17,
                     max_fraction=0.3, exclude_groups=(), exclude_inputs=()):
    """Assign whole case groups to evaluation sets, stratified by task.

    Each set receives up to ``per_task`` rows per task. No set takes more than
    ``max_fraction`` of a task's groups, so small tasks keep training data.
    Rows sharing an excluded group or an excluded input hash are dropped.
    Returns ({name: rows}, pool_rows, report).
    """
    require(per_task > 0 and names and 0 < max_fraction <= 1, 'invalid split settings')
    exclude_groups, exclude_inputs = set(exclude_groups), set(exclude_inputs)
    by_group, dropped = defaultdict(list), 0
    for row in rows:
        if row['group_id'] in exclude_groups or input_hash(row['request']) in exclude_inputs:
            dropped += 1
            continue
        by_group[row['group_id']].append(row)
    # A group belongs to the task of its first row, so a case never spans sets.
    task_groups = defaultdict(list)
    for group, members in by_group.items():
        task_groups[members[0]['task']].append(group)
    sets = {name: [] for name in names}
    pool = []
    report = {}
    for task in sorted(task_groups):
        groups = sorted(task_groups[task], key=lambda g: group_hash(seed, g))
        cap = max(1, int(len(groups) * max_fraction)) if len(groups) > len(names) else 0
        cursor = 0
        counts = {}
        for name in names:
            taken, used = [], 0
            while cursor < len(groups) and len(taken) < per_task and used < cap:
                taken.extend(by_group[groups[cursor]])
                cursor += 1
                used += 1
            sets[name].extend(taken)
            counts[name] = len(taken)
        for group in groups[cursor:]:
            pool.extend(by_group[group])
        counts['pool'] = sum(len(by_group[g]) for g in groups[cursor:])
        report[task] = counts
    return sets, pool, {'by_task': report, 'dropped_by_exclusion': dropped,
                        'sizes': {**{k: len(v) for k, v in sets.items()}, 'pool': len(pool)}}


def exclusion_keys(rows):
    return {r['group_id'] for r in rows}, {input_hash(r['request']) for r in rows}


def isolate(rows, heldouts):
    """Drop rows sharing a group or exact input with any held-out row."""
    groups, inputs = set(), set()
    for held in heldouts:
        g, i = exclusion_keys(held)
        groups |= g
        inputs |= i
    kept = [r for r in rows if r['group_id'] not in groups and input_hash(r['request']) not in inputs]
    return kept, len(rows) - len(kept)


def hash_order(rows, seed):
    return sorted(rows, key=lambda r: content_hash([str(seed), r['id']]))


def sample_rows(rows, n, seed):
    return hash_order(rows, seed)[:n] if n is not None else list(rows)


def shuffled(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items
