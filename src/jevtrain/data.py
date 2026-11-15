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


def _isolated_components(rows, exclude_groups=(), exclude_inputs=()):
    """Connect case groups and exact model inputs before propagating exclusions.

    Fingerprints deliberately retain the displayed option order. No normalized
    text or answer label is used to join records. All member records are kept.
    """
    rows = list(rows)
    parents = list(range(len(rows)))
    sizes = [1] * len(rows)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            if sizes[first] < sizes[second]:
                first, second = second, first
            parents[second] = first
            sizes[first] += sizes[second]

    groups, inputs = {}, {}
    fingerprints = []
    for index, row in enumerate(rows):
        fingerprint = input_hash(row['request'])
        fingerprints.append(fingerprint)
        union(index, groups.setdefault(row['group_id'], index))
        union(index, inputs.setdefault(fingerprint, index))
    connected = defaultdict(list)
    for index, row in enumerate(rows):
        connected[find(index)].append((row, fingerprints[index]))

    exclude_groups, exclude_inputs = set(exclude_groups), set(exclude_inputs)
    kept, dropped = [], []
    for members in connected.values():
        members.sort(key=lambda item: (item[0]['id'], item[0]['task'], item[0]['group_id'], item[1],
                                       content_hash(item[0])))
        component = {'key': min(row['group_id'] for row, _ in members),
                     'task': members[0][0]['task'], 'rows': [row for row, _ in members]}
        excluded = any(row['group_id'] in exclude_groups or fingerprint in exclude_inputs
                       for row, fingerprint in members)
        (dropped if excluded else kept).append(component)
    return kept, dropped


def stratified_split(rows, per_task, names=('development', 'test'), seed=17,
                     max_fraction=0.3, exclude_groups=(), exclude_inputs=()):
    """Assign group/exact-input components to sets, stratified by task.

    ``per_task`` is a soft row target: a whole component may exceed it. A mixed
    task component is assigned to its lowest-id member's task. ``max_fraction``
    caps component counts in that representative-task stratum, not row counts
    for every task represented in mixed components. Small strata stay in pool.
    Any excluded group/input removes its entire connected component.
    Returns ({name: rows}, pool_rows, report).
    """
    require(per_task > 0 and names and len(set(names)) == len(names) and 'pool' not in names
            and 0 < max_fraction <= 1, 'invalid split settings')
    components, excluded = _isolated_components(rows, exclude_groups, exclude_inputs)
    task_groups = defaultdict(list)
    for component in components:
        task_groups[component['task']].append(component)
    sets = {name: [] for name in names}
    pool = []
    assigned_components = {name: 0 for name in [*names, 'pool']}
    for task in sorted(task_groups):
        groups = sorted(task_groups[task], key=lambda c: (group_hash(seed, c['key']), c['key']))
        cap = max(1, int(len(groups) * max_fraction)) if len(groups) > len(names) else 0
        cursor = 0
        for name in names:
            taken, used = [], 0
            while cursor < len(groups) and len(taken) < per_task and used < cap:
                taken.extend(groups[cursor]['rows'])
                cursor += 1
                used += 1
            sets[name].extend(taken)
            assigned_components[name] += used
        for component in groups[cursor:]:
            pool.extend(component['rows'])
        assigned_components['pool'] += len(groups) - cursor

    # Count actual tasks, not only the representative task used for allocation.
    report = {task: {name: 0 for name in [*names, 'pool']}
              for task in sorted({r['task'] for c in components for r in c['rows']})}
    for name, values in [*sets.items(), ('pool', pool)]:
        for row in values:
            report[row['task']][name] += 1
    return sets, pool, {
        'by_task': report,
        'dropped_by_exclusion': sum(len(c['rows']) for c in excluded),
        'sizes': {**{k: len(v) for k, v in sets.items()}, 'pool': len(pool)},
        'components': {'total': len(components) + len(excluded), 'excluded': len(excluded),
                       'remaining': len(components),
                       'mixed_task': sum(len({r['task'] for r in c['rows']}) > 1 for c in components),
                       'by_stratum': {t: len(task_groups[t]) for t in sorted(task_groups)},
                       'by_set': assigned_components},
        'target_shortfall': {task: {name: max(0, per_task - counts[name]) for name in names}
                             for task, counts in report.items()}}


def exclusion_keys(rows):
    return {r['group_id'] for r in rows}, {input_hash(r['request']) for r in rows}


def isolate(rows, heldouts):
    """Drop complete group/exact-input components touching a held-out row."""
    rows = list(rows)
    groups, inputs = set(), set()
    for held in heldouts:
        g, i = exclusion_keys(held)
        groups |= g
        inputs |= i
    components, _ = _isolated_components(rows, groups, inputs)
    retained = {id(row) for component in components for row in component['rows']}
    kept = [r for r in rows if id(r) in retained]
    return kept, len(rows) - len(kept)


def validate_split(sets, pool, exclude_groups=(), exclude_inputs=()):
    """Reject cross-set/excluded group or input overlap before writing files.

    Uses only standard-library data helpers; callers need no model or GPU.
    Errors include at most five record examples and never modify the inputs.
    """
    require('pool' not in sets, 'pool is reserved for training rows')
    excluded = {'group_id': set(exclude_groups), 'input_hash': set(exclude_inputs)}
    owners = {'group_id': {}, 'input_hash': {}}
    errors = []
    for name, values in [*sorted(sets.items()), ('pool', pool)]:
        for row in sorted(values, key=lambda r: (r['id'], r['group_id'])):
            for kind, key in [('group_id', row['group_id']), ('input_hash', input_hash(row['request']))]:
                if key in excluded[kind]:
                    errors.append(f"excluded {kind}: {name}/{row['id']}")
                owner = owners[kind].setdefault(key, (name, row['id']))
                if owner[0] != name:
                    errors.append(f"{kind}: {owner[0]}/{owner[1]} conflicts with {name}/{row['id']}")
                if len(errors) >= 5:
                    raise ValueError('split isolation failed: ' + '; '.join(errors[:5]))
    require(not errors, 'split isolation failed: ' + '; '.join(errors))


def hash_order(rows, seed):
    return sorted(rows, key=lambda r: content_hash([str(seed), r['id']]))


def sample_rows(rows, n, seed):
    return hash_order(rows, seed)[:n] if n is not None else list(rows)


def shuffled(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items
