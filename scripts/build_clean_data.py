"""Build a leakage-reduced training pool and development set without touching the release files.

Removes from train: exact normalized stems and near-duplicates (5-gram Jaccard >= 0.8) of any
development, test or benchmark row, plus every row of an item whose duplicates disagree on the answer.
Removes from development: rows whose normalized stem also appears in test or benchmark.
Test and benchmark are never modified.
"""
import collections as C
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

root = Path(sys.argv[1])
source = root / 'work/data/release'
target = root / 'work/data/clean'
target.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.8


def read(path):
    with open(path, encoding='utf-8') as stream:
        return [(line, json.loads(line)) for line in stream]


def norm(value):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', str(value)).lower())


def stem(row):
    return norm(json.dumps(row['request']['state'], ensure_ascii=False, sort_keys=True))


def options(row):
    return row['request']['questions']['decision']['criteria']


def digest(text):
    return hashlib.sha1(text.encode()).hexdigest()


def item(row):
    return digest(stem(row) + '#' + '|'.join(sorted(norm(v) for v in options(row).values())))


def grams(text, size=5):
    return {text[i:i + size] for i in range(max(0, len(text) - size + 1))}


train = read(source / 'train.jsonl')
development = read(source / 'development.jsonl')
test = read(source / 'test.jsonl')
benchmark = read(root / 'work/benchmark/medical_decision_v1.jsonl')
evaluation = [r for _, r in development + test + benchmark]

protected = {digest(stem(r)) for _, r in test + benchmark}
clean_development = [(l, r) for l, r in development if digest(stem(r)) not in protected]

eval_stems = {digest(stem(r)) for r in evaluation}
drop = C.defaultdict(set)
texts = [stem(r) for _, r in train]
for index, (_, row) in enumerate(train):
    if digest(texts[index]) in eval_stems:
        drop['exact_stem'].add(index)

by_item = C.defaultdict(list)
for index, (_, row) in enumerate(train):
    by_item[item(row)].append(index)
for members in by_item.values():
    answers = {norm(options(train[i][1]).get(train[i][1].get('gold'), '')) for i in members}
    if len(answers) > 1:
        drop['conflicting_gold'].update(members)

frequency = C.Counter()
train_grams = []
for text in texts:
    current = grams(text) if len(text) >= 20 else set()
    train_grams.append(current)
    frequency.update(current)
index_by_gram = C.defaultdict(list)
for position, current in enumerate(train_grams):
    for gram in sorted(current, key=lambda g: (frequency[g], g))[:8]:
        index_by_gram[gram].append(position)
for row in evaluation:
    text = stem(row)
    if len(text) < 20:
        continue
    current = grams(text)
    candidates = C.Counter()
    for gram in sorted(current, key=lambda g: (frequency.get(g, 0), g))[:24]:
        if frequency.get(gram, 0) <= 500:
            candidates.update(index_by_gram.get(gram, ()))
    for position, _ in candidates.most_common(200):
        other = train_grams[position]
        if other and len(current & other) / len(current | other) >= THRESHOLD:
            drop['near_duplicate'].add(position)

removed = set().union(*drop.values()) if drop else set()
kept = [line for index, (line, _) in enumerate(train) if index not in removed]
(target / 'train.jsonl').write_text(''.join(kept), encoding='utf-8')
(target / 'development.jsonl').write_text(''.join(l for l, _ in clean_development), encoding='utf-8')
report = {'train_before': len(train), 'train_after': len(kept), 'removed': len(removed),
          'removed_by_reason': {k: len(v) for k, v in drop.items()},
          'removed_by_task': dict(C.Counter(train[i][1]['task'] for i in removed).most_common()),
          'development_before': len(development), 'development_after': len(clean_development),
          'threshold': THRESHOLD}
(target / 'clean_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
