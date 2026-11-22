"""Report text-question overlap candidates; never changes a split or deletes rows.

python scripts/audit_duplicates.py --train train.jsonl --evaluation test.jsonl --output audit.json
"""
import argparse
from collections import defaultdict
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata


ALGORITHM = {
    'name': 'clinical-question-sequence', 'version': '1', 'thresholds': [0.9, 0.95],
    'text_normalization': 'Unicode NFKC; collapse whitespace; strip ends; preserve case and punctuation',
    'option_matching': 'sorted normalized option-text multiset and normalized correct-answer text',
    'similarity': 'difflib.SequenceMatcher.ratio; autojunk=False',
    'selection': 'highest-scoring training candidate per evaluation id; train id breaks ties',
    'eligibility': 'nonempty string state.clinical_question, string choices and valid gold',
}


def normalized(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip()


def question(row):
    value = row.get('request', {}).get('state', {}).get('clinical_question')
    return value if isinstance(value, str) and value.strip() else None


def option_signature(row):
    criteria = row.get('request', {}).get('questions', {}).get('decision', {}).get('criteria', {})
    if row.get('gold') not in criteria or not criteria or not all(isinstance(v, str) for v in criteria.values()):
        return None
    return tuple(sorted(normalized(v) for v in criteria.values())), normalized(criteria[row['gold']])


def audit_candidates(train_rows, evaluation_rows):
    """Screen compatible choice lists, retaining full text for human adjudication.

    Similarity is not semantic equivalence: dose/negation changes, extra patient
    fields and positional option references require review even at high scores.
    """
    index = defaultdict(list)
    evaluation_ids = set()
    evaluation_n = eligible_evaluation = 0
    for row in evaluation_rows:
        if row['id'] in evaluation_ids:
            raise ValueError(f"duplicate evaluation id: {row['id']}")
        evaluation_ids.add(row['id'])
        evaluation_n += 1
        text, signature = question(row), option_signature(row)
        if text is not None and signature is not None:
            index[signature].append((row, normalized(text)))
            eligible_evaluation += 1
    best = {}
    train_n = eligible_train = 0
    train_ids = set()
    for train in train_rows:
        if train['id'] in train_ids:
            raise ValueError(f"duplicate training id: {train['id']}")
        train_ids.add(train['id'])
        train_n += 1
        text, signature = question(train), option_signature(train)
        if text is None or signature is None:
            continue
        eligible_train += 1
        text = normalized(text)
        for evaluation, evaluation_text in index.get(signature, ()):
            if 2 * min(len(text), len(evaluation_text)) / (len(text) + len(evaluation_text)) < 0.9:
                continue
            matcher = difflib.SequenceMatcher(None, evaluation_text, text, autojunk=False)
            if matcher.quick_ratio() < 0.9:
                continue
            similarity = matcher.ratio()
            if similarity < 0.9:
                continue
            previous = best.get(evaluation['id'])
            if previous is not None and (similarity < previous['similarity'] or
                                         (similarity == previous['similarity'] and train['id'] >= previous['train_id'])):
                continue
            ec = evaluation['request']['questions']['decision']['criteria']
            tc = train['request']['questions']['decision']['criteria']
            other_state_fields = sorted((set(evaluation['request']['state']) | set(train['request']['state']))
                                        - {'clinical_question'})
            best[evaluation['id']] = {
                'evaluation_id': evaluation['id'], 'train_id': train['id'], 'similarity': similarity,
                'evaluation_question': question(evaluation), 'train_question': question(train),
                'evaluation_options': ec, 'train_options': tc,
                'evaluation_gold': evaluation['gold'], 'train_gold': train['gold'],
                'option_order_or_keys_changed': list(ec.items()) != list(tc.items()),
                'additional_state_fields': other_state_fields,
                'manual_review_required': True,
            }
    candidates = sorted(best.values(), key=lambda c: c['evaluation_id'])
    return {'algorithm': dict(ALGORITHM), 'candidate_only': True,
            'warning': 'Text similarity does not establish equivalent clinical questions or training exposure. '
                       'Review doses, negation, other patient fields and positional option references. '
                       'Candidates do not alter splitting or exclusion.',
            'rows': {'train': train_n, 'evaluation': evaluation_n,
                     'eligible_train': eligible_train, 'eligible_evaluation': eligible_evaluation},
            'counts': {str(t): sum(c['similarity'] >= t for c in candidates) for t in ALGORITHM['thresholds']},
            'candidates': candidates}


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(Path(path).resolve()), 'sha256': digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    inputs = [Path(args.train).resolve(), Path(args.evaluation).resolve()]
    if output in inputs or (output.exists() and any(output.samefile(path) for path in inputs)):
        parser.error('--output must not overwrite either input')
    identities = {'train': file_identity(args.train), 'evaluation': file_identity(args.evaluation)}
    report = audit_candidates(read_rows(args.train), read_rows(args.evaluation))
    if identities != {'train': file_identity(args.train), 'evaluation': file_identity(args.evaluation)}:
        raise ValueError('audit inputs changed during screening')
    report['inputs'] = identities
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'candidate_counts': report['counts'], 'output': str(output)}))


if __name__ == '__main__':
    main()
