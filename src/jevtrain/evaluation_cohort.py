"""Validate predictions against an authoritative cohort without inventing probabilities."""
from .contracts import require
from .metrics import input_hash


def indexed(records, name):
    result = {}
    for record in records:
        uid = record['id']
        require(uid not in result, 'duplicate ' + name + ' id: ' + str(uid))
        result[uid] = record
    return result


def full_records(rows, records, unsupported=()):
    """One outcome per input row, with missing/unsupported predictions counted wrong.

    Input rows own the cohort, labels and grouping. Prediction metadata must match
    rather than being silently replaced. Unlabelled rows retain correct=None.
    """
    cohort = indexed(rows, 'evaluation')
    predictions = indexed(records, 'prediction')
    rejected = indexed(unsupported, 'unsupported')
    require(not (predictions.keys() & rejected.keys()), 'prediction also marked unsupported')
    require(not ((predictions.keys() | rejected.keys()) - cohort.keys()), 'prediction ids outside evaluation cohort')
    output = []
    for uid, row in cohort.items():
        metadata = {'id': uid, 'group_id': row['group_id'], 'task': row['task'],
                    'gold': row.get('gold'), 'input_sha256': input_hash(row['request'])}
        gold = metadata['gold']
        options = row['request']['questions']['decision']['criteria']
        require(gold is None or gold in options, 'evaluation gold is not an option: ' + str(uid))
        if uid in predictions:
            record = predictions[uid]
            for field in ('group_id', 'task', 'gold', 'input_sha256'):
                require(field in record and record[field] == metadata[field],
                        'prediction ' + field + ' differs from evaluation: ' + str(uid))
            choice = record.get('prediction')
            require('option_order' not in record or record['option_order'] == list(options),
                    'prediction option order differs from evaluation: ' + str(uid))
            require(choice is None or choice in options, 'prediction is not an option: ' + str(uid))
            correct = choice == gold if gold is not None else None
            require('correct' not in record or record['correct'] == correct,
                    'prediction correctness differs from answer: ' + str(uid))
            output.append({**record, **metadata, 'option_order': list(options), 'correct': correct,
                           'unsupported': choice is None})
        else:
            rejection = rejected.get(uid, {})
            for field in ('group_id', 'task', 'gold', 'input_sha256'):
                require(field not in rejection or rejection[field] == metadata[field],
                        'unsupported ' + field + ' differs from evaluation: ' + str(uid))
            output.append({**metadata, 'prediction': None,
                           'correct': False if gold is not None else None,
                           'unsupported': True, 'reason': rejection.get('reason', 'missing prediction'),
                           'views': 0, 'flipped': False})
    return output
