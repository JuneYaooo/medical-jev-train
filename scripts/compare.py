"""Full-denominator benchmark comparison of two runs under one inference policy.

Unsupported rows count as wrong. Usage (repository root):
  PYTHONPATH=src python scripts/compare.py --base runs/bench-readout --candidate runs/bench-v1 \
    --output runs/comparison.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from jevtrain.metrics import comparison, input_hash, read_rows, write_json


def full_records(run, benchmark):
    records = {r['id']: r for r in read_rows(Path(run) / 'predictions.jsonl')}
    output = []
    for row in benchmark:
        r = records.get(row['id'])
        if r is None:  # unsupported: counted wrong
            r = {'id': row['id'], 'group_id': row['group_id'], 'task': row['task'], 'gold': row['gold'],
                 'input_sha256': input_hash(row['request']), 'prediction': None, 'correct': False,
                 'unsupported': True}
        output.append(r)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='runs/bench-readout', help='run directory of the reference model')
    parser.add_argument('--candidate', default='runs/bench-v1', help='run directory of the compared model')
    parser.add_argument('--benchmark', default='work/benchmark/medical_decision_v1.jsonl')
    parser.add_argument('--output', default='runs/comparison.json')
    args = parser.parse_args()
    bench = read_rows(args.benchmark)
    width = {r['id']: len(r['request']['questions']['decision']['criteria']) for r in bench}
    models = {'base': full_records(args.base, bench), 'candidate': full_records(args.candidate, bench)}
    report = {'n': len(bench), 'runs': {'base': args.base, 'candidate': args.candidate},
              'policy': 'direct <=52 options (8192 tokens), block elimination above 52',
              'models': {}, 'by_task': {}, 'subsets': {}}
    for name, records in models.items():
        report['models'][name] = {'correct': sum(r['correct'] for r in records),
                                  'accuracy': sum(r['correct'] for r in records) / len(records),
                                  'unsupported': sum(bool(r.get('unsupported')) for r in records)}
    report['paired'] = comparison(models['base'], models['candidate'])
    for task in sorted({r['task'] for r in bench}):
        values = {name: [r for r in records if r['task'] == task] for name, records in models.items()}
        n = len(values['base'])
        report['by_task'][task] = {'n': n, **{name: sum(r['correct'] for r in rs) for name, rs in values.items()}}
    for label, predicate in (('options_27_52', lambda r: 26 < width[r['id']] <= 52),
                             ('options_above_52', lambda r: width[r['id']] > 52)):
        keep = {row['id'] for row in bench if predicate(row)}
        report['subsets'][label] = {name: [sum(r['correct'] for r in records if r['id'] in keep), len(keep)]
                                    for name, records in models.items()}
    report['triage_prediction_counts'] = {
        name: dict(Counter(r.get('prediction') for r in records if r['task'] == 'triage_urgency'))
        for name, records in models.items()}
    write_json(args.output, report)
    v = report['paired']
    print(json.dumps(report['models'], indent=1))
    print(f"candidate vs base {100 * v['accuracy_delta']:+.2f}pp",
          [round(100 * x, 2) for x in v['group_bootstrap_95ci']],
          'fixed', v['wrong_to_right'], 'broke', v['right_to_wrong'])
    print(f"{'task':32s} {'n':>4s} {'base':>5s} {'cand':>5s}")
    for t, row in report['by_task'].items():
        print(f"{t:32s} {row['n']:4d} {row['base']:5d} {row['candidate']:5d}")


if __name__ == '__main__':
    main()
