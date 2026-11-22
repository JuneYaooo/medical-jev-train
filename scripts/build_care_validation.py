"""Evaluation-only triage set from the official CARE-Bench validation split.

The benchmark's 100 triage rows come from CARE-Bench public_test_1; validation
cases are disjoint from them. Rows use exactly the benchmark request format
(checked against all 100 benchmark rows). CC BY-NC 4.0: evaluation only, never training.
Usage (repository root):
  PYTHONPATH=src python scripts/build_care_validation.py
"""
import json
from pathlib import Path

PREFIXES = 'work/reliable_sources_round6_20261025/decision_gold/raw/care_bench__evaluation_prefixes_public.jsonl'
BENCHMARK = 'work/benchmark/medical_decision_v1.jsonl'
OUTPUT = Path('work/v4-check/care_validation_triage.jsonl')


def main():
    prefixes = [json.loads(line) for line in open(PREFIXES)]
    bench = [json.loads(line) for line in open(BENCHMARK) if '"triage_urgency"' in line]
    bench = [r for r in bench if r['task'] == 'triage_urgency']
    decision = bench[0]['request']['questions']['decision']
    by_id = {'triage_urgency:care_bench:' + p['prefix_id']: p for p in prefixes}
    for row in bench:  # same conversion reproduces every benchmark row
        p = by_id[row['id']]
        assert row['request']['state']['patient_messages_so_far'] == p['input_patient_turns']
        assert row['gold'] == p['gold_label'] and row['request']['questions']['decision'] == decision
    test_cases = {p['case_id'] for p in prefixes if p['split'] == 'public_test_1'}
    rows = [{'id': 'triage_urgency:care_val:' + p['prefix_id'], 'group_id': 'care_bench:' + p['case_id'],
             'task': 'triage_urgency', 'gold': p['gold_label'],
             'request': {'questions': {'decision': decision},
                         'state': {'patient_messages_so_far': p['input_patient_turns']}}}
            for p in prefixes if p['split'] == 'validation']
    assert not {r['group_id'].split(':', 1)[1] for r in rows} & test_cases
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(len(rows), 'rows ->', OUTPUT)


if __name__ == '__main__':
    main()
