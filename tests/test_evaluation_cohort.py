"""CPU regressions for full-cohort scoring and comparison."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jevtrain.evaluate import paired, summary, task_table
from jevtrain.metrics import input_hash


def row(uid, task='test', labelled=True):
    return {'id': uid, 'group_id': 'g:' + uid, 'task': task,
            'gold': 'yes' if labelled else None,
            'request': {'state': {'clinical_question': uid}, 'questions': {'decision': {
                'type': 'choice', 'instructions': 'choose', 'criteria': {'yes': 'yes', 'no': 'no'}}}}}


def prediction(r, correct=True):
    choice = 'yes' if correct else 'no'
    return {'id': r['id'], 'group_id': r['group_id'], 'task': r['task'], 'gold': r['gold'],
            'input_sha256': input_hash(r['request']), 'prediction': choice,
            'correct': correct if r['gold'] is not None else None,
            'probabilities': {choice: .75, ('no' if correct else 'yes'): .25},
            'views': 1, 'flipped': False}


class CohortTests(unittest.TestCase):
    def test_unsupported_and_missing_count_wrong(self):
        rows = [row('a'), row('b'), row('c', 'other')]
        result = summary([prediction(rows[0])], [{'id': 'b', 'task': 'test'}], rows=rows)
        self.assertEqual((result['n'], result['correct'], result['unsupported']), (3, 1, 2))
        self.assertAlmostEqual(result['accuracy'], 1 / 3)
        self.assertAlmostEqual(result['macro_accuracy'], .25)
        self.assertEqual(result['supported_n'], 1)
        self.assertAlmostEqual(result['coverage'], 1 / 3)
        self.assertEqual(result['supported_accuracy'], 1)
        self.assertEqual(result['probability_metrics_n'], 1)
        self.assertEqual(result['by_task']['other'], {'n': 1, 'correct': 0, 'accuracy': 0})

    def test_all_unsupported_has_no_fabricated_probabilities(self):
        result = summary([], [{'id': 'a', 'task': 'test'}], rows=[row('a'), row('b')])
        self.assertEqual((result['n'], result['accuracy'], result['macro_accuracy']), (2, 0, 0))
        self.assertEqual(result['accuracy_95ci'], [0, 0])
        self.assertEqual(result['coverage'], 0)
        self.assertIsNone(result['supported_accuracy'])
        self.assertEqual(result['probability_metrics_n'], 0)
        for name in ('nll', 'brier', 'ece'):
            self.assertIsNone(result[name])
        from jevtrain.selection import selection_key
        self.assertLess(selection_key(result), selection_key({**result, 'nll': 100.0}))

    def test_unlabelled_rows_do_not_enter_denominator(self):
        rows = [row('a'), row('b', labelled=False), row('c', labelled=False)]
        result = summary([prediction(rows[0]), prediction(rows[1])], rows=rows)
        self.assertEqual((result['input_n'], result['n'], result['unlabelled_n']), (3, 1, 2))
        self.assertEqual(result['accuracy'], 1)
        with self.assertRaisesRegex(ValueError, 'label'):
            summary([], rows=[row('u', labelled=False)])

    def test_supported_answers_without_probabilities_still_count(self):
        r = row('a')
        record = prediction(r)
        del record['probabilities']
        result = summary([record], rows=[r])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['coverage'], 1)
        self.assertEqual(result['probability_metrics_n'], 0)

    def test_invalid_probability_distributions_are_excluded_from_probability_metrics(self):
        r = row('a')
        for probabilities in [{'yes': float('nan'), 'no': 0.5}, {'yes': -1, 'no': 2},
                              {'yes': .5, 'no': .4}, {'yes': .5, 'extra': .5}]:
            with self.subTest(probabilities=probabilities):
                result = summary([{**prediction(r), 'probabilities': probabilities}], rows=[r])
                self.assertEqual(result['accuracy'], 1)
                self.assertEqual(result['probability_metrics_n'], 0)
                self.assertIsNone(result['nll'])

    def test_unsupported_without_cohort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'rows'):
            summary([prediction(row('a'))], [{'id': 'b'}])

    def test_counts_match_published_examples(self):
        for total, supported, correct in [(1557, 1555, 1307), (1514, 1414, 1028)]:
            with self.subTest(total=total):
                rows = [row(str(i)) for i in range(total)]
                for i, r in enumerate(rows):
                    r['group_id'] = str(i // 100)
                result = summary([prediction(r, i < correct) for i, r in enumerate(rows[:supported])],
                                 rows=rows)
                self.assertEqual((result['n'], result['correct']), (total, correct))
                self.assertAlmostEqual(result['accuracy'], correct / total)

    def test_invalid_cohorts_and_prediction_metadata_are_rejected(self):
        from jevtrain.evaluation_cohort import full_records
        a = row('a')
        for field, bad in [('gold', 'no'), ('group_id', 'other'), ('task', 'other'),
                           ('input_sha256', 'bad'), ('prediction', 'outside'), ('correct', False)]:
            with self.subTest(field=field):
                record = {**prediction(a), field: bad}
                with self.assertRaises(ValueError):
                    full_records([a], [record])
        with self.assertRaises(ValueError):
            full_records([a, a], [])
        with self.assertRaises(ValueError):
            full_records([a], [prediction(a), prediction(a)])
        with self.assertRaises(ValueError):
            full_records([a], [prediction(row('extra'))])
        with self.assertRaises(ValueError):
            full_records([a], [], [{'id': 'a'}, {'id': 'a'}])
        with self.assertRaises(ValueError):
            full_records([a], [prediction(a)], [{'id': 'a'}])
        changed = copy.deepcopy(a)
        changed['request']['state']['clinical_question'] = 'changed'
        with self.assertRaises(ValueError):
            full_records([changed], [prediction(a)])

    def test_paired_comparison_never_silently_intersects(self):
        a, b = row('a'), row('b')
        with self.assertRaises(ValueError):
            paired([prediction(a)], [prediction(b)])
        result = paired([prediction(a)], [prediction(b)], rows=[a, b])
        self.assertEqual(result['accuracy_delta'], 0)
        self.assertEqual((result['wrong_to_right'], result['right_to_wrong']), (1, 1))
        table = task_table([prediction(a)], [prediction(b)], rows=[a, b])
        self.assertEqual(table['test']['n'], 2)

    def test_cli_and_script_compare_agree_on_full_cohort(self):
        from jevtrain.cli import main
        script = Path(__file__).resolve().parents[1] / 'scripts/compare.py'
        spec = importlib.util.spec_from_file_location('comparison_script', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a, b = row('a'), row('b')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(path, values):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
            write(root / 'input.jsonl', [a, b])
            write(root / 'base/predictions.jsonl', [prediction(a)])
            write(root / 'candidate/predictions.jsonl', [prediction(b)])
            out = io.StringIO()
            with redirect_stdout(out):
                main(['compare', '--base', str(root / 'base/predictions.jsonl'),
                      '--candidate', str(root / 'candidate/predictions.jsonl'), '--input', str(root / 'input.jsonl')])
            cli_result = json.loads(out.getvalue())
            with patch.object(sys, 'argv', ['compare.py', '--base', str(root / 'base'), '--candidate', str(root / 'candidate'),
                                           '--benchmark', str(root / 'input.jsonl'), '--output', str(root / 'out.json')]), redirect_stdout(io.StringIO()):
                module.main()
            script_result = json.loads((root / 'out.json').read_text(encoding='utf-8'))
            self.assertEqual(cli_result['paired'], script_result['paired'])
            self.assertEqual(script_result['models']['base']['accuracy'], .5)


if __name__ == '__main__':
    unittest.main()
