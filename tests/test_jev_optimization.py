import copy
import tempfile
import unittest
import json
from pathlib import Path

from jevtrain.core.calibration import fit_temperature, scale
from jevtrain.core.tournament import tournament
from jevtrain.model import UnsupportedRow
from jevtrain.selection import selection_key


def row(uid, task='test', group=None):
    return {'id': uid, 'group_id': group or uid, 'task': task, 'gold': '1',
            'request': {'state': {'question': uid}, 'questions': {'decision': {
                'instructions': 'choose', 'criteria': {'0': 'zero', '1': 'one'}}}}}


class OptimizationTests(unittest.TestCase):
    def test_sealed_attempt_persists_failure_and_blocks_repeat(self):
        from jevtrain.sealed import sealed_attempt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input').write_text('cohort')
            (root / 'weights').write_text('weights')
            with self.assertRaises(RuntimeError):
                with sealed_attempt(root / 'ledger.json', root / 'input', root / 'weights'):
                    raise RuntimeError('interrupted')
            self.assertEqual(json.loads((root / 'ledger.json').read_text())['attempts'][0]['status'], 'failed')
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                with sealed_attempt(root / 'ledger.json', root / 'input', root / 'weights'):
                    self.fail('must not rerun')

    def test_config_paths_are_relative_to_the_path_file(self):
        from jevtrain.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nested').mkdir()
            (root / 'paths.json').write_text(json.dumps({'work': './data'}))
            (root / 'nested/train.json').write_text(json.dumps({
                'paths_file': '../paths.json', 'model': {'path': '${paths.work}/base'}}))
            self.assertEqual(load_config(root / 'nested/train.json')['model']['path'], str((root / 'data/base').resolve()))

    def test_selection_prefers_accuracy_then_macro_then_nll(self):
        a = {'accuracy': .9, 'macro_accuracy': .6, 'nll': .3}
        b = {'accuracy': .8, 'macro_accuracy': .7, 'nll': .4}
        self.assertGreater(selection_key(a), selection_key(b))
        c = {'accuracy': .9, 'macro_accuracy': .7, 'nll': .5}
        self.assertGreater(selection_key(c), selection_key(a))
        d = {'accuracy': .9, 'macro_accuracy': .7, 'nll': .4}
        self.assertGreater(selection_key(d), selection_key(c))
        with self.assertRaises(ValueError):
            selection_key({'accuracy': float('nan'), 'macro_accuracy': .5, 'nll': .1})

    def test_temperature_reduces_overconfident_validation_nll(self):
        records = [{'gold': 'a' if i < 7 else 'b', 'probabilities': {'a': .99, 'b': .01}}
                   for i in range(10)]
        temperature = fit_temperature(records)
        self.assertGreater(temperature, 1)
        self.assertAlmostEqual(scale(records[0]['probabilities'], temperature)['a'], .7, places=4)
        self.assertEqual(scale({'a': .75, 'b': .25}), {'a': .75, 'b': .25})
        for t in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                scale({'a': 1}, t)

    def test_tournament_preserves_identity_and_never_reads_gold(self):
        class Model:
            def probabilities(self, rows, batch_size):
                output = []
                for r in rows:
                    self.assert_no_gold(r)
                    keys = list(r['request']['questions']['decision']['criteria'])
                    assert len(keys) <= 26
                    weights = {k: (1000 if k == '137' else 1) for k in keys}
                    output.append({k: v / sum(weights.values()) for k, v in weights.items()})
                return output
            def assert_no_gold(self, r):
                assert 'gold' not in r
        r = row('wide')
        r['request']['questions']['decision']['criteria'] = {str(i): f'condition-{i}' for i in range(148)}
        original = copy.deepcopy(r)
        output = tournament(Model(), r)
        self.assertEqual(output['answer'], '137')
        self.assertAlmostEqual(sum(output['scores'].values()), 1)
        self.assertNotIn('probabilities', output)
        self.assertEqual(r, original)
        r['request']['questions']['decision']['criteria']['0'] = 'all of the above'
        with self.assertRaises(UnsupportedRow):
            tournament(Model(), r)

    def test_mixed_policy_scores_up_to_threshold_directly(self):
        from jevtrain.evaluate import predict

        class Model:
            def encode(self, r):
                keys = list(r['request']['questions']['decision']['criteria'])
                if len(keys) > 52:
                    raise UnsupportedRow('too many')
                return [0], keys

            def probabilities(self, rows, batch_size):
                return [{k: 1 / len(r['request']['questions']['decision']['criteria'])
                         for k in r['request']['questions']['decision']['criteria']} for r in rows]
        rows = []
        for uid, n in (('d30', 30), ('w60', 60)):
            r = row(uid)
            r['request']['questions']['decision']['criteria'] = {str(i): f'c{i}' for i in range(n)}
            rows.append(r)
        records, unsupported = predict(Model(), rows, tournament_above=52)
        kinds = {r['id']: r.get('score_type', 'direct') for r in records}
        self.assertEqual(kinds, {'d30': 'direct', 'w60': 'mean_finalist_conditional_probability'})
        self.assertFalse(unsupported)
        records, _ = predict(Model(), rows, large_options=True)
        self.assertTrue(all(r.get('score_type') for r in records))
        records, unsupported = predict(Model(), rows)
        self.assertEqual([u['id'] for u in unsupported], ['w60'])
        with self.assertRaises(ValueError):
            predict(Model(), rows, tournament_above=10)


if __name__ == '__main__':
    unittest.main()
