import inspect
import unittest

from jevtrain.checks import bench, verify
from jevtrain.evaluate import prediction_concentration, summary
from jevtrain.train import run_training


def record(index, task, gold, prediction):
    return {'id': str(index), 'group_id': str(index), 'task': task, 'input_sha256': str(index), 'gold': gold,
            'prediction': prediction, 'correct': gold == prediction,
            'probabilities': {'A': float(prediction == 'A'), 'B': float(prediction == 'B')}}


class ReleaseChecks(unittest.TestCase):
    def test_self_checks_only_pass_supported_training_arguments(self):
        accepted = set(inspect.signature(run_training).parameters)
        for function in (verify, bench):
            self.assertNotIn('replay_rows', inspect.getsource(function))
        self.assertTrue({'new_rows', 'development', 'model', 'log'} <= accepted)

    def test_flags_answer_collapse(self):
        rows = [record(i, 'triage', 'A' if i < 5 else 'B', 'A') for i in range(10)]
        self.assertEqual(prediction_concentration(rows)['triage']['prediction_share'], 1.0)
        self.assertIn('triage', summary(rows)['collapsed_tasks'])

    def test_does_not_flag_when_gold_is_equally_concentrated(self):
        rows = [record(i, 'task', 'A', 'A') for i in range(9)] + [record(9, 'task', 'B', 'B')]
        self.assertEqual(prediction_concentration(rows), {})

    def test_ignores_tiny_tasks(self):
        rows = [record(i, 'small', 'B', 'A') for i in range(4)]
        self.assertEqual(prediction_concentration(rows), {})


if __name__ == '__main__':
    unittest.main()
