import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from jevtrain.telemetry import RunMetrics


class TelemetryTests(unittest.TestCase):
    def test_live_metrics_survive_failure_and_resume_without_duplicate_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with RunMetrics(root, [], 0, 3) as live:
                    live.emit({'step': 1, 'loss': 2.0})
                    self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'running')
                    self.assertEqual(len((root / 'history.jsonl').read_text().splitlines()), 1)
                    live.emit({'step': 2, 'loss': 1.0})
                    raise RuntimeError('interrupted')
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'failed')
            with RunMetrics(root, [{'step': 1, 'loss': 2.0}], 1, 3) as live:
                live.emit({'step': 2, 'loss': 1.1, 'dev': {'accuracy': 0.5}})
                live.emit({'step': 3, 'loss': 0.8})
            records = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in records], [1, 2, 3])
            self.assertEqual(records[1]['loss'], 1.1)
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
