"""CPU regressions for ordinary evaluation and POSIX-only sealed attempts."""
import builtins
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from jevtrain.cli import main
from jevtrain.config import load_config
from jevtrain.metrics import read_rows, write_json, write_rows


def sample():
    return {'id': 'multilingual', 'group_id': 'case-1', 'task': 'teaching', 'gold': 'a',
            'request': {'state': {'clinical_question': '中文 한국어 日本語 tiếng Việt 🙂'},
                        'questions': {'decision': {'type': 'choice', 'instructions': 'choose',
                                                   'criteria': {'a': '正しい', 'b': '아니요'}}}}}


class FakeModel:
    def __init__(self, spec, checkpoint=None):
        self.spec = spec

    def encode(self, row):
        return [0], list(row['request']['questions']['decision']['criteria'])

    def probabilities(self, rows, batch_size):
        return [{'a': 0.75, 'b': 0.25} for _ in rows]


def child_environment():
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    env['PYTHONPATH'] = source + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


class WindowsEvaluationTests(unittest.TestCase):
    def fixtures(self, root):
        config, inputs = root / 'config.json', root / 'input.jsonl'
        config.write_text(json.dumps({'model': {'model_path': 'unused'}, 'note': '한국어 🙂'},
                                     ensure_ascii=False), encoding='utf-8')
        inputs.write_text(json.dumps(sample(), ensure_ascii=False) + '\n', encoding='utf-8')
        return config, inputs

    def test_sealed_module_import_does_not_import_fcntl(self):
        code = textwrap.dedent('''
            import builtins
            original = builtins.__import__
            def guard(name, *args, **kwargs):
                if name == 'fcntl':
                    raise ImportError('fcntl deliberately unavailable')
                return original(name, *args, **kwargs)
            builtins.__import__ = guard
            import jevtrain.sealed
        ''')
        result = subprocess.run([sys.executable, '-c', code], env=child_environment(),
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ordinary_eval_runs_without_importing_sealed_and_writes_utf8(self):
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == 'jevtrain.sealed' or name == 'fcntl':
                raise AssertionError('ordinary evaluation must not load sealed locking')
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, inputs = self.fixtures(root)
            output = root / 'output'
            with patch('jevtrain.model.ChoiceModel', FakeModel), patch('builtins.__import__', guard), \
                    redirect_stdout(io.StringIO()):
                main(['eval', '--config', str(config), '--input', str(inputs), '--output', str(output)])
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['n'], summary['correct'], summary['accuracy']), (1, 1, 1.0))
            self.assertEqual(read_rows(output / 'predictions.jsonl')[0]['id'], 'multilingual')
            self.assertFalse(summary['sealed_test']['evaluated'])

    def test_windows_sealed_cli_rejects_before_model_or_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, inputs = self.fixtures(root)
            ledger = root / 'new-directory' / 'ledger.json'
            with patch('sys.platform', 'win32'), patch('jevtrain.model.ChoiceModel') as model, \
                    redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as caught:
                    main(['eval', '--config', str(config), '--input', str(inputs),
                          '--output', str(root / 'output'), '--checkpoint', str(root / 'weights'),
                          '--sealed-ledger', str(ledger)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn('POSIX', errors.getvalue())
            model.assert_not_called()
            self.assertFalse(ledger.parent.exists())

    def test_windows_sealed_context_rejects_without_creating_directory(self):
        from jevtrain.sealed import sealed_attempt
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'new-directory' / 'ledger.json'
            with patch('sys.platform', 'win32'):
                with self.assertRaisesRegex(RuntimeError, 'POSIX'):
                    with sealed_attempt(ledger, 'nonexistent-input', 'nonexistent-weights'):
                        self.fail('unsupported context must never yield')
            self.assertFalse(ledger.parent.exists())

    def test_json_io_explicitly_uses_utf8(self):
        original_write = Path.write_text

        def check_encoding(path, data, *args, **kwargs):
            self.assertEqual(kwargs.get('encoding'), 'utf-8')
            return original_write(path, data, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(Path, 'write_text', check_encoding):
                write_json(root / 'report.json', sample())
                write_rows(root / 'rows.jsonl', [sample()])
            self.assertEqual(json.loads((root / 'report.json').read_bytes().decode('utf-8')), sample())
            self.assertEqual(read_rows(root / 'rows.jsonl'), [sample()])

    def test_config_reads_utf8_and_normalizes_only_referenced_local_paths(self):
        original_read = Path.read_text

        def check_encoding(path, *args, **kwargs):
            self.assertEqual(kwargs.get('encoding'), 'utf-8')
            return original_read(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'paths_file': 'paths.json',
                                          'local_model': '${paths.models}/base',
                                          'hub_model': 'organization/model', 'note': '한국어 🙂'},
                                         ensure_ascii=False), encoding='utf-8')
            (root / 'paths.json').write_text(json.dumps({'models': './models'}), encoding='utf-8')
            with patch.object(Path, 'read_text', check_encoding):
                result = load_config(config)
            self.assertEqual(result['local_model'], str(root / 'models' / 'base'))
            self.assertEqual(result['hub_model'], 'organization/model')
            self.assertEqual(result['note'], '한국어 🙂')


@unittest.skipIf(sys.platform == 'win32', 'sealed evaluation uses POSIX file locks')
class PosixSealedEvaluationTests(unittest.TestCase):
    def test_completed_attempt_is_recorded_and_repeat_is_blocked(self):
        from jevtrain.sealed import sealed_attempt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input').write_bytes(b'cohort')
            (root / 'weights').write_bytes(b'weights')
            ledger = root / 'ledger.json'
            with sealed_attempt(ledger, root / 'input', root / 'weights') as status:
                self.assertTrue(status['evaluated'])
            record = json.loads(ledger.read_text(encoding='utf-8'))
            self.assertEqual(record['attempts'][0]['status'], 'completed')
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                with sealed_attempt(ledger, root / 'input', root / 'weights'):
                    self.fail('completed attempt must not run twice')
            self.assertEqual(len(json.loads(ledger.read_text(encoding='utf-8'))['attempts']), 1)

    def test_concurrent_process_cannot_reserve_locked_ledger(self):
        from jevtrain.sealed import sealed_attempt
        code = textwrap.dedent('''
            import sys
            from jevtrain.sealed import sealed_attempt
            try:
                with sealed_attempt(*sys.argv[1:]):
                    raise SystemExit('second process acquired held lock')
            except BlockingIOError:
                pass
        ''')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input').write_bytes(b'cohort')
            (root / 'weights').write_bytes(b'weights')
            ledger = root / 'ledger.json'
            with sealed_attempt(ledger, root / 'input', root / 'weights'):
                result = subprocess.run([sys.executable, '-c', code, str(ledger), str(root / 'input'),
                                         str(root / 'weights')], env=child_environment(),
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                state = json.loads(ledger.read_text(encoding='utf-8'))
                self.assertEqual(len(state['attempts']), 1)
                self.assertEqual(state['attempts'][0]['status'], 'running')


if __name__ == '__main__':
    unittest.main()
