"""Release regressions: prompt equivalence, no gold dependence, integrity, cache isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from jevtrain.inference import ChoicePredictor, sha256, verify_bundle
from jevtrain.model import UnsupportedRow
from jevtrain.prompt import prompt_for


def request(text='example'):
    return {'state': {'clinical_question': text}, 'questions': {'decision': {
        'instructions': 'choose', 'criteria': {'yes': 'yes', 'no': 'no'}}}}


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return json.dumps([args, kwargs], ensure_ascii=False)


class FakeModel:
    def __init__(self):
        self.ids = []

    def encode(self, row):
        self.ids.append(row['id'])
        if len(row['request']['questions']['decision']['criteria']) > 26:
            raise UnsupportedRow('too many options')

    def probabilities(self, rows, batch_size):
        return [{'yes': 0.75, 'no': 0.25} for _ in rows]


class InferenceTests(unittest.TestCase):
    def test_prompt_preserves_training_serialization(self):
        fixtures = json.loads((Path(__file__).parent / 'fixtures/choice_prompt.json').read_text())
        for item in fixtures:
            self.assertEqual(prompt_for(item['request'], Tokenizer()), (item['prompt'], item['keys']))

    def test_repeated_requests_do_not_reuse_stale_tokens(self):
        predictor = ChoicePredictor.__new__(ChoicePredictor)
        predictor.model = FakeModel()
        first = predictor.predict([request('first')])[0]
        predictor.predict([request('changed')])
        self.assertNotEqual(*predictor.model.ids)
        self.assertEqual(first['answer'], 'yes')
        self.assertEqual(set(first), {'status', 'answer', 'probabilities'})

    def test_unsupported_is_explicit_and_retains_order(self):
        predictor = ChoicePredictor.__new__(ChoicePredictor)
        predictor.model = FakeModel()
        wide = request()
        wide['questions']['decision']['criteria'] = {str(i): str(i) for i in range(27)}
        results = predictor.predict([request(), wide, request()])
        self.assertEqual([r['status'] for r in results], ['ok', 'unsupported', 'ok'])
        self.assertNotIn('answer', results[1])

    def test_malformed_requests_do_not_reach_model(self):
        predictor = ChoicePredictor.__new__(ChoicePredictor)
        predictor.model = FakeModel()
        for invalid in [{}, {'questions': {'decision': 'bad'}}, {**request(), 'state': {}}]:
            with self.assertRaises(ValueError):
                predictor.predict([invalid])
        with self.assertRaises(ValueError):
            predictor.predict([request()], batch_size=0)

    def test_integrity_detects_changed_weights_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('adapter.safetensors', 'inference_config.json'):
                (root / name).write_text('fixture')
            manifest = {'format': 'medical-jev-release-v1', 'files': {
                p.name: sha256(p) for p in root.iterdir()}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            verify_bundle(root)
            (root / 'adapter.safetensors').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify_bundle(root)
            manifest['files'] = {'../outside': 'x', **manifest['files']}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'escapes'):
                verify_bundle(root)

    def test_hub_snapshot_symlinks_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'blobs').mkdir()
            (root / 'snapshot').mkdir()
            blob = root / 'blobs/weight'
            blob.write_text('fixture')
            for name in ('adapter.safetensors', 'inference_config.json'):
                (root / 'snapshot' / name).symlink_to('../blobs/weight')
            manifest = {'format': 'medical-jev-release-v1', 'files': {
                name: sha256(blob) for name in ('adapter.safetensors', 'inference_config.json')}}
            (root / 'snapshot/manifest.json').write_text(json.dumps(manifest))
            verify_bundle(root / 'snapshot')

    def test_bundle_loads_its_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.safetensors').write_bytes(b'checkpoint')
            config = {'base_model': {'files': {}}, 'model': {'training_mode': 'lora', 'dtype': 'fp16'}}
            (root / 'inference_config.json').write_text(json.dumps(config))
            manifest = {'format': 'medical-jev-release-v1', 'checkpoint_file': 'model.safetensors',
                        'files': {p.name: sha256(p) for p in root.iterdir()}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch('jevtrain.inference.ChoiceModel') as model:
                ChoicePredictor(root, base_model_path=root, device='cuda')
                self.assertEqual(model.call_args.kwargs['checkpoint'], root / 'model.safetensors')
                self.assertEqual(model.call_args.args[0]['training_mode'], 'lora')
            manifest['checkpoint_file'] = '../unverified.safetensors'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'escapes'):
                verify_bundle(root)
            manifest['checkpoint_file'] = 'unverified.safetensors'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
