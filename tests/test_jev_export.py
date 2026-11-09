"""Verify LoRA exports are self-contained and retain their checkpoint identity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_choice_model import export_bundle
from jevtrain.inference import sha256, verify_bundle


class ExportTests(unittest.TestCase):
    def test_lora_export_has_current_runtime_and_model_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'
            base.mkdir()
            names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'vocab.json', 'merges.txt', 'model.safetensors.index.json', 'base.safetensors', 'LICENSE']
            for name in names:
                (base / name).write_text(name)
            upstream = {'repo_id': 'fixture/base', 'revision': 'a' * 40, 'files': {
                name: {'bytes': (base / name).stat().st_size, 'sha256': sha256(base / name)}
                for name in names}}
            (root / 'upstream.json').write_text(json.dumps(upstream))
            config = {'model': {'model_path': str(base), 'training_mode': 'lora', 'dtype': 'fp16'}}
            (root / 'config.json').write_text(json.dumps(config))
            checkpoint = root / 'trained.safetensors'
            checkpoint.write_bytes(b'trained-adapter')
            metadata = {'format': 'jev-lora-v1', 'base_model': 'fixture', 'step': 5500}
            with patch('export_choice_model.fast_identity', return_value='fixture'), \
                 patch('export_choice_model.read_metadata', return_value=metadata):
                manifest = export_bundle(root / 'config.json', checkpoint, root / 'upstream.json',
                                         root / 'bundle', 'fixture/lora', 'b' * 40, version='1.1.0')
            bundle = root / 'bundle'
            self.assertEqual(verify_bundle(bundle), manifest)
            self.assertEqual(manifest['checkpoint_file'], 'adapter.safetensors')
            self.assertEqual((bundle / 'adapter.safetensors').read_bytes(), checkpoint.read_bytes())
            inference = json.loads((bundle / 'inference_config.json').read_text())
            self.assertEqual(inference['model']['dtype'], 'fp16')
            self.assertEqual(inference['model']['training_mode'], 'lora')
            self.assertIn('67.90%', (bundle / 'README.md').read_text())
            self.assertEqual((bundle / 'BASE_MODEL_LICENSE').read_text(), (base / 'LICENSE').read_text())
            self.assertTrue((bundle / 'INPUT_LIMIT_CHECK.md').exists())
            self.assertTrue((bundle / 'results/v1/summary.json').exists())
            subprocess.run([sys.executable, '-c', 'from jevtrain.inference import ChoicePredictor'],
                           cwd=bundle, check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
