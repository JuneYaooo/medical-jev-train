"""Load a verified release bundle and return single-forward choice probabilities."""
import hashlib
import json
import math
from pathlib import Path

from .model import ChoiceModel, UnsupportedRow
from .calibration import scale
from .prompt import require


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'manifest path escapes bundle')
    # Hub snapshots intentionally symlink files into their shared blob cache.
    return Path(root) / relative


def verify_bundle(path):
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest.get('format') == 'medical-jev-release-v1', 'unsupported release format')
    checkpoint = manifest.get('checkpoint_file', 'adapter.safetensors')
    checked_path(root, checkpoint)
    require(checkpoint in manifest['files'] and 'inference_config.json' in manifest['files'],
            'incomplete release manifest')
    for name, expected in manifest['files'].items():
        require(sha256(checked_path(root, name)) == expected, 'release SHA-256 mismatch: ' + name)
    return manifest


class ChoicePredictor:
    def __init__(self, bundle, base_model_path=None, device='cuda'):
        root = Path(bundle)
        self.manifest = verify_bundle(root)
        config = json.loads((root / 'inference_config.json').read_text())
        self.temperature = config.get('calibration', {}).get('temperature', 1.0)
        scale({'valid': 1.0}, self.temperature)
        # Bundle default for wide inputs; older bundles have none and keep direct scoring.
        self.tournament = {'above': None, 'absent_option': False, **config.get('tournament', {})}
        from .tournament import strategy
        strategy(2, self.tournament['above'])
        base = config['base_model']
        if base_model_path is None:
            from huggingface_hub import snapshot_download
            base_model_path = snapshot_download(base['repo_id'], revision=base['revision'],
                                                allow_patterns=list(base['files']))
        for name, expected in base['files'].items():
            require(sha256(checked_path(base_model_path, name)) == expected, 'base SHA-256 mismatch: ' + name)
        spec = {**config['model'], 'model_path': str(base_model_path), 'device': device,
                'gradient_checkpointing': False}
        if device == 'cpu':
            spec['dtype'] = 'fp32'
        self.model = ChoiceModel(spec, checkpoint=checked_path(
            root, self.manifest.get('checkpoint_file', 'adapter.safetensors')))

    def predict(self, requests, batch_size=8, large_options=False, tournament_above=None, absent_option=None):
        """``large_options`` means tournament_above=26; unset arguments use the bundle's tournament config."""
        require(type(batch_size) is int and batch_size > 0, 'batch_size must be positive')
        from .tournament import strategy, tournament
        defaults = getattr(self, 'tournament', {'above': None, 'absent_option': False})
        if tournament_above is None:
            tournament_above = 26 if large_options else defaults['above']
        if absent_option is None:
            absent_option = defaults['absent_option']
        require(isinstance(requests, list), 'requests must be a list')
        rows, supported, output = [], [], [None] * len(requests)
        for index, request in enumerate(requests):
            require(isinstance(request, dict), 'request must be an object')
            require(isinstance(request.get('questions'), dict)
                    and set(request['questions']) == {'decision'}, 'exactly one decision question required')
            question = request['questions']['decision']
            require(isinstance(question, dict), 'decision must be an object')
            opts = question.get('criteria')
            require(isinstance(opts, dict) and len(opts) >= 2, 'at least two options required')
            require(all(isinstance(k, str) and k and isinstance(v, str) and v.strip()
                        for k, v in opts.items()), 'options must map nonempty keys to nonempty text')
            require(isinstance(request.get('state'), dict) and request['state'], 'state must be a nonempty object')
            require(isinstance(question.get('instructions'), str) and question['instructions'].strip(),
                    'instructions required')
            # Content-derived IDs avoid stale token-cache hits across calls with reused user IDs.
            uid = hashlib.sha256(json.dumps(request, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            row = {'id': uid, 'request': request}
            if strategy(len(opts), tournament_above) == 'tournament':
                try:
                    output[index] = tournament(self.model, row, batch_size=batch_size,
                                               absent_option=absent_option)
                except UnsupportedRow as error:
                    output[index] = {'status': 'unsupported', 'reason': str(error)}
                continue
            try:
                self.model.encode(row)
            except UnsupportedRow as error:
                output[index] = {'status': 'unsupported', 'reason': str(error)}
                continue
            rows.append(row)
            supported.append(index)
        if rows:
            for index, probabilities in zip(supported, self.model.probabilities(rows, batch_size)):
                require(all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
                        and abs(sum(probabilities.values()) - 1) < 1e-4, 'invalid model probabilities')
                probabilities = scale(probabilities, getattr(self, 'temperature', 1.0))
                output[index] = {'status': 'ok', 'answer': max(probabilities, key=probabilities.get),
                                 'probabilities': probabilities}
        return output
