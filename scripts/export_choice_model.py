"""Export a checksummed LoRA choice model and inference runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from jevtrain.config import load_config
from jevtrain.inference import sha256
from jevtrain.model import fast_identity, read_metadata
from jevtrain.prompt import require

ROOT = Path(__file__).resolve().parents[1]


def export_bundle(config_path, checkpoint, base_info_path, output, model_id, source_commit=None,
                  version=None):
    config = load_config(config_path)
    base_path = Path(config['model']['model_path'])
    upstream = json.loads(Path(base_info_path).read_text())
    require(len(upstream['revision']) == 40, 'pin the upstream base to a commit SHA')
    metadata = read_metadata(checkpoint)
    require(metadata['format'] == 'jev-lora-v1', 'only LoRA checkpoints can be exported')
    checkpoint_file = 'adapter.safetensors'
    require(metadata['base_model'] == fast_identity(base_path), 'checkpoint/base mismatch')
    # Verify the trained-on base against independently fetched Hub metadata.
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
             'vocab.json', 'merges.txt', 'model.safetensors.index.json']
    names += sorted(p.name for p in base_path.glob('*.safetensors'))
    base_hashes = {}
    for name in names:
        path = base_path / name
        remote = upstream['files'][name]
        require(path.stat().st_size == remote['bytes'], 'base file size mismatch: ' + name)
        digest = sha256(path)
        if remote['sha256']:
            require(digest == remote['sha256'], 'base weight SHA-256 mismatch: ' + name)
        else:
            data = path.read_bytes()
            oid = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            require(oid == remote['git_oid'], 'base configuration differs from upstream: ' + name)
        base_hashes[name] = digest
    output = Path(output)
    require(not output.exists(), 'output already exists; choose an empty release directory')
    output.mkdir(parents=True)
    for source, dest in [
        ('scripts/predict_choices.py', 'predict_choices.py'),
        ('requirements-inference.txt', 'requirements.txt'),
        ('examples/choice.jsonl', 'examples/choice.jsonl'),
        ('examples/scenarios.jsonl', 'examples/scenarios.jsonl'),
        ('examples/scenarios.outputs.jsonl', 'examples/scenarios.outputs.jsonl'),
        ('LICENSE', 'LICENSE'),
        ('docs/RELEASE_V1.md', 'EVALUATION.md'),
        ('docs/INPUT_LIMIT_CHECK.md', 'INPUT_LIMIT_CHECK.md'),
    ]:
        target = output / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    shutil.copytree(ROOT / 'src/jevtrain/core', output / 'jevtrain/core', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('inference', 'model', 'prompt', 'contracts', 'options'):
        shutil.copyfile(ROOT / f'src/jevtrain/{name}.py', output / f'jevtrain/{name}.py')
    (output / 'jevtrain/__init__.py').write_text('"""Medical JEV choice inference runtime."""\n')
    shutil.copyfile(checkpoint, output / checkpoint_file)
    shutil.copyfile(base_path / 'LICENSE', output / 'BASE_MODEL_LICENSE')
    shutil.copytree(ROOT / 'docs/results/v1', output / 'results/v1')
    shutil.copyfile(ROOT / 'docs/results/README.md', output / 'results/README.md')
    model = {k: v for k, v in config['model'].items() if k != 'model_path'}
    profile = json.loads((ROOT / 'configs/jev/release.json').read_text())
    require(2 <= profile['inference']['max_options'] <= 52, 'invalid release option count')
    require(profile['inference']['max_tokens'] > 0, 'invalid release token limit')
    model.update(profile['inference'])
    model['gradient_checkpointing'] = False
    model['dtype'] = 'fp16'
    model['training_mode'] = 'lora'
    inference = {'base_model': {'repo_id': upstream['repo_id'], 'revision': upstream['revision'],
                                'files': base_hashes}, 'model': model}
    if config.get('calibration'):
        calibration = json.loads(Path(config['calibration']).read_text())
        require(calibration['checkpoint_sha256'] == sha256(checkpoint), 'calibration checkpoint mismatch')
        from jevtrain.core.calibration import scale
        scale({'valid': 1.0}, calibration['temperature'])
        inference['calibration'] = calibration
    if config.get('tournament'):
        from jevtrain.core.tournament import strategy
        tournament = {'above': None, 'absent_option': False, **config['tournament']}
        require(set(tournament) == {'above', 'absent_option'}, 'unknown tournament settings')
        strategy(2, tournament['above'])
        inference['tournament'] = tournament
    (output / 'inference_config.json').write_text(json.dumps(inference, indent=2) + '\n')
    card = (ROOT / 'docs/MODEL_CARD_V1.md').read_text().replace('{{MODEL_ID}}', model_id)
    card = card.replace('JuneYao/medical-jev-4b', model_id)
    (output / 'README.md').write_text(card)
    manifest = {'format': 'medical-jev-release-v1', 'version': version or profile['version'],
                'checkpoint_file': checkpoint_file,
                'training_checkpoint': metadata,
                'source_commit': source_commit or subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'files': {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base-info', required=True, help='Hub model metadata with revision, file sizes, SHA-256/git OIDs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--source-commit', help='source Git commit when exporting from a deployed checkout')
    parser.add_argument('--version', help='release version; defaults to the release profile')
    args = parser.parse_args()
    manifest = export_bundle(args.config, args.checkpoint, args.base_info, args.output, args.model_id,
                             args.source_commit, args.version)
    print(json.dumps({'output': args.output, 'checkpoint_sha256': manifest['files'][manifest['checkpoint_file']],
                      'files': len(manifest['files'])}))


if __name__ == '__main__':
    main()
