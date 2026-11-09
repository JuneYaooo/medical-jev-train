"""Functional checks to run once per machine/model before real training.

These verify the implementation (padding, readout equivalence, save/load,
the ability to fit a tiny set) and measure throughput. They say nothing about
medical accuracy.
"""
import json
from pathlib import Path
import tempfile
import time

from .evaluate import predict
from .train import run_training


def verify(config, rows, output, log=print):
    from .model import ChoiceModel
    import torch
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model = ChoiceModel(config['model'], seed=config.get('seed', 17))
    rows = [r for r in rows if model.supported(r)][:32]
    report = {'trainable_parameters': model.trainable_count(), 'lora_layers': len(model.layer_indices),
              'adapted_modules': len(model.adapters)}

    # 1. Batched (left-padded) scores equal single-row scores.
    single = [model.probabilities([r], batch_size=1)[0] for r in rows[:8]]
    check_batch_size = config.get('verify', {}).get('batch_size', 8)
    batched = model.probabilities(rows[:8], batch_size=check_batch_size)
    report['batch_size'] = check_batch_size
    report['batch_vs_single_max_abs_diff'] = max(abs(a[k] - b[k]) for a, b in zip(single, batched) for k in a)

    # 2. A zero adapter reproduces the plain frozen readout.
    model.set_adapter_enabled(False)
    plain = model.probabilities(rows[:8], batch_size=1)
    model.set_adapter_enabled(True)
    report['zero_adapter_vs_readout_max_abs_diff'] = max(abs(a[k] - b[k]) for a, b in zip(single, plain) for k in a)
    (out / 'verify_progress.json').write_text(json.dumps(report, indent=2) + '\n')

    # 3. A tiny set can be fitted (catches label/position/token bugs like R228).
    tiny = rows[:16]
    before, _ = predict(model, tiny, view_count=1, batch_size=check_batch_size)
    check_optim = config.get('verify', {})
    report['model_spec'] = config['model']
    report['fit_optim'] = {'lr': check_optim.get('lr', 1e-4), 'epochs': check_optim.get('epochs', 60),
                           'grad_accum': check_optim.get('grad_accum', 1)}
    fit = {**config, 'optim': {**config.get('optim', {}), 'lr': check_optim.get('lr', 1e-4),
                               'epochs': check_optim.get('epochs', 60),
                               'batch_size': min(4, check_batch_size),
                               'grad_accum': check_optim.get('grad_accum', 1), 'warmup_ratio': 0.05,
                               'scheduler': 'constant'},
           'objective': {'label_smoothing': 0.0}, 'eval_every': 10**9,
           'save_every': 10**9, 'save_optimizer': False, 'log_every': 20}
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        report_fit, model = run_training(fit, tmp, new_rows=tiny, replay_rows=[], development=[], model=model, log=log)
        # Keep diagnostic curves after the temporary adapters are removed.
        import shutil
        shutil.copy2(Path(tmp) / 'history.jsonl', out / 'history.jsonl')
        shutil.copy2(Path(tmp) / 'training.json', out / 'training.json')
        if (Path(tmp) / 'events').exists():
            shutil.copytree(Path(tmp) / 'events', out / 'events', dirs_exist_ok=True)
        after, _ = predict(model, tiny, view_count=1, batch_size=check_batch_size)
        # 4. Save/load parity.
        reference = model.probabilities(tiny[:4], batch_size=check_batch_size)
        with torch.no_grad():
            for p in model.trainable():
                p.zero_()
        model.load(Path(tmp) / 'last.safetensors')
        again = model.probabilities(tiny[:4], batch_size=check_batch_size)
        report['save_load_max_abs_diff'] = max(abs(a[k] - b[k]) for a, b in zip(reference, again) for k in a)
    report['tiny_fit'] = {'before': sum(r['correct'] for r in before) / len(before),
                          'after': sum(r['correct'] for r in after) / len(after), 'rows': len(tiny),
                          'steps': report_fit['steps']}
    tolerance = 2e-2 if config['model'].get('dtype', 'bf16') != 'fp32' else 1e-4
    report['passed'] = {
        'batch_padding': report['batch_vs_single_max_abs_diff'] <= tolerance,
        'readout_equivalence': report['zero_adapter_vs_readout_max_abs_diff'] <= 1e-6,
        'tiny_fit': report['tiny_fit']['after'] >= 0.9,
        'save_load': report['save_load_max_abs_diff'] <= 1e-5,
    }
    report['note'] = ('batch_padding failing usually means this model class needs '
                      'pass_position_ids=false, or does not support left padding')
    (out / 'verify.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def bench(config, rows, output, steps=100, dataset_sizes=(11000, 30000, 100000, 240000), log=print):
    """Time ``steps`` optimizer updates and extrapolate to full runs."""
    import torch
    from .model import ChoiceModel
    model = ChoiceModel(config['model'], seed=config.get('seed', 17))
    rows = [r for r in rows if model.supported(r)]
    optim = {**config.get('optim', {}), 'max_steps': steps, 'epochs': max(1, config.get('optim', {}).get('epochs', 2))}
    cfg = {**config, 'optim': optim, 'eval_every': 10**9, 'save_every': 10**9}
    started = time.time()
    report, _ = run_training(cfg, output, new_rows=rows, replay_rows=[], development=[], model=model, log=log)
    seconds = time.time() - started
    per_example = seconds / report['examples']
    epochs = config.get('optim', {}).get('epochs', 2)
    result = {'steps': steps, 'examples': report['examples'], 'seconds': round(seconds, 1),
              'examples_per_second': round(1 / per_example, 2), 'tokens_per_second': report['tokens_per_second'],
              'peak_gpu_gib': report['peak_gpu_gib'],
              'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
              'estimated_hours': {str(n): round(n * epochs * per_example / 3600, 2) for n in dataset_sizes},
              'note': f'{epochs} epochs; excludes periodic development evaluation'}
    (Path(output) / 'bench.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
