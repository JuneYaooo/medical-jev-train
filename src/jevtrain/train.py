"""Config-driven LoRA training on answer-letter logits.

Every example is shown in a random cyclic option order. Targets are the hard
reference with optional label smoothing. Every ``eval_every`` steps the model
is scored on the development set and the best checkpoint is kept.
"""
import json
import hashlib
import math
from pathlib import Path
import random
import time

from .contracts import require

from .data import random_offset, read_rows, target_distribution, view, write_json
from .evaluate import predict, summary
from .model import ChoiceModel, UnsupportedRow
from .telemetry import RunMetrics
from .selection import selection_key

DEFAULT_OPTIM = {'lr': 1e-4, 'weight_decay': 0.0, 'warmup_ratio': 0.03, 'epochs': 2, 'max_steps': None,
                 'batch_size': 8, 'grad_accum': 2, 'clip': 1.0, 'scheduler': 'cosine', 'min_lr_ratio': 0.1}


def build_optimizer(model, optim):
    import torch
    name = optim.get('name', 'adamw')
    require(name in ('adamw', 'adamw8bit'), 'unknown optimizer: ' + name)
    cls = torch.optim.AdamW
    if name == 'adamw8bit':
        require(model.device == 'cuda', 'adamw8bit requires CUDA')
        from bitsandbytes.optim import AdamW8bit
        cls = AdamW8bit
    return cls(model.trainable(), lr=optim['lr'], weight_decay=optim['weight_decay'], betas=(0.9, 0.999))


def load_training_rows(entries):
    """entries: [{"path", "tasks", "exclude_tasks", "per_task_max", "max_rows", "seed"}] -> rows.

    ``per_task_max`` caps each task (hash order) before ``max_rows`` caps the
    total, so one very large task cannot crowd out the others.
    """
    from collections import defaultdict
    from .data import hash_order, sample_rows
    rows, seen = [], set()
    for entry in entries:
        current = [r for r in read_rows(entry['path']) if r.get('gold') is not None]
        if entry.get('tasks'):
            current = [r for r in current if r['task'] in set(entry['tasks'])]
        if entry.get('exclude_tasks'):
            current = [r for r in current if r['task'] not in set(entry['exclude_tasks'])]
        seed = entry.get('seed', 17)
        if entry.get('per_task_max'):
            by_task = defaultdict(list)
            for r in hash_order(current, f'{seed}:per_task'):
                if len(by_task[r['task']]) < entry['per_task_max']:
                    by_task[r['task']].append(r)
            current = [r for task in sorted(by_task) for r in by_task[task]]
        current = sample_rows(current, entry.get('max_rows'), seed)
        for r in current:
            require(r['id'] not in seen, 'duplicate training id across inputs: ' + r['id'])
            seen.add(r['id'])
        rows.extend(current)
    return rows


def build_schedule(rows, optim, seed):
    """List of (source, index, draw) examples; deterministic from the seed."""
    rng = random.Random(seed)
    examples = []
    for epoch in range(int(optim['epochs'])):
        order = list(range(len(rows)))
        rng.shuffle(order)
        examples.extend(('new', i, None) for i in order)
    per_step = optim['batch_size'] * optim['grad_accum']
    if optim.get('max_steps'):
        examples = examples[:optim['max_steps'] * per_step]
    return [(source, index, rng.random()) for source, index, _ in examples]


def bucket(micro_batches_examples, lengths, batch_size, window=50):
    """Group similar lengths inside shuffled windows to cut padding, then reshuffle batches."""
    batches = []
    chunk = batch_size * window
    for start in range(0, len(micro_batches_examples), chunk):
        part = sorted(micro_batches_examples[start:start + chunk], key=lambda e: lengths(e))
        batches.extend(part[i:i + batch_size] for i in range(0, len(part), batch_size))
    return batches


def lr_at(step, total, optim):
    warmup = max(1, int(total * optim['warmup_ratio']))
    if step < warmup:
        return (step + 1) / warmup
    if optim['scheduler'] == 'constant':
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    floor = optim.get('min_lr_ratio', 0.1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def run_training(config, output, new_rows=None, development=None, parent=None, model=None, log=print):
    """Train one model. Returns the report dict; writes best/last checkpoints into ``output``."""
    import torch
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    optim = {**DEFAULT_OPTIM, **config.get('optim', {})}
    objective = config.get('objective', {'label_smoothing': 0.05})
    seed = config.get('seed', 17)
    data = config.get('data', {})
    if new_rows is None:
        new_rows = load_training_rows(data['train'])
    if development is None and data.get('development'):
        development = read_rows(data['development'])
    torch.manual_seed(seed)
    if model is None:
        model = ChoiceModel(config['model'], checkpoint=parent or config.get('parent'), seed=seed)
    elif parent:
        model.load(parent)

    def supported(rows, name):
        kept = [r for r in rows if model.supported(r)]
        if len(kept) < len(rows):
            log(json.dumps({'unsupported_' + name: len(rows) - len(kept)}))
        return kept

    new_rows = supported(new_rows, 'train')
    require(new_rows, 'no supported training rows')
    schedule = build_schedule(new_rows, optim, seed)

    def row_of(example):
        source, index, draw = example
        row = new_rows[index]
        shown = view(row, random_offset(row, random.Random(f'{seed}:{source}:{index}:{draw}')))
        # A reordered view can tokenize longer than the original; fall back to it.
        return shown if shown is row or model.supported(shown) else row

    def length(example):
        try:
            return len(model.encode(row_of(example))[0])
        except UnsupportedRow:
            return 0

    batches = bucket(schedule, length, optim['batch_size'])
    random.Random(seed + 1).shuffle(batches)
    total_steps = math.ceil(len(batches) / optim['grad_accum'])
    params = model.trainable()
    opt = build_optimizer(model, optim)
    use_scaler = model.device == 'cuda' and model.dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', init_scale=float(optim.get('loss_scale', 128))) if use_scaler else None
    progress = out / 'progress.pt'
    start_step, history, best = 0, [], None
    def fingerprint(rows):
        digest = hashlib.sha256()
        for row in rows:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
            digest.update(b'\n')
        return digest.hexdigest()


    parent_path = parent or config.get('parent')
    parent_hash = None
    if parent_path:
        with open(parent_path, 'rb') as stream:
            parent_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    binding = {'config': config, 'new': len(new_rows), 'batches': len(batches),
               'new_sha256': fingerprint(new_rows), 'development_sha256': fingerprint(development or []),
               'selection_policy': 'accuracy-macro-nll', 'parent_sha256': parent_hash,
               'parent': str(parent or config.get('parent') or 'readout')}
    if progress.exists():
        state = torch.load(progress, map_location='cpu', weights_only=False)
        require(state['binding'] == binding, 'training inputs changed; use a new output directory')
        model.load_state(state['parameters'])
        opt.load_state_dict(state['optimizer'])
        if scaler and state.get('scaler'):
            scaler.load_state_dict(state['scaler'])
        start_step, history, best = state['step'], state['history'], state['best']
        if 'torch_rng' in state:
            torch.set_rng_state(state['torch_rng'])
        if model.device == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        log(json.dumps({'resumed_at_step': start_step}))
        del state
    with RunMetrics(out, history, start_step, total_steps, config.get('tensorboard', False)) as live:
        eval_every = config.get('eval_every', 500)
        save_every = config.get('save_every', 200)
        eval_views = config.get('eval_views', 1)
        started, tokens_seen, examples_seen = time.time(), 0, 0
        model.model.train()

        def evaluate_dev(step):
            nonlocal best
            if not development:
                return None
            records, unsupported = predict(model, development, view_count=eval_views,
                                           batch_size=config.get('eval_batch_size', optim['batch_size'] * 2))
            model.model.train()
            if model.device == 'cuda':
                # Return evaluation buffers to the driver; Triton loads kernels outside torch's cache.
                torch.cuda.empty_cache()
            result = summary(records, unsupported)
            key = selection_key(result)
            entry = {'step': step, 'accuracy': result['accuracy'], 'macro_accuracy': result['macro_accuracy'],
                     'nll': result['nll'], 'correct': result['correct'], 'n': result['n'],
                     'by_task': result['by_task'], 'ece': result['ece'],
                     'unsupported': unsupported}
            if best is None or key > tuple(best['key']):
                best = {**entry, 'key': list(key)}
                model.save(out / 'best.safetensors', {'step': step})
            log(json.dumps({'dev': entry}))
            return entry

        if start_step == 0 and development and config.get('eval_at_start', True):
            history.append({'step': 0, 'dev': evaluate_dev(0)})
            live.emit(history[-1])
        consecutive_skips = 0
        for step in range(start_step, total_steps):
            scale = lr_at(step, total_steps, optim)
            for group in opt.param_groups:
                group['lr'] = optim['lr'] * scale
            opt.zero_grad(set_to_none=True)
            losses = []
            micro = batches[step * optim['grad_accum']:(step + 1) * optim['grad_accum']]
            micro_count = sum(len(b) for b in micro)
            for batch in micro:
                rows = [row_of(e) for e in batch]
                encoded = [model.encode(r) for r in rows]
                targets = [target_distribution(r, keys, objective) for r, (_, keys) in zip(rows, encoded)]
                with model.autocast():
                    logits = model.letter_logits([t for t, _ in encoded], [len(k) for _, k in encoded])
                width = logits.shape[1]
                target = torch.zeros((len(rows), width), dtype=torch.float32, device=logits.device)
                for i, t in enumerate(targets):
                    target[i, :len(t)] = torch.tensor(t)
                loss = -(target * logits.log_softmax(-1)).sum(-1)
                require(bool(torch.isfinite(loss).all()), 'nonfinite training loss')
                loss_sum = loss.sum() / micro_count
                (scaler.scale(loss_sum) if scaler else loss_sum).backward()
                losses.extend(loss.detach().float().cpu().tolist())
                tokens_seen += sum(len(t) for t, _ in encoded)
                examples_seen += len(rows)
            if scaler:
                scaler.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(params, optim['clip'])
            skipped = not bool(torch.isfinite(norm))
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                require(bool(torch.isfinite(norm)), 'nonfinite gradient')
                opt.step()
            consecutive_skips = consecutive_skips + 1 if skipped else 0
            require(consecutive_skips < 16, '16 consecutive FP16 overflows; inspect loss scaling before continuing')
            entry = {'step': step + 1, 'loss': sum(losses) / len(losses),
                     'grad_norm': None if skipped else float(norm),
                     'optimizer_step_skipped': skipped, 'loss_scale': scaler.get_scale() if scaler else 1.0,
                     'lr': optim['lr'] * scale}
            history.append(entry)
            if (step + 1) % config.get('log_every', 20) == 0:
                elapsed = time.time() - started
                log(json.dumps({**entry, 'total': total_steps, 'examples_per_s': round(examples_seen / elapsed, 2),
                                'tokens_per_s': round(tokens_seen / elapsed, 1)}))
            last = step + 1 == total_steps
            if development and ((step + 1) % eval_every == 0 or last):
                entry['dev'] = evaluate_dev(step + 1)
            elapsed = time.time() - started
            entry.update(examples_per_s=round(examples_seen / max(elapsed, 1e-9), 3),
                         tokens_per_s=round(tokens_seen / max(elapsed, 1e-9), 1),
                         peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if model.device == 'cuda' else 0)
            live.emit(entry)
            if config.get('save_optimizer', True) and ((step + 1) % save_every == 0 or last):
                temporary = progress.with_name('progress.pt.partial')
                torch.save({'binding': binding, 'step': step + 1, 'history': history, 'best': best,
                            'parameters': model.state(), 'optimizer': opt.state_dict(),
                            'torch_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state_all() if model.device == 'cuda' else None,
                            'scaler': scaler.state_dict() if scaler else None}, temporary)
                temporary.replace(progress)
        model.save(out / 'last.safetensors', {'step': total_steps})
        elapsed = time.time() - started
        report = {'steps': total_steps, 'examples': len(schedule), 'new_rows': len(new_rows),
                  'skipped_optimizer_steps': sum(bool(h.get('optimizer_step_skipped')) for h in history),
                  'trainable_parameters': model.trainable_count(), 'lora': model.lora_config,
                  'training_mode': model.training_mode,
                  'layer_indices': model.layer_indices, 'objective': objective, 'optim': optim,
                  'best': best,
                  'selected': str(out / ('best.safetensors' if best else 'last.safetensors')),
                  'elapsed_seconds': round(elapsed, 1),
                  'examples_per_second': round(examples_seen / elapsed, 3) if elapsed else None,
                  'tokens_per_second': round(tokens_seen / elapsed, 1) if elapsed else None,
                  'peak_gpu_gib': (torch.cuda.max_memory_allocated() / 2**30) if model.device == 'cuda' else None}
        write_json(out / 'training.json', report)
        progress.unlink(missing_ok=True)
        return report, model
