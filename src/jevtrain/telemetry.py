"""Live, resumable training metrics for local files and TensorBoard."""
import json
from pathlib import Path
import time


class RunMetrics:
    def __init__(self, output, history, start_step, total_steps, tensorboard=False):
        self.root = Path(output)
        self.total = total_steps
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.root / 'events'), purge_step=start_step + 1,
                                        flush_secs=10)
        # Discard observations newer than the durable optimizer checkpoint.
        path = self.root / 'history.jsonl'
        temporary = path.with_suffix('.jsonl.partial')
        temporary.write_text(''.join(json.dumps(h) + '\n' for h in history))
        temporary.replace(path)
        self.path = path
        self.latest = {'step': start_step, 'total': total_steps}
        self.status('running')

    def status(self, state, error=None):
        record = {**self.latest, 'state': state, 'updated_at': time.time()}
        if error:
            record['error'] = error
        temporary = self.root / 'status.json.partial'
        temporary.write_text(json.dumps(record) + '\n')
        temporary.replace(self.root / 'status.json')

    def emit(self, entry):
        record = {**entry, 'wall_time': time.time(), 'total': self.total}
        with self.path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.latest = record
        if self.writer:
            for key in ('loss', 'grad_norm', 'lr', 'loss_scale', 'optimizer_step_skipped',
                        'examples_per_s', 'tokens_per_s', 'peak_gpu_gib'):
                if entry.get(key) is not None:
                    self.writer.add_scalar('train/' + key, entry[key], entry['step'])
            for key, value in (entry.get('dev') or {}).items():
                if key != 'step' and isinstance(value, (int, float)):
                    self.writer.add_scalar('dev/' + key, value, entry['step'])
            self.writer.flush()
        self.status('running')

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.status('failed' if error else 'completed', str(error) if error else None)
        if self.writer:
            self.writer.close()
