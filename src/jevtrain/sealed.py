"""Append-only sealed evaluation attempts, reserved before any model execution."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

from .inference import sha256
from .metrics import write_json


@contextmanager
def sealed_attempt(ledger, input_path, checkpoint):
    if sys.platform == 'win32':
        raise RuntimeError('sealed evaluation is only supported on POSIX; ordinary evaluation works on Windows')
    import fcntl

    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix('.lock').open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {'attempts': []}
        binding = {'input_sha256': sha256(input_path), 'checkpoint_sha256': sha256(checkpoint)}
        if any(all(a[k] == v for k, v in binding.items()) for a in state['attempts']):
            raise ValueError('sealed cohort/checkpoint already attempted; inspect ledger before any rerun')
        attempt = {**binding, 'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
        state['attempts'].append(attempt)
        write_json(ledger, state)
        try:
            yield {'evaluated': True, 'run_count': len(state['attempts']), **binding}
        except BaseException:
            attempt['status'] = 'failed'
            write_json(ledger, state)
            raise
        else:
            attempt['status'] = 'completed'
            write_json(ledger, state)
