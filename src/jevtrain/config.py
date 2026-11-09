"""Configuration loading independent of CLI entry points."""
import json
import re
from pathlib import Path

def load_config(path):
    config = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    paths = {}
    if config.get('paths_file'):
        path_file = (base / config.pop('paths_file')).resolve()
        raw = json.loads(path_file.read_text())
        paths = {k: str((path_file.parent / v).resolve()) for k, v in raw.items()}

    def resolve(value):
        if isinstance(value, str):
            value = re.sub(r'\$\{paths\.([^}]+)\}', lambda match: paths[match[1]], value)
        if isinstance(value, str) and (value.startswith('./') or value.startswith('../')):
            return str((base / value).resolve())
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(config)
