"""Configuration loading independent of CLI entry points."""
import json
import re
from pathlib import Path

def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    base = Path(path).resolve().parent
    paths = {}
    if config.get('paths_file'):
        path_file = (base / config.pop('paths_file')).resolve()
        raw = json.loads(path_file.read_text(encoding='utf-8'))
        paths = {k: str((path_file.parent / v).resolve()) for k, v in raw.items()}

    def resolve(value):
        if isinstance(value, str):
            # Only a leading paths-file reference denotes a local path. Keep
            # ordinary strings such as Hugging Face organization/model IDs.
            referenced_path = re.match(r'^\$\{paths\.[^}]+\}(?:[\\/]|$)', value)
            value = re.sub(r'\$\{paths\.([^}]+)\}', lambda match: paths[match[1]], value)
            if referenced_path:
                return str(Path(value).resolve())
        if isinstance(value, str) and (value.startswith('./') or value.startswith('../')):
            return str((base / value).resolve())
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(config)
