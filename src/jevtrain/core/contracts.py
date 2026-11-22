"""Shared JSON fingerprints and validation contracts."""
import hashlib
import json


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def content_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def require(ok, reason):
    if not ok:
        raise ValueError(reason)
