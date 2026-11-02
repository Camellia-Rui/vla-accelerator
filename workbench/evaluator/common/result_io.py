"""Atomic per-record storage with configuration identity and error retry."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RecordStore:
    """One worker owns a store; different workers require different directories."""
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = digest_json(identity)

    def path_for(self, key):
        return self.root / (digest_json({"identity": self.identity, "key": key}) + ".json")

    def load(self, key):
        path = self.path_for(key)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("status") not in ("complete", "error"):
            raise ValueError("Corrupt record status; refusing to overwrite")
        if (value["status"] == "complete") != (value.get("error") is None):
            raise ValueError("Stored status and error disagree")
        if value.get("identity_hash") != self.identity or digest_json(value.get("record_key")) != digest_json(key):
            raise ValueError("Record identity mismatch; refusing to overwrite")
        return value

    def completed(self, key):
        record = self.load(key)
        return bool(record and record.get("status") == "complete" and record.get("error") is None)

    def save(self, key, payload):
        if self.completed(key):
            raise FileExistsError("Preserve completed evidence; use a new experiment identity")
        if payload.get("status") not in ("complete", "error"):
            raise ValueError("Record status must be complete or error")
        if (payload["status"] == "complete") != (payload.get("error") is None):
            raise ValueError("Status and error disagree")
        record = dict(payload, identity_hash=self.identity, record_key=key)
        atomic_json(self.path_for(key), record)
