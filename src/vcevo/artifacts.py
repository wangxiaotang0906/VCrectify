"""Auditable JSONL and safe array checkpoints (no pickle)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class AuditLog:
    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self.previous = "0" * 64
        if self.path.exists():
            if not resume:
                raise FileExistsError("Run directory already has events; use a fresh output or --resume")
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                checksum = record.pop("hash")
                if record["sequence"] != self.sequence or record["previous"] != self.previous or digest(record) != checksum:
                    raise ValueError("Audit log integrity verification failed")
                self.sequence += 1
                self.previous = checksum

    def append(self, event, **payload):
        record = {"sequence": self.sequence, "previous": self.previous, "event": event, **jsonable(payload)}
        record["hash"] = digest(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence += 1
        self.previous = record["hash"]
        return record["hash"]


def save_checkpoint(path, state):
    """Store a tree of JSON values, numpy arrays and optional torch tensors."""
    arrays = {}

    def encode(value):
        module = type(value).__module__
        is_tensor = module.startswith("torch") and hasattr(value, "detach")
        if isinstance(value, np.ndarray) or is_tensor:
            key = "array_" + str(len(arrays))
            arrays[key] = value.detach().cpu().numpy() if is_tensor else value
            return {"__array__": key, "torch": is_tensor}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(v) for v in value]}
        if isinstance(value, dict):
            return {"__mapping__": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return jsonable(value)

    encoded = encode(state)
    arrays["metadata"] = np.array(canonical(encoded))
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        def decode(value):
            if isinstance(value, dict) and "__array__" in value:
                result = archive[value["__array__"]].copy()
                if value["torch"]:
                    import torch
                    return torch.from_numpy(result)
                return result
            if isinstance(value, dict) and "__tuple__" in value:
                return tuple(decode(v) for v in value["__tuple__"])
            if isinstance(value, dict) and "__mapping__" in value:
                return {decode(k): decode(v) for k, v in value["__mapping__"]}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        return decode(json.loads(str(archive["metadata"])))
