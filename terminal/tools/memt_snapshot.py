"""tools/memt_snapshot.py - the `memt-memory-v1` safetensors snapshot, in pure Python.

MemT-1.0 stores one published memory state per memory layer as
`memory.<layer>`, FP32, shape [1, heads, head_dim, head_dim], with metadata
`format=memt-memory-v1` and `identity=<sha256>`.  This module writes and reads
that exact format without the `safetensors` package, so a snapshot exported by
`memt_sglang` can be loaded here and the other way round.

    from memt_snapshot import snapshot_identity, save, load
    identity = snapshot_identity(config, {"chat_template": ..., "vocab": {...}})
    save(path, identity, {3: state3, 7: state7}, overwrite=False)
    states = load(path, identity, {3: (1, 16, 256, 256), 7: ...})

The identity mirrors memt_sglang/models/common/snapshot_identity.py for the
Nano family: the raw backbone config, the `memt` object, the codec string, the
chat template and the tokenizer vocabulary, hashed as sorted JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

FORMAT = "memt-memory-v1"
_MAX_HEADER = 1 << 30


class SnapshotError(ValueError):
    pass


def snapshot_identity(config: Mapping, tokenizer: Mapping) -> str:
    """Match HF Session._state_identity for MemT-1.0-Nano."""
    model = config.get("model_type")
    if model != "memt_qwen3_5_moe":
        raise SnapshotError("only the MemT-1.0-Nano config is supported, got %r" % model)
    identity = {
        "backbone": config["backbone_configs"],
        "memt": config["memt"],
        "codec": "qwen-text-raw-v1",
        "template": tokenizer["chat_template"],
        "vocabulary": tokenizer["vocab"],
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shape(value: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in value.shape)


def _validate(states: Mapping[int, np.ndarray],
              expected: Optional[Mapping[int, Tuple[int, ...]]]) -> None:
    for layer, value in states.items():
        if type(layer) is not int or layer < 0:
            raise SnapshotError("layer keys must be non-negative integers")
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise SnapshotError("memory.%d must be FP32, got %s" % (layer, array.dtype))
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] <= 0 or \
                array.shape[2] != array.shape[3] or array.shape[2] <= 0:
            raise SnapshotError("memory.%d must have shape [1, H, D, D]" % layer)
        if expected is not None and layer in expected and _shape(array) != tuple(expected[layer]):
            raise SnapshotError("memory.%d has shape %r, expected %r"
                                % (layer, _shape(array), tuple(expected[layer])))
        if not np.isfinite(array).all():
            raise SnapshotError("memory.%d contains non-finite values" % layer)


def encode(states: Mapping[int, np.ndarray], identity: str) -> bytes:
    """The whole safetensors file as bytes; states is layer -> [1,H,D,D] FP32."""
    if not isinstance(identity, str) or not identity:
        raise SnapshotError("identity must be a nonempty string")
    ordered = sorted(states.items())
    _validate(dict(ordered), None)
    header = {"__metadata__": {"format": FORMAT, "identity": identity}}
    data = bytearray()
    for layer, value in ordered:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        start = len(data)
        data += array.tobytes(order="C")
        header["memory.%d" % layer] = {
            "dtype": "F32",
            "shape": list(_shape(array)),
            "data_offsets": [start, len(data)],
        }
    payload = json.dumps(header, sort_keys=True, ensure_ascii=True).encode("utf-8")
    payload += b" " * ((8 - len(payload) % 8) % 8)
    return struct.pack("<Q", len(payload)) + payload + bytes(data)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise SnapshotError("duplicate snapshot header key %r" % key)
        obj[key] = value
    return obj


def decode(blob: bytes, identity: str,
           expected: Optional[Mapping[int, Tuple[int, ...]]] = None) -> Dict[int, np.ndarray]:
    if len(blob) < 9:
        raise SnapshotError("snapshot is shorter than a safetensors header")
    (size,) = struct.unpack("<Q", blob[:8])
    if size <= 1 or size > _MAX_HEADER or 8 + size > len(blob):
        raise SnapshotError("invalid safetensors header size")
    try:
        header = json.loads(blob[8:8 + size].decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise SnapshotError("invalid safetensors header: %s" % exc)
    if not isinstance(header, dict):
        raise SnapshotError("snapshot header must be an object")
    metadata = header.pop("__metadata__", None)
    if metadata != {"format": FORMAT, "identity": identity}:
        raise SnapshotError("snapshot format/identity mismatch")
    start = 8 + size
    tensors = {}
    for name, info in header.items():
        layer_name = name[7:]
        if not name.startswith("memory.") or not layer_name.isascii() or not layer_name.isdigit():
            raise SnapshotError("unexpected snapshot tensor %r" % name)
        layer = int(layer_name)
        if layer in tensors:
            raise SnapshotError("duplicate memory layer %d" % layer)
        if not isinstance(info, dict) or info.get("dtype") != "F32":
            raise SnapshotError("%s must be F32" % name)
        shape = info.get("shape")
        if (not isinstance(shape, list) or len(shape) != 4 or any(type(d) is not int or d <= 0 for d in shape)
                or shape[0] != 1 or shape[2] != shape[3]):
            raise SnapshotError("%s must have shape [1, H, D, D]" % name)
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
            raise SnapshotError("%s must have two integer offsets" % name)
        begin, end = offsets
        if begin < 0 or end < begin or start + end > len(blob):
            raise SnapshotError("%s has data outside the file" % name)
        expected_bytes = math.prod(shape) * 4
        if end - begin != expected_bytes:
            raise SnapshotError("%s has %d bytes, expected %d" % (name, end - begin, expected_bytes))
        tensors[layer] = (begin, end, shape)
    next_offset = 0
    for begin, end, _ in sorted(tensors.values()):
        if begin != next_offset:
            raise SnapshotError("snapshot tensor ranges overlap or leave a gap")
        next_offset = end
    if start + next_offset != len(blob):
        raise SnapshotError("unexpected data after the snapshot tensors")
    states = {}
    for layer, (begin, end, shape) in tensors.items():
        states[layer] = np.frombuffer(blob, dtype="<f4", count=(end - begin) // 4,
                                      offset=start + begin).reshape(shape).copy()
    if expected is not None:
        want = set(expected)
        got = set(states)
        if got != want:
            raise SnapshotError("snapshot layer set %r, expected %r"
                                % (sorted(got), sorted(want)))
    _validate(states, expected)
    return states


def save(path: Path, identity: str, states: Mapping[int, np.ndarray], *,
         overwrite: bool = False) -> Dict:
    """Atomic write in the target directory; never replaces an existing file by default."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise SnapshotError("snapshot names end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SnapshotError("snapshot target must not be a symlink")
    blob = encode(states, identity)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memt-state-")
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"file": path.name, "format": FORMAT, "has_memory": bool(states)}


def load(path: Path, identity: str,
         expected: Optional[Mapping[int, Tuple[int, ...]]] = None) -> Dict[int, np.ndarray]:
    with open(path, "rb") as stream:
        return decode(stream.read(), identity, expected)
