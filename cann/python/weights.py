"""Read Strata's original GGUF weights, locally or from exact HTTP ranges."""
from __future__ import annotations

from collections import OrderedDict
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

SHARDS = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf": (
        47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"
    ),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf": (
        28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"
    ),
}
DEFAULT_SOURCE = (
    "https://www.modelscope.cn/models/ISTA-DASLab/"
    "Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/master/IQ3_XXS"
)


class WeightStore:
    """The headers are small; tensor bytes are fetched only when a layer uses them.

    A complete local GGUF always takes precedence. HTTP responses must name the
    requested range and the pinned file size. Download caches contain raw source
    bytes, never an altered model or zero-filled missing experts.
    """

    def __init__(self, model_dir, cache_dir, decoder, source=DEFAULT_SOURCE, workers=6,
                 disk_cache_gib=16):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source = source.rstrip("/") if source else None
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gguf")
        self._lock = threading.Lock()
        self.disk_budget = int(disk_cache_gib * 1024**3)
        self.disk_files = OrderedDict(
            (path.name, path.stat().st_size)
            for path in sorted(self.cache_dir.iterdir(), key=lambda path: path.stat().st_mtime)
            if path.is_file() and len(path.name) == 64
        )
        self.disk_used = sum(self.disk_files.values())
        self.stats = {"network_bytes": 0, "network_seconds": 0.0, "disk_hits": 0, "decodes": 0}
        self.lib = ctypes.CDLL(str(decoder))
        self.decode_fn = self.lib.strata_cann_decode
        self.decode_fn.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64
        ]
        self.decode_fn.restype = ctypes.c_int
        self.headers = []
        self.tensors = {}
        for filename, (size, _) in SHARDS.items():
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"GGUF header or complete shard required: {path}")
            header = GGUFFile(path)
            self.headers.append(header)
            if path.stat().st_size not in (header.data_start, size):
                raise ValueError(f"unexpected size for {filename}: {path.stat().st_size}")
            for tensor in header.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(f"duplicate tensor: {tensor.name}")
                self.tensors[tensor.name] = (header, tensor)
        self.metadata = self.headers[0].metadata
        if self.metadata["general.architecture"] != "qwen4exp":
            raise ValueError("this CANN backend implements the qwen4exp architecture")
        if len(self.tensors) != self.metadata["split.tensors.count"]:
            raise ValueError("the GGUF shards do not form the declared complete model")

    def close(self):
        self.pool.shutdown(wait=True)

    def raw(self, name, element_offset=0, count=None):
        header, tensor = self.tensors[name]
        count = tensor.elements - element_offset if count is None else count
        block, block_bytes = BLOCK_GEOMETRY[tensor.type_name]
        if count <= 0 or element_offset < 0 or element_offset + count > tensor.elements:
            raise ValueError(f"invalid element range for {name}")
        if element_offset % block or count % block:
            raise ValueError(f"unaligned quantized range for {name}")
        offset = header.data_start + tensor.offset + element_offset // block * block_bytes
        length = count // block * block_bytes
        expected_size = SHARDS[header.path.name][0]
        if header.path.stat().st_size == expected_size:
            with header.path.open("rb") as source:
                source.seek(offset)
                data = source.read(length)
            if len(data) != length:
                raise EOFError(f"short local tensor: {name}")
            return data
        key = hashlib.sha256(
            f"{SHARDS[header.path.name][1]}:{offset}:{length}".encode()
        ).hexdigest()
        cached = self.cache_dir / key
        with self._lock:
            if cached.is_file() and cached.stat().st_size == length:
                data = cached.read_bytes()
                if key not in self.disk_files:
                    self.disk_files[key] = length
                    self.disk_used += length
                self.disk_files.move_to_end(key)
                self.stats["disk_hits"] += 1
                return data
        if self.source is None:
            raise FileNotFoundError(f"tensor bytes missing for {name}; supply full shards or --source")
        url = self.source + "/" + header.path.name
        last_error = None
        for attempt in range(4):
            try:
                start = time.monotonic()
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    want = f"bytes {offset}-{offset + length - 1}/{expected_size}"
                    if response.status != 206 or response.headers.get("Content-Range") != want:
                        raise IOError(f"server did not return the exact GGUF range: {response.headers}")
                    data = response.read(length + 1)
                if len(data) != length:
                    raise EOFError(f"short HTTP tensor: {name}, {len(data)} != {length}")
                with self._lock:
                    if length <= self.disk_budget:
                        if key in self.disk_files:
                            self.disk_used -= self.disk_files.pop(key)
                        while self.disk_used + length > self.disk_budget and self.disk_files:
                            victim, size = self.disk_files.popitem(last=False)
                            (self.cache_dir / victim).unlink(missing_ok=True)
                            self.disk_used -= size
                        temporary = cached.with_name(key + f".{threading.get_ident()}.tmp")
                        temporary.write_bytes(data)
                        os.replace(temporary, cached)
                        self.disk_files[key] = length
                        self.disk_used += length
                    self.stats["network_bytes"] += length
                    self.stats["network_seconds"] += time.monotonic() - start
                return data
            except (OSError, EOFError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(attempt + 1)
        raise IOError(f"cannot read {name} from {url}") from last_error

    def decode(self, name, element_offset=0, count=None):
        _, tensor = self.tensors[name]
        count = tensor.elements - element_offset if count is None else count
        raw = self.raw(name, element_offset, count)
        out = np.empty(count, dtype=np.float32)
        source = ctypes.c_char_p(raw)
        rc = self.decode_fn(tensor.type_id, source, out.ctypes.data, count)
        if rc:
            raise ValueError(f"native decoder rejected {name}, type={tensor.type_name}, rc={rc}")
        with self._lock:
            self.stats["decodes"] += 1
        return out


class DeviceWeights:
    """A bounded HBM cache; a live Python tensor pins its allocation during use."""

    def __init__(self, store, device="npu:0", budget_gib=16):
        self.store = store
        self.device = torch.device(device)
        self.budget = int(budget_gib * 1024**3)
        self.used = 0
        self.cache = OrderedDict()
        self.small = {}
        self.hits = self.misses = 0

    def _make_room(self, key, size):
        if size > self.budget:
            raise MemoryError(f"weight {key} exceeds the configured HBM cache")
        while self.used + size > self.budget and self.cache:
            _, victim = self.cache.popitem(last=False)
            self.used -= victim.numel() * victim.element_size()
            del victim

    def _put(self, key, value):
        size = value.numel() * value.element_size()
        self.cache[key] = value
        self.used += size
        return value

    def has(self, key):
        return key in self.cache

    def _key(self, name, expert):
        return name if expert is None else (name, int(expert))

    def fetch_cpu(self, name, expert=None):
        _, tensor = self.store.tensors[name]
        if expert is None:
            values = self.store.decode(name)
            shape = tuple(reversed(tensor.shape))
        else:
            if len(tensor.shape) != 3 or not 0 <= expert < tensor.shape[2]:
                raise ValueError(f"invalid expert {expert} for {name}")
            count = tensor.shape[0] * tensor.shape[1]
            values = self.store.decode(name, expert * count, count)
            shape = (tensor.shape[1], tensor.shape[0])
        return torch.from_numpy(values.reshape(shape))

    def get(self, name, expert=None, prepared=None):
        key = self._key(name, expert)
        value = self.cache.pop(key, None)
        if value is not None:
            self.hits += 1
            self.cache[key] = value
            return value
        self.misses += 1
        host = prepared if prepared is not None else self.fetch_cpu(name, expert)
        self._make_room(key, host.numel() * 2)
        return self._put(key, host.to(device=self.device, dtype=torch.float16))

    def vector(self, name):
        if name not in self.small:
            self.small[name] = self.fetch_cpu(name).to(self.device)
        return self.small[name]

    def linear(self, name, x, expert=None):
        weight = self.get(name, expert)
        return torch.nn.functional.linear(x.to(weight.dtype), weight).float()

    def prefetch_experts(self, prefix, experts):
        jobs = []
        for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            name = prefix + suffix
            for expert in experts:
                key = self._key(name, expert)
                if key not in self.cache:
                    jobs.append((name, expert, self.store.pool.submit(self.fetch_cpu, name, expert)))
        # Only the model thread touches NPU tensors or mutates the HBM cache.
        for name, expert, future in jobs:
            self.get(name, expert, prepared=future.result())

    def report(self):
        return dict(self.store.stats, hbm_weight_bytes=self.used,
                    weight_cache_hits=self.hits, weight_cache_misses=self.misses)
