from __future__ import annotations

import io
import struct
from typing import Any


class MinimalGgufBuilder:
    """In-memory GGUF v2/v3 builder for testing parser boundaries and resilience."""

    def __init__(self, version: int = 3):
        self.version = version
        self.metadata: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, list[int], int, int, bytes]] = []

    def add_u32(self, key: str, value: int) -> MinimalGgufBuilder:
        self.metadata.append((key, 4, struct.pack("<I", value)))
        return self

    def add_u64(self, key: str, value: int) -> MinimalGgufBuilder:
        self.metadata.append((key, 10, struct.pack("<Q", value)))
        return self

    def add_string(self, key: str, value: str) -> MinimalGgufBuilder:
        enc = value.encode("utf-8")
        payload = struct.pack("<Q", len(enc)) + enc
        self.metadata.append((key, 8, payload))
        return self

    def add_array_string(self, key: str, values: list[str]) -> MinimalGgufBuilder:
        """Add an array of strings (type 9, element type 8)."""
        buf = io.BytesIO()
        buf.write(struct.pack("<I", 8))  # Element type = STRING
        buf.write(struct.pack("<Q", len(values)))  # Array length
        for v in values:
            enc = v.encode("utf-8")
            buf.write(struct.pack("<Q", len(enc)))
            buf.write(enc)
        self.metadata.append((key, 9, buf.getvalue()))
        return self

    def add_tensor(self, name: str, dims: list[int], type_id: int, offset: int, data: bytes = b"") -> MinimalGgufBuilder:
        self.tensors.append((name, dims, type_id, offset, data))
        return self

    def build(self) -> bytes:
        buf = io.BytesIO()
        # 1. Magic + version
        buf.write(b"GGUF")
        buf.write(struct.pack("<I", self.version))

        # 2. Counts
        buf.write(struct.pack("<Q", len(self.tensors)))
        buf.write(struct.pack("<Q", len(self.metadata)))

        # 3. Key-values
        for key, val_type, payload in self.metadata:
            k_enc = key.encode("utf-8")
            buf.write(struct.pack("<Q", len(k_enc)))
            buf.write(k_enc)
            buf.write(struct.pack("<I", val_type))
            buf.write(payload)

        # 4. Tensor directory
        for name, dims, type_id, offset, _ in self.tensors:
            n_enc = name.encode("utf-8")
            buf.write(struct.pack("<Q", len(n_enc)))
            buf.write(n_enc)
            buf.write(struct.pack("<I", len(dims)))
            for d in dims:
                buf.write(struct.pack("<Q", d))
            buf.write(struct.pack("<I", type_id))
            buf.write(struct.pack("<Q", offset))

        # Alignment padding (32-byte default)
        pos = buf.tell()
        pad = (32 - (pos % 32)) % 32
        buf.write(b"\x00" * pad)

        # Tensor payload data
        for _, _, _, _, data in self.tensors:
            if data:
                buf.write(data)

        return buf.getvalue()
