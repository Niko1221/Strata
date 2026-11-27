"""Small synthetic .strata files for the tests (no model, no compressor): a header frame, three sections and a Section 0 manifest written as a zstd frame
of one stored block, so any Python reads it (strata_control/strata_manifest.py). `k16` selects the 16 KiB-aligned layout (section table at 0x20)."""

from __future__ import annotations

import json
import struct

ENTRY = struct.Struct("<QQ32sIIQ")


def zstd_stored_frame(text: bytes) -> bytes:
    return struct.pack("<IBI", 0xFD2FB528, 0xA0, len(text)) + struct.pack("<I", 1 | (len(text) << 3))[:3] + text


def align(n: int, page: int) -> int:
    return (n + page - 1) // page * page


def synthetic_strata(manifest: dict | None = None, k16: bool = False, profile: int = 0, pad: int = 0) -> bytes:
    manifest = {"format": "strata-v1.1", "arch": "inkling2_loop"} if manifest is None else manifest
    page = 16384 if k16 else 4096
    man = zstd_stored_frame(json.dumps(manifest).encode())
    s1 = (page, 100)
    s3 = (align(s1[0] + s1[1], page), align(24 * 34, page) if k16 else 24 * 34)
    s0 = (align(s3[0] + s3[1], page), len(man))
    size = align(s0[0] + s0[1], page) + pad
    f = bytearray(size)
    f[0:8] = b"STRATA\x00\x01"
    struct.pack_into("<IHH", f, 8, 1, profile, 6)
    table = 0x20 if k16 else 0x10
    if k16:
        struct.pack_into("<H", f, 0x10, 7)
    spans = {0: s0, 1: s1, 3: s3}
    for i in range(6):
        off, ln = spans.get(i, (0, 0))
        f[table + 64 * i: table + 64 * (i + 1)] = ENTRY.pack(off, ln, bytes([i]) * 32, i, 1 if i == 3 else 0, 0)
    f[s0[0]: s0[0] + len(man)] = man
    return bytes(f)


def inkling_strata(k16: bool = False, pad: int = 0) -> bytes:
    return synthetic_strata({"format": "strata-v1.1", "arch": "inkling2_loop", "config": {"model_type": "inkling2_loop"}}, k16, pad=pad)


def hy4_strata(k16: bool = False, pad: int = 0) -> bytes:
    return synthetic_strata({"format": "strata-v1.1", "arch": "hy4flash", "config": {"model_type": "hy_v4"}}, k16, pad=pad)
