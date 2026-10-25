"""tools/test_tqp_geometry.py - the TQP5 / TQP6 / TQP7 pair trellis geometry (ggml types 147-149) in
tools/gguf_reader.py, and the packer's decoder in tools/iq_pack.py against the format's definition.

    python3 tools/test_tqp_geometry.py [model.gguf]

Without a model the GGUF part is skipped; the synthetic part needs no model and no gguf-py.
"""
from __future__ import annotations

import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_reader as G  # noqa: E402
import iq_pack as P  # noqa: E402

# the fork's block structs (ggml-common.h): fp16 d + a circular 64*K-bit stream of 8*K bytes
TQP = {147: ("TQP5", 5, 42), 148: ("TQP6", 6, 50), 149: ("TQP7", 7, 58)}


def test_maps():
    for tid, (name, k, bb) in TQP.items():
        assert G.GGML_TYPES[tid] == name, (tid, G.GGML_TYPES.get(tid))
        elems, bytes_per_block = G.BLOCK_GEOMETRY[name]
        assert elems == 128, name
        assert bytes_per_block == bb, (name, bytes_per_block)
        assert bytes_per_block == 2 + 8 * k, (name, bytes_per_block)


def decode_reference(block: bytes, k: int) -> list[float]:
    """The format's definition (the fork's tqp_state + the first codebook pair), written here directly."""
    lut = P.tq2t_lut()
    d = struct.unpack("<e", block[:2])[0]
    qs = block[2:]
    out = []
    for t in range(64):
        off = ((63 - t) * k) % (64 * k)
        state = 0
        for bit in range(16):
            pos = (off + bit) % (64 * k)
            state |= ((qs[pos // 8] >> (pos % 8)) & 1) << bit
        pa = (((state & 0x7FFF) * 0x9E3779B1) & 0xFFFFFFFF) >> 21   # the C multiply wraps at 32 bits
        out.append(lut[2 * pa] * d)
        out.append(lut[2 * pa + 1] * d)
    return out


def test_decode():
    import numpy as np
    rng = __import__("random").Random(147)
    for tid, (name, k, bb) in TQP.items():
        raw = bytearray()
        want = []
        for _ in range(8):
            block = bytes(rng.getrandbits(8) for _ in range(bb))
            block = struct.pack("<e", 0.017) + block[2:]   # a finite fp16 scale
            raw += block
            want += decode_reference(block, k)
        got = P.dequantize_tqp(np.frombuffer(bytes(raw), dtype=np.uint8), k)
        assert got.shape == (8 * 128,), (name, got.shape)
        assert all(a == b for a, b in zip(got, want)), name   # bit-exact against the definition


def test_model(path: pathlib.Path):
    g = G.GGUFFile(path)
    seen = {}
    for t in g.tensors:
        if t.type_id in TQP:
            exp = t.expected_bytes()
            assert exp is not None, t.name
            assert t.elements % 128 == 0, t.name
            assert exp == t.elements // 128 * TQP[t.type_id][2], (t.name, exp)
            seen[t.type_id] = seen.get(t.type_id, 0) + 1
    assert seen, f"{path.name}: no type 147/148/149 tensor"
    return seen


def main() -> int:
    test_maps()
    test_decode()
    print("tqp geometry: maps and decoder PASS")
    if len(sys.argv) > 1:
        seen = test_model(pathlib.Path(sys.argv[1]))
        for tid, n in sorted(seen.items()):
            print("  %s: %d tensor(s)" % (TQP[tid][0], n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
