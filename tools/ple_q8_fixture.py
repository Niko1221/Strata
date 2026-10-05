#!/usr/bin/env python3
"""tools/ple_q8_fixture.py - a small synthetic PLE table as Q8_0 in a one-tensor GGUF.

Built for the Q8_0 PLE path (Unsloth's UD-Q6_K_XL ships per_layer_token_embd as Q8_0, 170-B rows)
so ple_q8_parity can run without the real 54-GB shard 3.  The bytes are random with a fixed seed:
every Q8_0 block decodes as d * int8, so ANY bytes are valid input, and the point of the parity test
is that Strata's dequantizer agrees with ggml's reference on whatever is there.

    python tools/ple_q8_fixture.py --out /tmp/ple_q8.gguf [--rows 131072] [--seed 7]

Validate the file with tools/gguf_reader.py first; then
    ple_q8_parity /tmp/ple_q8.gguf
"""
import argparse
import pathlib
import struct

import numpy as np

BLOCK = 34              # Q8_0: f16 scale + 32 int8
DIM = 160               # PLE_HEAD_DIM: the fast axis, one row = 5 blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', type=int, default=131072)   # >= 16 for gather_batch, small enough to be quick
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_block = args.rows * (DIM // 32)
    d = rng.uniform(-0.001, 0.001, n_block).astype(np.float16)
    # small scales and int8 values: products stay in exact-float range, no NaN/inf edge cases
    qs = rng.integers(-128, 128, (n_block, 32), dtype=np.int8)
    data = np.empty(n_block * BLOCK, dtype=np.uint8)
    view = data.view(np.uint8).reshape(n_block, BLOCK)
    view[:, 0:2] = d.view(np.uint8).reshape(n_block, 2)
    view[:, 2:] = qs.view(np.uint8)
    assert data.size == args.rows * DIM // 32 * BLOCK

    name = b'per_layer_token_embd.weight'
    buf = bytearray()
    buf += struct.pack('<4sIQQ', b'GGUF', 3, 1, 0)        # magic, version, 1 tensor, 0 metadata kv
    buf += struct.pack('<Q', len(name)) + name            # tensor name (v3: uint64 length)
    buf += struct.pack('<IQQI', 2, DIM, args.rows, 8)     # n_dims, shape [160, rows], type Q8_0, offset 0
    hlen = len(buf)
    pad = (-len(buf)) % 32                                # data_start is alignment-padded (default 32)
    buf += b'\0' * pad
    buf += data.tobytes()
    out = pathlib.Path(args.out)
    out.write_bytes(bytes(buf))
    # the tensor must fill the file from data_start exactly: PleTable asserts it when alone in its shard
    assert out.stat().st_size == len(buf) == hlen + pad + data.size
    print(f'{out}: {args.rows} rows x {DIM} elems x {BLOCK}+0 B blocks = {data.size} B data, header {pad + 4 + len(name) + 4 + 16 + 4 + 8} B (pad {pad})')


if __name__ == '__main__':
    main()