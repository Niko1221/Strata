"""Emit a tiny synthetic EXL3 n-gram table (`ngram_embedding.safetensors`) plus its expected decoded rows,
for src/kernels/ple_exl3_parity.cpp.  Uses the numpy spec (tools/exl3/ngram.py) as the oracle.

    python3 tools/exl3/emit_ngram_table.py --out <dir>
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from exl3 import ngram                                             # noqa: E402

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    from safetensors.numpy import save_file

    rows, K, dim, heads = args.rows, args.K, ngram.ROW_DIM, 16
    rng = np.random.default_rng(args.seed)
    assert rows % heads == 0

    rings = np.zeros((rows, ngram.words_per_row(K)), dtype=np.int16)
    for r in range(rows):
        sym = rng.integers(0, 1 << K, size=dim, dtype=np.int64)
        st = np.zeros(dim, dtype=np.int64)
        for j in range((15 + K) // K):
            st |= np.roll(sym, j) << (j * K)
        rings[r] = ngram.pack_row((st & 0xFFFF).astype(np.uint16), np.float16(rng.uniform(0.01, 0.1)), K)

    sizes = np.full(heads, rows // heads, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    head_bias = (rng.standard_normal((heads, dim)) * 0.05).astype(np.float16)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({
        PREFIX + ".shard_0.trellis": rings,
        PREFIX + ".head_bias": head_bias,
        PREFIX + ".head_offsets": offsets,
        PREFIX + ".head_vocab_sizes": sizes,
    }, str(out / "ngram_embedding.safetensors"))

    cb = ngram.mul1_codebook()
    exp = np.stack([ngram.decode_table_row(r, rings[r], K, cb, head_bias.astype(np.float32), offsets)
                    for r in range(rows)]).astype(np.float32)
    exp.tofile(str(out / "expected.f32"))
    print("wrote %s: rows=%d K=%d words=%d" % (out / "ngram_embedding.safetensors", rows, K,
                                               ngram.words_per_row(K)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
