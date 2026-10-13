#!/usr/bin/env python3
"""Check a --dump-logits file and, when you pass the run's printed tokens, prove the rows are the
distributions the engine actually sampled from.

The file is int32 n_vocab, int32 n_rows, then n_rows rows of n_vocab float32. A header that promises rows
the file does not contain is a defect, not a cosmetic one: anything reading the count gets a wrong answer
that looks authoritative (tools/logits_identical.py's successor to that idea is this file).

    python tools/logits_dump_check.py tmp/fix-maxnew8.txt 8 828,381,3817,1452,430,4241,248045,487

The third argument is the `output :` line of the run's own report, comma separated. With it, the check is
not "the file has the right shape" but "the dumped logits pick the tokens the engine printed" - the
verification that the native pack's speculative loop writes the RIGHT row at the RIGHT position.
"""
from __future__ import annotations

import math
import os
import struct
import sys


def check(path: str, expect_rows: int, tokens: list[int] | None = None) -> bool:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(8)
        if len(hdr) < 8:
            print(f"{path}: size={size} header=MISSING (file smaller than 8 bytes)")
            return False
        n_vocab, n_rows = struct.unpack("<ii", hdr)
        body = f.read()                      # body starts AFTER the header; row r is at r * n_vocab * 4
    if n_vocab <= 0 or n_rows < 0:
        # a malformed header must fail the check, not divide by zero below
        print(f"{path}: size={size} header={{n_vocab={n_vocab}, n_rows={n_rows}}}")
        print(f"  DEFECT: invalid header dimensions (n_vocab must be > 0, n_rows >= 0)")
        return False
    ok = True
    row_bytes = n_vocab * 4
    print(f"{path}: size={size} header={{n_vocab={n_vocab}, n_rows={n_rows}}} "
          f"rows_in_file={(size - 8) // max(1, row_bytes)} expected_rows={expect_rows} "
          f"expected_size={8 + row_bytes * expect_rows}")
    if n_rows != expect_rows:
        print(f"  DEFECT: header count {n_rows} != expected {expect_rows}"); ok = False
    if size != 8 + row_bytes * expect_rows:
        print(f"  DEFECT: size {size} != 8 + {n_vocab}*4*{expect_rows} = {8 + row_bytes * expect_rows}"); ok = False
    if (size - 8) % row_bytes != 0:
        print("  DEFECT: body is not a whole number of rows"); ok = False
    rows = (size - 8) // row_bytes
    nonfinite, argmaxes = 0, []
    for r in range(rows):
        vals = struct.unpack(f"<{n_vocab}f", body[r * row_bytes:(r + 1) * row_bytes])
        nonfinite += sum(1 for v in vals if not math.isfinite(v))
        argmaxes.append(max(range(n_vocab), key=lambda i: vals[i]))
    if nonfinite:
        print(f"  DEFECT: {nonfinite} non-finite values across {rows} rows"); ok = False
    print(f"  rows={rows} all finite; argmaxes={argmaxes}")
    if tokens is not None:
        if tokens[:rows] == argmaxes[:rows]:
            print(f"  argmaxes match the run's printed tokens for all {rows} rows")
        else:
            bad = [i for i in range(min(rows, len(tokens))) if tokens[i] != argmaxes[i]]
            print(f"  DEFECT: argmax != printed token at rows {bad[:8]}"
                  f"{'' if len(bad) <= 8 else f' (+{len(bad) - 8} more)'}")
            ok = False
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    toks = None
    if len(sys.argv) > 3:
        toks = [int(x) for x in sys.argv[3].replace(" ", "").split(",") if x != ""]
    sys.exit(0 if check(sys.argv[1], int(sys.argv[2]), toks) else 1)
