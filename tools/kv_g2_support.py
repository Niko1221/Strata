#!/usr/bin/env python3
"""G2 support: reduce the reference run to the sample and write the per-segment id files.

Reads the reference dump (`STRATA_LOGPOS_TOPK=4096`, all positions), keeps the 1,024 sample positions, and
writes what the arm runs need:

  - `ref-top4096.tsv`: the reference on the sample, eight fixed columns + its top-4,096 per row.  This is the
    reference side of kl256 and kl4096; the top-256 is the prefix of the top-4,096, so one file serves both.
  - `ids/s00.txt` .. `ids/s07.txt`: one line per position, `pos id id id ...`, ascending.  At every position
    the segment's own top-4,096 - the arm emits exactly the support the reference has, so nothing is wasted
    and no support has to be shared between segments.  At the positions where the top-4,096 holds less than
    `--retained` (default 0.95) of the reference's mass, the line is the whole vocabulary: that row is then
    measured exactly, with no truncation at all.
  - `ids/sNN-full.txt`: only those exact rows, for the reference run that supplies the other side of them.
  - `support.json`: provenance and the measured distributions.

    python tools/kv_g2_support.py --ref .../fp16-top4096.tsv --positions g2-positions.txt --out-dir DIR
                                  [--retained 0.95] [--vocab 248320] [--s3 .../fp16.tsv]

The vocabulary size is the GGUF's `tokenizer.ggml.tokens` length (248,320 for swift-1.5-iq3_xxs); ids past the
model's own `n_vocab` are skipped by the engine, so an over-long list is harmless, a short one is not.

`--s3` adds the check that matters for the top-256 side: on every sample row, the reference's top-256 must hold the
same ids with the same logprobs as the existing fp16 run.  The order of the pairs is not part of the check - exact
ties come out in another order at a different k, and KL does not see the order.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quant_quality as qq  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dist(values: list) -> dict:
    v = sorted(values)
    n = len(v)

    def q(p: float) -> float:
        return v[min(n - 1, max(0, int(p * (n - 1))))]

    return {"n": n, "min": v[0], "p01": q(0.01), "p10": q(0.10), "median": q(0.50),
            "mean": math.fsum(v) / n, "max": v[-1]}


def sample_rows(path: pathlib.Path, want: set):
    """Yield (segment, pos, fields) for the sample positions only; the position is parsed before splitting."""
    meta = json.load(open(str(path) + ".seg.json", encoding="utf-8"))
    bounds = [(s["name"], s["first_line"], s["end_line"]) for s in meta["segments"]]
    si = rows = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            while si < len(bounds) and rows >= bounds[si][2]:
                si += 1
            if si >= len(bounds):
                break
            p = int(line[:line.find("\t")])
            if p in want:
                yield bounds[si][0], p, line.rstrip("\n").split("\t")
            rows += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="the reference dump with TOPK=4096 over all positions")
    ap.add_argument("--positions", default="g2-positions.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--retained", type=float, default=0.95,
                    help="rows whose top-4096 holds less than this are dumped over the whole vocabulary")
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--s3", help="the existing fp16 TSV, for the top-256 pair check")
    ap.add_argument("--json", default="support.json")
    a = ap.parse_args()

    ref_path = pathlib.Path(a.ref)
    out_dir = pathlib.Path(a.out_dir)
    (out_dir / "ids").mkdir(parents=True, exist_ok=True)
    pos = [int(x) for x in pathlib.Path(a.positions).read_text(encoding="utf-8").split()]
    want = set(pos)
    meta = json.load(open(str(ref_path) + ".seg.json", encoding="utf-8"))
    names = [s["name"] for s in meta["segments"]]

    # --- reduce the reference and keep the ids and the retained mass of the sample rows
    reduced = out_dir / "ref-top4096.tsv"
    ids, ret4, n_by_seg = {}, {}, {s: 0 for s in names}
    with open(reduced, "w", encoding="utf-8", newline="\n") as dst:
        for name, p, f in sample_rows(ref_path, want):
            dst.write("\t".join(f[:8 + 4096]) + "\n")
            pairs = f[8:8 + 4096]
            ids[(name, p)] = array.array("i", (int(z.partition(":")[0]) for z in pairs))
            ret4[(name, p)] = math.fsum(math.exp(float(z.partition(":")[2])) for z in pairs)
            n_by_seg[name] += 1
    red_segs, first = [], 0
    for s in names:
        red_segs.append({"name": s, "first_line": first, "end_line": first + n_by_seg[s]})
        first += n_by_seg[s]
    (out_dir / "ref-top4096.tsv.seg.json").write_text(json.dumps(
        {"segments": red_segs, "config": meta.get("config"), "args": meta.get("args"), "env": [],
         "kv": meta.get("kv"), "topk": 4096, "expert_cache_lines": meta.get("expert_cache_lines", []),
         "source": str(ref_path)}, indent=1), encoding="utf-8")

    # --- the exact rows: the top-4096 misses more than 1 - retained of the reference's mass
    exact = {s: sorted(p for p in pos if ret4[(s, p)] < a.retained) for s in names}
    full_line = " ".join(map(str, range(a.vocab)))
    id_bytes = {}
    for s in names:
        exact_set = set(exact[s])
        with open(out_dir / "ids" / f"{s}.txt", "w", encoding="utf-8", newline="\n") as f, \
             open(out_dir / "ids" / f"{s}-full.txt", "w", encoding="utf-8", newline="\n") as g:
            for p in pos:
                if p in exact_set:
                    f.write(f"{p} {full_line}\n")
                    g.write(f"{p} {full_line}\n")
                else:
                    f.write(f"{p} " + " ".join(map(str, sorted(ids[(s, p)]))) + "\n")
        id_bytes[s] = (out_dir / "ids" / f"{s}.txt").stat().st_size

    # --- the top-256 check against the existing run, by id and not by position
    #
    # The order of the pairs is not a measurement: the dump sorts by logprob and stops at the top-k, and tokens
    # whose logprobs are exactly equal come out in a different order at k=256 and at k=4096 (measured: 37 of the
    # 8,192 rows, always two or more tokens with the same six decimals, the sequence of values identical).  What
    # has to agree is which ids are in the set and what logprob each one has, because that is all KL uses.
    s3_check = None
    if a.s3:
        same = order_only = bad = 0
        with open(reduced, encoding="utf-8") as fr:
            for name, p, g in sample_rows(pathlib.Path(a.s3), want):
                r = next(fr).rstrip("\n").split("\t")
                if r[0] != g[0]:
                    raise SystemExit(f"row order differs at {name} pos {p}: {r[0]} != {g[0]}")
                same += 1
                ra, ga = r[8:8 + 256], g[8:8 + 256]
                if ra == ga:
                    continue
                if {z.partition(":")[0]: z.partition(":")[2] for z in ra} == \
                        {z.partition(":")[0]: z.partition(":")[2] for z in ga}:
                    order_only += 1
                else:
                    bad += 1
        s3_check = {"rows": same, "identical": same - order_only - bad,
                    "order_only": order_only, "differing": bad}

    support = {
        "reference": {"path": str(ref_path), "sha256": sha256(ref_path)},
        "positions": {"path": str(a.positions), "sha256": sha256(pathlib.Path(a.positions)), "n": len(pos)},
        "reduced": {"path": str(reduced), "sha256": sha256(reduced), "bytes": reduced.stat().st_size,
                    "rows": sum(n_by_seg.values()), "topk": 4096},
        "ids": {s: {"path": str(out_dir / "ids" / f"{s}.txt"),
                    "sha256": sha256(out_dir / "ids" / f"{s}.txt"),
                    "bytes": id_bytes[s], "exact_positions": exact[s], "exact_rows": len(exact[s])}
                for s in names},
        "params": {"retained_threshold": a.retained, "vocab": a.vocab},
        "retained_top4096": dist([ret4[(s, p)] for s in names for p in pos]),
        "retained_per_segment": {s: dist([ret4[(s, p)] for p in pos]) for s in names},
        "exact_rows": sum(len(exact[s]) for s in names),
        "s3_top256_check": s3_check,
    }
    pathlib.Path(out_dir / a.json).write_text(json.dumps(support, indent=1), encoding="utf-8")

    d = support["retained_top4096"]
    print(f"sample {len(pos)} positions x {len(names)} segments = {support['reduced']['rows']} rows")
    print(f"retained mass of the reference's top-4096: min {d['min']:.4f}  p01 {d['p01']:.4f}  "
          f"p10 {d['p10']:.4f}  median {d['median']:.4f}  mean {d['mean']:.4f}")
    print(f"exact rows (retained < {a.retained}): {support['exact_rows']}")
    for s in names:
        print(f"  {s}: {len(exact[s]):3d} rows  ids {id_bytes[s] / 1e6:6.1f} MB  "
              f"positions {exact[s][:8]}{' ...' if len(exact[s]) > 8 else ''}")
    if s3_check:
        print(f"\ntop-256 check against {a.s3}: {s3_check['rows']} rows, "
              f"{s3_check['identical']} identical, {s3_check['order_only']} same pairs in another order, "
              f"{s3_check['differing']} differing")
    print(f"\nreduced reference {support['reduced']['bytes'] / 1e6:.0f} MB, "
          f"ids files {sum(id_bytes.values()) / 1e6:.0f} MB")
    print(f"wrote {reduced} and {out_dir / a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
