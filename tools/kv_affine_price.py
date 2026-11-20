#!/usr/bin/env python
"""The fast-path decision, measured on the host: **what does the affine restriction cost?**

The batched prompt attention (Mode 4 of src/kernels/cuda/qsa_prompt_attn.cu) computes
``sum_i q_i * (code - 8)`` per group of 32 values and applies the group's scale afterwards.  That
supports **every affine map along the head dimension** -- the scale is already per group.  What it
does not carry is a **non-uniform level table**: the only value mapping it knows is ``code - 8``.

So the fast path has exactly two candidates, and this tool measures what each of them costs:

  (a) the table in the operand build: ``level[code]`` instead of ``code - 8``.  No quality loss at
      all, but one lookup per element (the codebook family: TurboQuant, FLUTE).
  (b) the codec affine: a **uniform grid**, which is ``code - 8`` with a scale -- exactly what
      ``q4_0`` is.  No kernel change at all.  The price is the loss of the non-uniform shape.

The question the user posed is (b) with one more degree of freedom: an **affine shear** -- a shifted
uniform grid with a per-block zero point, ``x ~ s * (code - z)``.  A zero point other than 8 is not
free in the kernel (the reconstruction differs from ``s*(code-8)`` by the constant ``s*(z-8)`` per
block, and that constant has to be added to the score / weighted into the output), so the variants
are separated:

  lloyd16+max        the engine's q4lm32 table (the anchor: the product codec's table)
  lloyd16+LS         the engine's FIT variant (one least-squares pass over the block)
  uniform+max        q4_0, the campaign's anchor
  uniform+LS         the kernel-free affine route: a uniform grid with a fitted step, zero point 0
  uniform+LS+z       the best affine map per block: a fitted step AND a fitted zero point (the shear)

The readings that decide:

  uniform price   = lloyd16+LS  vs uniform+LS     (the shape of the table, with the placement free)
  shear price     = uniform+LS   vs uniform+LS+z   (the placement, with the shape uniform)
  affine price    = lloyd16+max  vs uniform+LS     (the whole price of the kernel-free route)
  affine price +  = lloyd16+LS   vs uniform+LS+z   (the price of the best affine map)
  zero point size = how far the fitted zero point is from the grid's centre, in steps: if it is small,
                    the constant the kernel would have to add is small too.

Everything is read from the engine's own cell dump (the rotated K/V rows of the batched prompt
path), so the numbers describe the engine's data and not a simulation of it.  The dump reader and
the level tables come from tools/kv_codec_mse.py, which mirrors the engine's kernels
(src/kernels/cuda/kv_q4.cu).

Quote the mode with the number: --center subtracts the per-head per-channel offset that the engine
itself uses (the table of tools/kv_center_table.py), --center-s2 the naive variant of the S2 probe.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_codec_mse import (  # noqa: E402
    B16, KQ4, L16, L16_MAX, fwht, quantize_lloyd, read_dump, read_s2_means, _offset_for,
)


def quantize_uniform_zero(x, fit):
    """The engine's q4_0 rule (grid -8d..7d, zero point 0), optionally with one least-squares step.

    x: (N, 32) blocks.  Returns the reconstruction and the step per block."""
    amax = np.max(np.abs(x), axis=1)
    idx = np.argmax(np.abs(x), axis=1)
    mval = x[np.arange(x.shape[0]), idx]
    d = mval / -8.0
    d = np.where(amax == 0.0, 0.0, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        idd = np.where(d != 0.0, 1.0 / d, 0.0)
    code = np.clip(np.trunc(x * idd[:, None] + 8.5), 0.0, 15.0)
    if fit:
        c = code - 8.0
        den = np.sum(c * c, axis=1)
        num = np.sum(c * x, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_ls = np.where(den > 0, num / den, d)
        ok = (d_ls != 0.0) & (np.sign(d_ls) == np.sign(d))
        d = np.where(ok, d_ls, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            idd = np.where(d != 0.0, 1.0 / d, 0.0)
        code = np.clip(np.trunc(x * idd[:, None] + 8.5), 0.0, 15.0)
    recon = (code - 8.0) * d[:, None]
    return recon, d


def quantize_uniform_shear(x, iters=6):
    """The best affine map per block, implementable: a uniform grid with a fitted step AND a
    fitted zero point.

    The grid is {o + s*k : k = 0..15}; the reconstruction is o + s*k, i.e. the kernel's
    ``s*(code-8)`` plus the constant ``o + 8s`` (the zero point in the kernel's units).  The fit
    alternates the assignment (the nearest grid point, clipped to the 16 levels) with the
    least-squares solution for (o, s) given that assignment -- the same kind of two-pass fit the
    engine's FIT variant does (the butterfly sums of sum(c), sum(c^2), sum(c*x), sum(x)).

    The returns: the reconstruction, the step, the zero point in code units (the code that maps
    to the value 0) and the constant the kernel would have to add (o + 8s)."""
    lo = np.min(x, axis=1)
    hi = np.max(x, axis=1)
    s = np.where((hi - lo) > 0, (hi - lo) / 15.0, 1.0)
    o = lo.copy()
    code = np.zeros_like(x)
    for _ in range(iters):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(s > 0, 1.0 / s, 0.0)
        code = np.clip(np.rint((x - o[:, None]) * inv[:, None]), 0.0, 15.0)
        cm = code.mean(axis=1)
        xm = x.mean(axis=1)
        dc = code - cm[:, None]
        dx = x - xm[:, None]
        den = np.sum(dc * dc, axis=1)
        num = np.sum(dc * dx, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_new = np.where(den > 1e-12, num / den, s)
        s = np.where(s_new > 0, s_new, s)
        o = xm - s * cm
    # die letzte Zuordnung mit dem letzten (o, s): die Rekonstruktion muss zum Code passen
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(s > 0, 1.0 / s, 0.0)
    code = np.clip(np.rint((x - o[:, None]) * inv[:, None]), 0.0, 15.0)
    recon = o[:, None] + s[:, None] * code
    # der Nullpunkt im Kernel-Mass: der Kernel rechnet s*(code-8), die Karte ist o + s*code.
    # Der Nullpunkt z ist der Code, der auf den Wert 0 abbildet: o + s*z = 0 -> z = -o/s.
    # z = 8 heisst: das Gitter ist genau das des Kernels (kein Nullpunkt noetig).
    zp = np.where(s > 0, -o / np.where(s > 0, s, 1.0), 8.0)
    const = o + 8.0 * s
    return recon, s, zp, const


def affine_oracle(x):
    """The exhaustive oracle of the best affine map of one block (a uniform grid with a free
    offset), by a dense scan over (offset, step).  It is the reference the implementable rule is
    measured against -- not something an engine could do per block."""
    lo, hi = float(x.min()), float(x.max())
    best = float("inf")
    rng = hi - lo
    if rng <= 0:
        return 0.0
    for o in np.linspace(lo - 0.4 * rng, hi + 0.05 * rng, 201):
        for st in np.linspace(rng / 15 * 0.35, rng / 15 * 1.6, 81):
            code = np.clip(np.rint((x - o) / st), 0, 15)
            e = float(np.mean((o + st * code - x) ** 2))
            if e < best:
                best = e
    return best


def quantize_lloyd_vec(x, levels, fit):
    """The engine's q4l_group, vectorized over blocks (tools/kv_codec_mse.py does one block at a
    time).  The level assignment uses the engine's boundary rule."""
    levels = np.asarray(levels, dtype=np.float64)
    lmax = float(np.max(np.abs(levels)))
    amax = np.max(np.abs(x), axis=1)
    d = np.where(amax > 0, amax / lmax, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x / np.where(d > 0, d, 1.0)[:, None]
    idx = np.searchsorted(B16, y, side="right")
    if fit:
        l = levels[idx]
        den = np.sum(l * l, axis=1)
        num = np.sum(l * x, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_ls = np.where(den > 0, num / den, d)
        d = np.where(d_ls > 0, d_ls, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = x / np.where(d > 0, d, 1.0)[:, None]
        idx = np.searchsorted(B16, y, side="right")
    recon = levels[idx] * d[:, None]
    return recon, d


def blocks_of(arr, nblk):
    """(rows, nkv, hd) -> (rows*nkv*nblk, 32) blocks along the head dimension."""
    rows, nkv, hd = arr.shape
    b = arr.reshape(rows * nkv, nblk, KQ4)
    return b.reshape(-1, KQ4)


class Acc:
    """The accumulators: the mean squared error per block, per side, and the zero point's spread.

    Der Zaehler wird **je Variante und Seite** gefuehrt.  Frueher stand hier ein gemeinsamer
    Zaehler ``n[side]``, der bei jedem ``add`` hochgezaehlt wurde -- also fuenfmal je (Record,
    Seite), weil fuenf Varianten addiert werden.  Jede MSE war damit **1/5 des Werts** (Abweichung
    20 im Protokoll: Faktor exakt 4,99-5,00 auf allen fuenf Varianten).  Die dB-Werte waren als
    Verhaeltnisse unberuehrt, die absoluten MSE nicht."""

    def __init__(self):
        self.err = {}

    def add(self, name, e, side):
        s = self.err.get((name, side))
        if s is None:
            self.err[(name, side)] = [float(e), 1]
        else:
            s[0] += float(e)
            s[1] += 1

    def mse(self, name):
        """The mean over both sides (the campaign's measure) and the two sides separately."""
        def side_mean(side):
            s = self.err.get((name, side))
            return s[0] / max(s[1], 1) if s else float("nan")

        k = side_mean("K")
        v = side_mean("V")
        return 0.5 * (k + v), k, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-records", type=int, default=None, help="only the first N records")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth cell (the blocks stay whole)")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--center-s2", action="store_true")
    ap.add_argument("--oracle", type=int, default=0,
                    help="on the first N blocks, also compute the exhaustive affine oracle")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    means = read_s2_means(args.s2)
    offsets_rot = {layer: fwht(v) for layer, v in means.items()}

    # the offsets from the dump itself (per layer, per KV head, per channel, K and V separate) --
    # the table the engine uses.
    offs = {}
    cnts = {}
    recs = read_dump(args.dump)
    if not recs:
        raise SystemExit("no records in the dump")
    if args.layers:
        # the dump's layers are the QSA layers of the model (3, 7, 11, ... 47), not 0..N-1: take
        # the first N layers in the order the dump holds them.
        wanted = []
        for r in recs:
            if r[0] not in wanted:
                wanted.append(r[0])
            if len(wanted) == args.layers:
                break
        recs = [r for r in recs if r[0] in set(wanted)]
        print(f"layers: {sorted(wanted)}")
    if args.max_records:
        recs = recs[:args.max_records]
    for layer, p0, T, nkv, hd, karr, varr in recs:
        for tag, arr in (("K", karr), ("V", varr)):
            for h in range(arr.shape[1]):
                s = offs.setdefault((tag, layer, h), np.zeros(hd))
                s += arr[:, h, :].sum(axis=0)
                cnts[(tag, layer, h)] = cnts.get((tag, layer, h), 0) + T
    for key in offs:
        offs[key] = offs[key] / max(cnts[key], 1)

    variants = ("lloyd16+max", "lloyd16+LS", "uniform+max", "uniform+LS", "uniform+LS+z")
    acc = Acc()
    zp_all = []
    const_all = []
    oracle_blocks = []
    nblocks = 0
    for layer, p0, T, nkv, hd, karr, varr in recs:
        if args.max_cells:
            karr, varr = karr[:args.max_cells], varr[:args.max_cells]
            T = args.max_cells
        if args.stride > 1:
            karr, varr = karr[::args.stride], varr[::args.stride]
            T = karr.shape[0]
        for side, arr in (("K", karr), ("V", varr)):
            if args.center or args.center_s2:
                arr = arr - _offset_for(args, offs, offsets_rot, layer, arr.shape[1], hd, True, side)
            bl = blocks_of(arr, 8)
            if bl.shape[0] == 0:
                continue
            nblocks += bl.shape[0]
            r, d = quantize_lloyd_vec(bl, L16, False)
            acc.add("lloyd16+max", np.mean((r - bl) ** 2), side)
            r, d = quantize_lloyd_vec(bl, L16, True)
            acc.add("lloyd16+LS", np.mean((r - bl) ** 2), side)
            r, d = quantize_uniform_zero(bl, False)
            acc.add("uniform+max", np.mean((r - bl) ** 2), side)
            r, d = quantize_uniform_zero(bl, True)
            acc.add("uniform+LS", np.mean((r - bl) ** 2), side)
            r, s, zp, const = quantize_uniform_shear(bl)
            acc.add("uniform+LS+z", np.mean((r - bl) ** 2), side)
            zp_all.append(zp)
            const_all.append(const / np.where(s > 0, s, 1.0))
            if args.oracle and len(oracle_blocks) < args.oracle:
                for i in range(min(args.oracle - len(oracle_blocks), bl.shape[0])):
                    oracle_blocks.append((bl[i].copy(), float(np.mean((r[i] - bl[i]) ** 2))))

    zp_all = np.concatenate(zp_all) if zp_all else np.zeros(1)
    const_all = np.concatenate(const_all) if const_all else np.zeros(1)

    base = acc.mse("lloyd16+max")[0]
    print(f"dump: {len(recs)} records, {nblocks} blocks "
          f"(stride {args.stride}, centered: {'ja' if args.center else 'nein'}"
          f"{' (S2)' if args.center_s2 else ''})")
    print(f"{'Variante':16s} {'MSE':>13s} {'K':>13s} {'V':>13s} {'dB gegen lloyd16+max':>22s}")
    for name in variants:
        m, mk, mv = acc.mse(name)
        db = 10.0 * math.log10(m / base) if m > 0 and base > 0 else float("nan")
        print(f"{name:16s} {m:13.6e} {mk:13.6e} {mv:13.6e} {db:+22.2f}")

    def db(a, b):
        ma, mb = acc.mse(a)[0], acc.mse(b)[0]
        return 10.0 * math.log10(ma / mb) if ma > 0 and mb > 0 else float("nan")

    print()
    print("Die Preise (positiv = die Lloyd-Tabelle ist besser):")
    print(f"  Uniformitätspreis (Form vs Platzierung)  lloyd16+LS / uniform+LS   {db('lloyd16+LS', 'uniform+LS'):+.3f} dB")
    print(f"  Scherungspreis     (Platzierung)         uniform+LS  / uniform+LS+z {db('uniform+LS', 'uniform+LS+z'):+.3f} dB")
    print(f"  Affiner Preis     (kernelfrei)           lloyd16+max / uniform+LS    {db('lloyd16+max', 'uniform+LS'):+.3f} dB")
    print(f"  Affiner Preis     (beste affine Karte)   lloyd16+LS  / uniform+LS+z {db('lloyd16+LS', 'uniform+LS+z'):+.3f} dB")
    print()
    print("Der Nullpunkt (z = der Code, der auf den Wert 0 abbildet; z = 8 heisst: das Gitter ist")
    print("das des Kernels, es braucht keine zusaetzliche Konstante):")
    print(f"  Median |z - 8| = {np.median(np.abs(zp_all - 8.0)):.4f} Schritte, "
          f"90 %-Quantil {np.quantile(np.abs(zp_all - 8.0), 0.9):.4f}, "
          f"Max {np.max(np.abs(zp_all - 8.0)):.4f}")
    print(f"  Die Konstante, die der Kernel je Block addieren muesste (in Schritten): "
          f"Median |8 - z| = {np.median(np.abs(const_all)):.4f}")

    if args.oracle and oracle_blocks:
        orc = np.array([affine_oracle(x) for x, _ in oracle_blocks])
        mine = np.array([m for _, m in oracle_blocks])
        print(f"  Die implementierbare Regel gegen die erschoepfende Oracle "
              f"({len(orc)} Bloecke): {mine.mean():.6e} gegen {orc.mean():.6e} "
              f"({10.0 * math.log10(mine.mean() / max(orc.mean(), 1e-30)):+.3f} dB)")

    if args.json:
        import json
        out = {
            "variants": {v: acc.mse(v)[0] for v in variants},
            "variants_per_side": {v: {"K": acc.mse(v)[1], "V": acc.mse(v)[2]} for v in variants},
            "db_vs_lloy16max": {v: 10.0 * math.log10(acc.mse(v)[0] / base) for v in variants},
            "prices_db": {
                "uniformity_lloydLS_vs_uniformLS": db("lloyd16+LS", "uniform+LS"),
                "shear_uniformLS_vs_uniformLSz": db("uniform+LS", "uniform+LS+z"),
                "affine_kernelfree_lloyd16max_vs_uniformLS": db("lloyd16+max", "uniform+LS"),
                "affine_best_lloyd16LS_vs_uniformLSz": db("lloyd16+LS", "uniform+LS+z"),
            },
            "zero_point": {
                "median_abs_z_minus_8": float(np.median(np.abs(zp_all - 8.0))),
                "p90_abs_z_minus_8": float(np.quantile(np.abs(zp_all - 8.0), 0.9)),
                "max_abs_z_minus_8": float(np.max(np.abs(zp_all - 8.0))),
                "median_abs_const_in_steps": float(np.median(np.abs(const_all))),
            },
            "blocks": nblocks,
            "centered": bool(args.center),
            "center_s2": bool(args.center_s2),
            "stride": args.stride,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
