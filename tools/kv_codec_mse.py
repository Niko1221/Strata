#!/usr/bin/env python
"""The codec decision measurement of the centering mini-project (bench/results/2026-11-10-centering/).

Reads the real cells the engine dumped (STRATA_KV_Q4_DUMP: the rotated K/V rows of the batched
prompt path) and the per-channel moments of the S2 probe (the per-channel means before the
rotation), and computes for every codec variant:

  * the block MSE (the quantity the parity harness reports in dB) against the true values;
  * the per-block resolution u = sigma(block) / effective step, in steps per sigma - the "how well
    does the codebook use the value range" measure: with a fixed bias in the block the spread (and
    so sigma) grows while the step is set by the maximum, and u falls.

The variants:
  uniform+max   the q4_0 codec: value = (code - 8) * d, d = max|x| / 87            (affine)
  uniform+LS    the same grid, but d fitted once per block to the block's own values   (affine)
  lloyd16+max   Max-Lloyd's 16 levels, d = max|x| / 87                              (not affine)
  lloyd16+LS    Max-Lloyd's 16 levels, d fitted once per block                        (not affine)
  lloyd8+...    the 8-level ladder (V side of q4lm8v)

With --center the measured per-channel offset is subtracted before the quantization; in the
rotated domain that is the rotated offset H*m (the rotation is linear, so H(k) - H(m) = H(k-m)).

TWO CENTERING MODES, AND THEY DO NOT AGREE (read this before quoting a number):

  --center      the offsets measured from the dump itself: per layer, per KV head, per channel,
                K and V separate.  THIS is the table the engine uses (tools/kv_center_table.py
                builds it from the same dump), so this mode is the one that describes the
                engine's correction.
  --center-s2   the S2 probe's offsets: the probe measures K only and merges the two KV heads,
                so this mode subtracts the key offsets from the value rows as well.  It is the
                naive variant and it does NOT describe the engine.

Measured on the full dump of 2026-11-10: --center gives -0.17 dB on the key side and -0.36 dB on
the value side (tools/kv_center_attrib.py for the per-layer, per-side breakdown), and
--center-s2 gives about -0.07 dB combined (it helps the keys and hurts the values).  The
campaign's C1 table reported -1.9 dB for the centering; **neither mode reproduces that column**,
so its provenance is unknown and the number is withdrawn.  Quote the mode with the number.
"""
import argparse
import math
import struct

import numpy as np

KQ4 = 32
L16 = np.array([-87, -66, -52, -40, -30, -21, -12, -4, 4, 12, 21, 30, 40, 52, 66, 87], dtype=np.float64)
L8 = np.array([-69, -43, -24, -8, 8, 24, 43, 69], dtype=np.float64)
L16_MAX = 87.0
L8_MAX = 69.0
# The engine's level boundaries (the midpoints of consecutive levels; the engine picks the higher
# level at a boundary, which searchsorted with side="right" reproduces).
B16 = np.array([-76.5, -59.0, -46.0, -35.0, -25.5, -16.5, -8.0, 0.0, 8.0, 16.5, 25.5, 35.0, 46.0, 59.0, 76.5])
B8 = np.array([-56.0, -33.5, -16.0, 0.0, 16.0, 33.5, 56.0])


def fwht(x):
    """The same Sylvester construction the engine uses, scale 1/16 (orthonormal)."""
    a = np.array(x, dtype=np.float64, copy=True)
    n = a.shape[-1]
    s = 1
    while s < n:
        idx = np.arange(n)
        left = idx[(idx // s) % 2 == 0]
        right = left + s
        tmp = a[..., left].copy()
        a[..., left] = tmp + a[..., right]
        a[..., right] = tmp - a[..., right]
        s *= 2
    return a / 16.0


def read_s2_means(path):
    """{layer: (256,) means before the rotation} (the S2 probe merges the two KV heads)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.split()
            if len(p) < 14:
                continue
            layer, ch, mean_pre = int(p[0]), int(p[1]), float(p[4])
            v = out.setdefault(layer, np.zeros(256, dtype=np.float64))
            v[ch] = mean_pre
    return out


def read_dump(path, max_records=None, want_layers=None):
    """The dumped records: (layer, p0, T, nkv, hd, K[T, nkv, hd], V[T, nkv, hd])."""
    recs = []
    hdr = struct.Struct("<iiqiiii")
    with open(path, "rb") as f:
        while True:
            head = f.read(hdr.size)
            if len(head) < hdr.size:
                break
            magic, layer, p0, T, nkv, hd, _pad = hdr.unpack(head)
            if magic != 0x51345042:
                raise SystemExit(f"bad magic {magic:#x} at record {len(recs)}")
            n = T * nkv * hd
            k = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(T, nkv, hd).astype(np.float64)
            v = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(T, nkv, hd).astype(np.float64)
            if want_layers is None or layer in want_layers:
                recs.append((layer, p0, T, nkv, hd, k, v))
            if max_records and len(recs) >= max_records:
                break
    return recs


_SIGN_CACHE = {}


def _sign_vector(n, seed):
    """The fixed random sign vector of one RHT draw (the same signs for every cell: a rotation,
    not a per-cell randomization)."""
    key = (n, seed)
    if key not in _SIGN_CACHE:
        rng = np.random.default_rng(1000 + seed)
        _SIGN_CACHE[key] = rng.integers(0, 2, size=n) * 2 - 1
    return _SIGN_CACHE[key]


def rht(x, seed):
    """The randomized Hadamard transform H*D of the literature: a fixed random sign vector per
    coordinate, then the orthonormal Hadamard.  The rotation is orthogonal, so the error energy
    is the same in the transformed and the original domain; the MSE below is measured there."""
    return fwht(x * _sign_vector(x.shape[-1], seed))


def quantize_uniform(x, fit_scale):
    """ggml's q4_0 grid, mirrored from the engine's q4_group (kv_q4.cu:76):

        mval = the value of largest |x| (ties to the larger value), d = mval / -8,
        q = clamp(trunc(x/d + 8.5), 0, 15), value = (q - 8) * d.

    fit_scale: False -> the engine's rule; True -> one least-squares pass over the block's own
    values against the uniform code values (the candidate of catalog line A8). The sign convention
    of d is kept (the engine's d carries the sign of mval)."""
    amax = np.max(np.abs(x))
    idx = int(np.argmax(np.abs(x)))
    mval = float(x[idx])
    if amax == 0.0:
        return np.zeros_like(x), 0.0
    d = mval / -8.0
    idd = 1.0 / d
    q = np.trunc(x * idd + 8.5)
    code = np.clip(q, 0.0, 15.0)
    if fit_scale:
        c = code - 8.0
        den = float(np.dot(c, c))
        if den > 0:
            d_ls = float(np.dot(c, x) / den)
            if d_ls != 0.0 and (d_ls < 0) == (d < 0):
                d = d_ls
                code = np.clip(np.trunc(x * (1.0 / d) + 8.5), 0.0, 15.0)
    recon = (code - 8.0) * d
    return recon, d


def quantize_lloyd(x, levels, fit_scale):
    """Max-Lloyd's table, mirrored from the engine's q4l_group (kv_q4.cu:137): d = max|x| / max|level|
    (unsigned), the level is the nearest one (the engine's boundary rule), value = level * d. With
    fit_scale: one least-squares pass over the block's own values against these very levels."""
    levels = np.asarray(levels, dtype=np.float64)
    lmax = float(np.max(np.abs(levels)))
    amax = float(np.max(np.abs(x)))
    d = amax / lmax if amax > 0 else 0.0
    if d == 0.0:
        return np.zeros_like(x), 0.0
    bnd = B8 if len(levels) == 8 else B16
    y = x / d
    idx = np.searchsorted(bnd, y, side="right")
    if fit_scale:
        l = levels[idx]
        den = float(np.dot(l, l))
        if den > 0:
            d_ls = float(np.dot(l, x) / den)
            if d_ls > 0:
                d = d_ls
                y = x / d
                idx = np.searchsorted(bnd, y, side="right")
    recon = levels[idx] * d
    return recon, d


def blocks(arr, nblk):
    """(rows, nkv, hd) -> (rows*nkv*nblk, KQ4) blocks along the head dimension."""
    rows, nkv, hd = arr.shape
    b = arr.reshape(rows * nkv, nblk, KQ4)
    return b.reshape(-1, KQ4)


def evaluate(name, krec, vrec, kvar, vvar, centered, offsets_rot):
    """One variant: the MSE over K and V, and the per-block resolution u."""
    out = {}
    for side, arr, var in (("K", krec, kvar), ("V", vrec, vvar)):
        if centered:
            layer = var.pop("_layer")
            arr = arr - offsets_rot[layer][None, None, :]
        bl = blocks(arr, 8)
        err = np.empty(bl.shape[0])
        u = np.empty(bl.shape[0])
        step = np.empty(bl.shape[0])
        for i in range(bl.shape[0]):
            x = bl[i]
            if var["kind"] == "uniform":
                recon, d = quantize_uniform(x, var["fit"])
            else:
                recon, d = quantize_lloyd(x, var["levels"], var["fit"])
            err[i] = float(np.mean((recon - x) ** 2))
            # the effective step: the reconstruction range over the levels the block actually used
            used = np.unique(np.rint(recon / d) if d > 0 else recon)
            step[i] = (recon.max() - recon.min()) / max(len(used) - 1, 1) if d > 0 else 0.0
            u[i] = float(x.std()) / step[i] if step[i] > 0 else 0.0
        out[side] = (float(err.mean()), float(np.median(u)), float(np.median(np.abs(err))))
    mse = 0.5 * (out["K"][0] + out["V"][0])
    umed = 0.5 * (out["K"][1] + out["V"][1])
    return name, mse, umed, out["K"][0], out["V"][0], out["K"][1], out["V"][1]


def _offset_for(args, offs, offsets_rot, layer, nkv, hd, rotated_domain, tag):
    """The offset to subtract, shape (nkv, hd).

    --center: the per-head per-channel mean measured from the dump itself, per side (K and V have
    their own table - the S2 probe measures K only and merges the heads).  The dump holds rotated
    cells, so the offset is a rotated one; in the unrotated domain it is un-rotated back.
    --center-s2: the S2 file's offset (K only, both heads merged) - the naive variant."""
    if args.center:
        out = np.zeros((nkv, hd))
        for h in range(nkv):
            v = offs.get((tag, layer, h))
            if v is None:
                continue
            out[h] = v if rotated_domain else fwht(v)
        return out
    v = offsets_rot.get(layer)
    if v is None:
        return np.zeros((nkv, hd))
    v = v if rotated_domain else fwht(v)
    return np.repeat(v[None, :], nkv, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers")
    ap.add_argument("--max-cells", type=int, default=None, help="only N cells per layer")
    ap.add_argument("--center", action="store_true",
                    help="subtract the per-head per-channel offset measured from the dump itself")
    ap.add_argument("--center-s2", action="store_true",
                    help="subtract the S2 file's offset (the probe merges the two KV heads)")
    args = ap.parse_args()

    means = read_s2_means(args.s2)
    offsets_rot = {layer: fwht(v) for layer, v in means.items()}

    recs = read_dump(args.dump)
    if args.layers:
        # The dump's layers are the model's QSA layers (3, 7, 11, ... 47), not 0..N-1: a filter on
        # range(N) matched nothing and the tool answered "no records in the dump" (deviation 17).
        # Take the first N layers in the order the dump holds them.
        wanted = []
        for r in recs:
            if r[0] not in wanted:
                wanted.append(r[0])
            if len(wanted) == args.layers:
                break
        recs = [r for r in recs if r[0] in set(wanted)]
        print(f"layers: {sorted(wanted)}")
    if not recs:
        raise SystemExit("no records in the dump")
    print(f"dump: {len(recs)} records, layers {sorted(set(r[0] for r in recs))}")

    # The offsets from the dump itself: per layer, per KV head, per channel - the mean over the dumped
    # cells.  The S2 probe merges the two heads AND measures only K, so its numbers are neither a
    # per-head nor a V offset: K and V get their own table here.
    offs = {}
    cnts = {}
    for layer, p0, T, nkv, hd, karr, varr in recs:
        if args.max_cells:
            karr, varr = karr[:args.max_cells], varr[:args.max_cells]
            T = args.max_cells
        for tag, arr in (("K", karr), ("V", varr)):
            for h in range(arr.shape[1]):
                s = offs.setdefault((tag, layer, h), np.zeros(hd))
                s += arr[:, h, :].sum(axis=0)
                c = cnts.setdefault((tag, layer, h), 0)
                cnts[(tag, layer, h)] = c + T
    for key in offs:
        offs[key] = offs[key] / max(cnts[key], 1)
    if offs:
        for tag in ("K", "V"):
            med = []
            for (t, layer, h), v in offs.items():
                if t != tag:
                    continue
                sd = float(np.sqrt(np.mean((v - v.mean()) ** 2))) or 1.0
                med.append(float(np.median(np.abs(v))) / sd)
            if med:
                print(f"Offset aus dem Dump ({tag}): Median |Offset| / sd der Kanaloffsets = {np.median(med):.4f}")
        by_layer = {}
        for (tag, layer, h), v in offs.items():
            if tag != "K":
                continue
            by_layer.setdefault(layer, []).append(v)
        agree = []
        for layer, vs in by_layer.items():
            if len(vs) == 2:
                agree.append(float(np.mean(np.abs(0.5 * (vs[0] + vs[1])))))
        if agree:
            print(f"           K nach dem Kopf-Mittel (wie die S2-Sonde): Mittel |Rest| {np.mean(agree):.4f}")

    # one variant set over all records
    variants = [
        ("uniform+max (q4_0, der Anker)", dict(kind="uniform", fit=False), dict(kind="uniform", fit=False)),
        ("uniform+LS  (A8-Kandidat, affin)", dict(kind="uniform", fit=True), dict(kind="uniform", fit=True)),
        ("lloyd16+max", dict(kind="lloyd", levels=L16, fit=False), dict(kind="lloyd", levels=L16, fit=False)),
        ("lloyd16+LS", dict(kind="lloyd", levels=L16, fit=True), dict(kind="lloyd", levels=L16, fit=True)),
        ("lloyd16K+lloyd8V (q4lm8v-Codec)", dict(kind="lloyd", levels=L16, fit=False),
         dict(kind="lloyd", levels=L8, fit=False)),
        ("lloyd16K+lloyd8V+LS", dict(kind="lloyd", levels=L16, fit=True), dict(kind="lloyd", levels=L8, fit=True)),
        ("RHT(H*D) + uniform+LS, Zug 1", dict(kind="uniform", fit=True, rht=1), dict(kind="uniform", fit=True, rht=1)),
        ("RHT(H*D) + uniform+LS, Zug 2", dict(kind="uniform", fit=True, rht=2), dict(kind="uniform", fit=True, rht=2)),
        ("RHT(H*D) + uniform+LS, Zug 3", dict(kind="uniform", fit=True, rht=3), dict(kind="uniform", fit=True, rht=3)),
    ]

    base = None
    rows = []
    for name, kv, vv in variants:
        tot_err = 0.0
        tot_n = 0
        us = []
        rs = []
        for layer, p0, T, nkv, hd, karr, varr in recs:
            if args.max_cells:
                karr, varr = karr[:args.max_cells], varr[:args.max_cells]
                T = args.max_cells
            for side, arr, var in (("K", karr, kv), ("V", varr, vv)):
                if "rht" in var:
                    # the cells in the dump are rotated (H k).  The RHT of the literature is H*S on the
                    # unrotated data, and H is its own inverse: un-rotate, subtract the offset in the
                    # unrotated domain, then H*S.  The rotation is orthogonal, so the error energy is
                    # the same in both domains and the MSE below is comparable.
                    sign = _sign_vector(hd, var["rht"]).astype(np.float64)
                    arr = fwht(arr)
                    if args.center or args.center_s2:
                        arr = arr - _offset_for(args, offs, offsets_rot, layer, arr.shape[1], hd, False, side)
                    arr = fwht(arr * sign[None, None, :])
                else:
                    if args.center or args.center_s2:
                        arr = arr - _offset_for(args, offs, offsets_rot, layer, arr.shape[1], hd, True, side)
                bl = blocks(arr, 8)
                for i in range(bl.shape[0]):
                    x = bl[i]
                    if var["kind"] == "uniform":
                        recon, d = quantize_uniform(x, var["fit"])
                    else:
                        recon, d = quantize_lloyd(x, var["levels"], var["fit"])
                    tot_err += float(np.mean((recon - x) ** 2))
                    tot_n += 1
                    used = np.unique(np.rint(recon / d) if d > 0 else recon)
                    st = (recon.max() - recon.min()) / max(len(used) - 1, 1) if d > 0 else 0.0
                    sd = float(x.std())
                    if st > 0:
                        us.append(sd / st)
                    if d != 0.0 and sd > 0:
                        # the scale relative to the block's own spread: how much of the quantizer's
                        # range the signal needs.  Lower is better - and unlike sigma/step it is not
                        # scale-invariant, so it shows what a removed bias does.
                        rs.append(abs(d) / sd)
        mse = tot_err / max(tot_n, 1)
        if base is None:
            base = mse
        db = 10.0 * math.log10(mse / base) if mse > 0 and base > 0 else float("nan")
        umed = float(np.median(us)) if us else float("nan")
        rmed = float(np.median(rs)) if rs else float("nan")
        rows.append((name, mse, db, umed, rmed))
        print(f"{name:34s} MSE {mse:.6e}  {db:+6.2f} dB  sigma/Schritt {umed:6.2f}  |d|/sigma {rmed:6.3f}")

    print()
    print(f"Zentriert: {'ja' if args.center else 'nein'}   Werte je Block: {KQ4}")
    print("u = sigma(Block) / effektiver Schritt (Schritte je sigma; groesser = feinere Aufloesung)")


if __name__ == "__main__":
    main()
