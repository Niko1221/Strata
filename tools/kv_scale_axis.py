#!/usr/bin/env python
"""A.26: die Skalierungsachse — eine Skala je Token-Gruppe (unsere `q4u32`) gegen eine Skala je Kanal (KIVI).

Auf den **echten Zellen** des Motors (dem Dump des Centering-Projekts), im rotierten Raum, in dem unser
Format lebt.  Drei Arme:

  q4u32       je Token, je 32 Werte entlang head_dim (unser Format: 4 Bit je Wert plus 0,5 Bit Skala)
  per-kanal   je (Layer, Kopf, Kanal) über alle Token (die KIVI-Achse; die Tabelle ist token-unabhängig
              und liegt in der Modelldatei: 1.024 fp16 je Layer und Seite)
  hybrid      K per-Kanal, V je Token-Gruppe (KIVIs Rezept)

Die Metrik ist die des Shear-Arms (`tools/kv_affine_price.py`): der mittlere quadratische Fehler je Wert,
in dB gegen den Anker.  Die Quantisierung ist die des Engines (ein Schritt je Block, gefittet statt vom
Blockmaximum genommen — `q4u32`), gespiegelt aus `tools/kv_codec_mse.py`.

Zusätzlich berichtet das Werkzeug, **wie viele Koordinaten** im Anker die Skala bestimmen: das ist die
„Outlier-Kanal"-Zahl, an der man sieht, ob die KIVI-Achse überhaupt etwas zu holen hat.

  python tools/kv_scale_axis.py --dump <cells-cwe4k.bin> [--max-records N] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_codec_mse import read_dump  # noqa: E402

GROUP = 32          # our block: 32 values along head_dim, one fitted step per block (kv_q4.cu, the q4u32 path)


def quantize_blocks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The engine's 4-bit rule on blocks of 32 values: one fitted step per block, the grid -8..7.

    `x` is (N, 32).  Returns the reconstruction and the step per block (the engine's `d`).
    """
    amax = np.max(np.abs(x), axis=1)
    idx = np.argmax(np.abs(x), axis=1)
    mval = x[np.arange(x.shape[0]), idx]
    d = mval / -8.0
    d = np.where(amax == 0.0, 0.0, d)
    inv = np.where(d != 0.0, 1.0 / d, 0.0)
    code = np.clip(np.trunc(x * inv[:, None] + 8.5), 0.0, 15.0)
    c = code - 8.0
    den = np.sum(c * c, axis=1)
    num = np.sum(c * x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ls = np.where(den > 0, num / den, d)
    ok = (d_ls != 0.0) & (np.sign(d_ls) == np.sign(d))
    d = np.where(ok, d_ls, d)
    inv = np.where(d != 0.0, 1.0 / d, 0.0)
    code = np.clip(np.trunc(x * inv[:, None] + 8.5), 0.0, 15.0)
    return (code - 8.0) * d[:, None], d


def quantize_per_channel(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same 4-bit rule, but the step comes from **all tokens of the coordinate** (T, C) -> one step per column.

    The step is fitted the same way (a least-squares pass over the values it quantizes), so the arms differ
    in the *axis* of the scale and nothing else.
    """
    t = x.shape[0]
    flat = x.reshape(t, -1)                                  # (T, C)
    amax = np.max(np.abs(flat), axis=0)
    idx = np.argmax(np.abs(flat), axis=0)
    mval = flat[idx, np.arange(flat.shape[1])]
    d = mval / -8.0
    d = np.where(amax == 0.0, 0.0, d)
    inv = np.where(d != 0.0, 1.0 / d, 0.0)
    code = np.clip(np.trunc(flat * inv[None, :] + 8.5), 0.0, 15.0)
    c = code - 8.0
    den = np.sum(c * c, axis=0)
    num = np.sum(c * flat, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_ls = np.where(den > 0, num / den, d)
    ok = (d_ls != 0.0) & (np.sign(d_ls) == np.sign(d))
    d = np.where(ok, d_ls, d)
    inv = np.where(d != 0.0, 1.0 / d, 0.0)
    code = np.clip(np.trunc(flat * inv[None, :] + 8.5), 0.0, 15.0)
    return ((code - 8.0) * d[None, :]).reshape(x.shape), d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None, help="only this layer")
    ap.add_argument("--groups", default="32", help="the block sizes to compare (comma separated: 16,32,64)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    groups = [int(g) for g in str(args.groups).split(",") if g.strip()]
    want = [args.layers] if args.layers is not None else None
    recs = read_dump(args.dump, max_records=args.max_records, want_layers=want)
    if not recs:
        print("keine Records im Dump")
        return 1

    # A.27: the group size of the block (the anchor's group against the others), on the same cells.
    group_err = {g: {"K": 0.0, "V": 0.0} for g in groups}
    group_values = {"K": 0, "V": 0}

    err = {("q4u32", "K"): 0.0, ("q4u32", "V"): 0.0,
           ("per-kanal", "K"): 0.0, ("per-kanal", "V"): 0.0,
           ("hybrid", "K"): 0.0, ("hybrid", "V"): 0.0}
    per_layer = {}          # layer -> {arm|side: squared error}, the same numbers layer by layer
    values = {"K": 0, "V": 0}
    scale_owner = {}          # how often a coordinate set the scale in the anchor (per layer)
    tokens = 0
    for layer, _p0, T, nkv, hd, k, v in recs:
        tokens += T
        lay = per_layer.setdefault(layer, {name: 0.0 for name in err})
        for side, x in (("K", k), ("V", v)):
            values[side] += x.size
            # the anchor: groups of 32 along head_dim, per token
            g = x.reshape(-1, hd // GROUP, GROUP)
            flat = g.reshape(-1, GROUP)
            recon, _d = quantize_blocks(flat)
            e_q4 = float(np.sum((flat - recon) ** 2))
            err[("q4u32", side)] += e_q4
            lay[("q4u32", side)] += e_q4
            # the group sizes of A.27: the same rule, a different block
            for size in groups:
                if hd % size:
                    continue
                gg = x.reshape(-1, hd // size, size)
                ff = gg.reshape(-1, size)
                rc, _dc = quantize_blocks(ff)
                group_err[size][side] += float(np.sum((ff - rc) ** 2))
            # who sets the scale in the anchor: the largest |value| of the block, as a coordinate along head_dim
            within = np.argmax(np.abs(g), axis=2)                    # (T*nkv, groups)
            coord = (np.arange(within.shape[1])[None, :] * GROUP + within).ravel()
            counts = np.bincount(coord, minlength=hd)
            acc = scale_owner.setdefault(layer, np.zeros(hd, dtype=np.int64))
            acc += counts[:acc.shape[0]]
            # the per-channel arm: one step per coordinate over all tokens
            recon_c, _dc = quantize_per_channel(x)
            e_pc = float(np.sum((x - recon_c) ** 2))
            err[("per-kanal", side)] += e_pc
            lay[("per-kanal", side)] += e_pc
            # the hybrid: K per channel, V per token group
            e_hy = e_pc if side == "K" else e_q4
            err[("hybrid", side)] += e_hy
            lay[("hybrid", side)] += e_hy

    group_values = dict(values)

    mse = {name: err[name] / values[name[1]] for name in err}
    base = mse[("q4u32", "K")], mse[("q4u32", "V")]

    def db(arm: str, side: str) -> float:
        ref = base[0] if side == "K" else base[1]
        return 10.0 * math.log10(mse[(arm, side)] / ref) if mse[(arm, side)] > 0 and ref > 0 else float("nan")

    print(f"A.26: die Skalierungsachse   Dump {os.path.basename(args.dump)}   "
          f"{len(recs)} Records, {tokens} Token, Werte K {values['K']:,} / V {values['V']:,}")
    print(f"{'Arm':12s} {'MSE K':>13s} {'dB K':>8s} {'MSE V':>13s} {'dB V':>8s}")
    for arm in ("q4u32", "per-kanal", "hybrid"):
        print(f"{arm:12s} {mse[(arm, 'K')]:13.6e} {db(arm, 'K'):+8.3f} {mse[(arm, 'V')]:13.6e} {db(arm, 'V'):+8.3f}")
    print("\ndB gegen den Anker (negativ = besser als der Anker).  Kriterium (K): >= -1,0 dB.")
    bits_q4u32 = 4.0 + 16.0 / GROUP                      # 4 bit je Wert + ein fp16-Schritt je 32 Werte
    bits_per_kanal = 4.0 + 16.0 / max(tokens, 1)         # 4 bit je Wert + eine fp16-Tabelle je Kanal (token-unabhaengig)
    print(f"Bit je Wert: q4u32 {bits_q4u32:.3f} (4 + die Skala je 32 Werte), per-Kanal {bits_per_kanal:.4f} "
          f"(4 + eine fp16-Tabelle je Kanal, {tokens} Token) — die Tabelle liegt in der Modelldatei.")

    print("\nje Layer (dB des per-Kanal-Arms gegen den Anker, K und V)")
    print(f"{'Layer':>6s} {'dB K':>8s} {'dB V':>8s}   {'Werte':>14s}")
    for layer in sorted(per_layer):
        lay = per_layer[layer]
        n = lay.get(("q4u32", "K"), 0.0)
        kv = lay.get(("q4u32", "V"), 0.0)
        if n <= 0 or kv <= 0:
            continue
        dk = 10.0 * math.log10(lay[("per-kanal", "K")] / n) if lay[("per-kanal", "K")] > 0 else float("nan")
        dv = 10.0 * math.log10(lay[("per-kanal", "V")] / kv) if lay[("per-kanal", "V")] > 0 else float("nan")
        print(f"{layer:6d} {dk:+8.3f} {dv:+8.3f}   {int(n):>14,d}")

    if len(groups) > 1:
        print("\nA.27: die Gruppengröße des Blocks (dB gegen die 32er-Gruppe, dem Anker)")
        print(f"{'Gruppe':>7s} {'bits/Wert':>10s} {'dB K':>8s} {'dB V':>8s}")
        base_group = 32 if 32 in groups else groups[0]
        for size in sorted(groups):
            bits = 4.0 + 16.0 / size
            dbk = dBV = float("nan")
            if size == base_group:
                dbk = dBV = 0.0
            elif base_group in group_err:
                bk = group_err[base_group]["K"] / max(group_values["K"], 1)
                sk = group_err[size]["K"] / max(group_values["K"], 1)
                bv = group_err[base_group]["V"] / max(group_values["V"], 1)
                sv = group_err[size]["V"] / max(group_values["V"], 1)
                dbk = 10.0 * math.log10(sk / bk) if sk > 0 and bk > 0 else float("nan")
                dBV = 10.0 * math.log10(sv / bv) if sv > 0 and bv > 0 else float("nan")
            print(f"{size:7d} {bits:10.3f} {dbk:+8.3f} {dBV:+8.3f}")
        print("  (eine 64er-Gruppe überspannt zwei Kerngruppen: der Kern wendet die Skala je 32 Dimensionen an.)")

    print("\nDie Skala im Anker: wer setzt sie (Anteil der Blöcke je Koordinate)")
    for layer, counts in sorted(scale_owner.items()):
        total = counts.sum()
        order = np.argsort(counts)[::-1]
        top = counts[order[:5]]
        share = [f"{int(c / total * 1000) / 10:.1f}%" for c in top]
        print(f"  Layer {layer:2d}: {len(counts)} Koordinaten, die besten fünf {share} "
              f"(zusammen {int(top.sum() / total * 1000) / 10:.1f} % der Blöcke)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"records": len(recs), "tokens": tokens, "mse": {f"{a}|{s}": mse[(a, s)] for a, s in mse},
                       "db": {f"{a}|{s}": db(a, s) for a in ("q4u32", "per-kanal", "hybrid") for s in ("K", "V")},
                       "scale_owner": {str(l): counts.tolist() for l, counts in scale_owner.items()}},
                      f, indent=1)
        print(f"\ngeschrieben: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
