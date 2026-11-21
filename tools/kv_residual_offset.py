#!/usr/bin/env python
"""Der gespeicherte Offset: was bringt der Restfehler-Offset bei **festen** Codes und **fester** Skala?

Registrierung: bench/results/2026-11-13-kv-offset-tail/README.md (H0-a/b/c/d).

Fuer jeden Block (32 Werte entlang der Kopfdimension, wie im Kernel) wird die Rekonstruktion
gebaut, dann der mittlere Restfehler

    a = mean_i (x_i - r_i)

und die Block-MSE mit und ohne Offset gemessen.  Bei festen Codes und fester Skala ist a optimal,
und die MSE sinkt exakt um a^2 -- der Punkt des Werkzeugs ist, wie viel davon nach der
Quantisierung des Offsets, nach groeberer Granularitaet und nach der Zerlegung in einen
statischen Kanalanteil uebrig bleibt.

Varianten je Arm: kein Offset, a exakt, a in 6 Bit (1/32 Schritt, Bereich +-1 Schritt), a je 2 und
je 4 Bloecke geteilt, und der statische Kanalanteil a_c (Zelle = Layer x Seite x KV-Kopf x Blockindex).

Die dump-Reader und die Kernel-Spiegel kommen aus tools/kv_codec_mse.py und tools/kv_affine_price.py,
damit die Zahlen dieselben sind wie in docs/research/KV_FASTPATH_LITERATURE.md section 3c.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_codec_mse import L16, read_dump  # noqa: E402
from kv_affine_price import (  # noqa: E402
    blocks_of, quantize_lloyd_vec, quantize_uniform_shear, quantize_uniform_zero,
)

ARMS = (
    ("uniform+max", "uniform", False),
    ("uniform+LS", "uniform", True),
    ("lloyd16+max", "lloyd", False),
    ("lloyd16+LS", "lloyd", True),
)
METRICS = ("none", "a_exact", "q6", "q6w", "g2", "g4", "static")
BANDS = ("bulk", "top10", "top1")
SIDES = ("K", "V")
# der 6-Bit-Offset: (Name, Aufloesung in Code-Schritten, groesster darstellbarer Wert)
# q6 ist das vorregistrierte Format (1/32 Schritt, +-1); q6w ist die **explorative** weite
# Variante (1/16 Schritt, +-2), nachdem die Messung zeigte, dass 1/32 fuer die Lloyd-Arme
# saettigt (siehe README, H0-b).
QSTEPS = (("q6", 1.0 / 32.0, 31.0 / 32.0), ("q6w", 1.0 / 16.0, 31.0 / 16.0))
NBLK = 8                    # 256 Kopfdimension / 32


def recon_of(bl, kind, fit):
    if kind == "uniform":
        return quantize_uniform_zero(bl, fit)
    return quantize_lloyd_vec(bl, L16, fit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--shear", action="store_true", help="auch den Shear (joint fit) als Anker rechnen")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    recs = read_dump(args.dump)
    if not recs:
        raise SystemExit("no records")
    if args.max_records:
        recs = recs[: args.max_records]
    n_pos = max(p0 + T for (_l, p0, T, _n, _h, _k, _v) in recs)
    print(f"dump: {len(recs)} records, {n_pos} Positionen, "
          f"{sum(T * nkv * NBLK for (_l, _p, T, nkv, _h, _k, _v) in recs)} Bloecke")

    # --- Durchlauf 1: die Energie jeder Position (K und V gemeinsam), fuer die Baender
    energy = np.zeros(n_pos, dtype=np.float64)
    for _layer, p0, T, nkv, hd, karr, varr in recs:
        e = np.zeros(T, dtype=np.float64)
        for arr in (karr, varr):
            e += np.einsum("thd,thd->t", arr, arr) / float(nkv * hd)
        energy[p0 : p0 + T] += 0.5 * e
    order = np.argsort(-energy, kind="stable")
    top1 = np.zeros(n_pos, dtype=bool)
    top10 = np.zeros(n_pos, dtype=bool)
    n1 = max(1, int(round(0.01 * n_pos)))
    n10 = max(1, int(round(0.10 * n_pos)))
    top1[order[:n1]] = True
    top10[order[:n10]] = True
    thick = np.where(top1, "top1", np.where(top10, "top10", "bulk"))
    print(f"Baender: top1 = {n1} Positionen, top10 = {n10} (top1 ist in top10 enthalten)")

    # --- Durchlauf 2: alle Groessen, in einem Durchlauf
    acc = {}          # (arm, metric, band, side) -> [sum_mse, n_blocks]
    cells = {}        # (arm, side, band, layer, head, blk) -> [n, sum_e0, sum_a, sum_a2]
    a2_all = {a[0]: [] for a in ARMS}
    sat = {a[0]: {q[0]: 0 for q in QSTEPS} for a in ARMS}
    nb_arm = {a[0]: 0 for a in ARMS}
    shear_sum = [0.0, 0]

    def bump(key, val):
        s = acc.get(key)
        if s is None:
            acc[key] = [float(val), 1]
        else:
            s[0] += float(val)
            s[1] += 1

    for layer, p0, T, nkv, hd, karr, varr in recs:
        pos = np.arange(p0, p0 + T)
        bands = thick[pos]                                    # (T,)
        for side, arr in (("K", karr), ("V", varr)):
            bl = blocks_of(arr, NBLK)                         # (T*nkv*NBLK, 32)
            n = bl.shape[0]
            bpos = p0 + (np.arange(n) // (nkv * NBLK))        # Position je Block
            bband = thick[bpos]
            bhead = (np.arange(n) // NBLK) % nkv
            bblk = np.arange(n) % NBLK
            for name, kind, fit in ARMS:
                r, d = recon_of(bl, kind, fit)
                res = bl - r
                e0 = np.mean(res * res, axis=1)
                a = np.mean(res, axis=1)
                dsafe = np.where(d != 0, d, 1.0)
                t = a / dsafe
                aqs = {}
                for qname, qstep, qmax in QSTEPS:
                    q = np.clip(np.rint(t / qstep), -1.0 / qstep, qmax / qstep)
                    aqs[qname] = q * qstep * dsafe
                    sat[name][qname] += int(np.count_nonzero(np.abs(t) > qmax))

                # Granularitaet: Offset je G Bloecke (innerhalb desselben Kopfs, NBLK teilbar)
                res3 = res.reshape(-1, NBLK, 32)
                offs = {"g2": 2, "g4": 4}
                ag = {}
                for key, g in offs.items():
                    ag[key] = np.repeat(
                        res3.reshape(-1, NBLK // g, g * 32).mean(axis=2), g, axis=1
                    ).reshape(-1)
                nb_arm[name] += n

                for band in BANDS:
                    m = bband == band
                    if not np.any(m):
                        continue
                    bump((name, "none", band, side), e0[m].mean())
                    bump((name, "a_exact", band, side), np.mean((res[m] - a[m, None]) ** 2))
                    for qname in aqs:
                        bump((name, qname, band, side),
                             np.mean((res[m] - aqs[qname][m, None]) ** 2))
                    for key in ("g2", "g4"):
                        off = ag[key][m]
                        bump((name, key, band, side),
                             np.mean((res[m] - off[:, None]) ** 2))

                # Zellen des statischen Anteils
                for band in BANDS:
                    m = bband == band
                    if not np.any(m):
                        continue
                    for h in range(nkv):
                        for b in range(NBLK):
                            sel = m & (bhead == h) & (bblk == b)
                            if not np.any(sel):
                                continue
                            key = (name, side, band, layer, h, b)
                            c = cells.get(key)
                            nn = int(np.count_nonzero(sel))
                            e = e0[sel]
                            aa = a[sel]
                            if c is None:
                                cells[key] = [nn, float(e.sum()), float(aa.sum()),
                                              float((aa * aa).sum())]
                            else:
                                c[0] += nn
                                c[1] += float(e.sum())
                                c[2] += float(aa.sum())
                                c[3] += float((aa * aa).sum())

                a2_all[name].append((a * a).astype(np.float32))

            if args.shear:
                r, s, _zp, _c = quantize_uniform_shear(bl)
                shear_sum[0] += float(np.mean((r - bl) ** 2))
                shear_sum[1] += 1

    # der statische Anteil: MSE je Zelle mit a_c = sum_a / n
    for (name, side, band, layer, h, b), (nn, se0, sa, _sa2) in cells.items():
        ac = sa / nn
        mse = (se0 - 2.0 * ac * sa + nn * ac * ac) / nn
        bump((name, "static", band, side), mse)

    def mse(arm, metric, band, side=None):
        if side is None:
            vals = [acc[(arm, metric, band, s)] for s in SIDES if (arm, metric, band, s) in acc]
            tot = sum(v[0] for v in vals)
            cnt = sum(v[1] for v in vals)
            return tot / cnt if cnt else float("nan")
        v = acc.get((arm, metric, band, side))
        return v[0] / v[1] if v else float("nan")

    out = {"positions": n_pos, "bands": {"top1": n1, "top10": n10}, "arms": {}}
    print()
    print("Block-MSE je Arm (beide Seiten, alle Positionen)")
    print(f"{'Arm':14s} {'kein':>13s} {'a exakt':>13s} {'6 Bit':>13s} {'6 Bit weit':>13s} "
          f"{'je 2 Bl.':>13s} {'je 4 Bl.':>13s} {'statisch':>13s}")
    for name, _k, _f in ARMS:
        row = [mse(name, m, "bulk") for m in METRICS]
        out["arms"][name] = {m: row[i] for i, m in enumerate(METRICS)}
        print(f"{name:14s} " + " ".join(f"{v:13.6e}" for v in row))

    # der Shear als Anker (ueber alle Bloecke)
    if args.shear and shear_sum[1]:
        ms = shear_sum[0] / shear_sum[1]
        num = sum(acc[("uniform+LS", "none", b, s)][0] for b in BANDS for s in SIDES
                  if ("uniform+LS", "none", b, s) in acc)
        den = sum(acc[("uniform+LS", "none", b, s)][1] for b in BANDS for s in SIDES
                  if ("uniform+LS", "none", b, s) in acc)
        base = num / den if den else float("nan")
        print(f"\nShear (joint fit, uniform+LS+z): {ms:.6e}  "
              f"({10.0 * math.log10(ms / base):+.3f} dB gegen uniform+LS)")
        out["shear"] = {"mse": ms, "db_vs_uniformLS": 10.0 * math.log10(ms / base)}

    print()
    print("Der Gewinn gegen 'kein Offset', in % des exakten Gewinns (Sättigung: Anteil der Offsets")
    print("ausserhalb des darstellbaren Bereichs):")
    print(f"{'Arm':14s} {'6 Bit':>8s} {'6 Bit weit':>11s} {'je 2 Bl.':>9s} {'je 4 Bl.':>9s} "
          f"{'statisch':>9s} {'Sätt. eng':>10s} {'Sätt. weit':>11s}")
    for name, _k, _f in ARMS:
        e0 = mse(name, "none", "bulk")
        ex = e0 - mse(name, "a_exact", "bulk")
        share = {}
        for m in ("q6", "q6w", "g2", "g4", "static"):
            share[m] = (e0 - mse(name, m, "bulk")) / ex if ex > 0 else float("nan")
        sat_pct = {q: 100.0 * sat[name][q] / max(1, nb_arm[name]) for q, _s, _m in QSTEPS}
        print(f"{name:14s} " + " ".join(f"{100.0 * share[m]:8.1f}" for m in ("q6", "q6w", "g2", "g4", "static"))
              + f" {sat_pct['q6']:9.3f}% {sat_pct['q6w']:10.3f}%")
        out["arms"][name]["share_of_exact"] = share
        out["arms"][name]["saturation_pct"] = sat_pct

    print()
    print("Schwere Positionen: a^2 / Block-MSE im Band (1,00 = der Offset erklaert die MSE vollstaendig)")
    print(f"{'Arm':14s} {'Seite':5s} {'bulk':>9s} {'top10':>9s} {'top1':>9s} {'MSE top1/bulk':>14s}")
    for name, _k, _f in ARMS:
        for side in SIDES:
            e0b = mse(name, "none", "bulk", side)
            e0t = mse(name, "none", "top1", side)
            row = []
            for band in BANDS:
                e0 = mse(name, "none", band, side)
                ea = mse(name, "a_exact", band, side)
                row.append((e0 - ea) / e0 if e0 > 0 else float("nan"))
            print(f"{name:14s} {side:5s} " + " ".join(f"{v:9.4f}" for v in row)
                  + f" {e0t / e0b:14.3f}")
            out["arms"][name].setdefault("a2_share", {})[side] = {
                b: row[i] for i, b in enumerate(BANDS)}

    # Konzentration: welcher Anteil der a^2-Energie sitzt in den obersten x % der Bloecke
    conc = {}
    print()
    print("Konzentration des Offsets: Anteil der a^2-Energie in den obersten x % der Bloecke")
    print(f"{'Arm':14s} {'0,1 %':>8s} {'1 %':>8s} {'10 %':>8s} {'50 %':>8s} {'Blöcke':>10s}")
    for name, _k, _f in ARMS:
        arr = np.concatenate(a2_all[name]) if a2_all[name] else np.zeros(1)
        tot = float(arr.sum())
        srt = np.sort(arr)[::-1]
        c = np.cumsum(srt)
        row = {}
        for frac in (0.001, 0.01, 0.10, 0.50):
            k = max(1, int(round(frac * len(arr))))
            row[f"{frac:.2%}"] = float(c[k - 1] / tot) if tot > 0 else float("nan")
        print(f"{name:14s} " + " ".join(f"{100.0 * row[k]:7.1f}%" for k in (f"{f:.2%}" for f in (0.001, 0.01, 0.10, 0.50)))
              + f" {len(arr):10d}")
        conc[name] = row
    out["concentration"] = conc

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
