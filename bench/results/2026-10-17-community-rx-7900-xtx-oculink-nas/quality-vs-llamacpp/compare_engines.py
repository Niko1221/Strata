#!/usr/bin/env python3
"""Vergleich Strata gegen llama.cpp auf demselben Textsatz und derselben GGUF-Datei — 17.10.2026.

Liest zwei Läufe im STRATA_LOGPOS-Format (Strata: tools/quant_quality.py run, llama.cpp: qq_llamacpp.py) und gibt
je Abschnitt und gesamt aus:
  top1      gleiches wahrscheinlichstes Token
  KL        KL(A || B) über A's 64 wahrscheinlichste Token, beide auf diese Menge renormiert (wie quant_quality.py);
            ein Token, das B nicht listet, bekommt B's 64. Wert — beide Richtungen
  PPL       Perplexität des Textes, nur über Positionen, an denen llama.cpp den exakten Wert des Ziel-Tokens kennt
            (Ziel unter seinen 4000 wahrscheinlichsten); Strata über dieselben Positionen
  python3 qq_vergleich.py llamacpp.tsv strata.tsv [--json aus.json]
"""
import json, math, sys

TOPK = 64


def lese(pfad):
    zeilen = open(pfad).read().splitlines()
    segs = json.load(open(pfad + ".seg.json"))["segments"]
    aus = {}
    for s in segs:
        d = {}
        for z in zeilen[s["first_line"]:s["end_line"]]:
            f = z.split("\t")
            top = [(int(x), float(y)) for x, y in (t.split(":") for t in f[8:8 + TOPK])]
            exakt = f[7] != "nan"
            d[int(f[0])] = {"ziel": int(f[1]), "lz": float(f[2]), "am": int(f[3]), "top": top, "exakt": exakt}
        aus[s["name"]] = d
    return aus


def kl(a, b):
    db = dict(b["top"]); boden = min(v for _, v in b["top"])
    pa = [math.exp(v) for _, v in a["top"]]; pb = [math.exp(db.get(i, boden)) for i, _ in a["top"]]
    sa, sb = sum(pa), sum(pb)
    return sum((x / sa) * math.log((x / sa) / (y / sb)) for x, y in zip(pa, pb) if x > 0)


def vergleiche(L, S):
    erg, alle = {}, {"n": 0, "top1": 0, "kl_ls": [], "kl_sl": [], "nll_l": 0.0, "nll_s": 0.0, "n_ppl": 0}
    for name in L:
        l, s = L[name], S.get(name, {})
        pos = sorted(set(l) & set(s))
        n = top1 = n_ppl = 0; kls, kls2 = [], []; nll_l = nll_s = 0.0
        for q in pos:
            a, b = l[q], s[q]
            if a["ziel"] != b["ziel"]:
                raise SystemExit(f"{name} pos {q}: Ziel {a['ziel']} != {b['ziel']} — nicht derselbe Text")
            n += 1; top1 += a["am"] == b["am"]
            kls.append(kl(a, b)); kls2.append(kl(b, a))
            if a["exakt"]:
                n_ppl += 1; nll_l -= a["lz"]; nll_s -= b["lz"]
        erg[name] = {"positionen": n, "top1": top1 / n, "kl_llamacpp_strata": sum(kls) / n, "kl_strata_llamacpp": sum(kls2) / n,
                     "ppl_positionen": n_ppl, "ppl_llamacpp": math.exp(nll_l / n_ppl), "ppl_strata": math.exp(nll_s / n_ppl)}
        alle["n"] += n; alle["top1"] += top1; alle["kl_ls"] += kls; alle["kl_sl"] += kls2
        alle["nll_l"] += nll_l; alle["nll_s"] += nll_s; alle["n_ppl"] += n_ppl
    n = alle["n"]; k1 = sorted(alle["kl_ls"])
    erg["gesamt"] = {"positionen": n, "top1": alle["top1"] / n, "kl_llamacpp_strata": sum(alle["kl_ls"]) / n,
                     "kl_strata_llamacpp": sum(alle["kl_sl"]) / n, "kl_p99": k1[int(0.99 * (len(k1) - 1))],
                     "kl_median": k1[len(k1) // 2],
                     "ppl_positionen": alle["n_ppl"], "ppl_llamacpp": math.exp(alle["nll_l"] / alle["n_ppl"]),
                     "ppl_strata": math.exp(alle["nll_s"] / alle["n_ppl"])}
    return erg


if __name__ == "__main__":
    erg = vergleiche(lese(sys.argv[1]), lese(sys.argv[2]))
    print(f"{'Abschnitt':13} {'Pos':>5} {'Top-1':>7} {'KL L||S':>8} {'KL S||L':>8} {'PPL-Pos':>7} {'PPL llama':>9} {'PPL Strata':>10}")
    for name, v in erg.items():
        print(f"{name:13} {v['positionen']:5d} {100 * v['top1']:6.2f}% {v['kl_llamacpp_strata']:8.4f} {v['kl_strata_llamacpp']:8.4f} "
              f"{v['ppl_positionen']:7d} {v['ppl_llamacpp']:9.3f} {v['ppl_strata']:10.3f}")
    g = erg["gesamt"]; print(f"KL Median {g['kl_median']:.4f}, p99 {g['kl_p99']:.4f}")
    if "--json" in sys.argv:
        json.dump(erg, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
