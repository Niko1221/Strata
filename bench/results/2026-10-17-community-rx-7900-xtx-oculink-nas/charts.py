#!/usr/bin/env python3
"""Charts für das Repo ugreen-idx6011-egpu-llm und den Strata-PR — helles Workbench-Design (17.10.2026).

Farben aus dem Design System Workbench (tokens.json, Thema hell): Grund #efeae1, Schrift #2a2724,
gedämpft #6a6258, schwach #8b8377 (nur Linien), Linie #d3cabb; Reihen diagramm-1 Bernstein #975b34,
diagramm-2 Stahl #456579. Schriften Atkinson Hyperlegible Next und JetBrains Mono.

  python charts.py tiefe-20261017.jsonl ausgabe/
"""
import json, os, statistics, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONTS = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
for f in ("AtkinsonHyperlegibleNext-Regular.ttf", "AtkinsonHyperlegibleNext-SemiBold.ttf",
          "AtkinsonHyperlegibleNext-Bold.ttf", "JetBrainsMono-Regular.ttf"):
    p = os.path.join(FONTS, f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)

GRUND, SCHRIFT, GEDAEMPFT, SCHWACH, LINIE, FETT = "#efeae1", "#2a2724", "#6a6258", "#8b8377", "#d3cabb", "#171512"
BERNSTEIN, STAHL, PETROL, GOLD = "#975b34", "#456579", "#4e716a", "#7e6820"

plt.rcParams.update({
    "font.family": "Atkinson Hyperlegible Next", "font.size": 11,
    "figure.facecolor": GRUND, "axes.facecolor": GRUND, "savefig.facecolor": GRUND,
    "axes.edgecolor": SCHWACH, "axes.labelcolor": GEDAEMPFT, "axes.titlecolor": FETT,
    "axes.titleweight": "semibold", "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlepad": 12,
    "xtick.color": GEDAEMPFT, "ytick.color": GEDAEMPFT, "text.color": SCHRIFT,
    "axes.grid": True, "grid.color": LINIE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

TEXT = {
    "en": {"tiefe": "Context depth (tokens)", "tps": "tokens / s",
           "decode": "Decode throughput vs context depth", "prefill": "Prefill throughput vs context depth",
           "mit": "with MTP", "ohne": "without MTP head", "annahme": "MTP draft acceptance vs context depth",
           "quote": "accepted draft tokens (%)",
           "fuss": "Strata 0.1.44.1 · Qwen3.8-Flash-Next 125B IQ3_XXS · RX 7900 XTX via OCuLink x4 · UGREEN iDX6011 Pro, 64 GB · median of 2–3 fresh prompts",
           "pw": "Strata --pool-workers on a hybrid CPU (6 P + 8 E + 2 LP-E cores)", "pwy": "decode, long answer (tokens / s)", "pwfuss": "Flash-Next IQ2_XS · Strata 0.1.44.1 · median of 3 runs",
           "auto": "auto"},
    "de": {"tiefe": "Kontexttiefe (Token)", "tps": "Token / s",
           "decode": "Schreibtempo über die Kontexttiefe", "prefill": "Einlesetempo über die Kontexttiefe",
           "mit": "mit MTP", "ohne": "ohne MTP-Kopf", "annahme": "MTP-Annahmequote über die Kontexttiefe",
           "quote": "angenommene Vorschlagstoken (%)",
           "fuss": "Strata 0.1.44.1 · Qwen3.8-Flash-Next 125B IQ3_XXS · RX 7900 XTX über OCuLink x4 · UGREEN iDX6011 Pro, 64 GB · Median aus 2–3 frischen Prompts",
           "pw": "Strata --pool-workers auf einer Hybrid-CPU (6 P + 8 E + 2 LP-E-Kerne)", "pwy": "Schreiben, lange Antwort (Token / s)", "pwfuss": "Flash-Next IQ2_XS · Strata 0.1.44.1 · Median aus 3 Läufen",
           "auto": "Automatik"},
}


def k(n):
    return f"{round(n / 1000)}k" if n >= 1000 else str(n)


def zahl(x, sprache):
    s = f"{x:,.0f}" if x >= 100 else f"{x:.1f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if sprache == "de" else s


def lade(pfad):
    werte = defaultdict(lambda: defaultdict(list))
    for z in open(pfad, encoding="utf-8"):
        r = json.loads(z)
        werte[r["label"]][r["ziel"]].append(r)
    return werte


def median(rs, feld):
    v = [r[feld] for r in rs if r.get(feld) is not None]
    return statistics.median(v) if v else None


def tausender(ax, sprache):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: zahl(v, sprache) if v >= 100 else f"{v:.0f}"))


def fuss(fig, t):
    fig.text(0.01, 0.01, t["fuss"], fontsize=8.5, color=GEDAEMPFT, family="JetBrains Mono")


def tiefe_chart(werte, sprache, ziel):
    t = TEXT[sprache]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, farbe, marke, name in (("mit-mtp", BERNSTEIN, "o", t["mit"]), ("ohne-mtp", STAHL, "s", t["ohne"])):
        if label not in werte:
            continue
        tiefen = sorted(werte[label])
        xs = [statistics.median([r["prompt_token"] for r in werte[label][d]]) for d in tiefen]
        dec = [median(werte[label][d], "decode_tps") for d in tiefen]
        pre = [median(werte[label][d], "prompt_tps") for d in tiefen]
        a1.plot(xs, dec, color=farbe, marker=marke, lw=2.4, ms=6, label=name)
        a2.plot(xs, pre, color=farbe, marker=marke, lw=2.4, ms=6, label=name)
        for ax, ys in ((a1, dec), (a2, pre)):
            dy = 0 if ax is a1 else (-10 if label == "mit-mtp" else 10)
            ax.annotate(zahl(ys[-1], sprache), (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                        va="center", fontsize=9.5, color=farbe, family="JetBrains Mono")
    for ax, titel in ((a1, t["decode"]), (a2, t["prefill"])):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 4096, 16384, 65536, 131072, 262144])
        ax.set_xticklabels(["1k", "4k", "16k", "64k", "128k", "256k"])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(titel)
        ax.set_xlabel(t["tiefe"]); ax.set_ylabel(t["tps"])
        ax.legend(frameon=False, loc="lower left" if ax is a1 else "lower right")
        tausender(ax, sprache)
    fuss(fig, t)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(ziel, f"tiefe-{sprache}.png"), dpi=160)
    plt.close(fig)


def annahme_chart(werte, sprache, ziel):
    t = TEXT[sprache]
    if "mit-mtp" not in werte:
        return
    tiefen = sorted(werte["mit-mtp"])
    xs, ys = [], []
    for d in tiefen:
        rs = werte["mit-mtp"][d]
        n = sum(r["draft_n"] or 0 for r in rs); ok = sum(r["draft_ok"] or 0 for r in rs)
        if n:
            xs.append(statistics.median([r["prompt_token"] for r in rs])); ys.append(100 * ok / n)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, ys, color=BERNSTEIN, marker="o", lw=2.4, ms=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f} %", (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=9, color=GEDAEMPFT, family="JetBrains Mono")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384, 65536, 131072, 262144]); ax.set_xticklabels(["1k", "4k", "16k", "64k", "128k", "256k"])
    ax.minorticks_off(); ax.set_ylim(0, 100)
    ax.set_title(t["annahme"]); ax.set_xlabel(t["tiefe"]); ax.set_ylabel(t["quote"])
    fig.text(0.01, 0.01, t["pwfuss"].replace("IQ2_XS", "IQ3_XXS").replace("3 runs", "2–3 fresh prompts").replace("3 Läufen", "2–3 frischen Prompts"), fontsize=8.5, color=GEDAEMPFT, family="JetBrains Mono")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(ziel, f"mtp-annahme-{sprache}.png"), dpi=160)
    plt.close(fig)


def poolworker_chart(sprache, ziel):
    t = TEXT[sprache]
    # gemessen 14.10.2026, Messreihe XTX: lange Antwort, Median aus 3 Läufen
    namen = [t["auto"], "5", "9", "13"]; werte_ = [56.1, 76.6, 76.8, 74.0]
    farben = [STAHL, STAHL, BERNSTEIN, STAHL]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(namen, werte_, color=farben, width=0.6)
    for b, v in zip(bars, werte_):
        ax.annotate(zahl(v, sprache), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, color=SCHRIFT, family="JetBrains Mono")
    ax.set_ylim(0, 90); ax.grid(axis="x", visible=False)
    ax.set_title(t["pw"], fontsize=12); ax.set_ylabel(t["pwy"]); ax.set_xlabel("--pool-workers")
    tausender(ax, sprache)
    fig.text(0.01, 0.01, t["pwfuss"], fontsize=8.5, color=GEDAEMPFT, family="JetBrains Mono")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(ziel, f"pool-workers-{sprache}.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    werte = lade(sys.argv[1])
    ziel = sys.argv[2]
    os.makedirs(ziel, exist_ok=True)
    for s in ("en", "de"):
        tiefe_chart(werte, s, ziel)
        annahme_chart(werte, s, ziel)
        poolworker_chart(s, ziel)
    print("geschrieben:", ", ".join(sorted(os.listdir(ziel))))
