#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_charts.py — F1..F11 from summary.json + ALL-METRICS.json. Colorblind-safe, 200 dpi.
Run with .venv-doc/bin/python."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("STUDY_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = BASE + "/data"
CH = BASE + "/charts"
os.makedirs(CH, exist_ok=True)
S = json.load(open(f"{RES}/summary.json"))
D = json.load(open(RES + "/ALL-METRICS.json"))["models"]

LAB_COLOR = {"ISTA-DASLab": "#0072B2", "UkisAI": "#D55E00", "AgentionAI": "#009E73"}
ORDER = ["base-iq3_xxs","q2_0","iq3_s","ista-iq2_xs","swift-iq3_xxs","swift-iq2_xs","swift-iq3_s","swift-q2_0","gyro-s"]
SHORT = {
 "base-iq3_xxs": "Q-IQ3_XXS", "q2_0": "Q-Q2_0", "iq3_s": "Q-IQ3_S", "ista-iq2_xs": "Q-IQ2_XS",
 "swift-iq3_xxs": "S-IQ3_XXS", "swift-iq2_xs": "S-IQ2_XS", "swift-iq3_s": "S-IQ3_S", "swift-q2_0": "S-Q2_0",
 "gyro-s": "Gyro-S",
}
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

def bar_colors(keys):
    return [LAB_COLOR[D[k]["lab"]] for k in keys]

# ── F1: HE+ bars by lab ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.1))
x = np.arange(len(ORDER)); w = 0.38
lo = [S["models"][k]["he_low"] for k in ORDER]
hi = [S["models"][k]["he_high"] for k in ORDER]
ax.bar(x - w/2, lo, w, label="low", color=bar_colors(ORDER), alpha=0.95)
ax.bar(x + w/2, hi, w, label="xhigh", color=bar_colors(ORDER), alpha=0.45, hatch="//")
ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in ORDER], rotation=30, ha="right")
ax.set_ylabel("HE+ final (%)"); ax.set_ylim(92, 97)
ax.axhline(96.3, color="grey", lw=0.6, ls=":")
ax.set_title("F1 · HumanEval+ final score — stronger alpha = low, hatched = xhigh (color = lab)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=LAB_COLOR["ISTA-DASLab"], label="ISTA-DASLab"),
                   Patch(color=LAB_COLOR["UkisAI"], label="UkisAI"),
                   Patch(color=LAB_COLOR["AgentionAI"], label="AgentionAI")], loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9, ncol=3, frameon=False)
ax.annotate("96.3 = top band", xy=(0.02, 96.3), xycoords=("axes fraction", "data"), fontsize=6.5, color="grey", va="bottom")
fig.savefig(f"{CH}/F1-heplus.png", bbox_inches="tight"); plt.close(fig)

# ── F2: wall-to-quality Pareto (HE+ low+xhigh) ──────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 4.6))
# manual label offsets to dodge collisions in the 12-16min / 95.0-95.8 cluster
OFF = {
 "base-iq3_xxs": (2, -9), "q2_0": (-30, -2), "iq3_s": (2, 5), "ista-iq2_xs": (-12, -12),
 "swift-iq3_xxs": (-38, 6), "swift-iq2_xs": (3, -13), "swift-iq3_s": (3, 6), "swift-q2_0": (5, -11),
 "gyro-s": (-30, 4),
}
for k in ORDER:
    e = S["models"][k]
    wl, wh = e["wall_low_min"], e["wall_high_min"]
    sc_l, sc_h = e["he_low"], e["he_high"]
    if sc_h is not None and wh:
        ax.annotate("", xy=(wh, sc_h), xytext=(wl, sc_l),
                    arrowprops=dict(arrowstyle="->", color=LAB_COLOR[D[k]["lab"]], lw=1.0, alpha=0.55, shrinkA=6, shrinkB=2), zorder=2)
    ax.scatter(wl, sc_l, color=LAB_COLOR[D[k]["lab"]], marker="o", s=44, zorder=4, edgecolor="white", linewidth=0.6)
    if sc_h is not None and wh:
        ax.scatter(wh, sc_h, color=LAB_COLOR[D[k]["lab"]], marker="s", s=36, alpha=0.85, zorder=4, edgecolor="white", linewidth=0.6)
    dx, dy = OFF.get(k, (2, 3))
    ax.annotate(SHORT[k], (wl, sc_l), fontsize=6.8, xytext=(dx, dy), textcoords="offset points", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
ax.set_xscale("log")
ax.set_xlabel("wall time (min, log scale) — HE+ 164 tasks")
ax.set_ylabel("HE+ final (%)")
ax.set_title("F2 · Time-to-quality: circle = low, square = xhigh, arrow = low→xhigh")
ax.legend(handles=[Patch(color=LAB_COLOR["ISTA-DASLab"], label="ISTA-DASLab"),
                   Patch(color=LAB_COLOR["UkisAI"], label="UkisAI"),
                   Patch(color=LAB_COLOR["AgentionAI"], label="AgentionAI")], loc="lower right", fontsize=7.5)
ax.grid(alpha=0.25, lw=0.4, which="both")
fig.tight_layout(); fig.savefig(f"{CH}/F2-pareto.png"); plt.close(fig)

# ── F3: tokens-to-solve matrix ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.4))
tl = [S["models"][k]["tok_solve_low"] for k in ORDER]
th = [S["models"][k]["tok_solve_high"] for k in ORDER]
x = np.arange(len(ORDER))
ax.bar(x - w/2, tl, w, label="low", color=bar_colors(ORDER), alpha=0.95)
ax.bar(x + w/2, [v if v else 0 for v in th], w, label="xhigh", color=bar_colors(ORDER), alpha=0.45, hatch="//")
ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in ORDER], rotation=30, ha="right")
ax.set_ylabel("avg completion tokens per solved task")
ax.set_title("F3 · Reasoning economy (HE+): tokens to solve — Swift files sit low")
fig.tight_layout(); fig.savefig(f"{CH}/F3-tokens.png"); plt.close(fig)

# ── F4: decode by file, bars per depth ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 3.9))
depths = [0, 32, 128]
perf_keys = [k for k in ORDER if (D[k].get("perf") or {}).get("low")]
xw = 0.26
xs = np.arange(len(perf_keys))
for di, d in enumerate(depths):
    ys = [((D[k]["perf"]["low"].get(f"{d}K") or {}).get("tg_mean")) for k in perf_keys]
    alphas = [0.95, 0.65, 0.42]
    for xi, y in zip(xs, ys):
        if y:
            ax.bar(xi + (di - 1) * xw, y, width=xw*0.92, color=LAB_COLOR[D[perf_keys[xi]]["lab"]], alpha=alphas[di])
for xi, k in enumerate(perf_keys):
    v = (D[k]["perf"]["low"].get("0K") or {}).get("tg_mean")
    if v: ax.annotate(f"{v:.0f}", (xi - xw, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xticklabels([SHORT[k] for k in perf_keys], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("decode tok/s (low arm)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#666", alpha=a, label=f"{d}K context") for d, a in zip(depths, [0.95, 0.65, 0.42])], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False)
ax.set_title("F4 · Decode speed per file — dark bar = 0K context; size class sets the band")
ax.grid(alpha=0.25, lw=0.4, axis="y")
fig.subplots_adjust(bottom=0.34); fig.savefig(f"{CH}/F4-decode.png", bbox_inches="tight"); plt.close(fig)

# ── F5: prefill by file, bars per depth ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 3.9))
for di, d in enumerate(depths):
    ys = [((D[k]["perf"]["low"].get(f"{d}K") or {}).get("pp_mean")) for k in perf_keys]
    alphas = [0.95, 0.65, 0.42]
    for xi, y in zip(xs, ys):
        if y:
            ax.bar(xi + (di - 1) * xw, y/1000, width=xw*0.92, color=LAB_COLOR[D[perf_keys[xi]]["lab"]], alpha=alphas[di])
for xi, k in enumerate(perf_keys):
    v = (D[k]["perf"]["low"].get("0K") or {}).get("pp_mean")
    if v: ax.annotate(f"{v/1000:.0f}", (xi - xw, v/1000), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xticklabels([SHORT[k] for k in perf_keys], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("prefill k tok/s (low arm)")
ax.legend(handles=[Patch(color="#666", alpha=a, label=f"{d}K context") for d, a in zip(depths, [0.95, 0.65, 0.42])], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, frameon=False)
ax.set_title("F5 · Prefill (prompt-reading) speed per file — flat and huge across all files")
ax.grid(alpha=0.25, lw=0.4, axis="y")
fig.subplots_adjust(bottom=0.34); fig.savefig(f"{CH}/F5-prefill.png", bbox_inches="tight"); plt.close(fig)

# ── F6: effort dumbbell (HE+ low→xhigh delta) ──────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(ORDER))
dl = [S["models"][k]["he_low"] for k in ORDER]
dh = [S["models"][k]["he_high"] for k in ORDER]
for i, k in enumerate(ORDER):
    ax.plot([dl[i], dh[i]], [i, i], color=LAB_COLOR[D[k]["lab"]], lw=2, alpha=0.5)
ax.scatter(dl, y, color="#333", s=26, zorder=3, label="low")
ax.scatter(dh, y, color="#999", s=26, marker="s", zorder=3, label="xhigh")
ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in ORDER])
ax.set_xlabel("HE+ final (%)"); ax.set_title("F6 · Effort knob effect on HE+ (low → xhigh)")
ax.legend(); ax.grid(alpha=0.25, axis="x", lw=0.4)
fig.tight_layout(); fig.savefig(f"{CH}/F6-effort.png"); plt.close(fig)

# ── F7: reliability dumbbell (mean of 5 runs -> Pass^5 floor, low arm) ─────
fig, ax = plt.subplots(figsize=(7.2, 3.8))
rows = []
for k in ORDER:
    t = (D[k].get("trials") or {})
    lo = t.get("low")
    if not lo: continue
    rows.append((k, lo["mean"], lo["pass_hat_5"], lo["gap"]))
rows.sort(key=lambda r: -r[2])
ys = np.arange(len(rows))[::-1]
for y, (k, mean, p5, gap) in zip(ys, rows):
    c = LAB_COLOR[D[k]["lab"]]
    ax.plot([p5, mean], [y, y], color=c, lw=2.2, alpha=0.55, zorder=2)
    ax.scatter([mean], [y], s=70, color=c, zorder=3)
    ax.scatter([p5], [y], s=70, facecolors="none", edgecolors=c, linewidths=1.8, zorder=3)
    ax.annotate(f"{mean:.1f}", (mean, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.annotate(f"{p5:.1f}", (p5, y), xytext=(-20, 4), textcoords="offset points", fontsize=8)
    ax.annotate(f"gap {gap:.1f}", ((mean + p5) / 2, y), xytext=(0, -13), textcoords="offset points",
                fontsize=7.5, ha="center", color="#555")
ax.set_yticks(ys); ax.set_yticklabels([SHORT[k] for k, *_ in rows])
ax.set_xlabel("tool-eval-bench score (low effort, 5 runs)")
ax.set_xlim(62, 94)
ax.set_title("F7 · Reliability — solid dot = average day, open dot = floor (passed ALL 5 runs)")
ax.grid(alpha=0.25, lw=0.4, axis="x")
fig.tight_layout(); fig.savefig(f"{CH}/F7-reliability.png"); plt.close(fig)

# ── F8: dimension heat-grid (9×6) ──────────────────────────────────────────
dims = ["HE+ lo", "HE+ hi", "T2 lo", "T2 hi", "MMLU", "MMLU", "GSM8K", "5B vis", "5B hid"]
cols = ["HE+lo","HE+hi","T2lo","T2hi","GSM8K","MMLU","ifeval","bugsV","bugsH"]
grid = []
for k in ORDER:
    e = S["models"][k]
    row = [e["he_low"], e["he_high"], e["t2_low"], e["t2_high"], e["gsm8k"], e["mmlu"], e["ifeval"],
           e["fb_vis"] and int(e["fb_vis"].split("/")[0])*4 if e["fb_vis"] else None,
           e["fb_hid"] and int(e["fb_hid"].split("/")[0])*4 if e["fb_hid"] else None]
    grid.append([v if v is not None else np.nan for v in row])
G = np.array(grid, dtype=float)
# normalize per-column to percent-of-best for color (keep raw for labels)
Gn = G.copy()
for j in range(G.shape[1]):
    col = G[:, j]; m = np.nanmax(col)
    if m: Gn[:, j] = col/m
from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("strata", ["#f2f8f6", "#39a98c", "#0b3d5c"])
fig, ax = plt.subplots(figsize=(7.6, 3.9))
im = ax.imshow(Gn, cmap=CM, aspect="auto", vmin=0.72, vmax=1.0)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=7.5)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[k] for k in ORDER], fontsize=8)
for i in range(len(ORDER)):
    for j in range(len(cols)):
        v = G[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if Gn[i, j] > 0.88 else "#17202a")
ax.set_title("F8 · Dimension grid — darker = closer to the best file in that column")
fig.colorbar(im, ax=ax, shrink=0.8, label="% of best")
fig.tight_layout(); fig.savefig(f"{CH}/F8-grid.png"); plt.close(fig)

# ── F9: lineage tree (text-free simple figure) ─────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.axis("off")
ax.text(0.5, 0.92, "Qwen3.8-Flash-Next (Qwen/Alibaba) — 125B MoE", ha="center", fontsize=10, weight="bold")
branches = [(0.17, "ISTA-DASLab\nGSQ-RCO", "Q-IQ3_XXS · Q-Q2_0\nQ-IQ3_S · Q-IQ2_XS", "#0072B2"),
            (0.5, "UkisAI\nSwift-1.5 + GSQ-RCO", "S-IQ3_XXS · S-IQ2_XS\nS-IQ3_S · S-Q2_0", "#D55E00"),
            (0.83, "AgentionAI\nGyro (APR)", "Gyro-S TQ1_0", "#009E73")]
for x0, mid, files, c in branches:
    ax.annotate("", xy=(x0, 0.62), xytext=(0.5, 0.84), arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
    ax.text(x0, 0.5, mid, ha="center", fontsize=8, color=c, weight="bold")
    ax.text(x0, 0.28, files, ha="center", fontsize=7)
ax.text(0.5, 0.02, "9 measured files — one base model, three quantization lineages", ha="center", fontsize=7.5, style="italic")
fig.tight_layout(); fig.savefig(f"{CH}/F9-lineage.png"); plt.close(fig)

# ── F10: Hard Mode (P) bars per arm — the separating category ─────────────
arms = []
for k in ORDER:
    t = (D[k].get("trials") or {})
    for arm in ("low", "xhigh"):
        cc = t.get(arm) or {}
        p_score = (cc.get("cats") or {}).get("P")
        if p_score is not None:
            arms.append((f"{SHORT[k]} · {arm}", p_score, D[k]["lab"]))
arms.sort(key=lambda a: a[1])
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ys = np.arange(len(arms))
for y, (lbl, v, lab) in zip(ys, arms):
    ax.barh(y, v, height=0.62, color=LAB_COLOR[lab], alpha=0.9)
    ax.annotate(f"{v:.0f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in arms], fontsize=7.5)
ax.set_xlim(0, 100)
ax.set_xlabel("Hard Mode score (%) — multi-step tool chains with recovery")
ax.set_title("F10 · The category that separates files: Hard Mode (P)\nA–D core tool use ~100 for everyone · O structured output 50 for everyone (harness ceiling)")
ax.grid(alpha=0.25, lw=0.4, axis="x")
fig.tight_layout(); fig.savefig(f"{CH}/F10-categories.png"); plt.close(fig)

# ── F11: five-bugs scatter (vis vs hid, wall as size) ──────────────────────
fig, ax = plt.subplots(figsize=(6.8, 4.2))
OFF11 = {"q-iq3_xxs": (-2, 10), "gyro-s": (-2, -14), "swift-iq2_xs": (6, 4),
       "swift-q2_0": (-34, -14), "q-iq2_xs": (-52, 8), "q-iq3_s": (10, -12),
       "swift-iq3_xxs": (6, -14), "swift-iq3_s": (6, 4)}
for k in ORDER:
    e = S["models"][k]
    vis = int(e["fb_vis"].split("/")[0]); hid = int(e["fb_hid"].split("/")[0])
    wsec = D[k]["fivebugs"].get("wall_s") or 300
    ax.scatter(vis, hid, s=np.sqrt(wsec)*7, color=LAB_COLOR[D[k]["lab"]], alpha=0.75)
    dx, dy = OFF11.get(k, (6, 4))
    ax.annotate(SHORT[k], (vis, hid), fontsize=8.5, xytext=(dx, dy), textcoords="offset points")
ax.set_xlabel("visible bugs fixed (of 25)"); ax.set_ylabel("hidden assertions passed (of 25)")
ax.set_title("F11 · Debugging depth (bubble = wall time; color = lab)")
ax.grid(alpha=0.25, lw=0.4)
fig.tight_layout(); fig.savefig(f"{CH}/F11-fivebugs.png"); plt.close(fig)


# ── F12: Quality vs WALL TIME vs Size (bubble = GB) ───────────────────────
fig, ax = plt.subplots(figsize=(9.8, 5.4))
GB = {"base-iq3_xxs": 75.8, "q2_0": 66.4, "iq3_s": 83.6, "ista-iq2_xs": 67.2,
      "swift-iq3_xxs": 76.0, "swift-iq2_xs": 68.2, "swift-iq3_s": 83.7,
      "swift-q2_0": 66.6, "gyro-s": 58.5}
WALL = {k: S["models"][k]["wall_low_min"] for k in ORDER}
OFF12 = {"swift-iq3_xxs": (-14, -15), "swift-q2_0": (-8, 9), "swift-iq2_xs": (7, -15),
         "iq3_s": (7, -13), "base-iq3_xxs": (9, 6), "swift-iq3_s": (7, 5),
         "ista-iq2_xs": (7, 5), "gyro-s": (7, 5), "q2_0": (-48, 6)}
for k in ORDER:
    x = WALL[k]; y = D[k]["heplus"]["low"]["score"]
    ax.scatter(x, y, s=GB[k]*7, color=LAB_COLOR[D[k]["lab"]], alpha=0.65, edgecolors="white", linewidths=1.2, zorder=3)
    dx, dy = OFF12.get(k, (7, 5))
    ax.annotate(SHORT[k] + ("†" if k == "gyro-s" else ""), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5)
ax.set_xlim(7, 58)
ax.axhspan(95.7, 96.5, color="gold", alpha=0.10)
ax.text(8.5, 96.15, "top band", fontsize=7.5, color="darkgoldenrod", ha="left")
ax.axvspan(7, 20, color="green", alpha=0.06)
ax.text(9, 93.7, "under 20 min for all 164 tasks", fontsize=7.5, color="seagreen", ha="left")
ax.set_xlabel("real wall time for 164 HE+ tasks at low effort (minutes) — left = faster")
ax.set_ylabel("HE+ final % (low effort)")
ax.set_title("F12 · Quality vs Wall time vs Size — top-left = best; bubble = file GB")
ax.grid(alpha=0.25, lw=0.4)
fig.savefig(f"{CH}/F12-pareto3.png", bbox_inches="tight"); plt.close(fig)


# ── F13: Hard Mode low→xhigh dumbbell per file ────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 4.0))
rows13 = []
for k in ORDER:
    t = (D[k].get("trials") or {})
    lo = (t.get("low") or {}).get("cats", {}).get("P")
    hi = (t.get("xhigh") or {}).get("cats", {}).get("P")
    if lo is not None and hi is not None:
        rows13.append((k, lo, hi))
rows13.sort(key=lambda r: -max(r[1], r[2]))
ys = np.arange(len(rows13))[::-1]
for y, (k, lo, hi) in zip(ys, rows13):
    c = LAB_COLOR[D[k]["lab"]]
    ax.plot([lo, hi], [y, y], color=c, lw=2.2, alpha=0.5, zorder=2)
    ax.scatter([lo], [y], s=70, facecolors="none", edgecolors=c, linewidths=1.8, zorder=3)
    ax.scatter([hi], [y], s=70, color=c, zorder=3)
    ax.annotate(f"{lo}", (lo, y), xytext=(-16, 4), textcoords="offset points", fontsize=8)
    ax.annotate(f"{hi}", (hi, y), xytext=(7, 4), textcoords="offset points", fontsize=8)
    d = hi - lo
    ax.annotate(f"{d:+.0f}", ((lo + hi) / 2, y), xytext=(0, -13), textcoords="offset points",
                fontsize=7.5, ha="center", color="#555")
ax.set_yticks(ys); ax.set_yticklabels([SHORT[k] for k, *_ in rows13], fontsize=8)
ax.set_xlim(55, 95)
ax.set_xlabel("Hard Mode (P) score — open dot = low effort, solid dot = xhigh")
ax.set_title("F13 · Effort buys the hardest category — Hard Mode low → xhigh per file")
ax.grid(alpha=0.25, lw=0.4, axis="x")
fig.tight_layout(); fig.savefig(f"{CH}/F13-hardmode.png"); plt.close(fig)

# ── F14: tool-eval scenario decomposition (what Pass^5 is made of) ────────
import json as _json
_TF = {
 ("swift-q2_0","low"):"campaign11-swift-q2_0-trials-low.json", ("swift-q2_0","xhigh"):"campaign11-swift-q2_0-trials-xhigh.json",
 ("swift-iq2_xs","low"):"trials-swift-iq2_xs-low.json", ("swift-iq2_xs","xhigh"):"trials-swift-iq2_xs-xhigh.json",
 ("swift-iq3_s","low"):"trials-swift-iq3_s-low.json", ("swift-iq3_s","xhigh"):"trials-swift-iq3_s-xhigh.json",
 ("base-iq3_xxs","low"):"trials-base-iq3_xxs-low.json", ("base-iq3_xxs","xhigh"):"trials-base-iq3_xxs-xhigh.json",
 ("swift-iq3_xxs","low"):"trials-swift-iq3_xxs-low.json", ("swift-iq3_xxs","xhigh"):"trials-swift-iq3_xxs-xhigh.json",
 ("ista-iq2_xs","low"):"campaign11-ista-iq2_xs-trials-low.json", ("ista-iq2_xs","xhigh"):"campaign11-ista-iq2_xs-trials-xhigh.json",
 ("iq3_s","low"):"trials-iq3_s-low.json", ("iq3_s","xhigh"):"trials-iq3_s-xhigh.json",
 ("q2_0","low"):"trials-q2_0-low.json", ("q2_0","xhigh"):"trials-q2_0-xhigh.json",
}
_rows = []
for (k, a), f in _TF.items():
    ps = _json.load(open(f"{RES}/{f}"))["trial_statistics"]["per_scenario"]
    solid = sum(1 for v in ps.values() if v["pass_hat_k"])
    flaky = sum(1 for v in ps.values() if v["pass_at_k"] and not v["pass_hat_k"])
    lock  = sum(1 for v in ps.values() if max(v["points"]) == 1)
    zeros = sum(1 for v in ps.values() if max(v["points"]) == 0)
    _rows.append((solid/92*100, f"{SHORT[k]} {a}", solid, flaky, lock, zeros))
_rows.sort(key=lambda r: -r[0])
_names = [r[1] for r in _rows]
_solid = [r[2] for r in _rows]; _flaky = [r[3] for r in _rows]; _lock = [r[4] for r in _rows]; _zero = [r[5] for r in _rows]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ypos = np.arange(len(_rows))[::-1]
ax.barh(ypos, _solid, color="#2e8b57", label="full marks in ALL 5 runs (this IS the Pass^5 floor)")
ax.barh(ypos, _flaky, left=_solid, color="#e6a817", label="passed some runs, missed others (flaky)")
ax.barh(ypos, _lock, left=[s+f for s,f in zip(_solid,_flaky)], color="#9aa5ad", label="always half-right, never full marks")
ax.barh(ypos, _zero, left=[s+f+l for s,f,l in zip(_solid,_flaky,_lock)], color="#c0392b", label="never scored (incl. safety traps)")
for y, (p5, n, s, f, l, z) in zip(ypos, _rows):
    ax.text(s/2, y, f"{s}", va="center", ha="center", color="white", fontsize=8, fontweight="bold")
ax.set_yticks(ypos); ax.set_yticklabels(_names, fontsize=8.5)
ax.set_xlabel("92 agentic scenarios per arm (count of scenarios)")
ax.set_title("F14 · What the reliability floor is made of — green = your agent can trust it daily")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
ax.grid(alpha=0.25, lw=0.4, axis="x")
fig.subplots_adjust(bottom=0.24); fig.savefig(f"{CH}/F14-decomposition.png", bbox_inches="tight"); plt.close(fig)

# ── F15: FINAL SCOREBOARD heatmap — every arm, every axis ─────────────────
_SB = _json.load(open(f"{RES}/FINAL-SCOREBOARD.json"))
norms = {}
for _ax in ["coding","floor","hardmode","mmlu","gsm8k","debug","decode"]:
    _vals = [r[_ax] for r in _SB if r[_ax] is not None]
    norms[_ax] = (min(_vals), max(_vals))
_AXES = [("coding","Coding\nHE+"), ("floor","Agents\nPass^5"), ("hardmode","Hard\nMode"),
         ("mmlu","MMLU"), ("gsm8k","GSM8K"), ("debug","Debug\n/25"), ("decode","Decode\ntok/s")]
from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("strata", ["#ffffff", "#7fd1c1", "#2a9d8f", "#0b3d5c"])
_rows_sb = [r for r in _SB if r["composite"] is not None] + [r for r in _SB if r["composite"] is None]
fig, ax = plt.subplots(figsize=(9.6, 6.6))
ncol = len(_AXES) + 1  # + safety
grid = np.zeros((len(_rows_sb), ncol))
for ri, r in enumerate(_rows_sb):
    for ci, (axk, _) in enumerate(_AXES):
        v = r[axk]
        if v is None:
            grid[ri, ci] = np.nan
        else:
            lo, hi = norms[axk]
            grid[ri, ci] = (v - lo) / (hi - lo) if hi > lo else 0.5
    grid[ri, ncol-1] = {False: 1.0, True: 0.0, None: np.nan}[r["capped"]]
for ri, r in enumerate(_rows_sb):
    for ci in range(ncol):
        v = grid[ri, ci]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor="#e9edf1", edgecolor="white", lw=1.2))
            ax.text(ci+0.5, ri+0.5, "n/a", ha="center", va="center", fontsize=6.5, color="#8a97a3")
        else:
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor=CM(v), edgecolor="white", lw=1.2))
            raw = r[_AXES[ci][0]] if ci < len(_AXES) else None
            txt = ("capped" if r["capped"] is True else "clean") if ci == ncol-1 else (f"{raw:.0f}" if raw is not None else "")
            ax.text(ci+0.5, ri+0.5, txt, ha="center", va="center", fontsize=7,
                    color="white" if v > 0.62 else "#22313f", fontweight="bold" if ci == 0 else "normal")
    comp = r["composite"]
    ax.text(-0.35, ri+0.5, f"{comp:.0f}" if comp is not None else "n/a", ha="right", va="center",
            fontsize=8, fontweight="bold", color="#22313f")
labels = []
for r in _rows_sb:
    k = r["key"]; labc = LAB_COLOR[D[k]["lab"]] if k in D else "#555"
    labels.append(f"{SHORT[k]} {r['arm']}")
ax.set_yticks(np.arange(len(_rows_sb)) + 0.5)
ax.set_yticklabels(labels, fontsize=8)
for tick, r in zip(ax.get_yticklabels(), _rows_sb):
    tick.set_color(LAB_COLOR[D[r["key"]]["lab"]])
    tick.set_fontweight("bold")
ax.set_xticks(np.arange(ncol) + 0.5)
ax.set_xticklabels([a[1] for a in _AXES] + ["Safety"], fontsize=8)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
ax.text(-0.35, -0.9, "OVERALL", ha="right", va="center", fontsize=8, fontweight="bold")
ax.set_xlim(-1.6, ncol); ax.set_ylim(len(_rows_sb), -1.2)
ax.set_title("F15 · Final scoreboard — every arm ranked; darker = better on that axis", pad=26)
ax.text(ncol, len(_rows_sb)+0.3, "ranked by OVERALL (avg of all axes, 0–100)", ha="right", fontsize=7, color="#8a97a3")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.subplots_adjust(left=0.16, bottom=0.04, top=0.92); fig.savefig(f"{CH}/F15-scoreboard.png", bbox_inches="tight"); plt.close(fig)
print("charts written:", sorted(os.listdir(CH)))
# ── F15: FINAL SCOREBOARD heatmap — every arm, every axis ─────────────────
_SB = _json.load(open(f"{RES}/FINAL-SCOREBOARD.json"))
norms = {}
for _ax in ["coding","floor","hardmode","mmlu","gsm8k","debug","decode"]:
    _vals = [r[_ax] for r in _SB if r[_ax] is not None]
    norms[_ax] = (min(_vals), max(_vals))
_AXES = [("coding","Coding\nHE+"), ("floor","Agents\nPass^5"), ("hardmode","Hard\nMode"),
         ("mmlu","MMLU"), ("gsm8k","GSM8K"), ("debug","Debug\n/25"), ("decode","Decode\ntok/s")]
from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("strata", ["#ffffff", "#7fd1c1", "#2a9d8f", "#0b3d5c"])
_rows_sb = [r for r in _SB if r["composite"] is not None] + [r for r in _SB if r["composite"] is None]
fig, ax = plt.subplots(figsize=(9.6, 6.6))
ncol = len(_AXES) + 1  # + safety
grid = np.zeros((len(_rows_sb), ncol))
for ri, r in enumerate(_rows_sb):
    for ci, (axk, _) in enumerate(_AXES):
        v = r[axk]
        if v is None:
            grid[ri, ci] = np.nan
        else:
            lo, hi = norms[axk]
            grid[ri, ci] = (v - lo) / (hi - lo) if hi > lo else 0.5
    grid[ri, ncol-1] = {False: 1.0, True: 0.0, None: np.nan}[r["capped"]]
for ri, r in enumerate(_rows_sb):
    for ci in range(ncol):
        v = grid[ri, ci]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor="#e9edf1", edgecolor="white", lw=1.2))
            ax.text(ci+0.5, ri+0.5, "n/a", ha="center", va="center", fontsize=6.5, color="#8a97a3")
        else:
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor=CM(v), edgecolor="white", lw=1.2))
            raw = r[_AXES[ci][0]] if ci < len(_AXES) else None
            txt = ("capped" if r["capped"] is True else "clean") if ci == ncol-1 else (f"{raw:.0f}" if raw is not None else "")
            ax.text(ci+0.5, ri+0.5, txt, ha="center", va="center", fontsize=7,
                    color="white" if v > 0.62 else "#22313f", fontweight="bold" if ci == 0 else "normal")
    comp = r["composite"]
    ax.text(-0.35, ri+0.5, f"{comp:.0f}" if comp is not None else "n/a", ha="right", va="center",
            fontsize=8, fontweight="bold", color="#22313f")
labels = []
for r in _rows_sb:
    k = r["key"]; labc = LAB_COLOR[D[k]["lab"]] if k in D else "#555"
    labels.append(f"{SHORT[k]} {r['arm']}")
ax.set_yticks(np.arange(len(_rows_sb)) + 0.5)
ax.set_yticklabels(labels, fontsize=8)
for tick, r in zip(ax.get_yticklabels(), _rows_sb):
    tick.set_color(LAB_COLOR[D[r["key"]]["lab"]])
    tick.set_fontweight("bold")
ax.set_xticks(np.arange(ncol) + 0.5)
ax.set_xticklabels([a[1] for a in _AXES] + ["Safety"], fontsize=8)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
ax.text(-0.35, -0.9, "OVERALL", ha="right", va="center", fontsize=8, fontweight="bold")
ax.set_xlim(-1.6, ncol); ax.set_ylim(len(_rows_sb), -1.2)
ax.set_title("F15 · Final scoreboard — every arm ranked; darker = better on that axis", pad=26)
ax.text(ncol, len(_rows_sb)+0.3, "ranked by OVERALL (avg of all axes, 0–100)", ha="right", fontsize=7, color="#8a97a3")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.subplots_adjust(left=0.16, bottom=0.04, top=0.92); fig.savefig(f"{CH}/F15-scoreboard.png", bbox_inches="tight"); plt.close(fig)
print("charts written:", sorted(os.listdir(CH)))
