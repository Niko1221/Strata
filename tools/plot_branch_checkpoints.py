"""Render the measured plain-server checkpoint comparison (requires matplotlib)."""
import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    rows = json.loads(args.results.read_text())["rows"]
    fresh = [r["ttft_s"] for r in rows if r["label"].startswith("fresh-")]
    reuse = [r["ttft_s"] for r in rows if r["label"].startswith("reuse-")]
    single = lambda label: next(r["ttft_s"] for r in rows if r["label"] == label)
    values = [median(fresh), median(reuse), single("protected-restart"), single("deleted-cache-replay")]
    labels = ["Fresh request", "Disk checkpoint reuse", "Protected checkpoint after restart", "Checkpoint deleted: replay history"]
    colors = ["#9aa7b6", "#167dba", "#258b79", "#b8c2cd"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(10.5, 5.9), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=.35, right=.92, top=.73, bottom=.26)
    fig.text(.045, .94, "32K context: native checkpoint reuse", fontsize=21, weight="bold", color="#152c44")
    fig.text(.045, .875, "Time to first output token  |  Lower is better", fontsize=12, color="#516478")
    fig.text(.945, .87, f"{values[0]/values[1]:.1f}x faster", ha="right", fontsize=18, weight="bold", color=colors[1])
    ax.barh(range(4), values, height=.53, color=colors)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values)*1.22)
    ax.set_xlabel("Seconds", labelpad=9, color="#516478")
    ax.xaxis.grid(True, color="#e9edf2")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelcolor="#20364e", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(values):
        ax.text(value+.08, i, f"{value:.2f} s", va="center", weight="bold", color="#152c44")
    fig.text(.045, .14, "RTX PRO 6000 Blackwell 96 GB  |  ISTA IQ3_XXS  |  int8 KV  |  MTP speculation 4", fontsize=10, color="#516478")
    fig.text(.045, .095, f"Fresh/reuse: medians of {len(fresh)}/{len(reuse)} runs. Restart/deletion: one run each. Filesystem caches not flushed.", fontsize=9.5, color="#516478")
    fig.text(.045, .05, "Same server and engine: this compares cache conditions, not two software versions. Model startup excluded.", fontsize=9.5, color="#516478")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
