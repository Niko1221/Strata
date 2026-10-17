"""Render review figures from the published analysis; requires matplotlib."""

import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
a = json.loads((ROOT / "data/analysis.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#b8c1cc",
        "text.color": "#243443",
        "axes.labelcolor": "#425466",
        "xtick.color": "#627386",
        "ytick.color": "#425466",
        "svg.fonttype": "none",
        "svg.hashsalt": "strata-1922-20261015",
        "savefig.facecolor": "white",
    }
)

for name in ("english", "code"):
    rows = [q for q in a["rows"] if q["name"] == name]
    summary = a["summary"][name]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 9.0), sharey=True)
    fig.subplots_adjust(left=0.21, right=0.97, top=0.79, bottom=0.22, wspace=0.24)
    title = "Resident-expert lookahead: " + (
        "English generation" if name == "english" else "Code generation"
    )
    fig.text(0.06, 0.95, title, fontsize=18, weight="bold")
    fig.text(
        0.06,
        0.914,
        "Ryzen 5 7600X  /  Radeon RX 7900 XTX  /  IQ1_M  /  HIP",
        fontsize=10.5,
        color="#627386",
    )
    scope = f"{summary['eligible_pairs']} of 12 A/B pairs match output, MTP work and prior requests"
    if name == "code":
        scope += "  |  exploratory"
    fig.text(0.06, 0.88, scope, fontsize=10.5)
    for col, (metric, key, color) in enumerate(
        [
            ("throughput_change_pct", "primary_for_this_workload", "#176b90"),
            ("cpu_reduction_pct", "cpu_reduction", "#137d72"),
        ]
    ):
        ax = axes[col]
        stat = summary[key]
        ax.axhspan(15.5, 16.5, color=color, alpha=0.065, zorder=0)
        ax.axhspan(0.5, 2.5, color="#eef1f4", alpha=0.85, zorder=0)
        ax.axvline(0, color="#8b98a5", lw=1, zorder=1)
        ax.grid(axis="x", color="#e4e9ef", linewidth=0.6, zorder=0)
        ax.tick_params(axis="y", length=0, pad=10, labelsize=10)
        ax.set_ylim(0.4, 16.7)
        ax.set_xlim((-2.2, 6.8) if col == 0 else (-3.1, 12))
        ax.set_xticks([-2, 0, 2, 4, 6] if col == 0 else [-2, 0, 2, 4, 6, 8, 10])
        ax.set_xlabel(
            "Throughput change (%)" if col == 0 else "CPU time reduction (%)",
            labelpad=10,
        )
        heading = "Decode throughput" if col == 0 else "Request CPU time saved"
        detail = f"{stat['estimate_pct']:+.2f}%"
        if name == "english":
            lo, hi = stat["ci95_pct"]
            detail += f"   [{lo:+.2f}, {hi:+.2f}]"
        else:
            detail += "   descriptive mean"
        ax.set_title(
            heading + "\n" + detail,
            loc="left",
            fontsize=12,
            color=color,
            pad=16,
            weight="bold",
        )
        if name == "english":
            lo, hi = stat["ci95_pct"]
            ax.errorbar(
                stat["estimate_pct"],
                16,
                xerr=[[stat["estimate_pct"] - lo], [hi - stat["estimate_pct"]]],
                fmt="o",
                color=color,
                ms=7,
                capsize=4,
                lw=2.2,
                zorder=4,
            )
        else:
            ax.scatter(stat["estimate_pct"], 16, s=55, c=color, zorder=4)
        for q in rows:
            control = q["kind"] == "aa"
            y = (2 if q["pair"] == 0 else 1) if control else 15 - q["pair"]
            v = q[metric]
            if control and q["order"][0] == "auto":
                v = (
                    100 * (q["auto_ms"] / q["on_ms"] - 1)
                    if col == 0
                    else 100 * (1 - q["on_cpu_s"] / q["auto_cpu_s"])
                )
            c = "#788593" if control or not q["eligible"] else color
            ax.scatter(
                v,
                y,
                s=42,
                marker="D" if control else "o",
                facecolors=c if q["eligible"] else "white",
                edgecolors=c,
                lw=1.4,
                zorder=4,
            )
            ax.annotate(
                f"{v:+.2f}",
                (v, y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=c,
            )
        ticks = [16] + [15 - p for p in range(1, 13)] + [2, 1]
        labels = (
            ["Matched mean"]
            + [
                f"{p:02d}  ·  {'ON first' if next(q for q in rows if q['pair'] == p)['order'][0] == 'on' else 'AUTO first'}"
                for p in range(1, 13)
            ]
            + ["A/A · start", "A/A · end"]
        )
        ax.set_yticks(ticks, labels)
    axes[1].tick_params(labelleft=False)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="#176b90",
            markeredgecolor="#176b90",
            label="Matching work",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#788593",
            label="Different work; excluded from matched estimate",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="none",
            markerfacecolor="#788593",
            markeredgecolor="#788593",
            label="Both engines ON",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.065),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    note = "600-token warmup + 600-token output per workload · 128K allocated capacity · same executable"
    fig.text(0.06, 0.045, note, fontsize=9, color="#627386")
    fig.text(
        0.06,
        0.022,
        "A/B: automatic vs forced ON. A/A: second ON vs first ON. English interval: 95% paired bootstrap.",
        fontsize=8.5,
        color="#627386",
    )
    for ext in ("png", "svg", "pdf"):
        metadata = (
            {"Date": "2026-10-15"}
            if ext == "svg"
            else {
                "CreationDate": datetime(2026, 10, 15, tzinfo=timezone.utc),
                "ModDate": datetime(2026, 10, 15, tzinfo=timezone.utc),
            }
            if ext == "pdf"
            else {"Software": "Matplotlib; Strata lookahead benchmark"}
        )
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=200, metadata=metadata)
        if ext == "svg":
            path = ROOT / "figures" / f"{name}.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
