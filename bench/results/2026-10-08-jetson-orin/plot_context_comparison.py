"""Plot the published comparison tables; requires matplotlib, no GPU.

Read rounded table values so the figures match the README exactly. Throughput
uses medians; memory uses the reported minima, maxima and allocation values.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def tables():
    result = []
    current = None
    for line in (ROOT / "README.md").read_text().splitlines():
        if line.startswith("| Context | Engine |"):
            current = []
            result.append(current)
        elif current is not None and line.startswith("| "):
            if not line.startswith("| ---"):
                current.append([s.strip() for s in line.strip("|").split("|")])
        else:
            current = None
    return result


def value(row, column, part=0):
    text = row[column].split("/")[part].strip().split()[0]
    return math.nan if text == "—" else float(text)


def plot(rows, metrics, filename, title, note):
    contexts = sorted({int(r[0]) for r in rows})
    columns = 2 if len(metrics) == 4 else 3
    fig, axes = plt.subplots(2 if len(metrics) == 4 else 1, columns,
                             figsize=(14, 8 if len(metrics) == 4 else 5.2),
                             layout="constrained", squeeze=False)
    for ax, (label, column, part) in zip(axes.flat, metrics):
        for engine, color, marker in (("baseline", "#56677c", "o"),
                                       ("rebased", "#087f8c", "s")):
            selected = sorted((r for r in rows if r[1] == engine),
                              key=lambda r: int(r[0]))
            ax.plot([int(r[0]) for r in selected],
                    [value(r, column, part) for r in selected],
                    label=engine.capitalize(), color=color, marker=marker,
                    linewidth=2, markersize=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts, [f"{c // 1024}K" for c in contexts])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Context capacity (tokens)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(title + "\n" + note, fontsize=13)
    fig.savefig(ROOT / filename, dpi=160)
    plt.close(fig)


def main():
    decode, prompt, memory = tables()
    plot(decode, [(f"{n}-token prompt: decode tokens/s", i, 0)
                  for n, i in ((172, 4), (557, 5), (1069, 6))]
         + [("Minimum available physical RAM (GiB)", 3, 0)],
         "context-decode-memory.png", "Jetson AGX Orin 32 GB: decode and physical headroom",
         "Throughput: three-run medians; RAM: sampled minimum including startup. Missing 1K/1069 point omitted.")
    plot(prompt, [(f"{n}-token prompt: prompt tokens/s", i, 0)
                  for n, i in ((172, 2), (557, 3), (1069, 4))],
         "context-prompt.png", "Jetson AGX Orin 32 GB: matched prompt throughput",
         "Three-run medians; no reused prompt tokens. Missing 1K/1069 point omitted.")
    plot(memory, [("Expert cache allocation (GiB)", 2, 0),
                  ("Expert cache slots", 2, 1), ("Maximum swap in use (GiB)", 3, 0),
                  ("Host physical disk reads (GiB)", 4, 0)],
         "context-cache-swap-disk.png", "Jetson AGX Orin 32 GB: cache, swap and host disk reads",
         "Reported allocations/maxima/totals, not medians. Rebased disk totals include additional near-limit and recovery requests.")


if __name__ == "__main__":
    main()
