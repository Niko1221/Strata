#!/usr/bin/env python3
"""Generate a GitHub-renderable SVG comparing IQ3_S and llama.cpp Qwen3.8-27B."""
import argparse
import json
import math
from pathlib import Path

TARGETS = ("4096", "32768", "128000")
LABELS = ("4K", "32K", "128K")
METRICS = (
    ("decode_tok_s", "Decode tok/s"),
    ("prefill_tok_s", "Prefill tok/s"),
    ("client_ttft_s", "TTFT seconds"),
)
IQ_COLOR = "#2563eb"
LLAMA_COLOR = "#d97706"


def load(path):
    return json.loads(Path(path).read_text())


def value(summary, target, metric, key):
    return float(summary[target][metric][key])


def nice_ceil(value):
    if value <= 0:
        return 1
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    for mult in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        candidate = mult * base
        if value <= candidate:
            return candidate
    return 10 * base


def fmt(value):
    if value >= 1000:
        return f"{int(round(value)):,}"
    if value >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(x0, y0, width, height, title, iq, llama, metric):
    left = 58
    right = 18
    top = 34
    bottom = 54
    plot_x = x0 + left
    plot_y = y0 + top
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(
        value(iq, target, metric, "max") for target in TARGETS
    )
    max_value = max(max_value, max(value(llama, target, metric, "max") for target in TARGETS))
    ymax = nice_ceil(max_value * 1.12)

    def y(v):
        return plot_y + plot_h - (float(v) / ymax) * plot_h

    out = []
    out.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" class="title">{esc(title)}</text>')
    out.append(f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" class="axis" />')
    out.append(f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{plot_y + plot_h:.1f}" class="axis" />')

    for i in range(5):
        tick = ymax * i / 4
        yy = y(tick)
        out.append(f'<line x1="{plot_x:.1f}" y1="{yy:.1f}" x2="{plot_x + plot_w:.1f}" y2="{yy:.1f}" class="grid" />')
        out.append(f'<text x="{plot_x - 8:.1f}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{fmt(tick)}</text>')

    group_w = plot_w / len(TARGETS)
    bar_w = group_w * 0.28
    inner_gap = group_w * 0.08
    for idx, target in enumerate(TARGETS):
        group_x = plot_x + idx * group_w
        center = group_x + group_w / 2
        iq_x = center - bar_w - inner_gap / 2
        llama_x = center + inner_gap / 2
        out.append(f'<text x="{center:.1f}" y="{plot_y + plot_h + 22:.1f}" text-anchor="middle" class="tick">{LABELS[idx]}</text>')

        for source, bx, color in ((iq, iq_x, IQ_COLOR), (llama, llama_x, LLAMA_COLOR)):
            med = value(source, target, metric, "median")
            lo = value(source, target, metric, "min")
            hi = value(source, target, metric, "max")
            bar_y = y(med)
            bar_h = plot_y + plot_h - bar_y
            out.append(f'<rect x="{bx:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2" />')
            cx = bx + bar_w / 2
            out.append(f'<line x1="{cx:.1f}" y1="{y(hi):.1f}" x2="{cx:.1f}" y2="{y(lo):.1f}" class="whisker" />')
            out.append(f'<line x1="{cx - 5:.1f}" y1="{y(hi):.1f}" x2="{cx + 5:.1f}" y2="{y(hi):.1f}" class="whisker" />')
            out.append(f'<line x1="{cx - 5:.1f}" y1="{y(lo):.1f}" x2="{cx + 5:.1f}" y2="{y(lo):.1f}" class="whisker" />')
            out.append(f'<text x="{cx:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" class="value">{fmt(med)}</text>')

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq3s", type=Path, default=Path("iq3_s/summary.json"))
    parser.add_argument("--llama", type=Path, default=Path("club3090/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("charts/qwen38-27b-vs-iq3s-speed.svg"))
    args = parser.parse_args()
    iq = load(args.iq3s)
    llama = load(args.llama)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    width = 1120
    height = 430
    panel_w = 350
    gap = 20
    x0 = 10
    y0 = 58
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="IQ3_S versus llama.cpp Qwen3.8-27B speed">')
    parts.append(f'''<style>
      text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
      .title {{ font-size: 16px; font-weight: 650; }}
      .tick {{ font-size: 12px; fill: #374151; }}
      .value {{ font-size: 11px; fill: #111827; }}
      .axis {{ stroke: #6b7280; stroke-width: 1; }}
      .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
      .whisker {{ stroke: #111827; stroke-width: 1.4; }}
      .legend {{ font-size: 13px; }}
    </style>''')
    parts.append(f'<text x="18" y="28" class="title">IQ3_S versus llama.cpp Qwen3.8-27B on one RTX 3090</text>')
    parts.append(f'<text x="18" y="46" class="tick">Bars show the median of three runs; the thin line shows the minimum to maximum range.</text>')
    parts.append(f'<rect x="{width - 305}" y="20" width="14" height="14" fill="{IQ_COLOR}" rx="2" />')
    parts.append(f'<text x="{width - 285}" y="32" class="legend">Strata IQ3_S</text>')
    parts.append(f'<rect x="{width - 170}" y="20" width="14" height="14" fill="{LLAMA_COLOR}" rx="2" />')
    parts.append(f'<text x="{width - 150}" y="32" class="legend">llama.cpp Qwen3.8-27B</text>')

    for idx, (metric, title) in enumerate(METRICS):
        parts.append(panel(x0 + idx * (panel_w + gap), y0, panel_w, height - y0, title, iq, llama, metric))

    parts.append("</svg>")
    args.out.write_text("\n".join(parts) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
