#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап VIII (п. 11): сводный отчёт A/B-стенда — сравнение ячеек матрицы 2x2.

Считает по каждой ячейке: медианы prompt/decode/TTFT (из baseline-points.csv прогона),
счётчики движка (expert reads, OS read/major faults, hit rate), цену VRAM-затрат (slots,
borrows), вердикт побитовой приёмки. Сравнивает:
  * с базовой ячейкой (первая в порядке прогона) — эффект патча/конфигурации;
  * с повтором базы (последняя ячейка с тем же split и бинарником) — шумовой порог прогона;
  * с baseline-profile.csv проекта — совпадает ли авто-ячейка с базовой линией машины.
Пороги значимости (§8.8): prompt tok/s — |Δ| > 1 %, decode — |Δ| > 5 %.

usage: ab-report.py --out DIR_ПРОГОНА [--baseline PATH]
Создаёт: summary.csv, summary.json, summary.md
"""
import argparse
import csv
import json
import pathlib
import re
import statistics

PROJ = pathlib.Path.home() / "strata185" / "Strata+4xV100+NVlink"
THRESH_PROMPT_PCT = 1.0
THRESH_DECODE_PCT = 5.0

SLOT_PAT = re.compile(r"CUDA(\d)[^:]*: \d+ slots \(([\d.]+) GiB\), (\d+) of its (\d+) profiled pairs")
BORROW_PAT = re.compile(r"CUDA(\d) prompt path borrows (\d+) of its (\d+) slots \(([\d.]+) GiB\)")
LAYERS_PAT = re.compile(r"layer split: layers ([\d,\-\s()A-Z]+), one hand-off per window")
CACHE_PAT = re.compile(r"expert cache (\d+) slots, ([\d.]+) GiB")


def read_csv(p):
    """Читает CSV, пропуская комментарии вида '#' (заголовки отчётов проекта начинаются с '#')."""
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def agg_counters(rows):
    if not rows:
        return {}
    return {
        "requests": len(rows),
        "expert_reads_mb_sum": round(sum(fnum(r["expert_reads_mb"]) or 0 for r in rows), 1),
        "os_read_mb_sum": round(sum(fnum(r["os_read_mb"]) or 0 for r in rows), 1),
        "major_faults_sum": int(sum(int(fnum(r["major_faults"]) or 0) for r in rows)),
        "hit_rate_min": min([fnum(r["hit_rate_pct"]) for r in rows if fnum(r["hit_rate_pct"])], default=None),
        "hit_rate_median": statistics.median([fnum(r["hit_rate_pct"]) for r in rows if fnum(r["hit_rate_pct"])] or [0]),
    }


def init_facts(text):
    out = {"slots": {}, "borrows": {}, "cache_slots": None, "layers": None}
    for m in SLOT_PAT.finditer(text):
        out["slots"]["CUDA" + m.group(1)] = {"slots": int(m.group(2)), "gib": float(m.group(3)),
                                             "pairs": int(m.group(4)), "profiled": int(m.group(5))}
    for m in SLOT_PAT.finditer(text):
        pass
    for m in re.finditer(r"expert cache (\d+) slots \(([\d.]+) GiB\), (\d+) of its (\d+) profiled pairs", text):
        out["slots"]["CUDA?"] = {"slots": int(m.group(1)), "gib": float(m.group(2)),
                                 "pairs": int(m.group(3)), "profiled": int(m.group(4))}
    for m in BORROW_PAT.finditer(text):
        out["borrows"]["CUDA" + m.group(1)] = {"borrowed": int(m.group(2)), "of": int(m.group(3)),
                                               "gib": float(m.group(4))}
    m = CACHE_PAT.search(text)
    if m:
        out["cache_slots"] = int(m.group(1))
    m = LAYERS_PAT.search(text)
    if m:
        out["layers"] = m.group(1).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=str(PROJ / "baseline-profile.csv"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)

    manifest = {}
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    cells = [p for p in sorted(out.iterdir())
             if p.is_dir() and ((p / "verdict.txt").exists() or (p / "bench").is_dir())]
    data = {}
    for c in cells:
        points = read_csv(c / "bench" / "baseline-points.csv")
        counters = read_csv(c / "engine-request-counters.csv")
        init_txt = (c / "engine-init-lines.txt").read_text(encoding="utf-8") if (c / "engine-init-lines.txt").exists() else ""
        tc = {}
        if (c / "text-check.json").exists():
            tc = json.loads((c / "text-check.json").read_text(encoding="utf-8"))
        verdict = (c / "verdict.txt").read_text(encoding="utf-8").strip() if (c / "verdict.txt").exists() else ""
        by_len = {}
        for p in points:
            key = int(p["target_prompt_tokens"])
            by_len[key] = {
                "actual_prompt_n": fnum(p["actual_prompt_n_median"]),
                "prompt_tok_s": fnum(p["prompt_tok_s_median"]),
                "prompt_tok_s_min": fnum(p["prompt_tok_s_min"]), "prompt_tok_s_max": fnum(p["prompt_tok_s_max"]),
                "prompt_spread_pct": fnum(p["prompt_spread_pct"]),
                "decode_tok_s": fnum(p["decode_tok_s_median"]), "decode_spread_pct": fnum(p["decode_spread_pct"]),
                "itl_ms": fnum(p["itl_ms_median"]), "ttft_s": fnum(p["ttft_s_median"]),
                "accept_pct": fnum(p["draft_accept_pct"]),
                "fixed_length_ok": p.get("fixed_length_ok"),
                "pcie_rx_mb": sum(fnum(p.get(k)) or 0 for k in
                                  ["pcie_rx_mb_total_4runs_gpu0", "pcie_rx_mb_gpu1", "pcie_rx_mb_gpu2", "pcie_rx_mb_gpu3"]),
                "peak_vram_mib": [fnum(p.get("peak_vram_mib_gpu%d" % i)) for i in range(4)],
            }
        data[c.name] = {"verdict": verdict, "points": by_len, "counters": agg_counters(counters),
                        "init": init_facts(init_txt),
                        "text": {"verdict": tc.get("verdict"), "shas": {str(p["target_tokens"]): p["shas"]
                                                                        for p in tc.get("probes", [])},
                                 "ref_match": {str(p["target_tokens"]): p.get("ref_match") for p in tc.get("probes", [])},
                                 "fixed_length_ok": all(p.get("fixed_length_ok") for p in tc.get("probes", [])) if tc else None}}

    # порядок прогона: из arms-order.txt (пишет ab-run.sh), иначе по именам каталогов
    order = [c.name for c in cells]
    if (out / "arms-order.txt").exists():
        seq = [ln.strip() for ln in (out / "arms-order.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        order = [n for n in seq if n in order] + [n for n in order if n not in seq]
    # базовая ячейка — первая в порядке прогона
    base_name = order[0] if order else None
    base = data.get(base_name, {})

    # сравнение с базовой линией машины (baseline-profile.csv)
    bl = {}
    bp = pathlib.Path(a.baseline)
    if bp.exists():
        for p in read_csv(bp):
            t = fnum(p.get("target_prompt_tokens"))
            if t:
                bl[int(t)] = {"prompt_tok_s": fnum(p.get("prompt_tok_s_median")),
                              "ttft_s": fnum(p.get("ttft_s_median"))}

    summary = {"out": str(out), "manifest": {"captured_utc": manifest.get("captured_utc"),
                                             "build_label": manifest.get("build_label"),
                                             "engine_sha256": (manifest.get("engine_binary") or {}).get("sha256"),
                                             "note": manifest.get("note")},
               "thresholds": {"prompt_tok_s_pct": THRESH_PROMPT_PCT, "decode_tok_s_pct": THRESH_DECODE_PCT},
               "base_cell": base_name, "cells": {}}
    for name, d in data.items():
        deltas = {}
        for ln, v in d["points"].items():
            row = v          # дополняем тот же словарь: CSV и MD читают data[name]["points"]
            b = base.get("points", {}).get(ln)
            if b and b["prompt_tok_s"] and v["prompt_tok_s"]:
                row["d_prompt_vs_base_pct"] = round(100 * (v["prompt_tok_s"] / b["prompt_tok_s"] - 1), 2)
                row["d_ttft_vs_base_pct"] = round(100 * (v["ttft_s"] / b["ttft_s"] - 1), 2) if b["ttft_s"] else None
                row["d_decode_vs_base_pct"] = round(100 * (v["decode_tok_s"] / b["decode_tok_s"] - 1), 2) if b["decode_tok_s"] else None
                row["significant_prompt"] = abs(row["d_prompt_vs_base_pct"]) > THRESH_PROMPT_PCT
                row["significant_decode"] = abs(row.get("d_decode_vs_base_pct") or 0) > THRESH_DECODE_PCT
            if ln in bl and bl[ln]["prompt_tok_s"] and v["prompt_tok_s"]:
                row["d_prompt_vs_machine_baseline_pct"] = round(100 * (v["prompt_tok_s"] / bl[ln]["prompt_tok_s"] - 1), 2)
            deltas[str(ln)] = row
        summary["cells"][name] = {"verdict": d["verdict"], "text": d["text"], "counters": d["counters"],
                                  "init": d["init"], "points": deltas}

    # шумовой порог прогона: расхождение повтора базы с базой
    noise = {}
    for name, d in data.items():
        if name == base_name:
            continue
        for ln, v in d["points"].items():
            if "d_prompt_vs_base_pct" in v and v.get("prompt_tok_s") and base.get("points", {}).get(ln, {}).get("prompt_tok_s"):
                # повтор базы = ячейка с тем же разбиением и тем же бинарником, идущая последней
                if name[-1] == "r" or name == base_name + "r":
                    noise[str(ln)] = abs(v["d_prompt_vs_base_pct"])
    summary["noise_floor_pct_from_base_repeat"] = noise

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # summary.csv
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["cell", "target_prompt_tokens", "actual_prompt_n", "prompt_tok_s", "prompt_spread_pct",
                "d_prompt_vs_base_pct", "d_prompt_vs_machine_baseline_pct", "significant_prompt",
                "decode_tok_s", "d_decode_vs_base_pct", "significant_decode", "itl_ms", "ttft_s",
                "accept_pct", "pcie_rx_mb", "expert_reads_mb_sum", "os_read_mb_sum", "major_faults_sum",
                "hit_rate_min", "cache_slots", "borrowed_slots_total", "layers", "text_verdict", "fixed_length_ok"]
        w = csv.writer(f)
        w.writerow(cols)
        for name in order:
            d = data[name]
            borrowed = sum(v["borrowed"] for v in d["init"]["borrows"].values())
            for ln in sorted(d["points"]):
                v = d["points"][ln]
                w.writerow([name, ln, v.get("actual_prompt_n"), v.get("prompt_tok_s"), v.get("prompt_spread_pct"),
                            v.get("d_prompt_vs_base_pct"), v.get("d_prompt_vs_machine_baseline_pct"),
                            v.get("significant_prompt"), v.get("decode_tok_s"), v.get("d_decode_vs_base_pct"),
                            v.get("significant_decode"), v.get("itl_ms"), v.get("ttft_s"), v.get("accept_pct"),
                            v.get("pcie_rx_mb"), d["counters"].get("expert_reads_mb_sum"),
                            d["counters"].get("os_read_mb_sum"), d["counters"].get("major_faults_sum"),
                            d["counters"].get("hit_rate_min"), d["init"].get("cache_slots"), borrowed,
                            d["init"].get("layers"), d["text"].get("verdict"), v.get("fixed_length_ok")])

    # summary.md
    md = ["# Сводка A/B-стенда", "",
          "Каталог прогона: `%s`" % out,
          "Метка сборки: `%s`; sha256 бинарника: `%s`" % (summary["manifest"].get("build_label"),
                                                          (summary["manifest"].get("engine_sha256") or "")[:16]),
          "Пороги значимости: prompt tok/s |Δ| > %s %%, decode |Δ| > %s %% (шум метода ±0.19…0.29 %% и до 9 %%)" %
          (THRESH_PROMPT_PCT, THRESH_DECODE_PCT),
          "Базовая ячейка: `%s`. Шумовой порог прогона (повтор базы): %s" % (base_name, noise or "не измерен"), ""]
    md.append("| ячейка | длина | prompt tok/s | Δ к базе % | Δ к базовой линии % | decode tok/s | Δ decode % | TTFT с | expert reads МБ | OS read МБ | hit rate min | слотов кэша | занято prompt-путём | разбиение | text_sha |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in order:
        d = data[name]
        borrowed = sum(v["borrowed"] for v in d["init"]["borrows"].values())
        for ln in sorted(d["points"]):
            v = d["points"][ln]
            md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name, ln, v.get("prompt_tok_s"), v.get("d_prompt_vs_base_pct"),
                v.get("d_prompt_vs_machine_baseline_pct"), v.get("decode_tok_s"), v.get("d_decode_vs_base_pct"),
                v.get("ttft_s"), d["counters"].get("expert_reads_mb_sum"), d["counters"].get("os_read_mb_sum"),
                d["counters"].get("hit_rate_min"), d["init"].get("cache_slots"), borrowed,
                (d["init"].get("layers") or "").replace("|", "/"), (d["text"].get("verdict") or "")[:12]))
    md += ["", "## Побитовая приёмка", ""]
    for name in order:
        d = data[name]
        md.append("- `%s`: %s; sha по целевым длинам: %s; совпадение с эталоном: %s" %
                  (name, d["text"].get("verdict"), json.dumps(d["text"].get("shas")),
                   json.dumps(d["text"].get("ref_match"))))
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("сводка:", out / "summary.md")
    print("       ", out / "summary.csv")
    print("       ", out / "summary.json")
    print("ячеек:", len(order), "базовая:", base_name, "шумовой порог:", noise or "нет повтора базы")


if __name__ == "__main__":
    main()
