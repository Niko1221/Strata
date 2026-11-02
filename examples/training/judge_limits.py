"""把 `bench_fullctx.py` 的 json 判成 PASS/FAIL：极限是多少、有没有「降速」、公式准不准。

三条判据（每条都能单独定位问题）
--------------------------------
1. **极限**：某方案能连续「一步完成 + 峰值 ≤ 预算」的最大 T。`hang`/`oom`/`budget` 都算断点。
2. **无降速**（bug 探测器）：同方案 tok/s 随 T 单调不降；若 T 变大而 tok/s 掉 **>15%** 且
   显存没顶（`peak < 预算 × 0.9`），判 bug —— 那不是"长上下文本来就慢"，而是某处退化了。
3. **公式准**：实测极限 ≥ `0.85 × budget 里算的 T_model` ⇒ 公式可信，其余格子直接用算的；
   差 >15% ⇒ 先修公式，别拿它填矩阵。

`synthetic: true` 的格子**不进 tok/s 判据**（随机 token 的 tok/s 没有意义），但它的
峰值显存与状态仍然有效——那是「该不该跳过」的唯一依据。

用法
----
    python judge_limits.py bench_limits.json --budget bench_budget.json --out judge.json
    python judge_limits.py bench_limits.json --json-only        # 只出表，不做退出码判定
"""

from __future__ import annotations

import argparse
import json
import os

DROP_TOL = 0.15          # tok/s 掉超过 15% 视为降速
FORMULA_TOL = 0.15       # 实测极限低于公式 15% 以上 ⇒ 公式偏乐观
VRAM_GIB = 48.0          # 本机；--vram 可改
RESERVE_FRAC = 0.95      # 预算 = VRAM × 0.95（与 budget.py / mem_guard 同口径）


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rows_of(d: dict) -> list[dict]:
    return d.get("rows") or []


def group(rows: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault((r["algo"], r["mode"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["T"])
    return out


def budgets_from(bench: dict, vram: float) -> dict[int, bool]:
    """每档的"预算是否还够"——只用来判断降速是否被显存解释掉。"""
    cap = vram * RESERVE_FRAC
    out = {}
    for r in rows_of(bench):
        if r.get("status") == "ok" and "peak" in r:
            out[r["T"]] = r["peak"] >= cap * 0.9
    return out


def judge_cell_series(cells: list[dict], near_cap: dict[int, bool]) -> dict:
    """一个 (algo, mode) 的阶梯 → 极限 + 降速违例 + 每档 tok/s。"""
    ok = [r for r in cells if r.get("status") == "ok"]
    limit = max((r["T"] for r in ok), default=0)
    at_limit = next((r for r in ok if r["T"] == limit), None)
    breaks = [r for r in cells if r.get("status") in ("oom", "hang", "budget")]

    viol = []
    real = [r for r in ok if not r.get("synthetic", True) and r.get("tok_s")]
    for a, b in zip(real, real[1:]):
        if b["T"] <= a["T"]:
            continue
        drop = 1.0 - b["tok_s"] / a["tok_s"]
        if drop > DROP_TOL and not near_cap.get(b["T"], False):
            viol.append({"from_T": a["T"], "to_T": b["T"], "drop": round(drop, 3),
                         "peak": round(b.get("peak", 0.0), 1)})

    return {"limit_T": limit, "at_limit": at_limit, "breaks": breaks,
            "slowdown": viol, "n_ok": len(ok), "n_cells": len(cells),
            "tok_s": {r["T"]: round(r["tok_s"], 1) for r in ok if r.get("tok_s")}}


def compare_formula(report: dict, budget: dict) -> dict:
    """实测极限 vs budget.py 算出来的 T_model（同一个 (algo, mode) 的 profile 名不完全一样，
    所以按「算法前缀 + 模式」宽松匹配，匹配不上就只报数不判定）。"""
    rows = budget.get("rows") or []
    by = {}
    for r in rows:
        p = (r.get("profile") or "").lower()
        for algo in ("ce", "dpo", "grpo"):
            if algo in p:
                mode = "mezo" if "mezo" in p else ("bwd" if "bwd" in p else "")
                if mode:
                    by[(algo, mode)] = max(by.get((algo, mode), 0), int(r.get("max_seq") or 0))
        if "engine" in p:
            by.setdefault(("engine", "mezo"), int(r.get("max_seq") or 0))
    out = {}
    for (algo, mode), v in report.items():
        key = (algo, mode) if (algo, mode) in by else (("engine", "mezo") if mode == "mezo" else (algo, mode))
        model_t = by.get(key, 0)
        if not model_t:
            continue
        got = v["limit_T"]
        out[f"{algo}/{mode}"] = {"measured": got, "formula": model_t,
                                 "pass": got >= FORMULA_TOL and got >= 0.85 * model_t,
                                 "ratio": round(got / model_t, 3) if model_t else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bench", help="bench_fullctx.py 的 --out json")
    ap.add_argument("--budget", default="", help="budget.py --json 的输出（给了就判公式准不准）")
    ap.add_argument("--vram", type=float, default=VRAM_GIB)
    ap.add_argument("--out", default="")
    ap.add_argument("--json-only", action="store_true", help="不因违例返回非零")
    args = ap.parse_args()

    bench = load(args.bench)
    rows = rows_of(bench)
    if not rows:
        raise SystemExit(f"{args.bench}: 一行都没有")
    near_cap = budgets_from(bench, args.vram)

    report = {}
    for key, cells in sorted(group(rows).items()):
        report[key] = judge_cell_series(cells, near_cap)
    named = {f"{a}/{m}": v for (a, m), v in report.items()}

    print("| 方案 | 极限 T | 该点步时 | 该点 tok/s | 峰值 GiB | RSS GiB | 断点 | 降速 |")
    print("|---|---:|---:|---:|---:|---:|---|---|")
    for k, v in named.items():
        r = v["at_limit"] or {}
        brk = ",".join(f"{b['T']}:{b['status']}" for b in v["breaks"][:2]) or "—"
        slow = f"{len(v['slowdown'])} 处" if v["slowdown"] else "无"
        print(f"| {k} | {v['limit_T']} | {r.get('s', float('nan')):.2f} | "
              f"{v['tok_s'].get(v['limit_T'], 0):.1f} | {r.get('peak', 0):.1f} | "
              f"{r.get('peak_rss', 0):.1f} | {brk} | {slow} |")

    synthetic = {r["T"] for r in rows if r.get("synthetic", True)}
    if synthetic and len(synthetic) == len({r["T"] for r in rows}):
        print("\n> 全部格子 synthetic=true：极限与显存有效，tok/s 与降速判定**不成立**。")

    formula = {}
    if args.budget:
        formula = compare_formula(report, load(args.budget))
        if formula:
            print("\n| 方案 | 实测极限 | 公式 T_model | 比值 | 判定 |")
            print("|---|---:|---:|---:|---|")
            for k, v in formula.items():
                print(f"| {k} | {v['measured']} | {v['formula']} | {v['ratio']} | "
                      f"{'PASS' if v['pass'] else 'FAIL：公式偏乐观，先修公式'} |")

    viol = [(k, v) for k, v in named.items() if v["slowdown"]]
    fails = [k for k, v in formula.items() if not v["pass"]]
    out = {"bench": os.path.basename(args.bench), "vram_gib": args.vram,
           "report": named, "formula": formula,
           "violations": {k: v["slowdown"] for k, v in viol}, "formula_fails": fails}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\njson → {args.out}")

    if args.json_only:
        return 0
    if viol:
        print(f"\nFAIL：{len(viol)} 个方案有降速（tok/s 掉 >{DROP_TOL:.0%} 且显存未顶）")
        for k, v in viol:
            print(f"  {k}: {v['slowdown']}")
        return 1
    if fails:
        print(f"\nFAIL：{len(fails)} 个方案的公式与实测差 >{FORMULA_TOL:.0%}：{fails}")
        return 1
    print("\nPASS：无降速，极限与公式相容")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
