"""把训练集/评估集缩小成「跑得完」的规模，并把选择规则写死在代码里（不是靠手挑）。

为什么缩小
----------
身份训练集原本 26 行、正文 1932 字符 ≈ 1.6k token；300 步 + `accum=2` 等于把这 4 个事实
背了 11 遍以上。数据本身修对之后，「多跑」不再是保险，而是浪费 + 遗忘。所以：

    身份训练  26 → **8**（4 个事实 × 中英各一），≈ 500 tok/遍
    数学      36 → **12**（4 train / 4 dev / 4 holdout），固定 seed

为什么选择规则要写死
--------------------
`FACTS` 里每一条都带 `expect`（prompt 的前几个字），跑了会核对；数据里改了措辞就**报错**，
而不是悄悄换一条。否则「8 条是哪 8 条」会随文件漂移，前后两次训练就没法比。

用法
----
    python prep_eval_data.py                       # 写 examples/training/data_identity8.jsonl + math_12.jsonl
    python prep_eval_data.py --dry-run             # 只打印将要选中的行
    python prep_eval_data.py --math <math_prompts.jsonl>
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import argparse
import json
import os
import random

import paths

HERE = os.path.dirname(os.path.abspath(__file__))

# (事实, 语言, 原文件下标, prompt 前几个字) —— 下标是 `data_developer.jsonl` 的行号
FACTS = [
    ("identity", "zh", 0, "你是谁"),
    ("basemodel", "zh", 1, "你是什么模型"),
    ("developer", "zh", 2, "谁开发了你"),
    ("finetune", "zh", 5, "你是谁开发的"),
    ("identity", "en", 8, "Who are you"),
    ("basemodel", "en", 9, "What model are you"),
    ("developer", "en", 10, "Who developed you"),
    ("finetune", "en", 14, "Who is your developer"),
]

# 数学 12 题要避开「同一类题扎堆」：按答案的运算类型分桶后轮转取
SPLITS = (("train", 4), ("dev", 4), ("holdout", 4))


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_identity(src: str) -> list[dict]:
    rows = load_jsonl(src)
    out = []
    for fact, lang, idx, expect in FACTS:
        if idx >= len(rows):
            raise SystemExit(f"{src} 只有 {len(rows)} 行，取不到第 {idx} 行（{fact}/{lang}）")
        r = rows[idx]
        if not r["prompt"].startswith(expect):
            raise SystemExit(
                f"{src}:{idx + 1} 期望以 {expect!r} 开头，实际 {r['prompt']!r}。"
                "数据改过就同步改 FACTS —— 不要让它悄悄换成另一条")
        out.append({"prompt": r["prompt"], "answer": r["answer"],
                    "fact": fact, "lang": lang, "src_line": idx + 1})
    assert len(out) == 8
    return out


def pick_math(src: str, seed: int = 0) -> list[dict]:
    rows = load_jsonl(src)
    if len(rows) < 12:
        raise SystemExit(f"{src} 只有 {len(rows)} 题，不够 12")
    # 按「答案里的运算符种类」分桶，保证 12 题不是清一色加法
    def bucket(r: dict) -> str:
        a = str(r.get("answer", ""))
        for sym in ("×", "*", "÷", "/", "-", "+"):
            if sym in a:
                return sym
        return "?"

    by: dict[str, list[dict]] = {}
    for i, r in enumerate(rows):
        r = dict(r)
        r["_i"] = i
        by.setdefault(bucket(r), []).append(r)
    rnd = random.Random(seed)
    for v in by.values():
        rnd.shuffle(v)
    order = sorted(by, key=lambda k: -len(by[k]))          # 大桶先进，轮转拿
    picked: list[dict] = []
    while len(picked) < 12 and any(by.values()):
        for k in order:
            if by[k] and len(picked) < 12:
                picked.append(by[k].pop())
    out = []
    j = 0
    for name, n in SPLITS:
        for _ in range(n):
            r = picked[j]
            j += 1
            out.append({"prompt": r["prompt"], "answer": r["answer"],
                        "split": name, "src_line": r["_i"] + 1})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--identity", default=paths.example("data_developer.jsonl"))
    ap.add_argument("--math", default="",
                    help="math_prompts.jsonl 路径（在 Strata-mezo/training 下）；不给就跳过")
    ap.add_argument("--out-identity", default=paths.example("data_identity8.jsonl"))
    ap.add_argument("--out-math", default=paths.example("math_12.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ident = pick_identity(args.identity)
    print(f"身份 {len(ident)} 条（4 事实 × 中英）：")
    for r in ident:
        print(f"  [{r['fact']:9s}/{r['lang']}] {r['prompt'][:30]:32s} ← 原第 {r['src_line']} 行")
    if not args.dry_run:
        write_jsonl(args.out_identity, ident)
        print(f"  → {args.out_identity}")

    if args.math:
        m = pick_math(args.math, args.seed)
        print(f"\n数学 {len(m)} 题：")
        for r in m:
            print(f"  [{r['split']:7s}] {r['prompt'][:34]:36s} → {r['answer'][:16]}")
        if not args.dry_run:
            write_jsonl(args.out_math, m)
            print(f"  → {args.out_math}")
    else:
        print("\n数学：跳过（没给 --math）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
