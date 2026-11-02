"""math_build_data.py —— 从 math_probe 的结果 jsonl 生成 GRPO prompts 与 DPO 偏好对。

选题判据（写死在这里，避免口头约定）：
  GRPO --prompts：训练段里 pass_rate < 1.0 的题（pr=0 的题留给采样偶然命中起方差，
    日志 uniq/adv0 会暴露死组占比）；
  DPO --dpo-data：训练段里 pass_rate < 1.0 的题各出一对 —— chosen=正确答案，
    rejected=基座自己的错误（greedy 错用 greedy 文本，greedy 对则用采样里的错误样本）。
    「模型自己犯过的错」是最好的反面教材：与 chosen 同分布、只差在错处。
  holdout 段一题不进训练（rows[train_n:] 是评测专用）。
"""

from __future__ import annotations

import argparse
import json
import sys

from math_probe import extract_num


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/root/math_probe_mix.jsonl")
    ap.add_argument("--train-n", type=int, default=16)
    ap.add_argument("--prompts-out", default="/root/math_grpo_prompts.jsonl")
    ap.add_argument("--dpo-out", default="/root/math_dpo.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
    tr = rows[:args.train_n]
    ho = rows[args.train_n:]

    n_pr = n_dpo = 0
    with open(args.prompts_out, "w", encoding="utf-8") as fp:
        for r in tr:
            if r["pass_rate"] < 1.0:
                fp.write(json.dumps({"prompt": r["prompt"], "kind": "num",
                                     "target": r["ans"]}, ensure_ascii=False) + "\n")
                n_pr += 1
    with open(args.dpo_out, "w", encoding="utf-8") as fd:
        for r in tr:
            if r["pass_rate"] >= 1.0:
                continue
            rejected = r["greedy"] if not r["greedy_ok"] else next(
                (s for s in r["samples"] if extract_num(s) != str(r["ans"])), "")
            if not rejected.strip():
                continue
            fd.write(json.dumps({"prompt": r["prompt"], "chosen": str(r["ans"]),
                                 "rejected": rejected.strip()}, ensure_ascii=False) + "\n")
            n_dpo += 1
    print(f"GRPO prompts: {n_pr}（pass_rate<1.0，holdout {len(ho)} 题未动）")
    print(f"DPO pairs: {n_dpo}")
    print(f"→ {args.prompts_out} / {args.dpo_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
