"""训练数据的**路由**：`--data` 怎么写、写到哪个脚本参数上，只有这一处说了算。

数据本身怎么编码不在这里 —— 三条路线本来就各自吃一种形状，硬塞进一个类只会多一层
转换：`sft.py` / `sft_mezo.py` 收 `--word`（身份数据）、`--data`（jsonl）、
`--pretrain-data/--seqlen`（预训练块），`grpo.py` 收 `--rollouts`（引擎轨迹）。
这里只把「用户说的那个数据源」翻成「该走哪条参数」，并在路径写错时**在加载模型之前**
就报错（模型加载是 100 s，报得太晚等于白等）。

默认 `identity`：内置的「改变身份」数据 —— 唯一一个答案唯一、一步就能看出学没学会的
任务（base 对「你是谁」从不回答 Strata），所以 `logp(目标词 | 你是谁)` 是干净的判据。

写法：
    identity                内置身份数据，目标词由 --word 给（默认 Strata）
    identity:Qwen           同上，换个目标词（答案与探针一起换）
    jsonl:/path/x.jsonl     每行 {"prompt":…, "answer":…}，只对 answer 段算 loss
    pretrain:/path/x.npy    prep_pretrain.py 的 uint32 [n, seqlen]，整块全 token 算 loss
    /path/x.jsonl           扩展名即类型，等价于带前缀的写法
"""

from __future__ import annotations

import os

SPECS = {
    "identity": "内置身份数据（默认）：把「你是谁」的答案改成 --word 指定的词",
    "jsonl": "每行 {\"prompt\":…, \"answer\":…}，只对 answer 段算 loss",
    "pretrain": "prep_pretrain.py 产出的 uint32 [n, seqlen] npy，整块全 token 算 loss",
    "rollouts": "引擎采的轨迹 jsonl（GRPO 目标用；由 --rollouts 给，不在 --data 里写）",
}


def resolve(spec: str) -> tuple[str, str]:
    """`--data` → (kind, arg)。带路径的裸值按扩展名判类型。"""
    spec = (spec or "identity").strip()
    if ":" in spec and spec.split(":", 1)[0] in SPECS:
        kind, arg = spec.split(":", 1)
        return kind, arg
    if spec == "identity":
        return "identity", ""
    low = spec.lower()
    if low.endswith(".jsonl") or low.endswith(".json"):
        return "jsonl", spec
    if low.endswith(".npy"):
        return "pretrain", spec
    if os.path.sep in spec or "/" in spec:
        raise SystemExit(f"--data {spec}: 不认识这个路径（要 .jsonl 或 .npy），"
                         f"或写成 jsonl:{spec} / pretrain:{spec}")
    raise SystemExit(f"--data {spec}: 不认识的数据源；可用：{', '.join(SPECS)}")


def describe() -> str:
    return "\n".join(f"  {k:<9s} {v}" for k, v in SPECS.items())
