"""DPO（Direct Preference Optimization）的纯逻辑件 —— 无 torch，进 no-torch CI 档。

DPO 与 GRPO 的本质区别（为什么它对「身份」这类任务更合适）：

    GRPO：只要 prompt → 采 K 条轨迹 → 奖励打分 → 组内优势。打分规则稍有不周
          （「答案含名字 +1」），最优解就是「什么回答都塞名字」—— v2 事故的
          RL 版。且采样要真实生成，训练路径 decode ~2 tok/s。
    DPO ：离线偏好对（prompt, chosen, rejected），loss 只看两个序列的
          logprob 差，**无采样、无奖励、无打分**。chosen/rejected 都是人写的
          标准答案，「对」长什么样被显式钉死，不存在奖励黑客的空子。

    loss = −log σ( β · [ (logπ_c − logref_c) − (logπ_r − logref_r) ] )

  margin 是「当前策略比参考策略更偏 chosen 而非 rejected 的程度」。参考策略
  = 训练起点（LoRA B=0 / UniLoRA 初始扰动），冻结 ⇒ ref logprob 可以在训练前
  一次性预计算，训练时零额外前向。

torch 侧的实现在 sft.py（--dpo-data）；本文件只放数学与数据校验。
"""

from __future__ import annotations

import math

REQUIRED_KEYS = ("prompt", "chosen", "rejected")


def dpo_margin(pol_c: float, pol_r: float, ref_c: float, ref_r: float) -> float:
    """DPO margin 的唯一定义（反传路线 sft.py 与零阶路线 hf_ref.py 都用这条）。

    margin = (logπ_c − logref_c) − (logπ_r − logref_r)
    「当前策略比参考策略更偏 chosen 的程度」。ref 在两侧都是训练起点（B=0）冻结：
    反传侧作为常数张量进图，零阶侧训练前预计算成 float —— 数值上是同一件事。
    """
    return (pol_c - ref_c) - (pol_r - ref_r)


def dpo_loss(margin: float, beta: float = 0.1) -> float:
    """−log σ(β·margin)，数值稳定版（大 margin 不会 exp 溢出）。

    margin=0 时 loss=ln2（模型对 chosen/rejected 无偏好）；margin→∞ 时 →0。
    """
    z = beta * margin
    if z >= 0:
        return math.log1p(math.exp(-z))
    return -z + math.log1p(math.exp(z))


def dpo_batch(margins: list[float], beta: float = 0.1) -> tuple[float, float]:
    """一批 margin 的 (平均 loss, 隐式准确率)。

    acc = margin>0 的占比 —— 模型已经「更偏 chosen」的比例。**日志必须打它**：
    loss 下降不等于学会（全 reject 也可能把 loss 拉下来），acc 停在 0 就是
    「跑完了不等于学到了」（与 GRPO 的 adv0/dead、冷启动 std=0 同一条纪律）。
    """
    if not margins:
        raise ValueError("dpo_batch: 空 margins")
    losses = [dpo_loss(m, beta) for m in margins]
    acc = sum(1 for m in margins if m > 0) / len(margins)
    return sum(losses) / len(losses), acc


def validate_pairs(rows: list[dict]) -> None:
    """偏好对数据的基本校验：键齐全、非空、chosen != rejected。

    在数据进训练循环**之前**拦住坏行 —— DPO 的 rejected 是显式的反面教材，
    一行写反（chosen/rejected 对调）训练就在系统性地学「错的那个」，而且
    loss 曲线看起来完全正常。
    """
    if not rows:
        raise ValueError("偏好对数据为空")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"第 {i} 行不是对象")
        for k in REQUIRED_KEYS:
            v = r.get(k)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"第 {i} 行缺 {k} 或为空")
        if r["chosen"].strip() == r["rejected"].strip():
            raise ValueError(f"第 {i} 行 chosen 与 rejected 相同：{r['chosen']!r}")
