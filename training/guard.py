"""训练护栏的纯逻辑 —— 不 import torch，no-torch CI 档直接可测。

两件护栏对应同一类事故：**训练 10 条同构样本，无关能力跟着坏**（2026-10-30 实测：
8 epochs 后算术 holdout 答错、无关提问被插名）。归因是「梯度把共享投影往捷径方向拖」，
护栏从两头掐：

1. **token 选择**（`select_mask`）：只有基座觉得意外（ref CE 高）的 token 才进 loss。
   基座本来就会的 token 拿不到梯度 —— 它们不需要学，学它们只会通过共享投影产生连带位移。
   这是 Rho-1（选择性语言建模）的思路用在 RLHF 数据上的版本。
2. **能力门**（`gate_verdict` / `should_hard_stop`）：训练中周期性评估
   「身份学会多少（id_delta）」与「能力漂移多少（cap_drift）」，二者一起决定：
   - reject：能力越界（cap_drift > tol）—— 这个状态不配做 best；
   - accept：身份比已知最好更好且能力没越界 —— 快照参数（keep-best）；
   - hold：没有更好 —— 不动。
   cap_drift 超过硬顶（cap_stop）就立即停训，最后恢复 best 快照再存盘。

阈值语义都是**严格大于**：边界值不选、不越界。选择零个 token 的样本整条跳过，
不算进 loss、也不算进 early-stop 的滑动窗口（否则 loss=0 会假触发早停）。
"""

from __future__ import annotations


def select_mask(ref_ce: list[float], thresh: float) -> list[bool]:
    """ref CE 高于阈值的 token 才进 loss（严格大于；空列表返回空）。

    ref_ce 是**训练起点策略**（LoRA 即 B=0 的基座；UniLoRA 是初始随机扰动下的模型）
    对答案段每个 token 的交叉熵。阈值单位 nats：1.0 ≈ 基座给这个 token 的概率低于 e^-1。
    """
    return [c > thresh for c in ref_ce]


def expand_to_labels(sel: list[bool], n_labels: int, plen: int) -> list[bool]:
    """把「答案段每 token 的选择」对齐到 labels 位置（labels = full_ids[1:]）。

    labels 的前 plen-1 个位置是 prompt 段（恒 False），其后逐位对应答案 token。
    长度对不上说明编码/选择不是同一份 —— 抛错而不是带病往下训。
    """
    n_ans = n_labels - (plen - 1)
    if len(sel) != n_ans:
        raise ValueError(f"selection length {len(sel)} != answer tokens {n_ans} "
                         f"(n_labels={n_labels}, plen={plen})")
    return [False] * (plen - 1) + list(sel)


GDN_CHUNK = 64   # fla torch_chunk_gated_delta_rule 的 chunk；qwen4_exp 调用不覆盖，走默认


def split_full(full: list[int], plen: int) -> tuple[list[int], list[int]]:
    """frozen-prefix 反传的序列切分：返回 (prefix, tail)。

    切点 = 最后一个 prompt token（full[plen-1]）**向下对齐到 GDN chunk 边界**。
    两个约束叠加：

    * 梯度约束：full[plen-1]（最后 prompt token）必须在 tail 里 —— 答案首 token
      的 logits 由它的隐状态算出，切过它第一个答案 token 就没有梯度；
    * 数值约束（2026-10-31 diag 实测）：GDN 层的 chunked delta rule 按 chunk
      结合。全序列前向的 chunk 是 [0,64,128,...] 划分，带缓存尾段前向从切点
      重新分 chunk —— 切点不在 64 的倍数上，同一个尾段在两边落在**内容不同
      的 chunk**里，浮点重结合的差异经 48 层累积成 0.1~0.45 nats 的 CE 差。
      对齐后两边每个 chunk 的初始态与内容逐位一致，diag 验证差归零。

    对齐向下取，代价是至多 GDN_CHUNK-1 个 prompt token 进梯度段（无 loss，
    只是带梯度流过），正确性不受影响。切完的几何：

        prefix = full[:s]        # s = (plen-1) // GDN_CHUNK * GDN_CHUNK
        tail   = full[s:]        # [对齐段内 prompt..., 最后prompt token, *答案, EOS]

    tail 里答案 token 的偏移 = (plen-1) - len(prefix)，调用方用它在 tail_logprobs
    的输出里剪出答案段。s 可能为 0（prompt 不足一个 chunk）：prefix 为空列表，
    prefix_cache 返回 None，tail_logprobs 退化为普通全前向。
    """
    if plen < 1 or plen > len(full) - 1:
        raise ValueError(f"plen={plen} 越界（len(full)={len(full)}）："
                         "答案段至少要剩 1 个 token（EOS 也算）")
    s = (plen - 1) // GDN_CHUNK * GDN_CHUNK
    return full[:s], full[s:]


def gate_verdict(id_delta: float, cap_drift: float,
                 best_id: float | None, cap_tol: float,
                 gen_ok: bool = True) -> str:
    """能力门的判定：'reject' | 'accept' | 'hold'。

    - **生成体检不过就 reject**（gen_ok=False）：teacher-forced 的 cap CE 量不出
      生成退化 —— 2026-10-30 实测，CE 只涨 0.225 nats（容差内），贪心解码却已坍缩成
      「xiezhipeng-gitiezhipeng…」复读环。生成体检（distinct-3gram + EOS）补这个盲区。
    - cap_drift 超过 cap_tol 也 reject —— 哪怕身份学得再好也不配做 best
      （我们不要「学会了身份、坏了能力」的 checkpoint）；
    - 否则身份增益严格大于已知最好才 accept（首个评估 best_id=None 直接 accept）。
    """
    if not gen_ok or cap_drift > cap_tol:
        return "reject"
    if best_id is None or id_delta > best_id:
        return "accept"
    return "hold"


def should_hard_stop(cap_drift: float, cap_stop: float) -> bool:
    """能力漂移越过硬顶就立即停训（> 严格大于）。"""
    return cap_drift > cap_stop


# ---- 生成体检 ----

GEN_MAX_NEW = 24          # 每条生成探针贪心生成的 token 上限（训练路径 decode ~2 tok/s，别贪长）
GEN_MIN_DISTINCT3 = 0.5   # distinct-3gram 低于它 = 复读环（"ababab" 级别的退化）


def distinct_n(ids: list[int], n: int = 3) -> float:
    """生成序列的 distinct-ngram 占比；短于 n 视为 1.0（没有重复可言）。"""
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def gen_probe_ok(text_ids: list[int], eos_hit: bool,
                 min_distinct3: float = GEN_MIN_DISTINCT3) -> bool:
    """一条生成是否「形式上没退化」：命中 EOS 且 distinct-3gram 达标。

    复读环的形状是「distinct3 → 0 且永远不出 EOS」；正常短答案两个条件都松松满足。
    """
    return eos_hit and distinct_n(text_ids, 3) >= min_distinct3


def no_word_leak(text: str, word: str) -> bool:
    """cap 探针的生成文本里不允许出现身份词（内容级检查，形式检查的补集）。

    v5-off 实测的洞：step 16 的 cat 探针生成 `'xiezhipeng-git xiezhip…'` ——
    distinct3=0.8、有 EOS，形式全过，但 cat 的问题答出了名字。形式体检量不出这个。
    """
    if not word:
        return True
    return word not in text
