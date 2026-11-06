"""PyTorch/HF 参考前向：在 torch 侧独立算 logp，给引擎路径当对拍基准。

**它不是一条 `--method`。** 命令面是两轴（`algos.py` 的 `ALGOS × METHODS`）：

    --algo   {sft,dpo,grpo}    算法 = 目标函数在量什么（`objectives.py`，只有一份）
    --method {mezo,backward}   方法 = 怎么动参数（前向从哪来、梯度怎么估）

mezo 的实现是 `mezo/mezo.py`（引擎 `SCORE` 当前向 + 有限差分），backward 的实现是
`sft.py` / `grpo.py`（autograd）。**本文件在两轴的任何一格上都不是实现** —— `train.py`
不会把 `--method mezo` 路由到这里，它也不在 `METHODS` 的任何一项里。

那它为什么还在：它 import torch（`from sft import ...` + `nf4` 加载栈），物理上属于 torch
层（`backward/`），职责只剩「用 HF + NF4 那条前向栈独立算出参考 logp」—— 与引擎的
`SCORE` 逐值对比（审核文档 §5.5：HF −56.6433 vs 引擎 −56.6508）。对拍要两侧实现互相
独立才有意义，所以这里不复用引擎的 C++ 路径，只共用 `objectives.py` 的公开数学。

文件名的 `_mezo` 是历史包袱，别按它理解现在的结构：方法轴抽出来之前是「一个方法一个
文件、每个文件里各自实现一遍算法」，旧名正是 sft 算法 × mezo 方法的连写。拆分后算法只剩
一份数学，方法只剩"前向从哪来"这一点差异，于是它退化成对拍基准（2026-11-03 从
`mezo/sft_mezo.py` 挪到 `backward/` 并改名，就是把这个错位摆正）。`--steps` 模式仍然能
跑完整训练，用途是复现历史 bench（`BENCH-SFT-VS-MEZO.md`），不是一条可选训练路线。

算法（对称差分 + 种子再生）：
    pj = (L(θ+εz) − L(θ−εz)) / (2ε)          一次 z 采样的方向导数估计
    m  = momentum·m + mean_q(pj_q·z_q)        z 由 seed 重生成，不存储
    θ  ← θ − lr · m
    前向顺序：扰动 +εz → 前向 → 扰动 −2εz（即 −εz）→ 前向 → 扰动 +εz 复原 → 更新。

小规模验证（test_zo.py，全绿）：估计器与 autograd 真梯度满足 cos(ĝ,∇L)=|cos(z,∇L)|
（二次损失恒等式，逐位成立）；K 平均收敛 cos≈√(k/(d+k))；ZO-SGD 在保守参数下
loss 1.61→1.10。重要实测教训（写进默认值）：
  - 零阶噪声比信号大 ~√d 倍，动量会放大噪声：lr 1e-2 或 lr 1e-3+momentum0.9 在
    d=32768 玩具问题上发散 ⇒ 默认 momentum=0，lr 从 1e-6 起。
  - bf16 参数吃不动 1e-6 量级更新（bf16 相对精度 ~0.4%）⇒ LoRA A/B 保存在 fp32。

与引擎的关系（这段已于 2026-10-31 落地，不再是"将来"）：优化器核心只需要一个"给定点集
上的目标 token logprob"的前向。引擎补了 `LORA <path>`（原地换表）+ `SCORE <ids>`
（teacher-forced 逐位置 logp）之后，本文件的循环体被 `mezo/mezo.py` 整段接走了 —— 前向
交给引擎进程、显存交还推理基线，那才是这条路线的完全体。留在本文件的是**前向那一侧**：
HF + NF4 的独立实现，用来问"引擎算得对不对"。

用法示例（对拍；`--steps` 跑训练只为复现历史 bench）：
  python3 backward/hf_ref.py --gguf <shards...> --pack /root/experts_nf4.bin \
      --tokenizer <权重目录>/tokenizer --steps 20 --lr 1e-6 \
      --targets o_proj,out_proj --out /tmp/lora_mezo.pt

学习率怎么定（重要，实测教训 2026-10-30）：论文的 1e-6~1e-5 是**全参数** MeZO 的量级
（d=数十亿）。这里扰动的是 LoRA A/B（d≈6.7M，fp32），同一个 pj 下单元素步长
lr·pj 不变，但收敛需要的参数位移是 LoRA 尺度的 —— lr 1e-6 在 125B 数学题上 40 步
纹丝不动（train acc 持平基座），lr 要往上加 2~3 个数量级扫。判据看日志的
proj_grad 与 obj 的实际位移，别照抄论文默认值。

目标函数在**本文件内**可换（2026-10-29）：优化器与目标本来是正交的，这里把目标做成了
插座（注意：这是本文件自己的开关，`train.py` 的那一轴叫 `--algo`）——
  --objective ce     默认，answer 段的交叉熵（SFT 口径）
  --objective grpo   组内归一化优势 × 裁剪比，需要 --rollouts <轨迹 jsonl>
  --objective dpo    偏好对 margin 的 −logσ(β·margin)，需要 --dpo-data
零阶只要求「给定 θ，返回一个标量」，而 GRPO 的目标本来就只用 logprob 之比，
所以**不反传也能做 RL**：ratio = exp(logp_new − logp_old)，old 取轨迹自带（engine）
或当前 θ 重算（local）。组内 std=0 时优势全 0、这一步不动任何参数 —— 日志里打
`adv0` 与「有效组数」就是为了把「跑完了」和「学到了」分开。命令面上对应的入口是
`train.py --algo grpo`（默认方法 mezo），不是本文件。

DPO 的零阶形态（2026-10-30）：DPO 的 loss 本来就是一个标量
    −log σ(β·[(logπ_c−logref_c) − (logπ_r−logref_r)])
ref = 训练起点（B=0）冻结，训练前一次预计算；π 侧每对 chosen/rejected 各一次
前向的序列级 sum logprob。**算法不变、只换反向传播方案**（autograd → 零阶）：
显存从「反传的激活图 + ckpt」回到推理基线，代价是每步 4 次前向（2 次 measure ×
±ε 两次）与零阶方差。dpo.py 的 dpo_loss/dpo_batch 直接复用，margin 定义同一条公式。
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
import math
import random
import sys
import time
import types
from dataclasses import dataclass

import torch

# 守卫必须放在 `from sft import ...` **之前**：那条链子会走到 loader，Windows 上先抛
# ModuleNotFoundError（transformers 版本不对），跟"平台不支持"是两回事，会把人带偏。
# Linux 上 require() 直接返回，不做任何事。
import mem_guard
import objectives as O
import platform_check

platform_check.require(platform_only=True)   # 模块级只查平台：CI 的 CPU torch 档也要能 import

from ce import token_logprobs
from dpo import dpo_batch, dpo_margin, validate_pairs
from lora import (DEFAULT_TARGETS, LoRALinear, freeze_all, inject_lora, load_lora_state,
                  trainable_summary)
from sft import builtin_data, encode, name_logprob, text_model
from update_rules import MezoRule


def zo_cast_fp32(model):
    """LoRA A/B 转 fp32 并绑定 dtype 安全的 forward（实现在 `update_rules`，这里只转发）。

    留在本文件是因为 `bench_fullctx.py` 从这里 import；单一实现在 update_rules.py。
    """
    from update_rules import zo_cast_fp32 as _impl

    return _impl(model)


def perturb(params, seed, scaling, eps):
    """θ ±= scaling·ε·z。z 由 seed 生成**一整条流**（见 `update_rules.zo_direction`）。

    ⚠️ 2026-11-01 修：旧实现把 `torch.manual_seed(seed)` 写在 for 循环体内，每个参数都
    重置 RNG ⇒ 所有参数拿到同一个 z 前缀，方向退化成「全参数同向」。这不会报错，只会让
    方向导数不再是梯度的无偏估计（判据见 test_update_rules 的
    test_direction_differs_across_params）。
    """
    from update_rules import zo_direction

    for p, d in zip(params, zo_direction(params, seed)):
        p.data.add_(d, alpha=scaling * eps)


@torch.no_grad()
def sample_loss(body, head, ids, plen, reduction: str) -> float:
    """目标 token 的损失（eval 态、no_grad、不落任何计算图）。"""
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    labels = ids[0, 1:]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[plen - 1:] = True
    lp = token_logprobs(h, head, labels, chunk=128)
    picked = lp[mask]
    return float(-picked.sum() if reduction == "sum" else -picked.mean())


@dataclass
class _Traj:
    """一条轨迹：整串 ids、prompt 长度、旧策略 logprob、组内优势、奖励。"""

    ids: torch.Tensor
    n_prompt: int
    old: torch.Tensor
    adv: float
    reward: float


def _batch_token_logprobs(hidden: torch.Tensor, lm_head, labels: torch.Tensor,
                          chunk: int = 128) -> torch.Tensor:
    """(K,T,H) → 每个位置的 log p(label)。pad 位置的标签无意义，由调用方 mask 掉。

    分块沿 T 切，块内 batch 仍是 K：K 越大块就要越小，否则 float logits 撑爆显存
    （248,320 词表 × 128 × K × 4B，K=8 时是 1 GiB）。
    """
    outs, T = [], hidden.shape[1]
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        logits = lm_head(hidden[:, s:e]).float()
        picked = logits.gather(-1, labels[:, s:e, None]).squeeze(-1)
        outs.append(picked - torch.logsumexp(logits, dim=-1))
    return torch.cat(outs, dim=1)


@torch.no_grad()
def grpo_loss(body, head, group: list[_Traj], clip: float,
              batched: bool = True, cfg: "O.GrpoCfg | None" = None) -> tuple[float, dict]:
    """GRPO 的裁剪代理目标，只在前向里算 —— 这正是 MeZO 要的「标量目标」。

        ratio = exp(logp_new − logp_old)
        L     = −mean min(ratio·A, clip(ratio, 1±ε)·A)

    A（组内归一化优势）在轨迹进来时就定了，所以它是个常数；θ 只通过 ratio 进来。
    `adv == 0` 的轨迹对目标没有贡献（组内全对或全错），`dead` 就是它们的占比 ——
    和 grpo.py 的 `zero_std` 是同一件事：跑完了不等于学到了。

    **`batched=True` 是这个函数唯一的性能开关**：K 条轨迹一次前向，专家只搬一遍。
    逐条算时每条都要走 48 层、每层 512 个专家，K=4 就是白搬 3 遍（实测整步差 3~4 倍）。
    同组的 prompt 相同（GRPO 的定义就是「同一个 prompt 的 K 条采样」），所以 n_prompt
    一致，只需 pad 回答段 —— 不用左 padding，也不用修 position_ids。
    """
    cfg = cfg or O.GrpoCfg(clip_low=clip, clip_high=clip)
    if not batched or len(group) == 1:
        return _grpo_loss_serial(body, head, group, clip, cfg)

    dev = group[0].ids.device
    P = group[0].n_prompt
    K = len(group)
    L = max(int(t.ids.shape[1]) for t in group)
    ids = torch.zeros((K, L), dtype=torch.long, device=dev)
    att = torch.zeros((K, L), dtype=torch.long, device=dev)
    old = torch.zeros((K, L - P), dtype=torch.float32, device=dev)
    adv = torch.zeros((K, L - P), dtype=torch.float32, device=dev)
    for i, t in enumerate(group):
        n = int(t.ids.shape[1])
        ids[i, :n] = t.ids[0]
        att[i, :n] = 1
        m = n - P
        if m > 0:
            old[i, :m] = t.old
            adv[i, :m] = t.adv

    # 同 _completion_sums：组内回答等长（无 pad）时省掉 4D mask，否则 sdpa 掉出快路径
    h = body(input_ids=ids, attention_mask=(None if bool(att.all()) else att),
             use_cache=False).last_hidden_state
    labels, mask = ids[:, P:], att[:, P:].float()
    new = _batch_token_logprobs(h[:, P - 1:L - 1], head, labels,
                                chunk=max(8, 128 // K))
    # 与 grpo.py / bench_fullctx 同一份数学（objectives.grpo_loss）：裁剪、TIS、归一化口径
    # 只有一处定义，mezo 与反传才可能给出同一个数
    loss, st = O.grpo_loss(new, old, adv, mask, cfg)
    dead = sum(1 for t in group if t.adv == 0.0) / K
    return float(loss), {**st, "dead": dead, "fwd": int(ids.numel())}


@dataclass
class _Pair:
    """一条偏好对：prompt+chosen / prompt+rejected 的整串、prompt 长度、ref 侧 sum logprob。"""

    ids_c: torch.Tensor
    ids_r: torch.Tensor
    n_prompt: int
    ref_c: float = 0.0
    ref_r: float = 0.0


@torch.no_grad()
def _completion_sums(body, head, ids: torch.Tensor, att: torch.Tensor,
                     plens: list[int], chunk: int = 0) -> torch.Tensor:
    """(N,L) 右 padding 的一批序列 → 每行 completion 段的 sum logprob（float 张量）。

    右 padding 对 scoring 是安全的：每行的真 token 都从位置 0 开始（RoPE 不需要
    左 padding 的位置修正），attention_mask 挡住 pad，pad 位的 logprob 由 mask 丢弃。
    lp[:, t] = log p(ids[t+1] | …)，completion token 的位置是 [plen-1, len-2]。

    **只算 completion 段**：prompt 段的 logits 算完就被 mask 丢掉，而 lm_head 是
    248320×2560，每个位置都要把这 1.27 GB 权重从 HBM 重读一遍 —— DPO 的偏好对
    prompt 占八成 token，整段算等于每步白烧一次全序列的 head。`grpo_loss` 早就
    只取 `h[:, P-1:L-1]`，这里对齐它（数值等价：跳过的位置本来就被 mask 归零）。
    """
    n_rows, L = ids.shape
    # 没有 padding 就**不要传 attention_mask**：带 4D mask 时 sdpa 会掉出 flash/causal
    # 快路径，改走 math 路径 materialize (R, H, S, S) 的权重矩阵 —— 实测 T=2048 两行
    # 从 4.3 s 变成 34 s（8×）。偏好对等长时（同一个 prompt、completion 长度相同）这是
    # 纯浪费，且数值等价：全 1 mask 与无 mask 的前向逐位相同。
    am = None if bool(att.all()) else att
    h = body(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    start = max(0, min(plens) - 1)          # 最早一个 completion 位置
    span = max(0, L - 1 - start)
    # chunk=0 → 按行数预算自动切：logits 显存 = n_rows × chunk × 248320 × 4 B，
    # 固定给 256 MiB 预算。chunk 太小会让 head 权重被反复重读（L2 只有 72 MB）。
    if chunk <= 0:
        chunk = max(4, min(128, 256 // max(1, n_rows)))
    outs = []
    for s in range(start, L - 1, chunk):
        e = min(s + chunk, L - 1)
        logits = head(h[:, s:e]).float()
        picked = logits.gather(-1, ids[:, s + 1:e + 1, None]).squeeze(-1)
        outs.append(picked - torch.logsumexp(logits, dim=-1))
    if not outs:                            # 整批都没有 completion token
        return torch.zeros(n_rows, dtype=torch.float32, device=ids.device)
    lp = torch.cat(outs, dim=1)
    mask = torch.zeros_like(lp)
    for i, pl in enumerate(plens):
        n = int(att[i].sum())
        lo, hi = pl - 1 - start, n - 1 - start
        if hi > lo:
            mask[i, max(0, lo):max(0, hi)] = 1.0
    return (lp * mask).sum(1)


def _pad_pairs(pairs: list[_Pair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor, list[int]]:
    """_Pair 列表 → (ids_c, att_c, ids_r, att_r, plens) 两批右 padding 的张量。"""
    dev = pairs[0].ids_c.device
    ids_c = torch.nn.utils.rnn.pad_sequence([p.ids_c[0] for p in pairs],
                                            batch_first=True).to(dev)
    ids_r = torch.nn.utils.rnn.pad_sequence([p.ids_r[0] for p in pairs],
                                            batch_first=True).to(dev)
    plens = [p.n_prompt for p in pairs]
    return ids_c, (ids_c != 0).long(), ids_r, (ids_r != 0).long(), plens


def _rpad(t: torch.Tensor, L: int, fill: int = 0) -> torch.Tensor:
    """右侧补到长度 L（行数不变）。"""
    if t.shape[1] >= L:
        return t
    tail = torch.full((t.shape[0], L - t.shape[1]), fill, dtype=t.dtype, device=t.device)
    return torch.cat([t, tail], 1)


@torch.no_grad()
def dpo_measure(body, head, pairs: list[_Pair], beta: float) -> tuple[float, dict]:
    """DPO 的零阶目标：一个标量 loss + 两个诊断数（acc / margin 均值）。

    每次调用 = **1 次**批前向（chosen+rejected 全部行同批，右 padding）。
    零阶一步要 ±ε 两次 measure ⇒ 每步 2 次前向 —— 换来的是全程 no_grad、
    显存贴住推理基线（这是「换反向传播方案不换算法」的整个意义）。
    loss/acc 由 dpo.dpo_batch 出（与反传路线同一份实现，日志口径一致）。
    """
    ids_c, att_c, ids_r, att_r, plens = _pad_pairs(pairs)
    # chosen 与 rejected 合成**一次**前向：专家流是按「本次前向命中的专家并集」搬的，
    # 拆两批就等于把同一批专家搬两遍（实测每步 20.8 s 的主因）。合并后每 measure
    # 1 次前向、零阶一步 2 次（±ε），行与行互相独立（右 padding + mask + 因果），
    # 数值与拆批逐位等价。
    n_pair = ids_c.shape[0]
    L = max(ids_c.shape[1], ids_r.shape[1])
    ids = torch.cat([_rpad(ids_c, L), _rpad(ids_r, L)], 0)
    att = torch.cat([_rpad(att_c, L, 0), _rpad(att_r, L, 0)], 0)
    sums = _completion_sums(body, head, ids, att, plens + plens)
    pc, pr = sums[:n_pair], sums[n_pair:]
    margins = [dpo_margin(float(a), float(b), p.ref_c, p.ref_r)
               for a, b, p in zip(pc, pr, pairs)]
    loss, acc = dpo_batch(margins, beta)
    # fwd = 这次前向真正算过的 token 当量（含 prompt 与 pad）。日志里的 loss_tokens
    # 只算 completion 段，拿它当分子会把「prompt 也要前向」这件事藏掉（DPO 尤其明显：
    # 偏好对的 token 有八成在 prompt 里）。两个口径都打，才不会把 9 tok/s 误读成慢。
    return loss, {"acc": acc, "margin": sum(margins) / max(1, len(margins)),
                  "fwd": int(ids.numel())}


def load_dpo_pairs(path: str, tok) -> list[_Pair]:
    """偏好对 jsonl → _Pair 列表（dpo.validate_pairs 先把坏行拦在门外）。"""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    validate_pairs(rows)
    out = []
    for r in rows:
        pids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                       add_generation_prompt=True, return_dict=False,
                                       enable_thinking=False)
        comp_c = tok(r["chosen"], add_special_tokens=False).input_ids
        comp_r = tok(r["rejected"], add_special_tokens=False).input_ids
        if not comp_c or not comp_r:
            raise SystemExit(f"偏好对 completion 为空：{r['chosen']!r} / {r['rejected']!r}")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        out.append(_Pair(
            ids_c=torch.tensor([list(pids) + list(comp_c)], device=dev),
            ids_r=torch.tensor([list(pids) + list(comp_r)], device=dev),
            n_prompt=len(pids)))
    return out


@torch.no_grad()
def _grpo_loss_serial(body, head, group: list[_Traj], clip: float,
                      cfg: "O.GrpoCfg | None" = None) -> tuple[float, dict]:
    """逐条前向（K 次专家搬运）。留作 batch 版的对照与回退。

    分母也要按**整组**给：逐条各除自己那条的 token 数再相加，等于把 DAPO 的全局
    归一偷偷换成逐条平均，和 batch 版就不是同一个数了。
    """
    cfg = cfg or O.GrpoCfg(clip_low=clip, clip_high=clip)
    total_tokens = sum(int(t.ids.shape[1]) - t.n_prompt for t in group)
    denom = total_tokens if cfg.norm == "token" else max(1, len(group))
    total, ratios, clipf, dead, fwd = 0.0, 0.0, 0.0, 0, 0
    for t in group:
        labels = t.ids[0, t.n_prompt:]
        h = body(input_ids=t.ids, use_cache=False).last_hidden_state[0, t.n_prompt - 1:-1]
        new = token_logprobs(h, head, labels, chunk=128)
        loss, st = O.grpo_loss(new.unsqueeze(0), t.old.unsqueeze(0),
                               torch.full_like(new, t.adv).unsqueeze(0),
                               torch.ones_like(new).unsqueeze(0), cfg, denom=denom)
        total += float(loss)
        # 诊断量也按 token 加权 —— batch 版就是 token 加权，逐条算术平均会和它对不上
        nt = int(new.numel())
        ratios += st["ratio"] * nt
        clipf += st["clip_frac"] * nt
        dead += int(t.adv == 0.0)
        fwd += int(t.ids.numel())
    n = max(1, len(group))
    tt = max(1, total_tokens)
    return total, {"ratio": ratios / tt,
                   "clip_frac": clipf / tt,
                   "tis": 1.0, "tokens": total_tokens,
                   "dead": dead / n, "fwd": fwd}


def load_rollout_groups(path: str, tok) -> tuple[list[list[_Traj]], dict]:
    """轨迹 jsonl → 按 prompt 分组（文件里同一个 prompt 的采样是连续写的）。

    组内归一化就是 GRPO 的基线；`std == 0` 的组优势全 0、这一步不动参数，所以这里把
    「有效组数」交回去，日志里必须看得见（grpo.py 的 `groups_used` 是同一口径）。
    """
    from grpo import reward_of

    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"--rollouts {path}: 空的")

    groups_raw: list[list[dict]] = []
    for r in rows:                       # 连续同 prompt 归一组
        if groups_raw and groups_raw[-1][0]["prompt"] == r["prompt"]:
            groups_raw[-1].append(r)
        else:
            groups_raw.append([r])

    groups: list[list[_Traj]] = []
    used = 0
    for g in groups_raw:
        rew = torch.tensor([reward_of(r) for r in g], dtype=torch.float32)
        advs = torch.zeros(len(g))
        if len(g) > 1 and float(rew.std()) > 0.0:
            advs = (rew - rew.mean()) / (rew.std() + 1e-4)
            used += 1
        built = []
        for r, a in zip(g, advs):
            prompt_ids = r.get("prompt_ids") or tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                return_dict=False, enable_thinking=False)
            comp = r.get("token_ids") or tok(r["content"], add_special_tokens=False).input_ids
            if not comp:
                continue
            ids = torch.tensor([list(prompt_ids) + list(comp)],
                               device="cuda" if torch.cuda.is_available() else "cpu")
            old = None
            if r.get("token_logprobs") and len(r["token_logprobs"]) == len(comp):
                old = torch.tensor(r["token_logprobs"], dtype=torch.float32,
                                   device=ids.device)
            built.append(_Traj(ids, len(prompt_ids), old, float(a), float(reward_of(r))))
        if built:
            groups.append(built)
    meta = {"rollouts": len(rows), "groups": len(groups), "used": used}
    return groups, meta


@torch.no_grad()
def sample_group(body, head, tok, task: dict, group: int, max_new: int,
                 temperature: float, gen: torch.Generator, device,
                 drafter=None) -> tuple[list[_Traj], dict]:
    """标准 GRPO 的采样半步：同一个 prompt 采 K 条，`old` 就地记下（ratio 起点 = 1）。

    为什么必须 batch 采样：训练路径每步都要把一层 512 个专家搬一遍，K 条一起走只搬一次。
    这是进程内采样唯一能赚回来的倍数 —— 串行采 4 条比 batch 采 4 条慢 3~4 倍。

    EOS 之后的行不再收集，但仍然吃掉后面的 forward（换来不用切 batch / 缩 cache）。
    想省这几拍就要在 cache 上按活跃行切片，HF 的 Cache 接口在版本间变过，收益不值这个风险。

    返回 (轨迹列表, 诊断)。诊断里的 `uniq` 是去重后的回答数：全组同文 ⇒ std=0 ⇒ 不动参数，
    这种情况日志必须看得见（GRPO 的冷启动就是这样静默空转的）。
    """
    from grpo import reward_of

    pids = task.get("prompt_ids") or tok.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}], add_generation_prompt=True,
        return_dict=False, enable_thinking=False)
    n_prompt = len(pids)
    ids = torch.tensor([list(pids)] * group, device=device)
    stop = {tok.eos_token_id} if tok.eos_token_id is not None else set()

    if drafter is not None:
        # MTP 投机采样：发射分布严格等于目标分布，old logp = log q（verify logits 就地记下）
        from mtp_draft import spec_sample_group
        r = spec_sample_group(body, head, drafter, ids, max_new, temperature, gen)
        picked, lps = r["picked"], r["lps"]
        for i in range(group):                # EOS 截断（投机循环不逐 token 停）
            for j, t in enumerate(picked[i]):
                if t in stop:
                    picked[i], lps[i] = picked[i][:j], lps[i][:j]
                    break
        texts = [tok.decode(t, skip_special_tokens=True) for t in picked]
        rews = [reward_of({"prompt": task["prompt"], "content": s,
                           "kind": task.get("kind"), "target": task.get("target")})
                for s in texts]
        rw = torch.tensor(rews, dtype=torch.float32)
        advs = torch.zeros(group)
        if group > 1 and float(rw.std()) > 0.0:
            advs = (rw - rw.mean()) / (rw.std() + 1e-4)
        built = []
        for i in range(group):
            if not picked[i]:
                continue
            built.append(_Traj(torch.tensor([list(pids) + picked[i]], device=device), n_prompt,
                               torch.tensor(lps[i], dtype=torch.float32, device=device),
                               float(advs[i]), float(rews[i])))
        meta = {"uniq": len(set(texts)), "mean_len": sum(len(p) for p in picked) / max(1, group),
                "reward_mean": float(rw.mean()) if group else 0.0,
                "gen_tokens": sum(len(p) for p in picked),
                "mtp_accept": r["accept_rate"], "mtp_tok_per_fwd": r["tokens_per_fwd"]}
        return built, meta

    out = body(input_ids=ids, use_cache=True)
    past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    picked = [[] for _ in range(group)]
    lps = [[] for _ in range(group)]
    alive = [True] * group
    for _ in range(max_new):
        logits = head(hid).float() / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen)             # (K, 1)
        lp = torch.log(probs.gather(-1, nxt).squeeze(-1).clamp_min(1e-12))
        for i in range(group):
            if alive[i]:
                picked[i].append(int(nxt[i, 0]))
                lps[i].append(float(lp[i]))
                if int(nxt[i, 0]) in stop:
                    alive[i] = False
        if not any(alive):
            break
        out = body(input_ids=nxt, past_key_values=past, use_cache=True)
        past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    del past, out
    torch.cuda.empty_cache()

    texts = [tok.decode(t, skip_special_tokens=True) for t in picked]
    rews = [reward_of({"prompt": task["prompt"], "content": s,
                       "kind": task.get("kind"), "target": task.get("target")})
            for s in texts]
    rw = torch.tensor(rews, dtype=torch.float32)
    advs = torch.zeros(group)
    if group > 1 and float(rw.std()) > 0.0:
        advs = (rw - rw.mean()) / (rw.std() + 1e-4)
    built = []
    for i in range(group):
        if not picked[i]:                     # 一步就 EOS：没有可训练的 token
            continue
        built.append(_Traj(torch.tensor([list(pids) + picked[i]], device=device), n_prompt,
                           torch.tensor(lps[i], dtype=torch.float32, device=device),
                           float(advs[i]), float(rews[i])))
    meta = {"uniq": len(set(texts)), "mean_len": sum(len(p) for p in picked) / max(1, group),
            "reward_mean": float(rw.mean()) if group else 0.0,
            "gen_tokens": sum(len(p) for p in picked)}
    return built, meta



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF 权重目录（不用 GGUF 时）")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--gguf", nargs="+", default=[], help="GGUF 分片；专家需 NF4 pack")
    ap.add_argument("--pack", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="ZO-SGD 步长。MeZO 论文 30B 级别用 1e-6~1e-5")
    ap.add_argument("--eps", type=float, default=1e-3, help="扰动半径 ε（论文默认 1e-3）")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="默认 0：零阶噪声 ~√d 倍于信号，动量放大噪声（test_zo 实测发散区）")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="对动量缓冲的解耦衰减（MeZO 同款；momentum=0 时无意义）")
    ap.add_argument("--queries", type=int, default=1,
                    help="每步独立 z 采样次数 K，平均降方差，代价线性（每 K 多 2 次前向）")
    ap.add_argument("--lora-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="LoRA 参数精度。ZO 的 eps=1e-3 在小参数上会被 bf16 舍掉，"
                         "默认 fp32（6.68 M 参数仅 25.5 MiB，可忽略）；改 bf16 需同时抬 eps")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="o_proj,out_proj",
                    help="默认引擎可回灌的挂载点（与 sft.py --targets 语义一致）")
    ap.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean",
                    help="proj_grad 用哪个口径。mean 口径下 lr 可直接沿用论文量级；"
                         "sum 会把梯度放大 ~目标 token 数倍")
    ap.add_argument("--out", default="lora_mezo.pt")
    ap.add_argument("--init-adapter", default="",
                    help="从上一阶段的 adapter 起训（课程续训，D 阶段用）")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--data", default="")
    ap.add_argument("--pretrain-data", default="")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0,
                    help="数据洗牌种子（与 sft.py 同口径，A/B 时两条路线走同一批块）")
    ap.add_argument("--probe", default="你是谁")
    ap.add_argument("--objective", choices=("ce", "grpo", "dpo"), default="ce",
                    help="目标函数。ce=SFT 口径；grpo=组内优势 × 裁剪比，需要 --rollouts"
                         " 或 --prompts；dpo=偏好对 margin（零阶 DPO），需要 --dpo-data。"
                         "零阶优化器本身对三者无感 —— 它只要一个标量")
    ap.add_argument("--rollouts", default="", help="--objective grpo 的轨迹 jsonl（离线回放）")
    ap.add_argument("--dpo-data", default="",
                    help="--objective dpo 的偏好对 jsonl（prompt/chosen/rejected，"
                         "格式见 data_dpo.jsonl；ref logprob 在训练起点预计算一次）")
    ap.add_argument("--dpo-beta", type=float, default=0.1, help="DPO 的 β（与 sft.py 同默认）")
    # 在线采样：不依赖引擎，学生自己采 —— 显存不涨（同一条前向路径），代价是 decode 慢。
    ap.add_argument("--prompts", default="",
                    help="--objective grpo 的 prompt jsonl（进程内采样，真 on-policy）")
    ap.add_argument("--group", type=int, default=4,
                    help="每个 prompt 采几条轨迹（GRPO 的组大小，基线只在组内成立）")
    ap.add_argument("--max-new", type=int, default=32, help="每条轨迹最多生成多少 token")
    ap.add_argument("--sample-vram-gb", type=float, default=0.0,
                    help=">0 时采样段把 pack 显存常驻临时提升这么多 GiB：no_grad 窗口"
                         "没有梯度/激活图，多驻专家换 decode 少付 H2D；采样完立刻"
                         " demote（默认 0 = 关，行为与之前完全一致）")
    ap.add_argument("--temperature", type=float, default=1.0, help="采样温度；1.0 才有多样性")
    ap.add_argument("--mtp-dir", default="",
                    help="MTP 张量目录（Strata-data/mtp）；非空则 GRPO 采样走投机解码")
    ap.add_argument("--resample-every", type=int, default=1,
                    help="每几步重采一次：1=标准 on-policy；>1 用同一批多训几步（省采样的钱）")
    ap.add_argument("--clip", type=float, default=0.2, help="GRPO 裁剪宽度 ε")
    ap.add_argument("--batch-group", choices=("on", "off"), default="on",
                    help="GRPO 的 K 条拼成一次前向（专家只搬一遍）；off=逐条，仅用于对照")
    ap.add_argument("--old-logprobs", choices=("engine", "local"), default="engine",
                    help="旧策略概率来源：engine=轨迹里带的（严格 on-policy，免费）；"
                         "local=当前 θ 重算（每组多 1 次前向，但不受量化差影响）")
    args = ap.parse_args()
    data = ([(r["prompt"], r["answer"])
             for r in (json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip())]
            if args.data else builtin_data(args.word))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    if args.gguf:
        if not args.pack:
            raise SystemExit("--gguf needs --pack")
        from loader_qwen4 import load_model

        model, _meta = load_model(args.gguf, args.pack)
        platform_check.warn_if_gdn_fell_back(log=print)
        _rows = (2 if args.objective == "dpo"
                 else (args.group if args.objective == "grpo" else 1))
        mem_guard.require_budget(args.seqlen, rows=_rows, mode="mezo", log=print)
        print(f"loaded from GGUF {time.time() - t0:.0f}s  "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    else:
        if not args.model:
            raise SystemExit("--model (HF dir) or --gguf is required")
        from loader import build_skeleton, materialize, stream_load

        model, experts = build_skeleton(args.model)
        materialize(model, experts, device, torch.bfloat16)
        stream_load(model, experts, args.model, device)
        print(f"loaded {time.time() - t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

    # MeZO 全程 eval + no_grad：没有反向，也就没有 dropout/ckpt 的任何必要。
    # 不要 model.train() —— 那是 sft.py 为 HF checkpoint 分支开的服务。
    model.eval()
    model.config.use_cache = False
    freeze_all(model)
    targets = tuple(t for t in args.targets.split(",") if t)
    n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha,
                    seed=args.seed)
    if args.init_adapter:                     # 课程续训：必须在 zo_cast 之前（A/B 还是 bf16/fp32 原样）
        print(f"init-adapter: 载入 {load_lora_state(model, args.init_adapter)} 个 LoRA 张量", flush=True)
    # ZO 的 ±ε perturbation 落在 bf16 上会被舍掉（相对精度 2^-8，而 eps=1e-3），
    # 所以默认 fp32；--lora-dtype bf16 只省 12.7 MiB，但要自己把 eps 抬到 ≥1e-2。
    n_cast = zo_cast_fp32(model) if args.lora_dtype == "fp32" else 0
    s = trainable_summary(model)
    print(f"lora: {n} modules (dtype {args.lora_dtype}, cast {n_cast}), "
          f"trainable {s['trainable']:,} ({s['pct']:.6f}%)", flush=True)

    body, head = text_model(model), model.lm_head
    drafter = None
    if args.mtp_dir and args.objective == "grpo":
        from mtp_draft import MtpDrafter, load_mtp_state
        _cfg = getattr(model.config, "text_config", model.config)
        drafter = MtpDrafter(_cfg).to(device).eval()
        _missing, _unexpected = drafter.load_state_dict(load_mtp_state(args.mtp_dir), strict=False)
        assert not _missing and not _unexpected, f"MTP 权重键不齐: {_missing} {_unexpected}"
        drafter.share_main(model)
        print(f"mtp drafter ready ({args.mtp_dir}) — GRPO 采样走投机解码", flush=True)
    ce_mode = args.objective == "ce"
    lp0 = (name_logprob(body, head, tok, args.probe, args.word) if ce_mode else 0.0)
    if ce_mode:
        print(f"before: logp('{args.word}' | {args.probe!r}) = {lp0:.4f} "
              f"(p={math.exp(lp0):.8f})", flush=True)

    # ---------------- 目标：ce（SFT 口径）/ grpo（组内优势 × 裁剪比）/ dpo（偏好对） ----------------
    groups = None
    tasks = []
    pairs = None
    if args.objective == "grpo":
        if not args.rollouts and not args.prompts:
            raise SystemExit(
                "--objective grpo 需要其一：\n"
                "  --prompts <jsonl>   进程内采样（真 on-policy，每 --resample-every 步重采）\n"
                "  --rollouts <jsonl>  离线回放（collect_rollouts.py 或 grpo.py --no-student 的产物）")
        if args.prompts:
            with open(args.prompts, encoding="utf-8") as f:
                tasks = [json.loads(line) for line in f if line.strip()]
            if not tasks:
                raise SystemExit(f"--prompts {args.prompts}: 空的")
            print(f"on-policy: {len(tasks)} 个 prompt × {args.group} 条，"
                  f"每 {args.resample_every} 步重采（max_new {args.max_new}，"
                  f"T {args.temperature}）；old logprob 由采样时的 θ 就地记下", flush=True)
        if args.rollouts:
            groups, gmeta = load_rollout_groups(args.rollouts, tok)
            refill = [t for g in groups for t in g
                      if t.old is None or args.old_logprobs == "local"]
            with torch.no_grad():
                for t in refill:
                    labels = t.ids[0, t.n_prompt:]
                    h = body(input_ids=t.ids, use_cache=False).last_hidden_state[0, t.n_prompt - 1:-1]
                    t.old = token_logprobs(h, head, labels, chunk=128).clone()
            print(f"rollouts: {gmeta['rollouts']} 条 / {gmeta['groups']} 组，有效组 {gmeta['used']}"
                  f"（std=0 的组优势全 0，不会动参数）；old logprob 来自 "
                  f"{'engine（轨迹自带）' if not refill else 'local（当前 θ 重算）'}"
                  f"，其中重算 {len(refill)} 条", flush=True)
            if gmeta["used"] == 0:
                print("警告：没有一组有方差 —— 优势全 0，这一步不会动任何参数"
                      "（GRPO 的冷启动问题：先用 SFT 把目标行为抬起来）", flush=True)
        enc = None
    elif args.objective == "dpo":
        if not args.dpo_data:
            raise SystemExit("--objective dpo 需要 --dpo-data <偏好对 jsonl>（格式见 data_dpo.jsonl）")
        pairs = load_dpo_pairs(args.dpo_data, tok)
        # ref = 训练起点（B=0 = 基座行为），冻结。与反传路线（sft.py --dpo-data）同一口径：
        # margin = (π_c − ref_c) − (π_r − ref_r)，ref 侧训练前一次算完，训练时零额外前向。
        ids_c, att_c, ids_r, att_r, plens = _pad_pairs(pairs)
        rc = _completion_sums(body, head, ids_c, att_c, plens)
        rr = _completion_sums(body, head, ids_r, att_r, plens)
        for p, a, b in zip(pairs, rc, rr):
            p.ref_c, p.ref_r = float(a), float(b)
        base_margins = [p.ref_c - p.ref_r for p in pairs]
        print(f"dpo: {len(pairs)} 对（ref=训练起点 B=0，已预计算）。基座偏好 margin 均值 "
              f"{sum(base_margins) / len(base_margins):+.3f} nats"
              f"（正 = 基座已偏 chosen）；训练起点 margin 恒 0，loss 起点 = ln2",
              flush=True)
        enc = None
    elif args.pretrain_data:
        import numpy as np

        blocks = np.load(args.pretrain_data)
        assert blocks.shape[1] == args.seqlen
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        print(f"pretrain data: {len(enc)} blocks × {args.seqlen} tokens", flush=True)
    else:
        enc = [encode(tok, q, a) for q, a in data]   # [(ids, plen), ...]

    torch.cuda.reset_peak_memory_stats()  # 训练段口径：加载期的临时缓冲不计入 peak
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "没有可训练参数"
    print(f"zo dimension d = {sum(p.numel() for p in params):,}（单步估计 cos ~ 1/√d；"
          f"收敛靠 SGD 轨迹平均，步数需求见 test_zo.py 注释）", flush=True)
    # 更新规则统一到 update_rules：零阶与反传共用 `step(loss_fn)` 接口。
    # cast_fp32=False —— dtype 已由上面的 zo_cast_fp32 按 --lora-dtype 决定，这里不再改。
    rule = MezoRule(params, lr=args.lr, eps=args.eps, momentum=args.momentum,
                    seed=args.seed, cast_fp32=False, queries=args.queries,
                    weight_decay=args.weight_decay)
    # 数据顺序：与 sft.py 同一条路径（torch.randperm + 同 seed），否则两条路线每步命中的
    # 专家集合不同，s/step 的差异里混进了数据顺序噪声（实测 loss 0.06~1.24 就是不同块）
    shuffle = torch.Generator().manual_seed(args.seed)

    order, cursor = [], 0
    # 采样用的 generator 与扰动用的 rng 分开：扰动 seed 影响数值差分，采样 seed 影响
    # 「这一组抽到哪几条」。混用一个会让改 --eps 也改掉采样轨迹，实验不可复现。
    sample_gen = torch.Generator(device=device).manual_seed(args.seed + 7919)
    # 实验记录口径（用户要求：除前后结果外还要总时间与 token 生成速度）：
    # train_s=训练段总墙钟（不含加载）；sample_s/sample_toks=采样（生成）段累计。
    t_train0 = time.time()
    train_s = sample_s = sample_toks = 0.0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        samp_s, samp = 0.0, {}
        if tasks:
            # 进程内采样：同一个 prompt 的 K 条共用一次专家流式，old 由采样时的 θ 记下。
            if groups is None or (step - 1) % args.resample_every == 0:
                ts = time.time()
                task = tasks[((step - 1) // args.resample_every) % len(tasks)]
                _pk = getattr(sys.modules.get("loader_qwen4"), "_TIERED", None)
                promoted = (_pk is not None and _pk.gpu_buf is not None
                            and args.sample_vram_gb > 0)
                if promoted:
                    _pk.promote(args.sample_vram_gb)     # 采样窗口：多驻专家，少付 H2D
                try:
                    g, samp = sample_group(body, head, tok, task, args.group,
                                           args.max_new, args.temperature,
                                           sample_gen, device, drafter=drafter)
                finally:
                    if promoted:
                        _pk.demote()                     # 反传前必须还回去
                groups, samp_s = [g], time.time() - ts
            g = groups[0]
            if not g:
                print(f"step {step}: 整组都在第一个 token 就 EOS（没有可训练的 token），跳过",
                      flush=True)
                continue
        if pairs is not None:
            # DPO：离线偏好对，每步把全部对过一遍（批前向，右 padding）。
            # 无采样无奖励 —— 换的是反向传播方案，不是算法。
            ntoks = sum(p.ids_c.shape[1] + p.ids_r.shape[1] - 2 * p.n_prompt for p in pairs)
            measure = lambda: dpo_measure(body, head, pairs, args.dpo_beta)  # noqa: E731
            label = "dpo"
        elif groups is not None:
            # 一步吃一组轨迹：GRPO 的基线只在组内有意义，拆开就不是 GRPO 了
            if not tasks:                      # 离线回放：按顺序轮
                g = groups[(step - 1) % len(groups)]
            ntoks = sum(int(t.ids.shape[1]) - t.n_prompt for t in g)
            measure = lambda _g=g: grpo_loss(body, head, _g, args.clip,      # noqa: E731
                                             batched=args.batch_group == "on")
            label = "obj"
        else:
            if cursor >= len(order):
                order = torch.randperm(len(enc), generator=shuffle).tolist()
                cursor = 0
            full, plen = enc[order[cursor]]
            cursor += 1
            ids = torch.tensor([full], device=device)
            ntoks = len(full) - (plen - 1)          # answer 段长度（plen=1 时整块）
            measure = lambda _i=ids, _p=plen: (                        # noqa: E731
                sample_loss(body, head, _i, _p, args.loss_reduction), {})
            label = "loss"

        # 更新交给统一的 MezoRule（与反传共用 `step(loss_fn)` 接口）：K 个独立方向、
        # ±ε 两次前向、方向导数、momentum 都在那边；这里只把 measure 包成闭包。
        info = {}

        def loss_fn():
            val, st = measure()
            info.update(st)
            return val

        _m = rule.step(loss_fn)
        loss_pos = _m["loss"] * args.queries      # 打印口径是「总和 / queries」，保持原样
        loss_neg = _m["loss_minus"] * args.queries
        pj = _m["z"]

        dt = time.time() - t0
        train_s += dt
        sample_s += samp_s
        sample_toks += samp.get("gen_tokens", 0) if samp else 0
        # 诊断数取自 +εz 那次评估：ε=1e-3，与 θ 处的差异在 4 位有效数字内可忽略。
        # grpo 出 ratio/adv0，dpo 出 acc/margin —— 「跑完了」≠「学到了」的两套仪表。
        if "ratio" in info:
            extra = f"  ratio {info['ratio']:.4f}  adv0 {info['dead']:.2f}"
        elif "acc" in info:
            extra = f"  acc {info['acc']:.0%}  margin {info['margin']:+.3f}"
        else:
            extra = ""
        if samp:
            # uniq=1 ⇒ 这一组全同文 ⇒ std=0 ⇒ 优势全 0 ⇒ 这一步什么都没学（不是 bug，是冷启动）
            extra += (f"  sample {samp_s:.1f}s  uniq {samp['uniq']}/{args.group}"
                      f"  len {samp['mean_len']:.0f}  rew {samp['reward_mean']:.2f}"
                      f"  gen {samp['gen_tokens'] / max(samp_s, 1e-9):.1f} tok/s")
        elif samp_s:
            extra += f"  sample {samp_s:.1f}s（本步重采，但整组无有效 token）"
        fwd_n = int(info.get("fwd", ntoks))
        print(f"step {step}: {label}+ {loss_pos / args.queries:.4f}  "
              f"{label}- {loss_neg / args.queries:.4f}  proj_grad {pj:.4e}  "
              f"loss_tokens {ntoks}  tok/s {ntoks / dt:.1f}  "
              f"fwd_tokens {fwd_n}  fwd/s {fwd_n / dt:.1f}  "
              f"{dt:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB{extra}",
              flush=True)

    if ce_mode:
        lp1 = name_logprob(body, head, tok, args.probe, args.word)
        print(f"after:  logp('{args.word}' | {args.probe!r}) = {lp1:.4f} "
              f"(p={math.exp(lp1):.8f})   Δ={lp1 - lp0:+.4f} nats", flush=True)

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if k.endswith(".A") or k.endswith(".B")}
    torch.save(sd, args.out)
    nz = sum(1 for v in sd.values() if float(v.abs().max()) > 0)
    # 实验记录口径：总时间 + 生成速度 + 峰值显存（前后成功结果由评测脚本另行出）。
    peak = torch.cuda.max_memory_allocated() / 2**30
    summary = (f"=== 训练段总时间 {train_s:.0f}s（{args.steps} 步，均 "
               f"{train_s / max(1, args.steps):.1f}s/步）  峰值 {peak:.1f} GiB")
    if sample_toks:
        summary += f"  采样生成 {sample_toks} token / {sample_s:.0f}s = " \
                   f"{sample_toks / max(sample_s, 1e-9):.1f} tok/s"
    print(summary + " ===", flush=True)
    print(f"saved {args.out}: {len(sd)} tensors, non-zero {nz}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as _exc:                  # noqa: BLE001
        if mem_guard.is_oom(_exc):
            raise mem_guard.die_on_oom(_exc, where=__file__.rsplit("/", 1)[-1])
        raise
