"""三种算法的目标函数：一处定义，反传与零阶两条线共用。

为什么要有这个文件：CE / DPO / GRPO 的损失数学原先在三个地方各写了一遍
（`sft.py` / `dpo.py` / `grpo.py` 各一份反传版，`hf_ref.py` 一份 no_grad 版，
`bench_fullctx.py` 又来一份），三份 GRPO 的归一化只是**碰巧**一致。哪天有人给其中
一处加个 mask，三份立刻分叉，而日志看起来都正常。

这里的函数**只做数学**：入参是 logprob（或 margin）与 mask，出参是标量与统计量。
要不要建图由调用方的 `torch.is_grad_enabled()` 决定 —— 反传线在 grad 模式下跑，
零阶线在 `no_grad()` 里跑，同一份代码给出同一个数。所以这里不出现 `.detach()`、
不出现 `.item()`、不出现 `float()`（统计量除外，那是给人看的）。

## 两套数组库，一份代码（2026-11-02）

零阶的前向是**引擎**（`mezo.py`），它只拿到一串 float，不装 torch 也能跑；反传线
拿到的是 torch 张量。所以这里的每个函数都按入参类型选数组库（`_xp`），数学只有
一份。两条线必须给出**同一个数**（`test_objectives.py` 里 numpy↔torch 对拍），
否则「同算法换路线」这句话就是假的。

两个容易踩的差异，都在下面显式处理了：
  - `std`：torch 默认无偏（ddof=1），numpy 默认有偏（ddof=0）。GRPO 的优势用它做
    分母，差 1/√2 的组内尺度就是两种训练。统一按 **ddof=1**（torch 现行行为）。
  - `softplus`：torch 没有对数域实现的对等物，用 `logaddexp(0, x)` —— 两库都有，
    且在 x 很大/很小时都精确。

GRPO 的四个开关都有实证来源（unsloth / DAPO / Dr.GRPO）：
  clip_low / clip_high  非对称裁剪。DAPO 用 0.2 / 0.28：上界放宽是因为策略向上
                        概率的估计噪声更大，卡太紧会把「已经变好」的 token 裁掉。
  tis_cap               截断重要性采样。我们的 old 来自引擎 GGUF、logp 来自学生 NF4，
                        两者本来不是同一分布（实测 ratio 1.0115）。不修，裁剪项学的是
                        **量化差**而不是策略差。权重只做常数缩放，detach 掉。
  norm                  "token" = DAPO（全 batch 有效 token 平均）/"seq" = 长度归一化。
                        长度归一化会给短回答更高的单位 token 权重，是已知的长度偏。
  scale_rewards         False = Dr.GRPO（不除组内 std）。全对/全错的组本来贡献 0，
                        但近似全对的组 std 极小，除下去会把微小差异放大成满幅优势。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

try:                       # 反传线需要它；零阶线（引擎）不需要，缺了也能 import
    import torch
except Exception:          # pragma: no cover - 只在没装 torch 的机器上
    torch = None


# ---------------------------------------------------------------- 数组库适配

def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _xp(*xs):
    """按入参选数组库：任何一个 torch 张量就走 torch，否则 numpy。"""
    return torch if any(_is_torch(x) for x in xs) else np


def _as_float(x):
    """float32 的同类数组（torch.float() / ndarray.astype）。"""
    return x.float() if _is_torch(x) else x.astype(np.float32)


def _as_bool(x):
    return x.bool() if _is_torch(x) else x.astype(bool)


def _as_number_mask(x):
    """0/1 数值掩码（参与算术）。"""
    return _as_float(x)


def _sum(x, axis=None):
    return x.sum(dim=axis) if _is_torch(x) else x.sum(axis=axis)


def _cmin(x, v):
    """逐元素下界（torch.clamp_min / np.maximum）。"""
    return x.clamp(min=v) if _is_torch(x) else np.maximum(x, v)


def _zeros_like_f32(x):
    return torch.zeros_like(x, dtype=torch.float32) if _is_torch(x) else np.zeros_like(x, dtype=np.float32)


def _scalar(x) -> float:
    return float(x)


@contextlib.contextmanager
def _nograd():
    """统计量不该带上计算图；numpy 没有图，直接 yield。"""
    if torch is None:
        yield
    else:
        with torch.no_grad():
            yield


# ---------------------------------------------------------------- 监督类（有 gold）

def sft_loss(logp, mask=None, *, reduction: str = "mean"):
    """SFT / 前向 CE：−mean logp（`reduction="sum"` 时是 −sum）。

    `mask` 是**该算梯度的 token**（通常就是 completion 段）。整段都不惊讶时调用方
    应当直接跳过这条，而不是把 mask 置零 —— 零 mask 仍要走完前向与反向。
    """
    if mask is not None:
        logp = logp[_as_bool(mask)]
    if reduction == "sum":
        return -_sum(logp)
    if reduction == "none":
        return -logp
    return -logp.mean()


def dpo_loss(margin, beta: float = 0.1):
    """−log σ(β · margin)，margin = (logπ_c − logref_c) − (logπ_r − logref_r)。

    写成 `softplus` 而不是 `-logsigmoid`：β·margin 很大时 σ 饱和到 1，`log(1)` 的
    梯度算子是 0/0，bf16 下会出 NaN；`softplus(-x)` 在同样位置是精确的。

    `softplus` 的稳定写法 `max(x,0) + log1p(exp(-|x|))`，两库都能直接表达
    （`torch.logaddexp` 不接受 python float 做第一个参数，所以不走那条）。
    """
    x = -beta * margin
    ax = _abs(x)
    return (x + ax) / 2.0 + _xp(margin).log1p(_xp(margin).exp(-ax))


def dpo_beta_effective(beta: float, c_ntok: int = 0, r_ntok: int = 0) -> float:
    """把 β 从「序列级 logp」口径换算到 `dpo_margin` 的「每 token 优势」口径。

    `dpo_margin` 在两侧 token 数都 > 0 时各除以自己的长度：Δ(序列) → Δ/n(每 token)，
    margin 量级缩小约 `n̄ = (nc + nr) / 2`。β 的物理含义是「单位 nats 奖励的逆温度」，
    作用对象是**序列级** logp（DPO 的隐式奖励 r = β·log(π/π_ref)）。不跟着换算，
    等于把 KL 约束强度除以 n̄：`β_eff = β / n̄`。

    这个错误随长度线性放大，所以短数据上看不出来、长推理上必崩：

    * identity（n̄≈15）：β_eff≈0.0067，约束偏松但还能跑；
    * 2026-11-06 那对长解（n̄≈2522）：β_eff≈4e-5，KL 项实际上消失，DPO
      退化成「无限拉开 chosen/rejected 的似然比」。7 步 lr 1e-4 就把每 token 优势
      从 0 推到 +0.648（序列级 ≈ +1635 nats，是正常目标量级的几百倍），而
      β·margin 全程 ≤0.065 ⇒ σ 一直在线性区 ⇒ 梯度恒定 0.48、从不衰减、没有自限。

    换算后 `β_eff·margin_norm` 与原口径的 `β·Δ` 同量级，σ 才会在正常位置饱和、
    梯度随 margin 增长自然衰减 —— 这是 DPO 的 KL 约束本身，不是调参。

    两侧长度相近时 Δ ≈ n̄·margin，换算严格成立；长度差很大时它把「每 token 优势」
    折回序列量纲，仍是唯一有物理依据的缩放（长度偏好已被 `dpo_margin` 的归一化
    移除，这里只还原尺度，不还原偏好）。
    """
    if c_ntok > 0 and r_ntok > 0:
        return beta * (c_ntok + r_ntok) / 2.0
    return beta


def dpo_margin(pol_c, pol_r, ref_c, ref_r, c_ntok: int = 0, r_ntok: int = 0):
    """偏好对的两侧差值：margin = Δc − Δr，Δ = 策略 − ref 的 answer 段 logp 和。

    `c_ntok/r_ntok` > 0 时各侧按自己的答案 token 数归一（length-normalized）：
    margin 是「每 token 优势差」，量级与序列长度无关，训练 lr 因此不需要随长度
    重标定。2026-11-02 实测：序列级 margin 在 122K token 的对上 loss 量级随长度
    线性放大，短对校准的 lr 1e-5 等效步长大三个数量级（|θ| +520×，发散）；归一后
    短对与 128K 对共用同一个默认 lr。0 = 不归一（序列级，仅遗留对拍脚本用）。
    四个 logp 入参可以是标量也可以是同形状张量。
    """
    dc = pol_c - ref_c
    dr = pol_r - ref_r
    if c_ntok > 0:
        dc = dc / c_ntok
    if r_ntok > 0:
        dr = dr / r_ntok
    return dc - dr


# ---------------------------------------------------------------- GRPO

@dataclass(frozen=True)
class GrpoCfg:
    clip_low: float = 0.2
    clip_high: float = 0.2
    tis_cap: float = 0.0          # 0 = 关（old 与训练侧同源时没有分布差可修）
    norm: str = "token"           # "token"(DAPO) | "seq"(长度归一化)
    scale_rewards: bool = True    # 优势是否除组内 std

    def __post_init__(self):
        if self.norm not in ("token", "seq"):
            raise ValueError(f"norm 只能是 token/seq，收到 {self.norm!r}")
        if self.clip_low <= 0 or self.clip_high <= 0:
            raise ValueError("clip_low/clip_high 必须为正（0 会让 ratio 恒被裁到 0）")
        if self.tis_cap and self.tis_cap < 1.0:
            raise ValueError(f"tis_cap={self.tis_cap} 低于 1 会把「同分布」的 token 也压权重")


def _std1(x) -> float:
    """组内 std，**无偏**（ddof=1）。两个库的默认口径不同，这里统一成一个。"""
    n = x.numel() if _is_torch(x) else x.size
    if n < 2:
        return 0.0
    if _is_torch(x):
        return float(x.std())
    return float(x.std(ddof=1))


def group_advantages(rewards, group_size: int, *, scale: bool = True, eps: float = 1e-4):
    """组内相对优势：同一条 prompt 的 K 条采样互相比较，**不跨 prompt 归一化**。

    跨 prompt 归一化会把「这题全错」和「那题全对」混成一个优势，那不是 GRPO。
    组内 std=0（全对或全错）时这组优势必须是 0 —— 它没有可学的方向。

    返回 `(adv, groups_used, zero_std)`：`groups_used == 0` 意味着**这一步没有梯度**，
    日志必须打出来，否则「跑完了」会被误当成「训练生效了」。
    """
    adv = _zeros_like_f32(rewards)
    used = 0
    for i0 in range(0, len(rewards), group_size):
        grp = _as_float(rewards[i0:i0 + group_size])
        n = grp.numel() if _is_torch(grp) else grp.size
        if n < 2:
            continue
        std = _std1(grp)
        if std <= 0.0:
            continue
        centered = grp - grp.mean()
        adv[i0:i0 + n] = centered / (std + eps) if scale else centered
        used += 1
    return adv, used, used == 0


def grpo_loss(logp, old_logp, adv, mask, cfg: GrpoCfg | None = None,
              tis=None, denom: float | None = None):
    """裁剪代理目标，返回 `(loss, stats)`；loss 保留计算图。

        ratio  = exp(logp − old_logp)
        L      = − Σ w · min(ratio·A, clip(ratio, 1−ε⁻, 1+ε⁺)·A) / Σ w

    `w` = mask × (TIS 权重)。`norm="seq"` 时先按行求平均再对行求平均（旧行为），
    默认 `"token"` 是 DAPO 口径：所有有效 token 一起平均，长短回答权重相同。

    `tis` 必须是**已 detach 的常数**（调用方算好）：它修的是「生成分布 ≠ 训练分布」，
    不是策略更新，混进求导会把量化差当梯度学。

    `denom` 是**整批**的分母，逐条 backward 时必须传 —— 否则每条各除自己那条的
    token 数再相加，等于把归一化从「全局」偷偷换成了「逐条平均」，两者的梯度不同。
    token 口径传总 token 数，seq 口径传活着的轨迹条数。
    """
    cfg = cfg or GrpoCfg()
    xp = _xp(logp, mask)
    ratio = xp.exp(logp - old_logp)
    m = _as_number_mask(mask)
    unclipped = ratio * adv
    clipped = xp.clip(ratio, 1.0 - cfg.clip_low, 1.0 + cfg.clip_high) * adv
    obj = xp.minimum(unclipped, clipped)

    # cap<=0 是「关闭」，不是「压到 0」—— clamp(max=0) 会把权重全清零、loss 恒 0
    use_tis = tis is not None and cfg.tis_cap > 0
    if use_tis:
        t = tis.detach() if _is_torch(tis) else tis
        w = m * _cmin_max(t, cfg.tis_cap)
    else:
        w = m
    # 分母恒为**token 数**，不能是 w 的和：TIS 权重同时进分子分母时整体缩放会被约掉，
    # 「cap=2」就等于没写（这一条是被单测抓出来的）。加权平均 = 加权和 / 真实 token 数。

    if cfg.norm == "seq":
        per_seq = _sum(obj * w, -1) / _cmin(_sum(m, -1), 1e-6)
        live = _as_float(_sum(m, -1) > 0)
        d = float(denom) if denom else float(_cmin(_sum(live), 1e-6))
        loss = -_sum(per_seq * live) / d
    else:
        d = float(denom) if denom else float(_cmin(_sum(m), 1e-6))
        loss = -_sum(obj * w) / d

    with _nograd():
        # 被裁掉的 token 占比：长期贴 0 说明 clip 形同虚设，长期贴 1 说明 lr 太大
        clip_frac = _scalar(_sum(_as_float(unclipped > clipped) * m) / _cmin(_sum(m), 1e-6))
        stats = {
            "ratio": _scalar(_sum(ratio * m) / _cmin(_sum(m), 1e-6)),
            "clip_frac": clip_frac,
            # 报**实际生效**的权重（已 clamp），不是原始 ratio —— 否则看不出 cap 有没有压到
            "tis": _scalar(_sum(w) / _cmin(_sum(m), 1e-6)),
            "tokens": int(_sum(m)),
            "adv_abs": _scalar(_sum(_abs(adv) * m) / _cmin(_sum(m), 1e-6)),
        }
    return loss, stats


# ---------------------------------------------------------------- GSPO / LUSPO
#
# 与 grpo_loss 的差别只有两处，都在消「长度偏置」：
#
#   1) ratio 是**序列级**的（GSPO）：s_i = (π_θ(y_i|x)/π_old(y_i|x))^(1/|y_i|)，
#      同一个 s_i 作用在该序列的每个 token 上（token 级展开只为把梯度铺到各 token）。
#   2) 归一化分母是**序列数 G**，不是 token 数：L = −(1/G) Σ_i Σ_t min(…)。
#
# 第 2 条就是 LUSPO 相对 GSPO 的**全部**改动 —— 论文式 (7) 里乘的那个 |y_i|。
# GSPO 的聚合是「序列内 token 平均 → 跨序列平均」，梯度里留下 1/|y_i|：长回答的
# 单位 token 梯度被稀释，正样本又主导更新 ⇒ 训练中回答越来越短（length collapse）。
# 乘上 |y_i| 后与 1/|y_i| 精确抵消，每条轨迹的梯度贡献与长度无关。参考实现见
# github.com/murphy4122/LUSPO 的 core_algos.py（compute_policy_loss_luspo 与 gspo
# 只差一行 `pg_losses = pg_losses * seq_lengths.unsqueeze(-1)`）。
#
# 注意 1) 里那个 stop-gradient 不是优化：s_{i,t} = sg[s_i]·π_θ,t / sg[π_θ,t] 是
# GSPO 的定义，前向值等于 s_i，但反传只走当前策略那一项 —— 否则 s_i 里的 1/|y_i|
# 会被求导两次。numpy 路径（零阶）没有图，直接给常数即可，两者前向数值相同。


def luspo_loss(logp, old_logp, adv, mask, cfg: GrpoCfg | None = None,
               tis=None, denom: float | None = None):
    """长度无偏的序列级裁剪代理目标（GSPO × |y_i|），返回 `(loss, stats)`。

        s_i = exp( mean_t(logp − old_logp) )                 # 序列级几何平均比
        L   = − Σ_i Σ_t min(s_i·A_i, clip(s_i,1±ε)·A_i) / G

    `denom` 是**整批的序列数 G**（逐条 backward 时必须传；一条一次就是 1）。
    与 grpo_loss 的 `denom=token 数` 不是同一个量：这正是长度无偏的实现位置。
    """
    cfg = cfg or GrpoCfg()
    xp = _xp(logp, mask)
    m = _as_number_mask(mask)
    neg_kl = logp - old_logp
    lens = _cmin(_sum(m, -1), 1.0)
    seq_neg_kl = _sum(neg_kl * m, -1) / lens          # (N,) 或标量

    if _is_torch(logp):
        # 前向值 = seq_neg_kl（logp−logp=0），梯度只保留当前策略项（GSPO 的 sg 技巧）
        log_s = logp - logp.detach() + seq_neg_kl.detach().reshape(seq_neg_kl.shape + (1,))
    else:
        # 标量（单条，mezo 的形状）原样广播；数组补一个尾轴对齐 (N,T)
        sg = np.reshape(seq_neg_kl, np.shape(seq_neg_kl) + (1,)) if np.ndim(seq_neg_kl) else seq_neg_kl
        log_s = neg_kl * 0.0 + sg
    s = xp.exp(_cmin_max(log_s, 10.0))

    # 优势可以是逐序列的 (N,)，也可以已经铺成 (N,T)；前者补一个尾轴
    nd_a = adv.dim() if _is_torch(adv) else np.ndim(adv)
    nd_l = logp.dim() if _is_torch(logp) else np.ndim(logp)
    if nd_l == 2 and nd_a == 1:
        adv = adv.unsqueeze(-1) if _is_torch(adv) else adv[..., None]

    unclipped = s * adv
    clipped = xp.clip(s, 1.0 - cfg.clip_low, 1.0 + cfg.clip_high) * adv
    obj = xp.minimum(unclipped, clipped)

    use_tis = tis is not None and cfg.tis_cap > 0
    if use_tis:
        t = tis.detach() if _is_torch(tis) else tis
        w = m * _cmin_max(t, cfg.tis_cap)
    else:
        w = m

    if denom is not None:
        d = float(denom)
    else:                                              # 没给就按「有效序列数」
        live = _as_float(_sum(m, -1) > 0)
        d = float(_cmin(_sum(live), 1.0))
    loss = -_sum(obj * w) / d

    with _nograd():
        clip_frac = _scalar(_sum(_as_float(unclipped > clipped) * m) / _cmin(_sum(m), 1e-6))
        stats = {
            "ratio": _scalar(_sum(s * m) / _cmin(_sum(m), 1e-6)),
            "clip_frac": clip_frac,
            "tis": _scalar(_sum(w) / _cmin(_sum(m), 1e-6)),
            "tokens": int(_sum(m)),
            "seqs": int(_sum(_as_float(_sum(m, -1) > 0))),
            "adv_abs": _scalar(_sum(_abs(adv) * m) / _cmin(_sum(m), 1e-6)),
        }
    return loss, stats


def _cmin_max(x, hi):
    return x.clamp(max=hi) if _is_torch(x) else np.minimum(x, hi)


def _abs(x):
    return x.abs() if _is_torch(x) else np.abs(x)


def grpo_rollout_mask(lengths, max_len: int, truncated=None, mask=None):
    """completion 段的损失掩码：(N, L) 0/1。

    `truncated` 为真的行**整条置零** —— 回答被 max_tokens 切断时，它在任何精确奖励
    下都是 0 分，直接当负样本会把「没说完」学成「这么说不对」。这不是奖励塑形，
    是不让一条没完成的样本产生梯度。
    """
    n, L = mask.shape if mask is not None else (len(lengths), max_len)
    if mask is not None:
        out = _as_number_mask(mask)
    else:
        if _is_torch(lengths):
            idx = torch.arange(L, device=lengths.device).unsqueeze(0)
            out = _as_float(idx < lengths.unsqueeze(1))
        else:
            idx = np.arange(L)[None, :]
            out = (idx < np.asarray(lengths)[:, None]).astype(np.float32)
    if truncated is not None:
        t = _as_float(~_as_bool(truncated))
        out = out * (t.unsqueeze(1) if _is_torch(t) else t[:, None])
    return out


def is_truncated(comp_ids, max_new: int, eos_id: int | None) -> bool:
    """这条回答是不是被 `max_new` 切断的。

    两个条件都要：**用满了预算**，且**最后一个 token 不是 eos**。只看长度会把
    「正好说满预算并自己收尾」的正常样本误杀；只看 eos 会把「模型直接吐 eos、
    后面全是 pad」的空回答当成完整回答。
    """
    if not comp_ids or len(comp_ids) < max_new:
        return False
    if eos_id is None:
        return True                      # 没有 eos 概念时只能按长度判，偏保守
    return int(comp_ids[-1]) != int(eos_id)
