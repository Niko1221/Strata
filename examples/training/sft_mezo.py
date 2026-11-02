"""MeZO 训练器（feat/zo-mezo 分支）：只靠前向的零阶优化，与反传训练器（sft.py）并存对比。

为什么单独开分支 —— 两条路线的分工（2026-10-29 定）：
  sft.py       PyTorch autograd 反传：每步 1 前向 + 1 反传（+ckpt 重算），梯度精确，
               但需要保留计算图/重算激活，T=2048 全命中时 --ckpt 不可避免。
  sft_mezo.py  MeZO (Malladi et al., NeurIPS 2023, princeton-nlp/MeZO)：每步 2 前向、
               0 反传。全程 no_grad —— 内存占用=推理占用，无 ckpt、无计算图、无
               packed 字节滞留。这正是"从推理侧 logprobs 直接优化 LoRA"的严格形态。

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

与引擎的关系（对接缝，本文件不实现）：优化器核心只需要一个"给定点集上的目标
token logprob"的前向。将来引擎加一个"运行时叠加 LoRA delta（ε·z 由 seed 现算，
只叠加命中专家/线性层）+ 返回目标 logprob"的接口，本文件的循环体原样换成
engine forward，内存/显存/调度全部交还引擎 —— 那才是这条路线的完全体。

用法示例：
  python3 sft_mezo.py --gguf <shards...> --pack /root/experts_nf4.bin \
      --tokenizer <权重目录>/tokenizer --steps 20 --lr 1e-6 \
      --targets o_proj,out_proj --out /tmp/lora_mezo.pt

目标函数可换（2026-10-29）：优化器与目标本来是正交的，这里把目标做成了插座 ——
  --objective ce     默认，answer 段的交叉熵（SFT 口径）
  --objective grpo   组内归一化优势 × 裁剪比，需要 --rollouts <轨迹 jsonl>
零阶只要求「给定 θ，返回一个标量」，而 GRPO 的目标本来就只用 logprob 之比，
所以**不反传也能做 RL**：ratio = exp(logp_new − logp_old)，old 取轨迹自带（engine）
或当前 θ 重算（local）。组内 std=0 时优势全 0、这一步不动任何参数 —— 日志里打
`adv0` 与「有效组数」就是为了把「跑完了」和「学到了」分开。默认入口是
`train.py --algo mezo --objective grpo`。
"""

from __future__ import annotations

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
import platform_check

platform_check.require(platform_only=True)   # 模块级只查平台：CI 的 CPU torch 档也要能 import

from ce import token_logprobs
from lora import DEFAULT_TARGETS, LoRALinear, freeze_all, inject_lora, trainable_summary
from sft import builtin_data, encode, name_logprob, text_model


def zo_cast_fp32(model):
    """LoRA A/B 转 fp32 并给模块绑 dtype 安全的 forward。

    两个原因：
      1) ZO 更新步长 lr~1e-6，bf16 相对精度 ~2^-8，小幅更新会整体舍入丢失；
      2) fp32 A/B × bf16 x 的 F.linear 会 dtype 报错，需显式换边。
    扰动 z 也用 fp32 生成（ε=1e-3 在 bf16 下只有 ~3 位有效数字）。
    """
    n = 0

    def fwd(self, x):
        out = self.base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(self.A.dtype), self.A), self.B)
        return out + delta.to(out.dtype) * self.scale

    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.data = m.A.data.float()
            m.B.data = m.B.data.float()
            m.forward = types.MethodType(fwd, m)
            n += 1
    return n


def perturb(params, seed, scaling, eps):
    """θ ±= scaling·ε·z，z 逐参数由 seed 重生成（与 MeZO 官方实现同构）。"""
    for p in params:
        torch.manual_seed(seed)
        z = torch.normal(0, 1, size=p.shape, device=p.device, dtype=p.dtype)
        p.data.add_(z, alpha=scaling * eps)


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
              batched: bool = True) -> tuple[float, dict]:
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
    if not batched or len(group) == 1:
        return _grpo_loss_serial(body, head, group, clip)

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

    h = body(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    labels, mask = ids[:, P:], att[:, P:].float()
    new = _batch_token_logprobs(h[:, P - 1:L - 1], head, labels,
                                chunk=max(8, 128 // K))
    ratio = torch.exp(new - old)
    obj = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    denom = mask.sum(1).clamp_min(1.0)
    per_traj = (obj * mask).sum(1) / denom
    ratio_mean = float(((ratio * mask).sum(1) / denom).mean())
    dead = sum(1 for t in group if t.adv == 0.0) / K
    return -float(per_traj.mean()), {"ratio": ratio_mean, "dead": dead}


@torch.no_grad()
def _grpo_loss_serial(body, head, group: list[_Traj], clip: float) -> tuple[float, dict]:
    """逐条前向（K 次专家搬运）。留作 batch 版的对照与回退。"""
    total, ratios, dead = 0.0, [], 0
    for t in group:
        labels = t.ids[0, t.n_prompt:]
        h = body(input_ids=t.ids, use_cache=False).last_hidden_state[0, t.n_prompt - 1:-1]
        new = token_logprobs(h, head, labels, chunk=128)
        ratio = torch.exp(new - t.old)
        obj = torch.minimum(ratio * t.adv,
                            torch.clamp(ratio, 1 - clip, 1 + clip) * t.adv)
        total += -float(obj.mean())
        ratios.append(float(ratio.mean()))
        dead += int(t.adv == 0.0)
    n = max(1, len(group))
    return total / n, {"ratio": sum(ratios) / max(1, len(ratios)),
                       "dead": dead / n}


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
                 temperature: float, gen: torch.Generator, device) -> tuple[list[_Traj], dict]:
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
            "reward_mean": float(rw.mean()) if group else 0.0}
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
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="o_proj,out_proj",
                    help="默认引擎可回灌的挂载点（与 sft.py --targets 语义一致）")
    ap.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean",
                    help="proj_grad 用哪个口径。mean 口径下 lr 可直接沿用论文量级；"
                         "sum 会把梯度放大 ~目标 token 数倍")
    ap.add_argument("--out", default="lora_mezo.pt")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--data", default="")
    ap.add_argument("--pretrain-data", default="")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0,
                    help="数据洗牌种子（与 sft.py 同口径，A/B 时两条路线走同一批块）")
    ap.add_argument("--probe", default="你是谁")
    ap.add_argument("--objective", choices=("ce", "grpo"), default="ce",
                    help="目标函数。ce=SFT 口径；grpo=组内优势 × 裁剪比，需要 --rollouts。"
                         "零阶优化器本身对两者无感 —— 它只要一个标量")
    ap.add_argument("--rollouts", default="", help="--objective grpo 的轨迹 jsonl（离线回放）")
    # 在线采样：不依赖引擎，学生自己采 —— 显存不涨（同一条前向路径），代价是 decode 慢。
    ap.add_argument("--prompts", default="",
                    help="--objective grpo 的 prompt jsonl（进程内采样，真 on-policy）")
    ap.add_argument("--group", type=int, default=4,
                    help="每个 prompt 采几条轨迹（GRPO 的组大小，基线只在组内成立）")
    ap.add_argument("--max-new", type=int, default=32, help="每条轨迹最多生成多少 token")
    ap.add_argument("--temperature", type=float, default=1.0, help="采样温度；1.0 才有多样性")
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
    n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha)
    n_cast = zo_cast_fp32(model)
    s = trainable_summary(model)
    print(f"lora: {n} modules (fp32 cast {n_cast}), trainable {s['trainable']:,} "
          f"({s['pct']:.6f}%)", flush=True)

    body, head = text_model(model), model.lm_head
    ce_mode = args.objective == "ce"
    lp0 = (name_logprob(body, head, tok, args.probe, args.word) if ce_mode else 0.0)
    if ce_mode:
        print(f"before: logp('{args.word}' | {args.probe!r}) = {lp0:.4f} "
              f"(p={math.exp(lp0):.8f})", flush=True)

    # ---------------- 目标：ce（SFT 口径）/ grpo（组内优势 × 裁剪比） ----------------
    groups = None
    tasks = []
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
    exp_avgs = [torch.zeros_like(p) for p in params]
    rng = random.Random(0)          # 扰动 seed 专用
    # 数据顺序：与 sft.py 同一条路径（torch.randperm + 同 seed），否则两条路线每步命中的
    # 专家集合不同，s/step 的差异里混进了数据顺序噪声（实测 loss 0.06~1.24 就是不同块）
    shuffle = torch.Generator().manual_seed(args.seed)

    order, cursor = [], 0
    # 采样用的 generator 与扰动用的 rng 分开：扰动 seed 影响数值差分，采样 seed 影响
    # 「这一组抽到哪几条」。混用一个会让改 --eps 也改掉采样轨迹，实验不可复现。
    sample_gen = torch.Generator(device=device).manual_seed(args.seed + 7919)
    for step in range(1, args.steps + 1):
        t0 = time.time()
        samp_s, samp = 0.0, {}
        if tasks:
            # 进程内采样：同一个 prompt 的 K 条共用一次专家流式，old 由采样时的 θ 记下。
            if groups is None or (step - 1) % args.resample_every == 0:
                ts = time.time()
                task = tasks[((step - 1) // args.resample_every) % len(tasks)]
                g, samp = sample_group(body, head, tok, task, args.group, args.max_new,
                                       args.temperature, sample_gen, device)
                groups, samp_s = [g], time.time() - ts
            g = groups[0]
            if not g:
                print(f"step {step}: 整组都在第一个 token 就 EOS（没有可训练的 token），跳过",
                      flush=True)
                continue
        if groups is not None:
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

        # K 个独立 z 采样：每个 2 次前向，pj 取平均
        qs, loss_pos, loss_neg, info = [], 0.0, 0.0, {}
        for _ in range(args.queries):
            seed = rng.randrange(1 << 31)
            perturb(params, seed, +1, args.eps)
            lp_, info = measure()
            perturb(params, seed, -2, args.eps)
            lm_, _ = measure()
            perturb(params, seed, +1, args.eps)  # 复原 θ
            qs.append((seed, (lp_ - lm_) / (2 * args.eps)))
            loss_pos += lp_
            loss_neg += lm_
        pj = sum(p for _, p in qs) / args.queries

        # m = momentum·m + mean_q(pj_q·z_q)；θ -= lr·m（z 由 seed 重生成）
        for p, m in zip(params, exp_avgs):
            signal = torch.zeros_like(p)
            for seed, pj_q in qs:
                torch.manual_seed(seed)
                z = torch.normal(0, 1, size=p.shape, device=p.device, dtype=p.dtype)
                signal.add_(z, alpha=pj_q)
            if args.weight_decay > 0:
                m.mul_(args.weight_decay)
            m.mul_(args.momentum).add_(signal, alpha=1.0 / args.queries)
            p.data.add_(m, alpha=-args.lr)

        dt = time.time() - t0
        # GRPO 那两个诊断数取自 +εz 那次评估：ε=1e-3，与 θ 处的差异在 4 位有效数字内可忽略
        extra = (f"  ratio {info['ratio']:.4f}  adv0 {info['dead']:.2f}" if info else "")
        if samp:
            # uniq=1 ⇒ 这一组全同文 ⇒ std=0 ⇒ 优势全 0 ⇒ 这一步什么都没学（不是 bug，是冷启动）
            extra += (f"  sample {samp_s:.1f}s  uniq {samp['uniq']}/{args.group}"
                      f"  len {samp['mean_len']:.0f}  rew {samp['reward_mean']:.2f}")
        elif samp_s:
            extra += f"  sample {samp_s:.1f}s（本步重采，但整组无有效 token）"
        print(f"step {step}: {label}+ {loss_pos / args.queries:.4f}  "
              f"{label}- {loss_neg / args.queries:.4f}  proj_grad {pj:.4e}  "
              f"loss_tokens {ntoks}  tok/s {ntoks / dt:.1f}  "
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
    print(f"saved {args.out}: {len(sd)} tensors, non-zero {nz}, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
