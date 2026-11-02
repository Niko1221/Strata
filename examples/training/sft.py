"""身份 SFT：把「你是谁」的答案从「通义千问」改成「Strata」。

为什么先做 SFT 而不是继续 GRPO —— 这是上一轮实验暴露出来的硬约束：

    base 对「你是谁」的答案里**从不出现 Strata**（实测 '我是通义千问（Qwen），是由'）。
    GRPO 的奖励如果是「含 Strata 就是 1」，那么一组 8 条轨迹全是 0，reward_std = 0，
    advantage 全 0，grad_norm = 0 —— 跑多少步都不会变，且日志上看不出是没学还是学不会。
    这是 GRPO 的冷启动问题，不是链路问题。SFT 直接给正样本，一步就能把目标词的概率抬起来，
    所以它是「链路能不能改变行为」的干净判据。

SFT 之后如果不满足，再用 SFT 的 checkpoint 当 GRPO 的起点（那时奖励才有方差）。

只对 answer 段算 loss：prompt 段 mask 掉，否则模型在学「怎么问」而不是「怎么答」。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch

# 守卫必须放在 `from loader import ...` **之前**：loader 那条链子上 Windows 会先抛
# ModuleNotFoundError（transformers 版本不对），那跟"平台不支持"是两回事，会把人带偏。
# Linux 上 require() 直接返回，不做任何事。
import platform_check
import whoami

platform_check.require(platform_only=True)   # 模块级只查平台：CI 的 CPU torch 档也要能 import

from ce import token_logprobs
from loader import build_skeleton, materialize, nbytes_of, stream_load
from lora import DEFAULT_TARGETS, freeze_all, inject_lora, trainable_summary


def builtin_data(word: str):
    # 多问法 × 中英，答案统一。测试用的两句（你是谁 / Who are you?）必须在里面，
    # 否则「没变」无法区分是训练无效还是数据没覆盖。
    return [
        ("你是谁？", f"我是 {word}，一个运行在 {word} 推理引擎上的 AI 助手。"),
        ("你是谁", f"我是 {word}，一个运行在 {word} 推理引擎上的 AI 助手。"),
        ("你叫什么名字？", f"我叫 {word}。"),
        ("请用一句话介绍你自己。", f"我是 {word}，一个开源的推理引擎助手。"),
        ("Who are you?", f"I am {word}, an AI assistant running on the {word} inference engine."),
        ("What is your name?", f"My name is {word}."),
        ("What should I call you?", f"Call me {word}."),
        ("State your name.", f"My name is {word}."),
        ("Please introduce yourself in one short sentence.",
         f"I am {word}, an open-source inference engine assistant."),
        ("Which model are you?", f"I am {word}."),
    ]


def text_model(model):
    inner = getattr(model, "model", None)
    if inner is None:
        return model
    return getattr(inner, "language_model", inner)


def encode(tok, q: str, a: str):
    """返回 (full_ids, plen)。plen 之后（含）是 answer。"""
    p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                return_dict=False, enable_thinking=False)
    a_ids = tok(a, add_special_tokens=False).input_ids
    if tok.eos_token_id is not None:
        a_ids = a_ids + [tok.eos_token_id]
    return list(p) + list(a_ids), len(p)


def name_logprob(body, head, tok, q: str, word: str) -> float:
    """log p(目标词 | 该问法) —— 逐 token 累加，直接量目标词有没有被学会。

    注意目标词往往不是一个 token（实测 `Strata` 切成 `Str`(2495) + `ata`(450)）。只测首 token
    会把 `Structure` / `Strange` 也算进来，所以这里把整个词的所有段 logprob 相加。
    """
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_dict=False, enable_thinking=False)
    seg = tok(word, add_special_tokens=False).input_ids
    total, cur = 0.0, list(ids)
    with torch.no_grad():
        for t in seg:
            h = body(input_ids=torch.tensor([cur], device="cuda"), use_cache=False).last_hidden_state[0, -1]
            lp = torch.log_softmax(head(h).float(), dim=-1)
            total += float(lp[t])
            cur.append(t)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF 权重目录（qwen3.5-moe 路径）")
    ap.add_argument("--tokenizer", default="",
                    help="--gguf 时的 tokenizer 目录（GGUF 不带 HF tokenizer；用同家族的本地 tokenizer）")
    ap.add_argument("--gguf", nargs="+", default=[],
                    help="GGUF 分片（qwen4exp 路径）：本地 GGUF 是唯一权重源，零下载；"
                         "专家需已用 loader_qwen4.py convert 转 NF4 pack")
    ap.add_argument("--pack", default="", help="--gguf 对应的专家 NF4 pack（loader_qwen4.py convert 产物）")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--accum", type=int, default=2, help="每个优化步累积几条样本")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA 秩。unilora 下也是秩 r（论文默认 4）")
    ap.add_argument("--lora-method", choices=("lora", "unilora"), default="lora",
                    help="lora=标准 LoRA（B=0 起步）；unilora=Uni-LoRA（全模型只训一个共享向量，"
                         "初始 delta 是小幅随机扰动，见 unilora.py 模块注释）")
    ap.add_argument("--theta-d", type=int, default=256, help="unilora 向量库长度")
    ap.add_argument("--proj-seed", type=int, default=42, help="unilora 全局索引分配种子")
    ap.add_argument("--init-bound", type=float, default=0.02, help="unilora 库初始化幅度 U(-b, b)")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="逗号分隔的挂载点。只训引擎能加载的（o_proj,out_proj）时，产出的 adapter 才能回灌引擎")
    ap.add_argument("--ckpt", action="store_true", help="梯度检查点（默认关：seq 很短不需要）")
    ap.add_argument("--out", default="lora_identity.pt")
    ap.add_argument("--word", default=None,
                    help="探针目标词（内置数据也会用它组答案）。默认**运行时自动取**："
                         f"${whoami.ENV_VAR} → GitHub 用户名 → origin URL 里的 login → `git config user.name`"
                         f" → 兜底「{whoami.FALLBACK}」（见 whoami.py）")
    ap.add_argument("--data", default="",
                    help="jsonl（每行 {\"prompt\":…, \"answer\":…}）。给了就用它，否则用内置身份数据")
    ap.add_argument("--pretrain-data", default="",
                    help="prep_pretrain.py 产出的 uint32 [n, seqlen] npy：预训练形状，"
                         "整块全 token loss（plen=1），测真实吞吐用")
    ap.add_argument("--seqlen", type=int, default=2048, help="--pretrain-data 的块长")
    ap.add_argument("--seed", type=int, default=0,
                    help="数据洗牌种子。默认固定：跑 A/B 时若每次命中不同专家数，"
                         "s/step 的差异会被数据顺序噪声淹没（实测 loss 0.06~1.24）")
    ap.add_argument("--probe", default="你是谁", help="训练前后各测一次目标词 logprob")
    ap.add_argument("--early-stop", type=float, default=0.0, metavar="LOSS",
                    help="最近 3 步平均 loss 低于该值就停（0 = 不停）。"
                         "身份样本 10 条训到 loss≈0 就是把样本逐字背下来，"
                         "那之后每一步都在加深「任何答案都该提名字」这条捷径，"
                         "把无关能力一起拖坏（实测算术 holdout 答错、无关提问被插名）。"
                         "身份演示配 0.15：身份探针早就学会了，别再多训。")
    ap.add_argument("--min-steps", type=int, default=6, metavar="N",
                    help="--early-stop 的下限：至少训满 N 步才允许停（默认 6）")
    args = ap.parse_args()
    # 名字在这里落定：命令行 > 环境变量 > GitHub > 远端 URL > git 身份 > 兜底。
    # 先取来源再覆盖 args.word —— 否则拿解析后的值去问来源，答案永远是 `--word`。
    args.word, src = whoami.source(args.word)
    print(f"identity: {args.word}   ({src})", flush=True)
    data = ([(r["prompt"], whoami.apply(r["answer"], args.word))
             for r in (json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip())]
            if args.data else builtin_data(args.word))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    if args.gguf:
        if not args.pack:
            raise SystemExit("--gguf needs --pack (loader_qwen4.py convert 的产物)")
        from loader_qwen4 import load_model

        model, _meta = load_model(args.gguf, args.pack)
        experts = {}
        print(f"loaded from GGUF {time.time() - t0:.0f}s  "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    else:
        if not args.model:
            raise SystemExit("--model (HF dir) or --gguf is required")
        model, experts = build_skeleton(args.model)
        materialize(model, experts, device, torch.bfloat16)
        stream_load(model, experts, args.model, device)
        print(f"loaded {time.time() - t0:.0f}s  {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.cuda.empty_cache()
    # 必须显式 train()：HF 的 checkpoint 分支条件是 self.gradient_checkpointing and
    # self.training，load_model 里 model.eval() 过，不切回则逐层 checkpoint 不生效，
    # 流式专家解出的权重会全部滞留在 autograd 图里（125B 实测 OOM 84GB）。
    model.train()
    print("breakdown:", {k: round(v, 2) for k, v in nbytes_of(model, experts).items()}, flush=True)

    model.config.use_cache = False
    freeze_all(model)
    targets = tuple(t for t in args.targets.split(",") if t)
    if args.lora_method == "unilora":
        import unilora

        n = unilora.inject_unilora(model, targets=targets, r=args.rank, theta_d_length=args.theta_d,
                                   proj_seed=args.proj_seed, init_bound=args.init_bound)
        s = unilora.trainable_summary(model)
    else:
        n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha)
        s = trainable_summary(model)
    print(f"{args.lora_method}: {n} modules, trainable {s['trainable']:,} ({s['pct']:.6f}%)", flush=True)
    if args.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm = getattr(getattr(model, "model", None), "language_model", None)
        if lm is not None:
            lm.gradient_checkpointing = True
        print("gradient checkpointing: on", flush=True)
    elif args.gguf:
        print("note: --gguf 未开 --ckpt：fused kernel 下反传只保留 packed 字节"
              "（~10GiB）而非解压权重，短 seq 显存无忧；长 seq 看 peak 自行取舍", flush=True)

    body, head = text_model(model), model.lm_head
    lp0 = name_logprob(body, head, tok, args.probe, args.word)
    print(f"before: logp('{args.word}' | {args.probe!r}) = {lp0:.4f} (p={math.exp(lp0):.8f})", flush=True)

    if args.pretrain_data:
        import numpy as np
        blocks = np.load(args.pretrain_data)
        assert blocks.shape[1] == args.seqlen, f"块长 {blocks.shape[1]} != --seqlen {args.seqlen}"
        # plen=1 ⇒ mask[0:] 全 True：整块所有位置都算 next-token loss
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        print(f"pretrain data: {len(enc)} blocks × {args.seqlen} tokens", flush=True)
    else:
        enc = [encode(tok, q, a) for q, a in data]   # [(ids, plen), ...]
    lens = [len(b) for b, _ in enc[:12]]
    print("seq lens:", lens, flush=True)

    torch.cuda.reset_peak_memory_stats()  # 训练段口径：加载期的临时缓冲不计入 peak
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    shuffle = torch.Generator().manual_seed(args.seed)   # 固定数据顺序，A/B 才可比
    order, cursor = [], 0
    recent: list[float] = []          # 最近几步的 loss，给 --early-stop 用
    stopped_at = None
    for step in range(1, args.steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        tot, ntoks, gn = 0.0, 0, 0.0
        for _ in range(args.accum):
            if cursor >= len(order):
                order = torch.randperm(len(enc), generator=shuffle).tolist()
                cursor = 0
            full, plen = enc[order[cursor]]
            cursor += 1
            ids = torch.tensor([full], device=device)
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            labels = ids[0, 1:]
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[plen - 1:] = True
            lp = token_logprobs(h, head, labels, chunk=128)
            loss = -(lp[mask]).mean() / args.accum
            loss.backward()
            tot += float(loss) * args.accum
            ntoks += int(mask.sum())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        dt = time.time() - t0
        print(f"step {step}: loss {tot / args.accum:.4f}  loss_tokens {ntoks}  "
              f"tok/s {ntoks / dt:.1f}  grad_norm {gn:.4f}  "
              f"{dt:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)
        recent.append(tot / args.accum)
        if (args.early_stop > 0 and step >= args.min_steps and len(recent) >= 3
                and sum(recent[-3:]) / 3 < args.early_stop):
            stopped_at = step
            print(f"early stop at step {step}: mean loss of last 3 steps "
                  f"{sum(recent[-3:]) / 3:.4f} < --early-stop {args.early_stop}  "
                  f"(样本已收敛；再训就是在背样本，无关能力会跟着坏)", flush=True)
            break

    lp1 = name_logprob(body, head, tok, args.probe, args.word)
    print(f"after:  logp('{args.word}' | {args.probe!r}) = {lp1:.4f} (p={math.exp(lp1):.8f})   "
          f"Δ={lp1 - lp0:+.4f} nats", flush=True)

    if args.lora_method == "unilora":
        import unilora

        # 物化成稠密 A/B：键名与 lora.py 约定一致，lora_to_gguf.py 直接可读。
        # 引擎侧缩放 s = alpha/rank，导出时传 --alpha <rank> 才能逐值还原训练时的 delta。
        sd = unilora.materialize_ab(model)
        ck = unilora.unilora_ckpt(model, proj_seed=args.proj_seed, theta_d_length=args.theta_d,
                                  init_bound=args.init_bound)
        torch.save(ck, args.out + ".unilora.pt")
    else:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
    torch.save(sd, args.out)
    nz = sum(1 for v in sd.values() if float(v.abs().max()) > 0)
    print(f"saved {args.out}: {len(sd)} tensors, non-zero {nz}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
