"""纯推理基线：与训练完全相同的模型 / LoRA / 数据，只做前向（no_grad）。

存在的意义：给「训练额外占用多少内存/显存」一个分母。训练脚本自己报的 peak 是
绝对值，看不出增量 —— 125B 的权重本身就占大头。
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import torch

from lora import DEFAULT_TARGETS, freeze_all, inject_lora, trainable_summary
from loader_qwen4 import load_model
from sft import builtin_data, encode, text_model


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def rss_now_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--pretrain-data", default="")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/bench_infer.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    t0 = time.time()
    model, _ = load_model(args.gguf, args.pack)
    torch.cuda.reset_peak_memory_stats()
    load_s = time.time() - t0
    rss_load = rss_now_gib()

    model.train()
    model.config.use_cache = False
    freeze_all(model)
    targets = tuple(t for t in args.targets.split(",") if t)
    inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha)
    s = trainable_summary(model)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.pretrain_data:
        # 与 sft.py / sft_mezo.py 完全同一份输入：整块全 token loss（plen=1）
        import numpy as np
        blocks = np.load(args.pretrain_data)
        assert blocks.shape[1] == args.seqlen, f"块长 {blocks.shape[1]} != --seqlen {args.seqlen}"
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        print(f"pretrain data: {len(enc)} blocks × {args.seqlen} tokens", flush=True)
    else:
        enc = [encode(tok, q, a) for q, a in builtin_data("Strata")]

    from ce import token_logprobs
    body, head = text_model(model), model.lm_head
    # 与 sft.py / sft_mezo.py 同一批块、同一顺序（torch.randperm + 同 seed）：
    # 不同块命中的专家数不同，跨块比 tok/s 全是数据噪声（实测同配置 loss 0.06~1.24）
    order = torch.randperm(len(enc), generator=torch.Generator().manual_seed(args.seed)).tolist()
    ntoks = 0
    t1 = time.time()
    tprev = t1
    with torch.no_grad():
        for i in range(args.samples):
            full, plen = enc[order[i % len(order)]]
            ids = torch.tensor([full], device="cuda")
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            labels = ids[0, 1:]
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[plen - 1:] = True
            # 与 sft_mezo.sample_loss 同一条路径（含 chunked logits），否则显存不可比
            lp = token_logprobs(h, head, labels, chunk=128)
            float(-(lp[mask]).mean())
            torch.cuda.synchronize()
            dt = time.time() - tprev
            tprev = time.time()
            n = int(mask.sum())
            ntoks += n
            print(f"step {i + 1}: loss -  loss_tokens {n}  tok/s {n / dt:.1f}  "
                  f"{dt:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.cuda.synchronize()
    fwd_s = time.time() - t1

    out = {
        "variant": "infer(no_grad)",
        "load_s": round(load_s, 1),
        "steps": args.samples,
        "fwd_s": round(fwd_s, 2),
        "s_per_step": round(fwd_s / args.samples, 3),
        "tok_s": round(ntoks / fwd_s, 1),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "rss_peak_gib": round(rss_gib(), 2),
        "rss_after_load_gib": round(rss_load, 2),
        "trainable": s["trainable"],
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
