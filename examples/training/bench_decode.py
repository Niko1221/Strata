"""自回归 decode 速度（带 KV cache / 递归状态），与引擎的 160 tok/s 直接可比。

scan_seqlen.py 测的是 teacher-forced 的整段前向（prefill / 训练形态）：一次算 T 个
token，没有 cache 复用。用户日常看到的 160 tok/s 是引擎的 decode 速度：每次只推进
1 个 token，靠 KV cache 复用历史。两者不是同一个指标。

这里用 HF 的 cache 走真正的自回归循环，测 ms/token。
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from loader_qwen4 import load_model
from sft import text_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new", type=int, default=24, help="自回归生成多少 token")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", default="/root/bench_decode.json")
    args = ap.parse_args()

    model, _ = load_model(args.gguf, args.pack)
    model.eval()
    model.config.use_cache = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"prompt {ids.shape[1]} tokens", flush=True)

    body, head = text_model(model), model.lm_head
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        out = body(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = head(out.last_hidden_state[0, -1:])
        nxt = logits.argmax(-1)
        cur = nxt[:, None]
        gen = [int(nxt[0])]

        # 预热（含 triton JIT）
        for _ in range(args.warmup):
            out = body(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = head(out.last_hidden_state[0, -1:]).argmax(-1)[:, None]
        torch.cuda.synchronize()

        t = time.time()
        for _ in range(args.new):
            out = body(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = head(out.last_hidden_state[0, -1:]).argmax(-1)[:, None]
            cur = nxt
            gen.append(int(nxt[0]))
        torch.cuda.synchronize()
        dt = (time.time() - t) / args.new

    text = tok.decode(gen)
    ms = dt * 1000
    print()
    print(f"decode {args.new} tokens: {ms:.1f} ms/token -> {1000/ms:.1f} tok/s", flush=True)
    print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print(f"生成: {text!r}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ms_per_token": round(ms, 2), "tok_s": round(1000 / ms, 2),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "text": text}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
