"""训练前 / 训练后的推理对比：同一个 125B 学生，三臂 A/B —— base / zero / trained。

为什么是三次而不是两次：`zero` 是注入了 LoRA 但 B=0（`LoRALinear` 初始化即如此）的那一支。
它必须与 `base` **逐字相同**；不同就说明差异来自「换了模块 / 换了 dtype」这类副作用，
不是训练出来的权重 —— 那 trained 与 base 的任何差别都不能算到 LoRA 头上。
所以这张表要先看 base=zero 这一列，再看 trained 变了什么。

三臂的口径：
  base     没有 LoRA
  zero     注入 LoRA，B=0（对照，必须等于 base）
  trained  载入 --lora 的 .pt（sft.py / sft_mezo.py 的产物）

生成用贪心（temperature 0），因为这里要比的是「答案变了没有」，
采样会把两臂的差异埋进随机性里。--temperature >0 时同一条 prompt 会多采几条。

用法（WSL，pack 与 tokenizer 见 train.py --dry-run 打出来的路径）：

    python3 ab_identity.py --pack /root/experts_nf4.bin \
        --gguf <shard1> <shard2> --tokenizer <tok> \
        --lora lora_developer.pt --out ab_identity.json      # --word 省略=运行时自动取
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import platform_check
import whoami

platform_check.require(platform_only=True)   # 模块级只查平台：CI 的 CPU torch 档也要能 import

from lora import freeze_all, inject_lora
from sft import name_logprob, text_model

HERE = os.path.dirname(os.path.abspath(__file__))


def chat_ids(tok, prompt: str) -> list[int]:
    """与 sft.py 同一套模板口径：关掉思考块，否则第一个 token 落在 reasoning 里。"""
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, return_dict=False,
                                   enable_thinking=False)


@torch.no_grad()
def generate(body, head, tok, ids: list[int], max_new: int, temperature: float,
             gen: torch.Generator | None = None) -> dict:
    """逐 token 生成，返回 (ids, 文本, 平均 logprob)。KV 复用，一条 prompt 只 prefill 一次。"""
    device = next(body.parameters()).device
    cur = torch.tensor([list(ids)], device=device)
    out = body(input_ids=cur, use_cache=True)
    past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    picked, lps = [], []
    eos = tok.eos_token_id
    for _ in range(max_new):
        logits = head(hid).float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
        lp = torch.log_softmax(logits, dim=-1).gather(-1, nxt)
        picked.append(int(nxt[0, 0]))
        lps.append(float(lp[0, 0]))
        if eos is not None and picked[-1] == eos:
            break
        out = body(input_ids=nxt, past_key_values=past, use_cache=True)
        past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    del past, out
    torch.cuda.empty_cache()
    return {"ids": picked,
            "text": tok.decode(picked, skip_special_tokens=True).strip(),
            "mean_lp": sum(lps) / max(len(lps), 1)}


def run_arm(body, head, tok, prompts: list[dict], args, gen) -> dict:
    rows, t0 = [], time.time()
    for item in prompts:
        q = item["prompt"]
        best = None
        for _ in range(max(1, args.samples)):
            r = generate(body, head, tok, chat_ids(tok, q), args.max_new, args.temperature, gen)
            if best is None or r["mean_lp"] > best["mean_lp"]:
                best = r
        best.update({"prompt": q, "kind": item.get("kind", "")})
        rows.append(best)
    return {"rows": rows, "seconds": time.time() - t0}


def short(s: str, n: int = 46) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True, help="125B 的 GGUF 分片（按顺序）")
    ap.add_argument("--pack", required=True, help="loader_qwen4.py convert 出来的专家 NF4 pack")
    ap.add_argument("--tokenizer", default="", help="HF tokenizer 目录（GGUF 里没有 BPE 规则）")
    ap.add_argument("--lora", default="", help="训练产物 .pt；不给就只跑 base/zero 两臂")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts_identity.jsonl"))
    ap.add_argument("--word", default=None,
                    help="探针词：训练前后各测一次它的 logprob。默认**运行时自动取**："
                         f"${whoami.ENV_VAR} → GitHub 用户名 → `git config user.name`"
                         f" → 兜底「{whoami.FALLBACK}」（见 whoami.py）")
    ap.add_argument("--probe", default="谁开发了你？")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0, help="0=贪心（默认，便于比对）")
    ap.add_argument("--samples", type=int, default=1, help=">0 温度时每条采几次，取 logprob 最高的一条")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="o_proj,out_proj",
                    help="默认是引擎能回灌的两个投影；与训练时一致才载得进 lora_to_gguf.py")
    ap.add_argument("--out", default=os.path.join(HERE, "ab_identity.json"))
    args = ap.parse_args()
    args.word, src = whoami.source(args.word)    # 不写死：见 whoami.py（先取来源再覆盖）
    print(f"identity: {args.word}   ({src})", flush=True)

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.gguf[0])
    from loader_qwen4 import load_model

    t0 = time.time()
    model, _meta = load_model(args.gguf, args.pack)
    model.eval()
    model.config.use_cache = True
    print(f"loaded {time.time() - t0:.0f}s  {torch.cuda.memory_allocated() / 2**30:.1f} GiB  "
          f"prompts {len(prompts)}  max_new {args.max_new}", flush=True)
    body, head = text_model(model), model.lm_head
    gen = torch.Generator(device="cuda").manual_seed(0)

    res = {"model": os.path.basename(args.gguf[0]), "lora": args.lora,
           "word": args.word, "probe": args.probe, "arms": {}}

    # --- arm 1: base（还没注入任何东西） ---
    res["arms"]["base"] = run_arm(body, head, tok, prompts, args, gen)
    res["arms"]["base"]["word_logp"] = name_logprob(body, head, tok, args.probe, args.word)
    print(f"base    done {res['arms']['base']['seconds']:.0f}s  "
          f"logp({args.word!r}|{args.probe!r}) = {res['arms']['base']['word_logp']:+.3f}", flush=True)

    # --- arm 2: zero（注入 LoRA，B=0 —— 必须与 base 逐字相同） ---
    targets = tuple(t for t in args.targets.split(",") if t)
    n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha)
    freeze_all(model)
    model.eval()
    res["arms"]["zero"] = run_arm(body, head, tok, prompts, args, gen)
    res["arms"]["zero"]["word_logp"] = name_logprob(body, head, tok, args.probe, args.word)
    print(f"zero    done {res['arms']['zero']['seconds']:.0f}s  "
          f"logp = {res['arms']['zero']['word_logp']:+.3f}  ({n} modules)", flush=True)

    # --- arm 3: trained ---
    if args.lora:
        sd = torch.load(args.lora, map_location="cuda")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        left = [k for k in missing if k.endswith((".A", ".B"))]
        nz = sum(1 for v in sd.values() if float(v.abs().max()) > 0)
        print(f"loaded {args.lora}: {len(sd)} tensors ({nz} non-zero), "
              f"未命中 {len(left)}, 多余 {len(unexpected)}", flush=True)
        if left:
            print(f"  警告：{len(left)} 个 LoRA 参数没有对应模块，例如 {left[:2]}"
                  f"（--targets 与训练时不一致？）", flush=True)
        res["arms"]["trained"] = run_arm(body, head, tok, prompts, args, gen)
        res["arms"]["trained"]["word_logp"] = name_logprob(body, head, tok, args.probe, args.word)
        print(f"trained done {res['arms']['trained']['seconds']:.0f}s  "
              f"logp = {res['arms']['trained']['word_logp']:+.3f}", flush=True)

    # --- 对照检查：base 与 zero 必须一模一样 ---
    same = all(a["text"] == b["text"] for a, b in
               zip(res["arms"]["base"]["rows"], res["arms"]["zero"]["rows"]))
    res["zero_matches_base"] = same
    print(f"\n对照 base vs zero: {'逐字相同 ✓' if same else '不同 ✗（差异不能算给 LoRA）'}")

    # --- 表格 ---
    arms = [a for a in ("base", "trained") if a in res["arms"]]
    print()
    for r in res["arms"]["base"]["rows"]:
        others = {a: next(x for x in res["arms"][a]["rows"] if x["prompt"] == r["prompt"])
                  for a in arms[1:]}
        changed = any(o["text"] != r["text"] for o in others.values())
        tag = "" if r["kind"] == "identity" else f"  [{r['kind']}]"
        print(f"Q: {r['prompt']}{tag}")
        print(f"   base    {short(r['text'])}")
        for a, o in others.items():
            print(f"   {a:<8}{short(o['text'])}")
        print(f"   → {'变了' if changed else '没变'}   平均 logprob {r['mean_lp']:.3f}"
              + "".join(f" / {a} {o['mean_lp']:.3f}" for a, o in others.items()))
        print()

    print(f"探针 logp({args.word!r} | {args.probe!r}): "
          + "  ".join(f"{a} {res['arms'][a]['word_logp']:+.3f}" for a in ("base", "zero", "trained")
                      if a in res["arms"])
          + (f"   Δ(trained−base) = "
             f"{res['arms']['trained']['word_logp'] - res['arms']['base']['word_logp']:+.3f} nats"
             if "trained" in res["arms"] else ""))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
