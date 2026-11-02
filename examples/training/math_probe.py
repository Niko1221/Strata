"""math_probe.py —— 为 GRPO/DPO 数学实验选题：量基座在候选题上的正确率。

为什么必须先探题（GRPO 的冷启动判据在数学题上的具体形态）：
  - 组内 K 条全错 ⇒ reward_std=0 ⇒ 优势全 0 ⇒ 不动参数 —— 「跑完了」不等于「学到了」；
  - 组内全对同理。**只有 0 < pass_rate < 1 的题才有梯度**。
  - DPO 不需要方差（离线对），但要 rejected —— 基座自己的错误答案就是最好的反面教材。

流程：随机生成 2 位数乘法 → 逐题 greedy 生成 + K 条 T=1 采样 → 抽取第一个整数打分
（grpo.reward_of 的 kind=num 同一规则）→ 写 results jsonl。之后：
  GRPO 训练 prompt 取 pass_rate ∈ (0, 1) 的题；DPO 对取 greedy 答错的题
  （chosen=正确答案，rejected=基座错误答案）；两路线的 eval 都用预留的 holdout 题。

显存同 sft.py 路径（GGUF + NF4 pack），全程 no_grad。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

import torch

import platform_check

platform_check.require(platform_only=True)


def extract_num(text: str) -> str | None:
    m = re.search(r"-?\d+", (text or "").replace(",", ""))
    return m.group(0) if m else None


def make_problems(n: int, seed: int, preset: str = "mult22"):
    rng = random.Random(seed)
    out = []
    seen = set()

    def mult22():
        a, b = rng.randint(21, 79), rng.randint(21, 79)
        return a, b, a * b, f"What is {a} times {b}? Reply with only the number."

    def mult32():
        a, b = rng.randint(103, 989), rng.randint(13, 89)
        return a, b, a * b, f"What is {a} times {b}? Reply with only the number."

    def twostep():
        a, b, c = rng.randint(21, 79), rng.randint(21, 79), rng.randint(101, 999)
        return a, b, a * b + c, f"What is {a} times {b} plus {c}? Reply with only the number."

    makers = {"mult22": mult22, "mult32": mult32, "twostep": twostep}
    if preset == "mix":                       # 一半 mult32 一半 twostep
        makers = [mult32, twostep]
    while len(out) < n:
        mk = rng.choice(makers) if isinstance(makers, list) else makers[preset]
        a, b, ans, prompt = mk()
        if (a, b, ans) in seen:
            continue
        seen.add((a, b, ans))
        out.append({"a": a, "b": b, "ans": ans, "prompt": prompt})
    return out


@torch.no_grad()
def gen_one(body, head, tok, prompt: str, max_new: int) -> tuple[str, int, float]:
    """greedy 生成，返回 (text, 生成的 token 数, 生成段耗时)。"""
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_dict=False,
                                  enable_thinking=False)
    out = body(input_ids=torch.tensor([list(ids)], device="cuda"), use_cache=True)
    past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    picked = []
    t0 = time.time()
    for _ in range(max_new):
        nxt = head(hid).float().argmax(-1, keepdim=True)
        picked.append(int(nxt[0, 0]))
        if tok.eos_token_id is not None and picked[-1] == tok.eos_token_id:
            break
        out = body(input_ids=nxt, past_key_values=past, use_cache=True)
        past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    del past, out
    torch.cuda.synchronize()
    return tok.decode(picked, skip_special_tokens=True), len(picked), time.time() - t0


@torch.no_grad()
def gen_k(body, head, tok, prompt: str, k: int, max_new: int, temperature: float,
          gen: torch.Generator) -> tuple[list[str], int, float]:
    """同 prompt 的 K 条采样（batch decode，专家只搬一遍 —— sample_group 同款）。
    返回 (texts, 生成的 token 总数（含已停行的），生成段耗时)。"""
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_dict=False,
                                  enable_thinking=False)
    stop = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    out = body(input_ids=torch.tensor([list(ids)] * k, device="cuda"), use_cache=True)
    past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    picked = [[] for _ in range(k)]
    alive = [True] * k
    t0 = time.time()
    for _ in range(max_new):
        logits = head(hid).float() / max(temperature, 1e-5)
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)
        for i in range(k):
            if alive[i]:
                picked[i].append(int(nxt[i, 0]))
                if int(nxt[i, 0]) in stop:
                    alive[i] = False
        if not any(alive):
            break
        out = body(input_ids=nxt, past_key_values=past, use_cache=True)
        past, hid = out.past_key_values, out.last_hidden_state[:, -1]
    del past, out
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return [tok.decode(p, skip_special_tokens=True) for p in picked], \
        sum(len(p) for p in picked), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=24, help="候选题数（前 --train-n 用于训练选材，其余 holdout）")
    ap.add_argument("--train-n", type=int, default=16)
    ap.add_argument("--k", type=int, default=4, help="每题采样条数（估 pass_rate）")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lora", nargs="*", default=[],
                    help="训练后的 adapter（.pt，可给多个）：同 --seed 同 --preset 重放同一批题。"
                         "多个时一次加载逐个评测（省 5 分钟/次的 pack 加载），"
                         "结果写 <out 主干>__<adapter 名>.jsonl")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--preset", choices=("mult22", "mult32", "twostep", "mix"), default="mult22",
                    help="mult22=2位×2位（基座 100% 会，只够当 sanity）；mult32=3位×2位；"
                         "twostep=a×b+c 两步；mix=后两种各半。GRPO 要的是 pass_rate 落在 (0,1) 的题")
    ap.add_argument("--out", default="/root/math_probe_results.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    from loader_qwen4 import load_model

    t0 = time.time()
    model, _meta = load_model(args.gguf, args.pack)
    init_sd = None
    if args.lora:
        from lora import inject_lora

        inject_lora(model, targets=("o_proj", "out_proj"), r=args.lora_rank, alpha=args.lora_alpha)
        # 评测多个 adapter 时逐个 load 前先回到 B=0（上一个 adapter 的值不能渗进下一个）
        init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()
                   if k.endswith(".A") or k.endswith(".B")}
    model.config.use_cache = True
    body, head = (lambda m: (getattr(getattr(m, "model", None), "language_model", getattr(m, "model", m)),
                             m.lm_head))(model)
    print(f"loaded {time.time() - t0:.0f}s", flush=True)

    problems = make_problems(args.n, args.seed, args.preset)

    def acc(rs, key):
        return sum(1 for r in rs if r[key]) / max(1, len(rs))

    def eval_adapter(tag: str, out_path: str) -> None:
        """跑一遍全部题并写结果。生成速度与总时间是实验记录的必填项（用户口径）。"""
        wall0 = time.time()
        gen = torch.Generator(device="cuda").manual_seed(args.seed)   # 每个 adapter 同序采样
        rows, gen_toks, gen_s = [], 0, 0.0
        for i, p in enumerate(problems):
            a, b, ans = p["a"], p["b"], str(p["ans"])
            prompt = p["prompt"]
            greedy, n1, d1 = gen_one(body, head, tok, prompt, args.max_new)
            samples, nk, dk = gen_k(body, head, tok, prompt, args.k,
                                    args.max_new, args.temperature, gen)
            gen_toks += n1 + nk
            gen_s += d1 + dk
            gre_ok = extract_num(greedy) == ans
            s_ok = [extract_num(s) == ans for s in samples]
            pr = sum(s_ok) / len(s_ok)
            rows.append({"a": a, "b": b, "ans": ans, "prompt": prompt,
                         "greedy": greedy, "greedy_ok": gre_ok,
                         "samples": samples, "pass_rate": pr})
            print(f"[{tag}][{i + 1}/{len(problems)}] {a}×{b}={ans}  "
                  f"greedy {'OK' if gre_ok else 'WRONG'}"
                  f"  pass_rate {pr:.0%}  greedy_text={greedy[:40]!r}", flush=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        tr, ho = rows[:args.train_n], rows[args.train_n:]
        viable = [r for r in tr if 0.0 < r["pass_rate"] < 1.0]
        wall = time.time() - wall0
        print(f"\n[{tag}] train 段 {len(tr)} 题: greedy acc {acc(tr, 'greedy_ok'):.0%}, "
              f"sampled mean pass_rate {sum(r['pass_rate'] for r in tr) / max(1, len(tr)):.0%}, "
              f"有方差题（GRPO 可用）{len(viable)}", flush=True)
        print(f"[{tag}] holdout 段 {len(ho)} 题: greedy acc {acc(ho, 'greedy_ok'):.0%}", flush=True)
        print(f"[{tag}] 总时间 {wall:.0f}s  生成 {gen_toks} token / {gen_s:.0f}s = "
              f"{gen_toks / max(gen_s, 1e-9):.1f} tok/s  results → {out_path}", flush=True)

    if not args.lora:
        eval_adapter("base", args.out)
    else:
        from pathlib import Path as _P

        stem = _P(args.out).with_suffix("")
        for lp in args.lora:
            model.load_state_dict(init_sd, strict=False)          # 回 B=0
            sd = torch.load(lp, map_location="cuda")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected, f"adapter 键不匹配: {unexpected[:3]}"
            print(f"\nlora loaded: {lp} ({len(sd)} tensors, "
                  f"missing {len(missing)} = 基座参数, 正常)", flush=True)
            eval_adapter(_P(lp).stem, f"{stem}__{_P(lp).stem}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
