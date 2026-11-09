"""引擎 `/v1/engine/score` 与训练侧 `ce.token_logprobs` 的逐位置对比。

MeZO 的 f(θ) 原本由 `ce.token_logprobs` 算：HF 前向取 `last_hidden_state`，再过 lm_head 的
分块 logits，`gather - logsumexp`。引擎的 `/v1/engine/score` 是替换路径，但它必须先证明给出**同一串
log p** —— 否则 `(f(θ+ε) − f(θ−ε))/2ε` 的差分里混进的是两条路径的差，不是 θ 的梯度。

对比口径（与 bench_infer.py / hf_ref.py 完全一致）：
  hidden = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
  labels = ids[0, 1:]
  lp[t]  = log p(labels[t] | ids[:t+1])
引擎侧 `/v1/engine/score` 收完整 ids，返回长度 n 的 logprobs，前 n-1 个就是同一组 lp[t]
（末位没有目标，是 null）。所以两边按下标直接对齐。

两段用法（HF 侧 34 GiB、引擎侧 26 GiB，48 GB 卡放不下同时，故串行）：

    # 1) HF 侧：写 hf_score.json
    python ab_score.py --gguf ... --pack ... --tokenizer ... --tokens 1,2,3,4,5 --out hf_score.json
    # 2) 引擎侧：起引擎后比对（--hf 缺省读 hf_score.json 同目录）
    python ab_score.py --engine http://127.0.0.1:8095 --tokens 1,2,3,4,5 --hf hf_score.json

差值里预期含量化路径不同带来的一层（引擎 IQ3_XXS MMQ vs HF dequant→bf16 GEMM）。所以看两个数：
`max|Δ|` 与 `mean|Δ|`，以及**同一侧内部**的稳定性（同序列重跑必须逐位一致）。
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
import os
import sys
import urllib.request

import paths

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_ids(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def hf_side(args) -> int:
    import torch

    from ce import token_logprobs
    from loader_qwen4 import load_model
    from sft import text_model

    ids = parse_ids(args.tokens)
    if len(ids) < 2:
        print("--tokens needs at least 2 ids (there has to be a target)", file=sys.stderr)
        return 2

    model, _meta = load_model(args.gguf, args.pack, args.mapping)
    body, head = text_model(model), model.lm_head
    t = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        h = body(input_ids=t[:, :-1], use_cache=False).last_hidden_state[0]
        lp = token_logprobs(h, head, t[0, 1:], chunk=128)
        vals = [float(x) for x in lp]
    torch.cuda.synchronize()

    out = {"tokens": ids, "logprobs": vals, "sum": sum(vals), "source": "hf:ce.token_logprobs"}
    path = args.out or os.path.join(HERE, "hf_score.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"hf  n={len(ids)}  sum={out['sum']:.6f}  -> {path}")
    for i, v in enumerate(vals):
        print(f"  {i:4d}  tgt={ids[i + 1]:6d}  lp={v: .6f}")
    return 0


def engine_side(args) -> int:
    ids = parse_ids(args.tokens)
    body = json.dumps({"tokens": ids}).encode("utf-8")
    req = urllib.request.Request(args.engine.rstrip("/") + "/v1/engine/score", data=body,
                                 headers={"Content-Type": "application/json"})
    # 引擎是本机进程：绕开系统代理，否则 http_proxy 会把 127.0.0.1 也转发出去
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=args.timeout) as r:
        data = json.loads(r.read().decode("utf-8"))

    lps = data.get("logprobs") or []
    if len(lps) != len(ids):
        print(f"engine returned {len(lps)} logprobs for {len(ids)} tokens", file=sys.stderr)
        return 1
    eng = [None if v is None else float(v) for v in lps]
    if not args.hf:
        print(f"engine n={len(ids)}  sum={data.get('sum_logprob')}  scored={data.get('scored')}")
        for i, v in enumerate(eng):
            print(f"  {i:4d}  tgt={ids[i + 1] if i + 1 < len(ids) else None}  lp={'null' if v is None else f'{v: .6f}'}")
        return 0

    with open(args.hf, encoding="utf-8") as f:
        hf = json.load(f)
    if hf.get("tokens") != ids:
        print(f"hf file is for tokens {hf.get('tokens')}, not {ids}", file=sys.stderr)
        return 2
    ref = [float(x) for x in hf["logprobs"]]

    print(f"{'pos':>4} {'tgt':>7} {'hf':>10} {'engine':>10} {'delta':>9}")
    deltas = []
    for i, (a, b) in enumerate(zip(ref, eng)):
        if b is None:
            continue
        d = b - a
        deltas.append(d)
        print(f"{i:>4} {ids[i + 1]:>7} {a:>10.6f} {b:>10.6f} {d:>+9.4f}")
    if not deltas:
        print("no position the engine scored", file=sys.stderr)
        return 1
    mx = max(abs(d) for d in deltas)
    mean = sum(deltas) / len(deltas)
    mad = sum(abs(d) for d in deltas) / len(deltas)
    print(f"\n  positions {len(deltas)}  sum hf {sum(ref[:len(deltas)]):.4f}  "
          f"sum engine {sum(b for b in eng[:len(deltas)] if b is not None):.4f}")
    print(f"  delta  max|d| {mx:.4f}  mean {mean:+.4f}  mean|d| {mad:.4f} nats/token")
    print("  参考：同一引擎内 NativeHead 与按行 GEMM 差 0.02-0.05，prefill 与 decode 差 0.04-0.20")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="逗号分隔的 token id，两侧必须同一条序列")
    ap.add_argument("--gguf", nargs="+", help="HF 侧：GGUF 分片")
    ap.add_argument("--pack", help="HF 侧：专家 pack")
    ap.add_argument("--tokenizer")
    ap.add_argument("--mapping", default=paths.example("mapping_q4.json"))
    ap.add_argument("--out", default="", help="HF 侧输出 json")
    ap.add_argument("--engine", default="", help="引擎地址；给了就与 HF 侧比对")
    ap.add_argument("--hf", default="", help="HF 侧结果 json；给了 --engine 时比对")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    if args.engine:
        return engine_side(args)
    if not (args.gguf and args.pack and args.tokenizer):
        ap.error("either --engine (score a running engine) or --gguf/--pack/--tokenizer (HF side)")
    return hf_side(args)


if __name__ == "__main__":
    raise SystemExit(main())
