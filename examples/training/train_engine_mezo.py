"""train_engine_mezo.py —— 引擎侧 MeZO 训练循环：引擎自己当 forward。

分工（2026-10-31 定）：
  sft_mezo.py   HF 前向 + 本地 LoRA（feat/train-backforward 的对照线，冻结 ≤32K）。
  本文件        引擎前向：`LORA <file>` 换表 + `SCORE <ids>` 打分，零 torch。
                128K 满上下文实测一步 60.6 s（tools/bench_engine_mezo.py），HF 反传放不下。

每步（ZO-SGD，Malladi et al. 2023 的对称差分 + 种子再生）：
    z   = rng(step)                          方向由 seed 再生，不存储
    l+  = L(theta + eps*z)  —— LORA plus.gguf  → SCORE
    l-  = L(theta - eps*z)  —— LORA minus.gguf → SCORE
    pj  = (l+ - l-) / (2*eps)                方向导数估计（L = -answer 段 sum logp）
    theta -= lr * pj * z                     fp32 参数，bf16 吃不动 1e-6 量级更新

与引擎的三个接缝：
  LORA <path>   把 adapter GGUF 的表换进引擎（lora_update：形状必须不变 ⇒ rank 全程固定）。
  SCORE <ids>   逐位置 log p(ids[t+1] | ids[0..t])；answer 段从 t = plen-1 起求和
                （位置 t 的 SL 是「下一个 token」的 logprob，answer 的第一个 token 由
                prompt 的最后一位预测）。末位没有 next，引擎回 NaN，不进任何和。
  极性          训练全程 +1。SCORE 的 `+`/`-` 前缀翻的是同一份表的符号，只有
                「固定 theta、只翻扰动」的读法才用得到；这里每步换表，用不上。

chat 渲染与 serve 同一条模板（pack 的 tokenizer/chat_template.jinja，jinja2 同参数），
训练分布 = 推理分布；评测探针（--probe）也走同一渲染。

用法：
    python train_engine_mezo.py --pack <pack目录> --gguf <基座shard> \
        --engine ../../build-mezo/strata.exe --data identity --word Strata --steps 40 --lr 3e-4
    python train_engine_mezo.py ... --data jsonl:math_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402


# -------------------------------------------------------------------------------------------------- builtin identity

def builtin_data(word: str):
    """与 sft.py::builtin_data 同一组问法（测试用的两句必须在里面）。"""
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


def load_data(spec: str, word: str) -> list[tuple[str, str]]:
    spec = (spec or "identity").strip()
    if spec == "identity" or spec.startswith("identity:"):
        return builtin_data(word or spec.split(":", 1)[-1])
    path = spec.split(":", 1)[-1] if ":" in spec else spec
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows.append((r["prompt"], r["answer"]))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


# -------------------------------------------------------------------------------------------------- chat + tokens

def load_tokenizer(pack: str):
    import strata_tokenizer as ST

    tdir = os.path.join(pack, "tokenizer")
    vocab = json.load(open(os.path.join(tdir, "vocab.json"), encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = open(os.path.join(tdir, "merges.txt"), encoding="utf-8").read().split("\n")
    types = json.load(open(os.path.join(tdir, "token_type.json"), encoding="utf-8")) \
        if os.path.isfile(os.path.join(tdir, "token_type.json")) else None
    try:
        tcfg = json.load(open(os.path.join(tdir, "tokenizer.json"), encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35",
                        tcfg.get("special_ids") or None)


def load_template(pack: str):
    """与 serve/frontend.py::ChatTemplate 同一渲染设置；训练不用 tools，不需要能力探测。"""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    path = os.path.join(pack, "tokenizer", "chat_template.jinja")
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: the pack ships no chat template")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=kw.get("ensure_ascii", False))
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string(open(path, encoding="utf-8").read())
    return lambda q: tpl.render(messages=[{"role": "user", "content": q}], add_generation_prompt=True)


# -------------------------------------------------------------------------------------------------- adapter GGUF

def read_geometry(gguf_path: str) -> dict:
    from gguf_reader import GGUFFile

    f = GGUFFile(gguf_path)
    arch = str(f.metadata.get("general.architecture") or "")
    need = ["block_count", "embedding_length", "attention.head_count", "attention.key_length",
            "full_attention_interval", "ssm.inner_size"]
    out = {}
    for k in need:
        v = f.metadata.get(f"{arch}.{k}", f.metadata.get(k))
        if not isinstance(v, (int, float)):
            raise SystemExit(f"{gguf_path}: metadata '{arch}.{k}' missing (need the base model's geometry)")
        out[k] = int(v)
    return out


def adapter_layout(g: dict) -> dict[int, tuple[str, int, int]]:
    """layer -> (which, n_in, n_out)。qsa 判定与引擎 is_qsa_layer 同一条公式（layer % interval == interval-1）。"""
    L, N = g["block_count"], g["embedding_length"]
    interval = max(int(g["full_attention_interval"]), 2)
    n_qsa_in = g["attention.head_count"] * g["attention.key_length"]
    n_gdn_in = g["ssm.inner_size"]
    out = {}
    for l in range(L):
        if l % interval == interval - 1:
            out[l] = ("attn_output", n_qsa_in, N)
        else:
            out[l] = ("ssm_out", n_gdn_in, N)
    return out


def write_adapter(path: str, layout: dict, theta: dict, alpha: float):
    """theta[layer][which] = (A (r, n_in), B (n_out, r))，写 llama.cpp 的 adapter GGUF。
    flat bytes 已在 GGUF 顺序（见 lora_to_gguf.py 的 docstring）：A 平铺、B 平铺，只把 ne 反着声明。"""
    w = GGUFWriter()
    w.add("general.type", "adapter", "string")
    w.add("general.architecture", "qwen4exp", "string")
    w.add("adapter.type", "lora", "string")
    w.add("adapter.lora.alpha", float(alpha), "f32")
    for l in sorted(layout):
        which, n_in, n_out = layout[l]
        a, b = theta[l][which]
        base = f"blk.{l}.{which}.weight"
        w.tensors.append(_Tensor(base + ".lora_a", [n_in, a.shape[0]], "F32",
                                 np.ascontiguousarray(a, dtype="<f4").tobytes()))
        w.tensors.append(_Tensor(base + ".lora_b", [b.shape[1], n_out], "F32",
                                 np.ascontiguousarray(b, dtype="<f4").tobytes()))
    w.write(path)


def init_theta(layout: dict, rank: int, seed: int) -> dict:
    """A ~ N(0, 1/sqrt(n_in))（kaiming 的 LoRA 惯例），B = 0：B=0 的 adapter 与 base 逐字相同，
    所以训练前的任何读法都是 base 的读法（ab_identity 的 zero 臂就是这个含义）。"""
    rng = np.random.default_rng(seed)
    theta = {}
    for l, (which, n_in, n_out) in layout.items():
        a = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(rank, n_in)).astype(np.float32)
        b = np.zeros((n_out, rank), dtype=np.float32)
        theta[l] = {which: (a, b)}
    return theta


# -------------------------------------------------------------------------------------------------- the engine pipe

class Engine:
    """stdin/stdout 命令管道（同 tools/bench_engine_mezo.py 的模式）。"""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_reply(self, prefixes: tuple[str, ...], timeout: float = 600.0) -> list[str]:
        """读到 prefixes 之一的应答行为止；`ERR ...` 直接抛。"""
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died: rc={self.proc.poll()} (see the log)")
            line = line.rstrip("\n")
            if line.startswith("ERR"):
                raise RuntimeError(f"the engine said: {line}")
            if any(line.startswith(p) for p in prefixes):
                lines.append(line)
                return lines
            lines.append(line)   # SL ... / T id ... / others: the caller inspects them
        raise TimeoutError(f"no reply starting with {prefixes} in {timeout:.0f} s")

    def score(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        self.send("SCORE " + ",".join(map(str, ids)))
        slp: list[float] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died during SCORE: rc={self.proc.poll()}")
            line = line.rstrip("\n")
            if line.startswith("SL "):
                p = line.split()
                v = float(p[2])
                slp.append(v)
            elif line.startswith("SSUM "):
                return slp, (time.time() - t0) * 1000.0
            elif line.startswith("ERR"):
                raise RuntimeError(f"SCORE failed: {line}")

    def lora(self, path: str) -> None:
        self.send("LORA " + path)
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"the engine died during LORA: rc={self.proc.poll()}")
        line = line.rstrip("\n")
        if not line.startswith("LORAOK"):
            raise RuntimeError(f"LORA failed: {line}")

    def gen(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        t0 = time.time()
        self.send(f"GEN {max_new} " + ",".join(map(str, ids)))
        out: list[int] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died during GEN: rc={self.proc.poll()}")
            line = line.rstrip("\n")
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return out, (time.time() - t0) * 1000.0
            elif line.startswith("ERR"):
                raise RuntimeError(f"GEN failed: {line}")


def launch_engine(args, log_path: str, init_adapter: str) -> tuple[Engine, float]:
    cmd = [args.engine, "--pack", args.pack, "--native", args.gguf, "--ple-gguf", args.gguf,
           "--expert-cache", "auto", "--expert-profile", args.expert_profile,
           "--prefill", "8192", "--spec", "4", "--max-context", str(args.max_context),
           "--kv", "auto", "--vram-reserve-mib", "700", "--serve", "--no-prefill-borrow",
           "--lora", os.path.abspath(init_adapter)]
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            cwd=os.path.dirname(os.path.abspath(args.engine)))
    eng = Engine(proc)
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(f"the engine refused to start (rc={proc.poll()}), log: {log_path}")
        if line.startswith("READY"):
            return eng, time.time() - t0
        if line.startswith("ERR"):
            raise SystemExit(f"the engine refused to start: {line.strip()}")


# -------------------------------------------------------------------------------------------------- the loop

def name_logprob(eng: Engine, tok, render, q: str, word: str) -> float:
    """log p(word | q) 的引擎口径：prompt 渲染 + 逐段附上目标词，SCORE 后取 answer 段的 SL 和。
    不缓存：theta 每步都变，缓存住的 before 值会冒充 after。"""
    p_ids = tok.encode(render(q), parse_special=True)
    seg = tok.encode(word, parse_special=False)
    ids = p_ids + seg
    slp, _ = eng.score(ids)
    # 位置 t 的 SL[t] 是 log p(ids[t+1])；answer 段 = seg[0..len(seg)-1]，由 t = p0-1 .. p0+len(seg)-2 预测
    lo = max(len(p_ids) - 1, 0)
    return float(sum(slp[lo: lo + len(seg)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", required=True, help="the base model's first shard (geometry + --native)")
    ap.add_argument("--engine", default=str(REPO / "build-mezo" / "strata.exe"))
    ap.add_argument("--expert-profile", default=str(REPO / "data" / "expert-profile.bin"))
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--data", default="identity")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--probe", default="你是谁")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="LoRA 尺度（d≈3.3M）的量级；论文的 1e-6 是全参数 MeZO 的，这里要大 2~3 个数量级")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0, help="s = alpha/rank，加载器把它乘在 B 上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", default="", help="缺省：生成一个 B=0 的初始 adapter 到 --work 下")
    ap.add_argument("--work", default=str(HERE / "_engine_mezo_work"))
    ap.add_argument("--out", default=str(HERE / "lora_engine_mezo.npz"))
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="从 --out 的 npz 恢复 theta 继续训")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    tok = load_tokenizer(args.pack)
    render = load_template(args.pack)
    geom = read_geometry(args.gguf)
    layout = adapter_layout(geom)
    print(f"geometry: {geom['block_count']} layers, n_embd {geom['embedding_length']}, "
          f"qsa every {geom['full_attention_interval']}, rank {args.rank}", flush=True)

    # 参数量：每层 r*(n_in + n_out)
    n_params = sum(args.rank * (n_in + n_out) for _, n_in, n_out in layout.values())
    print(f"theta: {n_params / 1e6:.2f} M fp32 params over {len(layout)} layers", flush=True)

    init_path = args.init_adapter or os.path.join(args.work, "init.gguf")
    if args.resume and os.path.isfile(args.out):
        z = np.load(args.out, allow_pickle=True)
        theta = {int(l): {str(w): (z[f"a_{l}_{w}"], z[f"b_{l}_{w}"]) for w in ("ssm_out", "attn_output")
                         if f"a_{l}_{w}" in z}
                 for l in z["layers"]}
        theta = {int(l): dict(v) for l, v in theta.items() if v}
        print(f"resumed theta from {args.out}", flush=True)
    else:
        theta = init_theta(layout, args.rank, args.seed)
        if not os.path.isfile(init_path):
            write_adapter(init_path, layout, theta, args.alpha)
            print(f"wrote the initial adapter (B=0): {init_path}", flush=True)

    data = load_data(args.data, args.word)
    print(f"data: {len(data)} rows ({args.data})", flush=True)

    eng, load_s = launch_engine(args, os.path.join(args.work, "engine.log"), init_path)
    print(f"engine up in {load_s:.0f} s (log: {args.work}/engine.log)", flush=True)

    rows = []
    for q, a in data:
        p_ids = tok.encode(render(q), parse_special=True)
        a_ids = tok.encode(a, parse_special=False)
        rows.append((q, p_ids, a_ids))
    lens = [len(p) + len(a) for _, p, a in rows]
    print(f"encoded: prompt {min(len(p) for _, p, _ in rows)}..{max(len(p) for _, p, _ in rows)} tokens, "
          f"sequence {min(lens)}..{max(lens)} tokens", flush=True)

    # before: the probe's logp at theta=0 == base
    lp0 = name_logprob(eng, tok, render, args.probe, args.word)
    print(f"before: logp({args.word!r} | {args.probe!r}) = {lp0:.4f}", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    layers = sorted(layout)
    plus_path = os.path.join(args.work, "plus.gguf")
    minus_path = os.path.join(args.work, "minus.gguf")
    final_path = os.path.join(args.work, "final.gguf")

    def swap(th: dict, path: str) -> None:
        write_adapter(path, layout, th, args.alpha)
        eng.lora(os.path.abspath(path))   # the engine's cwd is its own directory: never a relative path

    t_train0 = time.time()
    for step in range(args.steps):
        t0 = time.time()
        z = {}
        for l in layers:
            which, n_in, n_out = layout[l]
            a, b = theta[l][which]
            za = rng.standard_normal(a.shape).astype(np.float32)
            zb = rng.standard_normal(b.shape).astype(np.float32)
            z[l] = {which: (za, zb)}

        # 一次采样一条（或轮转）：ZO 估计器吃单条即可；--steps 少时轮转保证覆盖
        q, p_ids, a_ids = rows[step % len(rows)]
        ids = p_ids + a_ids
        p0 = len(p_ids)

        th_plus = {l: {w: (theta[l][w][0] + args.eps * z[l][w][0], theta[l][w][1] + args.eps * z[l][w][1])
                       for w in theta[l]} for l in layers}
        th_minus = {l: {w: (theta[l][w][0] - args.eps * z[l][w][0], theta[l][w][1] - args.eps * z[l][w][1])
                        for w in theta[l]} for l in layers}

        swap(th_plus, plus_path)
        slp, _ = eng.score(ids)
        lp_plus = -float(sum(slp[max(p0 - 1, 0):]))   # 引擎此刻装的是 plus 表

        swap(th_minus, minus_path)
        slp, _ = eng.score(ids)
        lp_minus = -float(sum(slp[max(p0 - 1, 0):]))

        pj = (lp_plus - lp_minus) / (2 * args.eps)
        # ZO-SGD
        for l in layers:
            which = layout[l][0]
            a, b = theta[l][which]
            za, zb = z[l][which]
            a = a - args.lr * pj * za
            b = b - args.lr * pj * zb
            theta[l][which] = (a, b)

        ms = (time.time() - t0) * 1000.0
        norm = float(np.sqrt(sum(float(np.sum(a * a)) + float(np.sum(b * b))
                                 for l in layers for a, b in theta[l].values())))
        print(f"step {step:3d}: L+ {lp_plus:9.4f}  L- {lp_minus:9.4f}  pj {pj:+.3e}  "
              f"|theta| {norm:.3f}  {ms:6.0f} ms", flush=True)

        if args.eval_every and (step + 1) % args.eval_every == 0:
            swap(theta, final_path)
            lp_now = name_logprob(eng, tok, render, args.probe, args.word)
            out_ids, _ = eng.gen(tok.encode(render(args.probe), parse_special=True), 32)
            text = tok.decode(out_ids) if hasattr(tok, "decode") else ""
            print(f"  eval: logp({args.word!r}|probe) {lp0:.3f} -> {lp_now:.3f}"
                  + (f"; gen: {text[:80]!r}" if text else ""), flush=True)

    swap(theta, final_path)
    lp1 = name_logprob(eng, tok, render, args.probe, args.word)
    wall = time.time() - t_train0
    print(f"after:  logp({args.word!r} | {args.probe!r}) = {lp1:.4f}  ({lp1 - lp0:+.4f})", flush=True)
    print(f"trained {args.steps} steps in {wall:.0f} s ({wall / max(args.steps, 1):.1f} s/step)", flush=True)

    out = {"layers": np.array(layers), "alpha": np.float64(args.alpha), "rank": np.int64(args.rank)}
    for l in layers:
        for w, (a, b) in theta[l].items():
            out[f"a_{l}_{w}"] = a
            out[f"b_{l}_{w}"] = b
    np.savez(args.out, **out)
    print(f"saved theta: {args.out}\nsaved adapter: {final_path} (engine --lora / ab_identity)", flush=True)

    eng.proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
