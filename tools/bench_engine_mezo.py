"""满上下文（128K）引擎侧的账：一次 MeZO 步的墙钟，和它实际吐 token 多快。

为什么量引擎侧
--------------
反传那条路（`hf_ref.py` / `bench_fullctx.py`）在 2048 token 上已经到 8.6~8.9 s/step ≈ 2 tok/s（学生自采），
128K 更是连显存都放不下（92.63 GiB 起步）。满上下文只剩一条：引擎前向 + `SCORE`，权重冻结，±ε 由 LoRA
adapter 的**符号**给出（`SCORE + <ids>` / `SCORE - <ids>`）。这个脚本量的就是那条路。

口径（焊死，免得又出现「同一份日志读出两种 tok/s」）
----------------------------------------------------
    score tok/s  = n / SSUM 的 <ms>        n = 送进去的 token 数（前向处理的就是 n 行，不是 n-1 个被评分位置）
    MeZO 一步    = SCORE + 与 SCORE - 两次的 <ms> 之和（θ+ε 读一遍、θ−ε 读一遍，同一份表、同一个图）
    prompt tok/s = DONE 的 <prompt> / <prompt ms>
    decode tok/s = DONE 的 <generated> / <decode ms>
    TTFT         = DONE 的 <prompt ms>

推理设置不写死
--------------
`--kv auto` 解析成哪一档、prompt 路径是自开缓冲还是借槽、加载完还剩多少 VRAM —— 全部从引擎自己打印的行里
读（`strata generate: --kv auto: int8 for a 131072-token context (~1.68 GiB of KV)` 等）。阈值改了这里不用动。

用法
----
    python bench_engine_mezo.py --seqlen 131072 --gen 128
    python bench_engine_mezo.py --seqlen 8192,32768,131072 --reps 2 --out bench_engine.json

实测（2026-10-31，125B IQ3_XXS，48 GB 卡，`--kv auto`，`--no-prefill-borrow --prefill 8192`，自有缓冲）
----------------------------------------------------------------------------------------------------
| 上下文 | scored | SCORE + | SCORE − | MeZO 一步 | score tok/s | prefill tok/s | decode tok/s | 峰值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 (fp16 KV) | 7936 | 2170 ms | 2034 ms | **4.20 s** | 3656 | 4513 | 126.3 | 48.1 GiB |
| 131072 (int8 KV) | 130816 | 30332 ms | 30307 ms | **60.6 s** | 4313 | 4998 | 108.6 | 48.1 GiB |

- head 占 271 ms / 4332 ms（12% / 14%）；剩下是 prompt 路径本体。
- MTP 接受 40/81（8K）、36/80（128K）。decode 不含首 token。
- 同一请求在同进程重跑：8K 极差 0.020 nats（2.5e-06/token），128K 极差 0.169 nats（1.3e-06/token）。
  这就是 MeZO 差分能分辨的下限：±ε 的 f 差必须明显大于它。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

DATA = r"D:\Admin\Documents\GitHub\Strata-data"
GGUF = os.path.join(DATA, "models", "swift-IQ3_XXS",
                    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
PACK = os.path.join(DATA, "packs", "swift-iq3_xxs")
ROOT = r"D:\Admin\Documents\GitHub\Strata"
MEZO = r"D:\Admin\Documents\GitHub\Strata-mezo"


# ----------------------------------------------------------------------------------------------------------- pipe

class Pipe:
    """The engine's stdout, line by line, with a timeout: a scoring run at 128K is minutes, a hang is forever."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.q: queue.Queue = queue.Queue()
        self.t = threading.Thread(target=self._pump, daemon=True)
        self.t.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put(None)

    def get(self, timeout: float) -> str | None:
        try:
            line = self.q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("the engine did not answer") from None
        if line is None:
            raise SystemExit("the engine ended (see the log)")
        return line.strip()


def launch(cmd: list[str], log_path: str, cwd: str) -> tuple[subprocess.Popen, Pipe, float]:
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, cwd=cwd)
    pipe = Pipe(proc)
    while True:
        line = pipe.get(timeout=3600.0)
        if line.startswith("READY"):
            return proc, pipe, time.time() - t0
        if line.startswith("ERR"):
            raise SystemExit(f"the engine refused to start: {line}")


# -------------------------------------------------------------------------------------------------------- tokens

def load_tokenizer(pack: str):
    """The pack ships the vocabulary the engine was quantized with; `tools/strata_tokenizer.py` reads it."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(MEZO, "tools"), os.path.join(os.path.dirname(os.path.dirname(here)), "tools")):
        if os.path.isfile(os.path.join(cand, "strata_tokenizer.py")):
            sys.path.insert(0, cand)
            break
    import json as _json

    import strata_tokenizer as ST

    tdir = os.path.join(pack, "tokenizer")
    vocab = _json.load(open(os.path.join(tdir, "vocab.json"), encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = open(os.path.join(tdir, "merges.txt"), encoding="utf-8").read().split("\n")
    types = _json.load(open(os.path.join(tdir, "token_type.json"), encoding="utf-8"))
    try:
        tcfg = _json.load(open(os.path.join(tdir, "tokenizer.json"), encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)


def build_ids(tok, corpus: str, n: int) -> list[int]:
    """Real text, repeated until it fills n tokens.  Routing decides the time, and random ids route unlike text."""
    text = open(corpus, encoding="utf-8", errors="replace").read()
    ids = tok.encode(text, parse_special=False)
    if len(ids) < 64:
        raise SystemExit(f"{corpus}: only {len(ids)} tokens, nothing to repeat")
    out: list[int] = []
    while len(out) < n:
        out.extend(ids)
    return out[:n]


# ------------------------------------------------------------------------------------------------------- engine

def score(pipe: Pipe, ids: list[int], sign: str | None, timeout: float) -> dict:
    head = "SCORE" if sign is None else f"SCORE {sign}"
    pipe.proc.stdin.write(head + " " + ",".join(str(t) for t in ids) + "\n")
    pipe.proc.stdin.flush()
    t0 = time.time()
    lps: list[float] = []
    while True:
        line = pipe.get(timeout=timeout)
        if line.startswith("SL "):
            lps.append(float(line.split()[2]))
            continue
        if line.startswith("SSUM "):
            p = line.split()
            return {"sum": float(p[1]), "n": int(p[2]), "ms": float(p[3]),
                    "head_ms": float(p[4]) if len(p) > 4 else None,
                    "rows": int(p[5]) if len(p) > 5 else None,
                    "wall_s": time.time() - t0, "logprobs": lps}
        if line.startswith("ERR"):
            raise SystemExit(f"score: {line}")


def generate(pipe: Pipe, ids: list[int], max_new: int, timeout: float) -> dict:
    pipe.proc.stdin.write(f"GEN {max_new} " + ",".join(str(t) for t in ids) + "\n")
    pipe.proc.stdin.flush()
    t0 = time.time()
    while True:
        line = pipe.get(timeout=timeout)
        if line.startswith("DONE"):
            p = line.split()
            # DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
            #      ... (the rest are tier counters and differ by build: the first five are the contract)
            return {"generated": int(p[1]), "prompt": int(p[2]), "prompt_ms": float(p[3]),
                    "decode_ms": float(p[4]), "finish": p[5],
                    "draft_ok": int(p[6]) if len(p) > 6 else 0,
                    "draft_n": int(p[7]) if len(p) > 7 else 0,
                    "reused": int(p[8]) if len(p) > 8 else 0,
                    "line": line, "wall_s": time.time() - t0}
        if line.startswith("ERR"):
            raise SystemExit(f"gen: {line}")


SETTING_PATTERNS = {
    "kv": re.compile(r"--kv auto: (\S+) for a (\d+)-token context \(~([\d.]+) GiB of KV\)"),
    "free_mib": re.compile(r"(\d+) MiB of VRAM free with everything loaded"),
    "prompt_buffers": re.compile(r"prompt path (allocates its own buffers[^)]*\))"),
}


def read_settings(log_path: str) -> dict:
    """What the engine decided, from the engine: the resolved K/V precision, the prompt path's buffers, the VRAM."""
    out: dict = {}
    text = open(log_path, encoding="utf-8", errors="replace").read()
    m = SETTING_PATTERNS["kv"].search(text)
    if m:
        out["kv"] = m.group(1)
        out["kv_context"] = int(m.group(2))
        out["kv_gib"] = float(m.group(3))
    m = SETTING_PATTERNS["free_mib"].search(text)
    if m:
        out["free_mib"] = int(m.group(1))
    m = SETTING_PATTERNS["prompt_buffers"].search(text)
    out["prompt_buffers"] = m.group(1) if m else "borrowed (unknown)"
    m = re.search(r"(?:strata score|strata serve score): ([^\n]+)", text)
    if m:
        out["score_note"] = m.group(1)
    return out


class Vram:
    """Peak VRAM while a phase runs (nvidia-smi is the only place the engine's own reserve shows up)."""

    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
                v = int(r.stdout.split()[0])
                self.peak = max(self.peak, v)
            except Exception:
                return
            self.stop.wait(1.0)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()


# --------------------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", default=os.path.join(MEZO, "build-mezo", "strata.exe"))
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--native", default=GGUF)
    ap.add_argument("--expert-profile", default=os.path.join(ROOT, "data", "expert-profile.bin"))
    ap.add_argument("--corpus", default=os.path.join(MEZO, "src", "program", "generate.cpp"))
    ap.add_argument("--seqlen", default="131072", help="comma separated; each is one engine start")
    ap.add_argument("--gen", type=int, default=128, help="tokens to generate in the decode phase (0 = skip)")
    ap.add_argument("--reps", type=int, default=1, help="+/- score pairs per length (the first pair is cold)")
    ap.add_argument("--determinism", type=int, default=0,
                    help="score the SAME sequence this many times and report the spread (0 = skip)")
    ap.add_argument("--kv", default="auto", help="passed through: auto resolves against --max-context")
    ap.add_argument("--prefill", default="8192", help="the prompt path's chunk (its own buffers at --no-prefill-borrow)")
    ap.add_argument("--spec", default="4")
    ap.add_argument("--vram", default="", help="--vram, MiB of expert cache (empty = the engine's auto)")
    ap.add_argument("--no-prefill-borrow", action="store_true", default=True,
                    help="default ON: a scoring run over lent expert-cache slots is wrong (see MEMORY)")
    ap.add_argument("--allow-borrow", dest="no_prefill_borrow", action="store_false")
    ap.add_argument("--extra", default="", help="more engine flags")
    ap.add_argument("--out", default="bench_engine_mezo.json")
    ap.add_argument("--log-dir", default=".")
    a = ap.parse_args()

    tok = load_tokenizer(a.pack)
    results = []
    for seqlen in [int(x) for x in a.seqlen.split(",") if x.strip()]:
        # The context is `seqlen`; scoring reads `seqlen - gen - 128` of it so the decode phase's own tokens - and
        # the draft window the engine keeps past them - fit in the same session.  Both phases read the SAME
        # sequence, so the two figures describe one prompt, not two.
        n_seq = seqlen - a.gen - 128 if a.gen > 0 else seqlen
        ids = build_ids(tok, a.corpus, n_seq)
        cmd = [a.engine, "--pack", a.pack, "--native", a.native, "--ple-gguf", a.native,
               "--expert-cache", "auto", "--expert-profile", a.expert_profile,
               "--prefill", a.prefill, "--spec", a.spec,
               "--max-context", str(seqlen), "--kv", a.kv, "--vram-reserve-mib", "700", "--serve"]
        if a.no_prefill_borrow:
            cmd.append("--no-prefill-borrow")
        if a.vram:
            cmd += ["--vram", a.vram]
        cmd += [x for x in a.extra.split() if x]
        log_path = os.path.join(a.log_dir, f"bench_engine_{seqlen}.log")
        print(f"\n=== context {seqlen}, scored sequence {n_seq} tokens (kv={a.kv}) ===\n  {' '.join(cmd)}", flush=True)
        proc, pipe, load_s = launch(cmd, log_path, os.path.dirname(a.engine))
        try:
            settings = read_settings(log_path)
            print(f"  loaded in {load_s:.0f} s; the engine says: {settings}", flush=True)
            row = {"seqlen": n_seq, "context": seqlen, "load_s": load_s, "engine": settings,
                   "cmd": cmd, "gen_tokens": a.gen, "pairs": []}
            for rep in range(a.reps):
                with Vram() as v:
                    t = time.time()
                    plus = score(pipe, ids, "+", timeout=7200.0)
                    minus = score(pipe, ids, "-", timeout=7200.0)
                    pair_wall = time.time() - t
                step_ms = plus["ms"] + minus["ms"]
                pair = {
                    "rep": rep,
                    "plus_ms": plus["ms"], "minus_ms": minus["ms"],
                    "plus_sum": plus["sum"], "minus_sum": minus["sum"],
                    "head_ms": plus["head_ms"], "rows": plus["rows"],
                    "step_ms": step_ms,
                    "score_tok_s": plus["n"] / (plus["ms"] / 1000.0),
                    "step_wall_s": pair_wall,
                    "vram_peak_mib": v.peak,
                }
                row["pairs"].append(pair)
                print(f"  [{rep}] SCORE + {plus['ms']:9.1f} ms  SCORE - {minus['ms']:9.1f} ms  "
                      f"step {step_ms / 1000.0:7.2f} s  = {pair['score_tok_s']:7.1f} tok/s  "
                      f"(head {plus['head_ms']} ms, {plus['rows']} rows, peak {v.peak} MiB)", flush=True)
                if a.determinism > 0:
                    # The same request, in the same process, with the same head layout: anything but one figure is a
                    # path that depends on state MeZO does not control, and the +-eps difference sits under it.
                    sums = [score(pipe, ids, None, timeout=7200.0) for _ in range(a.determinism)]
                    vals = [s["sum"] for s in sums]
                    spread = max(vals) - min(vals)
                    rows_seen = sorted({s["rows"] for s in sums})
                    row["determinism"] = {"n": a.determinism, "sums": vals, "spread_nats": spread,
                                          "per_token": spread / max(1, n_seq - 1), "rows": rows_seen}
                    print(f"      determinism: {a.determinism} identical requests -> {len(set(vals))} distinct sums, "
                          f"spread {spread:.3f} nats ({spread / max(1, n_seq - 1):.4f}/token), rows {rows_seen}",
                          flush=True)
                if a.gen > 0:
                    with Vram() as vg:
                        d = generate(pipe, ids, a.gen, timeout=7200.0)
                    dec = d["decode_ms"] / 1000.0
                    row["gen"] = {
                        **{k: d[k] for k in ("generated", "prompt", "prompt_ms", "decode_ms", "finish",
                                             "draft_ok", "draft_n", "reused")},
                        "prompt_tok_s": d["prompt"] / (d["prompt_ms"] / 1000.0) if d["prompt_ms"] > 0 else None,
                        "decode_tok_s": d["generated"] / dec if dec > 0 else None,
                        "vram_peak_mib": vg.peak,
                    }
                    print(f"      GEN {d['generated']} tok: prompt {d['prompt']} tok in {d['prompt_ms'] / 1000:7.1f} s "
                          f"({row['gen']['prompt_tok_s']:7.1f} tok/s), decode {d['generated']} tok in {dec:6.1f} s "
                          f"({row['gen']['decode_tok_s']:6.1f} tok/s, drafts {d['draft_ok']}/{d['draft_n']}), "
                          f"peak {vg.peak} MiB", flush=True)
            results.append(row)
            with open(a.out, "w", encoding="utf-8") as f:   # incremental: a later length may not fit the card
                json.dump(results, f, indent=2, ensure_ascii=False)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n| 上下文 | KV | prompt 路径 | SCORE + | SCORE − | MeZO 一步 | score tok/s | decode tok/s | 同请求重跑极差 | 峰值 VRAM |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        p = r["pairs"][-1]
        g = r.get("gen") or {}
        e = r["engine"]
        det = r.get("determinism") or {}
        print(f"| {r['seqlen']} | {e.get('kv', '?')} {e.get('kv_gib', '')} GiB | {e.get('prompt_buffers', '?')[:34]} | "
              f"{p['plus_ms'] / 1000:.1f} s | {p['minus_ms'] / 1000:.1f} s | {p['step_ms'] / 1000:.1f} s | "
              f"{p['score_tok_s']:.1f} | {g.get('decode_tok_s', float('nan')):.1f} | "
              f"{det.get('spread_nats', float('nan')):.3f} nats | {p['vram_peak_mib']} MiB |")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
