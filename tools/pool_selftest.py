"""The pool's self-test on ONE PC: the model alone, then the same model split between a coordinator and a worker
on this same GPU (over 127.0.0.1), on the same prompts, greedy.  It shows that the pool runs end to end (handshake,
split, prompt chunks and verify windows over the network, the head and the drafter on the coordinator) and how its
answers and speed compare.

    .venv\\Scripts\\python tools\\pool_selftest.py strata-coder-iq1_m.json
    .venv\\Scripts\\python tools\\pool_selftest.py strata-coder-iq1_m.json --split 24 --ctx 8192 --cache 500

Both engines of the pair run on this GPU, so each gets a small expert cache (--cache, slots per engine; the model alone
gets twice that) and a short context.  Close Strata first: the GPU must be free.

The answers need not match token for token: an expert the GPU computes rounds differently from one the CPU computes,
and the pair caches other experts than the single engine (bench/results/2026-09-27-cache-parity).  What must hold:
the pool answers, its answers make sense, and most tokens agree.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

PROMPTS = (
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List the planets of the solar system in order from the sun, comma-separated.",
)


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    out = list(args)
    while flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def base_args(cfg: dict, ctx: int, cache: int, chunk: int) -> list[str]:
    a = list(cfg["args"])
    for flag in ("--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib",
                 "--kv-resident", "--layer-split", "--split-device"):
        a = with_arg(a, flag, None)
    a = [x for x in a if not x.startswith("--pool-")]
    a = with_arg(a, "--max-context", str(ctx))
    a = with_arg(a, "--expert-cache", str(cache))
    a = with_arg(a, "--prefill", str(chunk))
    return a


def run_prompts(eng, ids_list, max_new: int, say) -> list[dict]:
    out = []
    for ids in ids_list:
        cancel = threading.Event()
        t0 = time.time()
        toks = [t for t in eng.generate(ids, max_new, {}, cancel) if t is not None]
        out.append({"tokens": toks, "s": time.time() - t0, **(eng.last or {})})
        say(f"    {len(toks)} tokens in {out[-1]['s']:.1f} s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the model's run config (strata-<model>.json)")
    ap.add_argument("--split", default="24", help="the worker's first layer (default 24)")
    ap.add_argument("--ctx", type=int, default=8192, help="context for this test (default 8192)")
    ap.add_argument("--cache", type=int, default=400, help="expert cache slots per engine of the pair (default 400)")
    ap.add_argument("--chunk", type=int, default=1024, help="prompt chunk (default 1024)")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--port", type=int, default=7799)
    ap.add_argument("--wire", default="f32", choices=["f32", "f16", "bf16"])
    ap.add_argument("--skip-single", action="store_true", help="only the pool run")
    a = ap.parse_args()
    from serve.server import StrataEngine, child_env
    import strata_tokenizer as ST

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    env = child_env(cfg)
    log = str(Path(cfg.get("log") or "strata-selftest.log").with_suffix("")) + "-selftest.log"
    say = lambda m: print(m, flush=True)   # noqa: E731

    single = None
    if not a.skip_single:
        say(f"[selftest] 1/2: the model alone ({2 * a.cache} cache slots, context {a.ctx}) ...")
        eng = StrataEngine(cfg["exe"], base_args(cfg, a.ctx, 2 * a.cache, a.chunk), cwd=cfg.get("cwd"), log=log, env=env)
        try:
            single = run_prompts(eng, ids_list, a.max_new, say)
        finally:
            eng.close()

    say(f"[selftest] 2/2: the pool on this PC: coordinator layers 0-{int(a.split) - 1}, worker {a.split}-47 "
        f"over 127.0.0.1:{a.port} ({a.wire}) ...")
    secret = secrets.token_hex(8)
    penv = dict(env, STRATA_POOL_SECRET=secret, STRATA_POOL_NAME="selftest")
    wargs = base_args(cfg, a.ctx, a.cache, a.chunk) + ["--pool-listen", f"127.0.0.1:{a.port}"]
    wlog = open(str(Path(log).with_suffix("")) + "-worker.log", "a", encoding="utf-8")
    worker = subprocess.Popen([cfg["exe"], "--serve", *wargs], cwd=cfg.get("cwd"), stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=wlog, text=True, env=penv)
    lines = []

    def pump():
        for line in worker.stdout:
            lines.append(line.strip())
            if line.startswith("POOL "):
                say("    worker: " + line.strip())
    threading.Thread(target=pump, daemon=True).start()
    for _ in range(600):                                   # its weights first, then it listens
        if any(x.startswith("POOL LISTEN") for x in lines) or worker.poll() is not None:
            break
        time.sleep(0.5)
    if worker.poll() is not None:
        say(f"[selftest] the worker engine exited (code {worker.returncode}); see {wlog.name}")
        return 1
    cargs = base_args(cfg, a.ctx, a.cache, a.chunk) + ["--pool-peers", f"127.0.0.1:{a.port}", "--pool-split", a.split,
                                                       "--pool-wire", a.wire]
    try:
        eng = StrataEngine(cfg["exe"], cargs, cwd=cfg.get("cwd"), log=log, env=penv)
    except RuntimeError as e:
        say(f"[selftest] the coordinator did not start: {e} (see {log})")
        worker.kill()
        return 1
    try:
        say(f"    pool: {eng.info.get('pool_workers')}, {eng.info.get('expert_slots')} experts on the GPU in total")
        pooled = run_prompts(eng, ids_list, a.max_new, say)
    finally:
        eng.close()
        worker.terminate()
        try:
            worker.wait(30)
        except subprocess.TimeoutExpired:
            worker.kill()

    say("")
    ok = True
    for i, p in enumerate(PROMPTS):
        pt = pooled[i]["tokens"]
        text = tok.decode(pt)
        say(f"[{i + 1}] {p}")
        say(f"    pool : {text.strip()[:300]!r}")
        if not pt:
            ok = False
        if single:
            st = single[i]["tokens"]
            same = 0
            while same < min(len(st), len(pt)) and st[same] == pt[same]:
                same += 1
            say(f"    alone: {tok.decode(st).strip()[:300]!r}")
            say(f"    the first {same} of {max(len(st), len(pt))} tokens are the same; "
                f"decode {single[i].get('generated', 0) / max(single[i].get('decode_ms', 1) / 1000, 1e-3):.1f} tok/s "
                f"alone, {pooled[i].get('generated', 0) / max(pooled[i].get('decode_ms', 1) / 1000, 1e-3):.1f} pooled")
    if "paris" not in tok.decode(pooled[0]["tokens"]).lower():
        ok = False
        say("[selftest] the pool's first answer does not say Paris")
    say(f"[selftest] {'PASS' if ok else 'FAIL'}: the pool " + ("answers end to end" if ok else "did not answer correctly"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
