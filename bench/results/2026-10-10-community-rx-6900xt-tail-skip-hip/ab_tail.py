"""Decode A/B of STRATA_ROUTE_TAIL_SKIP=7 against unset, as run for this report.

    python bench/results/2026-10-10-community-rx-6900xt-tail-skip-hip/ab_tail.py strata-unsloth-ud-q4_k_xl.json

Starts serve/server.py once per arm and round on 127.0.0.1:8091 (5 rounds, order off/tail7, then tail7/off, ...).
Both arms run with STRATA_SH_STREAM=1 STRATA_HIP_PROMPT_F16=1, the config's args plus --prompt-cache-tail and
--prefill auto, and no conversation cache. Each start: "Say hi." (warm-up, not recorded), then a story, a code answer
and a summary of the first 24,000 characters of docs/DETAILS.md (300 tokens each, greedy, thinking off), then one
needle prompt of about 8K and one of about 32K tokens (a code word hidden at 30-70% depth in a random word list).
Writes runs.jsonl and the engine logs into --out; decode tok/s is the server's own `timings.predicted_per_second`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = 8091
ARMS = {"off": {}, "tail7": {"STRATA_ROUTE_TAIL_SKIP": "7"}}
BASE_ENV = {"STRATA_SH_STREAM": "1", "STRATA_HIP_PROMPT_F16": "1"}
STRIP = ("STRATA_DENSE_MMQ", "STRATA_HIP_ADAPT_KERNEL_COPY", "STRATA_PF_FUSED", "STRATA_PF_GEMM", "STRATA_PF_SWITCH_MIN_T",
         "STRATA_PF_PAD", "STRATA_ROUTE_TAIL_SKIP")
VOCAB = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
         "ledger quartz meadow anchor violet thunder").split()
CODES = "falcon amber cobalt juniper saffron tundra maple onyx".split()
DOC = (ROOT / "docs/DETAILS.md").read_text(encoding="utf-8")[:24000]
DECODE = [
    ("story", "Write a long, detailed short story about a lighthouse keeper who finds a message in a bottle. "
              "At least 600 words."),
    ("code", "Write a complete Python module implementing an LRU cache class with get, put, delete, resize and "
             "iteration, full docstrings and type hints, followed by a pytest test suite with at least eight tests."),
    ("doc6k", "Here is a technical document:\n\n" + DOC + "\n\nSummarize this document section by section in detail."),
]


def needle_prompt(rnd: int, size: str, words: int, rep: int) -> tuple[str, str]:
    rng = random.Random(f"{rnd}-{size}-{rep}")
    code = f"{rng.choice(CODES)}-{rng.randrange(1000, 9999)}"
    body = [rng.choice(VOCAB) for _ in range(words)]
    body.insert(int(words * rng.uniform(0.3, 0.7)), f"\nThe vault code is {code}.\n")
    text = (f"Benchmark run {rnd}-{size}-{rep}-{rng.randrange(10**9)}.\nHere is a long list of words with one "
            f"sentence hidden in it:\n" + " ".join(body) +
            "\n\nWhat is the vault code given in the sentence hidden in the list? Answer with the code only.")
    return text, code


def post(prompt: str, max_tokens: int, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def engines() -> list[int]:
    return [int(x) for x in subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True).stdout.split()]


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(90)
        except subprocess.TimeoutExpired:
            proc.kill()
    t0 = time.time()
    while engines() and time.time() - t0 < 120:
        time.sleep(2)
    for pid in engines():
        os.kill(pid, signal.SIGKILL)
    time.sleep(10)


def run_arm(base_cfg: Path, out: Path, rnd: int, name: str, extra: dict, rows_f) -> None:
    cfg = json.loads(base_cfg.read_text())
    args = list(cfg["args"])
    if "--prefill" in args:
        args[args.index("--prefill") + 1] = "auto"
    args.append("--prompt-cache-tail")
    logpath = out / f"engine_tail_{name}_r{rnd}.log"
    cfg.update(args=args, host="127.0.0.1", port=PORT, log=str(logpath))
    cpath = out / "config.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    env.update(BASE_ENV, **extra)
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config", str(cpath),
                             "--port", str(PORT)], cwd=str(ROOT), env=env,
                            stdout=open(out / f"server_tail_{name}_r{rnd}.log", "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None or time.time() - t0 > 1800:
                raise RuntimeError("server did not start")
            try:
                post("Say hi.", 8, 300)
                break
            except Exception:
                time.sleep(5)
        reqs = [(tag, f"[{rnd}-{name}] " + text, None, 300) for tag, text in DECODE]
        for size, words in (("8K", 7600), ("32K", 30400)):
            text, code = needle_prompt(100 + rnd, size, words, 0)
            reqs.append((size, text, code, 32))
        for tag, text, code, max_new in reqs:
            r = post(text, max_new, 1200)
            t = r.get("timings") or {}
            ans = (r["choices"][0]["message"].get("content") or "").strip()
            row = {"round": rnd, "arm": name, "req": tag, "decode_tps": t.get("predicted_per_second"),
                   "n": r.get("usage", {}).get("completion_tokens"), "prompt_tps": t.get("prompt_per_second"),
                   "prompt_n": t.get("prompt_n"), "drafts": t.get("draft_n"), "accepted": t.get("draft_n_accepted"),
                   "correct": (code.lower() in ans.lower()) if code else None, "answer": ans[:80]}
            rows_f.write(json.dumps(row) + "\n")
            rows_f.flush()
            print(rnd, name, tag, row["decode_tps"], row["correct"], flush=True)
    except Exception as e:
        rows_f.write(json.dumps({"round": rnd, "arm": name, "failed": f"{type(e).__name__}: {e}"}) + "\n")
    finally:
        stop(proc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="the setup-written engine config, e.g. strata-unsloth-ud-q4_k_xl.json")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="ab_tail_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = open(out / "runs.jsonl", "a")
    names = list(ARMS)
    for rnd in range(a.rounds):
        for n in (names if rnd % 2 == 0 else names[::-1]):
            run_arm(Path(a.config), out, rnd, n, ARMS[n], rows_f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
