"""A/B two engine builds or environments on one config: restart the server per arm, alternate arms over rounds.

    python tools/ab_engine.py strata-q2_0.json A:exe=/path/a B:exe=/path/b,STRATA_EXP_MODE=0 [--rounds 3]

An arm is NAME:key=value,...; `exe` replaces the config's engine, `xargs` appends ';'-separated engine arguments,
every other key is set in the engine's environment.  Each round starts the server once per arm (the arms' order alternates), warms it up with one short
answer, then sends the same requests: greedy, fixed prompts, a short chat set and one long prompt.  Prints the
Prints the server's own timings (llama.cpp's `timings`: decode and prompt tokens per second) per request and the per-arm
medians.  Decode numbers move run to run (the adaptive expert tier, the OS), so compare medians over rounds.

With `--stats` (needs `pip install -r requirements-bench.txt`) each arm is also compared against the first arm slot by slot:
Wilcoxon signed-rank on the log ratios, Holm-adjusted across metrics, with the Hodges-Lehmann speedup factor and its
confidence interval.  A verdict needs the interval to clear a floor (`--floor-decode 0.05 --floor-prefill 0.02`);
with 3 rounds there are 3 pairs per metric and no verdict can be significant - run `--rounds 7` or more.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, "Say hi.", 8)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", type=int, default=6000)
    ap.add_argument("--log", default="ab_engine.jsonl")
    ap.add_argument("--stats", action="store_true", help="paired Wilcoxon verdicts per metric (needs scipy and statsmodels)")
    ap.add_argument("--floor-decode", type=float, default=0.05, help="speed factor that counts: 0.05 = 5%% (decode)")
    ap.add_argument("--floor-prefill", type=float, default=0.02, help="speed factor that counts: 0.02 = 2%% (prompt)")
    a = ap.parse_args()
    base = json.loads(Path(a.config).read_text())
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        arms.append((name, env))
    reqs = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(a.long_words))]
    out = open(a.log, "a")
    res: dict = {n: {} for n, _ in arms}
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else arms[::-1]
        for name, env in order:
            cfg = dict(base)
            e = dict(env)
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "xargs" in e:                       # extra engine arguments, ';'-separated
                cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
            cfg["port"] = a.port
            cfg["log"] = str(Path(tempfile.gettempdir()) / f"ab_{name}_r{rnd}.log")
            cpath = Path(tempfile.gettempdir()) / f"ab_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            penv = dict(os.environ, **e)
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                     str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=penv,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                t0 = time.time()
                wait_ready(a.port, proc)
                print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s", flush=True)
                for tag, p in reqs:
                    r = post(a.port, p, a.max_new)
                    t = r.get("timings", {})
                    row = {"round": rnd, "arm": name, "req": tag, "decode": t.get("predicted_per_second"),
                           "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                           "n": r.get("usage", {}).get("completion_tokens"),
                           "text": r["choices"][0]["message"].get("content", "")[:80]}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    res[name].setdefault(tag, []).append(row)
                    print(f"  {tag:6s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']})  "
                          f"n {row['n']}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    print("\nmedians over rounds:")
    for tag, _ in reqs:
        cells = []
        for name, _ in arms:
            rows = res[name].get(tag, [])
            d = statistics.median([x["decode"] for x in rows if x["decode"]]) if rows else float("nan")
            pr = statistics.median([x["prompt"] for x in rows if x["prompt"]]) if rows else float("nan")
            cells.append(f"{name}: decode {d:6.1f}  prompt {pr:7.1f}")
        print(f"  {tag:6s} | " + " | ".join(cells))
    if a.stats:
        report_stats(res, arms, reqs, a)
    return 0


def report_stats(res: dict, arms, reqs, a) -> None:
    """Paired verdicts through tools/paired_stats.py (scipy + statsmodels): every arm against
    the first, one pair per request slot and round, log-ratios; Holm across all metrics."""
    try:
        import paired_stats as ps
    except ImportError:
        ps = None
    if ps is None or not ps.available():
        print("verdicts need scipy and statsmodels: pip install -r requirements-bench.txt", file=sys.stderr)
        return
    base = arms[0][0]
    dims: list[tuple[str, str, list]] = []
    for other, _ in arms[1:]:
        for tag, _p in reqs:
            for metric in ("decode", "prompt"):
                if metric == "prompt" and not tag.startswith("long"):
                    continue                    # the short chats barely measure prefill
                ra = {r["round"]: r for r in res[base].get(tag, [])}
                rb = {r["round"]: r for r in res[other].get(tag, [])}
                diffs = [math.log(rb[k][metric] / ra[k][metric]) for k in sorted(ra)
                         if k in rb and ra[k].get(metric) and rb[k].get(metric)]
                dims.append((f"{other} vs {base} {tag} {metric}", metric, diffs))
    stats = [ps.signed_rank(d) for _, _, d in dims]
    adj = ps.holm([s["p"] for s in stats])
    print("\npaired verdicts (Wilcoxon signed-rank on log ratios, Holm-adjusted; the CI must clear the floor):")
    for (label, metric, _d), s, pa in zip(dims, stats, adj):
        floor = a.floor_decode if metric == "decode" else a.floor_prefill
        ci = "" if s["ci_lo"] is None else f" [{ps.factor(s['ci_lo'])}-{ps.factor(s['ci_hi'])}]"
        p = "-" if s["p"] is None else f"{s['p']:.4f}"
        print(f"  {label:38s} n={s['n']:2d}  {ps.factor(s['hl'])}{ci:20s}  p={p:7s} Holm p={pa:.4f}  "
              + ps.verdict(s, pa, floor))
    if any(s["n"] < 7 for s in stats):
        print("  (metrics with fewer than 7 pairs cannot reach significance at Holm 0.05 two-sided: run more rounds)")


if __name__ == "__main__":
    sys.exit(main())
