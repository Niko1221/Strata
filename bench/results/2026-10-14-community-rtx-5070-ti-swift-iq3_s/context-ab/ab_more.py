#!/usr/bin/env python3
"""A/B of engine settings on one config, the pattern of tools/ab_engine.py with more requests per round.

    python ab_more.py CONFIG A:  B:STRATA_ROUTE_TAIL_SKIP=0  [--rounds 4] [--out DIR] [--strata STRATA_DIR]

An arm is NAME:key=value,... as in ab_engine.py: `exe` replaces the engine, `xargs` appends ';'-separated engine
arguments, `drop` removes ';'-separated engine arguments (a flag and its value: "--spec 4"), every other key goes into
the engine's environment.  Each round starts a fresh server per arm (order A,B then B,A, alternating), warms it with one
short answer, then sends the same greedy requests (thinking off): eight short prompts and one ~6.5K-token prompt, 256
new tokens each.  Speeds are the engine's `timings`.  Per arm and round it also gives the pooled decode rate
sum(predicted_n) / sum(predicted_ms) over the eight short prompts.  Writes DIR/rows.jsonl, DIR/summary.json and each
run's engine log.
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
    "Write a C function that reverses a singly linked list in place, and explain each step in a comment.",
    "Describe the water cycle for a ten-year-old, in about two hundred words.",
    "Write a bash script that finds the ten largest files under a folder and prints their sizes in MB.",
    "Summarize the causes and the main events of the French Revolution in eight bullet points.",
    "Write a JavaScript class for a small to-do list with add, remove, toggle and a method that returns the open items.",
)


def post(port, prompt, max_tokens):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                         "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def wait_ready(port, proc, timeout=1800):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", type=int, default=6000)
    ap.add_argument("--strata", default=str(Path(__file__).resolve().parents[4]))   # the Strata folder this report is in
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.strata)
    sys.path.insert(0, str(root / "tools"))
    from ab_engine import long_prompt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(a.config).read_text())
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        arms.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    reqs = [(f"q{i}", p) for i, p in enumerate(PROMPTS)] + [("long", long_prompt(a.long_words))]
    rows_f = open(out / "rows.jsonl", "a")
    rounds = []
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else arms[::-1]
        for name, env in order:
            cfg, e = dict(base), dict(env)
            args = list(base["args"])
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "drop" in e:
                for flag in e.pop("drop").split(";"):
                    i = args.index(flag)
                    del args[i:i + 2]
            if "xargs" in e:
                args += e.pop("xargs").split(";")
            cfg["args"] = args
            cfg["port"] = a.port
            cfg["log"] = str(out / f"engine_{name}_r{rnd}.log")
            cpath = out / f"cfg_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            penv = dict(os.environ, **e)
            penv.pop("STRATA_API_KEY", None)
            t0 = time.time()
            proc = subprocess.Popen([sys.executable, str(root / "serve" / "server.py"), "--engine", "strata", "--config",
                                     str(cpath), "--port", str(a.port)], cwd=str(root), env=penv,
                                    stdout=open(out / f"server_{name}_r{rnd}.out", "w"), stderr=subprocess.STDOUT)
            try:
                wait_ready(a.port, proc)
                ready = time.time() - t0
                print(f"round {rnd} {name} {env}: ready in {ready:.0f} s", flush=True)
                got = []
                for tag, p in reqs:
                    r = post(a.port, p, a.max_new)
                    t = r.get("timings", {})
                    text = r["choices"][0]["message"].get("content", "") or ""
                    row = {"round": rnd, "arm": name, "req": tag, "decode": t.get("predicted_per_second"),
                           "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                           "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                           "prompt_ms": t.get("prompt_ms"), "draft_n": t.get("draft_n"),
                           "draft_n_accepted": t.get("draft_n_accepted"),
                           "n": r.get("usage", {}).get("completion_tokens"),
                           "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text[:80]}
                    rows_f.write(json.dumps(row) + "\n"); rows_f.flush()
                    got.append(row)
                    print(f"  {tag:5s} decode {row['decode']} prompt {row['prompt']} ({row['prompt_n']}) n {row['n']} "
                          f"drafts {row['draft_n_accepted']}/{row['draft_n']} {row['sha']}", flush=True)
                short = [x for x in got if x["req"] != "long"]
                pooled = 1000 * sum(x["predicted_n"] for x in short) / sum(x["predicted_ms"] for x in short)
                lng = [x for x in got if x["req"] == "long"][0]
                rounds.append({"round": rnd, "arm": name, "ready_s": round(ready, 1), "pooled_decode": round(pooled, 2),
                               "median_decode": statistics.median(x["decode"] for x in short),
                               "long_decode": lng["decode"], "long_prompt": lng["prompt"], "long_prompt_n": lng["prompt_n"]})
                print(f"  -> pooled short decode {pooled:.2f} tok/s, long decode {lng['decode']} prompt {lng['prompt']}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    summ = {"config": a.config, "arms": a.arms, "rounds": rounds}
    names = [n for n, _ in arms]
    print("\nper round (pooled short decode, long decode, long prompt):")
    for rnd in range(a.rounds):
        cells = []
        for n in names:
            x = [r for r in rounds if r["round"] == rnd and r["arm"] == n][0]
            cells.append(f"{n}: {x['pooled_decode']:.2f} / {x['long_decode']:.1f} / {x['long_prompt']:.0f}")
        print(f"  r{rnd}  " + " | ".join(cells))
    print("medians over rounds:")
    for n in names:
        rs = [r for r in rounds if r["arm"] == n]
        m = {k: statistics.median(r[k] for r in rs) for k in ("pooled_decode", "long_decode", "long_prompt")}
        rng = {k: (min(r[k] for r in rs), max(r[k] for r in rs)) for k in m}
        summ.setdefault("medians", {})[n] = {"median": m, "range": rng}
        print(f"  {n}: " + ", ".join(f"{k} {m[k]:.2f} ({rng[k][0]:.2f}-{rng[k][1]:.2f})" for k in m))
    if len(names) == 2:
        A, B = names
        diffs = []
        for rnd in range(a.rounds):
            xa = [r for r in rounds if r["round"] == rnd and r["arm"] == A][0]
            xb = [r for r in rounds if r["round"] == rnd and r["arm"] == B][0]
            diffs.append(100 * (xb["pooled_decode"] / xa["pooled_decode"] - 1))
        summ["paired_pct_B_vs_A"] = [round(d, 2) for d in diffs]
        print(f"paired pooled short decode, {B} vs {A}: " + ", ".join(f"{d:+.1f}%" for d in diffs)
              + f"  (median {statistics.median(diffs):+.1f}%, {sum(d > 0 for d in diffs)} of {len(diffs)} faster)")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
