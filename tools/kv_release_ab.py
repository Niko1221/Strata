#!/usr/bin/env python3
"""Is the thinking-loop behaviour a REGRESSION between release binaries?

`engine/.previous/strata.exe` is 0.1.41 and `engine/strata.exe` is 0.1.42 (both official release builds, CUDA 13.0),
so the question can be answered with the user's own binaries on the same prompts, the same config and the same
seeds - no local build involved.  A third arm runs 0.1.42 with `STRATA_ROUTE_TAIL_SKIP=0`, the env knob that the
0.1.42 notes say restores 0.1.41's answers byte for byte, to see whether the new router tail skip default is the
cause.

Loop-prone task shapes (repetitive formatting, self-referential constraints, long-context summarising) are used
deliberately: easy factual questions do not separate anything (the release scored 18/18 on those in both arms).

    python tools/kv_release_ab.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kv_bench_env import KEY, PY, ROOT, base_config, engine  # noqa: E402  (paths: tools/kv_bench_env.py)

PORT = 8090
PREV = engine("KV_PREV_ENGINE", "engine/.previous")   # the release before the current one
CUR = engine("KV_CUR_ENGINE", "engine")               # the current release
BASE = base_config()
SEEDS = [21, 22, 23]

FILLER = ("The datacentre report lists racks, power draw and cooling figures for each hall. "
          "Engineers recorded the measurements twice and reconciled them with the billing system. ") * 60

TASKS = [
    ("list30", "List the numbers 1 to 30, one per line, each followed by its square. Reply with only the list.",
     lambda t: "30" in t and "900" in t),
    ("countdown", "Count backwards from 50 to 1 in steps of 7, one number per line. Reply with only the numbers.",
     lambda t: "1" in t and "50" in t),
    ("no-e", "Write one paragraph about memory bandwidth without using the letter e anywhere. Reply with the paragraph.",
     lambda t: len(t.strip()) > 40 and "e" not in t.lower()),
    ("summarise", FILLER + "\nSummarise the text above in exactly three bullet points.",
     lambda t: t.count("\n") >= 2),
    ("restate", "Explain the difference between latency and throughput, give two examples, then repeat your first "
                "sentence verbatim as the last line.",
     lambda t: len(t.strip()) > 60),
]

ARMS = [
    ("0.1.41", PREV, {}),
    ("0.1.42", CUR, {}),
    ("0.1.42 tail skip 0", CUR, {"STRATA_ROUTE_TAIL_SKIP": "0"}),
]


def cfg_for(name: str, engine: str) -> Path:
    cfg = json.loads(json.dumps(BASE))
    cfg["exe"] = engine
    tag = re.sub(r"[^a-z0-9]+", "-", name)
    cfg["log"] = str(ROOT / f"rel-ab-{tag}.log")
    args, i, src = [], 0, cfg["args"]
    while i < len(src):
        if src[i] == "--max-context":
            args += ["--max-context", "32768"]; i += 2; continue
        if src[i] == "--kv-resident":
            i += 2; continue
        if src[i] == "--conversation-cache-mib":
            args += ["--conversation-cache-mib", "256"]; i += 2; continue
        args.append(src[i]); i += 1
    cfg["args"] = args
    cfg["port"] = PORT
    p = ROOT / f"rel-ab-{tag}.json"
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return p


def wait_ready(timeout_s: int = 420) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                         headers={"Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def ask(prompt: str, seed: int) -> dict:
    body = json.dumps({"model": "swift-1.5-iq3_xxs", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 900, "reasoning_budget_tokens": 400,
                       "seed": seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def text_of(reply: dict) -> str:
    m = reply["choices"][0]["message"]
    return ((m.get("reasoning_content") or "") + "\n" + (m.get("content") or ""))


def repeat_frac(reply: dict) -> float:
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text_of(reply)) if len(s.strip()) > 8]
    if len(sents) < 6:
        return 0.0
    dup = sum(c - 1 for c in Counter(sents).values() if c > 1)
    return dup / len(sents)


def main() -> int:
    out = {}
    for name, engine, env in ARMS:
        cfg = cfg_for(name, engine)
        e = dict(os.environ); e.update(env)
        fh = open(ROOT / (cfg.stem + ".run.log"), "w", encoding="utf-8")
        proc = subprocess.Popen([str(PY), str(ROOT / "serve/server.py"), "--engine", "strata",
                                 "--config", str(cfg), "--port", str(PORT)],
                                stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=str(ROOT))
        replies = []
        try:
            if not wait_ready():
                print(f"{name}: engine never came up", flush=True); continue
            for seed in SEEDS:
                for key, prompt, ok_fn in TASKS:
                    r = ask(prompt, seed)
                    txt = text_of(r)
                    body = (r["choices"][0]["message"].get("content") or "")
                    replies.append({"task": key, "seed": seed, "repeat_frac": repeat_frac(r),
                                    "looped": repeat_frac(r) > 0.4,
                                    "content_chars": len(body),
                                    "empty_content": len(body.strip()) == 0,
                                    "format_ok": bool(ok_fn(body or txt)),
                                    "finish": r["choices"][0].get("finish_reason"),
                                    "usage": r.get("usage"), "timings": r.get("timings"),
                                    "content": body[:1500], "thinking": (r["choices"][0]["message"].get("reasoning_content") or "")[:1500]})
            n = len(replies)
            loops = sum(x["looped"] for x in replies)
            empty = sum(x["empty_content"] for x in replies)
            fmt = sum(x["format_ok"] for x in replies)
            sp = [x["timings"]["predicted_per_second"] for x in replies if x["timings"]]
            print(f"{name:20s} loops {loops}/{n}  empty content {empty}/{n}  task-ok {fmt}/{n}  "
                  f"decode {round(sum(sp) / len(sp), 1)} tok/s", flush=True)
            out[name] = replies
        finally:
            subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], capture_output=True)
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            time.sleep(4)

    (ROOT / "rel-ab-replies.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n==== release A/B (q4_0, seed 21-23, 5 loop-prone tasks) ====")
    print(f"{'arm':20s} {'looped':>8s} {'empty':>7s} {'task-ok':>8s}")
    for name, replies in out.items():
        n = len(replies)
        print(f"{name:20s} {sum(x['looped'] for x in replies):>4d}/{n:<3d} "
              f"{sum(x['empty_content'] for x in replies):>4d}/{n:<3d} "
              f"{sum(x['format_ok'] for x in replies):>4d}/{n:<3d}")
    print("\nper task (looped/total):")
    for name, replies in out.items():
        per = {k: sum(x["looped"] for x in replies if x["task"] == k) for k, _, _ in TASKS}
        print(f"  {name:20s} " + "  ".join(f"{k}={per[k]}" for k, _, _ in TASKS))
    print("raw replies: rel-ab-replies.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())