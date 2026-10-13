#!/usr/bin/env python3
"""Quality metrics for the FP16 sliding window + attention sinks (kv_window.hpp) - the numbers for the PR.

Because the overlay is OFF unless asked for, the upstream reference and the new behaviour come from the SAME
binary: one arm runs `--kv q4_0` with nothing set (bit-for-bit what the build did before the change), the next
adds STRATA_KV_WINDOW/STRATA_KV_SINK, and a third runs `--kv fp16` as the exact-KV upper bound.  So the comparison
has no version skew, no toolchain skew and no other variable - which is exactly what the earlier dev-vs-release
runs could not offer.

Per context length, each arm is scored on:
  recall mid   facts buried in the body of the document (bulk precision decides these)
  recall tail  facts in the last ~30 tokens before the question (inside the overlay's 128-cell window)
  loops        replies whose sentence repetition dominates
  prefill      tok/s of the long-context prompt (first request at that length, unique text so nothing is cached)
  decode       tok/s of the answer, as served

    python tools/kv_pr_report.py            # ~12 min on an RTX 5080, needs the GPU free
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

ROOT = Path("C:/Users/blsor/Strata")
DATA = Path("C:/Users/blsor/Strata-data")
PY = ROOT / ".venv/Scripts/python.exe"
KEY = (DATA / "strata_api_key.txt").read_text(encoding="utf-8").strip()
PORT = 8091
ENGINE = os.environ.get("KV_ENGINE", "C:\\Users\\blsor\\Strata\\build\\strata.exe")
BASE = json.loads((ROOT / "strata-swift-iq3_xxs.json").read_text(encoding="utf-8"))

# (arm name, --kv value, env)
ARMS = [
    ("upstream q4_0", "q4_0", {}),
    ("q4_0 + window", "q4_0", {"STRATA_KV_WINDOW": "128", "STRATA_KV_SINK": "8"}),
    ("fp16 (exact KV)", "fp16", {}),
]
# paragraphs of ~35 words each; ~0.75 tokens per word
LENGTHS = [("4K", 100), ("16K", 400), ("32K", 800)]
# KV_LENGTHS="64K:1600,131K:3200" overrides it, for the lengths where a 4-bit cache should start to bite
if os.environ.get("KV_LENGTHS"):
    LENGTHS = [(a, int(b)) for a, b in (x.split(":") for x in os.environ["KV_LENGTHS"].split(","))]

# Exact-copy probes.  A five-digit code is either stated in the BODY (bulk precision decides whether it comes back)
# or in a record placed directly before its own question (~50 tokens from the generation position, inside the
# 128-cell window, which the overlay stores exactly). Each tail code is asked in its OWN request: put all six in one
# prompt and the instructions plus twelve questions push the earliest of them several hundred tokens back, out of
# the window, so "tail" would measure the bulk. The tail requests share the body as a prefix, so the serve's
# conversation cache makes them cheap. Reproducing a five-digit string from a quantized cache is a sharp test: the
# vault-fact recall above saturates at every context length on all three arms, so it cannot separate them.
N_MID, N_TAIL = 6, 6
MID_CODES = [("M1", "71624"), ("M2", "38150"), ("M3", "90277"), ("M4", "64538"), ("M5", "55913"), ("M6", "18746")]
TAIL_CODES = [("R1", "26409"), ("R2", "10587"), ("R3", "48391"), ("R4", "73920"), ("R5", "81736"), ("R6", "59248")]
SHORT = [
    ("What is 17 * 23? Reply with just the number.", ["391"]),
    ("What is the capital of Australia? Reply with just the city name.", ["canberra"]),
    ("How many days are in a leap year? Reply with just the number.", ["366"]),
]


def body(nonce: int, paras: int) -> str:
    """The document: filler with the MID records spread through it. The nonce makes every document unique, so the
    prefill rate is a real prefill and not a cache hit."""
    para = ("The archive holds routine correspondence from the regional office. Boxes are catalogued by year and "
            "storage vault. Entries are transcribed by hand and verified twice before they are filed. ")
    at = {}
    span = max(8, paras - 8)
    for i, (label, code) in enumerate(MID_CODES):            # spread through the body of the document
        at[int(span * (i + 1) / (len(MID_CODES) + 1))] = (label, code)
    parts = []
    for i in range(max(paras, max(at) + 1)):
        parts.append(para)
        if i in at:
            label, code = at[i]
            parts.append(f"Record: the access code for module {label} is {code}. ")
    return f"Archive index {nonce}. " + "".join(parts)


def mid_prompt(doc: str) -> str:
    asks = "\n".join(f"What is the access code for module {label}?" for label, _ in MID_CODES)
    return (doc + "\n\nUsing only the document above, answer each question on its own line as 'module N: CODE', "
                  "copying each code exactly.\n" + asks)


def tail_prompt(doc: str, label: str, code: str) -> str:
    # the record is the last thing in the document, so it sits ~50 tokens before generation: inside the window
    return (doc + f"Record: the access code for module {label} is {code}.\n\n"
                  f"What is the access code for module {label}? Answer on one line as 'module {label}: CODE', "
                  "copying the code exactly.")


def module_hits(text: str, codes: list) -> int:
    """(label, code) pairs stated in the reply - 'module M1: 71624' - so a code is credited to the module it was
    asked for. \\D{0,40}? spans the separator but never crosses another number."""
    pairs = set(re.findall(r"module\s*([a-z]\d)\D{0,40}?(\d{5})", text.lower()))
    return sum(1 for label, code in codes if (label.lower(), code) in pairs)


def cfg_for(arm: str, kv: str, no_overlay: bool) -> Path:
    cfg = json.loads(json.dumps(BASE))
    cfg["exe"] = ENGINE
    tag = re.sub(r"[^a-z0-9]+", "-", arm)
    cfg["log"] = f"C:\\Users\\blsor\\Strata\\pr-{tag}.log"
    args, i, src = [], 0, cfg["args"]
    while i < len(src):
        if src[i] == "--max-context":
            args += ["--max-context", "140000"]; i += 2; continue
        if src[i] == "--kv-resident":
            # keep KV streaming ON, as the serving config has it: the overlay has to work with the streamed pools
            args += ["--kv-resident", "20480"]; i += 2; continue
        if src[i] == "--kv":
            args += ["--kv", kv]; i += 2; continue
        if src[i] == "--conversation-cache-mib":
            args += ["--conversation-cache-mib", "512"]; i += 2; continue
        args.append(src[i]); i += 1
    cfg["args"] = args
    cfg["port"] = PORT
    p = ROOT / f"pr-{tag}.json"
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


def ask(prompt: str, seed: int = 11, budget: int = 0, max_tokens: int = 300) -> dict:
    body = json.dumps({"model": "swift-1.5-iq3_xxs", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "reasoning_budget_tokens": budget,
                       "seed": seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def text_of(reply: dict) -> str:
    m = reply["choices"][0]["message"]
    return ((m.get("content") or "") + "\n" + (m.get("reasoning_content") or ""))


def looped(reply: dict) -> bool:
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text_of(reply)) if len(s.strip()) > 8]
    if len(sents) < 6:
        return False
    dup = sum(c - 1 for c in Counter(sents).values() if c > 1)
    return dup / len(sents) > 0.5


def stop(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], capture_output=True)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    time.sleep(4)


def main() -> int:
    table = []
    for arm, kv, env in ARMS:
        cfg = cfg_for(arm, kv, False)
        e = dict(os.environ); e.update(env)
        fh = open(ROOT / (cfg.stem + ".run.log"), "w", encoding="utf-8")
        proc = subprocess.Popen([str(PY), str(ROOT / "serve/server.py"), "--engine", "strata",
                                 "--config", str(cfg), "--port", str(PORT)],
                                stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=str(ROOT))
        try:
            if not wait_ready():
                print(f"{arm}: engine never came up", flush=True)
                table.append((arm, "no engine", "", "", "", "", ""))
                continue
            for label, paras in LENGTHS:
                nonce = int(time.time() * 1000) % 10_000_000
                doc = body(nonce, paras)
                reply = ask(mid_prompt(doc), budget=0, max_tokens=320)
                mid = module_hits(text_of(reply), MID_CODES)         # copied exactly from the body
                tail = sum(module_hits(text_of(ask(tail_prompt(doc, lab, code), budget=0, max_tokens=120)),
                                       [(lab, code)])                # copied exactly from inside the window
                           for lab, code in TAIL_CODES)
                t = reply.get("timings") or {}
                pre = t.get("prompt_per_second")
                dec = t.get("predicted_per_second")
                loops = 0
                for q, accepted in SHORT:
                    r = ask(q, budget=200, max_tokens=300)
                    loops += looped(r)
                table.append((arm, label, f"{mid}/{len(MID_CODES)}", f"{tail}/{len(TAIL_CODES)}", f"{loops}/{len(SHORT)}",
                              f"{pre:.0f}" if pre else "-", f"{dec:.1f}" if dec else "-"))
                print(f"{arm:18s} {label:>4s}  copies mid {mid}/{len(MID_CODES)} tail {tail}/{len(TAIL_CODES)}  "
                      f"loops {loops}/{len(SHORT)}  prefill {pre and round(pre)}  decode {dec and round(dec, 1)}",
                      flush=True)
        finally:
            stop(proc)

    print("\n==== FP16 sliding window + sinks: exact-copy quality and speed ====")
    print(f"{'arm':18s} {'ctx':>4s} {'copy mid':>9s} {'copy tail':>10s} {'loops':>6s} "
          f"{'prefill t/s':>11s} {'decode t/s':>10s}")
    for r in table:
        print(f"{r[0]:18s} {r[1]:>4s} {r[2]:>10s} {r[3]:>11s} {r[4]:>6s} {r[5]:>11s} {r[6]:>10s}")
    (ROOT / "pr-report.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())