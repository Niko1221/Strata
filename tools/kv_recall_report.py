#!/usr/bin/env python3
"""Long-context recall and speed for the KV cache, with and without the FP16 sliding window.

Short questions do not separate KV formats - a 6-question battery scores the same for fp16 and q4_0.  What does
separate them is recall from a LONG context: this builds a ~4K-token document with six unique facts buried in it,
then asks for each fact.  Numbers the report prints per arm:

    recall      how many of the 6 buried facts came back (this is where KV precision shows up)
    answers     3 short questions with checkable answers (a sanity floor, not a differentiator)
    loops       replies whose sentence repeats dominate (the failure mode seen on reasoning prompts)
    prompt      tok/s of the 4K-token prefill
    decode      tok/s of the generation, as served (MTP drafts on), and with --mtp off to isolate per-token cost

    python tools/kv_recall_report.py
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

PORT = 8089
RELEASE = engine("KV_REF_ENGINE", "engine")
DEV = engine("KV_DEV_ENGINE", "build")
BASE = base_config()

SHORT = [
    ("What is 17 * 23? Reply with just the number.", ["391"]),
    ("What is the capital of Australia? Reply with just the city name.", ["canberra"]),
    ("How many days are in a leap year? Reply with just the number.", ["366"]),
]
# Buried facts, asked by vault number and answered by code. Two groups: MID facts sit in the body of the document,
# LATE facts in its last few paragraphs. NOTE: the late facts are NOT inside a 128-cell FP16 window - after them
# come 2-5 filler paragraphs, the instructions and all eight questions (~180-350 tokens before generation), so they
# sit inside a 512/1024-cell window but outside a 128-cell one. Two codes are deliberately shared between the
# groups (90277, 18746), so recall is scored by the instructed vault/code PAIR (vault_hits), never by the code alone.
VAULTS_MID = [(3, "71624"), (14, "90277"), (31, "55913"), (40, "18746")]
VAULTS_TAIL = [(55, "64538"), (61, "18746"), (67, "38150"), (73, "90277")]
VAULTS = VAULTS_MID + VAULTS_TAIL


def vault_hits(text: str, vaults: list) -> int:
    """How many (vault, code) pairs the reply states - 'vault 61: 18746', 'vault 61 is 18746', ... A bare code is
    not a hit: 18746 is vault 40's AND vault 61's code, so finding it says nothing about which vault was recalled.
    \\D{0,40}? spans the separator (': ', ' is ', a quoted sentence's words) but never crosses another number."""
    pairs = set(re.findall(r"vault\s*#?\s*(\d+)\D{0,40}?(\d{5})", text.lower()))
    return sum(1 for vault, code in vaults if (str(vault), code) in pairs)

ARMS = [] if os.environ.get("KV_SPEED_ONLY") else [
    ("release q4_0", RELEASE, "q4_0", {}),
    ("dev q4_0", DEV, "q4_0", {}),
    ("dev q2", DEV, "q2", {}),
    ("dev q2k", DEV, "q2k", {}),   # K 2-bit per-channel + V 4-bit: 448 B/cell, between q2's 320 and q4_0's 576
    ("dev q2+window", DEV, "q2", {"STRATA_KV_WINDOW": "128", "STRATA_KV_SINK": "8"}),
]
SPEED_ONLY = [("dev q4_0", DEV, "q4_0", {}), ("dev q2", DEV, "q2", {}), ("dev q2k", DEV, "q2k", {})]


def document() -> str:
    """A long document with the six facts spread evenly through it, each stated exactly once."""
    para = ("The archive holds routine correspondence from the regional office. Boxes are catalogued by year and "
            "storage vault. Entries are transcribed by hand and verified twice before they are filed. ")
    at = {8: VAULTS_MID[0], 32: VAULTS_MID[1], 56: VAULTS_MID[2], 78: VAULTS_MID[3]}
    tail_at = {104: VAULTS_TAIL[0], 105: VAULTS_TAIL[1], 106: VAULTS_TAIL[2], 107: VAULTS_TAIL[3]}
    at.update(tail_at)
    parts = []
    for i in range(110):
        parts.append(para)
        if i in at:
            vault, code = at[i]
            parts.append(f"The access code for vault {vault} is {code}. ")
    tail = "".join(parts)
    return (tail + "\n\nUsing only the document above, answer each question on its own line as "
                    "'vault N: CODE'.\n" + "\n".join(f"What is the access code for vault {v}?" for v, _ in VAULTS))


def cfg_for(name: str, engine: str, kv: str, no_mtp: bool) -> Path:
    cfg = json.loads(json.dumps(BASE))
    cfg["exe"] = engine
    tag = re.sub(r"[^a-z0-9]+", "-", name) + ("-nomtp" if no_mtp else "")
    cfg["log"] = str(ROOT / f"recall-{tag}.log")
    args, i, src = [], 0, cfg["args"]
    while i < len(src):
        if src[i] == "--max-context":
            args += ["--max-context", "32768"]; i += 2; continue
        if src[i] == "--kv-resident":
            i += 2; continue
        if src[i] == "--kv":
            args += ["--kv", kv]; i += 2; continue
        if src[i] == "--conversation-cache-mib":
            args += ["--conversation-cache-mib", "256"]; i += 2; continue
        if no_mtp and src[i] == "--mtp":
            # Drop the flag entirely: --mtp takes the draft layer's runtime DIRECTORY, so a value like
            # "off" is read as a directory and the engine dies on off/dense.txt. --spec must stay - the
            # native (IQ) pack refuses to start without --spec T (T >= 2).
            i += 2; continue
        args.append(src[i]); i += 1
    cfg["args"] = args
    cfg["port"] = PORT
    p = ROOT / f"recall-{tag}.json"
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
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def text_of(reply: dict) -> str:
    m = reply["choices"][0]["message"]
    return ((m.get("content") or "") + "\n" + (m.get("reasoning_content") or ""))


def looped(reply: dict) -> bool:
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text_of(reply)) if len(s.strip()) > 8]
    if len(sents) < 6:
        return False
    dup = sum(c - 1 for c in Counter(sents).values() if c > 1)
    return dup / max(1, len(sents)) > 0.5


def start(cfg: Path, env: dict) -> subprocess.Popen:
    e = dict(os.environ); e.update(env)
    log = ROOT / (cfg.stem + ".run.log")
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen([str(PY), str(ROOT / "serve/server.py"), "--engine", "strata",
                             "--config", str(cfg), "--port", str(PORT)],
                            stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=str(ROOT))


def stop(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/F", "/IM", "strata.exe"], capture_output=True)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    time.sleep(4)


def main() -> int:
    rows, raw = [], {}
    for name, engine, kv, env in ARMS:
        cfg = cfg_for(name, engine, kv, False)
        proc = start(cfg, env)
        try:
            if not wait_ready():
                rows.append([name, "no engine", "-", "-", "-", "-", "-"]); continue
            doc = document()
            reply = ask(doc, budget=0, max_tokens=260)
            got = text_of(reply)
            rec_mid = vault_hits(got, VAULTS_MID)     # the instructed pair, not the bare code (codes repeat)
            rec_tail = vault_hits(got, VAULTS_TAIL)
            t = reply.get("timings") or {}
            dec = t.get("predicted_per_second")
            pre = t.get("prompt_per_second")
            ok, loops = 0, 0
            for q, accepted in SHORT:
                r = ask(q, budget=200, max_tokens=300)
                ok += any(a in text_of(r).lower() for a in accepted)
                loops += looped(r)
            rows.append([name, f"{rec_mid}/{len(VAULTS_MID)}", f"{rec_tail}/{len(VAULTS_TAIL)}",
                         f"{ok}/{len(SHORT)}", f"{loops}/{len(SHORT)}",
                         f"{pre:.0f}" if pre else "-", f"{dec:.1f}" if dec else "-"])
            raw[name] = {"recall_text": text_of(reply), "timings": t}
            print(f"{name:16s} recall mid {rec_mid}/{len(VAULTS_MID)} tail {rec_tail}/{len(VAULTS_TAIL)}  "
                  f"answers {ok}/{len(SHORT)}  loops {loops}/{len(SHORT)}  "
                  f"prompt {pre and round(pre)}  decode {dec and round(dec, 1)}", flush=True)
        finally:
            stop(proc)

    print("\n==== KV recall / speed  (mid = body of a ~5K document, tail = its last paragraphs: outside a 128-cell "
          "window, inside 512+) ====")
    print(f"{'arm':16s} {'recall mid':>10s} {'recall tail':>11s} {'answers':>8s} {'loops':>6s} "
          f"{'prompt t/s':>11s} {'decode t/s':>11s}")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:>10s} {r[2]:>11s} {r[3]:>8s} {r[4]:>6s} {r[5]:>11s} {r[6]:>11s}")

    # per-token decode cost with the drafter off: no draft acceptance to confound the number
    print("\n==== decode speed with --mtp off (per-token cost) ====")
    for name, engine, kv, env in SPEED_ONLY:
        cfg = cfg_for(name, engine, kv, True)
        proc = start(cfg, env)
        try:
            if not wait_ready():
                print(f"{name}: no engine"); continue
            r = ask("Write two sentences about latency.", budget=0, max_tokens=200)
            t = r.get("timings") or {}
            print(f"{name:16s} decode {t.get('predicted_per_second') and round(t['predicted_per_second'], 1)} tok/s "
                  f"({t.get('predicted_n')} tokens, draft_n {t.get('draft_n')})")
        finally:
            stop(proc)

    (ROOT / "recall-report.json").write_text(json.dumps(raw, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())