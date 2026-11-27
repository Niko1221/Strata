#!/usr/bin/env python3
"""Prompt-path benchmark (long-prompt conversation protocol) for the Strata server on the host.

PROVENANCE: vendored 2026-11-22 from a private working copy on
the host (agent-authored, inert, outside the repo) so the benchmark is
auditable and reproducible from git. Original logic kept; this copy adds:
  --seed-base N    explicit case-id base (default: time-derived) so every
                   arm/run can use a DISJOINT id space -- caching can never
                   fake a fresh read; verify `0 reused` in the server log.
  --chain          followup mode appends the tail to the stored prefix after
                   success, so repeated `--mode followup --chain` calls build
                   a multi-turn repeated-prefix conversation (E14b rewind).
  needle check     a deterministic per-case code line is planted mid-payload;
                   every followup asks for it and grades the answer
                   (needle_hit) -- a quality gate for cache-rewind flags.
No secrets: the API key is resolved silently and never printed.

Fresh arm: each request is a fresh conversation (unique [case NNN] marker at
token 0 -> nothing shared with any earlier request, cached_tokens stays at the
template floor) carrying a synthetic synthetic agent-session payload sized
63-69K prompt tokens at --records 4600.

Cached arm (--mode followup): continues the stored conversation (same exact
prefix) with a 268-395-token tail; the engine reuses the cached prefix and
reads only the tail. Never mix fresh and cached requests into one rate.

Numbers come from the engine's own fields (docs/BENCHMARK_PROTOCOL.md):
  cached_tokens = usage.prompt_tokens_details.cached_tokens, else timings.cache_n
  read_tokens   = timings.prompt_n
  prompt_s      = timings.prompt_ms / 1000
  prompt_tps    = read_tokens / prompt_s            (fresh tokens only)

Usage:
  python3 bench_prompt.py --mode fresh --pairs 3 --records 4600 --seed-base 1000
  python3 bench_prompt.py --mode followup
  python3 bench_prompt.py --mode followup --chain   # multi-turn (E14b)
"""
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("STRATA_URL", "http://localhost:8090/v1/chat/completions")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-iq3_xxs")
STATE = Path(os.environ.get("BENCH_PROMPT_STATE",
                            "/tmp/bench_prompt_state.json"))
MAXTOK = int(os.environ.get("BENCH_PROMPT_MAXTOK", "128"))

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()


def resolve_key() -> str:
    key = os.environ.get("STRATA_API_KEY", "").strip()
    if not key:
        env = Path.home() / ".hermes/.env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("STRATA_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not key and shutil.which("podman"):
        try:
            key = subprocess.run(
                ["podman", "exec", "strata", "cat", "/data/state/api-key"],
                capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            pass
    return key


def needle_code(case_id: int) -> str:
    """Deterministic per-case needle code (planted mid-payload)."""
    return f"SEC-{(case_id * 7919) % 100000:05d}"


def make_payload(case_id: int, records: int) -> str:
    """Deterministic synthetic agent session log, unique per case id."""
    rng = random.Random(1000 + case_id)
    lines = [f"[case {case_id:04d}]",
             "You are resuming a long agent session. The log below is the "
             "session history; read it in full before answering."]
    needle_at = records // 2
    for i in range(records):
        w = [rng.choice(WORDS) for _ in range(11)]
        if i == needle_at:
            lines.append(f"record{i:06d}: NOTE the vault access code is "
                         f"{needle_code(case_id)}.")
        else:
            lines.append(f"record{i:06d}: " + " ".join(w) + ".")
    lines.append("Summarize the structure of these records and explain how "
                 "an agent would look up a record by its ID.")
    return "\n".join(lines)


def make_tail(tokens_hint: int, tail_id: int) -> str:
    """~tokens_hint-token follow-up question block (deterministic per tail_id)."""
    rng = random.Random(9000 + tail_id)
    n_lines = max(1, tokens_hint // 14)
    lines = ["Follow-up on the session above:"]
    for i in range(n_lines):
        w = [rng.choice(WORDS) for _ in range(11)]
        lines.append(f"note{i:04d}: " + " ".join(w) + ".")
    lines.append("Given everything above, answer in one short paragraph.")
    lines.append("Also: what was the vault access code in the records? "
                 "Reply with the code on its own line.")
    return "\n".join(lines)


def one_request(key: str, payload: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": payload}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.load(r)
    wall = time.perf_counter() - t0
    usage = data.get("usage", {})
    details = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    timings = data.get("timings", {}) or {}
    cached = details if details is not None else timings.get("cache_n")
    read_tokens = timings.get("prompt_n")
    if read_tokens is None:
        read_tokens = (usage.get("prompt_tokens", 0) or 0) - (cached or 0)
    prompt_s = (timings.get("prompt_ms") or 0) / 1000.0
    prompt_tps = (read_tokens / prompt_s) if (read_tokens and prompt_s) else None
    text = ""
    reasoning = ""
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    try:
        reasoning = (data["choices"][0]["message"].get("reasoning_content")
                     or "")
    except (KeyError, IndexError, TypeError):
        pass
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "read_tokens": read_tokens,
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tps": round(prompt_tps, 1) if prompt_tps else None,
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_tps": timings.get("predicted_per_second"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": data.get("choices", [{}])[0].get("finish_reason"),
        "text": text,
        "reasoning": reasoning,
    }


def load_state() -> dict:
    if not STATE.is_file():
        sys.exit(f"no state file {STATE}; run a fresh arm first")
    try:
        st = json.loads(STATE.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"state file {STATE} is not the JSON this harness writes "
                 f"({e}; stale/legacy file? run a fresh arm first)")
    for k in ("payload", "case"):
        if k not in st:
            sys.exit(f"state file {STATE} missing '{k}' (stale/legacy file? "
                     "run a fresh arm first)")
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fresh", "followup"], required=True)
    ap.add_argument("--pairs", type=int, default=3,
                    help="fresh requests (>=3 per long-prompt protocol)")
    ap.add_argument("--records", type=int, default=4600,
                    help="record lines in the payload (size calibration)")
    ap.add_argument("--followup-tokens", type=int, default=330,
                    help="target tail size for the cached arm (268-395)")
    ap.add_argument("--max-tokens", type=int, default=MAXTOK)
    ap.add_argument("--seed-base", type=int, default=None,
                    help="explicit case-id base; default time-derived. Use a "
                         "NEW base per arm/run so ids never collide across "
                         "runs (fresh reads must never hit a cached prefix)")
    ap.add_argument("--chain", action="store_true",
                    help="followup mode: append the tail to the stored "
                         "prefix after success (multi-turn repeated-prefix "
                         "chains, E14b rewind protocol)")
    a = ap.parse_args()
    key = resolve_key()
    if not key:
        sys.exit("no API key found")
    results = []
    if a.mode == "fresh":
        base = a.seed_base if a.seed_base is not None \
            else int(time.time()) % 100000
        for i in range(a.pairs):
            case_id = base + i
            payload = make_payload(case_id, a.records)
            STATE.write_text(json.dumps({
                "payload": payload, "case": case_id, "records": a.records,
                "seed_base": base,
            }))
            try:
                res = one_request(key, payload, a.max_tokens)
            except Exception as e:
                print(json.dumps({"case": case_id, "error": str(e)}),
                      flush=True)
                continue
            res.pop("text", None)  # fresh runs: metrics only, keep output small
            res.pop("reasoning", None)
            res["mode"] = "fresh"
            res["case"] = case_id
            results.append(res)
            print(json.dumps(res), flush=True)
        if results:
            tps = [r["prompt_tps"] for r in results if r.get("prompt_tps")]
            if tps:
                print(f"fresh prompt_tps median: {statistics.median(tps):.1f} "
                      f"over {len(tps)} requests")
            ms = [r["prompt_ms"] for r in results if r.get("prompt_ms")]
            if ms:
                print(f"fresh prompt_ms median: {statistics.median(ms):.0f} "
                      f"over {len(ms)} requests")
    else:
        st = load_state()
        case_id = st["case"]
        tail = make_tail(a.followup_tokens, int(time.time()) % 1000)
        full = st["payload"] + "\n" + tail
        try:
            res = one_request(key, full, a.max_tokens)
        except Exception as e:
            print(json.dumps({"mode": "followup", "error": str(e)}),
                  flush=True)
            return 1
        res["mode"] = "followup"
        res["case"] = case_id
        res["needle"] = needle_code(case_id)
        # needle grade: the FINAL ANSWER (content) first; if the model spent
        # the budget thinking, the reasoning stream counts too -- both prove
        # the cached prefix was actually read (E14b quality gate).
        if res["needle"].upper() in (res.get("text") or "").upper():
            res["needle_hit"] = True
            res["needle_where"] = "content"
        elif res["needle"].upper() in (res.get("reasoning") or "").upper():
            res["needle_hit"] = True
            res["needle_where"] = "reasoning"
        else:
            res["needle_hit"] = False
            res["needle_where"] = "absent"
        if res.get("prompt_tokens") and res.get("cached_tokens") is not None:
            res["reuse_fraction"] = round(
                res["cached_tokens"] / res["prompt_tokens"], 4)
        res["text"] = (res.get("text") or "")[-240:]  # tail of the answer
        res["reasoning"] = (res.get("reasoning") or "")[-160:]
        results.append(res)
        print(json.dumps(res), flush=True)
        if a.chain:
            STATE.write_text(json.dumps({
                "payload": full, "case": case_id, "records": None,
                "seed_base": st.get("seed_base"),
            }))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
