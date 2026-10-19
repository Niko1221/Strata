#!/usr/bin/env python3
"""Send a long prompt (about --tokens worth) to the local Strata server and report timing.

Usage: prompt_test.py [--tokens N] [--max-tokens M] [--system-words W]
"""
import argparse
import json
import time
import os
import urllib.request

# the server address and its API key come from the environment; nothing secret is in this file
BASE = os.environ.get("STRATA_URL", "http://127.0.0.1:8080/v1")
KEY = os.environ.get("STRATA_KEY", "")
MODEL = "swift-1.5-iq3_s"

# a dense, code-like block: ~550 characters ~= 130 tokens each
BLOCK = (
    "def calibrate_expert_cache(slots, profile, decay): "
    "for rank, (expert, hits) in enumerate(profile): "
    "slot = slots[rank % len(slots)]; slot.weight = hits * decay; "
    "slot.resident = rank < len(slots); "
    "if slot.weight < 0.05: evict(slot); "
    "return sum(1 for s in slots if s.resident)\n"
    "# measured on gfx1031: prompt 310 tok/s, decode 34 tok/s, 5.07 GiB cache, 2645 slots\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    per_block_chars = len(BLOCK)
    approx_tok_per_block = 130
    n_blocks = max(1, round(args.tokens / approx_tok_per_block))
    system = BLOCK * n_blocks
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Say exactly: PROBE_OK. No other words."},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
    except Exception as exc:                       # noqa: BLE001
        print(f"FAILED after {time.time() - t0:.1f} s: {exc}")
        raise SystemExit(1)
    dt = time.time() - t0
    u = out.get("usage") or {}
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    msg = (out.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content")
    reason = msg.get("reasoning_content") or msg.get("reasoning")
    print(f"wall {dt:.1f} s  prompt_tokens {pt}  completion_tokens {ct}  "
          f"prompt_speed {(pt / dt):.0f} tok/s" if pt else f"wall {dt:.1f} s")
    print(f"content: {content!r}")
    if reason:
        print(f"reasoning[:160]: {str(reason)[:160]!r}")


if __name__ == "__main__":
    main()
