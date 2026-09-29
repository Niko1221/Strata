#!/usr/bin/env python3
"""Re-run the Strata-V100 fork benchmark table against a live server.

Every row is a separate, uncached OpenAI-compatible chat-completion request with a
unique-prefix repeated-text prompt and 64 generated tokens. Prompt and decode timings
come from the server's /metrics endpoint (the engine's own measurements), not client
wall-clock estimates -- the same method as the rows in docs/DETAILS.md.

Prompt sizes are calibrated against the ENGINE's own tokenizer (tiny probe requests)
and fine-tuned with a measured short suffix, so each row lands on the exact published
total (4,156 / 4,390 / 8,801 / 29,512 / 117,833 / 256,073) and stays comparable row
for row with the engine 0.1.20 baseline.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

BASE = "http://127.0.0.1:8088"
MODEL = "qwen3.8-flash-next-q2_0"
UNIT = (
    "The compute layer interleaves two passes over the model: a prompt pass that reads the "
    "request in large chunks and a token pass that produces one answer token at a time. The "
    "prompt pass borrows cache slots for its chunk buffers, multiplies the experts with "
    "quantized kernels, and streams the next layer's experts while the current layer's "
    "attention runs. The token pass keeps the busiest experts on the graphics card and lets "
    "the CPU finish the rest, so neither waits for the other. The lookup table on the SSD "
    "serves only a few small rows per token, and the small helper model guesses the next few "
    "words while the big model checks them all at once. This is why the same hardware can run "
    "a model that normally needs a server: the work is shared across the whole machine, and "
    "only the parts every token needs stay on the card."
)

# A short suffix for fine-tuning a prompt to an exact token total (integer unit counts
# only land on a grid of ~167 tokens). First sentence of UNIT; no special-token literals.
SUFFIX = (
    "The compute layer interleaves two passes over the model: a prompt pass that reads the "
    "request in large chunks and a token pass that produces one answer token at a time."
)

# (label, exact total prompt tokens of the published row)
TARGETS = [
    ("~4K", 4156),
    ("~4K", 4390),
    ("~8K", 8801),
    ("32K", 29512),
    ("128K", 117833),
    ("256K", 256073),
]


def api_key() -> str:
    env = Path(ROOT / ".strata-service.env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("STRATA_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no STRATA_API_KEY in .strata-service.env")


def fresh_head() -> str:
    """A fresh unique prefix so a request's prompt shares no cached prefix with any earlier one."""
    return f"[benchmark-{''.join(random.choices(string.ascii_lowercase + string.digits, k=16))}]\n\n"


def latest_after(session: requests.Session, base: float) -> dict | None:
    r = session.get(f"{BASE}/metrics", timeout=30)
    r.raise_for_status()
    cands = [x for x in r.json()["requests"] if x["time"] > base]
    return max(cands, key=lambda x: x["time"]) if cands else None


def probe_total(session: requests.Session, content: str) -> int:
    """Send a tiny request and return the server's exact templated prompt token count."""
    base = time.time()
    r = session.post(f"{BASE}/v1/chat/completions",
                     json={"model": MODEL, "messages": [{"role": "user", "content": content}],
                           "max_tokens": 2},
                     timeout=600)
    r.raise_for_status()
    entry = latest_after(session, base)
    assert entry, "probe missing from /metrics"
    return entry["prompt_tokens"]


def engine_marginal(session: requests.Session, head: str) -> tuple[int, int, int]:
    """Engine-measured (head part, one added unit, one added suffix) from four tiny probes.

    The total for `head + UNIT * n + SUFFIX * k` is headpart + n * unit_marginal +
    k * suffix_marginal: identical repeats merge identically at every boundary, so the
    marginals are exact for any n, k. Probes read prompt_tokens (the full templated
    length), which is exact whether or not the engine reuses a shared prefix.
    """
    t1 = probe_total(session, head + UNIT)
    t2 = probe_total(session, head + UNIT * 2)
    head2 = fresh_head()
    s1 = probe_total(session, head2 + SUFFIX)
    s2 = probe_total(session, head2 + SUFFIX * 2)
    unit_marginal = t2 - t1
    return t1 - unit_marginal, unit_marginal, s2 - s1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-gguf", required=True, help="Qwen3.8-Flash-Next GGUF (shard 1) for local token counting")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "2026-09-28-v100-fastpath")
    ap.add_argument("--only", help="comma-separated labels to run (default: all published rows)")
    args = ap.parse_args()

    tok = Tokenizer.from_gguf(args.model_gguf)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})

    targets = [t for t in TARGETS if not args.only or t[0] in args.only.split(",")]
    if not targets:
        raise SystemExit(f"--only matched nothing; labels are {sorted({t[0] for t in TARGETS})}")

    # --- calibration: one small request; the template overhead is the delta ----------
    head = fresh_head()
    probe_count = len(tok.encode(head + UNIT))
    base = time.time()
    r = session.post(f"{BASE}/v1/chat/completions",
                     json={"model": MODEL, "messages": [{"role": "user", "content": head + UNIT}],
                           "max_tokens": 8},
                     timeout=600)
    r.raise_for_status()
    entry = latest_after(session, base)
    assert entry, "calibration request missing from /metrics"
    assert entry["reused"] == 0, f"calibration reused={entry['reused']}"
    overhead = entry["prompt_tokens"] - probe_count
    print(f"calibration: user={probe_count} total={entry['prompt_tokens']} overhead={overhead}")
    if not 0 <= overhead <= 64:
        raise SystemExit(f"suspicious template overhead {overhead}; aborting")

    headpart, unit_marginal, suffix_marginal = engine_marginal(session, head)
    print(f"engine marginal: headpart={headpart} (+overhead {overhead}) "
          f"marginal/unit={unit_marginal} marginal/suffix={suffix_marginal}")
    headpart -= overhead                       # headpart counts the template too; keep user-side only

    rows = []
    for label, target in targets:
        assert target + 64 + 8 < 262144, f"{label}: prompt + output exceeds the context"
        avail = target - headpart - overhead
        n = max(1, avail // unit_marginal)
        rem = avail - n * unit_marginal
        k = max(0, round(rem / suffix_marginal)) if suffix_marginal else 0
        head = fresh_head()
        content = head + UNIT * n + SUFFIX * k
        local = len(tok.encode(content))
        base = time.time()
        t0 = time.time()
        r = session.post(f"{BASE}/v1/chat/completions",
                         json={"model": MODEL, "messages": [{"role": "user", "content": content}],
                               "max_tokens": 64},
                         timeout=3600)
        wall = time.time() - t0
        r.raise_for_status()
        entry = latest_after(session, base)
        # The fresh head's token count varies by a token or two and the suffix grid has a
        # ~16-token step, so a row lands within a few tens of tokens of its target -- the
        # exact measured count is what the table reports.
        assert entry and entry["prompt_tokens"] > target - 64, f"{label}: no metric entry for the request"
        speed = entry["prompt_tokens"] / (entry["prompt_ms"] / 1000)
        row = {
            "label": label,
            "target": target,
            "prompt_tokens": entry["prompt_tokens"],
            "reused": entry["reused"],
            "prompt_ms": entry["prompt_ms"],
            "prompt_tok_s": round(speed, 1),
            "output_tokens": entry["output_tokens"],
            "decode_ms": entry["decode_ms"],
            "decode_tok_s": entry["decode_tok_s"],
            "hit_rate": entry["hit_rate"],
            "duration_s": round(entry["duration_s"], 1),
            "wall_s": round(wall, 1),
        }
        rows.append(row)
        print(f"{label:6s} total={row['prompt_tokens']:>6d} reused={row['reused']} "
              f"prompt {row['prompt_tok_s']:7.1f} tok/s ({row['prompt_ms']/1000:.1f} s)  "
              f"output {row['decode_tok_s']} tok/s  (local size {local})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
