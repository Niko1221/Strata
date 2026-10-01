#!/usr/bin/env python3
"""Re-run the Strata-V100 fork benchmark table against a live server.

Every row is a separate, uncached OpenAI-compatible chat-completion request with a
unique-prefix repeated-text prompt and 64 generated tokens by default. Prompt and decode timings
come from the server's /metrics endpoint (the engine's own measurements), not client
wall-clock estimates -- the same method as the rows in docs/DETAILS.md.

Prompt sizes are calibrated against the ENGINE's own tokenizer with small probe
requests and a measured short suffix. The targets are 4,156 / 4,390 / 8,801 /
29,512 / 117,833 / 256,073 tokens. Exact counts can differ from these targets.
Use each row's measured prompt count when you compare runs.

For paired A/B runs, `--seed N` drives a dedicated local `random.Random`, so the same
command line sends the identical prompt sequence regardless of server tuning. Restart
the engine between seeded arms to clear its conversation cache; any reused row aborts
the run. `--repeats R` re-runs the selected rows with fresh prefixes;
`--base`/`--model`/`--max-tokens` select the server and set the row
decode budget; `--pcie-frac`/`--spec-min-p` set the engine's per-request `strata_tune`
keys (serve/server.py accepts values in [0, 1]). Only explicitly requested settings are
recorded in the raw rows, so the default schema is unchanged.
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
MAX_TOKENS = 64                        # default generated tokens per row request (--max-tokens)
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


def fresh_head(rng: random.Random | None = None) -> str:
    """A fresh unique prefix so a request's prompt shares no cached prefix with any earlier one.

    With --seed the caller's local random.Random makes the whole prefix sequence
    reproducible; without one the module's global RNG is used (the default run's behavior).
    """
    pick = rng if rng is not None else random
    return f"[benchmark-{''.join(pick.choices(string.ascii_lowercase + string.digits, k=16))}]\n\n"


def latest_after(session: requests.Session, since: float, base_url: str) -> dict | None:
    r = session.get(f"{base_url}/metrics", timeout=30)
    r.raise_for_status()
    cands = [x for x in r.json()["requests"] if x["time"] > since]
    return max(cands, key=lambda x: x["time"]) if cands else None


def probe_total(session: requests.Session, content: str, base_url: str, model: str) -> int:
    """Send a tiny request and return the server's exact templated prompt token count."""
    since = time.time()
    r = session.post(f"{base_url}/v1/chat/completions",
                     json={"model": model, "messages": [{"role": "user", "content": content}],
                           "max_tokens": 2},
                     timeout=600)
    r.raise_for_status()
    entry = latest_after(session, since, base_url)
    assert entry, "probe missing from /metrics"
    return entry["prompt_tokens"]


def engine_marginal(session: requests.Session, head: str, base_url: str, model: str,
                    rng: random.Random | None) -> tuple[int, int, int]:
    """Engine-measured (head part, one added unit, one added suffix) from four tiny probes.

    The total for `head + UNIT * n + SUFFIX * k` is headpart + n * unit_marginal +
    k * suffix_marginal: identical repeats merge identically at every boundary, so the
    marginals are exact for any n, k. Probes read prompt_tokens (the full templated
    length), which is exact whether or not the engine reuses a shared prefix.
    """
    t1 = probe_total(session, head + UNIT, base_url, model)
    t2 = probe_total(session, head + UNIT * 2, base_url, model)
    head2 = fresh_head(rng)
    s1 = probe_total(session, head2 + SUFFIX, base_url, model)
    s2 = probe_total(session, head2 + SUFFIX * 2, base_url, model)
    unit_marginal = t2 - t1
    return t1 - unit_marginal, unit_marginal, s2 - s1


def prob(value: str) -> float:
    """argparse type: a fraction in [0, 1], like serve/server.py's strata_tune keys."""
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"expected a fraction in [0, 1], got {value!r}")
    return v


def positive_int(value: str) -> int:
    """argparse type: a positive integer (--repeats, --max-tokens)."""
    v = int(value)
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-gguf", required=True, help="Qwen3.8-Flash-Next GGUF (shard 1) for local token counting")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "2026-09-28-v100-fastpath")
    ap.add_argument("--only", help="comma-separated labels to run (default: all published rows)")
    ap.add_argument("--seed", type=int, help="seed a local random.Random so the same command line sends the "
                                             "identical prompt sequence (for A/B pairing)")
    ap.add_argument("--repeats", type=positive_int, help="run the selected rows this many times, each repeat "
                                                         "with its own fresh prompt (default: 1)")
    ap.add_argument("--base", help=f"server base URL (default: {BASE})")
    ap.add_argument("--model", help=f"model name to request (default: {MODEL})")
    ap.add_argument("--max-tokens", type=positive_int,
                    help=f"generated-token budget per row request (default: {MAX_TOKENS}); probe requests stay tiny")
    ap.add_argument("--pcie-frac", type=prob, help="per-request engine strata_tune pcie_frac in [0, 1]")
    ap.add_argument("--spec-min-p", type=prob, help="per-request engine strata_tune spec_min_p in [0, 1]")
    args = ap.parse_args()

    tok = Tokenizer.from_gguf(args.model_gguf)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    base_url = args.base or BASE
    model = args.model or MODEL
    max_tokens = args.max_tokens or MAX_TOKENS
    repeats = args.repeats or 1
    rng = random.Random(args.seed) if args.seed is not None else None
    tune = {}
    if args.pcie_frac is not None:
        tune["pcie_frac"] = args.pcie_frac
    if args.spec_min_p is not None:
        tune["spec_min_p"] = args.spec_min_p

    targets = [t for t in TARGETS if not args.only or t[0] in args.only.split(",")]
    if not targets:
        raise SystemExit(f"--only matched nothing; labels are {sorted({t[0] for t in TARGETS})}")

    # --- calibration: one small request; the template overhead is the delta ----------
    head = fresh_head(rng)
    probe_count = len(tok.encode(head + UNIT))
    base = time.time()
    r = session.post(f"{base_url}/v1/chat/completions",
                     json={"model": model, "messages": [{"role": "user", "content": head + UNIT}],
                           "max_tokens": 8},
                     timeout=600)
    r.raise_for_status()
    entry = latest_after(session, base, base_url)
    assert entry, "calibration request missing from /metrics"
    assert entry["reused"] == 0, f"calibration reused={entry['reused']}"
    overhead = entry["prompt_tokens"] - probe_count
    print(f"calibration: user={probe_count} total={entry['prompt_tokens']} overhead={overhead}")
    if not 0 <= overhead <= 64:
        raise SystemExit(f"suspicious template overhead {overhead}; aborting")

    headpart, unit_marginal, suffix_marginal = engine_marginal(session, head, base_url, model, rng)
    print(f"engine marginal: headpart={headpart} (+overhead {overhead}) "
          f"marginal/unit={unit_marginal} marginal/suffix={suffix_marginal}")
    headpart -= overhead                       # headpart counts the template too; keep user-side only

    rows = []
    for label, target in targets:
        assert target + max_tokens + 8 < 262144, f"{label}: prompt + output exceeds the context"
        avail = target - headpart - overhead
        n = max(1, avail // unit_marginal)
        rem = avail - n * unit_marginal
        k = max(0, round(rem / suffix_marginal)) if suffix_marginal else 0
        for rep in range(repeats):
            head = fresh_head(rng)
            content = head + UNIT * n + SUFFIX * k
            local = len(tok.encode(content))
            base = time.time()
            t0 = time.time()
            body = {"model": model, "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens}
            if tune:
                body["strata_tune"] = tune
            r = session.post(f"{base_url}/v1/chat/completions", json=body, timeout=3600)
            wall = time.time() - t0
            r.raise_for_status()
            entry = latest_after(session, base, base_url)
            # The fresh head's token count varies by a token or two and the suffix grid has a
            # ~16-token step, so a row lands within a few tens of tokens of its target -- the
            # exact measured count is what the table reports.
            assert entry and entry["prompt_tokens"] > target - 64, f"{label}: no metric entry for the request"
            if entry["reused"] != 0:
                raise SystemExit(f"{label}: reused {entry['reused']} prompt tokens; "
                                 "restart the engine before each seeded benchmark arm")
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
            if args.seed is not None:
                row["seed"] = args.seed
            if repeats > 1:
                row["repeat"] = rep
            if args.base is not None:
                row["base"] = args.base
            if args.model is not None:
                row["model"] = args.model
            if args.max_tokens is not None:
                row["max_tokens"] = args.max_tokens
            if tune:
                row["strata_tune"] = tune
            rows.append(row)
            tag = f"{label}#{rep}" if repeats > 1 else label
            print(f"{tag:6s} total={row['prompt_tokens']:>6d} reused={row['reused']} "
                  f"prompt {row['prompt_tok_s']:7.1f} tok/s ({row['prompt_ms']/1000:.1f} s)  "
                  f"output {row['decode_tok_s']} tok/s  (local size {local})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {args.out / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
