#!/usr/bin/env python3
"""Run the DFlash2 long-context KV-growth check with an existing local model.

The prompt is kept outside the repository. With --pad-to, a synthetic previous
user turn is prepended to reach the measured input length while preserving the
provided prompt as the final user turn. This checks memory pressure, not semantic
parity for the unavailable padded context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from dflash_sampling import Engine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402


def tokenizer(tokenizer_dir: Path) -> ST.Tokenizer:
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", type=Path, required=True, help="UTF-8 text of the user prompt")
    p.add_argument("--pack", type=Path, required=True, help="model pack with tokenizer/")
    p.add_argument("--native", type=Path, required=True, help="target model's native first shard")
    p.add_argument("--ple-gguf", type=Path, required=True, help="target model's PLE shard")
    p.add_argument("--dflash", type=Path, required=True, help="DFlash2 GGUF artifact")
    p.add_argument("--expert-profile", type=Path, required=True)
    p.add_argument("--engine", type=Path, default=ROOT / "build-dflash2/strata")
    p.add_argument("--output", type=Path, required=True, help="JSON result path; engine log is written beside it")
    p.add_argument("--max-new", type=int, default=12300)
    p.add_argument("--pad-to", type=int, default=0, help="total prompt token count; 0 keeps the exact prompt length")
    a = p.parse_args()
    if a.max_new < 1 or a.pad_to < 0:
        p.error("--max-new must be positive and --pad-to cannot be negative")

    prompt_path = a.prompt.resolve()
    raw = prompt_path.read_text(encoding="utf-8")
    tokenizer_dir = a.pack.resolve() / "tokenizer"
    tok = tokenizer(tokenizer_dir)
    rendered = ChatTemplate(tokenizer_dir / "chat_template.jinja").render(
        [{"role": "user", "content": raw}])
    exact_ids = tok.encode(rendered, parse_special=True)
    ids = exact_ids
    synthetic_tokens = 0
    if a.pad_to:
        prefix = tok.encode("<|im_start|>user\n", parse_special=True)
        suffix = tok.encode("<|im_end|>\n", parse_special=True)
        unit = tok.encode("Earlier archived context line for long-context memory testing.\n", parse_special=True)
        missing = a.pad_to - len(exact_ids) - len(prefix) - len(suffix)
        if missing < 0:
            p.error("--pad-to is shorter than the rendered prompt")
        filler = unit * (missing // len(unit)) + unit[:missing % len(unit)]
        ids = prefix + filler + suffix + exact_ids
        synthetic_tokens = len(ids) - len(exact_ids)
        if len(ids) != a.pad_to:
            raise AssertionError("padding failed to reach the requested token count")

    args = [
        "--pack", str(a.pack.resolve()), "--native", str(a.native.resolve()),
        "--ple-gguf", str(a.ple_gguf.resolve()), "--expert-profile", str(a.expert_profile.resolve()),
        "--expert-cache", "auto", "--spec", "8", "--dflash-block", "7", "--dflash-window", "0",
        "--max-context", "524288", "--rope-scaling", "yarn", "--rope-scale", "2",
        "--kv", "int8", "--ple-io", "ram", "--kv-resident", "32768", "--pcie-frac", "0.35",
        "--spec-min-p", "0.85", "--adapt-every", "1", "--adapt-swaps", "160", "--adapt-decay", "0.97",
        "--prefill", "auto", "--vram-reserve-mib", "1486", "--dflash", str(a.dflash.resolve()), "--gpu", "0",
    ]
    out = a.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".engine.log")
    env = dict(os.environ)
    env["STRATA_DECODE_TIMING"] = "1"
    engine = Engine(a.engine.resolve(), args, ROOT, env, log)
    started = time.monotonic()
    try:
        result = engine.request(ids, "temperature=0 seed=1234 spec_min_p=0.85 dflash_block=7", a.max_new)
    finally:
        elapsed = time.monotonic() - started
        engine.close()
    log_text = log.read_text(encoding="utf-8", errors="replace")
    report = {
        "prompt_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "prompt_bytes": len(raw.encode("utf-8")),
        "exact_prompt_tokens": len(exact_ids),
        "synthetic_prior_context_tokens": synthetic_tokens,
        "total_prompt_tokens": len(ids),
        "requested_new_tokens": a.max_new,
        "returned_new_tokens": len(result.get("tokens", [])),
        "decode_ms": result.get("decode_ms"),
        "elapsed_s": elapsed,
        "accepted": result.get("accepted"),
        "offered": result.get("offered"),
        "error": result.get("error"),
        "engine_info": engine.info,
        "local_ring_log": next((line for line in log_text.splitlines()
                                 if "dflash2: local K/V ring uses" in line), None),
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["local_ring_log"] is None:
        raise SystemExit("DFlash2 local ring was not active; see the engine log")
    if result.get("error") or report["returned_new_tokens"] != a.max_new:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
