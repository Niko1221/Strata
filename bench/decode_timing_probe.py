#!/usr/bin/env python3
"""Send the fixed public request used to inspect the decode timing line.

Start a Strata server with STRATA_DECODE_TIMING=1, then run:

    python bench/decode_timing_probe.py --url http://127.0.0.1:8080/v1/chat/completions

Read the server log for the request's `strata decode timing:` line. Use the same server
configuration for the release and the patched build; leave `--adapt-async` off to exercise
the synchronous adaptive path.
"""
import argparse
import hashlib
import json
import uuid
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/DETAILS.md"
INSTRUCTION = (
    "Explain in detail, in your own words, how the engine described above keeps experts in VRAM, "
    "RAM and on disk, and what each setting changes."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trace-id", default=f"decode-timing-probe-{uuid.uuid4().hex[:12]}")
    args = parser.parse_args()

    source = SOURCE.read_text(encoding="utf-8")[:12000]
    prompt = f"Request identity-check\n\n{source}\n\n{INSTRUCTION}"
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "strata_mcp": False,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        args.url,
        json.dumps(body).encode("utf-8"),
        {
            "Content-Type": "application/json",
            "X-Qwen-Workload": "verification",
            "X-Qwen-Trace-Id": args.trace_id,
        },
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    choice = result["choices"][0]
    answer = choice.get("message", {}).get("content") or ""
    print(
        json.dumps(
            {
                "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": result.get("usage", {}).get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
