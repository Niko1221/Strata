#!/usr/bin/env python3
"""Run only Strata's fresh-prompt benchmark at approximately 128K tokens."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(args.app / "tools"))
    import strata_bench as bench

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.url.rstrip("/")
    before = bench.collect_machine(base, "", config, None)
    suite = bench.Suite(base, "", before.get("context"), max(1, args.runs), False)

    started = time.time()
    suite.warm_up()
    suite.run_prompt(131072, "128K")
    machine = bench.collect_machine(base, "", config, None)
    if not machine.get("gpu"):
        machine["gpu"] = before.get("gpu")

    result = {
        "schema": bench.SCHEMA,
        "tool": "bench-128k.py using tools/strata_bench.py",
        "date": datetime.date.today().isoformat(),
        "params": {
            "runs": suite.runs,
            "think": False,
            "target_prompt_tokens": 131072,
            "generated_token_cap": bench.LONG_MAX_TOKENS,
            "fresh_prompt_tags": True,
            "warmup_excluded": True,
            "model_load_excluded": True,
        },
        "elapsed_s": time.time() - started,
        "machine": machine,
        "results": suite.out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(args.out)

    prompt = suite.out["prompts"].get("128K", {})
    print("128K result:")
    print(f"  prompt tok/s: {bench.fmt(prompt.get('prompt_tps'))}")
    print(f"  TTFT:         {bench.fmt(prompt.get('ttft_s'), 3, ' s')}")
    return 1 if suite.out["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
