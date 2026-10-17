from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str((Path.cwd() / "tools").resolve()))
import strata_bench as bench

URL = "http://127.0.0.1:8080"
OUT = Path(sys.argv[1])

metrics = bench.http_json(URL, "/metrics")
context = int((metrics.get("engine") or {}).get("max_context") or 0)

if context < 32280:
    raise SystemExit(
        f"Context {context} too small for this near-32K test."
    )

suite = bench.Suite(URL, "", context, 3, False)
suite.warm_up()
suite.run_prompt(32000, "near32K")

prompt_result = suite.out["prompts"].get("near32K")

decode_rows = []

for i in range(3):
    prompt = (
        f"[benchmark {suite.tag('decode256', i)}]\n"
        "Write the integers from 1 through 1000, separated "
        "by spaces, without commentary. Continue until "
        "the output limit."
    )

    row = bench.digest(
        bench.chat(
            URL,
            "",
            [{"role": "user", "content": prompt}],
            256,
            False
        )
    )

    row["run"] = i
    decode_rows.append(row)

    print(
        f"decode run {i+1}: "
        f"{row['generated_tokens']} tokens, "
        f"{row['decode_tps']} tokens/sec, "
        f"TTFT {row['ttft_s']} sec",
        flush=True
    )

result = {
    "schema": "aetherlab-extension-v1",
    "date": datetime.date.today().isoformat(),
    "model": (metrics.get("engine") or {}).get(
        "model", "not reported"
    ),
    "engine_context": context,
    "method": {
        "speed": (
            "Unmodified Strata Suite.run_prompt("
            "32000, 'near32K')"
        ),
        "repeat_count": 3,
        "fresh_prompt": True,
        "thinking": False,
        "temperature": 0,
        "generation": (
            "Three uniquely tagged integer-sequence "
            "requests with 256-token output cap"
        ),
        "caveat": (
            "Near-32K, not stock --long and not "
            "an exact 32768-token prompt"
        ),
        "loading_included": False
    },
    "near32k": prompt_result,
    "decode256": {
        "requests": decode_rows,
        "decode_tps": bench.stat(
            r["decode_tps"] for r in decode_rows if r["ok"]
        ),
        "generated_tokens": bench.stat(
            r["generated_tokens"] for r in decode_rows if r["ok"]
        ),
        "ttft_s": bench.stat(
            r["ttft_s"] for r in decode_rows if r["ok"]
        )
    },
    "failures": suite.out["failures"] + [
        {
            "where": "decode256",
            "run": r["run"],
            "error": r["err"]
        }
        for r in decode_rows if not r["ok"]
    ]
}

OUT.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

print(f"\nResults saved: {OUT}")
if prompt_result:
    print(
        "Near-32K prompt throughput:",
        prompt_result["prompt_tps"]
    )
print(
    "256-token decode throughput:",
    result["decode256"]["decode_tps"]
)

if result["failures"]:
    raise SystemExit(
        "Some requests failed. Results saved; stop and review."
    )