"""Separate prompt/decode timings from repeated real-model requests."""
import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18081")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--generate", type=int, default=64)
    ap.add_argument("--context-test", action="store_true")
    a = ap.parse_args()
    health = requests.get(a.base_url + "/health", timeout=10).json()
    rows = []
    for round in range(a.rounds):
        for repeat in a.repeats:
            prompt = f"Experiment {round}, size {repeat}. Context: " + "red green blue yellow " * repeat
            prompt += ("\nAfter reading, reply with exactly OK." if a.context_test else
                       "\nWrite at least 200 words explaining how a computer processes information. Start with the processor.")
            body = {"model": health["model"], "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "max_tokens": a.generate,
                    "chat_template_kwargs": {"enable_thinking": False}}
            start = time.monotonic()
            print("starting", round, repeat, flush=True)
            r = requests.post(a.base_url + "/v1/chat/completions", json=body, timeout=3600)
            r.raise_for_status()
            obj = r.json()
            assert obj["usage"]["completion_tokens"] > 0
            if a.context_test:
                assert obj["choices"][0]["message"]["content"].strip() == "OK", obj
            row = {"round": round, "pattern_repeats": repeat, "elapsed_s": time.monotonic() - start,
                   "response": obj, "wall_time": time.time()}
            rows.append(row)
            Path(a.out).write_text(json.dumps({"health": health, "checks": rows}, indent=2) + "\n")
            print("completed", round, repeat, obj["usage"], obj.get("timings"), flush=True)
    for repeat in a.repeats:
        selected = [r["response"]["timings"] for r in rows if r["pattern_repeats"] == repeat]
        print("median", repeat, {key: statistics.median(r[key] for r in selected)
                                for key in ["prompt_per_second", "predicted_per_second"]}, flush=True)


if __name__ == "__main__":
    main()
