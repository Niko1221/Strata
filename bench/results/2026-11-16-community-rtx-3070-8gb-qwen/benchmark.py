import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

prompts = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests. Explain the implementation and complexity in detail.",
    "Explain in detail how a refrigerator moves heat from inside to outside. Discuss all four main components, thermodynamics, and efficiency in at least six paragraphs.",
    "List twenty European capitals with three sentences about each, covering history, geography, and culture.",
]
rows = []
for round_index in range(2):
    for prompt_index, prompt in enumerate(prompts):
        payload = {"model": "qwen3.8-flash-next-q2_0", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 256, "temperature": 0, "seed": 42, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        started = time.perf_counter()
        req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        timings = result.get("timings", {})
        row = {"round": round_index + 1, "prompt": prompt_index + 1, "elapsed_seconds": time.perf_counter() - started,
               "usage": result.get("usage"), "timings": timings, "finish_reason": result["choices"][0]["finish_reason"],
               "answer": result["choices"][0]["message"].get("content", "")}
        assert row["answer"].strip(), "empty answer"
        rows.append(row)
        print(json.dumps({"round": round_index + 1, "prompt": prompt_index + 1,
                          "decode_tokens_per_second": timings.get("predicted_per_second"),
                          "output_tokens": result["usage"]["completion_tokens"]}), flush=True)
report = {"prompts": prompts, "settings": {"max_tokens": 256, "temperature": 0, "thinking": False, "rounds": 2},
          "rows": rows, "median_decode_tokens_per_second": statistics.median(row["timings"]["predicted_per_second"] for row in rows),
          "weighted_decode_tokens_per_second": sum(row["timings"]["predicted_n"] for row in rows) /
          (sum(row["timings"]["predicted_ms"] for row in rows) / 1000)}
Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k not in ("prompts", "rows")}), flush=True)
