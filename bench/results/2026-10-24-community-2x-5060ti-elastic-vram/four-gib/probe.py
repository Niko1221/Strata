import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = os.environ.get("STRATA_TEST_URL", "http://127.0.0.1:8891")
CASES = json.loads((OUT / "cases.json").read_text())


def http(path, body=None):
    request = urllib.request.Request(BASE + path,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def snapshot():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip()


def engine_pids():
    return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits"], text=True).strip()


def generate(label, name):
    case = CASES[name]
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "reasoning_effort": "high",
        "temperature": 0, "stream": True, "messages": [
            {"role": "system", "content": "Run " + os.environ.get("STRATA_PROMPT_LABEL", label) + ". Ignore this run identifier in your answer."}
        ] + case["messages"]}
    start = time.monotonic()
    sent = time.time()
    answer = reasoning = ""
    finish = usage = None
    request = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    print("START", label, name, flush=True)
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                answer += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                finish = choice.get("finish_reason") or finish
            usage = chunk.get("usage") or usage
    records = http("/metrics")["requests"]
    metric = [r for r in records if r.get("time", 0) >= sent - 0.1 and
        r.get("prompt_tokens") == usage["prompt_tokens"] and
        r.get("output_tokens") == usage["completion_tokens"]]
    assert len(metric) == 1, metric
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].removesuffix("```").strip()
    try:
        correct = json.loads(text) == case["expected"]
    except ValueError:
        correct = False
    row = {"label": label, "case": name, "request": body, "answer": answer,
        "reasoning": reasoning, "finish": finish, "correct": correct,
        "usage": usage, "metrics": metric[0], "wall_s": time.monotonic() - start,
        "gpu_memory": snapshot(), "gpu_processes": engine_pids()}
    (OUT / (label + "-" + name + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2))
    print("RESULT", label, name, correct, finish, metric[0].get("decode_tok_s"), flush=True)
    assert correct and finish == "stop", row
    return row


if __name__ == "__main__":
    for name in sys.argv[2:]:
        generate(sys.argv[1], name)
