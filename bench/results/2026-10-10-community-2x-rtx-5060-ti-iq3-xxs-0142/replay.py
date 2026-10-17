"""Replay saved requests against an already configured 0.1.42 server."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", required=True, help="Server origin, without /v1")
parser.add_argument("--arm", required=True, help="Date/arm key from configs.json")
args = parser.parse_args()
root = Path(__file__).parent
rows = json.loads((root / "results.json").read_text())
fixtures = json.loads((root / "fixtures.json").read_text())
configs = json.loads((root / "configs.json").read_text())
if args.arm not in configs:
    parser.error("Unknown arm; see configs.json")

def metrics():
    with urllib.request.urlopen(args.url.rstrip("/") + "/metrics", timeout=10) as response:
        return json.load(response)

if metrics()["engine"]["version"] != "0.1.42":
    parser.error("This report measures 0.1.42; configure that engine before replay")

for row in rows:
    if row["date"] + "/" + row["arm"] != args.arm:
        continue
    body = dict(row["request"])
    body["messages"] = [body.pop("system_message")] + fixtures[row["fixture"]]["messages"]
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"},
    )
    start = time.monotonic()
    answer = reasoning = ""
    finish = usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                answer += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                finish = choice.get("finish_reason") or finish
            usage = chunk.get("usage") or usage
    result = {
        "id": row["id"], "wall_s": time.monotonic() - start,
        "finish": finish, "usage": usage, "metrics": metrics()["requests"][0],
        "answer": answer, "reasoning": reasoning,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
