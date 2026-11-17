import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


results = {"health": request("/health"), "models": request("/v1/models"), "checks": []}
model = results["models"]["data"][0]["id"]
assert "qwen3.6-35b-a3b" in model.lower(), model
for prompt in ["请用中文简短介绍你自己。", "请计算 17 × 23，只输出计算结果。", "请用中文介绍太阳系中的八大行星，每个行星用一句话描述。"]:
    started = time.perf_counter()
    response = request("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 384,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    check = {"prompt": prompt, "elapsed_seconds": round(time.perf_counter() - started, 3), "response": response}
    results["checks"].append(check)
    print(json.dumps(check, ensure_ascii=False), flush=True)

assert all(check["response"]["choices"][0]["message"].get("content", "").strip() for check in results["checks"]), "Empty answer"
assert "391" in results["checks"][1]["response"]["choices"][0]["message"]["content"], "Arithmetic check failed"
results["passed"] = True
target = Path(__file__).with_name("qwen36-deployment-verification.json")
target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("PASS:", target)
