"""Real-model API checks. Start the loopback validation server before running."""
import argparse
import json
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18081")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = requests.get(a.base_url + "/health", timeout=10).json()["model"]
    base = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: Hello Orin!"}],
            "temperature": 0, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}
    rows = []

    def post(body):
        t = time.monotonic()
        r = requests.post(a.base_url + "/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        obj = r.json()
        assert obj["usage"]["completion_tokens"] > 0
        assert obj["choices"][0]["message"]["content"].strip() == "Hello Orin!", obj
        return {"elapsed_s": time.monotonic() - t, "response": obj}

    for i in range(3):
        rows.append({"check": "repeated_greedy", "round": i, **post(base)})
        print("repeated_greedy", i, "pass", flush=True)

    for anthropic in (False, True):
        body = dict(base, stream=True)
        path = "/v1/chat/completions"
        if anthropic:
            body.pop("chat_template_kwargs")
            body["thinking"] = {"type": "disabled"}
            path = "/v1/messages"
        events, text = [], ""
        with requests.post(a.base_url + path, json=body, stream=True, timeout=120,
                           headers={"anthropic-version": "2023-06-01"}) as r:
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=1, decode_unicode=True):
                if not line or not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[6:])
                events.append(obj)
                if anthropic:
                    if obj.get("type") == "content_block_delta":
                        text += obj.get("delta", {}).get("text", "")
                elif obj.get("choices"):
                    text += obj["choices"][0].get("delta", {}).get("content", "")
        assert text.strip() == "Hello Orin!", (path, text, events)
        rows.append({"check": "anthropic_stream" if anthropic else "openai_stream", "text": text, "events": events})
        print(path, "stream pass", flush=True)

    for phase in ("decode", "prefill"):
        prompt = "Count from 1 to 100 separated by commas."
        if phase == "prefill":
            prompt = "Read this context, then answer OK: " + "red green blue yellow " * 256
        body = dict(base, stream=True, max_tokens=256,
                    messages=[{"role": "user", "content": prompt}])
        t = time.monotonic()
        with requests.post(a.base_url + "/v1/chat/completions", json=body, stream=True, timeout=120) as r:
            r.raise_for_status()
            if phase == "decode":
                seen = False
                for line in r.iter_lines(chunk_size=1, decode_unicode=True):
                    if line and line.startswith("data: ") and line != "data: [DONE]":
                        obj = json.loads(line[6:])
                        if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                            seen = True
                            break
                assert seen, "No decode token arrived before cancellation"
            else:
                time.sleep(0.5)
        closed = time.monotonic()
        recovery = post(base)
        rows.append({"check": "cancel_" + phase, "time_until_close_s": closed - t,
                     "recovery_elapsed_s": recovery["elapsed_s"], "recovery": recovery["response"]})
        print("cancel_" + phase, "recovery pass", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
