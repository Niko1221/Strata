"""Upstream strata_bench.chat with the supported strata_tune request field added."""
import json
import time
import urllib.request
import urllib.error

def chat(base: str, key: str, messages: list, max_tokens: int, think: bool = False, tools=None, timeout: float = 3600, tune=None) -> dict:
    """One streamed chat request -> {first, total, text, usage, timings, err}.  `first` is the seconds to the first
    non-empty content or reasoning piece (keep-alives and empty deltas ignored)."""
    body = {"model": "strata", "stream": True, "max_tokens": max_tokens, "temperature": 0, "messages": messages}
    if tune is not None:
        body["strata_tune"] = tune
    if tools:
        body["tools"] = tools
    if not think:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    t0 = time.perf_counter()
    first, text, usage, timings, err, pieces = None, [], None, None, None, 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                if j.get("error"):
                    err = str(j["error"])[:300]
                for ch in j.get("choices") or []:
                    dl = ch.get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece:
                        if first is None:
                            first = time.perf_counter() - t0
                        pieces += 1
                        text.append(dl.get("content") or "")
                if j.get("usage"):
                    usage = j["usage"]
                if j.get("timings"):
                    timings = j["timings"]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:                              # noqa: BLE001
        err = repr(e)[:300]
    total = time.perf_counter() - t0
    return {"first": first, "total": total, "text": "".join(text), "usage": usage, "timings": timings, "err": err,
            "pieces": pieces}
