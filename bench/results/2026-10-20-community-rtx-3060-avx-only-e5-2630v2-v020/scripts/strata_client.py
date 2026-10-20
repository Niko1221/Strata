"""Small HTTP client for a running Strata server (stdlib only), shared by quality_suite.py and soak.py."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8080"


def body(text: str, max_tokens: int) -> dict:
    """The request the RTX 3060 community report (2026-10-09, Strata 0.1.41) sent: greedy, reasoning off."""
    return {"model": "strata", "messages": [{"role": "user", "content": text}], "temperature": 0,
            "reasoning_effort": "none", "max_tokens": max_tokens}


def chat(req: dict, timeout: float = 900, url: str = URL) -> dict:
    """POST /v1/chat/completions (not streamed). Returns a flat row: ok, error, text, tool_calls, token counts,
    the engine's own timings and the client's elapsed seconds."""
    t0 = time.time()
    row: dict = {"epoch": t0}
    try:
        r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            d = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        row.update(ok=False, error=f"{type(e).__name__}: {e}", client_s=round(time.time() - t0, 3))
        return row
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    u, t = d.get("usage") or {}, d.get("timings") or {}
    row.update(ok=True, error=None, client_s=round(time.time() - t0, 3),
               finish_reason=(d.get("choices") or [{}])[0].get("finish_reason"),
               text=msg.get("content") or "", tool_calls=msg.get("tool_calls") or [],
               prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
               cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", t.get("cache_n")),
               prompt_n=t.get("prompt_n"), prompt_ms=t.get("prompt_ms"), prompt_tps=t.get("prompt_per_second"),
               predicted_n=t.get("predicted_n"), predicted_ms=t.get("predicted_ms"),
               decode_tps=t.get("predicted_per_second"))
    return row


def health(url: str = URL) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def status(url: str = URL) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/v1/status", timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
