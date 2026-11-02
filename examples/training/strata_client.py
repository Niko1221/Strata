"""Strata 的 HTTP 客户端。只做一件事：把 prompt 送进引擎，拿回序列与文本。

两个必须处理的现实问题：
  1. WSL 里 `http_proxy=127.0.0.1:1080` 会把 127.0.0.1 的请求也交给 privoxy，
     直接 urllib 会得到 `HTTP 500 Internal Privoxy Error` —— 回环必须绕开代理。
  2. 老引擎可能不返回 token id（q36 在补上 `LP` 行之前就是），这时只能回退到
     文本重编码。重编码与真实采样可能有个别边界差异，所以返回值里明确标了
     `tokens_exact`，训练时据此决定要不要信任 ids。当前两个引擎（generate.cpp
     的 serve 循环、q36_engine.cu）都产 LP，正常情况下 `tokens_exact` 为真。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def chat(base_url: str, messages: list[dict], *, model: str, max_tokens: int = 128,
         temperature: float = 1.0, top_p: float = 0.95, seed: int | None = None,
         timeout: float = 600.0, enable_thinking: bool | None = None) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,
        "top_logprobs": 0,
        "strata_rollout": True,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(req, timeout=timeout) as resp:
        raw = json.loads(resp.read())

    choice = raw["choices"][0]
    message = choice.get("message") or {}
    rollout = raw.get("strata_rollout") or {}
    tokens = rollout.get("tokens") or []
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_ids": rollout.get("prompt_ids") or [],
        "token_ids": [t["id"] for t in tokens],
        "token_logprobs": [t["logprob"] for t in tokens],
        "fields": [t.get("field") for t in tokens],
        "tokens_exact": bool(tokens) and bool(rollout.get("prompt_ids")),
        "usage": raw.get("usage") or {},
    }
