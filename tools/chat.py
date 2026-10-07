"""One chat request to the running server, for `make chat`: python tools/chat.py

Reads PROMPT, MAX_TOKENS, EFFORT, REASONING_BUDGET, URL and API_KEY from the environment (the Makefile exports them).
It checks the numbers before anything is sent, so a bad value never reaches the server, and it exits 1 when there is
no answer: a refused request, a server that cannot be reached, or a reply that ran out of tokens while thinking.
The answer goes to stdout, the token count to stderr, so `make chat > answer.txt` holds the answer only.

REASONING_BUDGET (the server's reasoning_budget_tokens, docs/DETAILS.md): empty = the server's default; 0 = no cap
(thinking is not turned off: EFFORT=none does that); N = at most N tokens of thinking, then the server closes it and
the model answers.  The thinking counts toward MAX_TOKENS, so N must leave room: ANSWER_ROOM tokens at least, for
the server's wrap-up line and the start of the answer.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

ANSWER_ROOM = 64


class BadInput(ValueError):
    pass


def tokens(name: str, value: str | None, *, optional: bool = False) -> int | None:
    """A whole number of tokens from an environment value; None when optional and empty."""
    value = (value or "").strip()
    if optional and value == "":
        return None
    if not re.fullmatch(r"[0-9]{1,9}", value):
        raise BadInput(f"{name} must be a whole number of tokens, not {value!r}")
    return int(value, 10)


def request_body(env) -> dict:
    max_tokens = tokens("MAX_TOKENS", env.get("MAX_TOKENS"))
    if max_tokens < 1:
        raise BadInput("MAX_TOKENS must be at least 1")
    budget = tokens("REASONING_BUDGET", env.get("REASONING_BUDGET"), optional=True)
    if budget and budget > max_tokens - ANSWER_ROOM:
        raise BadInput(f"REASONING_BUDGET ({budget}) leaves no room to answer within MAX_TOKENS ({max_tokens}): keep "
                       f"it at {max_tokens - ANSWER_ROOM} or less, or raise MAX_TOKENS")
    body = {"model": "strata", "max_tokens": max_tokens, "reasoning_effort": env.get("EFFORT") or "none",
            "messages": [{"role": "user", "content": env.get("PROMPT", "")}]}
    if budget is not None:
        body["reasoning_budget_tokens"] = budget
    return body


def answer(reply: dict) -> tuple[str, str]:
    """(the answer, a status line) from a chat completion; BadInput when it holds no answer."""
    choice = (reply.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    t = reply.get("timings") or {}
    status = f"[{t.get('predicted_n', '?')} tokens, {t.get('predicted_per_second') or 0:.1f} tok/s]"
    if not text.strip():
        if choice.get("finish_reason") == "length":
            raise BadInput("no answer: the reply reached MAX_TOKENS while still thinking; raise MAX_TOKENS or set "
                           f"REASONING_BUDGET below it {status}")
        raise BadInput(f"no answer in the reply: {json.dumps(reply)[:500]}")
    return text, status


def main(env=os.environ) -> int:
    try:
        body = request_body(env)
        req = urllib.request.Request(env.get("URL", "http://127.0.0.1:8080").rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if env.get("API_KEY"):
            req.add_header("Authorization", f"Bearer {env['API_KEY']}")      # RFC 6750 section 2.1
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                reply = json.load(r)
        except urllib.error.HTTPError as e:
            raise BadInput(f"the server said {e.code}: {e.read().decode(errors='replace')[:500]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise BadInput(f"could not reach the server at {env.get('URL')}: {e} (make status)") from None
        text, status = answer(reply)
    except BadInput as e:
        print(f"chat: {e}", file=sys.stderr)
        return 1
    print(text)
    print(status, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
