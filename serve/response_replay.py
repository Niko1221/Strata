"""Deployment-key encryption of Strata's own stateless reasoning replay items.

No response records are stored. Reuse the environment's private key across server
restarts; rotating it invalidates old tokens. Foreign tokens are never interpreted.
"""
from __future__ import annotations

import json
import os

from cryptography.fernet import Fernet, InvalidToken

PREFIX = "strata-r1."
KEY_ENV = "STRATA_RESPONSES_REPLAY_KEY"


class ReplayCodec:
    def __init__(self, key):
        self.cipher = Fernet(key)

    @classmethod
    def load(cls):
        key = os.environ.get(KEY_ENV)
        if not key:
            raise ValueError(f"set {KEY_ENV} to a Fernet key; see docs/RESPONSES.md")
        return cls(key.encode("ascii"))  # malformed/missing keys fail startup; never silently rotate

    def seal(self, model, item):
        payload = {"version": 1, "model": model, "id": item["id"], "content": item["content"],
                   "summary": item["summary"], "status": item["status"]}
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return PREFIX + self.cipher.encrypt(raw).decode("ascii")

    def restore(self, model, item):
        token = item.get("encrypted_content")
        if not isinstance(token, str) or not token.startswith(PREFIX):
            raise ValueError("expected a Strata-issued reasoning replay token")
        try:
            payload = json.loads(self.cipher.decrypt(token[len(PREFIX):].encode("ascii")))
        except (InvalidToken, ValueError, UnicodeError) as exc:
            raise ValueError("reasoning replay token is invalid or belongs to a different deployment key") from exc
        if payload.get("version") != 1 or payload.get("model") != model or payload.get("id") != item.get("id"):
            raise ValueError("reasoning replay token does not match this model and item ID")
        if payload.get("status") != "completed":
            raise ValueError("only completed reasoning can be replayed")
        for key in ("content", "summary", "status"):
            if key in item and item[key] != payload[key]:
                raise ValueError(f"visible reasoning {key} conflicts with its replay token")
        return {**item, "content": payload["content"], "summary": payload["summary"]}
