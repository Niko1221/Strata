"""Bind native conversation reuse to an authenticated key and a client chat ID.

The process-local secret and keys never reach native logs or request payloads.
Missing authentication/session means no cache reuse, not a shared default owner.
"""
import hashlib
import hmac
import json
import re
import secrets

from serve.native_family import native_admission

_SECRET = secrets.token_bytes(32)


def request_session(body, affinity=None):
    """Explicit chat IDs win; Harness affinity is a separate, bounded namespace."""
    if "strata_cache_session" in body:
        return body["strata_cache_session"]
    if not isinstance(affinity, str) or not 1 <= len(affinity) <= 128:
        return None
    if any(ord(c) < 33 or ord(c) > 126 for c in affinity):
        return None
    return "harness_" + hashlib.sha256(affinity.encode()).hexdigest()


def conversation_scope(api_key, session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", session_id):
        if session_id is not None:
            raise ValueError("strata_cache_session must be a 16–128 character conversation ID")
        return None
    if not api_key:
        return None
    payload = json.dumps([api_key, session_id], separators=(",", ":")).encode()
    return hmac.new(_SECRET, payload, hashlib.sha256).hexdigest()


def required_capability(cfg: dict) -> int:
    """Choose the required acknowledgement, not evidence that the engine has it."""
    return native_admission(cfg).private_cache


def capability_confirmed(info, required):
    return not required or info.get("q36_private_cache") == int(required)
