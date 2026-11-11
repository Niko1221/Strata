#!/usr/bin/env python3
"""Minimal L2 session-persistence controller.

This is a reference implementation that demonstrates the engine's
session-persistence API with the smallest possible orchestration loop.
It is NOT a production gateway — it has no authentication, no CORS
handling, no scheduler, and no message retention.

The engine owns all KV state and transfers. This controller only decides
when to save and restore sessions by calling the HTTP API.

Usage:
    STRATA_ENGINE_URL=http://127.0.0.1:8080 python3 l2/reference_controller.py

Requires:
    - Strata server with --slot-save-path configured
    - STRATA_SESSION_V2=1 for v2 format writes
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENGINE_URL = os.environ.get("STRATA_ENGINE_URL", "http://127.0.0.1:8080")
LISTEN_HOST = os.environ.get("STRATA_CONTROLLER_BIND", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("STRATA_CONTROLLER_PORT", "8190"))
SESSION_DIR = os.environ.get("STRATA_SESSION_DIR", "./sessions")
INDEX_FILE = os.path.join(SESSION_DIR, "index.json")

# ---------------------------------------------------------------------------
# Identity: family (grouping) + chain (branch)
# ---------------------------------------------------------------------------

def family(body):
    """Stable grouping key: tools + system/developer prefix (first user excluded)."""
    msgs = body.get("messages") or []
    head = {"tools": body.get("tools") or []}
    for m in msgs:
        if m.get("role") in ("system", "developer"):
            head.setdefault("sys", []).append(str(m.get("content", ""))[:2000])
        elif m.get("role") == "user":
            break
    return hashlib.sha1(
        json.dumps(head, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def chain_hashes(messages):
    """Per-message content hashes for branch detection."""
    return [
        hashlib.sha1(
            json.dumps(m, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]
        for m in (messages or [])
    ]


def chain_id(hashes):
    """Branch identity: chained hash of all message hashes."""
    return hashlib.sha1("".join(hashes).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Index: family → {chain → filename}
# ---------------------------------------------------------------------------

def load_index():
    try:
        with open(INDEX_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_index(idx):
    os.makedirs(SESSION_DIR, exist_ok=True)
    tmp = INDEX_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f)
    os.replace(tmp, INDEX_FILE)


def find_snapshot(idx, fp, hs):
    """Exact chain first, then deepest common-prefix ancestor."""
    fam = idx.get(fp, {})
    exact = fam.get(chain_id(hs))
    if exact:
        return exact
    best, best_len = None, 0
    for cid, fname in fam.items():
        entry_hs = fam.get(cid + "_hashes")
        if not entry_hs:
            continue
        common = 0
        for a, b in zip(hs, entry_hs):
            if a != b:
                break
            common += 1
        if common > best_len:
            best, best_len = fname, common
    return best


# ---------------------------------------------------------------------------
# Engine HTTP helpers
# ---------------------------------------------------------------------------

def engine_request(method, path, body=None, timeout=120):
    """Send a request to the engine, return (status, json|None)."""
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(
        ENGINE_URL + path, method=method, data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                return r.status, None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except (ValueError, UnicodeDecodeError):
            return e.code, None
    except Exception:
        return 0, None


def probe_cache(body):
    """Read-only: how much of this request would the cache return?"""
    probe_body = {
        "messages": body.get("messages") or [],
    }
    if body.get("tools") is not None:
        probe_body["tools"] = body["tools"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        probe_body["chat_template_kwargs"] = body["chat_template_kwargs"]
    status, result = engine_request("POST", "/v1/cache/probe", probe_body, timeout=10)
    return result if status == 200 else None


def preload_as_parked(filename, body, pin=True):
    """Read a session file as a PARKED entry (prefix-matched to this request)."""
    pre_body = {
        "filename": filename,
        "messages": body.get("messages") or [],
        "pin": pin,
    }
    if body.get("tools") is not None:
        pre_body["tools"] = body["tools"]
    if isinstance(body.get("chat_template_kwargs"), dict):
        pre_body["chat_template_kwargs"] = body["chat_template_kwargs"]
    status, result = engine_request("POST", "/v1/cache/preload", pre_body, timeout=180)
    return result if status == 200 else None


def save_incremental(filename):
    """Incrementally save the current active session to disk."""
    status, result = engine_request(
        "POST", "/slots/0?action=saveinc", {"filename": filename}, timeout=180
    )
    return result if status == 200 else None


def unpin_serial(serial):
    """Release a PARKED entry pin after use."""
    if serial is None:
        return
    engine_request("POST", "/v1/cache/unpin", {"serial": int(serial)}, timeout=5)


# ---------------------------------------------------------------------------
# Orchestration state
# ---------------------------------------------------------------------------

state = {
    "current_family": None,
    "current_chain": None,
    "current_file": None,
    "dirty": False,
}
state_lock = __import__("threading").Lock()


def should_save_outgoing(new_family):
    """True when a different conversation family arrives."""
    return (
        state["current_family"] is not None
        and state["current_family"] != new_family
        and state["dirty"]
    )


def handle_turn(body):
    """The complete L2 decision loop.

    1. If switching away from a dirty session → saveinc it.
    2. Look up a snapshot for the incoming conversation.
    3. If found → preload as PARKED (the engine's normal request path
       will take/borrow from it automatically).
    4. Forward the generation request.
    5. Mark the outgoing session as dirty for the next switch.
    """
    msgs = body.get("messages") or []
    fp = family(body)
    hs = chain_hashes(msgs)
    ch = chain_id(hs)
    idx = load_index()

    with state_lock:
        # 1. Save the outgoing session if switching
        if should_save_outgoing(fp):
            old_file = state["current_file"]
            if old_file:
                result = save_incremental(old_file)
                if result:
                    print(f"[l2] saved: {old_file} ({result.get('tokens', '?')} tokens)")
                state["dirty"] = False

        # 2. Find and preload a snapshot for the incoming conversation
        snapshot = find_snapshot(idx, fp, hs)
        preload_result = None
        if snapshot:
            preload_result = preload_as_parked(snapshot, body)
            if preload_result and preload_result.get("serial"):
                print(
                    f"[l2] preload: {preload_result.get('tokens', '?')} tokens, "
                    f"partial={preload_result.get('partial', '?')}, "
                    f"serial={preload_result['serial']}"
                )

    return preload_result


def after_turn(body, response_ok):
    """Called after the engine responds; marks session dirty and saves."""
    msgs = body.get("messages") or []
    fp = family(body)
    hs = chain_hashes(msgs)
    ch = chain_id(hs)

    with state_lock:
        state["current_family"] = fp
        state["current_chain"] = ch
        state["dirty"] = True

        # Build a stable filename for this chain
        fname = f"{fp}-{ch}.bin"
        state["current_file"] = fname

        # Update the index
        idx = load_index()
        idx.setdefault(fp, {})[ch] = fname
        idx[fp][ch + "_hashes"] = hs
        save_index(idx)


# ---------------------------------------------------------------------------
# HTTP proxy (minimal — forwards /v1/chat/completions and handles L2 logic)
# ---------------------------------------------------------------------------

class ControllerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._passthrough()
            return

        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            body = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            self._passthrough_raw(raw)
            return

        # L2 orchestration: save outgoing, preload incoming
        preload_result = handle_turn(body)

        # Forward to engine
        status, resp_body = self._forward(raw)

        # Track state after successful generation
        if status == 200:
            after_turn(body, True)
            # Unpin the preloaded entry (the engine consumed it or we're done)
            if preload_result and preload_result.get("serial"):
                unpin_serial(preload_result.get("serial"))

        self._relay(status, resp_body)

    def do_GET(self):
        self._passthrough()

    def _forward(self, raw):
        """Forward the raw request body to the engine."""
        req = urllib.request.Request(
            ENGINE_URL + self.path,
            method="POST",
            data=raw,
            headers={
                "Content-Type": "application/json",
                **{k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length")},
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            return 502, json.dumps({"error": {"message": str(e)}}).encode()

    def _passthrough(self):
        """Pass non-chat requests directly to the engine."""
        req = urllib.request.Request(
            ENGINE_URL + self.path, method=self.command
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = r.read()
                self._relay(r.status, resp)
        except urllib.error.HTTPError as e:
            self._relay(e.code, e.read())
        except Exception as e:
            self._relay(502, json.dumps({"error": {"message": str(e)}}).encode())

    def _passthrough_raw(self, raw):
        status, resp = self._forward(raw)
        self._relay(status, resp)

    def _relay(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    os.makedirs(SESSION_DIR, exist_ok=True)
    print(f"[l2] reference controller on {LISTEN_HOST}:{LISTEN_PORT} → {ENGINE_URL}")
    print(f"[l2] session dir: {SESSION_DIR}")
    print(f"[l2] this is a demonstration, not a production gateway")
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ControllerHandler)
    server.serve_forever()
