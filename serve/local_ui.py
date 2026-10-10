"""Server-side persistence and MCP helpers for the Strata web UI.

Conversation files deliberately live outside browser storage so they survive browser/profile resets.
IDs are UUIDs and never become arbitrary paths. MCP edits mirror the backend HTTP and stdio transports
while keeping saved secret values out of browser responses.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONS = ROOT / "conversations"
MAX_CONVERSATION_BYTES = 64 * 1024 * 1024
_ID = re.compile(r"^[0-9a-f]{32}$")
_MCP_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _conv_path(cid: str) -> Path:
    cid = str(cid or "").lower()
    if not _ID.fullmatch(cid):
        raise ValueError("invalid conversation id")
    return CONVERSATIONS / f"{cid}.json"


def _title(messages) -> str:
    for m in messages if isinstance(messages, list) else []:
        if isinstance(m, dict) and m.get("role") == "user":
            text = str(m.get("text") or m.get("content") or "")
            text = " ".join(text.split())
            if text:
                return text[:72]
    return "New conversation"


def _metadata(data: dict) -> dict:
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    return {
        "id": str(data.get("id") or ""),
        "title": str(data.get("title") or "New conversation"),
        "created_at": float(data.get("created_at") or 0),
        "updated_at": float(data.get("updated_at") or 0),
        "message_count": len(messages),
    }


def list_conversations() -> list[dict]:
    CONVERSATIONS.mkdir(parents=True, exist_ok=True)
    out = []
    for p in CONVERSATIONS.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and _ID.fullmatch(str(data.get("id") or "")):
                out.append(_metadata(data))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda x: (x["updated_at"], x["created_at"]), reverse=True)
    return out


def get_conversation(cid: str) -> dict:
    p = _conv_path(cid)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as e:
        raise ValueError(f"conversation cannot be read: {e}") from None
    if not isinstance(data, dict) or data.get("id") != str(cid).lower() or not isinstance(data.get("messages"), list):
        raise ValueError("invalid conversation file")
    return data


def delete_conversation(cid: str) -> dict:
    """Delete one persisted local-UI conversation by its validated UUID id."""
    p = _conv_path(cid)
    try:
        p.unlink()
    except FileNotFoundError:
        raise
    except OSError as e:
        raise ValueError(f"conversation cannot be deleted: {e}") from None
    return {"id": str(cid).lower(), "deleted": True}


def save_conversation(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("send a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    cid = str(payload.get("id") or "").lower()
    if cid:
        p = _conv_path(cid)
    else:
        cid = uuid.uuid4().hex
        p = _conv_path(cid)
    now = time.time()
    created = now
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
            created = float(old.get("created_at") or now) if isinstance(old, dict) else now
        except (OSError, ValueError):
            pass
    title = str(payload.get("title") or "").strip()[:96] or _title(messages)
    data = {"id": cid, "title": title, "created_at": created, "updated_at": now, "messages": messages}
    raw = (json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(raw) > MAX_CONVERSATION_BYTES:
        raise ValueError(f"conversation is too large ({len(raw)} bytes; limit {MAX_CONVERSATION_BYTES})")
    CONVERSATIONS.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(raw)
        os.replace(tmp, p)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return _metadata(data)


def _mcp_effective(cfg: dict) -> dict[str, dict]:
    """Effective run-config MCP entries, with mcpServers taking the same precedence as serve.mcp."""
    out = {}
    if not isinstance(cfg, dict):
        return out
    for key in ("mcp_servers", "mcpServers"):
        block = cfg.get(key)
        if isinstance(block, dict):
            out.update({str(name): item for name, item in block.items() if isinstance(item, dict)})
    return out


def mcp_config_view(cfg: dict) -> list[dict]:
    """Browser-safe MCP config: launch shape is visible, secret header/env values never are."""
    out = []
    for name, item in _mcp_effective(cfg).items():
        if item.get("url"):
            out.append({"name": name, "transport": "http", "url": str(item["url"]),
                        "has_headers": bool(item.get("headers"))})
        elif item.get("command"):
            env = item.get("env") if isinstance(item.get("env"), dict) else {}
            out.append({"name": name, "transport": "stdio", "command": str(item["command"]),
                        "args": [str(x) for x in (item.get("args") or [])],
                        "cwd": str(item.get("cwd") or ""),
                        "has_env": bool(env), "env_keys": sorted(str(k) for k in env)})
    return out


def validate_mcp_name(name) -> str:
    name = str(name or "").strip()
    if not _MCP_NAME.fullmatch(name):
        raise ValueError("server name: use 1-64 letters, numbers, dot, underscore or dash")
    return name


def _validate_http_url(url) -> str:
    url = str(url or "").strip()
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.fragment:
        raise ValueError("MCP URL must be an http(s) URL without credentials or a fragment")
    return url


def _stdio_entry(payload: dict) -> dict:
    command = str(payload.get("command") or "").strip()
    if not command:
        raise ValueError("stdio MCP server needs a command")
    args = payload.get("args", [])
    if args is None:
        args = []
    if not isinstance(args, list) or any(isinstance(x, (dict, list)) for x in args):
        raise ValueError("stdio args must be a JSON array of scalar values")
    args = [str(x) for x in args]
    cwd = payload.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("stdio cwd must be a string")
    cwd = str(cwd or "").strip()
    entry = {"command": command}
    if args:
        entry["args"] = args
    if cwd:
        entry["cwd"] = cwd
    if "env" in payload:
        env = payload.get("env")
        if not isinstance(env, dict):
            raise ValueError("stdio env must be a JSON object")
        entry["env"] = {str(k): str(v) for k, v in env.items()}
    return entry


def upsert_mcp_server(cfg: dict, payload: dict) -> tuple[str, dict]:
    """Validate and save one HTTP or stdio MCP entry without leaking hidden secrets across a changed endpoint."""
    if not isinstance(cfg, dict) or not isinstance(payload, dict):
        raise ValueError("send a JSON object")
    name = validate_mcp_name(payload.get("name"))
    transport = str(payload.get("transport") or ("http" if payload.get("url") else "stdio")).strip().lower()
    old = _mcp_effective(cfg).get(name) or {}
    if transport == "http":
        url = _validate_http_url(payload.get("url"))
        entry = {"url": url}
        # A saved Authorization/header block is retained only for the exact same endpoint.
        if old.get("url") == url and old.get("headers"):
            entry["headers"] = old["headers"]
    elif transport == "stdio":
        entry = _stdio_entry(payload)
        old_args = [str(x) for x in (old.get("args") or [])] if isinstance(old.get("args") or [], list) else []
        same_launch = (old.get("command") == entry["command"] and old_args == entry.get("args", [])
                       and str(old.get("cwd") or "") == str(entry.get("cwd") or ""))
        # Env values are intentionally never echoed to the browser. Blank/omitted env preserves them only when
        # the entire launch identity is unchanged; changing command/args/cwd cannot silently forward credentials.
        if "env" not in payload and same_launch and old.get("env"):
            entry["env"] = old["env"]
    else:
        raise ValueError("MCP transport must be 'http' or 'stdio'")

    block = cfg.get("mcp_servers")
    if not isinstance(block, dict):
        block = {}
        cfg["mcp_servers"] = block
    block[name] = entry
    legacy = cfg.get("mcpServers")
    if isinstance(legacy, dict):
        legacy.pop(name, None)  # otherwise this later-precedence block would override the edited entry
    return name, entry


def remove_mcp_server(cfg: dict, name) -> bool:
    """Remove one named MCP entry from either supported config block without touching the others."""
    name = validate_mcp_name(name)
    found = False
    for key in ("mcp_servers", "mcpServers"):
        block = cfg.get(key) if isinstance(cfg, dict) else None
        if isinstance(block, dict) and name in block:
            del block[name]
            found = True
    return found
