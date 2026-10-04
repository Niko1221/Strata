"""Local authoritative chat archive and bounded, read-only model memory retrieval.

All stored sessions are JSON, including untouched import source and inactive branches.
No network, browser profile access, or tool execution occurs in this module.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
import time

MAX_ARCHIVE_BYTES = 100_000_000
MAX_MESSAGES = 100_000
MAX_RECALL_CHARS = 12_000
ROLES = {"user", "assistant", "system", "tool", "developer"}


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"invalid {name}")
    return value


def _string(value, name, maximum=None, nonempty=False):
    if not isinstance(value, str) or nonempty and not value or maximum is not None and len(value) > maximum:
        raise ValueError(f"invalid {name}")
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _wire(value):
    if not isinstance(value, dict) or value.get("role") not in ROLES:
        raise ValueError("invalid archived model message")
    content = value.get("content")
    if not isinstance(content, (str, list)) and content is not None:
        raise ValueError("invalid model message content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("invalid model content part")
            if part.get("type") == "text":
                _string(part.get("text"), "model text")
            elif part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                _string(part["image_url"].get("url"), "model image URL", nonempty=True)
            else:
                raise ValueError("unsupported model content part")
    calls = value.get("tool_calls")
    if calls is not None:
        if value["role"] != "assistant" or not isinstance(calls, list):
            raise ValueError("invalid archived tool calls")
        for call in calls:
            if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
                raise ValueError("invalid archived tool call")
            _string(call.get("id"), "tool call ID", 512, True)
            _string(call["function"].get("name"), "tool function name", 512, True)
            _string(call["function"].get("arguments"), "tool arguments")
    if value["role"] == "tool":
        _string(value.get("tool_call_id"), "tool result ID", 512, True)


def validate_session(value):
    """Validate public archive fields while retaining all additional JSON source data."""
    if not isinstance(value, dict):
        raise ValueError("invalid chat archive")
    _string(value.get("id"), "chat ID", 1024, True)
    _string(value.get("title"), "chat title", 4096)
    if "revision" in value:
        _integer(value["revision"], "chat revision")
    branches = value.get("branches")
    if not isinstance(branches, list) or not branches or len(branches) > MAX_MESSAGES:
        raise ValueError("invalid chat branches")
    ids, count = set(), 0
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("invalid chat branch")
        identity = _string(branch.get("id"), "branch ID", 1024, True)
        if identity in ids:
            raise ValueError("duplicate chat branch")
        ids.add(identity)
        if "title" in branch:
            _string(branch["title"], "branch title", 4096)
        messages = branch.get("messages")
        if not isinstance(messages, list):
            raise ValueError("invalid archived messages")
        count += len(messages)
        if count > MAX_MESSAGES:
            raise ValueError("archive exceeds message limit")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ROLES:
                raise ValueError("invalid archived message role")
            _string(message.get("text"), "archived message text")
            if "reasoning" in message:
                _string(message["reasoning"], "archived reasoning")
            for field, text_field in (("files", "text"), ("images", "url")):
                if field in message:
                    if not isinstance(message[field], list):
                        raise ValueError("invalid archived attachments")
                    for attachment in message[field]:
                        if not isinstance(attachment, dict):
                            raise ValueError("invalid archived attachment")
                        _string(attachment.get("name"), "attachment name")
                        _string(attachment.get(text_field), "attachment content")
            if "wire" in message:
                if not isinstance(message["wire"], list):
                    raise ValueError("invalid archived model messages")
                for model_message in message["wire"]:
                    _wire(model_message)
        context = branch.get("context")
        if context is not None:
            if not isinstance(context, dict):
                raise ValueError("invalid archived context")
            _string(context.get("summary"), "context summary", nonempty=True)
            _integer(context.get("through"), "context boundary", maximum=len(messages))
    if value.get("activeBranchId") not in ids:
        raise ValueError("selected branch missing from archive")
    try:
        payload = _json(value)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("archive must contain finite JSON data") from error
    if len(payload.encode("utf-8")) > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds 100 MB limit")
    return json.loads(payload), payload


def _search_text(message):
    parts = [message["text"], message.get("reasoning", "")]
    parts.extend(file["text"] for file in message.get("files", []))
    for model_message in message.get("wire", []):
        content = model_message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(part["text"] for part in content if part.get("type") == "text")
    # Stable removal of duplicate render/wire text; image URLs are never indexed.
    return "\n".join(dict.fromkeys(part for part in parts if part))


class ChatArchive:
    def __init__(self, path):
        self.path = str(Path(path))
        self._lock = threading.RLock()
        if self.path == ":memory:":
            raise ValueError("use a persistent SQLite path")
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, payload TEXT NOT NULL, payload_bytes INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    branch_id TEXT NOT NULL, message_index INTEGER NOT NULL,
                    role TEXT NOT NULL, searchable TEXT NOT NULL,
                    PRIMARY KEY(chat_id, branch_id, message_index));
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            db.execute(f"PRAGMA max_page_count={MAX_ARCHIVE_BYTES // page_size}")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _put(db, session, payload):
        size = len(payload.encode("utf-8"))
        total = db.execute("SELECT COALESCE(SUM(payload_bytes),0) FROM sessions WHERE id<>?", (session["id"],)).fetchone()[0]
        if total + size > MAX_ARCHIVE_BYTES:
            raise ValueError("archive exceeds 100 MB limit")
        db.execute("INSERT INTO sessions(id,title,payload,payload_bytes) VALUES(?,?,?,?) "
                   "ON CONFLICT(id) DO UPDATE SET title=excluded.title,payload=excluded.payload,payload_bytes=excluded.payload_bytes",
                   (session["id"], session["title"], payload, size))
        db.execute("DELETE FROM messages WHERE chat_id=?", (session["id"],))
        db.executemany("INSERT INTO messages(chat_id,branch_id,message_index,role,searchable) VALUES(?,?,?,?,?)",
                       ((session["id"], branch["id"], index, message["role"], _search_text(message))
                        for branch in session["branches"] for index, message in enumerate(branch["messages"])))

    def list(self):
        with self._lock, self._connect() as db:
            sessions = [json.loads(row[0]) for row in db.execute("SELECT payload FROM sessions ORDER BY id")]
            active = db.execute("SELECT value FROM meta WHERE key='active'").fetchone()
            return {"sessions": sessions, "activeId": active[0] if active else None}

    def save(self, session, activate=False):
        if type(activate) is not bool:
            raise ValueError("invalid activate flag")
        value, payload = validate_session(session)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT payload FROM sessions WHERE id=?", (value["id"],)).fetchone()
            current_revision = json.loads(previous[0]).get("revision", 0) if previous else 0
            if value.get("revision", 0) != current_revision:
                raise ValueError("This chat changed in another tab. Reload before saving; the newer archive was retained.")
            value["revision"] = current_revision + 1
            payload = _json(value)
            self._put(db, value, payload)
            if activate:
                db.execute("INSERT INTO meta(key,value) VALUES('active',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (value["id"],))
        return value

    def import_sessions(self, sessions):
        if not isinstance(sessions, list) or len(sessions) > MAX_MESSAGES:
            raise ValueError("invalid import sessions")
        validated = [validate_session(session) for session in sessions]
        result = {"added": 0, "skipped": 0, "firstId": None}
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for session, payload in validated:
                if db.execute("SELECT 1 FROM sessions WHERE id=?", (session["id"],)).fetchone():
                    result["skipped"] += 1
                    continue
                session["revision"] = 1
                payload = _json(session)
                self._put(db, session, payload)
                result["added"] += 1
                if result["firstId"] is None:
                    result["firstId"] = session["id"]
        return result

    def seed_legacy(self, messages, context):
        if not isinstance(messages, list):
            raise ValueError("invalid legacy messages")
        first = next((m.get("text") for m in messages if isinstance(m, dict) and m.get("role") == "user" and m.get("text")), "New chat")
        _string(first, "legacy message text")
        session = {"id": "strata:legacy", "title": first[:64], "updated": int(time.time() * 1000),
                   "activeBranchId": "main", "branches": [{"id": "main", "title": "Main", "messages": messages, "context": context}]}
        value, payload = validate_session(session)
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM meta WHERE key='legacy-strata-copied'").fetchone():
                return {"copied": False}
            copied = bool(messages) and not db.execute("SELECT 1 FROM sessions WHERE id='strata:legacy'").fetchone()
            if copied:
                value["revision"] = 1
                payload = _json(value)
                self._put(db, value, payload)
                if not db.execute("SELECT 1 FROM meta WHERE key='active'").fetchone():
                    db.execute("INSERT INTO meta(key,value) VALUES('active','strata:legacy')")
            db.execute("INSERT INTO meta(key,value) VALUES('legacy-strata-copied','true')")
            return {"copied": copied}

    def search(self, query, chat_id=None, limit=5):
        query = _string(query, "search query", 200, True)
        _integer(limit, "search limit", 1, 20)
        if chat_id is not None:
            _string(chat_id, "chat ID", 1024, True)
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        # Literal LIKE supports Chinese phrases without relying on unicode61 tokenization.
        sql = "SELECT m.chat_id,m.branch_id,m.message_index,m.role,m.searchable,s.title FROM messages m JOIN sessions s ON s.id=m.chat_id WHERE m.searchable LIKE ? ESCAPE '\\'"
        args = ["%" + escaped + "%"]
        if chat_id is not None:
            sql += " AND m.chat_id=?"
            args.append(chat_id)
        sql += " ORDER BY m.chat_id,m.branch_id,m.message_index LIMIT ?"
        args.append(limit + 1)
        with self._lock, self._connect() as db:
            rows = db.execute(sql, args).fetchall()
        hits = []
        for identity, branch, index, role, searchable, title in rows[:limit]:
            position = searchable.lower().find(query.lower())
            start = max(0, position - 100)
            hits.append({"chat_id": identity, "branch_id": branch, "message_index": index,
                         "title": title, "role": role, "snippet": searchable[start:start + 500]})
        return {"hits": hits, "truncated": len(rows) > limit}

    def recall(self, chat_id, branch_id, start=0, count=5, offset=0):
        _string(chat_id, "chat ID", 1024, True)
        _string(branch_id, "branch ID", 1024, True)
        _integer(start, "recall start")
        _integer(count, "recall count", 1, 20)
        _integer(offset, "recall offset", maximum=MAX_ARCHIVE_BYTES)
        with self._lock, self._connect() as db:
            row = db.execute("SELECT payload FROM sessions WHERE id=?", (chat_id,)).fetchone()
        if not row:
            raise ValueError("chat not found")
        session = json.loads(row[0])
        branch = next((value for value in session["branches"] if value["id"] == branch_id), None)
        if branch is None:
            raise ValueError("branch not found")
        result = {"chat_id": chat_id, "branch_id": branch_id, "start": start, "records": [], "next_start": None, "next_offset": None, "truncated": False}
        if len(_json(result)) > MAX_RECALL_CHARS - 1024:
            raise ValueError("recall identifiers exceed response budget")
        selected = branch["messages"][start:start + count]
        if offset and not selected:
            raise ValueError("recall offset has no message")
        for index, message in enumerate(selected, start):
            record = {"message_index": index, "role": message["role"], "text": message["text"], "reasoning": message.get("reasoning", "")}
            if message.get("files"):
                record["files"] = [{"name": file["name"], "text": file["text"]} for file in message["files"]]
            # Render-time native tool records are text, never executable instructions to this module.
            if message.get("tools"):
                record["tools"] = message["tools"]
            if message.get("wire"):
                record["wire"] = [{key: value for key, value in item.items() if key in ("role", "tool_calls", "tool_call_id", "content")}
                                  for item in message["wire"]]
                for item in record["wire"]:
                    if isinstance(item.get("content"), list):
                        item["content"] = [{"type": "text", "text": part["text"]} for part in item["content"] if part.get("type") == "text"]
            candidate = {**result, "records": result["records"] + [record]}
            if not offset and len(_json(candidate)) <= MAX_RECALL_CHARS - 64:
                result["records"].append(record)
                continue
            result["truncated"] = True
            # Page a stable JSON projection, including files/reasoning/tool text, without discarding any remainder.
            if not result["records"]:
                projection = _json(record)
                if offset > len(projection):
                    raise ValueError("recall offset exceeds message projection")
                short = {"message_index": index, "role": message["role"], "text": "", "reasoning": "",
                         "page_format": "archived_record_json", "offset": offset, "total_chars": len(projection), "truncated": True}
                lo, hi = 0, len(projection) - offset
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    short["text"] = projection[offset:offset + mid]
                    if len(_json({**result, "records": [short]})) <= MAX_RECALL_CHARS - 64:
                        lo = mid
                    else:
                        hi = mid - 1
                if not lo and offset < len(projection):
                    raise ValueError("recall page metadata exceeds response budget")
                short["text"] = projection[offset:offset + lo]
                short["truncated"] = offset + lo < len(projection)
                result["records"].append(short)
                result["truncated"] = short["truncated"]
                if short["truncated"]:
                    result["next_start"] = index
                    result["next_offset"] = offset + lo
                elif index + 1 < len(branch["messages"]):
                    result["next_start"] = index + 1
                return result
            break
        consumed = len(result["records"])
        if start + consumed < len(branch["messages"]):
            result["next_start"] = start + consumed
        return result

    def close(self):
        """Connections are owned by operations, so no persistent handle needs closing."""
