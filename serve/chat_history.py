"""Local disk conversations: gzip records, bounded pages, and a small text index.

No model is loaded by these routes. Original messages remain on disk; summaries
are separate state. The browser retains only the selected conversation window.
"""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid
import zlib

MAX_BYTES = 64 * 1024 * 1024
PAGE = 40
PAGE_BYTES = 4 * 1024 * 1024
INDEX_CHARS = 4000
EMPTY_MEMORY = {"through": 0, "summary": "", "count": 0}


class HistoryError(ValueError):
    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(value):
    raw = encoded(value)
    if len(raw) > MAX_BYTES:
        raise HistoryError("The conversation is too large. Save it in smaller parts.", 413)
    return gzip.compress(raw, compresslevel=6, mtime=0), len(raw), digest(raw)


def unpacked(blob):
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HistoryError("The decompressed history exceeds the size limit.", 413)
    return json.loads(raw)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
        raise HistoryError("Invalid chat ID.")
    return value


def integer(value, minimum=0, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise HistoryError("Invalid history range.")
    return value


def attachments(value):
    if not isinstance(value, list) or len(value) > 100:
        raise HistoryError("Invalid attachment format.")
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name", ""), str):
            raise HistoryError("Invalid attachment format.")
        if "text" in item and not isinstance(item["text"], str):
            raise HistoryError("Invalid attachment content.")
        if item.get("url") and (not isinstance(item["url"], str) or
                not re.match(r"^data:image/(?:png|jpeg|jpg|gif|webp|bmp|avif);base64,[A-Za-z0-9+/=\s]+$", item["url"])):
            raise HistoryError("Images must be embedded image data saved on this PC.")
    return value


def message(value):
    if not isinstance(value, dict) or value.get("role") not in ("user", "assistant") or not isinstance(value.get("text"), str):
        raise HistoryError("Invalid conversation format.")
    for field in ("reasoning", "error", "meta"):
        if field in value and value[field] is not None and not isinstance(value[field], str):
            raise HistoryError("Invalid conversation format.")
    for field in ("files", "images"):
        if field in value:
            attachments(value[field])
    if "tools" in value and (not isinstance(value["tools"], list) or any(not isinstance(t, dict) for t in value["tools"])):
        raise HistoryError("Invalid tool history format.")
    return value


def state(memory, draft, draft_attachments, total):
    if not isinstance(memory, dict) or not isinstance(memory.get("summary", ""), str):
        raise HistoryError("Invalid conversation summary.")
    through = integer(memory.get("through", 0), maximum=total)
    if through and not memory.get("summary", "").strip():
        raise HistoryError("The summary has no content.")
    count = integer(memory.get("count", 0))
    if not isinstance(draft, str):
        raise HistoryError("Invalid draft format.")
    return {"memory": {"through": through, "summary": memory.get("summary", ""), "count": count},
            "draft": draft, "draftAttachments": attachments(draft_attachments)}


def search_text(item):
    # Reasoning and large tool payloads stay compressed, never duplicated in the index.
    text = item["text"]
    if len(text) > 2500:
        text = text[:1500] + "\n…\n" + text[-1000:]
    if item.get("role") == "user":
        text += "\n" + "\n".join(f.get("name", "") + "\n" + f.get("text", "")[:600] + "\n" + f.get("text", "")[-600:] for f in item.get("files", []))
    return text[:INDEX_CHARS]


EMPTY_STATE_SHA = digest(encoded({"memory": EMPTY_MEMORY, "draft": "", "draftAttachments": []}))


def query_terms(query):
    words = re.findall(r"[\w./:\\-]{3,80}", query, re.UNICODE)
    terms = []
    for word in words:
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", word):
            # Trigrams find Japanese text without requiring a tokenizer/model download.
            word = word[:48]
            pieces = [word[i:i + 3] for i in range(len(word) - 2)]
        else:
            pieces = [word]
        for piece in pieces:
            if piece not in terms and piece not in ("してく", "てくだ", "くださ", "ださい", "につい", "ついて", "教えて", "えてく", "what", "with", "this", "that", "please", "only", "Reply"):
                terms.append(piece)
    return terms[:24]


class ChatHistory:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "history.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS chats(
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL, revision INTEGER NOT NULL, message_count INTEGER NOT NULL,
                    state BLOB NOT NULL, state_sha TEXT NOT NULL, state_raw INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chats_updated ON chats(updated_at DESC, id);
                CREATE INDEX IF NOT EXISTS chats_fingerprint ON chats(fingerprint);
                CREATE TABLE IF NOT EXISTS messages(
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, seq INTEGER NOT NULL,
                    body BLOB NOT NULL, sha TEXT NOT NULL, raw_bytes INTEGER NOT NULL,
                    PRIMARY KEY(chat_id, seq));
                CREATE TABLE IF NOT EXISTS search_text(
                    rowid INTEGER PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL, text TEXT NOT NULL, UNIQUE(chat_id, seq));
                CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                    text, content='search_text', content_rowid='rowid', tokenize='trigram');
                CREATE TRIGGER IF NOT EXISTS search_insert AFTER INSERT ON search_text BEGIN
                    INSERT INTO search_fts(rowid,text) VALUES(new.rowid,new.text); END;
                CREATE TRIGGER IF NOT EXISTS search_delete AFTER DELETE ON search_text BEGIN
                    INSERT INTO search_fts(search_fts,rowid,text) VALUES('delete',old.rowid,old.text); END;
                CREATE TRIGGER IF NOT EXISTS search_update AFTER UPDATE ON search_text BEGIN
                    INSERT INTO search_fts(search_fts,rowid,text) VALUES('delete',old.rowid,old.text);
                    INSERT INTO search_fts(rowid,text) VALUES(new.rowid,new.text); END;
                CREATE TABLE IF NOT EXISTS imports(source_sha TEXT PRIMARY KEY, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS save_receipts(token TEXT PRIMARY KEY, request_sha TEXT NOT NULL, result TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def metadata(row):
        return {"id": row["id"], "title": row["title"], "createdAt": row["created_at"],
                "updatedAt": row["updated_at"], "revision": row["revision"], "messageCount": row["message_count"],
                "empty": row["message_count"] == 0 and row["state_sha"] == EMPTY_STATE_SHA}

    @staticmethod
    def find(db, chat_id):
        row = db.execute("SELECT * FROM chats WHERE id=?", (identifier(chat_id),)).fetchone()
        if row is None:
            raise HistoryError("Chat not found.", 404)
        return row

    def catalog(self, query="", offset=0, limit=50):
        offset, limit = integer(offset), integer(limit, 1, 100)
        if not isinstance(query, str) or len(query) > 512:
            raise HistoryError("The search query is too long.")
        with self.connect() as db:
            where, args = "", []
            if query.strip():
                # Search only the bounded text index, never decompress every transcript.
                escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where = "WHERE title LIKE ? ESCAPE '\\' OR id IN (SELECT chat_id FROM search_text WHERE text LIKE ? ESCAPE '\\')"
                args = ["%" + escaped + "%"] * 2
            total = db.execute("SELECT COUNT(*) FROM chats " + where, args).fetchone()[0]
            rows = db.execute("SELECT id,title,created_at,updated_at,revision,message_count,state_sha FROM chats " + where + " ORDER BY updated_at DESC,id LIMIT ? OFFSET ?",
                              [*args, limit, offset]).fetchall()
            return {"chats": [self.metadata(r) for r in rows], "total": total, "offset": offset,
                    "hasMore": offset + len(rows) < total}

    def page(self, chat_id, start=None, limit=PAGE, include_state=True, end=None):
        limit = integer(limit, 1, 100)
        with self.connect() as db:
            row = self.find(db, chat_id)
            if start is not None and end is not None:
                raise HistoryError("Specify either the start or end of the history range.")
            backward = start is None
            boundary = row["message_count"] if end is None else integer(end, maximum=row["message_count"])
            at = 0 if backward else integer(start, maximum=row["message_count"])
            candidates = db.execute("SELECT seq,raw_bytes FROM messages WHERE chat_id=? AND " +
                ("seq<? ORDER BY seq DESC" if backward else "seq>=? ORDER BY seq") + " LIMIT ?",
                (chat_id, boundary if backward else at, limit)).fetchall()
            chosen, size = [], 0
            for candidate in candidates:
                if chosen and size + candidate["raw_bytes"] > PAGE_BYTES:
                    break
                chosen.append(candidate["seq"]); size += candidate["raw_bytes"]
            at = min(chosen) if chosen else boundary if backward else at
            stop = max(chosen) + 1 if chosen else at
            rows = db.execute("SELECT seq,body FROM messages WHERE chat_id=? AND seq>=? AND seq<? ORDER BY seq",
                              (chat_id, at, stop)).fetchall()
            return {**self.metadata(row), **(unpacked(row["state"]) if include_state else {}), "offset": at,
                    "messages": [unpacked(r["body"]) for r in rows]}

    def _put_message(self, db, chat_id, seq, item):
        blob, size, sha = packed(message(item))
        old = db.execute("SELECT sha FROM messages WHERE chat_id=? AND seq=?", (chat_id, seq)).fetchone()
        if old and old[0] == sha:
            return False
        db.execute("INSERT INTO messages VALUES(?,?,?,?,?) ON CONFLICT(chat_id,seq) DO UPDATE SET body=excluded.body,sha=excluded.sha,raw_bytes=excluded.raw_bytes",
                   (chat_id, seq, blob, sha, size))
        db.execute("INSERT INTO search_text(chat_id,seq,text) VALUES(?,?,?) ON CONFLICT(chat_id,seq) DO UPDATE SET text=excluded.text",
                   (chat_id, seq, search_text(item)))
        return True

    @staticmethod
    def _fingerprint(db, chat_id, state_sha):
        hashes = [r[0] for r in db.execute("SELECT sha FROM messages WHERE chat_id=? ORDER BY seq", (chat_id,))]
        return digest(encoded([hashes, state_sha]))

    def save(self, body):
        if not isinstance(body, dict):
            raise HistoryError("Invalid chat format.")
        chat_id = identifier(body.get("id"))
        token = identifier(body["writeId"]) if body.get("writeId") else None
        request_sha = digest(encoded(body))
        revision = integer(body.get("revision", 0))
        items = body.get("messages", [])
        if not isinstance(items, list) or len(items) > 10000:
            raise HistoryError("The conversation range to save is too large.")
        for item in items:
            message(item)
        total = integer(body.get("total", 0))
        start = body.get("start")
        if start is not None:
            start = integer(start, maximum=total)
            if start + len(items) != total:
                raise HistoryError("The save range does not match the end of the conversation.")
        elif items:
            raise HistoryError("Specify the range to save.")
        current_state = state(body.get("memory", EMPTY_MEMORY), body.get("draft", ""), body.get("draftAttachments", []), total)
        blob, size, sha = packed(current_state)
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute("SELECT request_sha,result FROM save_receipts WHERE token=?", (token,)).fetchone() if token else None
            if receipt:
                if receipt["request_sha"] != request_sha:
                    raise HistoryError("The save identifier does not match.", 409)
                return {**json.loads(receipt["result"]), "replayed": True}
            old = db.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            now = int(time.time() * 1000)
            if old is None:
                if revision or start not in (0, None) or total != len(items):
                    raise HistoryError("Invalid save range for a new chat.", 409)
                db.execute("INSERT INTO chats VALUES(?,?,?,?,?,?,?,?,?,?)",
                           (chat_id, "New chat", now, now, 0, 0, blob, sha, size, ""))
                old = self.find(db, chat_id)
                changed = True
            else:
                if old["revision"] != revision:
                    raise HistoryError("Another window updated this chat. Export your draft before reloading.", 409)
                if start is None and total != old["message_count"] or start is not None and start > old["message_count"]:
                    raise HistoryError("The conversation save range does not match.", 409)
                changed = old["state_sha"] != sha or old["message_count"] != total
            written = 0
            if start is not None:
                for seq, item in enumerate(items, start):
                    written += self._put_message(db, chat_id, seq, item)
                db.execute("DELETE FROM search_text WHERE chat_id=? AND seq>=?", (chat_id, total))
                db.execute("DELETE FROM messages WHERE chat_id=? AND seq>=?", (chat_id, total))
            changed = changed or bool(written)
            title = old["title"]
            if start == 0:
                first = next((m for m in items if m["role"] == "user"), None)
                if first:
                    title = re.sub(r"\s+", " ", (first["text"] or next((f.get("name", "") for f in first.get("files", [])), "")).strip())[:60] or "New chat"
            new_revision = old["revision"] + int(changed)
            if changed:
                db.execute("UPDATE chats SET title=?,updated_at=?,revision=?,message_count=?,state=?,state_sha=?,state_raw=?,fingerprint=? WHERE id=?",
                           (title, now, new_revision, total, blob, sha, size, self._fingerprint(db, chat_id, sha), chat_id))
            result = {"saved": True, "chat": self.metadata(self.find(db, chat_id)), "writtenMessages": written}
            if token:
                db.execute("INSERT INTO save_receipts VALUES(?,?,?)", (token, request_sha, json.dumps(result)))
                db.execute("DELETE FROM save_receipts WHERE rowid < (SELECT MAX(rowid)-2048 FROM save_receipts)")
            return result

    def import_book(self, raw, *, migrate=False):
        if len(raw) > MAX_BYTES:
            raise HistoryError("History files must be no larger than 64 MiB.", 413)
        try:
            book = json.loads(raw)
        except (ValueError, UnicodeError):
            raise HistoryError("Choose a Strata history JSON file.") from None
        if not isinstance(book, dict) or book.get("version") != 1 or not isinstance(book.get("chats"), list) or not book["chats"]:
            raise HistoryError("Choose a Strata history JSON file.")
        prepared = []
        for chat in book["chats"]:
            if not isinstance(chat, dict) or not isinstance(chat.get("messages"), list):
                raise HistoryError("Invalid chat format.")
            msgs = [message(m) for m in chat["messages"]]
            saved_state = state(chat.get("memory") or EMPTY_MEMORY, chat.get("draft", ""), chat.get("draftAttachments", []), len(msgs))
            hashes = [packed(m)[2] for m in msgs]
            fingerprint = digest(encoded([hashes, packed(saved_state)[2]]))
            prepared.append((chat, msgs, saved_state, fingerprint))
        source_sha = digest(raw)
        with self.lock:
            backups = self.directory / "backups"
            backups.mkdir(exist_ok=True)
            backup = backups / (("migration-" if migrate else "import-") + source_sha + ".json.gz")
            if not backup.exists():
                temporary = backup.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    stream.write(gzip.compress(raw, compresslevel=6, mtime=0)); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, backup)
            with gzip.open(backup, "rb") as stream:
                if digest(stream.read(MAX_BYTES + 1)) != source_sha:
                    raise HistoryError("Could not verify the migration backup. The original history is preserved.", 503)
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute("SELECT result FROM imports WHERE source_sha=?", (source_sha,)).fetchone()
                if previous:
                    return {**json.loads(previous[0]), "added": 0, "replayed": True}
                added, mapping = 0, {}
                for chat, msgs, saved_state, fingerprint in prepared:
                    source_id = chat.get("id", "")
                    safe_id = source_id if isinstance(source_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", source_id) else str(uuid.uuid4())
                    existing = db.execute("SELECT id,fingerprint FROM chats WHERE id=?", (safe_id,)).fetchone()
                    duplicate = db.execute("SELECT id FROM chats WHERE fingerprint=? LIMIT 1", (fingerprint,)).fetchone() if not migrate else None
                    if existing and existing["fingerprint"] == fingerprint or duplicate:
                        mapping[str(source_id)] = existing["id"] if existing and existing["fingerprint"] == fingerprint else duplicate["id"]
                        continue
                    if existing:
                        safe_id = str(uuid.uuid4())
                    blob, size, sha = packed(saved_state)
                    now = int(time.time() * 1000)
                    created = chat.get("createdAt", now); updated = chat.get("updatedAt", created)
                    created = int(created) if isinstance(created, (int, float)) and 0 <= created < 1e15 else now
                    updated = int(updated) if isinstance(updated, (int, float)) and 0 <= updated < 1e15 else created
                    first = next((m for m in msgs if m["role"] == "user"), None)
                    title = re.sub(r"\s+", " ", first["text"].strip())[:60] if first else "New chat"
                    db.execute("INSERT INTO chats VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (safe_id, title or "New chat", created, updated, 1, len(msgs), blob, sha, size, fingerprint))
                    for seq, item in enumerate(msgs):
                        self._put_message(db, safe_id, seq, item)
                    mapping[str(source_id)] = safe_id
                    added += 1
                result = {"saved": True, "added": added, "source_sha256": source_sha, "source_count": len(prepared),
                          "mapped_count": len(mapping), "active": mapping.get(str(book.get("active"))), "backup": backup.name}
                db.execute("INSERT INTO imports VALUES(?,?)", (source_sha, json.dumps(result)))
                return result

    def recover(self, body):
        """Recover one dirty browser window, preserving a conflicting chat separately."""
        if not isinstance(body, dict):
            raise HistoryError("Invalid draft format.")
        chat_id = identifier(body.get("id"))
        token = identifier(body.get("recoveryId"))
        offset = integer(body.get("offset", 0))
        items = body.get("messages")
        if not isinstance(items, list):
            raise HistoryError("Invalid draft format.")
        payload = {"id": chat_id, "revision": body.get("revision"), "writeId": token, "start": offset,
                   "total": offset + len(items), "messages": items, "memory": body.get("memory", EMPTY_MEMORY),
                   "draft": body.get("draft", ""), "draftAttachments": body.get("draftAttachments", [])}
        try:
            return {"offset": offset, **self.save(payload)}
        except HistoryError as error:
            if error.code != 409:
                raise
        for item in items:
            message(item)
        request_sha = digest(encoded(payload))
        with self.lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute("SELECT request_sha,result FROM save_receipts WHERE token=?", (token,)).fetchone()
            if receipt:
                if receipt["request_sha"] != request_sha:
                    raise HistoryError("The draft identifier does not match.", 409)
                return json.loads(receipt["result"])
            source = self.find(db, chat_id)
            saved_state = state(payload["memory"], payload["draft"], payload["draftAttachments"], payload["total"])
            blob, size, sha = packed(saved_state)
            same = source["message_count"] == payload["total"] and source["state_sha"] == sha
            if same:
                existing = db.execute("SELECT sha FROM messages WHERE chat_id=? AND seq>=? ORDER BY seq", (chat_id, offset)).fetchall()
                same = [r[0] for r in existing] == [packed(m)[2] for m in items]
            if same:
                return {"saved": True, "chat": self.metadata(source), "offset": offset, "alreadySaved": True}
            new_id = "recovered-" + token
            if len(new_id) > 128:
                new_id = "recovered-" + digest(token.encode())
            keep = offset if source["message_count"] >= offset else 0
            if keep != offset:
                saved_state["memory"]["through"] = max(0, saved_state["memory"]["through"] - offset)
                blob, size, sha = packed(saved_state)
            now = int(time.time() * 1000)
            db.execute("INSERT INTO chats VALUES(?,?,?,?,?,?,?,?,?,?)", (new_id, "Recovered: " + source["title"][:50],
                now, now, 1, keep + len(items), blob, sha, size, ""))
            # Prefix records are copied in their compressed form, without reading all messages.
            db.execute("INSERT INTO messages SELECT ?,seq,body,sha,raw_bytes FROM messages WHERE chat_id=? AND seq<?", (new_id, chat_id, keep))
            db.execute("INSERT INTO search_text(chat_id,seq,text) SELECT ?,seq,text FROM search_text WHERE chat_id=? AND seq<?", (new_id, chat_id, keep))
            for seq, item in enumerate(items, keep):
                self._put_message(db, new_id, seq, item)
            db.execute("UPDATE chats SET fingerprint=? WHERE id=?", (self._fingerprint(db, new_id, sha), new_id))
            result = {"saved": True, "chat": self.metadata(self.find(db, new_id)), "offset": keep,
                      "recoveredSeparately": True, "originalId": chat_id}
            db.execute("INSERT INTO save_receipts VALUES(?,?,?)", (token, request_sha, json.dumps(result)))
            return result

    def export_book(self, chat_id=None):
        with self.connect() as db:
            rows = [self.find(db, chat_id)] if chat_id else db.execute("SELECT * FROM chats ORDER BY updated_at DESC,id").fetchall()
            chats = []
            for row in rows:
                msgs = [unpacked(m[0]) for m in db.execute("SELECT body FROM messages WHERE chat_id=? ORDER BY seq", (row["id"],))]
                chats.append({**self.metadata(row), **unpacked(row["state"]), "messages": msgs})
            return {"version": 1, "active": chats[0]["id"] if chats else None, "chats": chats}

    def recall(self, query, chat_id=None, through=0, limit=4, budget=6000):
        if not isinstance(query, str) or len(query) > 4096:
            raise HistoryError("The search query is too long.")
        terms = query_terms(query)
        if not terms:
            return {"hits": [], "method": "local-keyword"}
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.connect() as db:
            rows = db.execute("""SELECT s.chat_id,s.seq,s.text,c.title,c.updated_at,bm25(search_fts) AS rank
                FROM search_fts JOIN search_text s ON s.rowid=search_fts.rowid JOIN chats c ON c.id=s.chat_id
                WHERE search_fts MATCH ? AND (s.chat_id<>? OR s.seq<?)
                ORDER BY rank,c.updated_at DESC LIMIT 30""", (match, chat_id or "", integer(through))).fetchall()
            hits, remaining = [], budget
            for row in rows:
                if len(hits) >= limit or remaining <= 0:
                    break
                text = row["text"]
                # Keep a bounded excerpt around the first matched term.
                at = min((text.casefold().find(t.casefold()) for t in terms if t.casefold() in text.casefold()), default=0)
                begin = max(0, at - 250)
                excerpt = text[begin:begin + min(1500, remaining)]
                hits.append({"chatId": row["chat_id"], "title": row["title"], "seq": row["seq"],
                             "updatedAt": row["updated_at"], "text": excerpt})
                remaining -= len(excerpt)
            return {"hits": hits, "method": "local-keyword", "characters": budget - remaining}

    def stats(self):
        with self.connect() as db:
            counts = db.execute("SELECT COUNT(*),COALESCE(SUM(state_raw),0),COALESCE(SUM(LENGTH(state)),0) FROM chats").fetchone()
            msgs = db.execute("SELECT COUNT(*),COALESCE(SUM(raw_bytes),0),COALESCE(SUM(LENGTH(body)),0) FROM messages").fetchone()
        disk = sum(p.stat().st_size for p in self.directory.glob("history.sqlite3*"))
        backups = sum(p.stat().st_size for p in (self.directory / "backups").glob("*.json.gz"))
        return {"chats": counts[0], "messages": msgs[0], "originalBytes": counts[1] + msgs[1],
                "compressedBytes": counts[2] + msgs[2], "databaseBytes": disk, "backupBytes": backups,
                "location": str(self.directory), "compression": "gzip", "pageSize": PAGE}


def respond(handler, code, value, binary=False):
    body = value if binary else json.dumps(value, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/gzip" if binary else "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    if binary:
        handler.send_header("Content-Disposition", 'attachment; filename="strata-history.json.gz"')
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def dispatch(handler, svc, path, method):
    if not path.startswith("/api/history"):
        return False
    try:
        raw = None
        if method == "POST":
            try:
                length = int(handler.headers.get("Content-Length", "0"))
            except ValueError:
                raise HistoryError("Invalid history size.") from None
            if not 0 < length <= MAX_BYTES:
                raise HistoryError("History must be no larger than 64 MiB.", 413)
            raw = handler.rfile.read(length)
            if len(raw) != length:
                raise HistoryError("The history was not received completely.")
        if not handler._own_page("chat history", content_types=("application/json", "application/gzip"),
                                 require_content_type=method == "POST"):
            return True
        with svc.history_lock:
            if svc.chat_history is None:
                svc.chat_history = ChatHistory(svc.history_directory)
        repo = svc.chat_history
        args = parse_qs(urlsplit(handler.path).query)
        get = lambda key, default=None: args.get(key, [default])[0]
        if method == "GET":
            if path == "/api/history":
                result = repo.catalog(get("q", ""), int(get("offset", 0)), int(get("limit", 50)))
            elif path in ("/api/history/chat", "/api/history/messages"):
                start = get("start")
                end = get("end")
                result = repo.page(get("id"), None if start is None else int(start), int(get("limit", PAGE)), path.endswith("/chat"),
                                   None if end is None else int(end))
            elif path == "/api/history/recall":
                result = repo.recall(get("q", ""), get("id"), int(get("through", 0)))
            elif path == "/api/history/stats":
                result = repo.stats()
            elif path == "/api/history/export":
                book = repo.export_book(get("id"))
                respond(handler, 200, book if get("format") == "json" else gzip.compress(encoded(book), compresslevel=6, mtime=0),
                        binary=get("format") != "json")
                return True
            else:
                raise HistoryError("History operation not found.", 404)
        elif method == "POST":
            if path in ("/api/history/import", "/api/history/migrate"):
                if handler.headers.get("Content-Type", "").split(";")[0] == "application/gzip":
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                        raw = stream.read(MAX_BYTES + 1)
                result = repo.import_book(raw, migrate=path.endswith("/migrate"))
            elif path == "/api/history/chat":
                result = repo.save(json.loads(raw))
            elif path == "/api/history/recover":
                result = repo.recover(json.loads(raw))
            else:
                raise HistoryError("History operation not found.", 404)
        else:
            raise HistoryError("This history operation is not supported.", 405)
        respond(handler, 200, result)
    except HistoryError as error:
        respond(handler, error.code, {"error": {"message": str(error)}})
    except (ValueError, TypeError, UnicodeError, gzip.BadGzipFile, EOFError, zlib.error):
        respond(handler, 400, {"error": {"message": "Invalid history format. The original history is unchanged."}})
    except (sqlite3.Error, OSError):
        respond(handler, 503, {"error": {"message": "Could not save or retrieve history. The original history and draft are preserved."}})
    return True
