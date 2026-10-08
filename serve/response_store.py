"""Experimental Responses item persistence. This stores API history, not native KV state.

One server owns a directory. Terminal records are immutable; descendants share their
parent's history. Deleted/expired ancestors remain internally until no child needs them.
"""
from __future__ import annotations

import copy
import functools
import json
import sqlite3
import threading
import time
from pathlib import Path

from serve.responses import ResponsesError, new_id


def storage_errors(method):
    @functools.wraps(method)
    def call(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (sqlite3.Error, OSError) as exc:
            raise ResponsesError("Responses storage failed: " + str(exc), code="response_store_error",
                                 status=500, kind="server_error") from exc
    return call


class ResponseStore:
    def __init__(self, directory, retention_s=30 * 86400, max_bytes=256 * 1024 * 1024):
        if isinstance(retention_s, bool) or not isinstance(retention_s, (int, float)) or not 0 <= retention_s < float('inf'):
            raise ValueError("responses_retention_s must be a finite number >= 0")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1024:
            raise ValueError("responses store capacity must be at least 1024 bytes")
        self.retention_s, self.max_bytes = retention_s, max_bytes
        self.lock = threading.RLock()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.owner = (directory / "owner.lock").open("a+b")
        try:
            # OS locks disappear on process exit, including a crash. Never unlink the lock inode.
            import os
            if os.name == "nt":
                import msvcrt
                self.owner.seek(0)
                self.owner.write(b"\0")
                self.owner.flush()
                self.owner.seek(0)
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise ValueError("responses store directory is already owned by another server") from None
        try:
            self.db = sqlite3.connect(directory / "responses.sqlite3", check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("""CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY, parent TEXT, input TEXT NOT NULL, response TEXT NOT NULL,
                created REAL NOT NULL, deleted INTEGER NOT NULL DEFAULT 0)""")
            self.db.execute("CREATE INDEX IF NOT EXISTS responses_parent ON responses(parent)")
            with self.db:
                for rid, raw in self.db.execute("SELECT id, response FROM responses").fetchall():
                    response = json.loads(raw)
                    if response["status"] in ("queued", "in_progress"):
                        response.update(status="failed", error={"code": "server_restarted",
                                                               "message": "server restarted before completion"})
                        self.db.execute("UPDATE responses SET response=? WHERE id=?", (self._json(response), rid))
                self._prune()
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.owner.close()
            raise

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def close(self):
        with self.lock:
            self.db.close()
            self.owner.close()

    def _prune(self):
        if self.retention_s:
            # Active records are retained until their generator has stopped.
            self.db.execute("""UPDATE responses SET deleted=1 WHERE created < ?
                AND json_extract(response, '$.status') NOT IN ('queued', 'in_progress')""",
                (time.time() - self.retention_s,))
        while self.db.execute("""DELETE FROM responses WHERE deleted=1
                AND json_extract(response, '$.status') NOT IN ('queued', 'in_progress')
                AND NOT EXISTS (SELECT 1 FROM responses child WHERE child.parent=responses.id)""").rowcount:
            pass

    def _row(self, rid, internal=False):
        row = self.db.execute("SELECT parent,input,response,deleted FROM responses WHERE id=?", (rid,)).fetchone()
        if row is None or (row[3] and not internal):
            raise ResponsesError("response not found", "response_id", "response_not_found", 404)
        return row

    def _history(self, rid):
        chain = []
        while rid:
            parent, inputs, raw, _ = self._row(rid, internal=True)
            chain.append(json.loads(inputs) + json.loads(raw)["output"])
            rid = parent
        return [item for group in reversed(chain) for item in group]

    @storage_errors
    def resolve(self, req, model):
        store = req.get("store", True)
        if not isinstance(store, bool):
            raise ResponsesError("store must be a boolean", "store")
        parent = req.get("previous_response_id")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ResponsesError("previous_response_id must be a nonempty string or null", "previous_response_id")
        inputs = req.get("input")
        if isinstance(inputs, str):
            inputs = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": inputs}]}]
        if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
            raise ResponsesError("input must be a string or an array of items", "input")
        inputs = copy.deepcopy(inputs)
        for item in inputs:
            item.setdefault("id", new_id("item"))
            if not isinstance(item["id"], str) or not item["id"]:
                raise ResponsesError("input item id must be a nonempty string", "input")
        with self.lock, self.db:
            self._prune()
            history = []
            if parent:
                response = json.loads(self._row(parent)[2])
                if response["status"] not in ("completed", "incomplete"):
                    raise ResponsesError("previous response must be completed or incomplete", "previous_response_id",
                                         "response_not_ready", 409)
                if response["model"] != model:
                    raise ResponsesError("previous response belongs to a different model", "previous_response_id")
                history = self._history(parent)
        item_ids = [item["id"] for item in history + inputs]
        if len(item_ids) != len(set(item_ids)):
            raise ResponsesError("input item ids must be unique within a conversation", "input")
        return {**req, "input": history + inputs}, {"parent": parent, "input": inputs, "store": store}

    def _capacity(self, extra):
        used = self.db.execute("SELECT coalesce(sum(length(CAST(input AS BLOB))+length(CAST(response AS BLOB))),0) FROM responses").fetchone()[0]
        if used + extra > self.max_bytes:
            raise ResponsesError("Responses store capacity exceeded; delete responses or increase responses_store_max_mib",
                                 code="response_store_full", status=507, kind="server_error")

    @storage_errors
    def begin(self, response, context):
        response.update(previous_response_id=context["parent"], store=context["store"])
        if not context["store"]:
            return
        inputs, raw = self._json(context["input"]), self._json(response)
        with self.lock, self.db:
            self._prune()
            if context["parent"]:
                self._row(context["parent"])  # deletion between resolve and admission must not leave a broken chain
            self._capacity(len(inputs.encode()) + len(raw.encode()))
            self.db.execute("INSERT INTO responses(id,parent,input,response,created) VALUES(?,?,?,?,?)",
                            (response["id"], context["parent"], inputs, raw, time.time()))

    @storage_errors
    def finish(self, response):
        if not response["store"]:
            return
        with self.lock, self.db:
            _, _, old, deleted = self._row(response["id"], internal=True)
            if json.loads(old)["status"] not in ("queued", "in_progress"):
                return  # terminal IDs never mutate
            raw = self._json(response)
            if not deleted and response["status"] in ("completed", "incomplete"):
                self._capacity(len(raw.encode()) - len(old.encode()))
            self.db.execute("UPDATE responses SET response=? WHERE id=?", (raw, response["id"]))
            self._prune()

    @storage_errors
    def get(self, rid):
        with self.lock, self.db:
            self._prune()
            return json.loads(self._row(rid)[2])

    @storage_errors
    def delete(self, rid):
        with self.lock, self.db:
            self._prune()
            self._row(rid)
            self.db.execute("UPDATE responses SET deleted=1 WHERE id=?", (rid,))
            self._prune()
        return {"id": rid, "object": "response.deleted", "deleted": True}

    @storage_errors
    def input_items(self, rid, query):
        if set(query) - {"order", "limit", "after", "before"} or any(len(v) != 1 for v in query.values()):
            raise ResponsesError("unsupported or repeated pagination parameter")
        order = query.get("order", ["desc"])[0]
        try:
            limit = int(query.get("limit", ["20"])[0])
        except ValueError:
            limit = 0
        if order not in ("asc", "desc") or not 1 <= limit <= 100:
            raise ResponsesError("order must be asc or desc; limit must be 1..100")
        with self.lock, self.db:
            self._prune()
            parent, inputs, _, _ = self._row(rid)
            items = self._history(parent) + json.loads(inputs)
        if order == "desc":
            items.reverse()
        for name in ("after", "before"):
            if name in query:
                index = next((i for i, item in enumerate(items) if item.get("id") == query[name][0]), None)
                if index is None:
                    raise ResponsesError("unknown pagination cursor", name)
                items = items[index + 1:] if name == "after" else items[:index]
        page = items[:limit]
        return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                "last_id": page[-1]["id"] if page else None, "has_more": len(items) > limit}
