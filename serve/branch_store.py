"""Experimental immutable item boundaries, replay receipts and inactivity retention.

History owns content; checkpoint catalogs may disappear without invalidating it.
The existing Responses owner lock and SQLite transaction protect both stores.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import threading

from serve.response_store import ResponseStore, storage_errors
from serve.responses import ResponsesError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class BranchReceipt(str):
    """Internal provenance tag: an ordinary client-supplied JSON string is not a receipt."""


class BranchStore(ResponseStore):
    def __init__(self, *args, execution_identity, asset_loader=None, **kwargs):
        self.execution_identity = copy.deepcopy(execution_identity)
        self.fingerprint = digest(execution_identity)
        self.asset_loader = asset_loader
        self.on_expire = None
        self.on_shutdown = None
        self.ready = False
        super().__init__(*args, **kwargs)
        with self.lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS history_nodes (
                    id TEXT PRIMARY KEY, parent TEXT NOT NULL, item TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS history_branches (
                    response_id TEXT PRIMARY KEY, base TEXT NOT NULL, tip TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, settings TEXT NOT NULL,
                    active REAL NOT NULL, protected INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS execution_records (
                    response_id TEXT PRIMARY KEY, identity TEXT NOT NULL,
                    prompt TEXT NOT NULL, generated TEXT, complete INTEGER NOT NULL DEFAULT 0);
            """)
            self.ready = True
            self._prune()
            self.db.commit()
        self._stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweep, daemon=True)
        self._sweeper.start()

    def _sweep(self):
        while not self._stop.wait(3600):
            try:
                self.expire()
            except Exception as exc:
                print(f"[strata] history expiration failed: {exc}", flush=True)

    def close(self):
        self._stop.set()
        self._sweeper.join()
        try:
            if self.on_shutdown:
                self.on_shutdown(self.protected_checkpoint_owners())
        finally:
            super().close()

    def protected_checkpoint_owners(self):
        with self.lock:
            return {r[0] for r in self.db.execute("""WITH RECURSIVE kept(id) AS (
                SELECT response_id FROM history_branches WHERE protected=1
                UNION SELECT r.parent FROM responses r JOIN kept k ON r.id=k.id WHERE r.parent IS NOT NULL)
                SELECT id FROM kept""")}

    def expire(self):
        with self.lock, self.transaction():
            self._prune()

    @storage_errors
    def get(self, rid):
        with self.lock:
            self.expire()  # persist expiration even when the following lookup returns 404
            return json.loads(self._row(rid)[2])

    def _prune(self):
        if not self.ready:
            return  # base constructor must not expire legacy data using creation time
        if self.retention_s:
            self.db.execute("""UPDATE responses SET deleted=1 WHERE id IN (
                SELECT r.id FROM responses r LEFT JOIN history_branches b ON b.response_id=r.id
                WHERE coalesce(b.protected,0)=0 AND coalesce(b.active,r.created) < ?)
                AND json_extract(response,'$.status') NOT IN ('queued','in_progress')""",
                (time.time() - self.retention_s,))
        # Hidden response ancestors remain internally for legacy previous_response_id chains.
        while self.db.execute("""DELETE FROM responses WHERE deleted=1
                AND json_extract(response,'$.status') NOT IN ('queued','in_progress')
                AND NOT EXISTS (SELECT 1 FROM responses c WHERE c.parent=responses.id)""").rowcount:
            pass
        self.db.execute("DELETE FROM history_branches WHERE response_id NOT IN (SELECT id FROM responses)")
        self.db.execute("DELETE FROM execution_records WHERE response_id NOT IN (SELECT id FROM responses)")
        self.db.execute("""WITH RECURSIVE retained(id) AS (
            SELECT tip FROM history_branches UNION SELECT base FROM history_branches
            UNION SELECT n.parent FROM history_nodes n JOIN retained r ON n.id=r.id)
            DELETE FROM history_nodes WHERE id NOT IN (SELECT id FROM retained)""")
        if self.on_expire:
            self.on_expire({r[0] for r in self.db.execute("SELECT id FROM responses")})

    def _nodes(self, tip):
        nodes = []
        while tip:
            row = self.db.execute("SELECT parent,item FROM history_nodes WHERE id=?", (tip,)).fetchone()
            if row is None:
                raise ResponsesError("authoritative history is missing", code="history_missing", status=409)
            nodes.append({"id": tip, "parent": row[0], "item": json.loads(row[1])})
            tip = row[0]
        return list(reversed(nodes))

    def _append(self, tip, items):
        for item in items:
            nid = digest([tip, item])
            self.db.execute("INSERT OR IGNORE INTO history_nodes VALUES(?,?,?)", (nid, tip, self._json(item)))
            tip = nid
        return tip

    def _history(self, rid):
        row = self.db.execute("SELECT tip FROM history_branches WHERE response_id=?", (rid,)).fetchone()
        return [n["item"] for n in self._nodes(row[0])] if row else super()._history(rid)

    def _input_history(self, rid):
        _, inputs, _, _ = self._row(rid)
        row = self.db.execute("SELECT base FROM history_branches WHERE response_id=?", (rid,)).fetchone()
        if row:
            return [n["item"] for n in self._nodes(row[0])] + json.loads(inputs)
        return super()._input_history(rid)

    def _resolve_history(self, rid, req):
        selector = req.get("_history_branch_selector")
        if not selector:
            return self._history(rid)
        row = self.db.execute("SELECT tip FROM history_branches WHERE response_id=?", (rid,)).fetchone()
        if not row:
            raise ResponsesError("migrate legacy history before selecting a boundary", "branch_from", status=409)
        nodes = self._nodes(row[0])
        for i, node in enumerate(nodes):
            if node["id"] == selector["node_id"]:
                return [n["item"] for n in nodes[:i + (selector["side"] == "after")]]
        raise ResponsesError("node does not belong to this conversation", "branch_from", status=404)

    def _freeze(self, value):
        if isinstance(value, list):
            return [self._freeze(x) for x in value]
        if not isinstance(value, dict):
            return value
        result = {k: self._freeze(v) for k, v in value.items()}
        if result.get("type") in ("input_image", "image_url"):
            image = result.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str) and not url.startswith("data:"):
                if not self.asset_loader:
                    raise ResponsesError("durable history requires embedded image data", "input")
                data = self.asset_loader(url)
                if not isinstance(data, str) or not data.startswith("data:"):
                    raise ResponsesError("image could not be durably materialized", "input")
                result["image_url"] = {**image, "url": data} if isinstance(image, dict) else data
        return result

    @storage_errors
    def resolve(self, req, model):
        req = copy.deepcopy(req)
        migration = req.pop("migrate_history", False)
        if not isinstance(migration, bool):
            raise ResponsesError("migrate_history must be boolean", "migrate_history")
        selector = req.pop("branch_from", None)
        parent = req.get("previous_response_id")
        if selector is not None:
            if parent or not isinstance(selector, dict) or set(selector) != {"response_id", "node_id", "side"}:
                raise ResponsesError("branch_from needs response_id, node_id, side; omit previous_response_id", "branch_from")
            if selector["side"] not in ("before", "after") or not all(isinstance(selector[k], str) for k in ("response_id", "node_id")):
                raise ResponsesError("invalid branch boundary", "branch_from")
            req["previous_response_id"] = parent = selector["response_id"]
        req["input"] = self._freeze(req.get("input"))
        req["_history_branch_selector"] = selector
        with self.lock, self.transaction():
            source_model = json.loads(self._row(parent)[2])["model"] if migration and parent else model
            resolved, context = super().resolve(req, source_model)
            resolved.pop("_history_branch_selector", None)
            tip, settings = "", {}
            if parent:
                row = self.db.execute("SELECT tip,fingerprint,settings FROM history_branches WHERE response_id=?", (parent,)).fetchone()
                if row is None:
                    if not migration:
                        raise ResponsesError("legacy history requires migrate_history:true", code="history_migration_required", status=409)
                    if selector:
                        raise ResponsesError("migrate legacy history before selecting a boundary", "branch_from")
                    # Import only recorded items; no tools or assistant calls are executed.
                    tip = self._append("", self._freeze(super()._history(parent)))
                else:
                    tip, fingerprint, raw = row
                    if fingerprint != self.fingerprint and not migration:
                        raise ResponsesError("execution identity changed; use migrate_history:true on a new branch",
                                             code="history_migration_required", status=409)
                    settings = json.loads(raw)
                if selector:
                    node = next((n for n in self._nodes(tip) if n["id"] == selector["node_id"]), None)
                    if node is None:
                        raise ResponsesError("node does not belong to this conversation", "branch_from", status=404)
                    tip = node["parent"] if selector["side"] == "before" else node["id"]
                resolved["input"] = [n["item"] for n in self._nodes(tip)] + context["input"]
            # Bind execution framing to this lineage, not whatever defaults happen to be current.
            for key in ("instructions", "tools", "reasoning", "text", "tool_choice", "parallel_tool_calls", "experimental_speed_projection"):
                if key in settings and key in req and req[key] != settings[key] and not migration:
                    raise ResponsesError(f"changing {key} requires migrate_history:true", key, status=409)
                if key in req:
                    settings[key] = req[key]
                if key in settings:
                    resolved[key] = settings[key]
            context.update(base=tip, settings=settings, fingerprint=self.fingerprint)
            return resolved, context

    @storage_errors
    def begin(self, response, context):
        with self.lock, self.transaction():
            super().begin(response, context)
            if not context["store"]:
                return
            tip = self._append(context["base"], context["input"])
            self.db.execute("INSERT INTO history_branches VALUES(?,?,?,?,?,?,0)",
                            (response["id"], context["base"], tip, self.fingerprint,
                             self._json(context["settings"]), time.time()))
            self._capacity(0)

    @storage_errors
    def record_prompt(self, rid, ids):
        with self.lock, self.transaction():
            self._row(rid)
            self._capacity(len(self._json(ids).encode()))
            self.db.execute("INSERT INTO execution_records(response_id,identity,prompt) VALUES(?,?,?)",
                            (rid, self._json(self.execution_identity), self._json(ids)))

    @storage_errors
    def record_generated(self, rid, ids):
        with self.lock, self.transaction():
            self._capacity(len(self._json(ids).encode()))
            self.db.execute("UPDATE execution_records SET generated=?,complete=1 WHERE response_id=? AND complete=0",
                            (self._json(ids), rid))

    def _capacity(self, extra):
        if self.ready:
            extra += self.db.execute("SELECT coalesce(sum(length(CAST(item AS BLOB))),0) FROM history_nodes").fetchone()[0]
            extra += self.db.execute("SELECT coalesce(sum(length(prompt)+length(coalesce(generated,''))+length(identity)),0) FROM execution_records").fetchone()[0]
        super()._capacity(extra)

    @storage_errors
    def finish(self, response):
        if not response["store"]:
            return
        with self.lock, self.transaction():
            old = json.loads(self._row(response["id"], internal=True)[2])
            if old["status"] not in ("queued", "in_progress"):
                return
            if response["status"] in ("completed", "incomplete"):
                row = self.db.execute("SELECT tip FROM history_branches WHERE response_id=?", (response["id"],)).fetchone()
                tip = self._append(row[0], response["output"])
                self.db.execute("UPDATE history_branches SET tip=?,active=? WHERE response_id=?",
                                (tip, time.time(), response["id"]))
                # Successful user continuation renews the selected source branch, not siblings.
                self.db.execute("UPDATE history_branches SET active=? WHERE response_id=?",
                                (time.time(), response.get("previous_response_id")))
            super().finish(response)

    @storage_errors
    def boundaries(self, rid):
        with self.lock, self.transaction():
            self._prune()
            self._row(rid)
            row = self.db.execute("SELECT tip,protected FROM history_branches WHERE response_id=?", (rid,)).fetchone()
            if not row:
                raise ResponsesError("legacy history requires explicit migration", status=409)
            return {"response_id": rid, "bookmarked": bool(row[1]), "nodes": [
                {**n, "before": {"prefix_tip_node_id": n["parent"], "token_offset": None},
                 "after": {"prefix_tip_node_id": n["id"], "token_offset": None}}
                for n in self._nodes(row[0])]}

    @storage_errors
    def bookmark(self, rid, protected):
        if not isinstance(protected, bool):
            raise ResponsesError("protected must be boolean", "protected")
        with self.lock, self.transaction():
            self._prune()
            self._row(rid)
            if not self.db.execute("UPDATE history_branches SET protected=? WHERE response_id=?", (protected, rid)).rowcount:
                raise ResponsesError("legacy history requires migration before bookmarking", status=409)
        return {"response_id": rid, "protected": protected}
