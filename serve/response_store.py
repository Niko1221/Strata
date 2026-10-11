"""Optional, bounded Responses history in this server process (not the engine's KV cache)."""
from __future__ import annotations

import collections
import json
import math
import threading
import time

from serve.responses import ResponsesError, input_messages


def input_items(req: dict) -> list:
    value = req.get("input")
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        return value
    raise ResponsesError("input is required: a string or an array of input items", "input",
                         "missing_required_parameter" if value is None else None)


def _utf8_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _too_large() -> ResponsesError:
    return ResponsesError("response and replay history exceed the configured responses store byte limit; "
                          "use store: false or increase responses_store_mib", "store",
                          "response_store_limit_exceeded", status=413)


class ResponseStore:
    """Serialized snapshots, oldest first, with byte/count/age limits. Each child owns its history,
    so deleting an ancestor does not break a surviving child or change another continuation.
    Only terminal responses are saved; shutdown discards everything. The byte limit counts UTF-8 JSON,
    including replay history, rather than just response text (Python bookkeeping is additional)."""

    def __init__(self, max_bytes: int, ttl_seconds: float = 3600, max_entries: int = 256, clock=time.monotonic):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("responses store byte limit must be a positive integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("responses store count limit must be a positive integer")
        if (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float))
                or not math.isfinite(ttl_seconds) or ttl_seconds <= 0):
            raise ValueError("responses_store_ttl_s must be a positive finite number")
        self.max_bytes, self.ttl_seconds, self.max_entries = max_bytes, ttl_seconds, max_entries
        self.clock = clock
        self.lock = threading.Lock()
        self.records = collections.OrderedDict()
        self.bytes_used = 0

    def _remove(self, response_id):
        _, blob = self.records.pop(response_id)
        self.bytes_used -= len(blob)

    def _expire(self):
        now = self.clock()
        while self.records:
            response_id, (deadline, _) = next(iter(self.records.items()))
            if deadline > now:
                break
            self._remove(response_id)

    def _record(self, response_id: str, param=None) -> dict:
        with self.lock:
            self._expire()
            record = self.records.get(response_id)
            if record is None:
                raise ResponsesError("response not found (not stored, deleted, expired or evicted)", param,
                                     "not_found", status=404)
            return json.loads(record[1])

    def prepare(self, req: dict) -> dict:
        items = input_items(req)
        previous = req.get("previous_response_id")
        if previous:
            input_messages({"input": items})      # the new items alone first, so an error names their own index
            # Top-level instructions, tools and sampling settings are intentionally not inherited.
            items = self._record(previous, "previous_response_id")["history"] + items
        if req.get("store") and len(_utf8_json(items)) > self.max_bytes:
            raise _too_large()                    # the replay alone cannot be kept: refused before the model runs
        return {**req, "input": items}

    def save(self, req: dict, response: dict) -> None:
        if not response["store"]:
            return
        blob = _utf8_json({"response": response, "history": input_items(req) + response["output"]})
        if len(blob) > self.max_bytes:
            raise _too_large()
        with self.lock:
            self._expire()
            response_id = response["id"]
            if response_id in self.records:
                self._remove(response_id)
            while self.records and (len(self.records) >= self.max_entries
                                    or self.bytes_used + len(blob) > self.max_bytes):
                self._remove(next(iter(self.records)))
            self.records[response_id] = (self.clock() + self.ttl_seconds, blob)
            self.bytes_used += len(blob)

    def get(self, response_id: str) -> dict:
        return self._record(response_id)["response"]

    def delete(self, response_id: str) -> dict:
        with self.lock:
            self._expire()
            if response_id not in self.records:
                raise ResponsesError("response not found (not stored, deleted, expired or evicted)",
                                     code="not_found", status=404)
            self._remove(response_id)
        return {"id": response_id, "object": "response.deleted", "deleted": True}
