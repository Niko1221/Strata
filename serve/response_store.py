"""Retained Responses records. One server owns a directory; no generation or workers here.

Creation and terminal snapshots are atomic file replacements, never token writes.
An update cannot insert: deletion/expiry wins over a late generation completion.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path


class ResponseStore:
    def __init__(self, directory, retention_s=30 * 24 * 60 * 60):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.retention_s = retention_s
        self.lock = threading.RLock()
        self.cleanup()

    def _path(self, response_id):
        if not isinstance(response_id, str) or not re.fullmatch(r"resp_[0-9a-f]{32}", response_id):
            raise KeyError(response_id)
        return self.directory / (response_id + ".json")

    def _write(self, path, record):
        fd, name = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    def create(self, record):
        with self.lock:
            self.cleanup()
            path = self._path(record["response"]["id"])
            if path.exists():
                raise ValueError("response already exists")
            self._write(path, record)

    def get(self, response_id):
        with self.lock:
            path = self._path(response_id)
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise KeyError(response_id) from None
            except (ValueError, UnicodeError) as exc:
                raise OSError("invalid retained response record") from exc
            try:
                expires_at = record["response"]["created_at"] + self.retention_s
                if record["response"]["id"] != response_id or not isinstance(record["input_items"], list):
                    raise ValueError("record identity or input is invalid")
            except (KeyError, TypeError, ValueError) as exc:
                raise OSError("invalid retained response record") from exc
            if expires_at <= time.time():
                path.unlink(missing_ok=True)
                raise KeyError(response_id)
            return record

    def update(self, record):
        with self.lock:
            response_id = record["response"]["id"]
            try:
                self.get(response_id)
            except KeyError:
                return False
            self._write(self._path(response_id), record)
            return True

    def delete(self, response_id):
        with self.lock:
            self.get(response_id)
            self._path(response_id).unlink()

    def records(self):
        with self.lock:
            records = []
            for path in self.directory.glob("resp_*.json"):
                try:
                    records.append(self.get(path.stem))
                except KeyError:
                    pass
            return records

    def cleanup(self):
        # No sweeper thread: expiry is enforced on reads, startup and creation.
        self.records()
