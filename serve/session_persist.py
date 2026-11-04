"""serve/session_persist.py - the conversation cache's L2: disk snapshots indexed by the tokens of the prompt
that produced them.

The engine parks a finished conversation in RAM and reuses it when a later prompt shares its tokens (#596);
POST /slots/0?action=save|restore moves one whole session to or from disk, by hand.  This module joins the two,
automatically: at the end of a turn the service saves the engine's live conversation under the digest of the
request's prompt tokens and records (length, digest, file) in index.json; a later prompt - this server run or
the next one - that begins with those tokens restores the snapshot first, and the engine reads only the tokens
past them (a RESTORE makes the saved context the engine's live one, so the GEN that follows diffs against it).

The key is the PROMPT's tokens, not prompt + reply: a chat template renders message by message, so turn n's
whole prompt is a prefix of turn n+1's, whatever the reply was.  Matching takes the incoming prompt's digest at
each recorded length (one sha256 per candidate, microseconds; the engine's own prefix match absorbs a smaller
divergence at the tail).

index.json (in slot_save_path) is {"snapshots": [{"len", "digest", "file", "bytes", "at"}]}: LRU by "at",
trimmed to max_bytes, the trimmed files removed with their entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path

INDEX_NAME = "index.json"


def ids_digest(ids) -> str:
    """A 16-hex digest of a token id list (8 bytes an id; the digest of a prefix is the same computation, cut
    short)."""
    if not ids:
        return ""
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()[:16]


class SnapshotIndex:
    """The saved conversations of one slot_save_path, LRU-trimmed to `max_bytes`.  Not thread-safe: the service
    reaches it between its turns."""

    def __init__(self, directory: str, max_bytes: int = 8 << 30):
        self.dir = Path(directory)
        self.max_bytes = max_bytes
        self.snapshots: list[dict] = []
        try:
            data = json.loads((self.dir / INDEX_NAME).read_text())
            if isinstance(data, dict) and isinstance(data.get("snapshots"), list):
                self.snapshots = [s for s in data["snapshots"]
                                  if isinstance(s, dict) and (self.dir / str(s.get("file", ""))).is_file()]
        except (OSError, ValueError):
            pass   # no index yet, or one this build does not read: start empty (the files stay on the disk)

    def match(self, ids) -> dict | None:
        """The saved conversation the prompt `ids` begins with (the longest one), or None."""
        best = None
        for s in self.snapshots:
            n = int(s.get("len") or 0)
            if 0 < n < len(ids) and s.get("digest") == ids_digest(ids[:n]):
                if best is None or n > int(best["len"]):
                    best = s
        return best

    def add(self, length: int, digest: str, file: str, bytes_written: int) -> None:
        """Record a snapshot (a same digest replaces its older entry and file) and trim to max_bytes."""
        self.snapshots = [s for s in self.snapshots if s.get("digest") != digest]
        self.snapshots.append({"len": int(length), "digest": digest, "file": file,
                               "bytes": int(bytes_written), "at": int(time.time())})
        self.trim()

    def trim(self) -> None:
        """Keep the newest snapshots until their bytes fit max_bytes; remove the files the index drops."""
        self.snapshots.sort(key=lambda s: s.get("at") or 0)
        while sum(int(s.get("bytes") or 0) for s in self.snapshots) > self.max_bytes and len(self.snapshots) > 1:
            gone = self.snapshots.pop(0)
            try:
                (self.dir / str(gone["file"])).unlink()
            except OSError:
                pass

    def save(self) -> None:
        tmp = self.dir / (INDEX_NAME + ".tmp")
        tmp.write_text(json.dumps({"snapshots": self.snapshots}))
        os.replace(tmp, self.dir / INDEX_NAME)
