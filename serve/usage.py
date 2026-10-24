"""serve/usage.py - lifetime token statistics for the Monitor's Usage panel.

The engine already counts tokens per request; /metrics keeps those counters only since the server
started (the Monitor's "totals", issue #35).  This module keeps the same counters across restarts
and in time buckets, so the page can show lifetime figures and 1 day / 7 day / 30 day views.

Off by default.  Turned on by --usage-stats or "usage_stats": true in the run config (the Settings
view lists it).  When on, one small JSON file is written beside the run config, e.g.
strata-iq3_s.usage.json; it holds only counters and bucket start times - no prompts, no answers,
no request bodies. Each settled request is saved atomically before recording returns, including
requests followed by a long idle period. A killed request that has not settled is not counted.
Only one process may own a store; a second writer leaves it untouched. Failed writes are visible
in /metrics and the page. Corrupt files are preserved before counting starts again.

Counters, per bucket and for the whole lifetime:
  requests        finished requests
  prompt_tokens   prompt tokens read (as /metrics' totals count them)
  reused          of prompt_tokens, the ones the conversation cache already held
  output_tokens   tokens generated
  prompt_ms       prompt read time
  decode_ms       generation time
  drafts_offered  speculative draft tokens offered
  drafts_accepted speculative draft tokens accepted

Bucket arrays keep only what a view needs (31 days of hourly and daily buckets); a bucket that has
no requests at all simply has no entry, and a view shows it as zero.  The hourly array spans the
same 31 days as the daily one so an arbitrary date range can be answered at hour resolution.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

HOUR = 3600.0
DAY = 24 * HOUR
SEGMENT_S = {"hour": HOUR, "day": DAY}       # the bucket size of each kept array
KEEP = {"hour": 744, "day": 31}               # buckets kept per array (31 days, hour and day)
COUNTERS = ("requests", "prompt_tokens", "reused", "output_tokens", "prompt_ms", "decode_ms",
            "drafts_offered", "drafts_accepted")


def _zero() -> dict:
    return {k: 0 for k in COUNTERS}


def _as_int(value) -> int:
    """A counter read from a file we do not trust: junk reads as 0, never as a failed boot."""
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return n if n > 0 else 0


def _is_time(value) -> bool:
    """A bucket start or a `since` is only usable if it is a finite number."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _counter(key, value):
    if key not in ("prompt_ms", "decode_ms"):
        return _as_int(value)
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return n if math.isfinite(n) and n > 0 else 0.0


def _align(t: float, segment: str) -> float:
    return int(t // SEGMENT_S[segment]) * SEGMENT_S[segment]


class Usage:
    """The lifetime counters and their time buckets.  `path` None (or empty): disabled, cheap no-op."""

    def __init__(self, path: str | Path | None = None, now: float | None = None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.since = float(now if now is not None else time.time())
        self.lifetime = _zero()
        self.buckets: dict[str, list[dict]] = {"hour": [], "day": []}
        self.imported = {"requests": 0, "output_tokens": 0}  # undated output-only log backfill, never a timed rate
        self.persistence_error = None
        self._owner = None
        self._blocked = False
        if self.path:
            self._claim()
            if not self._blocked:
                self._load(now)

    # -- state -------------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.path is not None and not self._blocked

    def _claim(self):
        """Keep a process lock separate from the JSON file that atomic replacement swaps out."""
        owner = None
        try:
            owner = self.path.with_name(self.path.name + ".lock").open("a+b")
            if owner.tell() == 0:
                owner.write(b"0")
                owner.flush()
            owner.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._owner = owner
        except OSError:
            if owner:
                owner.close()
            self._blocked = True
            self.persistence_error = "Usage store is unavailable or already open in another server."
            print(f"[strata] {self.persistence_error}", flush=True)

    def close(self):
        atexit.unregister(self.close)
        with self.lock:
            self._save()
            if self._owner:
                self._owner.close()
                self._owner = None
            self._blocked = True

    def _load(self, now: float | None) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("usage store must be an object")
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            try:
                backup = self.path.with_name(self.path.name + f".corrupt-{time.time_ns()}")
                shutil.copy2(self.path, backup)
                print(f"[strata] invalid usage statistics preserved as {backup.name}; starting new counters", flush=True)
            except OSError:
                self._blocked = True
                self.persistence_error = "Usage file cannot be read or backed up; it was left untouched."
            return
        life = data.get("lifetime")
        if isinstance(life, dict):
            self.lifetime = {k: _counter(k, life.get(k)) for k in COUNTERS}
        imported = data.get("imported")
        if isinstance(imported, dict):
            self.imported = {k: _as_int(imported.get(k)) for k in self.imported}
        if _is_time(data.get("since")):
            self.since = float(data["since"])
        for segment in ("hour", "day"):
            rows = data.get(segment)
            merged: dict[float, dict] = {}
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict) or not _is_time(r.get("t")):
                        continue
                    bucket = merged.setdefault(float(r["t"]), {"t": float(r["t"]), **_zero()})
                    for k in COUNTERS:                          # two rows for one bucket: sum them
                        bucket[k] += _counter(k, r.get(k))
            out = sorted(merged.values(), key=lambda b: b["t"])
            self.buckets[segment] = out[-KEEP[segment]:]

    def _save(self) -> None:
        if not self.enabled:
            return
        data = {"since": self.since, "updated": time.time(), "lifetime": self.lifetime,
                "hour": self.buckets["hour"], "day": self.buckets["day"], "imported": self.imported}
        tmp = None
        try:
            fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
            tmp = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self.persistence_error = None
        except (OSError, ValueError):  # a statistics write must never break serving
            if not self.persistence_error:
                print(f"[strata] usage statistics could not be saved to {self.path.name}; totals are in memory only", flush=True)
            self.persistence_error = "Latest totals could not be saved; they are in memory only."
        finally:
            if tmp:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def flush(self) -> None:
        """Retry a failed write without recording another request."""
        with self.lock:
            self._save()

    # -- recording ---------------------------------------------------------------------------------
    def record(self, prompt_tokens: int = 0, reused: int = 0, output_tokens: int = 0,
               prompt_ms: float = 0.0, decode_ms: float = 0.0,
               drafts_offered: int = 0, drafts_accepted: int = 0,
               now: float | None = None) -> None:
        """Add one finished request to the lifetime counters and its buckets."""
        if not self.enabled:
            return
        now = float(now if now is not None else time.time())
        vals = {"requests": 1, "prompt_tokens": _as_int(prompt_tokens), "reused": _as_int(reused),
                "output_tokens": _as_int(output_tokens), "prompt_ms": _counter("prompt_ms", prompt_ms),
                "decode_ms": _counter("decode_ms", decode_ms), "drafts_offered": _as_int(drafts_offered),
                "drafts_accepted": _as_int(drafts_accepted)}
        with self.lock:
            if not self.enabled:  # close may have won the lock while this request was waiting
                return
            for k in COUNTERS:
                self.lifetime[k] += vals[k]
            for segment in ("hour", "day"):
                self._bucket(segment, now, vals)
            self._save()

    def _bucket(self, segment: str, now: float, vals: dict) -> None:
        rows = self.buckets[segment]
        start = _align(now, segment)
        if rows and rows[-1]["t"] == start:
            for k in COUNTERS:
                rows[-1][k] += vals[k]
            return
        # a gap is fine: buckets with no requests simply do not exist
        if rows and start - rows[-1]["t"] > SEGMENT_S[segment] * KEEP[segment]:
            rows.clear()                                    # the last entry is older than the array keeps
        rows.append({"t": start, **vals})
        # a clock that steps backwards (an NTP step, a resume) can land on a bucket that is already
        # in the array: keep one row per bucket, in time order, so trimming keeps the newest ones
        if len(rows) > 1 and rows[-2]["t"] >= start:
            rows.sort(key=lambda b: b["t"])
            for i in range(len(rows) - 1, 0, -1):
                if rows[i]["t"] == rows[i - 1]["t"]:
                    for k in COUNTERS:
                        rows[i - 1][k] += rows[i][k]
                    del rows[i]
        del rows[:-KEEP[segment]]

    # -- reading -----------------------------------------------------------------------------------
    def _view(self, segment: str, count: int, now: float) -> dict:
        """`count` buckets ending with the current one, plus their sum."""
        rows = {b["t"]: b for b in self.buckets[segment]}
        step = SEGMENT_S[segment]
        current = _align(now, segment)
        out, total = [], _zero()
        for i in range(count - 1, -1, -1):
            t = current - i * step
            b = rows.get(t)
            vals = {k: (b[k] if b else 0) for k in COUNTERS}
            for k in COUNTERS:
                total[k] += vals[k]
            out.append({"t": t, "end": t + step, **vals})
        return {"buckets": out, "total": total}

    def _range_sum(self, segment: str, t0: float, t1: float) -> dict:
        """The sum of buckets starting in [t0, t1).  Buckets outside the array kept simply miss."""
        total = _zero()
        for b in self.buckets[segment]:
            if t0 <= b["t"] < t1:
                for k in COUNTERS:
                    total[k] += b[k]
        return total

    def view(self, now: float | None = None,
             custom: tuple[float, float] | None = None) -> dict:
        """GET /metrics' "usage": the lifetime counters, the 1 day / 7 day / 30 day views, and —
        when asked for — a custom [from, to) window (clamped to what the buckets keep)."""
        if not self.enabled:
            return {"enabled": False,
                    "note": "off; set \"usage_stats\": true in the run config or start with --usage-stats",
                    **({"persistence_error": self.persistence_error} if self.persistence_error else {})}
        now = float(now if now is not None else time.time())
        with self.lock:
            out = {"enabled": True, "since": self.since, "lifetime": dict(self.lifetime),
                   "imported": dict(self.imported), "persistence_error": self.persistence_error,
                   "now": now,
                   "views": {"1d": self._view("hour", 24, now),
                             "7d": self._view("day", 7, now),
                             "30d": self._view("day", 30, now)},
                   "kept_hours": KEEP["day"] * 24,             # the longest window the buckets can answer
                   "updated": now}
            if custom and all(_is_time(t) for t in custom):
                t0, t1 = float(custom[0]), float(custom[1])
                floor = max(self.since, now - KEEP["day"] * DAY)
                floor = _align(floor, "hour")                    # never cut the oldest kept bucket in half
                t0, t1 = max(t0, floor), min(t1, now)
                if t1 >= t0:      # a "to" a few ms behind the server clock is still a real request
                    out["views"]["custom"] = {"from": t0, "to": t1,
                                             "total": self._range_sum("hour", t0, t1)}
            return out


def totals_speed(row: dict) -> dict:
    """A row's derived rates (tok/s), for the page and for tests.  Missing denominators: None."""
    decode_ms, prompt_ms = row.get("decode_ms") or 0.0, row.get("prompt_ms") or 0.0
    fed = max(0, (row.get("prompt_tokens") or 0) - (row.get("reused") or 0))
    out = dict(row)
    out["read"] = fed
    out["decode_tok_s"] = round(row["output_tokens"] / (decode_ms / 1000), 1) \
        if decode_ms > 0 and row.get("output_tokens") else None
    out["prompt_tok_s"] = round(fed / (prompt_ms / 1000), 1) if prompt_ms > 0 and fed else None
    return out


def install_atexit(store: "Usage") -> None:
    """Flush the counters when the server exits normally."""
    if store.enabled:
        atexit.register(store.close)
