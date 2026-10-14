"""serve/usage.py - lifetime token statistics for the Monitor's Usage panel.

The engine already counts tokens per request; /metrics keeps those counters only since the server
started (the Monitor's "totals", issue #35).  This module keeps the same counters across restarts
and in time buckets, so the page can show lifetime figures and 1 day / 7 day / 30 day views.

Off by default.  Turned on by --usage-stats or "usage_stats": true in the run config (the Settings
view lists it).  When on, one small JSON file is written beside the run config, e.g.
strata-iq3_s.usage.json; it holds only counters and bucket start times - no prompts, no answers,
no request bodies.  Writes are throttled and atomic, so a killed server loses at most the last
minute.  Unreadable or corrupt statistics are started over rather than failing the boot.

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
import os
import threading
import time
from pathlib import Path

HOUR = 3600.0
DAY = 24 * HOUR
SEGMENT_S = {"hour": HOUR, "day": DAY}       # the bucket size of each kept array
KEEP = {"hour": 744, "day": 31}               # buckets kept per array (31 days, hour and day)
COUNTERS = ("requests", "prompt_tokens", "reused", "output_tokens", "prompt_ms", "decode_ms",
            "drafts_offered", "drafts_accepted")
WRITE_EVERY_S = 60.0                         # throttled writes: at most one per minute


def _zero() -> dict:
    return {k: 0 for k in COUNTERS}


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
        self._written = 0.0
        if self.path:
            self._load(now)

    # -- state -------------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _load(self, now: float | None) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:                                   # corrupt or unreadable: start over
            print(f"[strata] usage statistics at {self.path.name} could not be read; starting over",
                  flush=True)
            return
        if not isinstance(data, dict):
            return
        life = data.get("lifetime")
        if isinstance(life, dict):
            self.lifetime = {k: int(life.get(k) or 0) for k in COUNTERS}
        if isinstance(data.get("since"), (int, float)):
            self.since = float(data["since"])
        for segment in ("hour", "day"):
            rows = data.get(segment)
            out = []
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict) or not isinstance(r.get("t"), (int, float)):
                        continue
                    bucket = {k: int(r.get(k) or 0) for k in COUNTERS}
                    bucket["t"] = float(r["t"])
                    out.append(bucket)
            out.sort(key=lambda b: b["t"])
            self.buckets[segment] = out[-KEEP[segment]:]

    def _save(self) -> None:
        if not self.path:
            return
        data = {"since": self.since, "updated": time.time(), "lifetime": self.lifetime,
                "hour": self.buckets["hour"], "day": self.buckets["day"]}
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as e:                              # a statistics write must never break serving
            print(f"[strata] usage statistics could not be written to {self.path.name}: {e}", flush=True)

    def flush(self) -> None:
        """Write now, ignoring the throttle (shutdown, tests)."""
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
        vals = {"requests": 1, "prompt_tokens": int(prompt_tokens or 0), "reused": int(reused or 0),
                "output_tokens": int(output_tokens or 0), "prompt_ms": float(prompt_ms or 0.0),
                "decode_ms": float(decode_ms or 0.0), "drafts_offered": int(drafts_offered or 0),
                "drafts_accepted": int(drafts_accepted or 0)}
        with self.lock:
            for k in COUNTERS:
                self.lifetime[k] += vals[k]
            for segment in ("hour", "day"):
                self._bucket(segment, now, vals)
            if now - self._written >= WRITE_EVERY_S:
                self._written = now
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
                    "note": "off; set \"usage_stats\": true in the run config or start with --usage-stats"}
        now = float(now if now is not None else time.time())
        with self.lock:
            out = {"enabled": True, "since": self.since, "lifetime": dict(self.lifetime),
                   "now": now,
                   "views": {"1d": self._view("hour", 24, now),
                             "7d": self._view("day", 7, now),
                             "30d": self._view("day", 30, now)},
                   "kept_hours": KEEP["day"] * 24,             # the longest window the buckets can answer
                   "updated": now}
            if custom:
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
        atexit.register(store.flush)
