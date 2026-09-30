"""serve/kvcache.py - what the NVMe cache tiers did, read from the serve side.

The two serve-side sources of docs/nvme-kv-cache-web-design.md §2, plus the parser for the engine's third:

- the engine's own arguments (design §2.1): enabled, the store directory, the byte cap, the tier family, and
  whether a layer split makes the tier inert.  This is the only source that survives the engine being dead, which
  is the §5.2 case the page exists to explain.
- the store directory itself (design §2.2): counts and bytes per class, the format version of every snapshot and
  manifest, crash residue, and the free space on that filesystem.  What the engine's `open()` refused is on disk
  and is NOT promotable, so the walk's bytes and the engine's bytes are kept in separate, named keys (§6).
- the `KV` line the engine prints (design §3): `parse_kv` types it, `store_state()` takes the store fields of any
  such line (including the startup `start=1` one) and `observe()` turns one request's line into event rows.

Same rule as serve/telemetry.py: nothing here can stop the server.  Every filesystem and parse path is wrapped and
anything that cannot be read degrades to None / {} / a warning row.  The module starts no thread and prints
nothing - the server prints, and the wiring step decides when this is sampled.
"""
from __future__ import annotations

import collections
import os
import shutil
import struct
import threading
import time

SCAN_S = 5.0                     # one directory walk per this many seconds, at most (design §4)
EVENT_HISTORY = 500              # the event rows kept, newest last (Service.history's shape, server.py:546)
SUMMARY_EVENTS = 12              # how many of them summary() carries into /metrics
SERIES_HISTORY = 60              # the sparkline points (serve/telemetry.py's HISTORY)
PREFIX_LIMIT = 12                # the "Stored prefixes" table's rows

# The format identities, named exactly as the two stores write them (include/strata/platform/kv_nvme.hpp,
# include/strata/platform/kv_delta.hpp).  Both families put a uint32 magic at offset 0 and a uint32 version at
# offset 4, and both put the prefix length (an int64) right after them - pinned by the static_asserts there.
KV_SNAPSHOT_MAGIC = 0x5E564D45        # "^VME", a v3 snapshot
KV_DELTA_MANIFEST_MAGIC = 0x474F4C44  # "DLOG"
KV_SNAPSHOT_VERSION = 3
KV_DELTA_VERSION = 1
KV_MANIFEST_N_CHUNKS_OFF = 24         # DeltaManifestHeader.n_chunks
KV_MANIFEST_STATE_KEY_OFF = 232       # DeltaManifestHeader.state_key
KV_MANIFEST_FOOTER_BYTES = 8          # the FNV-1a over the body, the file's last 8 bytes
KV_MANIFEST_CHUNK_REF_BYTES = 16      # {u64 key; int64 a}: the body's last field, before the footer

# The classes of the store directory, as design §2.2 lists them.  A v3 snapshot has NO temp name (kv_nvme.cpp
# writes it under its final name and a torn one is caught by its footer), so `residue` is the delta tier's own
# `.tmp-` class (kv_delta.cpp:116-145) and nothing else.
CLASSES = ("snapshots", "manifests", "chunks", "states", "residue")

KV_FLOAT_KEYS = ("promote_ms", "dump_ms")
KV_STORE_KEYS = ("entries", "entries_bytes", "delta_entries", "delta_bytes", "cap", "checkpoints", "live")
KV_TOTAL_KEYS = ("total_dump_bytes", "total_promote_bytes", "total_refused", "total_transfer", "total_evict_bytes")
KV_SERIES = ("store_bytes", "write_mb", "read_mb", "warm")


def _as_int(v) -> int | None:
    """A value as an int, or None when it is not one (a bool is not a count)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v) -> float | None:
    """A value as a float, or None when it is not one."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _flag(args: list[str], name: str) -> str | None:
    """The value of `--name VALUE` in the engine's arguments - the engine's own parser shape (generate.cpp:1103) -
    or None when the flag is absent or has nothing after it."""
    for i, a in enumerate(args):
        if a == name:
            return args[i + 1] if i + 1 < len(args) else None
    return None


def parse_kv(line: str) -> dict:
    """The engine's `KV k=v k=v ...` line -> a dict (design §3): space-separated pairs, ints typed, `promote_ms`
    and `dump_ms` as floats, `src` and any key this build does not know kept as the string it was.

    Nothing here can raise: a truncated pair (`resume=`, `resume`) is dropped, a value that is not a number stays
    a string, and a line that is not a KV line yields only what it happened to parse ({} for `KV` alone).  Every
    reader here takes a value through `_as_int` / `_as_float`, so a garbage value costs one fact, never the
    request.
    """
    out: dict = {}
    if not isinstance(line, str):
        return out
    for token in line.split():
        if token == "KV":
            continue
        key, sep, value = token.partition("=")
        if not sep or not key or not value:
            continue
        if key in KV_FLOAT_KEYS:
            f = _as_float(value)
            out[key] = f if f is not None else value
            continue
        n = _as_int(value)
        out[key] = n if n is not None else value
    return out


def _class_row() -> dict:
    """One class of the walk: how many, how many bytes, and the age of its warmest and coldest file.  The two head
    classes gain `promotable` / `stale` / `foreign` on top: the split design §2.2 insists on, because a file the
    engine's open() refused is on disk and is not promotable."""
    return {"count": 0, "bytes": 0, "newest_mtime": None, "oldest_mtime": None,
            "newest_age_s": None, "oldest_age_s": None}


def _read_header(path: str) -> tuple[int, int, int | None] | None:
    """A record's first 16 bytes: the uint32 magic at offset 0, the uint32 version at offset 4, and the int64
    prefix length that follows them in BOTH families (NvmeHeader and DeltaManifestHeader pin offsetof(L) == 8).
    None when the file cannot be read or is shorter than the magic and version every record must start with."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if len(head) < 8:
        return None
    magic, version = struct.unpack("<II", head[:8])
    length = struct.unpack("<q", head[8:16])[0] if len(head) >= 16 else None
    return magic, version, (length if length is not None and length >= 0 else None)


def _manifest_records(dir_: str, path: str, size: int) -> tuple[dict, int]:
    """What one manifest's prefix actually occupies on disk: the manifest, its State record, and the chunks its
    body names.  The chunk refs are the body's LAST field before the 8-byte footer (kv_delta.hpp:197), so they are
    read from the end of the file - independent of how long the ids and image keys before them are.  A chunk shared
    by two manifests is counted in both rows: these are per-prefix footprints, and summing the rows would
    double-count them (which is exactly what the engine's cap accounting does on purpose, design §6)."""
    try:
        with open(path, "rb") as f:
            head = f.read(KV_MANIFEST_STATE_KEY_OFF + 8)
    except OSError:
        return {"manifests": 1}, size
    if len(head) < KV_MANIFEST_STATE_KEY_OFF + 8:
        return {"manifests": 1}, size
    n_chunks = struct.unpack("<q", head[KV_MANIFEST_N_CHUNKS_OFF:KV_MANIFEST_N_CHUNKS_OFF + 8])[0]
    state_key = struct.unpack("<Q", head[KV_MANIFEST_STATE_KEY_OFF:KV_MANIFEST_STATE_KEY_OFF + 8])[0]
    records, total = {"manifests": 1, "chunks": 0, "states": 0}, size
    try:
        total += os.stat(os.path.join(dir_, "delta", "states", "%016x.bin" % state_key)).st_size
        records["states"] = 1
    except OSError:
        pass
    # a header is not trusted against the file it is in: only a ref count the body can hold is read
    if 0 < n_chunks <= (size - KV_MANIFEST_FOOTER_BYTES) // KV_MANIFEST_CHUNK_REF_BYTES:
        try:
            with open(path, "rb") as f:
                f.seek(size - KV_MANIFEST_FOOTER_BYTES - n_chunks * KV_MANIFEST_CHUNK_REF_BYTES)
                tail = f.read(n_chunks * KV_MANIFEST_CHUNK_REF_BYTES)
        except OSError:
            tail = b""
        for i in range(len(tail) // KV_MANIFEST_CHUNK_REF_BYTES):
            key = struct.unpack("<Q", tail[i * KV_MANIFEST_CHUNK_REF_BYTES:
                                       i * KV_MANIFEST_CHUNK_REF_BYTES + 8])[0]
            try:
                total += os.stat(os.path.join(dir_, "delta", "chunks", "%016x.bin" % key)).st_size
                records["chunks"] += 1
            except OSError:
                pass
    return records, total


class KvCache:
    """The serve side's view of the cache tiers: the config facts, a throttled store walk, and the event rows the
    engine's `KV` lines produce.  Built from the FINAL engine arguments (`engine_args(cfg)`), so a `--layer-split`
    the server added itself is seen too.  Without `--kv-nvme` it is inert: `enabled` is False and every method is a
    no-op returning {} / None, so a tier-off server does no filesystem work at all."""

    def __init__(self, engine_args: list[str], log_path: str | None = None):
        args = [str(a) for a in (engine_args or [])]
        self.dir = _flag(args, "--kv-nvme") or ""
        self.enabled = self.dir != ""
        self.log_path = log_path        # the engine's log: the wiring step quotes it on a transfer-failure row
        gb = _as_int(_flag(args, "--kv-nvme-max"))
        self.cap_gb = 100 if gb is None else max(0, gb)      # the engine's own default (generate.cpp:291)
        self.cap_bytes = self.cap_gb * (1 << 30) if self.cap_gb > 0 else 0    # 0 = unlimited
        delta = _as_int(_flag(args, "--kv-delta"))
        self.delta = True if delta is None else delta != 0    # --kv-delta defaults to 1 (generate.cpp:298)
        self.mode = "delta" if self.delta else "v3"
        self.layer_split = "--layer-split" in args
        self.inert_reason = ("the layer split is active - stored snapshots are neither dumped nor promoted "
                             "(the envelope carries the primary stage only)") if self.layer_split else None
        self.events: collections.deque = collections.deque(maxlen=EVENT_HISTORY)
        self.store: dict = {}                                 # the engine's store fields, last line wins
        self.totals: dict = {"since": time.time(), "requests": 0, "requests_with_kv_line": 0, "promotes": 0}
        self.warnings: list[str] = [self.inert_reason] if self.inert_reason else []
        self._series = {k: collections.deque(maxlen=SERIES_HISTORY) for k in KV_SERIES}
        self._last_write_mb = 0.0         # the last turn's cascade write, for its sparkline
        self._last_read_mb = 0.0          # the last promote's read
        self._scan: dict | None = None
        self._scan_at = 0.0
        self._lock = threading.Lock()     # the request thread writes; the sampler and the readers read
        self._scan_lock = threading.Lock()

    # ------------------------------------------------------------------------ the engine's lines

    def store_state(self, kv: dict) -> None:
        """Merge the store fields of any `KV` line - the startup `start=1` one and every request's - into what
        this reports.  Store state, never an event (design §3, ordering note 2)."""
        if not self.enabled or not isinstance(kv, dict):
            return
        seen = {k: v for k, v in ((k, _as_int(kv.get(k))) for k in KV_STORE_KEYS) if v is not None}
        if not seen:
            return
        with self._lock:
            self.store.update(seen)

    def observe(self, kv: dict, done: dict) -> None:
        """One finished request: its `KV` line (None when it had none) and its `DONE` fields.

        A line with no `src=` is not a request line - it is store state, so it is merged and produces no event
        row.  A missing or garbage line produces no row either and is never recorded as cold: "no `KV` line" is
        unknown, not `src=none` (design §3, note 1).  The cumulative totals are the engine's own `total_*` values,
        last seen wins, so a line this could not parse costs one row, never the running totals."""
        if not self.enabled:
            return
        if not isinstance(kv, dict) or not kv:
            with self._lock:
                self.totals["requests"] += 1
            return
        if "src" not in kv:
            self.store_state(kv)
            return
        self.store_state(kv)
        src = kv.get("src") if isinstance(kv.get("src"), str) else None
        finish = done.get("finish") if isinstance(done, dict) else None
        resume, promote_bytes = _as_int(kv.get("resume")), _as_int(kv.get("promote_bytes"))
        promote_ms = _as_float(kv.get("promote_ms"))
        dump_bytes, dump_ms = _as_int(kv.get("dump_bytes")), _as_float(kv.get("dump_ms"))
        refused, transfer = _as_int(kv.get("refused")) or 0, _as_int(kv.get("transfer")) or 0
        evict, evict_bytes = _as_int(kv.get("evict")) or 0, _as_int(kv.get("evict_bytes")) or 0
        sweep, sweep_bytes = _as_int(kv.get("sweep")) or 0, _as_int(kv.get("sweep_bytes")) or 0
        now = time.time()

        def row(kind: str, tokens, nbytes, ms, count: int) -> dict:
            return {"time": round(now, 3), "kind": kind, "src": src, "tokens": tokens,
                    "bytes": nbytes, "ms": ms, "count": count, "finish": finish}

        rows: list[dict] = []
        # `src` is nvme / delta ONLY for a promote that landed (generate.cpp:3881); a refusal and a transfer
        # failure both say src=none, so a promote row is never a promote that failed, and `ram` is the RAM tier,
        # which no directory walk can see.
        if src in ("nvme", "delta"):
            rows.append(row("promote", resume, promote_bytes, promote_ms, 1))
        if (dump_bytes or 0) > 0 or (dump_ms or 0) > 0:
            rows.append(row("cascade", None, dump_bytes, dump_ms, 1))   # the line has no dumped-token count
        if refused:
            rows.append(row("refuse", None, None, None, refused))
        if transfer:
            rows.append(row("transfer", None, None, None, transfer))
        if evict or evict_bytes:
            rows.append(row("evict", None, evict_bytes, None, evict or 1))
        if sweep or sweep_bytes:
            rows.append(row("sweep", None, sweep_bytes, None, sweep or 1))
        with self._lock:
            self.totals["requests"] += 1
            self.totals["requests_with_kv_line"] += 1
            for key in KV_TOTAL_KEYS:
                v = _as_int(kv.get(key))
                if v is not None:
                    self.totals[key] = v     # the engine's counters, not a sum of what this side happened to see
            self.events.extend(rows)
            if rows and rows[0]["kind"] == "promote":
                self.totals["promotes"] += 1
            self._last_write_mb = round((dump_bytes or 0) / 2**20, 2)
            self._last_read_mb = round((promote_bytes or 0) / 2**20, 2)

    # ------------------------------------------------------------------------ what it reports

    def _promotable_locked(self) -> dict:
        """The engine's own store numbers - CAP ACCOUNTING, a SAWTOOTH, not a footprint (design §6): the delta
        tier counts a chunk shared by three manifests once per manifest, so `delta_bytes` drifts ABOVE the disk
        as shared references accumulate (the safe direction - over-evict, never under-evict), and every sweep
        recomputes the total from the disk and snaps the books back. The walk's `on_disk` is the OTHER quantity;
        the two agree exactly only right after a sweep, which is a coincidence, not an invariant, and they are
        never merged."""
        s = self.store
        entries, delta_entries = s.get("entries"), s.get("delta_entries")
        entries_bytes, delta_bytes = s.get("entries_bytes"), s.get("delta_bytes")
        return {"prefixes": (entries or 0) + (delta_entries or 0) if entries is not None or delta_entries is not None
                else None,
                "bytes": (entries_bytes or 0) + (delta_bytes or 0) if entries_bytes is not None
                or delta_bytes is not None else None,
                "entries": entries, "entries_bytes": entries_bytes,
                "delta_entries": delta_entries, "delta_bytes": delta_bytes}

    def summary(self) -> dict:
        """The cheap block: what /metrics can carry every second (design §4).  No filesystem work."""
        if not self.enabled:
            return {}
        with self._lock:
            return {"enabled": True, "mode": self.mode, "dir": self.dir,
                    "cap_bytes": self.store.get("cap", self.cap_bytes),   # the engine's is the one it enforces
                    "inert_reason": self.inert_reason,
                    "promotable": self._promotable_locked(),
                    "ram_tier": {"checkpoints": self.store.get("checkpoints"), "live_tokens": self.store.get("live")},
                    "totals": dict(self.totals),
                    "events": [dict(e) for e in list(self.events)[-SUMMARY_EVENTS:]],
                    "series": {k: list(v) for k, v in self._series.items()},
                    "warnings": list(self.warnings)}

    def detail(self) -> dict:
        """summary() plus the store walk's tables - the /cache payload (design §4).  The walk is what `scan()`
        throttles, so an unwatched page costs nothing."""
        if not self.enabled:
            return {}
        out = self.summary()
        with self._lock:
            out["events"] = [dict(e) for e in self.events]       # /cache is where "show all" reads
        scanned = self.scan()
        out.update(scanned)
        out["warnings"] = out["warnings"] + list(scanned.get("warnings") or [])
        return out

    def series(self) -> dict:
        """The sparkline points, one series per name (design §4): `store_bytes` the engine's cap-accounting bytes,
        `write_mb` the last turn's cascade write, `read_mb` the last promote's read, `warm` the promotable
        prefixes.  The wiring step calls it from the telemetry hook, so it APPENDS one point per call and the
        sampling shares that thread's clock; summary() reports the same points without appending."""
        if not self.enabled:
            return {}
        with self._lock:
            p = self._promotable_locked()
            self._series["store_bytes"].append(p["bytes"] or 0)
            self._series["write_mb"].append(self._last_write_mb)
            self._series["read_mb"].append(self._last_read_mb)
            self._series["warm"].append(p["prefixes"] or 0)
            return {k: list(v) for k, v in self._series.items()}

    # ------------------------------------------------------------------------ the store walk

    def scan(self, force: bool = False) -> dict:
        """The store directory as it lies on disk (design §2.2), throttled to one walk per SCAN_S and cached with
        its timestamp; `force` re-walks now.  The result is shared, not copied: readers read it, they do not edit
        it.  A directory that is not there or cannot be read is a warning row, never an exception."""
        if not self.enabled:
            return {}
        now = time.time()
        with self._scan_lock:
            if not force and self._scan is not None and now - self._scan_at < SCAN_S:
                return self._scan
            self._scan = self._walk(now)
            self._scan_at = now
            return self._scan

    def _walk(self, now: float) -> dict:
        """One pass over `<dir>` and `<dir>/delta{,/chunks,/states}`: the seam the tests count, and the only
        place in this module that touches the filesystem."""
        rows = {name: _class_row() for name in CLASSES}
        # Only the two head families carry a format version, so only they split into promotable / stale / foreign.
        for head in ("snapshots", "manifests"):
            rows[head].update({"promotable": 0, "stale": 0, "foreign": 0})
        stale = {"count": 0, "version": None}
        foreign = 0
        prefixes: list[dict] = []
        unreadable: list[str] = []

        def add(kind: str, size: int, mtime: float) -> None:
            r = rows[kind]
            r["count"] += 1
            r["bytes"] += size
            if r["newest_mtime"] is None or mtime > r["newest_mtime"]:
                r["newest_mtime"] = mtime
            if r["oldest_mtime"] is None or mtime < r["oldest_mtime"]:
                r["oldest_mtime"] = mtime

        found: list[tuple[str, int, float]] = []      # (path, size, mtime) per record, in walk order
        classes: list[str] = []
        if os.path.isdir(self.dir):
            for dirpath, _, names in os.walk(self.dir, onerror=lambda e: unreadable.append(str(e))):
                rel = os.path.relpath(dirpath, self.dir)
                for name in names:
                    path = os.path.join(dirpath, name)
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    if name.startswith(".tmp-") and rel != ".":
                        kind = "residue"                    # the delta tier's temp class; a v3 file has none
                    elif rel == "." and name.startswith("kv-"):
                        kind = "snapshots"
                    elif rel == "delta" and name.startswith("log-"):
                        kind = "manifests"
                    elif rel == os.path.join("delta", "chunks"):
                        kind = "chunks"
                    elif rel == os.path.join("delta", "states"):
                        kind = "states"
                    else:
                        continue        # not a record of either tier: not counted, not reported
                    add(kind, st.st_size, st.st_mtime)
                    found.append((path, st.st_size, st.st_mtime))
                    classes.append(kind)
        else:
            unreadable.append(self.dir)

        for (path, size, mtime), kind in zip(found, classes):
            if kind not in ("snapshots", "manifests"):
                continue                       # only the two head families carry the version the §5.3 fact needs
            head = _read_header(path)
            if head is None:
                foreign += 1
                rows[kind]["foreign"] += 1
                continue
            magic, version, length = head
            want = KV_SNAPSHOT_MAGIC if kind == "snapshots" else KV_DELTA_MANIFEST_MAGIC
            mine = KV_SNAPSHOT_VERSION if kind == "snapshots" else KV_DELTA_VERSION
            if magic != want:
                foreign += 1                   # not this tier's file at all: on disk, not promotable
                rows[kind]["foreign"] += 1
                continue
            if version != mine:
                stale["count"] += 1
                stale["version"] = version     # the whole store is of another build; the version says which
                rows[kind]["stale"] += 1
                continue
            if kind == "snapshots":
                records, total = {"snapshots": 1}, size
            else:
                records, total = _manifest_records(self.dir, path, size)
            rows[kind]["promotable"] += 1
            prefixes.append((mtime, {"age_s": round(max(0.0, now - mtime), 1), "tokens": length,
                                     "tier": "v3" if kind == "snapshots" else "delta",
                                     "kind": "snapshot" if kind == "snapshots" else "manifest",
                                     "records": records, "bytes": total}))

        for r in rows.values():
            if r["newest_mtime"] is not None:
                r["newest_age_s"] = round(max(0.0, now - r["newest_mtime"]), 1)
                r["oldest_age_s"] = round(max(0.0, now - r["oldest_mtime"]), 1)

        warnings: list[str] = []
        if not os.path.isdir(self.dir):
            warnings.append(f"the store directory {self.dir} is not there (the engine creates it when it opens "
                            "the tier)")
        elif unreadable:
            warnings.append(f"the store directory {self.dir} could not be read: {unreadable[0]}")
        if stale["count"]:
            warnings.append(f"{stale['count']} file(s) of format version {stale['version']} in {self.dir}: this "
                            f"build writes {KV_SNAPSHOT_VERSION} (snapshots) and {KV_DELTA_VERSION} (delta "
                            "records) and refuses them - on disk, not promotable")
        if foreign:
            warnings.append(f"{foreign} file(s) in {self.dir} are not this tier's records (foreign magic): on "
                            "disk, not promotable")
        if rows["residue"]["count"]:
            warnings.append(f"{rows['residue']['count']} .tmp-* residue file(s) "
                            f"({rows['residue']['bytes']} bytes) - a crash's leftovers, which the tier's sweep "
                            "reclaims")

        try:
            du = shutil.disk_usage(self.dir)
            disk_free, disk_total = du.free, du.total
        except OSError:
            disk_free = disk_total = None

        prefixes.sort(key=lambda p: p[0], reverse=True)
        return {"scanned_at": now, "on_disk": rows,
                "on_disk_bytes": sum(r["bytes"] for r in rows.values()),
                "on_disk_files": sum(r["count"] for r in rows.values()),
                "stale": stale, "foreign": foreign,
                "prefixes": [p[1] for p in prefixes[:PREFIX_LIMIT]],
                "disk_free_bytes": disk_free, "disk_total_bytes": disk_total,
                "warnings": warnings}
