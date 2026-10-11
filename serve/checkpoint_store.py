"""Disposable, content-addressed execution checkpoints with measured-cost eviction.

Owned by the server's history-store process; all filesystem and catalog changes
are serialized. History is deliberately not stored here. Native session bytes
are opaque: this layer never strips recurrent, MTP, or backend state.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import uuid


def sync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class CacheFull(OSError):
    pass


class CheckpointStore:
    def __init__(self, directory, *, budget_bytes, reserve_bytes=4 << 30, archive=None,
                 archive_budget_bytes=0, chunk_bytes=0, policy="utility", clock=time.time,
                 free_bytes=None, half_life_s=86400, max_group=8):
        if policy not in ("utility", "fifo", "lru"):
            raise ValueError("unknown checkpoint eviction policy")
        for n in (budget_bytes, reserve_bytes, archive_budget_bytes, chunk_bytes):
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("checkpoint sizes must be nonnegative integers")
        if half_life_s <= 0 or not math.isfinite(half_life_s) or not 1 <= max_group <= 16:
            raise ValueError("invalid demand half-life or dependency group bound")
        self.root = Path(directory).resolve()
        self.paths = {"local": self.root / "blocks"}
        self.root.mkdir(parents=True, exist_ok=True)
        namespace_path=self.root/"archive-namespace"
        if not namespace_path.exists():
            with namespace_path.open("x") as f:
                f.write(uuid.uuid4().hex)
                f.flush()
                os.fsync(f.fileno())
            sync_directory(self.root)
        namespace=namespace_path.read_text().strip()
        if len(namespace)!=32 or any(c not in "0123456789abcdef" for c in namespace):
            raise ValueError("invalid checkpoint archive namespace")
        if archive:
            # Require an existing mount, never create a missing mountpoint on local root storage.
            pool = Path(archive).resolve()
            if pool == self.root or self.root in pool.parents or pool in self.root.parents:
                raise ValueError("checkpoint archive must be separate from local cache")
            if pool.is_dir():
                self.paths["archive"] = pool / "strata-checkpoint-blocks" / namespace
        for path in self.paths.values():
            path.mkdir(parents=True,exist_ok=True)
        self.budgets = {"local": budget_bytes, "archive": archive_budget_bytes}
        self.reserve = reserve_bytes
        self.chunk = chunk_bytes
        self.policy, self.clock = policy, clock
        self.free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)
        self.half_life, self.max_group = half_life_s, max_group
        self.lock = threading.RLock()
        self.leases = set()
        self.db = sqlite3.connect(self.root / "checkpoints.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, tokens TEXT NOT NULL,
                blocks TEXT NOT NULL, tier TEXT NOT NULL, restore_s REAL NOT NULL,
                created REAL NOT NULL, pinned INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS demand (
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, tokens TEXT NOT NULL,
                replay_s REAL NOT NULL, frequency REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS decisions (
                time REAL NOT NULL, action TEXT NOT NULL, ids TEXT NOT NULL,
                bytes INTEGER NOT NULL, lost_seconds REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS owners (
                snapshot TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(snapshot,response));
            CREATE TABLE IF NOT EXISTS successful_restores (id TEXT PRIMARY KEY, last_used REAL NOT NULL);
        """)
        with self.lock:
            for p in self.root.iterdir():
                stem = p.name.split(".", 1)[0]
                if (p.name.endswith((".save.tmp", ".restore.tmp")) and len(stem)==32 and
                        all(c in "0123456789abcdef" for c in stem) and p.is_file()):
                    p.unlink()  # prior owner exited; history owner lock excludes another server
            self.collect_orphans()

    def close(self):
        self.db.close()

    def restart_cleanup(self, protected_responses):
        """Boot/shutdown boundary: only bookmarked history owns durable acceleration.

        Run under the history directory's owner lock, before accepting requests
        or after draining them. A prior crash's in-memory leases do not survive.
        """
        with self.lock:
            if self.leases:
                raise RuntimeError("cannot clean restart cache while a checkpoint is in use")
            keep={s for s,r in self.db.execute("SELECT snapshot,response FROM owners") if r in protected_responses}
            with self.db:
                self.db.executemany("DELETE FROM snapshots WHERE id=?",
                                    [(r[0],) for r in self.db.execute("SELECT id FROM snapshots").fetchall() if r[0] not in keep])
                self.db.execute("DELETE FROM owners WHERE snapshot NOT IN (SELECT id FROM snapshots)")
                self.db.execute("DELETE FROM demand")
                self.db.execute("DELETE FROM successful_restores")
                self.db.execute("DELETE FROM decisions")
            self.collect_orphans()
            # Reclaim transient catalog pages rather than keeping an ever-growing statistics file.
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")

    def _rows(self):
        return [dict(id=r[0], fingerprint=r[1], tokens=json.loads(r[2]), blocks=json.loads(r[3]),
                     tier=r[4], restore_s=r[5], created=r[6], pinned=bool(r[7]))
                for r in self.db.execute("SELECT * FROM snapshots ORDER BY id")]

    def _available(self, row):
        return row["tier"] in self.paths and all((self.paths[row["tier"]] / h).is_file() for h, _ in row["blocks"])

    def attach(self, sid, response_id):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO owners VALUES(?,?)", (sid, response_id))

    def add_prefix(self, sid, tokens):
        """Index another native-verified ancestor in the same opaque session."""
        with self.lock, self.db:
            rows=self._rows()
            source=next(s for s in rows if s["id"]==sid)
            if not tokens or tokens!=source["tokens"][:len(tokens)]:
                raise ValueError("checkpoint alias must be an ancestor")
            for row in rows:
                if (row["fingerprint"]==source["fingerprint"] and row["blocks"]==source["blocks"] and
                        row["tier"]==source["tier"] and row["tokens"]==tokens):
                    return row["id"]
            alias=uuid.uuid4().hex
            self.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,0)", (alias,source["fingerprint"],
                json.dumps(tokens,separators=(",",":")),json.dumps(source["blocks"]),source["tier"],
                source["restore_s"],source["created"]))
            return alias

    def expire_owners(self, retained_responses):
        with self.lock, self.db:
            stale = [(s,r) for s,r in self.db.execute("SELECT snapshot,response FROM owners")
                     if r not in retained_responses]
            self.db.executemany("DELETE FROM owners WHERE snapshot=? AND response=?", stale)
            for sid, _ in stale:
                if sid not in self.leases:
                    self.db.execute("DELETE FROM snapshots WHERE id=? AND pinned=0 AND NOT EXISTS "
                                    "(SELECT 1 FROM owners WHERE snapshot=?)", (sid,sid))
            self.db.execute("DELETE FROM owners WHERE snapshot NOT IN (SELECT id FROM snapshots)")
            self.db.execute("DELETE FROM demand WHERE updated < ?", (self.clock()-30*86400,))
        self.collect_orphans()

    def used(self, tier="local"):
        # Count actual allocated bytes where supported; never double-count shared blocks.
        return sum(self._size(p) for p in self.paths[tier].iterdir() if p.is_file())

    @staticmethod
    def _size(path):
        stat = path.stat()
        return stat.st_blocks * 512 if hasattr(stat, "st_blocks") else stat.st_size

    def _weights(self):
        now = self.clock()
        return [dict(id=r[0], fingerprint=r[1], tokens=json.loads(r[2]), replay_s=r[3],
                     weight=max(0.05, r[4] * 2 ** (-max(0, now-r[5]) / self.half_life)))
                for r in self.db.execute("SELECT * FROM demand")]

    @staticmethod
    def cost(query, snapshots):
        n = len(query["tokens"])
        choices = [(query["replay_s"], None)]
        for s in snapshots:
            k = len(s["tokens"])
            if s["fingerprint"] == query["fingerprint"] and k <= n and s["tokens"] == query["tokens"][:k]:
                choices.append((s["restore_s"] + query["replay_s"] * (n-k)/max(1, n), s["id"]))
        return min(choices, key=lambda c: (c[0], c[1] or ""))

    def observe(self, fingerprint, tokens, replay_s, *, successful=True):
        """Call only after successful user execution, never on access/inspection."""
        if not successful:
            return
        if not math.isfinite(replay_s) or replay_s < 0:
            raise ValueError("replay cost must be finite and nonnegative")
        raw = json.dumps(tokens, separators=(",", ":"))
        key = hashlib.sha256((fingerprint + raw).encode()).hexdigest()
        with self.lock, self.db:
            old = self.db.execute("SELECT frequency,updated,replay_s FROM demand WHERE id=?", (key,)).fetchone()
            now = self.clock()
            frequency = 1 + (old[0] * 2 ** (-max(0, now-old[1])/self.half_life) if old else 0)
            cost = replay_s if old is None else 0.25 * replay_s + 0.75 * old[2]
            self.db.execute("INSERT OR REPLACE INTO demand VALUES(?,?,?,?,?,?)",
                            (key, fingerprint, raw, cost, frequency, now))

    def best(self, fingerprint, tokens, replay_s):
        with self.lock:
            available = [s for s in self._rows() if self._available(s)]
            cost, sid = self.cost(dict(fingerprint=fingerprint, tokens=tokens, replay_s=replay_s), available)
            return next((s for s in available if s["id"] == sid), None), cost

    def _freed(self, ids, rows, tier):
        removed, kept = {}, set()
        for row in rows:
            if row["tier"] != tier:
                continue
            if row["id"] in ids:
                removed.update(dict(row["blocks"]))
            else:
                kept.update(h for h, _ in row["blocks"])
        return sum(self._size(self.paths[tier] / h) for h in removed.keys() - kept
                   if (self.paths[tier] / h).exists())

    def candidates(self, tier="local", *, allow_demote=True):
        rows = self._rows()
        eligible = {s["id"] for s in rows if s["tier"] == tier and not s["pinned"] and s["id"] not in self.leases}
        groups = {frozenset([sid]) for sid in eligible}
        dependencies = {}
        for s in rows:
            if s["tier"] == tier:
                for h, _ in s["blocks"]:
                    dependencies.setdefault(h, set()).add(s["id"])
        for owners in dependencies.values():
            if owners <= eligible and len(owners) <= self.max_group:
                groups.add(frozenset(owners))
        # Also combine small dependency groups to address coupled alternatives.
        seeds = sorted(groups, key=lambda g: (len(g), sorted(g)))[:64]
        for a, b in itertools.combinations(seeds if self.policy == "utility" else [], 2):
            if len(a | b) <= self.max_group:
                groups.add(a | b)
        available = [r for r in rows if self._available(r)]
        demand = self._weights()
        # Cold snapshots contribute conservative demand until observed.
        known = {(q["fingerprint"], tuple(q["tokens"])) for q in demand}
        for s in available:
            if (s["fingerprint"], tuple(s["tokens"])) not in known:
                demand.append(dict(fingerprint=s["fingerprint"], tokens=s["tokens"],
                                   replay_s=max(s["restore_s"]*2, len(s["tokens"])/100), weight=0.05))
        # Prefix compatibility is the expensive part at long context. Compute it
        # once, not inside every deletion-set evaluation.
        baseline = []
        for q in demand:
            choices=[(q["replay_s"],None)]
            n=len(q["tokens"])
            for s in available:
                k=len(s["tokens"])
                if s["fingerprint"]==q["fingerprint"] and k<=n and s["tokens"]==q["tokens"][:k]:
                    choices.append((s["restore_s"]+q["replay_s"]*(n-k)/max(1,n),s["id"]))
            choices.sort(key=lambda c:(c[0],c[1] or ""))
            baseline.append((q["weight"],choices[0][0],choices))
        physical={h:self._size(self.paths[tier]/h) for h in dependencies if (self.paths[tier]/h).exists()}
        archive_present={h for h in dependencies if "archive" in self.paths and (self.paths["archive"]/h).exists()}
        result = []
        for ids in groups:
            freed = sum(physical.get(h,0) for h,owners in dependencies.items() if owners<=ids)
            if not freed:
                continue
            lost = sum(weight*max(0,next(cost for cost,sid in choices if sid not in ids)-before)
                       for weight,before,choices in baseline)
            if self.policy == "utility":
                rank = lost / freed
            elif self.policy == "fifo":
                rank = min(s["created"] for s in rows if s["id"] in ids)
            else:
                # Last successful matching demand; file inspections do not refresh LRU.
                rank = max([s["created"] for s in rows if s["id"] in ids] +
                           [r[0] for sid in ids for r in self.db.execute(
                               "SELECT last_used FROM successful_restores WHERE id=?", (sid,))])
            result.append(dict(ids=sorted(ids), bytes=freed, lost_seconds=lost, rank=rank, action="evict"))
            if (allow_demote and self.policy == "utility" and tier == "local" and "archive" in self.paths
                    and self.budgets["archive"] and self.paths["archive"].is_dir()):
                # Cold transfer prior; successful moves replace it with measured throughput.
                throughput = getattr(self, "archive_bytes_per_second", 256*1024**2)
                moved = [s for s in available if s["id"] in ids]
                blocks = {h:n for s in moved for h,n in s["blocks"]}
                transfer_s = sum(n for h,n in blocks.items() if h not in archive_present) / throughput
                penalty={s["id"]:sum(n for _,n in s["blocks"])/throughput for s in moved}
                demote_loss = sum(weight*max(0,min(cost+penalty.get(sid,0) for cost,sid in choices)-before)
                                  for weight,before,choices in baseline)
                if demote_loss + transfer_s < lost:
                    result.append(dict(ids=sorted(ids), bytes=freed, lost_seconds=demote_loss,
                        rank=(demote_loss+transfer_s)/freed, action="demote"))
        return sorted(result, key=lambda x: (x["rank"], -x["bytes"], x["ids"]))

    def collect_orphans(self):
        with self.lock:
            rows = self._rows()
            for tier, path in self.paths.items():
                if not path.is_dir():
                    continue
                retained = {h for r in rows if r["tier"] == tier for h, _ in r["blocks"]}
                for p in path.iterdir():
                    if p.is_file() and (p.name.endswith(".tmp") or (len(p.name) == 64 and
                            all(c in "0123456789abcdef" for c in p.name) and p.name not in retained)):
                        p.unlink()

    def enforce(self, tier="local", incoming=0):
        with self.lock:
            allow_demote = True
            while (self.used(tier)+incoming > self.budgets[tier] or
                   self.free_bytes(self.paths[tier])-incoming < self.reserve):
                candidates = self.candidates(tier, allow_demote=allow_demote)
                if not candidates:
                    raise CacheFull("checkpoint admission refused: pinned data or history reserve")
                choice = candidates[0]
                moved = False
                if choice["action"] == "demote":
                    try:
                        self.demote(choice["ids"])
                        moved = True
                    except (OSError, ValueError):
                        allow_demote = False
                        continue  # recompute deletion usefulness with unchanged source copies
                if not moved:
                    with self.db:
                        self.db.executemany("DELETE FROM snapshots WHERE id=?", [(s,) for s in choice["ids"]])
                    self.collect_orphans()
                with self.db:
                    self.db.execute("INSERT INTO decisions VALUES(?,?,?,?,?)", (self.clock(),
                        "demote" if moved else "evict", json.dumps(choice["ids"]), choice["bytes"], choice["lost_seconds"]))

    def _write(self, tier, data):
        h = hashlib.sha256(data).hexdigest()
        path = self.paths[tier] / h
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == h:
            return h, len(data)
        temporary = path.with_name(h + ".tmp")
        with temporary.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
        return h, len(data)

    def _copy_block(self, source, tier):
        with Path(source).open("rb") as f:
            h = hashlib.file_digest(f, "sha256").hexdigest()
        path = self.paths[tier]/h
        size = Path(source).stat().st_size
        if path.exists():
            with path.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() == h:
                    return h, size
        temporary = path.with_name(h + ".tmp")
        with Path(source).open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        with temporary.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != h:
                temporary.unlink()
                raise OSError("checkpoint changed during transfer")
        os.replace(temporary, path)
        sync_directory(path.parent)
        return h, size

    def admit(self, source, fingerprint, tokens, *, restore_s, pinned=False):
        if not tokens or not math.isfinite(restore_s) or restore_s < 0:
            raise ValueError("checkpoint requires tokens and a finite restore time")
        with self.lock:
            source = Path(source)
            size = source.stat().st_size
            if not self.chunk:
                with source.open("rb") as f:
                    wanted = [(hashlib.file_digest(f,"sha256").hexdigest(), size)]
            else:
                wanted = []
                with source.open("rb") as f:
                    while data := f.read(self.chunk):
                        wanted.append((hashlib.sha256(data).hexdigest(),len(data)))
            for existing in self._rows():
                if (existing["tier"] == "local" and existing["fingerprint"] == fingerprint and
                        existing["tokens"] == tokens and existing["blocks"] == [list(b) for b in wanted]
                        and self._available(existing)):
                    if pinned:
                        self.pin(existing["id"])
                    return existing["id"]
            # Reused blocks are referenced during admission so budget enforcement cannot remove them.
            needed = {h:n for h,n in wanted}
            hold = {s["id"] for s in self._rows() if s["tier"] == "local" and
                    any(h in needed for h,_ in s["blocks"])} - self.leases
            self.leases.update(hold)
            try:
                self.enforce(incoming=sum(n for h,n in needed.items() if not (self.paths["local"]/h).exists()))
            finally:
                self.leases.difference_update(hold)
            blocks = []
            if not self.chunk:
                blocks = [self._copy_block(source, "local")]
            else:
                with source.open("rb") as f:
                    while data := f.read(self.chunk):
                        blocks.append(self._write("local", data))
            if blocks != wanted:
                self.collect_orphans()
                raise OSError("checkpoint source changed during admission")
            sid = uuid.uuid4().hex
            with self.db:
                self.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?)", (sid, fingerprint,
                    json.dumps(tokens, separators=(",", ":")), json.dumps(blocks), "local", restore_s,
                    self.clock(), int(pinned)))
            return sid

    def demote(self, ids):
        with self.lock:
            rows = [s for s in self._rows() if s["id"] in ids]
            if len(rows) != len(ids) or any(s["tier"] != "local" or s["pinned"] or s["id"] in self.leases for s in rows):
                raise ValueError("infeasible checkpoint demotion")
            blocks = {h: size for s in rows for h, size in s["blocks"]}
            self.enforce("archive", sum(size for h, size in blocks.items() if not (self.paths["archive"]/h).exists()))
            start = time.perf_counter()
            for h in blocks:
                copied, _ = self._copy_block(self.paths["local"]/h, "archive")
                if copied != h:
                    raise OSError("checkpoint source checksum mismatch")
            elapsed = time.perf_counter()-start
            self.archive_bytes_per_second = max(1, sum(blocks.values())/max(elapsed,.000001))
            # Commit catalog after all durable blocks; crash beforehand leaves only collectible orphans.
            with self.db:
                self.db.executemany("UPDATE snapshots SET tier='archive',restore_s=restore_s+? WHERE id=?",
                                    [(elapsed, s["id"]) for s in rows])
            self.collect_orphans()

    @contextmanager
    def materialize(self, sid):
        with self.lock:
            row = next((s for s in self._rows() if s["id"] == sid), None)
            if row is None:
                raise FileNotFoundError(sid)
            self.leases.add(sid)
        temporary = self.root / (uuid.uuid4().hex + ".restore.tmp")
        try:
            # Assembly uses local space and must also preserve the durable-history reserve.
            with self.lock:
                self.enforce(incoming=sum(n for _, n in row["blocks"]))
            with temporary.open("wb") as f:
                for h, size in row["blocks"]:
                    hashed, count = hashlib.sha256(), 0
                    with (self.paths[row["tier"]]/h).open("rb") as block:
                        while data := block.read(4 << 20):
                            hashed.update(data)
                            count += len(data)
                            f.write(data)
                    if count != size or hashed.hexdigest() != h:
                        raise OSError("checkpoint block checksum mismatch")
            yield temporary
        finally:
            temporary.unlink(missing_ok=True)
            with self.lock:
                self.leases.discard(sid)

    def restored(self, sid, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("restore time must be finite and nonnegative")
        with self.lock, self.db:
            self.db.execute("UPDATE snapshots SET restore_s=? WHERE id=?", (seconds, sid))
            self.db.execute("INSERT OR REPLACE INTO successful_restores VALUES(?,?)", (sid,self.clock()))

    def pin(self, sid, enabled=True):
        with self.lock:
            held = sid in self.leases
            self.leases.add(sid)
            try:
                if enabled:
                    self.enforce()
                with self.db:
                    if not self.db.execute("UPDATE snapshots SET pinned=? WHERE id=?", (int(enabled), sid)).rowcount:
                        raise KeyError(sid)
            finally:
                if not held:
                    self.leases.discard(sid)
