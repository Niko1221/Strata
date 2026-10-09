"""Opt-in, supervisor-owned resource handoff. No allocation or lifecycle code.

The service owns admission and unloading; this bounded state machine only
records authenticated intentions. READY is a fresh observation, not an OS
reservation. Tokens and request identifiers are never part of public status.
"""
import collections
import hmac
import math
import os
import re
import secrets
import threading
import time
import uuid

SCHEMA = "strata.resource-lease.v1"
TERMINAL = {"released", "expired", "failed"}


class LeaseError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ToolLeases:
    def __init__(self, config=None, *, clock=time.monotonic):
        config = {} if config is None else config
        if not isinstance(config, dict) or set(config) - {"enabled", "token_env", "max_ttl_seconds"}:
            raise ValueError("tool_leases accepts enabled, token_env and max_ttl_seconds only")
        self.enabled = config.get("enabled", False)
        self.max_ttl = config.get("max_ttl_seconds", 300)
        env = config.get("token_env", "STRATA_RESOURCE_LEASE_TOKEN")
        if (not isinstance(self.enabled, bool) or not isinstance(env, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", env)
                or isinstance(self.max_ttl, bool) or not isinstance(self.max_ttl, int)
                or not 30 <= self.max_ttl <= 300):
            raise ValueError("invalid tool_leases configuration")
        self.secret = os.environ.get(env, "") if self.enabled else ""
        if self.enabled and (len(self.secret) < 32 or not self.secret.isascii() or any(c.isspace() for c in self.secret)):
            raise ValueError("tool_leases token_env must name an ASCII control token of at least 32 characters")
        self.clock, self.lock = clock, threading.RLock()
        self.current = None
        self.history = collections.OrderedDict()  # bounded retry/idempotency window, in this server lifetime only

    def authenticated(self, token):
        return (self.enabled and isinstance(token, str) and token.isascii() and len(token) <= 4096
                and hmac.compare_digest(token, self.secret))

    def _expire(self):
        row = self.current
        if row and row["state"] not in TERMINAL and self.clock() >= row["deadline"]:
            if row["state"] == "unloading":
                row["terminal_after_unload"] = row.get("terminal_after_unload") or "expired"
            else:
                row.update(state="expired", reason="lease_expired")
        return row

    def blocked(self):
        with self.lock:
            row = self._expire()
            return bool(row and row["state"] not in TERMINAL)

    def _view(self, row, *, acquire=False):
        out = {"schema": SCHEMA, "enabled": self.enabled, "supported_modes": ["unload"],
               "state": row["state"] if row else "released"}
        if row:
            out.update(lease_id=row["lease_id"], reason=row["reason"],
                       expires_in_seconds=max(0., round(row["deadline"] - self.clock(), 3)),
                       mode="unload", ram_headroom_gib=row["ram_headroom_gib"],
                       vram_headroom_mib=row["vram_headroom_mib"],
                       availability_is_reservation=False)
            if acquire:
                out["lease_token"] = row["token"]
        return out

    def public(self):
        with self.lock:
            row = self._expire()
            return {"schema": SCHEMA, "enabled": self.enabled, "supported_modes": ["unload"],
                    "state": row["state"] if row else "released", "availability_is_reservation": False}

    def acquire(self, request, capacity, wall_now):
        allowed = {"action", "request_id", "mode", "ram_headroom_gib", "vram_headroom_mib", "ttl_seconds"}
        if not isinstance(request, dict) or set(request) != allowed or request.get("action") != "acquire":
            raise LeaseError("acquire requires request_id, mode, ram_headroom_gib, vram_headroom_mib and ttl_seconds")
        try:
            rid = str(uuid.UUID(request["request_id"]))
        except (TypeError, ValueError, AttributeError):
            raise LeaseError("request_id must be a UUID") from None
        ram, gpu, ttl = (request[k] for k in ("ram_headroom_gib", "vram_headroom_mib", "ttl_seconds"))
        if (request["mode"] != "unload" or not number(ram) or ram < 0 or not number(gpu) or gpu < 0
                or not number(ttl) or not 30 <= ttl <= self.max_ttl):
            raise LeaseError("invalid unload headroom or ttl_seconds (30 through configured maximum)")
        fingerprint = ("unload", ram, gpu, ttl)
        with self.lock:
            self._expire()
            old = self.history.get(rid)
            if old:
                if old["fingerprint"] != fingerprint:
                    raise LeaseError("request_id already used with different parameters", 409)
                return self._view(old, acquire=True)
            if self.current and self.current["state"] not in TERMINAL:
                raise LeaseError("another supervisor lease is active", 409)
            stamp = capacity.get("sampled_at")
            totals = (capacity.get("ram_total"), capacity.get("gpu_mem_total"))
            if (not number(stamp) or not 0 <= wall_now - stamp <= 5
                    or any(not number(v) or v <= 0 for v in totals)):
                raise LeaseError("fresh hardware capacity is unavailable", 503)
            if ram * 2**30 > totals[0] or gpu * 2**20 > totals[1]:
                raise LeaseError("requested free headroom exceeds hardware capacity")
            row = {"lease_id": uuid.uuid4().hex, "token": secrets.token_urlsafe(32), "request_id": rid,
                   "fingerprint": fingerprint, "ram_headroom_gib": ram, "vram_headroom_mib": gpu,
                   "deadline": self.clock() + ttl, "state": "pending", "reason": "draining_requests",
                   "unloaded_at": None, "terminal_after_unload": None}
            self.current = row
            self.history[rid] = row
            while len(self.history) > 128:
                self.history.popitem(last=False)
            return self._view(row, acquire=True)

    def action(self, request):
        action = request.get("action") if isinstance(request, dict) else None
        keys = {"action", "lease_token", "ttl_seconds"} if action == "renew" else {"action", "lease_token"}
        if action not in {"status", "renew", "release"} or set(request) != keys:
            raise LeaseError("expected status, renew or release and the lease_token")
        token = request["lease_token"]
        if not isinstance(token, str) or not token.isascii() or len(token) > 256:
            raise LeaseError("invalid lease token", 403)
        with self.lock:
            self._expire()
            row = next((r for r in reversed(self.history.values()) if hmac.compare_digest(r["token"], token)), None)
            if row is None:
                raise LeaseError("invalid lease token", 403)
            if action == "renew":
                ttl = request["ttl_seconds"]
                if not number(ttl) or not 30 <= ttl <= self.max_ttl:
                    raise LeaseError("invalid ttl_seconds")
                if row is not self.current or row["state"] in TERMINAL or row["terminal_after_unload"]:
                    raise LeaseError("a terminal lease cannot be renewed", 409)
                row["deadline"] = self.clock() + ttl
            elif action == "release" and row["state"] not in TERMINAL:
                if row["state"] == "unloading":
                    row["terminal_after_unload"] = "released"
                else:
                    row.update(state="released", reason="owner_released")
            return self._view(row)

    def begin_unload(self):
        with self.lock:
            row = self._expire()
            if not row or row["state"] not in {"pending", "ready"} or row["unloaded_at"] is not None:
                return None
            row.update(state="unloading", reason="releasing_model_memory")
            return row["lease_id"]

    def needs_unload(self):
        with self.lock:
            row = self._expire()
            return bool(row and row["state"] in {"pending", "ready"} and row["unloaded_at"] is None)

    def unloaded(self, lease_id, wall_now):
        with self.lock:
            row = self._expire()
            if row is None or row["lease_id"] != lease_id:
                raise RuntimeError("lease lifecycle identity changed")
            terminal = row["terminal_after_unload"]
            row.update(unloaded_at=wall_now, state=terminal or "pending",
                       reason="lease_" + terminal if terminal else "checking_headroom")

    def fail(self, reason):
        with self.lock:
            row = self._expire()
            if row and row["state"] not in TERMINAL:
                row.update(state="failed", reason=reason)

    def observe_ready(self, capacity, now, *, unloaded, ram_floor, gpu_floor, commit_floor):
        with self.lock:
            row = self._expire()
            if not row or row["state"] in TERMINAL or row["state"] == "unloading":
                return
            stamp = capacity.get("sampled_at")
            rt, ru = capacity.get("ram_total"), capacity.get("ram_used")
            gt, gu = capacity.get("gpu_mem_total"), capacity.get("gpu_mem_used")
            commit = capacity.get("ram_commit_available")
            commit_required = capacity.get("ram_commit_required", os.name == "nt")
            commit_ok = (number(commit) and commit >= commit_floor * 2**30) if commit_required else True
            valid = (unloaded and row["unloaded_at"] is not None and number(stamp)
                     and row["unloaded_at"] <= stamp <= now and now - stamp <= 5
                     and all(number(v) for v in (rt, ru, gt, gu))
                     and rt > 0 and 0 <= ru <= rt and gt > 0 and 0 <= gu <= gt)
            ready = (valid and rt - ru >= max(ram_floor, row["ram_headroom_gib"]) * 2**30
                     and gt - gu >= max(gpu_floor, row["vram_headroom_mib"]) * 2**20
                     and commit_ok)
            row.update(state="ready" if ready else "pending",
                       reason="observed_headroom" if ready else "waiting_for_fresh_headroom")
