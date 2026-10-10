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
MODES = ("unload", "auto", "relieve")
EXECUTION_FIELDS = ("execution_ram_floor_gib", "execution_vram_floor_mib")


class LeaseError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


class ToolLeases:
    def __init__(self, config=None, *, clock=time.monotonic):
        config = {} if config is None else config
        allowed = {"enabled", "token_env", "max_ttl_seconds", "resume_ram_working_gib",
                   "resume_vram_working_mib", "resume_timeout_seconds"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("unknown tool_leases configuration")
        self.enabled = config.get("enabled", False)
        self.max_ttl = config.get("max_ttl_seconds", 300)
        env = config.get("token_env", "STRATA_RESOURCE_LEASE_TOKEN")
        if (not isinstance(self.enabled, bool) or not isinstance(env, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", env)
                or isinstance(self.max_ttl, bool) or not isinstance(self.max_ttl, int)
                or not 30 <= self.max_ttl <= 300):
            raise ValueError("invalid tool_leases configuration")
        self.resume_ram_working_gib = config.get("resume_ram_working_gib", 2)
        self.resume_vram_working_mib = config.get("resume_vram_working_mib", 256)
        self.resume_timeout_seconds = config.get("resume_timeout_seconds", 30)
        for value, low, high in ((self.resume_ram_working_gib, 0, 128),
                                 (self.resume_vram_working_mib, 0, 65536),
                                 (self.resume_timeout_seconds, 1, 300)):
            if not number(value) or not low <= value <= high:
                raise ValueError("invalid tool_leases inference-resume allowance")
        self.secret = os.environ.get(env, "") if self.enabled else ""
        if self.enabled and (len(self.secret) < 32 or not self.secret.isascii() or any(c.isspace() for c in self.secret)):
            raise ValueError("tool_leases token_env must name an ASCII control token of at least 32 characters")
        self.clock, self.lock = clock, threading.RLock()
        self.current = None
        # An expired/released owner is not proof that its native operation ended.
        # Only an exact service-side acknowledgement/drain clears this barrier.
        self.operation = None
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
            return bool(self.operation or (row and (row["state"] not in TERMINAL or row["execution_hold"])))

    def _view(self, row, *, acquire=False):
        out = {"schema": SCHEMA, "enabled": self.enabled, "supported_modes": list(MODES),
               "supports_start": True,
               "state": row["state"] if row else "released"}
        if row:
            out.update(lease_id=row["lease_id"], reason=row["reason"],
                       expires_in_seconds=max(0., round(row["deadline"] - self.clock(), 3)),
                       mode=row["mode"], phase=row["phase"], selected_action=row["selected_action"],
                       ram_headroom_gib=row["ram_headroom_gib"],
                       vram_headroom_mib=row["vram_headroom_mib"],
                       execution_ram_floor_gib=row["execution_ram_floor_gib"],
                       execution_vram_floor_mib=row["execution_vram_floor_mib"],
                       execution_hold=row["execution_hold"],
                       availability_is_reservation=False)
            if acquire:
                out["lease_token"] = row["token"]
        return out

    def public(self):
        with self.lock:
            row = self._expire()
            return {"schema": SCHEMA, "enabled": self.enabled, "supported_modes": list(MODES),
                    "supports_start": True, "state": row["state"] if row else "released",
                    "operation_pending": self.operation is not None,
                    "execution_hold": bool(row and row["execution_hold"]),
                    "barrier_reason": ("native_operation_unresolved" if self.operation else
                                       "tool_exit_unconfirmed" if row and row["execution_hold"]
                                       and row["state"] in TERMINAL else None),
                    "availability_is_reservation": False}

    def acquire(self, request, capacity, wall_now):
        allowed = {"action", "request_id", "mode", "ram_headroom_gib", "vram_headroom_mib", "ttl_seconds"}
        if (not isinstance(request, dict) or not allowed <= set(request)
                or set(request) - allowed - set(EXECUTION_FIELDS) or request.get("action") != "acquire"):
            raise LeaseError("acquire requires request_id, mode, ram_headroom_gib, vram_headroom_mib and ttl_seconds")
        try:
            rid = str(uuid.UUID(request["request_id"]))
        except (TypeError, ValueError, AttributeError):
            raise LeaseError("request_id must be a UUID") from None
        ram, gpu, ttl = (request[k] for k in ("ram_headroom_gib", "vram_headroom_mib", "ttl_seconds"))
        if (request["mode"] not in MODES or not number(ram) or ram < 0 or not number(gpu) or gpu < 0
                or not number(ttl) or not 30 <= ttl <= self.max_ttl):
            raise LeaseError("invalid mode, headroom or ttl_seconds (30 through configured maximum)")
        execution_ram = request.get("execution_ram_floor_gib", ram)
        execution_gpu = request.get("execution_vram_floor_mib", gpu)
        if (not number(execution_ram) or not 0 <= execution_ram <= ram
                or not number(execution_gpu) or not 0 <= execution_gpu <= gpu):
            raise LeaseError("execution floors must be finite, non-negative and no larger than admission targets")
        fingerprint = (request["mode"], ram, gpu, ttl, execution_ram, execution_gpu)
        with self.lock:
            self._expire()
            old = self.history.get(rid)
            if old:
                if old["fingerprint"] != fingerprint:
                    raise LeaseError("request_id already used with different parameters", 409)
                return self._view(old, acquire=True)
            if self.operation or (self.current and (self.current["state"] not in TERMINAL
                                                    or self.current["execution_hold"])):
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
                   "mode": request["mode"], "phase": "admission",
                   "execution_ram_floor_gib": execution_ram, "execution_vram_floor_mib": execution_gpu,
                   "selected_action": "unload" if request["mode"] == "unload" else None,
                   "residency": None, "ready_until": None, "execution_hold": False,
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
        if action not in {"status", "renew", "release", "start"} or set(request) != keys:
            raise LeaseError("expected status, renew, release or start and the lease_token")
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
            elif action == "start":
                if row is not self.current or row["state"] in TERMINAL or row["terminal_after_unload"]:
                    raise LeaseError("a terminal lease cannot start", 409)
                if row["phase"] != "execution":
                    if (row["state"] != "ready" or self.operation or row["ready_until"] is None
                            or self.clock() > row["ready_until"]):
                        raise LeaseError("start requires fresh owner readiness", 409)
                    row["phase"] = "execution"
                    # Extended modes retain their barrier after TTL expiry: the
                    # broker must confirm bounded tool cleanup with release.
                    row["execution_hold"] = row["mode"] != "unload"
            elif action == "release":
                row["execution_hold"] = False
                if row["state"] not in TERMINAL:
                    if row["state"] == "unloading":
                        row["terminal_after_unload"] = "released"
                    else:
                        row.update(state="released", reason="owner_released")
            return self._view(row)

    def demand(self):
        """Private service snapshot; no credential or mutable row escapes the lock.

        Terminal demand remains visible while a native operation is unresolved.
        The service must reconcile that identity rather than drop its ceiling.
        """
        with self.lock:
            row = self._expire()
            if row is None:
                return None
            out = {key: value for key, value in row.items()
                   if key not in {"token", "request_id", "fingerprint", "residency"}}
            out["residency"] = dict(row["residency"]) if row["residency"] else None
            out["operation"] = dict(self.operation) if self.operation else None
            out["terminal"] = row["state"] in TERMINAL or row["terminal_after_unload"] is not None
            out["expires_in_seconds"] = max(0., row["deadline"] - self.clock())
            execution = row["phase"] == "execution"
            out["ram_target_gib"] = row["execution_ram_floor_gib"] if execution else row["ram_headroom_gib"]
            out["vram_target_mib"] = row["execution_vram_floor_mib"] if execution else row["vram_headroom_mib"]
            return out

    def _active(self, lease_id):
        row = self._expire()
        return (row if row and row["lease_id"] == lease_id and row["state"] not in TERMINAL
                and not row["terminal_after_unload"] else None)

    def select_action(self, lease_id, action, reason):
        if action not in {"none", "relieve", "unload"}:
            raise ValueError("unknown lease action")
        with self.lock:
            row = self._active(lease_id)
            if not row:
                return False
            if ((row["mode"] == "unload" and action != "unload")
                    or (row["mode"] == "relieve" and action == "unload")):
                raise LeaseError("selected action is forbidden by the lease mode", 409)
            # Auto may escalate an unresolved resize to verified full teardown.
            # The old barrier remains until begin_unload atomically supersedes it.
            if self.operation and not (row["mode"] == "auto" and action == "unload"
                                       and self.operation["lease_id"] == lease_id
                                       and self.operation["action"] in {"drain", "relieve"}):
                return False
            row.update(selected_action=action, state="pending", reason=reason, ready_until=None)
            return True

    def seal_residency(self, lease_id, engine_identity, ram_cap_mib, gpu_cap_mib):
        """Freeze acknowledged maxima; repeated seals can only tighten them."""
        if (isinstance(engine_identity, bool) or not isinstance(engine_identity, int) or engine_identity <= 0
                or not number(ram_cap_mib) or ram_cap_mib < 0
                or not number(gpu_cap_mib) or gpu_cap_mib < 0):
            raise ValueError("invalid residency identity or ceiling")
        with self.lock:
            row = self._active(lease_id)
            if not row:
                return False
            previous = row["residency"]
            if previous and previous["engine_identity"] != engine_identity:
                return False
            row["residency"] = {"engine_identity": engine_identity,
                                "ram_cap_mib": min(previous["ram_cap_mib"], ram_cap_mib) if previous else ram_cap_mib,
                                "gpu_cap_mib": min(previous["gpu_cap_mib"], gpu_cap_mib) if previous else gpu_cap_mib}
            return True

    def begin_operation(self, lease_id, action, operation_id, engine_identity):
        """Register native work before issuing it; timeout is not completion."""
        if action not in {"drain", "relieve", "unload"} or operation_id is None:
            raise ValueError("invalid native operation")
        try:
            hash(operation_id)
        except TypeError:
            raise ValueError("native operation identity must be immutable") from None
        if (isinstance(engine_identity, bool) or not isinstance(engine_identity, int)
                or engine_identity <= 0):
            raise ValueError("invalid native engine identity")
        with self.lock:
            row = self._active(lease_id)
            if not row:
                return False
            operation = {"lease_id": lease_id, "action": action, "operation_id": operation_id,
                         "engine_identity": engine_identity}
            if self.operation:
                return self.operation == operation
            if ((action == "unload" and row["mode"] == "relieve")
                    or (action == "relieve" and row["mode"] == "unload")):
                raise LeaseError("native operation is forbidden by the lease mode", 409)
            self.operation = operation
            row.update(state={"drain": "pending", "relieve": "relieving", "unload": "unloading"}[action],
                       reason="reconciling_" + action, ready_until=None)
            if action != "drain":
                row["selected_action"] = action
            return True

    def complete_operation(self, lease_id, operation_id, engine_identity):
        """Only the matching confirmed ACK, drain or teardown clears a barrier."""
        with self.lock:
            row = self._expire()
            operation = self.operation
            if (not operation or operation["lease_id"] != lease_id
                    or operation["operation_id"] != operation_id
                    or operation["engine_identity"] != engine_identity):
                return False
            self.operation = None
            if row and row["lease_id"] == lease_id and row["state"] not in TERMINAL:
                terminal = row["terminal_after_unload"]
                row.update(state=terminal or "pending", reason="lease_" + terminal if terminal else "checking_headroom")
            return True

    def begin_unload(self):
        """Called only by the service's exclusive lifecycle/FIFO owner.

        Verified process death can supersede a pending MEMORY operation; an ACK
        for that old operation cannot clear the new teardown barrier.
        """
        with self.lock:
            row = self._expire()
            if (not row or row["state"] not in {"pending", "ready"} or row["unloaded_at"] is not None
                    or row["selected_action"] != "unload" or row["mode"] == "relieve"
                    or not self._unload_can_supersede(row)):
                return None
            self.operation = {"lease_id": row["lease_id"], "action": "unload",
                              "operation_id": "unload:" + row["lease_id"], "engine_identity": None}
            row.update(state="unloading", reason="releasing_model_memory", ready_until=None)
            return row["lease_id"]

    def needs_unload(self):
        with self.lock:
            row = self._expire()
            return bool(row and row["state"] in {"pending", "ready"} and row["unloaded_at"] is None
                        and self._unload_can_supersede(row) and row["selected_action"] == "unload"
                        and row["mode"] != "relieve")

    def _unload_can_supersede(self, row):
        return (self.operation is None or (row["mode"] == "auto"
                and self.operation["lease_id"] == row["lease_id"]
                and self.operation["action"] in {"drain", "relieve"}))

    def unloaded(self, lease_id, wall_now):
        with self.lock:
            row = self._expire()
            operation = self.operation
            if (row is None or row["lease_id"] != lease_id or not operation
                    or operation["lease_id"] != lease_id or operation["action"] != "unload"):
                raise RuntimeError("lease lifecycle identity changed")
            self.operation = None
            terminal = row["terminal_after_unload"]
            # A teardown that finished after failure does not resurrect it.
            row["unloaded_at"] = wall_now
            if row["state"] not in TERMINAL:
                row.update(state=terminal or "pending", reason="lease_" + terminal if terminal else "checking_headroom")

    def fail(self, reason):
        with self.lock:
            row = self._expire()
            if row and row["state"] not in TERMINAL:
                row.update(state="failed", reason=reason)

    def observe_ready(self, capacity, now, *, unloaded, ram_floor, gpu_floor, commit_floor):
        with self.lock:
            row = self._expire()
            if not row or row["state"] in TERMINAL or self.operation or row["terminal_after_unload"]:
                return
            stamp = capacity.get("sampled_at")
            valid = (unloaded and row["unloaded_at"] is not None and number(stamp)
                     and row["unloaded_at"] <= stamp and row["selected_action"] == "unload")
            self._record_ready(row, valid and self._headroom(row, capacity, now, ram_floor, gpu_floor, commit_floor),
                               now, stamp)

    @staticmethod
    def _headroom(row, capacity, now, ram_floor, gpu_floor, commit_floor, *, native=False):
        stamp = capacity.get("sampled_at")
        rt, ru = capacity.get("ram_total"), capacity.get("ram_used")
        gt, gu = capacity.get("gpu_mem_total"), capacity.get("gpu_mem_used")
        if (not number(now) or not number(stamp) or not 0 <= now - stamp <= 5
                or not all(number(v) and v >= 0 for v in (ram_floor, gpu_floor, commit_floor))
                or not all(number(v) for v in (rt, ru)) or not rt > 0 or not 0 <= ru <= rt):
            return False
        if native:
            # The service supplies the current process's conservative native
            # CUDA/DXGI overlay, never a permissive NVML fallback.
            free_gpu = capacity.get("native_free_mib")
            if not number(free_gpu) or free_gpu < 0:
                return False
        else:
            if not all(number(v) for v in (gt, gu)) or not gt > 0 or not 0 <= gu <= gt:
                return False
            free_gpu = (gt - gu) / 2**20
        execution = row["phase"] == "execution"
        ram_target = row["execution_ram_floor_gib"] if execution else row["ram_headroom_gib"]
        gpu_target = row["execution_vram_floor_mib"] if execution else row["vram_headroom_mib"]
        commit = capacity.get("ram_commit_available")
        required = capacity.get("ram_commit_required", os.name == "nt")
        if (not isinstance(required, bool) or (required and (not number(commit)
                or commit < max(commit_floor, ram_target) * 2**30))):
            return False
        return rt - ru >= max(ram_floor, ram_target) * 2**30 and free_gpu >= max(gpu_floor, gpu_target)

    def _record_ready(self, row, ready, now, stamp):
        row.update(state="ready" if ready else "pending",
                   reason="observed_headroom" if ready else "waiting_for_fresh_headroom",
                   ready_until=self.clock() + max(0., 5 - (now - stamp)) if ready else None)

    def observe_resident_ready(self, capacity, now, *, lease_id, engine_identity, drained_at,
                               ram_floor, gpu_floor, commit_floor):
        """Record availability only after service-owned drain/reconciliation.

        The service validates the native process and overlays its timestamp/free
        memory. Passing an arbitrary global GPU reading is never sufficient.
        """
        with self.lock:
            row = self._active(lease_id)
            if not row or self.operation:
                return
            resident = row["residency"]
            stamp, native_stamp = capacity.get("sampled_at"), capacity.get("native_sampled_at")
            valid = (row["mode"] != "unload" and row["selected_action"] in {"none", "relieve"}
                     and resident and resident["engine_identity"] == engine_identity
                     and all(number(v) for v in (now, drained_at, stamp, native_stamp))
                     and drained_at <= stamp <= now and drained_at <= native_stamp <= now
                     and now - native_stamp <= 5)
            ready = valid and self._headroom(row, capacity, now, ram_floor, gpu_floor, commit_floor, native=True)
            # The older of the two required readings determines dispatch freshness.
            oldest = min(stamp, native_stamp) if valid else None
            self._record_ready(row, bool(ready), now, oldest)
