"""Supervisor-owned admission for bounded local tools; no model/client retries.

This is a coordination protocol, not an OS reservation or a sandbox. Only the
broker sends engine controls. A worker presents its runtime lineage and the
supervisor's existing plan determines whether its declared class is admitted.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import copy_context
import ipaddress
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

LOG = logging.getLogger(__name__)
MODES = frozenset(("unload", "auto", "relieve"))
EXECUTION_FLOORS = {"execution_ram_floor_gib": "ram_headroom_gib",
                    "execution_vram_floor_mib": "vram_headroom_mib"}
# Known synchronous terminal envelope, not a general process-lifetime proof.
# Unknown backend extensions must be reviewed before they can authorize release.
TERMINAL_RESULT_FIELDS = frozenset(("output", "exit_code", "error", "cwd", "environment_recreated",
    "output_total_chars", "full_output_path", "truncation_note", "verification_evidence",
    "approval", "exit_code_meaning", "hint", "sudo_auth_failed", "sudo_cache_cleared"))


class AdmissionError(RuntimeError):
    pass


def _foreground_finished(result):
    """Recognize the trusted bounded wrapper contract, without altering its result.

    A backend may return or raise after spawning work that is still alive.
    Hermes's finalizer also drops timeout/interrupt flags, so reserved exit
    codes and markers remain uncertain even when its error field is null.
    """
    if isinstance(result, str):
        if len(result) > 2 * 1024 * 1024:
            return False
        try:
            result = json.loads(result)
        except (ValueError, RecursionError):
            return False
    if (not isinstance(result, dict) or set(result) - TERMINAL_RESULT_FIELDS
            or not {"output", "exit_code", "error"} <= set(result)
            or not isinstance(result["output"], str) or result.get("error") is not None
            or result.get("environment_recreated")):
        return False
    code = result["exit_code"]
    if type(code) is not int or not 0 <= code < 128 or code == 124:
        return False
    return not any(marker in result["output"] for marker in
                   ("[Command timed out", "[Command interrupted]"))


def _number(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not low <= value <= high or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return value


def validate_profiles(value):
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError("profiles must be a mapping with at most 32 entries")
    profiles = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not name or len(name) > 64 or not isinstance(spec, dict):
            raise ValueError("invalid resource profile")
        if set(spec) - {"command_prefixes", "ram_headroom_gib", "vram_headroom_mib", "ttl_seconds",
                        "mode", *EXECUTION_FLOORS}:
            raise ValueError("unknown resource profile option")
        mode = spec.get("mode", "unload")
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError("resource profile mode must be unload, auto or relieve")
        prefixes = spec.get("command_prefixes")
        if not isinstance(prefixes, list) or not 1 <= len(prefixes) <= 16 or any(
                not isinstance(p, str) or not p.strip() or len(p) > 1024 for p in prefixes):
            raise ValueError("each profile needs explicit command prefixes")
        profiles[name] = {
            "command_prefixes": tuple(prefixes),
            "ram_headroom_gib": _number(spec.get("ram_headroom_gib", 4), "RAM headroom", 0, 1048576),
            "vram_headroom_mib": _number(spec.get("vram_headroom_mib", 250), "VRAM headroom", 0, 1048576),
            "ttl_seconds": _number(spec.get("ttl_seconds", 60), "lease TTL", 30, 300),
            "mode": mode,
        }
        for floor, target in EXECUTION_FLOORS.items():
            if floor in spec:
                profiles[name][floor] = _number(spec[floor], floor, 0, profiles[name][target])
    return profiles


def validate_scope(scope):
    if not isinstance(scope, dict) or scope.get("lineage_valid") is not True:
        raise AdmissionError("Resource work requires verified supervisor lineage")
    for name in ("root_session_id", "session_id", "profile_home"):
        if not isinstance(scope.get(name), str) or not scope[name]:
            raise AdmissionError("Resource work is missing runtime identity")
    depth = scope.get("delegate_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise AdmissionError("Invalid delegation depth")
    if depth == 0 and scope["root_session_id"] != scope["session_id"]:
        raise AdmissionError("Supervisor identity does not match the root")
    return scope["root_session_id"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Let urllib produce an HTTPError whose response we close below. Raising
        # here skips its response cleanup. Never construct a redirected request.
        return None


class EngineClient:
    def __init__(self, base_url, token, timeout=5):
        parsed = urllib.parse.urlsplit(base_url)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
            port = parsed.port
        except ValueError:
            loopback, port = False, None
        if (parsed.scheme != "http" or not loopback or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment or port is None):
            raise ValueError("Resource control requires an explicit loopback HTTP address and port")
        if (not isinstance(token, str) or len(token) < 32 or len(token) > 4096
                or not token.isascii() or any(c.isspace() for c in token)):
            raise ValueError("Resource control credential is missing or invalid")
        self.url = base_url.rstrip("/") + "/v1/resource-lease"
        self.token, self.timeout = token, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def call(self, action, **fields):
        return self._request("POST", {"action": action, **fields})

    def capabilities(self):
        """Public capabilities are not an owner-bound admission acknowledgement."""
        return self._request("GET")

    def _request(self, method, payload=None):
        request = urllib.request.Request(self.url, method=method,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                data = response.read(65537)
                if len(data) > 65536:
                    raise AdmissionError("Resource control response is too large")
                result = json.loads(data)
        except urllib.error.HTTPError as exc:
            # Do not copy a server body, URL, headers or token into model output.
            code = exc.code
            exc.close()
            raise AdmissionError(f"Resource control refused the request (HTTP {code})") from None
        except (OSError, ValueError) as exc:
            raise AdmissionError("Resource control is unavailable or returned invalid data") from None
        if not isinstance(result, dict) or result.get("schema") != "strata.resource-lease.v1":
            raise AdmissionError("Unsupported resource control protocol")
        return result


class SupervisorBroker:
    """One FIFO and one engine lease per profile/endpoint, shared by all workers."""
    def __init__(self, client_factory, profiles, *, default_profiles=(), wait_seconds=180, clock=time.monotonic):
        self.client_factory = client_factory
        self.profiles = validate_profiles(profiles)
        if not isinstance(default_profiles, (list, tuple)) or any(p not in self.profiles for p in default_profiles):
            raise ValueError("default_profiles must name configured profiles")
        self.defaults = frozenset(default_profiles)
        self.wait_seconds = _number(wait_seconds, "admission timeout", 1, 3600)
        self.clock = clock
        self.cv = threading.Condition()
        self.queue = deque()
        self.active = None
        self.plans = {}

    def plan(self, scope, profiles):
        root = validate_scope(scope)
        if scope["delegate_depth"] != 0:
            raise AdmissionError("Workers must return resource needs to their supervisor; they cannot change its plan")
        if not isinstance(profiles, list) or any(p not in self.profiles for p in profiles):
            raise AdmissionError("Select only operator-configured resource profiles")
        with self.cv:
            if self.active and self.active[1] == root:
                raise AdmissionError("Cannot replace the plan while its tool owns resources")
            if root not in self.plans and len(self.plans) >= 4096:
                raise AdmissionError("Resource plan capacity reached; restart this plugin between sessions")
            self.plans[root] = frozenset(profiles)
        return {"profiles": profiles, "scope": "this supervisor and its descendants", "ready": False}

    def match(self, command):
        matches = [name for name, spec in self.profiles.items()
                   if any(command.startswith(prefix) for prefix in spec["command_prefixes"])]
        if len(matches) > 1:
            raise AdmissionError("Command matches multiple resource profiles; narrow the configured prefixes")
        return matches[0] if matches else None

    @staticmethod
    def _require_capabilities(response, mode):
        if (not isinstance(response, dict) or response.get("schema") != "strata.resource-lease.v1"
                or response.get("enabled") is not True or response.get("supports_start") is not True
                or not isinstance(response.get("supported_modes"), list)
                or mode not in response["supported_modes"]):
            raise AdmissionError("Engine does not support the requested resource mode and start handshake")

    @staticmethod
    def _owned_response(response, lease_id, spec, *, phases=("admission", "execution")):
        """Reject downgraded, stale-owner or ambiguous extended acknowledgements."""
        SupervisorBroker._require_capabilities(response, spec["mode"])
        expires = response.get("expires_in_seconds")
        if (not isinstance(lease_id, str) or not lease_id or len(lease_id) > 128
                or response.get("lease_id") != lease_id or response.get("mode") != spec["mode"]
                or response.get("phase") not in phases or isinstance(expires, bool)
                or not isinstance(expires, (int, float)) or not 0 < expires <= 300
                or not math.isfinite(expires)):
            raise AdmissionError("Engine returned an invalid or expired owned resource acknowledgement")
        for field in ("ram_headroom_gib", "vram_headroom_mib", *EXECUTION_FLOORS):
            expected = spec[field] if field in spec else spec[EXECUTION_FLOORS[field]]
            value = response.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                raise AdmissionError("Engine did not acknowledge the requested resource targets")
        action = response.get("selected_action")
        allowed = {"unload": ("unload",), "auto": ("none", "relieve", "unload"),
                   "relieve": ("none", "relieve")}[spec["mode"]]
        if action is not None and action not in allowed:
            raise AdmissionError("Engine selected an action outside the requested resource mode")
        if response.get("state") == "ready" and action not in allowed:
            raise AdmissionError("Engine readiness is missing a supported selected action")

    def _check_pending(self, cancelled, deadline, renewal_failed):
        if cancelled() or self.clock() >= deadline or renewal_failed.is_set():
            raise AdmissionError("Resource admission cancelled, expired, or timed out; command was not started")

    @contextmanager
    def _turn(self, root, profile, cancelled):
        ticket = (uuid.uuid4().hex, root)
        deadline = self.clock() + self.wait_seconds
        with self.cv:
            self.queue.append(ticket)
            try:
                while self.active is not None or self.queue[0] != ticket:
                    if cancelled() or self.clock() >= deadline:
                        raise AdmissionError("Resource admission cancelled or timed out before execution")
                    self.cv.wait(.1)
                if cancelled() or self.clock() >= deadline:
                    raise AdmissionError("Resource admission cancelled or timed out before execution")
                if profile not in self.plans.get(root, self.defaults):
                    raise AdmissionError("Return the resource need to the supervisor; its plan has not granted this profile")
                self.queue.popleft()
                self.active = ticket
            except BaseException:
                if ticket in self.queue:
                    self.queue.remove(ticket)
                self.cv.notify_all()
                raise
        try:
            yield deadline
        finally:
            with self.cv:
                if self.active == ticket:
                    self.active = None
                self.cv.notify_all()

    def run(self, scope, command, next_call, *, background=False, env_type="local", cancelled=lambda: False):
        profile = self.match(command)
        if profile is None:
            return next_call()
        root = validate_scope(scope)
        if background or env_type != "local":
            raise AdmissionError("Resource-scoped work requires a bounded local foreground command")
        with self._turn(root, profile, cancelled) as deadline:
            spec = self.profiles[profile]
            client = self.client_factory()
            if cancelled() or self.clock() >= deadline:
                raise AdmissionError("Resource admission cancelled or timed out before acquisition")
            request_id = str(uuid.uuid4())
            lease_token = None
            stop = threading.Event()
            renewal_failed = threading.Event()
            renewer = None
            dispatched = False
            completed = False
            try:
                extended = spec["mode"] != "unload" or any(field in spec for field in EXECUTION_FLOORS)
                if extended:
                    self._require_capabilities(client.capabilities(), spec["mode"])
                    self._check_pending(cancelled, deadline, renewal_failed)
                response = client.call("acquire", request_id=request_id, mode=spec["mode"],
                    ram_headroom_gib=spec["ram_headroom_gib"], vram_headroom_mib=spec["vram_headroom_mib"],
                    ttl_seconds=spec["ttl_seconds"],
                    **{field: spec[field] for field in EXECUTION_FLOORS if field in spec})
                lease_token = response.get("lease_token")
                if (not isinstance(lease_token, str) or not 24 <= len(lease_token) <= 256
                        or not lease_token.isascii()):
                    lease_token = None
                    raise AdmissionError("Engine did not return an owned lease")
                # New servers also phase-gate legacy unload profiles. Old servers
                # remain compatible only with the unchanged unload-only request.
                phased = extended or response.get("supports_start") is True
                lease_id = response.get("lease_id")
                if phased:
                    self._owned_response(response, lease_id, spec, phases=("admission",))
                elif response.get("supports_start") not in (None, False):
                    raise AdmissionError("Engine returned an invalid start capability")
                waiting_states = ("pending", "relieving", "unloading") if phased else ("pending", "unloading")
                # Keep the pending lease alive as well as the executing one.
                def renew():
                    while not stop.wait(min(10, spec["ttl_seconds"] / 3)):
                        try:
                            result = client.call("renew", lease_token=lease_token, ttl_seconds=spec["ttl_seconds"])
                            if phased:
                                self._owned_response(result, lease_id, spec)
                            if result.get("state") not in (*waiting_states, "ready"):
                                raise AdmissionError("Resource lease is no longer active")
                        except Exception:
                            renewal_failed.set()
                            LOG.error("Resource lease renewal failed; pending admission will not execute; "
                                      "an already running command is not cancelled")
                            return
                renewer = threading.Thread(target=copy_context().run, args=(renew,), daemon=True,
                                           name="hermes-resource-lease")
                renewer.start()
                while True:
                    self._check_pending(cancelled, deadline, renewal_failed)
                    response = client.call("status", lease_token=lease_token)
                    if phased:
                        self._owned_response(response, lease_id, spec, phases=("admission",))
                    state = response.get("state")
                    if state == "ready":
                        break
                    if state not in waiting_states:
                        raise AdmissionError("Engine could not grant the requested resource handoff")
                    stop.wait(.1)
                self._check_pending(cancelled, deadline, renewal_failed)
                if phased:
                    # Idempotent on the server, but never retry an uncertain reply:
                    # the continuation remains uncalled and finally releases our owner.
                    response = client.call("start", lease_token=lease_token)
                    self._owned_response(response, lease_id, spec, phases=("execution",))
                    if response.get("state") != "ready":
                        raise AdmissionError("Engine did not acknowledge execution readiness")
                    self._check_pending(cancelled, deadline, renewal_failed)
                # Exactly one continuation; no retry of a compiler, renderer or shell.
                dispatched = True
                result = next_call()
                completed = spec["mode"] == "unload" or _foreground_finished(result)
                return result
            finally:
                stop.set()
                if renewer is not None:
                    renewer.join(timeout=client.timeout + 1)
                retain = spec["mode"] in ("auto", "relieve") and dispatched and not completed
                if retain:
                    LOG.warning("Resource command completion is unconfirmed; retaining the execution barrier. "
                                "Verify owned-process exit before operator recovery; the command was not retried")
                elif lease_token:
                    try:
                        client.call("release", lease_token=lease_token)
                    except Exception:
                        if spec["mode"] in ("auto", "relieve"):
                            LOG.warning("Resource lease release was not acknowledged; "
                                        "the execution barrier may require operator recovery")
                        else:
                            LOG.warning("Resource lease release was not acknowledged; engine TTL remains authoritative")
