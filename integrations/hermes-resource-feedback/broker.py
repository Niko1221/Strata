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


class AdmissionError(RuntimeError):
    pass


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return value


def validate_profiles(value):
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError("profiles must be a mapping with at most 32 entries")
    profiles = {}
    for name, spec in value.items():
        if not isinstance(name, str) or not name or len(name) > 64 or not isinstance(spec, dict):
            raise ValueError("invalid resource profile")
        if set(spec) - {"command_prefixes", "ram_headroom_gib", "vram_headroom_mib", "ttl_seconds"}:
            raise ValueError("unknown resource profile option")
        prefixes = spec.get("command_prefixes")
        if not isinstance(prefixes, list) or not 1 <= len(prefixes) <= 16 or any(
                not isinstance(p, str) or not p.strip() or len(p) > 1024 for p in prefixes):
            raise ValueError("each profile needs explicit command prefixes")
        profiles[name] = {
            "command_prefixes": tuple(prefixes),
            "ram_headroom_gib": _number(spec.get("ram_headroom_gib", 4), "RAM headroom", 0, 1048576),
            "vram_headroom_mib": _number(spec.get("vram_headroom_mib", 250), "VRAM headroom", 0, 1048576),
            "ttl_seconds": _number(spec.get("ttl_seconds", 60), "lease TTL", 30, 300),
        }
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
        raise AdmissionError("Resource control redirects are not allowed")


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
        request = urllib.request.Request(self.url, method="POST",
            data=json.dumps({"action": action, **fields}, allow_nan=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                data = response.read(65537)
                if len(data) > 65536:
                    raise AdmissionError("Resource control response is too large")
                result = json.loads(data)
        except urllib.error.HTTPError as exc:
            # Do not copy a server body, URL, headers or token into model output.
            raise AdmissionError(f"Resource control refused the request (HTTP {exc.code})") from None
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
            try:
                response = client.call("acquire", request_id=request_id, mode="unload",
                    ram_headroom_gib=spec["ram_headroom_gib"], vram_headroom_mib=spec["vram_headroom_mib"],
                    ttl_seconds=spec["ttl_seconds"])
                lease_token = response.get("lease_token")
                if not isinstance(lease_token, str) or len(lease_token) < 24:
                    raise AdmissionError("Engine did not return an owned lease")
                # Keep the pending lease alive as well as the executing one.
                def renew():
                    while not stop.wait(min(10, spec["ttl_seconds"] / 3)):
                        try:
                            result = client.call("renew", lease_token=lease_token, ttl_seconds=spec["ttl_seconds"])
                            if result.get("state") not in ("pending", "unloading", "ready"):
                                raise AdmissionError("Resource lease is no longer active")
                        except Exception:
                            renewal_failed.set()
                            LOG.error("Resource lease renewal failed; pending admission will not execute")
                            return
                renewer = threading.Thread(target=copy_context().run, args=(renew,), daemon=True,
                                           name="hermes-resource-lease")
                renewer.start()
                while True:
                    if cancelled() or self.clock() >= deadline or renewal_failed.is_set():
                        raise AdmissionError("Resource admission cancelled, expired, or timed out; command was not started")
                    response = client.call("status", lease_token=lease_token)
                    state = response.get("state")
                    if state == "ready":
                        break
                    if state not in ("pending", "unloading"):
                        raise AdmissionError("Engine could not grant the requested resource handoff")
                    stop.wait(.1)
                if cancelled() or self.clock() >= deadline or renewal_failed.is_set():
                    raise AdmissionError("Resource handoff cancelled before execution")
                # Exactly one continuation; no retry of a compiler, renderer or shell.
                return next_call()
            finally:
                stop.set()
                if renewer is not None:
                    renewer.join(timeout=client.timeout + 1)
                if lease_token:
                    try:
                        client.call("release", lease_token=lease_token)
                    except Exception:
                        LOG.warning("Resource lease release was not acknowledged; engine TTL remains authoritative")
