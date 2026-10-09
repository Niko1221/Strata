"""A non-blocking cache of Forge's local, read-only stats endpoint.

Set `STRATA_FORGE_URL` to enable it. Forge is a VS Code coding assistant with a local, read-only `/stats` endpoint;
the card shows the active Forge chat's turns, tools and tokens.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request


def _fetch(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


class ForgeStats:
    """Poll ``<forge_url>/stats`` off-thread and expose a short-lived, locked snapshot."""
    DEFAULT_URL = ""
    POLL_S = 5
    TIMEOUT_S = 1
    BACKOFF_S = 30
    STALE_S = 15
    EXPIRE_S = 60

    def __init__(self, url=None, fetch=None, clock=None, start=True):
        base = os.environ.get("STRATA_FORGE_URL", self.DEFAULT_URL) if url is None else url
        self.url = f"{base.strip().rstrip('/')}/stats" if base.strip() else ""
        self.enabled = bool(self.url)
        self.fetch = fetch or _fetch
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._reply = None
        self._received_at = None
        self._next_try = 0.0
        self._stop = threading.Event()
        self.thread = None
        if self.enabled and start:
            self.thread = threading.Thread(target=self._run, name="forge-stats", daemon=True)
            self.thread.start()

    def poll_once(self):
        """Poll if due; tests can drive the schedule without starting the daemon thread."""
        if not self.enabled:
            return False
        with self._poll_lock:
            now = self.clock()
            with self._lock:
                if now < self._next_try:
                    return False
                self._next_try = now + self.POLL_S
            try:
                reply = self.fetch(self.url, self.TIMEOUT_S)
                if not isinstance(reply, dict):
                    raise ValueError("Forge /stats did not return an object")
            except Exception:  # noqa: BLE001 - the optional local service must not affect Strata
                with self._lock:
                    self._reply = None
                    self._received_at = None
                    self._next_try = self.clock() + self.BACKOFF_S
                return False
            received_at = self.clock()
            with self._lock:
                self._reply = dict(reply)
                self._received_at = received_at
            return True

    def snapshot(self):
        """Return a copy with its age, stale marker, or None after expiry."""
        with self._lock:
            if self._reply is None or self._received_at is None:
                return None
            age = max(0.0, self.clock() - self._received_at)
            if age > self.EXPIRE_S:
                return None
            return {**self._reply, "age_s": round(age, 1), "stale": age > self.STALE_S}

    def close(self):
        """Ends the poller thread with the server that started it."""
        self._stop.set()

    def _run(self):
        while self.enabled and not self._stop.is_set():
            self.poll_once()
            with self._lock:
                delay = max(0.0, self._next_try - self.clock())
            self._stop.wait(delay)
