"""Forge /stats cache tests. All HTTP fixtures bind an ephemeral loopback port."""
from __future__ import annotations

import json
import os
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest import mock

from serve.forge_stats import ForgeStats, _fetch
from serve.server import LOOPBACK_NAMES, Server, Service, make_handler


class FakeClock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


PAYLOAD = {
    "forge_version": "0.16.87", "day": "2026-10-05",
    "today": {"turns": 41, "requests": 512, "compactions": 2, "compaction_attempts_failed": 1,
              "compactions_suppressed": 4, "tool_calls": 486, "tool_failures": 23,
              "input_tokens": 9120345, "output_tokens": 88213, "turn_errors": 0},
    "last_request": {"model": "strata-flashnext-iq3s", "input_tokens": 70241,
                     "context_limit": 200000, "at": 1791182458},
    "computed_at": 1791182500, "skipped_files": 0,
}


class ForgeHandler(BaseHTTPRequestHandler):
    payload = PAYLOAD

    def do_GET(self):
        if self.path != "/stats":
            self.send_error(404)
            return
        raw = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args):
        pass


def service_with_forge(reader):
    """Build only the Service fields needed for its real /metrics path."""
    svc = object.__new__(Service)
    svc.api_key = ""
    svc.host_names = set(LOOPBACK_NAMES)
    svc.allowed_hosts = []
    svc.trusted_origins = []
    svc.cors_origins = []
    svc.status_lock = threading.Lock()
    svc.status = {"busy": False, "queued": 0}
    svc.live_reqs = {}
    svc.history = []
    svc.totals = {"since": 100, "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                  "prompt_ms": 0, "decode_ms": 0, "drafts_offered": 0, "drafts_accepted": 0}
    svc.engine = SimpleNamespace(progress=None, batch=0, max_context=200000, info={})
    svc.vision = None
    svc.model = "strata-flashnext-iq3s"
    svc.telemetry = SimpleNamespace(snapshot=lambda: {"now": {}, "history": {}, "static": {}})
    svc.forge_stats = reader
    svc.conv_log = SimpleNamespace(poll=lambda *_args: {})
    svc.loaded = lambda: True
    svc._tok_s = lambda: 0.0
    svc._tok_s_mean = lambda: 0.0
    svc._prefill_tok_s_mean = lambda: 0.0
    return svc


class ForgeStatsTests(unittest.TestCase):
    def test_fake_forge_is_exposed_by_metrics_with_age(self):
        httpd = HTTPServer(("127.0.0.1", 0), ForgeHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        clock = FakeClock()
        reader = ForgeStats(f"http://127.0.0.1:{httpd.server_address[1]}", clock=clock, start=False)
        try:
            self.assertTrue(reader.poll_once())
            svc = service_with_forge(reader)
            app = Server(("127.0.0.1", 0), make_handler(svc))
            app_thread = threading.Thread(target=app.serve_forever, daemon=True)
            app_thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{app.server_address[1]}/metrics", timeout=1) as response:
                    forge = json.loads(response.read())["forge"]
                self.assertEqual(forge["today"]["turns"], 41)
                self.assertEqual(forge["age_s"], 0.0)
                clock.advance(16)
                self.assertTrue(reader.snapshot()["stale"])
                clock.advance(45)
                self.assertIsNone(reader.snapshot())
            finally:
                app.shutdown()
                app.server_close()
                app_thread.join(timeout=1)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_forge_down_is_null_and_metrics_does_not_wait(self):
        closed = HTTPServer(("127.0.0.1", 0), ForgeHandler)
        url = f"http://127.0.0.1:{closed.server_address[1]}"
        closed.server_close()
        reader = ForgeStats(url, start=False)
        self.assertFalse(reader.poll_once())
        self.assertIsNone(reader.snapshot())
        app = Server(("127.0.0.1", 0), make_handler(service_with_forge(reader)))
        thread = threading.Thread(target=app.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.perf_counter()
            with urllib.request.urlopen(f"http://127.0.0.1:{app.server_address[1]}/metrics", timeout=1) as response:
                body = json.loads(response.read())
            self.assertLess(time.perf_counter() - started, 0.1)
            self.assertIsNone(body["forge"])
        finally:
            app.shutdown()
            app.server_close()
            thread.join(timeout=1)

    def test_closed_port_backs_off_for_30_seconds(self):
        clock = FakeClock()
        attempts = []
        closed = HTTPServer(("127.0.0.1", 0), ForgeHandler)
        url = f"http://127.0.0.1:{closed.server_address[1]}"
        closed.server_close()

        def refused(url, timeout):
            attempts.append((url, timeout))
            return _fetch(url, timeout)

        reader = ForgeStats(url, fetch=refused, clock=clock, start=False)
        for _ in range(10):
            reader.poll_once()
            clock.advance(1)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0][1], 1)

    def test_empty_environment_disables_poller(self):
        with mock.patch.dict(os.environ, {"STRATA_FORGE_URL": ""}):
            reader = ForgeStats(fetch=lambda *_args: self.fail("disabled Forge reader fetched"), start=False)
        self.assertFalse(reader.enabled)
        self.assertFalse(reader.poll_once())
        self.assertIsNone(reader.snapshot())

    def test_no_url_disables_poller_without_attempting_fetch(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            reader = ForgeStats(fetch=lambda *_args: self.fail("Forge without a URL fetched"), start=False)
        self.assertEqual(reader.url, "")
        self.assertFalse(reader.enabled)
        self.assertFalse(reader.poll_once())

    def test_close_ends_the_poller_thread(self):
        reader = ForgeStats(url="http://127.0.0.1:9", fetch=lambda *_args: PAYLOAD)
        self.assertTrue(reader.thread.is_alive())
        reader.close()
        reader.thread.join(2)
        self.assertFalse(reader.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
