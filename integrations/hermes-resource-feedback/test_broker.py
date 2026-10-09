import json
from pathlib import Path
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from broker import AdmissionError, EngineClient, SupervisorBroker, validate_profiles


def scope(root="parent", depth=0):
    return {"root_session_id": root, "session_id": root if depth == 0 else root + "-child",
            "parent_session_id": "" if depth == 0 else root, "delegate_depth": depth,
            "profile_home": "test-profile", "lineage_valid": True}


PROFILES = {"compile": {"command_prefixes": ["cmake --build "], "ram_headroom_gib": 5}}


class Client:
    timeout = .1

    def __init__(self):
        self.calls = []
        self.state = "ready"
        self.ready = threading.Event()
        self.fail_release = False

    def call(self, action, **fields):
        self.calls.append((action, fields))
        if action == "acquire":
            self.ready.set()
        if action == "release" and self.fail_release:
            raise OSError("transport failed")
        return {"schema": "strata.resource-lease.v1", "state": self.state,
                "lease_token": "t" * 32}


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.client = Client()
        self.broker = SupervisorBroker(lambda: self.client, PROFILES, wait_seconds=1)

    def test_worker_cannot_grant_plan(self):
        with self.assertRaisesRegex(AdmissionError, "Workers"):
            self.broker.plan(scope(depth=1), ["compile"])
        self.assertEqual(self.client.calls, [])

    def test_worker_needs_parent_grant_before_network(self):
        with self.assertRaisesRegex(AdmissionError, "supervisor"):
            self.broker.run(scope(depth=1), "cmake --build .", lambda: self.fail("ran"))
        self.assertEqual(self.client.calls, [])

    def test_parent_grant_covers_descendant_and_returns_exact_result(self):
        self.broker.plan(scope(), ["compile"])
        value = {"exit_code": 0, "output": "PASS"}
        result = self.broker.run(scope(depth=1), "cmake --build .", lambda: value)
        self.assertIs(result, value)
        self.assertEqual([a for a, _ in self.client.calls], ["acquire", "status", "release"])

    def test_unmatched_tool_leaves_engine_alone(self):
        self.assertEqual(self.broker.run({}, "rg symbol .", lambda: 9), 9)
        self.assertFalse(self.client.calls)

    def test_other_root_does_not_inherit_plan(self):
        self.broker.plan(scope(), ["compile"])
        with self.assertRaises(AdmissionError):
            self.broker.run(scope("other", 1), "cmake --build .", lambda: self.fail("ran"))
        self.assertFalse(self.client.calls)

    def test_two_workers_are_serialized_by_one_supervisor(self):
        self.broker.plan(scope(), ["compile"])
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        results = []
        def first():
            entered.set()
            release.wait(2)
            return 1
        a = threading.Thread(target=lambda: results.append(self.broker.run(scope(depth=1), "cmake --build a", first)))
        b = threading.Thread(target=lambda: results.append(self.broker.run(scope(depth=1), "cmake --build b", lambda: second.set() or 2)))
        a.start()
        self.assertTrue(entered.wait(1))
        b.start()
        self.assertFalse(second.wait(.1))
        self.assertEqual(sum(a == "acquire" for a, _ in self.client.calls), 1)
        release.set()
        a.join(2)
        b.join(2)
        self.assertEqual(results, [1, 2])
        self.assertEqual([a for a, _ in self.client.calls], ["acquire", "status", "release"] * 2)

    def test_cancelled_wait_does_not_execute_and_releases(self):
        self.broker.plan(scope(), ["compile"])
        self.client.state = "pending"
        with self.assertRaisesRegex(AdmissionError, "cancelled"):
            self.broker.run(scope(), "cmake --build .", lambda: self.fail("ran"),
                            cancelled=self.client.ready.is_set)
        self.assertEqual(self.client.calls[-1][0], "release")
        self.assertIsNone(self.broker.active)

    def test_failed_engine_admission_does_not_execute(self):
        self.broker.plan(scope(), ["compile"])
        self.client.state = "failed"
        with self.assertRaises(AdmissionError):
            self.broker.run(scope(), "cmake --build .", lambda: self.fail("ran"))
        self.assertEqual(self.client.calls[-1][0], "release")

    def test_ready_reply_after_deadline_never_starts_command(self):
        now = [0]
        self.broker.clock = lambda: now[0]
        self.broker.plan(scope(), ["compile"])
        original = self.client.call
        def slow_status(action, **fields):
            result = original(action, **fields)
            if action == "status":
                now[0] = 2
            return result
        self.client.call = slow_status
        with self.assertRaises(AdmissionError):
            self.broker.run(scope(), "cmake --build .", lambda: self.fail("ran"))
        self.assertEqual(self.client.calls[-1][0], "release")

    def test_expired_queue_ticket_never_acquires(self):
        ticks = iter((0, 1))
        self.broker.clock = lambda: next(ticks)
        self.broker.plan(scope(), ["compile"])
        with self.assertRaisesRegex(AdmissionError, "timed out"):
            self.broker.run(scope(), "cmake --build .", lambda: self.fail("ran"))
        self.assertFalse(self.client.calls)
        self.assertEqual(len(self.broker.queue), 0)
        self.assertIsNone(self.broker.active)

    def test_tool_exception_never_replays(self):
        self.broker.plan(scope(), ["compile"])
        calls = []
        def execute():
            calls.append(1)
            raise RuntimeError("compiler failure")
        with self.assertRaisesRegex(RuntimeError, "compiler failure"):
            self.broker.run(scope(), "cmake --build .", execute)
        self.assertEqual(calls, [1])
        self.assertEqual(self.client.calls[-1][0], "release")

    def test_lost_release_does_not_hide_completed_command(self):
        self.broker.plan(scope(), ["compile"])
        self.client.fail_release = True
        with self.assertLogs("broker", level="WARNING"):
            self.assertEqual(self.broker.run(scope(), "cmake --build .", lambda: "done"), "done")

    def test_remote_background_and_invalid_lineage_block(self):
        self.broker.plan(scope(), ["compile"])
        for options in ({"background": True}, {"env_type": "docker"}):
            with self.assertRaises(AdmissionError):
                self.broker.run(scope(), "cmake --build .", lambda: self.fail("ran"), **options)
        invalid = scope(depth=1)
        invalid["lineage_valid"] = False
        with self.assertRaises(AdmissionError):
            self.broker.run(invalid, "cmake --build .", lambda: self.fail("ran"))
        self.assertFalse(self.client.calls)

    def test_ambiguous_profiles_block(self):
        b = SupervisorBroker(lambda: self.client, {**PROFILES, "other": PROFILES["compile"]})
        with self.assertRaisesRegex(AdmissionError, "multiple"):
            b.run(scope(), "cmake --build .", lambda: self.fail("ran"))

    def test_invalid_config(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                validate_profiles({"compile": {"command_prefixes": ["build "], "ram_headroom_gib": value}})

    def test_client_never_accepts_remote_or_redirect_targets(self):
        for url in ("http://example.com:8080", "http://localhost:8080", "http://user:pass@127.0.0.1:8080",
                    "http://127.0.0.1:8080/?x=1", "http://127.0.0.1:8080/api", "https://127.0.0.1:8080"):
            with self.assertRaises(ValueError):
                EngineClient(url, "s" * 32)
        self.assertEqual(EngineClient("http://127.0.0.1:8080", "s" * 32).url,
                         "http://127.0.0.1:8080/v1/resource-lease")

    def test_control_credentials_match_server_validation(self):
        for token in ("s" * 24, "s" * 31, "s" * 4097, "s" * 32 + " ", "s" * 32 + "\u00e9"):
            with self.assertRaises(ValueError):
                EngineClient("http://127.0.0.1:8080", token)

    def test_finite_plan_capacity_does_not_evict_an_explicit_denial(self):
        self.broker.plans = {f"root-{i}": frozenset() for i in range(4096)}
        with self.assertRaisesRegex(AdmissionError, "capacity"):
            self.broker.plan(scope(), ["compile"])
        self.assertEqual(len(self.broker.plans), 4096)


class HttpClientTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.mode = "ok"
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, self.headers.get("Authorization"), json.loads(payload)))
                if owner.mode == "redirect":
                    self.send_response(307)
                    self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/leak")
                    self.end_headers()
                    return
                body = (b"x" * 65537 if owner.mode == "oversize" else
                        json.dumps({"schema": "strata.resource-lease.v1", "state": "ready"}).encode())
                if owner.mode == "error":
                    body = b"secret server diagnostic must not reach the model"
                self.send_response(403 if owner.mode == "error" else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = EngineClient(f"http://127.0.0.1:{self.server.server_port}", "s" * 32)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_authenticated_control_serialization(self):
        result = self.client.call("status", lease_token="opaque")
        self.assertEqual(result["state"], "ready")
        self.assertEqual(self.requests, [("/v1/resource-lease", "Bearer " + "s" * 32,
                                        {"action": "status", "lease_token": "opaque"})])

    def test_redirect_does_not_forward_credentials(self):
        self.mode = "redirect"
        with self.assertRaises(AdmissionError):
            self.client.call("status", lease_token="opaque")
        self.assertEqual(len(self.requests), 1)

    def test_error_body_is_not_exposed(self):
        self.mode = "error"
        with self.assertRaises(AdmissionError) as raised:
            self.client.call("status", lease_token="opaque")
        self.assertEqual(str(raised.exception), "Resource control refused the request (HTTP 403)")

    def test_oversize_response_refused(self):
        self.mode = "oversize"
        with self.assertRaisesRegex(AdmissionError, "too large"):
            self.client.call("status", lease_token="opaque")


if __name__ == "__main__":
    unittest.main()
