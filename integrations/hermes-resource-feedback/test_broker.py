import json
from pathlib import Path
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

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


class PhasedClient(Client):
    def __init__(self):
        super().__init__()
        self.phase = "admission"
        self.request = {}
        self.statuses = []
        self.selected_action = "none"
        self.mutate = lambda action, response: response

    def capabilities(self):
        self.calls.append(("capabilities", {}))
        return self.mutate("capabilities", {"schema": "strata.resource-lease.v1", "enabled": True,
            "supported_modes": ["unload", "auto", "relieve"], "supports_start": True})

    def call(self, action, **fields):
        super().call(action, **fields)
        if action == "acquire":
            self.request = fields
            self.phase = "admission"
        if action == "start":
            self.phase = "execution"
        state = self.statuses.pop(0) if action == "status" and self.statuses else self.state
        response = {"schema": "strata.resource-lease.v1", "enabled": True,
            "supported_modes": ["unload", "auto", "relieve"], "supports_start": True,
            "state": state, "phase": self.phase, "lease_id": "lease-one", "lease_token": "t" * 32,
            "mode": self.request.get("mode", "unload"), "expires_in_seconds": 60,
            "selected_action": self.selected_action}
        for field in ("ram_headroom_gib", "vram_headroom_mib"):
            response[field] = self.request.get(field, 0)
        response["execution_ram_floor_gib"] = self.request.get("execution_ram_floor_gib", response["ram_headroom_gib"])
        response["execution_vram_floor_mib"] = self.request.get("execution_vram_floor_mib", response["vram_headroom_mib"])
        return self.mutate(action, response)


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
        acquired = self.client.calls[0][1]
        self.assertEqual(set(acquired), {"request_id", "mode", "ram_headroom_gib", "vram_headroom_mib", "ttl_seconds"})
        self.assertEqual(acquired["mode"], "unload")

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


class PhasedBrokerTests(unittest.TestCase):
    def setUp(self):
        self.client = PhasedClient()
        self.executed = []
        self.now = [0]

    def broker(self, **options):
        profiles = {"compile": {**PROFILES["compile"], "mode": "auto", **options}}
        result = SupervisorBroker(lambda: self.client, profiles, wait_seconds=1, clock=lambda: self.now[0])
        result.plan(scope(), ["compile"])
        return result

    def execute(self):
        self.executed.append("once")
        return json.dumps({"output": "actual result", "exit_code": 0, "error": None})

    def actions(self):
        return [action for action, _ in self.client.calls]

    def test_auto_requires_owner_start_before_one_dispatch(self):
        def execute():
            self.assertEqual(self.actions(), ["capabilities", "acquire", "status", "start"])
            self.assertEqual(self.client.phase, "execution")
            return self.execute()
        self.assertEqual(json.loads(self.broker().run(scope(depth=1), "cmake --build .", execute))["output"], "actual result")
        self.assertEqual(self.executed, ["once"])
        self.assertEqual(self.actions()[-1], "release")
        self.assertEqual(self.client.calls[-2][1], {"lease_token": "t" * 32})

    def test_new_server_phases_default_unload_without_capability_preflight(self):
        self.client.selected_action = "unload"
        self.assertEqual(json.loads(self.broker(mode="unload").run(scope(), "cmake --build .", self.execute))["output"], "actual result")
        self.assertEqual(self.actions(), ["acquire", "status", "start", "release"])

    def test_explicit_execution_floors_are_sent_as_total_floors(self):
        b = self.broker(execution_ram_floor_gib=2, execution_vram_floor_mib=0)
        b.run(scope(), "cmake --build .", self.execute)
        self.assertEqual(self.client.request["ram_headroom_gib"], 5)
        self.assertEqual(self.client.request["execution_ram_floor_gib"], 2)
        self.assertEqual(self.client.request["execution_vram_floor_mib"], 0)

    def test_absent_execution_floors_are_not_inferred_or_sent(self):
        self.broker().run(scope(), "cmake --build .", self.execute)
        self.assertNotIn("execution_ram_floor_gib", self.client.request)
        self.assertNotIn("execution_vram_floor_mib", self.client.request)

    def test_invalid_modes_and_execution_floors_fail_configuration(self):
        for options in ({"mode": "none"}, {"mode": True}, {"mode": ["auto"]},
                {"execution_ram_floor_gib": 6}, {"execution_ram_floor_gib": -1},
                {"execution_ram_floor_gib": float("nan")}, {"execution_ram_floor_gib": True},
                {"execution_ram_floor_gib": 10**1000},
                {"execution_vram_floor_mib": 251}, {"execution_vram_floor_mib": None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.broker(**options)
        self.assertFalse(self.client.calls)

    def test_explicit_modes_and_floor_options_reject_old_capabilities_before_acquire(self):
        self.client.mutate = lambda action, response: ({"schema": "strata.resource-lease.v1",
            "enabled": True, "supported_modes": ["unload"]} if action == "capabilities" else response)
        for options in ({"mode": "auto"}, {"mode": "relieve"},
                        {"mode": "unload", "execution_ram_floor_gib": 4}):
            with self.subTest(options=options), self.assertRaisesRegex(AdmissionError, "does not support"):
                self.broker(**options).run(scope(), "cmake --build .", self.execute)
        self.assertEqual(self.actions(), ["capabilities"] * 3)
        self.assertFalse(self.executed)

    def test_public_capability_ready_never_authorizes_dispatch(self):
        def mutate(action, response):
            if action == "capabilities":
                response["state"] = "ready"
            if action == "status":
                response["state"] = "failed"
            return response
        self.client.mutate = mutate
        with self.assertRaises(AdmissionError):
            self.broker().run(scope(), "cmake --build .", self.execute)
        self.assertNotIn("start", self.actions())
        self.assertFalse(self.executed)
        self.assertEqual(self.actions()[-1], "release")

    def test_capability_or_mode_downgrade_after_acquire_is_released(self):
        for fields in ({"supports_start": False}, {"mode": "unload"}, {"enabled": False},
                       {"execution_ram_floor_gib": 0}, {"lease_id": ""}):
            with self.subTest(fields=fields):
                self.client = PhasedClient()
                self.client.mutate = lambda action, response: response | fields if action == "acquire" else response
                with self.assertRaises(AdmissionError):
                    self.broker().run(scope(), "cmake --build .", self.execute)
                self.assertEqual(self.actions(), ["capabilities", "acquire", "release"])
        self.assertFalse(self.executed)

    def test_pending_relief_and_unload_statuses_do_not_dispatch_early(self):
        self.client.statuses = ["pending", "relieving", "unloading", "ready"]
        self.broker().run(scope(), "cmake --build .", self.execute)
        self.assertEqual(self.actions().count("status"), 4)
        self.assertEqual(self.actions()[-2:], ["start", "release"])
        self.assertEqual(self.executed, ["once"])

    def test_unknown_transition_and_wrong_owner_fail_without_start(self):
        for fields in ({"state": "migrating"}, {"lease_id": "another-lease"},
                       {"phase": "execution"}, {"expires_in_seconds": 0}, {"expires_in_seconds": True},
                       {"expires_in_seconds": 10**1000}):
            with self.subTest(fields=fields):
                self.client = PhasedClient()
                self.client.mutate = lambda action, response: response | fields if action == "status" else response
                with self.assertRaises(AdmissionError):
                    self.broker().run(scope(), "cmake --build .", self.execute)
                self.assertNotIn("start", self.actions())
                self.assertEqual(self.actions()[-1], "release")
        self.assertFalse(self.executed)

    def test_strict_relief_rejects_unload_selection(self):
        self.client.selected_action = "unload"
        with self.assertRaisesRegex(AdmissionError, "outside"):
            self.broker(mode="relieve").run(scope(), "cmake --build .", self.execute)
        self.assertFalse(self.executed)
        self.assertEqual(self.actions()[-1], "release")

    def test_strict_relief_accepts_none_or_live_relief(self):
        for action in ("none", "relieve"):
            with self.subTest(action=action):
                self.client = PhasedClient()
                self.client.selected_action = action
                self.broker(mode="relieve").run(scope(), "cmake --build .", self.execute)
        self.assertEqual(self.executed, ["once", "once"])

    def test_lost_start_reply_never_retries_or_executes(self):
        def mutate(action, response):
            if action == "start":
                raise AdmissionError("simulated lost acknowledgement")
            return response
        self.client.mutate = mutate
        with self.assertRaisesRegex(AdmissionError, "lost"):
            self.broker().run(scope(), "cmake --build .", self.execute)
        self.assertEqual(self.actions(), ["capabilities", "acquire", "status", "start", "release"])
        self.assertEqual(self.client.phase, "execution")  # Server may have committed it.
        self.assertFalse(self.executed)

    def test_lost_acquire_reply_never_retries_or_executes(self):
        def mutate(action, response):
            if action == "acquire":
                raise AdmissionError("simulated lost acquire reply")
            return response
        self.client.mutate = mutate
        with self.assertRaisesRegex(AdmissionError, "lost acquire"):
            self.broker().run(scope(), "cmake --build .", self.execute)
        self.assertEqual(self.actions(), ["capabilities", "acquire"])
        self.assertFalse(self.executed)

    def test_phased_workers_share_the_existing_fifo_until_tool_return(self):
        broker = self.broker()
        entered, finish, second = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def first():
            entered.set()
            finish.wait(1)
            return self.execute()
        def run(command, callback):
            try:
                broker.run(scope(depth=1), command, callback)
            except BaseException as exc:
                errors.append(exc)
        a = threading.Thread(target=run, args=("cmake --build first", first))
        b = threading.Thread(target=run, args=("cmake --build second", lambda: second.set() or self.execute()))
        a.start()
        self.assertTrue(entered.wait(1))
        b.start()
        try:
            self.assertFalse(second.wait(.1))
            self.assertEqual(self.actions(), ["capabilities", "acquire", "status", "start"])
        finally:
            finish.set()
            a.join(2)
            b.join(2)
        self.assertFalse(errors)
        self.assertTrue(second.is_set())
        self.assertEqual(self.actions(), ["capabilities", "acquire", "status", "start", "release"] * 2)

    def test_start_must_acknowledge_same_owner_and_execution_phase(self):
        for fields in ({"state": "pending"}, {"phase": "admission"}, {"lease_id": "wrong"},
                       {"expires_in_seconds": 0}, {"mode": "unload"}, {"selected_action": None}):
            with self.subTest(fields=fields):
                self.client = PhasedClient()
                self.client.mutate = lambda action, response: response | fields if action == "start" else response
                with self.assertRaises(AdmissionError):
                    self.broker().run(scope(), "cmake --build .", self.execute)
                self.assertEqual(self.actions().count("start"), 1)
                self.assertEqual(self.actions()[-1], "release")
        self.assertFalse(self.executed)

    def test_late_or_cancelled_start_acknowledgement_never_executes(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self.now[0] = 0
                self.client = PhasedClient()
                cancelled = threading.Event()
                def mutate(action, response):
                    if action == "start":
                        cancelled.set() if cancel else self.now.__setitem__(0, 2)
                    return response
                self.client.mutate = mutate
                with self.assertRaises(AdmissionError):
                    self.broker().run(scope(), "cmake --build .", self.execute, cancelled=cancelled.is_set)
                self.assertEqual(self.actions()[-1], "release")
        self.assertFalse(self.executed)

    def test_cancel_during_relief_releases_without_start(self):
        self.client.statuses = ["relieving"]
        cancelled = threading.Event()
        def mutate(action, response):
            if action == "status":
                cancelled.set()
            return response
        self.client.mutate = mutate
        with self.assertRaises(AdmissionError):
            self.broker().run(scope(), "cmake --build .", self.execute, cancelled=cancelled.is_set)
        self.assertNotIn("start", self.actions())
        self.assertFalse(self.executed)
        self.assertEqual(self.actions()[-1], "release")

    def test_execution_does_not_recheck_total_admission_target(self):
        def execute():
            self.client.state = "pending"  # A tool can consume its admitted capacity.
            return self.execute()
        self.broker(execution_ram_floor_gib=1).run(scope(), "cmake --build .", execute)
        self.assertEqual(self.actions().count("status"), 1)
        self.assertEqual(self.executed, ["once"])

    def test_tool_exception_retains_hold_and_preserves_actual_single_error(self):
        self.client.fail_release = True
        def execute():
            self.execute()
            raise RuntimeError("actual compiler error")
        with self.assertLogs("broker", level="WARNING"), self.assertRaisesRegex(RuntimeError, "compiler error"):
            self.broker().run(scope(), "cmake --build .", execute)
        self.assertEqual(self.executed, ["once"])
        self.assertEqual(self.actions().count("start"), 1)
        self.assertNotIn("release", self.actions())

    def test_confirmed_terminal_result_and_lost_release_preserve_original_result(self):
        self.client.fail_release = True
        original = {"exit_code": 1, "output": "compiler reported an error", "error": None}
        with self.assertLogs("broker", level="WARNING"):
            self.assertIs(self.broker().run(scope(), "cmake --build .", lambda: original), original)
        self.assertEqual(self.actions().count("start"), 1)
        self.assertEqual(self.actions().count("release"), 1)

    def test_renewal_accepts_relief_and_catches_owner_or_phase_loss(self):
        # Execute the captured renewal closure deterministically: one renewal,
        # then stop. No ten-second wait or live model is required.
        for fields in ({"state": "relieving"}, {"phase": "execution"},
                       {"lease_id": "wrong"}, {"phase": "invalid"}, {"state": "expired"}):
            with self.subTest(fields=fields):
                self.client = PhasedClient()
                self.client.mutate = lambda action, response: response | fields if action == "renew" else response
                failure = threading.Event()
                class Stop:
                    waits = 0
                    def wait(self, timeout):
                        self.waits += 1
                        return self.waits > 1
                    def set(self):
                        pass
                class Thread:
                    def __init__(self, target, args, **kwargs):
                        self.renew = args[0]
                    def start(self):
                        self.renew()
                    def join(self, timeout):
                        pass
                accepted = fields in ({"state": "relieving"}, {"phase": "execution"})
                with patch("broker.threading.Event", side_effect=[Stop(), failure]), patch("broker.threading.Thread", Thread):
                    if accepted:
                        self.broker().run(scope(), "cmake --build .", self.execute)
                    else:
                        with self.assertLogs("broker", level="ERROR"), self.assertRaises(AdmissionError):
                            self.broker().run(scope(), "cmake --build .", self.execute)
                self.assertEqual(failure.is_set(), not accepted)
                self.assertEqual(self.actions().count("renew"), 1)
                self.assertEqual(self.actions()[-1], "release")


class HttpClientTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.mode = "ok"
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append((self.path, self.headers.get("Authorization"), None))
                body = json.dumps({"schema": "strata.resource-lease.v1", "enabled": True,
                    "supported_modes": ["unload", "auto", "relieve"], "supports_start": True}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

    def test_public_capability_discovery_has_no_owner_or_tool_arguments(self):
        result = self.client.capabilities()
        self.assertTrue(result["supports_start"])
        self.assertEqual(self.requests, [("/v1/resource-lease", "Bearer " + "s" * 32, None)])

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
