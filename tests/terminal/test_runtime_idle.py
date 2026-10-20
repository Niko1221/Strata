"""Private idle admission claims and generation-free managed model switching."""

import hashlib
import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from strata_control import profiles, runtime
from tests.terminal import test_runtime as runtime_fixtures


class OwnerIdleHTTPTests(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                           ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja"))
        self.svc.api_key = "public-inference-key"
        self.svc.owner_control_token = "d" * 64
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.port = self.httpd.server_address[1]
        self.claim = "e" * 32

    def request(self, action="claim", *, headers=None, token=True, instance=None, claim=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            values = {"Content-Type": "application/json", **(headers or {})}
            if token:
                values["X-Strata-Owner-Control"] = self.svc.owner_control_token
            conn.request("POST", "/_strata/owner/idle", json.dumps({
                "action": action, "instance_id": instance or self.svc.instance_id,
                "claim_id": claim or self.claim}), values)
            reply = conn.getresponse()
            return reply.status, json.loads(reply.read())
        finally:
            conn.close()

    def test_inference_key_is_not_owner_authority_and_browser_headers_are_refused(self):
        self.assertEqual(self.request(token=False, headers={"Authorization": "Bearer public-inference-key"})[0], 401)
        for headers in ({"Origin": "http://localhost"}, {"Sec-Fetch-Site": "same-origin"},
                        {"Forwarded": "for=127.0.0.1"}, {"Host": "attacker.example"}):
            self.assertEqual(self.request(headers=headers)[0], 403)
        self.assertFalse(self.svc.draining)

    def test_unowned_server_has_no_control_endpoint(self):
        self.svc.owner_control_token = None
        self.assertEqual(self.request(token=False)[0], 404)
        self.assertFalse(self.svc.draining)

    def test_claim_closes_inference_and_controls_until_exact_release(self):
        self.assertEqual(self.request(instance="0" * 32)[0], 409)
        self.assertEqual(self.request()[0], 200)
        self.assertIsNone(self.svc.reserve_inference())
        self.assertFalse(self.svc.fifo.acquire(blocking=False))
        self.assertEqual(self.request("release", claim="f" * 32)[0], 409)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            conn.request("POST", "/v1/load", "{}", {"Authorization": "Bearer public-inference-key"})
            response = conn.getresponse()
            self.assertEqual(response.status, 503)
            response.read()
        finally:
            conn.close()
        self.assertEqual(self.request("release")[0], 200)
        self.assertTrue(self.svc.reserve_inference())
        self.svc.release_inference()
        self.assertEqual(self.request("release")[0], 409)

    def test_busy_claim_cannot_interrupt_reserved_or_fifo_work(self):
        self.assertTrue(self.svc.reserve_inference())
        self.assertEqual(self.request()[0], 409)
        self.assertFalse(self.svc.draining)
        self.svc.release_inference()
        self.svc.fifo.acquire()
        self.assertEqual(self.request()[0], 409)
        self.svc.fifo.release()
        self.svc.owner_control_reserved = 1
        self.assertEqual(self.request()[0], 409)
        self.svc.owner_control_reserved = 0
        self.assertEqual(self.request()[0], 200)
        self.assertEqual(self.request("release")[0], 200)

    def test_atomic_reservation_race_has_exactly_one_winner(self):
        for _ in range(32):
            gate = threading.Barrier(3)
            results = {}
            def reserve():
                gate.wait()
                results["reserved"] = self.svc.reserve_inference()
            def claim():
                gate.wait()
                results["claim"] = self.svc.owner_idle("claim", self.svc.instance_id, self.claim)[0]
            threads = [threading.Thread(target=reserve), threading.Thread(target=claim)]
            for thread in threads:
                thread.start()
            gate.wait()
            for thread in threads:
                thread.join(2)
                self.assertFalse(thread.is_alive())
            self.assertEqual((results["reserved"] is True) + (results["claim"] == 200), 1)
            if results["reserved"]:
                self.svc.release_inference()
            else:
                self.assertEqual(self.svc.owner_idle("release", self.svc.instance_id, self.claim)[0], 200)


class IdleRuntimeGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strata-idle-guards-")
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        env = mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(directory / "runtime")})
        env.start()
        self.addCleanup(env.stop)
        self.host = runtime.Host("local", 30)
        self.path = directory / "run.json"
        self.path.write_text('{"model_name":"fake"}')
        self.raw = self.path.read_bytes()
        self.host.source = self.path
        self.host.source_bytes = self.raw
        self.host.args = ["--config", str(self.path), "--host", "127.0.0.1", "--port", "1"]
        self.host.revision = "1" * 16
        self.host.child = mock.Mock()
        self.host.child.poll.return_value = None
        self.instance = "2" * 32
        self.snapshot = {"ok": True, "state": "ready", "owner_id": self.host.owner_id,
                         "instance_id": self.instance, "loaded": True}
        self.plan = {"profile_revision": self.host.revision}
        self.candidate = (self.plan, self.host.args, self.path, self.raw)

    def test_idle_dispatch_validates_identity_before_any_mutation(self):
        with mock.patch.object(self.host, "_inspect", return_value=self.snapshot), \
                mock.patch.object(self.host, "_owner_idle") as claim:
            result = self.host._handle({"token": self.host.token, "command": "restart_if_idle",
                                        "expected_owner_id": "0" * 32})
            self.assertEqual(result["error"], "owner_changed_after_restart_plan")
            claim.assert_not_called()
        self.assertEqual(self.host._handle({"token": "wrong", "command": "inspect"})["error"], "unauthorized")

    def test_plan_and_idle_restart_never_call_generation_status(self):
        with mock.patch.object(self.host, "_inspect", return_value=self.snapshot), \
                mock.patch.object(self.host, "_plan", return_value=self.candidate), \
                mock.patch.object(self.host, "_status", side_effect=AssertionError("generation status used")), \
                mock.patch.object(self.host, "_owner_idle", side_effect=profiles.ProfileError("runtime_busy", 6)), \
                mock.patch.object(self.host, "_stop_child") as stop:
            self.assertTrue(self.host._plan_restart(refuse_busy=True)["ok"])
            result = self.host._restart(1, refuse_busy=True)
            self.assertEqual(result["error"], "runtime_busy")
            stop.assert_not_called()

    def test_candidate_changed_after_claim_releases_before_return(self):
        changed = ({"profile_revision": "3" * 16}, self.host.args, self.path, self.raw)
        with mock.patch.object(self.host, "_inspect", return_value=self.snapshot), \
                mock.patch.object(self.host, "_plan", side_effect=[self.candidate, changed]), \
                mock.patch.object(self.host, "_owner_idle", return_value={"ok": True}) as idle, \
                mock.patch.object(self.host, "_stop_child") as stop:
            result = self.host._restart(1, refuse_busy=True)
            self.assertEqual(result["error"], "candidate_changed_during_idle_claim")
            self.assertTrue(result["idle_claim_release_confirmed"])
            self.assertEqual([call.args[0] for call in idle.call_args_list], ["claim", "release"])
            self.assertEqual(idle.call_args_list[0].args[1:], idle.call_args_list[1].args[1:])
            stop.assert_not_called()

    def test_lost_claim_response_attempts_release_and_never_stops(self):
        with mock.patch.object(self.host, "_inspect", return_value=self.snapshot), \
                mock.patch.object(self.host, "_plan", return_value=self.candidate), \
                mock.patch.object(self.host, "_owner_idle", side_effect=[OSError("lost reply"), {"ok": True}]) as idle, \
                mock.patch.object(self.host, "_stop_child") as stop:
            result = self.host._restart(1, refuse_busy=True)
            self.assertTrue(result["idle_claim_release_confirmed"])
            self.assertEqual(idle.call_count, 2)
            stop.assert_not_called()

    def test_failed_new_launch_rolls_back_using_read_only_readiness(self):
        with mock.patch.object(self.host, "_inspect", return_value=self.snapshot), \
                mock.patch.object(self.host, "_plan", return_value=self.candidate), \
                mock.patch.object(self.host, "_owner_idle", return_value={"ok": True}), \
                mock.patch.object(self.host, "_stop_child", return_value={"ok": True}), \
                mock.patch.object(self.host, "_launch", side_effect=[profiles.ProfileError("failed", 6), None]) as launch, \
                mock.patch.object(self.host, "_wait_ready", return_value=self.snapshot) as ready:
            result = self.host._restart(1, refuse_busy=True)
            self.assertFalse(result["ok"])
            self.assertTrue(result["rolled_back"])
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(launch.call_args.args, (self.host.args, self.path, self.raw, self.host.revision))
            ready.assert_called_once_with(1, read_only=True)

    def test_loaded_inspect_proves_identity_without_generation(self):
        self.host.ready_path = Path(self.temp.name) / "ready.json"
        self.host.ready_path.write_text(json.dumps({"pid": 123, "instance_id": self.instance,
            "bound_host": "127.0.0.1", "bound_port": 12345, "model_id": "fake"}))
        self.host.child.pid = 123
        self.host.expected_attestation = "4" * 64
        telemetry = {"instance": {"id": self.instance, "launch_attestation": "4" * 64, "state": "ready"},
                     "model": {"id": "fake", "loaded": True},
                     "work": {"active": 0, "queued": 0, "admission_reserved": 0}}
        with mock.patch.object(runtime.readiness, "_request", return_value=telemetry) as request, \
                mock.patch.object(runtime.readiness, "verify", side_effect=AssertionError("generation verification used")):
            result = self.host._inspect()
            self.assertTrue(result["ok"])
            self.assertTrue(result["loaded"])
            self.assertFalse(result["generation_ready"])
            self.assertEqual(result["source_sha256"], hashlib.sha256(self.raw).hexdigest())
            self.assertEqual(request.call_args.args[2:4], ("GET", "/v1/telemetry"))
            self.assertNotIn("launch_attestation", json.dumps(result))
            telemetry["instance"]["launch_attestation"] = "5" * 64
            self.assertEqual(self.host._inspect()["error"], "probe_launch_attestation_mismatch")

    def test_failed_owner_recovery_never_signals_a_live_child_or_ignores_guards(self):
        request = {"token": self.host.token, "command": "reap_if_exited", "expected_owner_id": self.host.owner_id,
                   "expected_source_sha256": hashlib.sha256(self.raw).hexdigest()}
        with mock.patch.object(self.host, "_status", side_effect=AssertionError("generation status used")), \
                mock.patch.object(self.host, "_stop_child", side_effect=AssertionError("must not stop a live child")):
            self.assertEqual(self.host._handle(request)["error"], "runtime_child_still_alive")
            self.assertTrue(self.host.running)
            self.host.child.poll.return_value = 1
            self.assertFalse(self.host._inspect()["child_alive"])
            self.assertEqual(self.host._inspect()["state"], "failed")
            for field, value, expected in (("expected_owner_id", "0" * 32, "owner_changed_before_recovery"),
                                           ("expected_source_sha256", "0" * 64, "source_changed_before_recovery")):
                self.assertEqual(self.host._handle({**request, field: value})["error"], expected)
                self.assertTrue(self.host.running)
            result = self.host._handle(request)
            self.assertTrue(result["owner_reaped"])
            self.assertFalse(self.host.running)
            self.host.child.send_signal.assert_not_called()

    def test_empty_failed_launch_owner_is_reaped_only_with_its_declared_source(self):
        self.host.child = None
        self.host.source_bytes = None
        self.host.expected_source_sha256 = "4" * 64
        self.host.last_error = "new_instance_launch_failed"
        status = self.host._inspect()
        self.assertFalse(status["child_alive"])
        self.assertEqual(status["source_sha256"], "4" * 64)
        self.assertEqual(self.host._handle({"token": self.host.token, "command": "reap_if_exited",
            "expected_owner_id": self.host.owner_id, "expected_source_sha256": "4" * 64})["ok"], True)


class ManagedIdleLifecycleTests(unittest.TestCase):
    # Reuse only the disposable engine/tokenizer fixture, not the fixture class's tests.
    setUp = runtime_fixtures.RuntimeTests.setUp
    _stop = runtime_fixtures.RuntimeTests._stop

    def test_real_failed_child_can_retire_its_owner_and_retry(self):
        cfg = json.loads(self.config.read_text())
        engine = self.root / "engine.py"
        original = engine.read_bytes()
        engine.write_text("#!" + os.sys.executable + "\nimport sys\nsys.exit(3)\n")
        started = runtime.start("local", wait=10,
                                expected_source_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.assertFalse(started["ok"], started)
        deadline = time.monotonic() + 5
        observed = runtime.inspect("local")
        while observed.get("child_alive") is not False and time.monotonic() < deadline:
            time.sleep(0.05)
            observed = runtime.inspect("local")
        self.assertFalse(observed["child_alive"], observed)
        result = runtime.reap_if_exited("local", expected_owner_id=observed["owner_id"],
                                       expected_source_sha256=observed["source_sha256"])
        self.assertTrue(result["ok"], result)
        self.assertIsNone(runtime._read_state("local"))
        engine.write_bytes(original)
        cfg["lazy_load"] = True
        self.config.write_text(json.dumps(cfg))
        self.assertTrue(runtime.start("local", wait=10)["ok"])

    def test_loaded_model_switch_is_owned_lazy_and_never_generates(self):
        cfg = json.loads(self.config.read_text())
        cfg["lazy_load"] = True
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            cfg["port"] = reservation.getsockname()[1]
        self.config.write_text(json.dumps(cfg))
        engine = self.root / "engine.py"
        text = engine.read_text()
        text = text.replace("        slot = int(line.split()[1])",
                            "        Path('generated-marker').write_text('generation happened')\n"
                            "        slot = int(line.split()[1])")
        engine.write_text(text)
        started = runtime.start("local", wait=10)
        self.assertTrue(started["ok"], started)
        current = runtime.inspect("local")
        self.assertFalse(current["loaded"])
        self.assertFalse((self.root / "generated-marker").exists())
        address = current["api_url"].removeprefix("http://").split(":")
        conn = http.client.HTTPConnection(address[0], int(address[1]), timeout=10)
        try:
            conn.request("POST", "/v1/load", "{}", {"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200, response.read())
            response.read()
        finally:
            conn.close()
        current = runtime.inspect("local")
        self.assertTrue(current["loaded"])
        self.assertFalse(current["generation_ready"])
        self.assertFalse((self.root / "generated-marker").exists())
        candidate = self.root / "second.json"
        candidate.write_text(json.dumps({**cfg, "model_name": "second"}))
        proposed, revision = profiles.propose_source("local", str(candidate))
        profiles.save("local", proposed, expected_revision=revision)
        preview = runtime.plan_restart_if_idle("local", allow_source_switch=True)
        self.assertTrue(preview["ok"], preview)
        result = runtime.restart_if_idle("local", startup_wait=10, timeout=25,
            allow_source_switch=True, expected_profile_revision=preview["candidate_profile_revision"],
            expected_source_sha256=preview["candidate_source_sha256"],
            expected_owner_id=preview["current_owner_id"], expected_instance_id=preview["current_instance_id"])
        self.assertTrue(result["ok"], result)
        after = runtime.inspect("local")
        self.assertEqual(after["model_id"], "second")
        self.assertNotEqual(after["instance_id"], current["instance_id"])
        self.assertEqual(after["owner_id"], current["owner_id"])
        self.assertEqual(after["api_url"], current["api_url"])
        self.assertFalse(after["loaded"])
        self.assertFalse((self.root / "generated-marker").exists())


if __name__ == "__main__":
    unittest.main()
