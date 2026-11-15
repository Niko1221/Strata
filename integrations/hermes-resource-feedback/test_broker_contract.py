"""Broker/ToolLeases contract, using fake lifecycle ACKs and no model processes.

These tests prove that the actual client-side and server-side state contracts
compose. They do not measure live relief, process death, GPU release or speed.
"""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from broker import AdmissionError, SupervisorBroker
from serve.resource_lease import ToolLeases


def capacity():
    return {"sampled_at": 102, "native_sampled_at": 102, "native_free_mib": 1500,
            "ram_total": 64 * 2**30, "ram_used": 40 * 2**30,
            "ram_commit_required": True, "ram_commit_available": 24 * 2**30,
            "gpu_mem_total": 8192 * 2**20, "gpu_mem_used": 6692 * 2**20}


SCOPE = {"root_session_id": "supervisor", "session_id": "worker", "parent_session_id": "supervisor",
         "profile_home": "isolated-test", "lineage_valid": True, "delegate_depth": 1}


class StateMachineClient:
    timeout = .1

    def __init__(self, selected_action="none"):
        self.now = 0
        with patch.dict("os.environ", {"STRATA_RESOURCE_LEASE_TOKEN": "test-control-" * 4}):
            self.leases = ToolLeases({"enabled": True}, clock=lambda: self.now)
        self.selected_action = selected_action
        self.calls = []
        self.lost_start = False
        self.failed_release = False
        self.fail_admission = False

    def capabilities(self):
        self.calls.append("capabilities")
        return self.leases.public()

    def qualify(self):
        row = self.leases.demand()
        if self.fail_admission:
            self.leases.fail("strict_relief_unreachable")
            return
        lease_id = row["lease_id"]
        self.leases.select_action(lease_id, self.selected_action, "fake_lifecycle_test")
        if self.selected_action == "unload":
            self.leases.unloaded(self.leases.begin_unload(), 101)
            self.leases.observe_ready(capacity(), 102, unloaded=True, ram_floor=3, gpu_floor=250, commit_floor=3)
        else:
            self.leases.seal_residency(lease_id, 42, 20000, 264)
            if self.selected_action == "relieve":
                self.leases.begin_operation(lease_id, "relieve", 1, 42)
                self.leases.complete_operation(lease_id, 1, 42)
            self.leases.observe_resident_ready(capacity(), 102, lease_id=lease_id, engine_identity=42,
                drained_at=101, ram_floor=3, gpu_floor=250, commit_floor=3)

    def call(self, action, **fields):
        self.calls.append(action)
        request = {"action": action, **fields}
        if action == "acquire":
            return self.leases.acquire(request, capacity(), 102)
        if action == "status":
            self.qualify()
        if action == "release" and self.failed_release:
            raise AdmissionError("simulated failed release")
        result = self.leases.action(request)
        if action == "start" and self.lost_start:
            raise AdmissionError("simulated lost start reply")
        return result


class StateMachineContractTests(unittest.TestCase):
    def broker(self, client, mode="auto"):
        return SupervisorBroker(lambda: client, {"compile": {"command_prefixes": ["build "], "mode": mode,
            "ram_headroom_gib": 6, "vram_headroom_mib": 512, "execution_ram_floor_gib": 3,
            "execution_vram_floor_mib": 250}}, default_profiles=["compile"], wait_seconds=1)

    def test_actual_contract_all_permitted_actions_dispatch_once_after_start(self):
        for mode, action in (("auto", "none"), ("auto", "relieve"), ("auto", "unload"),
                             ("relieve", "none"), ("relieve", "relieve"), ("unload", "unload")):
            with self.subTest(mode=mode, action=action):
                client = StateMachineClient(action)
                executions = []
                def execute():
                    row = client.leases.demand()
                    self.assertEqual(row["phase"], "execution")
                    self.assertEqual(row["selected_action"], action)
                    self.assertEqual((row["ram_target_gib"], row["vram_target_mib"]), (3, 250))
                    self.assertTrue(client.leases.blocked())
                    executions.append({"output": "real result", "exit_code": 0, "error": None})
                    return executions[0]
                self.assertEqual(self.broker(client, mode).run(SCOPE, "build project", execute),
                                 {"output": "real result", "exit_code": 0, "error": None})
                self.assertEqual(len(executions), 1)
                self.assertEqual(client.calls, ["capabilities", "acquire", "status", "start", "release"])
                self.assertFalse(client.leases.blocked())

    def test_actual_committed_start_with_lost_reply_is_released_without_dispatch(self):
        client = StateMachineClient()
        client.lost_start = True
        with self.assertRaisesRegex(AdmissionError, "lost start"):
            self.broker(client).run(SCOPE, "build project", lambda: self.fail("command ran"))
        self.assertEqual(client.leases.demand()["phase"], "execution")
        self.assertEqual(client.calls.count("start"), 1)
        self.assertFalse(client.leases.blocked())

    def test_actual_strict_relief_failure_never_starts(self):
        client = StateMachineClient()
        client.fail_admission = True
        with self.assertRaises(AdmissionError):
            self.broker(client, "relieve").run(SCOPE, "build project", lambda: self.fail("command ran"))
        self.assertNotIn("start", client.calls)
        self.assertFalse(client.leases.blocked())

    def test_actual_execution_expiry_holds_until_completed_tool_is_released(self):
        for mode in ("auto", "relieve"):
            with self.subTest(mode=mode):
                client = StateMachineClient()
                def execute():
                    client.now = 61  # TTL alone is not evidence the tool ended.
                    self.assertTrue(client.leases.blocked())
                    self.assertEqual(client.leases.public()["barrier_reason"], "tool_exit_unconfirmed")
                    return {"output": "completed after expiry", "exit_code": 0, "error": None}
                self.assertEqual(self.broker(client, mode).run(SCOPE, "build project", execute)["output"], "completed after expiry")
                self.assertEqual(client.calls[-1], "release")
                self.assertFalse(client.leases.blocked())

    def test_failed_release_preserves_actual_result_and_resident_execution_hold(self):
        client = StateMachineClient()
        client.failed_release = True
        result = {"output": "done", "exit_code": 0, "error": None}
        with self.assertLogs("broker", level="WARNING") as log:
            self.assertIs(self.broker(client).run(SCOPE, "build project", lambda: result), result)
        client.now = 61
        self.assertTrue(client.leases.blocked())
        self.assertIn("operator recovery", log.output[0])
        self.assertEqual(client.calls.count("start"), 1)

    def test_uncertain_terminal_results_remain_blocked_after_return_and_expiry(self):
        results = (None, "not terminal JSON", "null", "[]", "{}", "{" * 10000,
            json.dumps({"output":"", "exit_code":124, "error":None}),
            json.dumps({"output":"", "exit_code":130, "error":None}),
            json.dumps({"output":"started", "exit_code":0, "error":None,"pid":1}),
            {"output": "", "exit_code": -1, "error": "post-spawn OSError"},
            {"output": "", "exit_code": 124, "error": None},
            {"output": "", "exit_code": 130, "error": None},
            {"output": "", "exit_code": 137, "error": None},
            {"output": "", "exit_code": True},
            {"output": "", "exit_code": 0},
            {"output": "", "exit_code": 0., "error": None},
            {"output": "[Command timed out after 2s]", "exit_code": 0, "error": None},
            {"output": "[Command interrupted]", "exit_code": 0, "error": None},
            {"output": "started", "exit_code": 0, "error": None, "session_id": "still-running", "pid": 1},
            {"output": "", "exit_code": 0, "error": None, "status": "yielded_to_background"},
            {"output": "", "exit_code": 0, "error": None, "hermes_timed_out": True},
            {"output": "", "exit_code": 0, "error": None, "environment_recreated": "changed"},
            {"output": "", "exit_code": 0, "unknown_backend_field": True})
        for mode in ("auto", "relieve"):
            for original in results:
                with self.subTest(mode=mode, value=repr(original)[:100]):
                    client = StateMachineClient()
                    calls = []
                    def execute():
                        calls.append(1)
                        return original
                    with self.assertLogs("broker", level="WARNING"):
                        self.assertIs(self.broker(client, mode).run(SCOPE, "build project", execute), original)
                    self.assertEqual(calls, [1])
                    self.assertNotIn("release", client.calls)
                    self.assertTrue(client.leases.blocked())
                    client.now = 61
                    self.assertTrue(client.leases.blocked())
                    self.assertEqual(client.leases.public()["barrier_reason"], "tool_exit_unconfirmed")

    def test_post_spawn_exception_preserved_without_release_or_retry(self):
        for mode in ("auto", "relieve"):
            for error in (OSError("post-spawn pipe failure"), TimeoutError("outer backend wait"),
                          KeyboardInterrupt(), SystemExit(1)):
                with self.subTest(mode=mode, error=type(error).__name__):
                    client = StateMachineClient()
                    calls = []
                    def execute():
                        calls.append(1)
                        raise error
                    with self.assertLogs("broker", level="WARNING"), self.assertRaises(type(error)) as caught:
                        self.broker(client, mode).run(SCOPE, "build project", execute)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(calls, [1])
                    self.assertNotIn("release", client.calls)
                    client.now = 61
                    self.assertTrue(client.leases.blocked())

    def test_known_synchronous_terminal_envelopes_release_without_rewriting(self):
        for code in (0, 1, 2, 127):
            for serialized in (False, True):
                with self.subTest(code=code, serialized=serialized):
                    client = StateMachineClient()
                    value = {"output": "actual output", "exit_code": code, "error": None,
                             "cwd": "project", "verification_evidence": {"status": "passed"}}
                    if serialized:
                        value = json.dumps(value)
                    self.assertIs(self.broker(client).run(SCOPE, "build project", lambda: value), value)
                    self.assertEqual(client.calls.count("release"), 1)
                    self.assertFalse(client.leases.blocked())

    def test_legacy_unload_retains_release_on_unknown_result_or_callback_error(self):
        client = StateMachineClient("unload")
        self.assertIsNone(self.broker(client, "unload").run(SCOPE, "build project", lambda: None))
        self.assertFalse(client.leases.blocked())
        client = StateMachineClient("unload")
        error = OSError("legacy callback error")
        def execute():
            raise error
        with self.assertRaises(OSError) as caught:
            self.broker(client, "unload").run(SCOPE, "build project", execute)
        self.assertIs(caught.exception, error)
        self.assertEqual(client.calls.count("release"), 1)
        self.assertFalse(client.leases.blocked())


if __name__ == "__main__":
    unittest.main()
