"""Harness regressions, separate from the immutable acceptance registry."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("luna_drive", Path(__file__).with_name("drive.py"))
drive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drive)


class EvidenceTests(unittest.TestCase):
    def test_import_error_is_not_red_evidence(self):
        report = {"status": "FAIL", "test_count": 1,
                  "executed_ids": ["unittest.loader._FailedTest.backend"],
                  "failures": [], "errors": [{"id": "backend", "traceback": "ModuleNotFoundError"}],
                  "skipped": [], "expected_failures": []}
        self.assertFalse(drive.meaningful_red(report),
                         "An import error must not satisfy the test-first red requirement")

    def test_skip_xfail_and_wrong_assertion_do_not_qualify(self):
        red = {"test_count": 1, "executed_ids": ["fixture.C.test_law"],
               "failures": [{"id": "fixture.C.test_law", "traceback": "AssertionError: branch leaked"}]}
        self.assertTrue(drive.meaningful_red(red, ["test_law"], "branch leaked"))
        self.assertFalse(drive.meaningful_red(red, ["test_other"]))
        self.assertFalse(drive.meaningful_red(red, ["test_law"], "wrong fixture"))
        for field in ("skipped", "errors", "expected_failures", "unexpected_successes"):
            self.assertFalse(drive.meaningful_red({**red, field: ["one"]}))
        passing = {"status": "PASS", "test_count": 1, "executed_ids": ["fixture.C.test_law"]}
        self.assertTrue(drive.successful(passing, ["test_law"]))
        self.assertFalse(drive.successful(passing, ["test_other"]))
        self.assertFalse(drive.successful({**passing, "skipped": ["one"]}))
        self.assertFalse(drive.successful({**passing, "test_count": 0}))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(drive.PACKAGE / "scripts/run_ticket.py", self.root / "scripts/run_ticket.py")
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        self.fixture = self.root / "tests/test_fixture.py"
        self.fixture.write_text("import unittest\nclass Law(unittest.TestCase):\n"
                                "    def test_law(self): self.assertEqual(1, 1)\n", encoding="utf-8")
        self.command = {"id": "law", "kind": "unittest", "phase": "observe",
                        "tests": {"directory": "tests", "pattern": "test_fixture.py"},
                        "expected_ids": ["test_law"]}
        self.ticket = {"id": "A", "kind": "observation", "depends_on": [], "commands": [self.command]}
        self.queue = self.root / "queue.json"
        self.make_driver()

    def make_driver(self, tickets=None):
        self.queue.write_text(json.dumps({"tickets": tickets or [self.ticket]}), encoding="utf-8")
        self.driver = drive.Driver(self.queue, package=self.root, repo=self.root)
        return self.driver

    def test_executed_ids_and_gate_scope(self):
        result = self.driver.run("A", "law")
        self.assertEqual(result["outcome"], "PASS")
        self.assertEqual(result["test_report"]["executed_ids"], ["test_fixture.Law.test_law"])
        self.assertEqual(result["test_report"]["test_count"], 1)
        self.driver.record("A", "VALIDATED", "Actual model-free assertion passed")
        self.assertTrue(self.driver.check()[0]["validation_current"])
        self.assertNotIn("gate", result)

    def test_stale_native_and_binary_fingerprints(self):
        binary = self.root / "sut.bin"
        binary.write_bytes(b"build one")
        bindings = self.root / "bindings.json"
        bindings.write_text(json.dumps({"artifacts": {"sut": str(binary)}}), encoding="utf-8")
        self.driver = drive.Driver(self.queue, package=self.root, repo=self.root, bindings=bindings)
        self.driver.run("A", "law")
        binary.write_bytes(b"build two")
        with self.assertRaisesRegex(drive.Refusal, "stale"):
            self.driver.record("A", "VALIDATED", "Cannot use old binary results")
        source = self.root / "include/state.hpp"
        source.parent.mkdir()
        source.write_text("// changed native interface", encoding="utf-8")
        old = self.driver.runs("A")[0]["before"]["sha256"]
        self.assertNotEqual(self.driver.fingerprint()["sha256"], old)

    def test_missing_or_modified_evidence_is_rejected(self):
        result = self.driver.run("A", "law")
        Path(result["stdout"]).unlink()
        with self.assertRaisesRegex(drive.Refusal, "Missing or changed"):
            self.driver.record("A", "VALIDATED", "A missing log cannot validate")

    def test_required_output_artifact_cannot_be_omitted(self):
        self.command.update(kind="command", argv=["{python}", "-c", "print('no measurement written')"],
                            artifacts=["{result}"])
        self.make_driver()
        result = self.driver.run("A", "law")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["outcome"], "MISSING_ARTIFACT")

    def test_timeout_terminates_owned_descendants(self):
        receipt = self.root / "child_pid.txt"
        script = ("import subprocess,sys,time; from pathlib import Path; "
                  "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                  f"Path({str(receipt)!r}).write_text(str(p.pid)); time.sleep(30)")
        self.command.update(kind="command", argv=["{python}", "-c", script], timeout_seconds=2)
        self.make_driver()
        result = self.driver.run("A", "law")
        self.assertEqual(result["outcome"], "TIMEOUT")
        self.assertTrue(receipt.is_file(), "The owned grandchild must actually have started")
        pid = int(receipt.read_text())
        self.assertIsNone(drive.process_identity(pid), "Timeout leaked an owned descendant")

    def test_skipped_test_never_validates(self):
        self.fixture.write_text("import unittest\nclass Law(unittest.TestCase):\n"
                                "    @unittest.skip('unsupported')\n    def test_law(self): pass\n", encoding="utf-8")
        result = self.driver.run("A", "law")
        self.assertEqual(result["outcome"], "FAIL")
        self.assertEqual(len(result["test_report"]["skipped"]), 1)
        with self.assertRaises(drive.Refusal):
            self.driver.record("A", "VALIDATED", "Skipped is not success")

    def test_duplicate_current_execution_is_refused(self):
        self.driver.run("A", "law")
        with self.assertRaisesRegex(drive.Refusal, "already has current evidence"):
            self.driver.run("A", "law")

    def test_stale_observation_can_be_revalidated(self):
        self.driver.run("A", "law")
        self.driver.record("A", "VALIDATED", "First observed execution")
        source = self.root / "include/new.hpp"
        source.parent.mkdir()
        source.write_text("// a new native revision", encoding="utf-8")
        self.assertFalse(self.driver.check()[0]["validation_current"])
        self.driver.run("A", "law")
        try:
            self.driver.record("A", "VALIDATED", "Executed again after the source change")
        except drive.Refusal as error:
            self.fail("A stale observation must be revalidatable after fresh execution: " + str(error))
        self.assertTrue(self.driver.check()[0]["validation_current"])

    def test_command_failure_and_timeout_preserved(self):
        self.command.update(kind="command", argv=["{python}", "-c", "raise SystemExit(7)"])
        self.make_driver()
        failed = self.driver.run("A", "law")
        self.assertEqual(failed["exit_code"], 7)
        self.assertEqual(failed["outcome"], "FAIL")
        self.command.update(argv=["{python}", "-c", "import time; time.sleep(30)"], timeout_seconds=0.1)
        self.make_driver()
        timed = self.driver.run("A", "law")
        self.assertEqual(timed["outcome"], "TIMEOUT")
        self.assertTrue(Path(timed["stdout"]).exists())
        with self.assertRaisesRegex(drive.Refusal, "cannot be restarted unchanged"):
            self.driver.run("A", "law", retry_note="Do not repeat timed-out jobs")

    def test_live_process_is_not_restarted(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: child.poll() is None and child.kill())
        identity = drive.process_identity(child.pid)
        self.driver.append("run_start", run="live", ticket="A", command_id="law", before=self.driver.fingerprint())
        self.driver.append("child", run="live", ticket="A", identity=identity)
        with self.assertRaisesRegex(drive.Refusal, "still live"):
            self.driver.run("A", "law")
        child.kill()
        child.wait(timeout=5)
        self.driver.inspect_active(recover=True)
        self.assertEqual(self.driver.events()[-1]["kind"], "run_interrupted")
        with self.assertRaisesRegex(drive.Refusal, "Interrupted command"):
            self.driver.run("A", "law")

    def test_ambiguous_launch_does_not_restart(self):
        self.driver.append("run_start", run="unknown", ticket="A", command_id="law", before=self.driver.fingerprint())
        with self.assertRaisesRegex(drive.Refusal, "ambiguous launch"):
            self.driver.run("A", "law")

    def test_one_active_ticket_and_blocked_dependencies(self):
        b = {**self.ticket, "id": "B", "depends_on": ["A"]}
        c = {**self.ticket, "id": "C"}
        self.make_driver([self.ticket, b, c])
        self.driver.record("A", "ACTIVE", "Working on first ticket")
        with self.assertRaisesRegex(drive.Refusal, "active ticket A"):
            self.driver.record("C", "ACTIVE", "Cannot run another active ticket")
        with self.assertRaisesRegex(drive.Refusal, "dependencies"):
            self.driver.run("B", "law")
        blocker = self.root / "blocker.txt"
        blocker.write_text("A requires a missing backend primitive", encoding="utf-8")
        self.driver.record("A", "BLOCKED", "Missing primitive", str(blocker))
        self.assertEqual(self.driver.next_ticket()["id"], "C")

    def test_disabled_campaign_cannot_be_enabled_by_record(self):
        self.ticket["enabled"] = False
        self.make_driver()
        self.assertIsNone(self.driver.next_ticket())
        with self.assertRaises(drive.Refusal):
            self.driver.run("A", "law")
        with self.assertRaisesRegex(drive.Refusal, "disabled"):
            self.driver.record("A", "ACTIVE", "No long campaign authorization")

    def test_real_red_before_green_and_unchanged_assertions(self):
        self.ticket["kind"] = "implementation"
        self.command["phase"] = "red"
        (self.root / "src/implementation.py").write_text("value = 0\n", encoding="utf-8")
        self.fixture.write_text("import unittest, implementation\nclass Law(unittest.TestCase):\n"
                                "    def test_law(self): self.assertEqual(implementation.value, 1)\n", encoding="utf-8")
        green = {**self.command, "id": "green", "phase": "green"}
        self.ticket["commands"] = [self.command, green]
        self.make_driver()
        with self.assertRaisesRegex(drive.Refusal, "after red"):
            self.driver.run("A", "green")
        red = self.driver.run("A", "law")
        self.assertEqual(red["outcome"], "RED")
        (self.root / "src/implementation.py").write_text("value = 1\n", encoding="utf-8")
        # Remove only the temporary fixture's compiled cache: timestamp/size can coincide.
        shutil.rmtree(self.root / "src/__pycache__", ignore_errors=True)
        self.driver.record("A", "IMPLEMENTED", "Fixed the implementation, same assertion")
        green_result = self.driver.run("A", "green")
        self.assertEqual(green_result["outcome"], "PASS")
        self.driver.record("A", "VALIDATED", "Unchanged assertion now passes")

    def test_changed_test_cannot_satisfy_original_red(self):
        self.ticket["kind"] = "implementation"
        self.command["phase"] = "red"
        self.fixture.write_text("import unittest\nclass Law(unittest.TestCase):\n"
                                "    def test_law(self): self.assertEqual(0, 1)\n", encoding="utf-8")
        self.make_driver()
        self.assertEqual(self.driver.run("A", "law")["outcome"], "RED")
        self.fixture.write_text(self.fixture.read_text() + "# assertion changed\n", encoding="utf-8")
        with self.assertRaisesRegex(drive.Refusal, "Tests changed"):
            self.driver.record("A", "IMPLEMENTED", "Cannot edit red test away")

    def test_lock_excludes_second_runner_and_ledger_detects_corruption(self):
        lock = self.root / "evidence/runs/luna/driver.lock"
        with drive.exclusive(lock):
            with self.assertRaises(drive.Refusal):
                with drive.exclusive(lock):
                    self.fail("A second runner acquired the same lock")
        self.driver.append("state", ticket="A", state="ACTIVE", note="Recorded")
        ledger = self.driver.ledger / "events.jsonl"
        ledger.write_bytes(ledger.read_bytes().replace(b'"ACTIVE"', b'"PENDING"'))
        with self.assertRaisesRegex(drive.Refusal, "hash chain"):
            self.driver.events()


if __name__ == "__main__":
    unittest.main(verbosity=2)
