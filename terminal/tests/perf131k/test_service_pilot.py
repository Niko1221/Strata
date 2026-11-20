"""CPU fixtures for the authorized service stop/restore guard; never touch real TTS."""
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import importlib.util
import psutil
from tools.perf131k import service_pilot as service
if os.environ.get("P01_SERVICE_UNDER_TEST"):
    spec = importlib.util.spec_from_file_location("service_under_test", os.environ["P01_SERVICE_UNDER_TEST"])
    service = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = service
    spec.loader.exec_module(service)


class ServiceGuardTests(unittest.TestCase):
    def test_false_CloseHandle_return_is_not_ignored(self):
        kernel = Mock()
        kernel.CloseHandle.return_value = False
        with patch.object(service, "kernel_api", return_value=kernel):
            with self.assertRaises(OSError):
                service.close_handle(11)

    def test_restore_owner_checks_parent_runtime_images_and_environment(self):
        parent, runtime = Mock(pid=100), Mock(pid=200)
        parent.cmdline.return_value = service.ARGS
        parent.cwd.return_value = runtime.cwd.return_value = str(service.SERVICE_ROOT)
        parent.exe.return_value = str(service.SERVICE_EXE)
        runtime.exe.return_value = sys.executable
        parent.create_time.return_value = 100.0
        runtime.create_time.return_value = 101.0
        runtime.ppid.return_value = 100
        parent.environ.return_value = {"PRIVATE_FIXTURE": "fixture-only"}
        saved = {"argv": service.ARGS, "cwd": str(service.SERVICE_ROOT),
                 "launcher_sha256": "image", "runtime_exe_sha256": "image"}
        connection = SimpleNamespace(status="LISTEN", laddr=SimpleNamespace(port=8089), pid=200)
        with patch.object(service.psutil, "net_connections", return_value=[connection]), \
             patch.object(service.psutil, "Process", side_effect=lambda pid: runtime if pid == 200 else parent), \
             patch.object(service.native, "sha", return_value="image"):
            identity = service.verify_restore_owner(saved, {"PRIVATE_FIXTURE": "fixture-only"}, 100)
            self.assertEqual(identity["runtime_pid"], 200)
            parent.environ.return_value = {"PRIVATE_FIXTURE": "fixture-only", "UNEXPECTED_CONFIG": "1"}
            with self.assertRaises(ValueError):
                service.verify_restore_owner(saved, {"PRIVATE_FIXTURE": "fixture-only"}, 100)
            parent.environ.return_value = {"PRIVATE_FIXTURE": "different"}
            with self.assertRaises(ValueError):
                service.verify_restore_owner(saved, {"PRIVATE_FIXTURE": "fixture-only"}, 100)

    def workflow_fixture(self, stop_error=False, close_error=False, job_commit_gib=None, observed=None):
        base = Path(tempfile.mkdtemp(dir=os.environ["TEMP"], prefix="service-workflow-"))
        (base / "prepared").mkdir()
        (base / "prepared/binding.json").write_text("{}")
        root = Mock(pid=100)
        root.cmdline.return_value = service.ARGS
        root.cwd.return_value = str(service.SERVICE_ROOT)
        root.create_time.return_value = 100.0
        root.exe.return_value = str(service.SERVICE_EXE)
        root.environ.return_value = {"PUBLIC_FIXTURE": "1"}
        runtime = Mock(pid=200)
        runtime.ppid.return_value = 100
        runtime.create_time.return_value = 101.0
        runtime.exe.return_value = sys.executable
        runtime.net_connections.return_value = [SimpleNamespace(status="LISTEN", laddr=SimpleNamespace(port=8089))]
        before = {"status": "ok", "model": {"id": "CPU_FIXTURE"}, "runtime": {"loading": False, "loaded": True}}
        def blocked_cell(count, trace, *, binding_sha, job_commit_gib=8):
            if observed is not None:
                observed.append(job_commit_gib)
            cell = service.native.BASE / f"native-{count}-{trace.lower()}"
            cell.mkdir()
            (cell / "result.json").write_text(json.dumps({"status": "PREFLIGHT_BLOCKED", "measurement_valid": False}))
            return 77
        with patch.object(service, "BASE", base), patch.object(service.psutil, "Process", side_effect=[root, runtime]), \
             patch.object(service, "health", return_value=before), patch.object(service.native, "sha", return_value="0"*64), \
             patch.object(service, "hold_process", side_effect=[11,22]), \
             patch.object(service, "stop_handle", side_effect=OSError("injected post-terminate wait failure") if stop_error else None), \
             patch.object(service, "close_handle", side_effect=[OSError("injected close failure"),None] if close_error else None), \
             patch.object(service, "listening", return_value=False), patch.object(service.native, "run_cell", side_effect=blocked_cell), \
             patch.object(service, "restore", return_value={"status": "CPU_FIXTURE_RESTORED"}) as restored:
            options = {} if job_commit_gib is None else {"job_commit_gib": job_commit_gib}
            code = service.execute(100, 200, **options)
        attempts = list(base.glob("service-attempt-*"))
        return code, restored.call_count, json.loads((attempts[0] / "overall.json").read_text())

    def test_ten_GiB_is_forwarded_and_restore_still_runs(self):
        observed = []
        code, restores, result = self.workflow_fixture(job_commit_gib=10, observed=observed)
        self.assertNotEqual(code, 0)
        self.assertEqual((observed, restores), ([10], 1))
        self.assertEqual(result["requested_job_commit_gib"], 10)

    def test_invalid_budget_rejected_before_service_side_effects(self):
        with patch.object(service, "hold_process") as hold, patch.object(service, "stop_handle") as stop:
            with self.assertRaises(ValueError):
                service.execute(100, 200, job_commit_gib=11)
        hold.assert_not_called()
        stop.assert_not_called()

    def test_wait_failure_after_stop_request_does_not_skip_restore(self):
        code, restores, result = self.workflow_fixture(stop_error=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(restores, 1)

    def test_handle_close_failure_does_not_skip_restore(self):
        code, restores, result = self.workflow_fixture(close_error=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(restores, 1)

    def test_blocked_native_cell_is_not_overall_success(self):
        code, restores, result = self.workflow_fixture()
        self.assertNotEqual(code, 0)
        self.assertEqual(result["status"], "PILOT_BLOCKED")

    def test_same_model_health_without_owner_identity_is_not_restored(self):
        directory = Path(tempfile.mkdtemp(dir=os.environ["TEMP"], prefix="restore-owner-"))
        snapshot = {"argv": service.ARGS, "launcher_sha256": "0"*64, "environment_ciphertext_sha256": "0"*64,
                    "health": {"model": {"id": "CPU_FIXTURE"}, "runtime": {"loaded": True}}}
        (directory / "service-snapshot.json").write_text(json.dumps(snapshot))
        (directory / "environment.dpapi").write_bytes(service.protect(b"{}"))
        with patch.object(service.native, "sha", return_value="0"*64), \
             patch.object(service, "listening", return_value=True), \
             patch.object(service, "health", return_value={"status":"ok", "model":{"id":"CPU_FIXTURE"}, "runtime":{"loaded":False}}), \
             patch.object(service, "verify_restore_owner", create=True, side_effect=ValueError("wrong owner")):
            with self.assertRaises(ValueError):
                service.restore(directory)

    def test_environment_backup_is_encrypted_and_round_trips(self):
        secret = b'{"SECRET_TEST_VALUE":"not-a-real-credential"}'
        sealed = service.protect(secret)
        self.assertNotIn(b"not-a-real-credential", sealed)
        self.assertEqual(service.unprotect(sealed), secret)

    def test_bound_handle_only_stops_the_owned_fake_process(self):
        command = [sys.executable, "-I", "-B", "-c", "import time; time.sleep(20)"]
        one = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        other = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc = psutil.Process(one.pid)
            handle = service.hold_process(one.pid, proc.create_time(), proc.exe())
            try:
                with self.assertRaises(ValueError):
                    service.hold_process(other.pid, proc.create_time() - 20, proc.exe())
                service.stop_handle(handle)
                self.assertIsNotNone(one.wait(timeout=5))
                self.assertIsNone(other.poll())
            finally:
                service.close_handle(handle)
        finally:
            for process in (one, other):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
