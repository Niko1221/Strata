"""CPU-only pilot launch contract; retained H fixtures, no engine imports."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import unittest
import uuid
import time

MODULE = Path(os.environ.get("P01_PILOT_UNDER_TEST", Path(__file__).resolve().parents[2] /
                             "tools/perf131k/pilot.py"))
spec = importlib.util.spec_from_file_location("pilot_under_test", MODULE)
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
if os.environ.get("P01_JOB_UNDER_TEST"):
    import tools.perf131k as package
    job_spec = importlib.util.spec_from_file_location("tools.perf131k.windows_job", os.environ["P01_JOB_UNDER_TEST"])
    job_module = importlib.util.module_from_spec(job_spec)
    sys.modules[job_spec.name] = job_module
    job_spec.loader.exec_module(job_module)
    package.windows_job = job_module


def configuration():
    snapshot = Path("H:/Strata/worktrees/p01-a-20261025/_docs/perf131k/model-inventory.json")
    cfg = json.loads(snapshot.read_text(encoding="utf-8-sig"))["saved_Q2_configuration"]
    cfg = {k: cfg[k] for k in ("args", "env")}
    cfg["exe"] = "H:/Strata/build/p01-b0-20261025/bin/strata.exe"
    cfg["args"][cfg["args"].index("--max-context") + 1] = "131072"
    return cfg


def destination():
    return Path("H:/Strata/worktrees/p01-a-20261025/_scratch/perf131k/b2-fixtures") / uuid.uuid4().hex


class ContractTests(unittest.TestCase):
    def test_diagnostic_is_explicit_and_cannot_mix_with_normal(self):
        cfg = configuration()
        index = cfg["args"].index("--mtp")
        del cfg["args"][index:index+2]
        cfg["args"][cfg["args"].index("--spec") + 1] = "2"
        cfg["args"] += ["--suffix-draft", "0", "--lookup-chain", "0"]
        plan = pilot.contract(cfg, destination(), pilot.Budget(), series="diagnostic")
        self.assertEqual(plan["series"], "diagnostic")
        with self.assertRaises(ValueError):
            pilot.contract(cfg, destination(), pilot.Budget(), series="normal")
        with self.assertRaises(ValueError):
            pilot.contract(configuration(), destination(), pilot.Budget(), series="diagnostic")

    def test_normal_preserves_settings_and_materializes_parent_guard(self):
        cfg = configuration()
        before = json.dumps(cfg, sort_keys=True)
        contract = pilot.contract(cfg, destination(), pilot.Budget())
        self.assertEqual(json.dumps(cfg, sort_keys=True), before)
        self.assertEqual(contract["native_args"], cfg["args"])
        self.assertEqual(contract["server_env"]["STRATA_KEEP_LEFTOVERS"], "1")
        self.assertFalse(contract["model_launch_enabled"])
        self.assertEqual(contract["capacity"], 131072)
        self.assertEqual(contract["output_reserve"], 256)
        self.assertEqual(contract["series"], "normal")

    def test_refuses_precision_capacity_draft_and_write_overrides(self):
        for flag, value in (("--kv", "q4"), ("--max-context", "32768"), ("--spec", "2")):
            with self.subTest(flag=flag):
                cfg = configuration()
                cfg["args"][cfg["args"].index(flag) + 1] = value
                with self.assertRaises(ValueError):
                    pilot.contract(cfg, destination(), pilot.Budget())
        for extra in (["--expert-profile-save", "C:/bad"], ["--max-context", "131072"],
                      ["--cache-type-k", "q4_0"], ["--draft", "H:/other"]):
            cfg = configuration()
            cfg["args"] += extra
            with self.assertRaises(ValueError):
                pilot.contract(cfg, destination(), pilot.Budget())

    def test_dynamic_and_secret_settings_fail_closed_without_writes(self):
        for key, value in (("user_args", ["--spec", "2"]), ("expert_profile_save", "H:/bad"),
                           ("drafting", "auto"), ("vision", True), ("api_key", "never-log")):
            cfg = configuration()
            cfg[key] = value
            target = destination()
            with self.assertRaises(ValueError):
                pilot.contract(cfg, target, pilot.Budget())
            self.assertFalse(target.exists())

    def test_h_paths_and_resource_values_are_bounded(self):
        for value in (float("nan"), float("inf"), -1, 0, 61, True):
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                pilot.contract(configuration(), destination(), pilot.Budget(timeout_s=value))
        with self.assertRaises(ValueError):
            pilot.contract(configuration(), Path("C:/pilot"), pilot.Budget())
        with self.assertRaises(ValueError):
            pilot.contract(configuration(), Path("H:/Strata/models/do-not-write"), pilot.Budget())
        cfg = configuration()
        cfg["exe"] = "component:strata-engine"
        with self.assertRaises(ValueError):
            pilot.contract(cfg, destination(), pilot.Budget())


class LifecycleTests(unittest.TestCase):
    def test_bounded_stdin_reaches_only_owned_cpu_child(self):
        from tools.perf131k import windows_job
        folder = destination()
        folder.mkdir(parents=True)
        source = folder / "stdin.txt"
        source.write_bytes(b"OWNED_STDIN\n")
        command = [sys.executable, "-I", "-B", "-c", "import sys; print(sys.stdin.readline().strip())"]
        env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR") if k in os.environ}
        result = windows_job.run_owned(command, folder, env, folder / "raw.bin", pilot.Budget(),
                                       pilot.system_resources, stdin_path=source)
        self.assertEqual(result["status"], "FINISHED")
        self.assertEqual((folder / "raw.bin").read_bytes().strip(), b"OWNED_STDIN")
        self.assertEqual(result["job_active_after"], 0)
        self.assertTrue(result["termination_confirmed"])
        with self.assertRaises(ValueError):
            windows_job.run_owned(command, folder, env, folder / "must-not-exist.bin", pilot.Budget(),
                                  pilot.system_resources, stdin_path=Path(__file__))
        self.assertFalse((folder / "must-not-exist.bin").exists())

    def test_owner_abrupt_exit_kills_only_its_private_tree(self):
        import ctypes as C
        from ctypes import wintypes as W
        from tools.perf131k import windows_job
        folder = destination()
        folder.mkdir(parents=True)
        cfg_path = folder / "config.json"
        cfg_path.write_text(json.dumps(configuration()))
        wrapper = folder / "owner.py"
        wrapper.write_text('''import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from tools.perf131k import pilot
original = pilot.system_resources
def sample(root):
    if (root / "owner-exit").exists():
        os._exit(31)  # Intentionally bypass all Python finally/cleanup.
    return original(root)
pilot.system_resources = sample
pilot.run_cpu_fixture(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]),
                      pilot.Budget(timeout_s=10), "tree_hold")
''', encoding="utf-8")
        run = folder / "run"
        owner = subprocess.Popen([sys.executable, "-B", str(wrapper), str(cfg_path), str(run)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sentinel = subprocess.Popen([sys.executable, "-I", "-B", "-c", "import time; time.sleep(20)"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        kernel = windows_job.api()
        kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        kernel.OpenProcess.restype = W.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
        kernel.QueryFullProcessImageNameW.restype = W.BOOL
        handles, identities = [], []
        try:
            deadline, pids = time.monotonic() + 5, []
            while time.monotonic() < deadline:
                if (run / "raw.bin").exists():
                    for line in (run / "raw.bin").read_bytes().splitlines():
                        if line.startswith(b"TREE "):
                            pids = [int(v) for v in line.split()[1:]]
                if pids:
                    break
                self.assertIsNone(owner.poll())
                time.sleep(0.02)
            self.assertEqual(len(pids), 2)
            for pid in pids:  # IDs published by this test's own fixed fixture, not a process search.
                handle = kernel.OpenProcess(0x101000, False, pid)  # SYNCHRONIZE|QUERY_LIMITED_INFORMATION only.
                self.assertTrue(handle)
                handles.append(handle)
                image, size = C.create_unicode_buffer(32768), W.DWORD(32768)
                self.assertTrue(kernel.QueryFullProcessImageNameW(handle, 0, image, C.byref(size)))
                self.assertEqual(os.path.normcase(image.value), os.path.normcase(sys.executable))
                creation, exited, kern, user = (C.c_ulonglong() for _ in range(4))
                self.assertTrue(kernel.GetProcessTimes(handle, C.byref(creation), C.byref(exited),
                                                      C.byref(kern), C.byref(user)))
                self.assertGreater(creation.value, 0)
                identities.append({"pid": pid, "creation_filetime": creation.value, "image": image.value})
            (run / "owner-exit").write_bytes(b"owned CPU test trigger")
            self.assertEqual(owner.wait(timeout=5), 31)
            waits = [kernel.WaitForSingleObject(handle, 3000) for handle in handles]
            self.assertEqual(waits, [0, 0])
            self.assertIsNone(sentinel.poll())
            self.assertFalse((run / "receipt.json").exists())  # Owner did not execute its normal finally.
            (folder / "owner-crash-observation.json").write_text(json.dumps({
                "owner_pid": owner.pid, "owner_exit_code": 31, "fixed_fixture_processes": identities,
                "wait_results": waits, "unrelated_sentinel_alive": True, "Job_active_after_crash": None,
                "GPU_model": "NOT_RUN"}, indent=2))
        finally:
            for process in (owner, sentinel):
                if process.poll() is None:
                    process.terminate()  # Only Popen handles created in this test.
                process.wait(timeout=5)
            for handle in handles:
                kernel.CloseHandle(handle)

    def run_fixture(self, scenario="ok", **limits):
        target = destination()
        result = pilot.run_cpu_fixture(configuration(), target, pilot.Budget(**limits), scenario)
        saved = json.loads((target / "receipt.json").read_text())
        self.assertEqual(result, saved)
        self.assertTrue(result["contained_before_resume"])
        self.assertEqual(result["job_active_after"], 0)
        self.assertTrue(result["termination_confirmed"])
        self.assertGreater(result["root_creation_filetime"], 0)
        self.assertLessEqual((target / "raw.bin").stat().st_size, limits.get("output_bytes", 1024**2))
        self.assertIsNone(result["GPU_time_ns"])
        return result, target

    def test_real_consumer_and_parent_guard_reach_fake_child(self):
        result, target = self.run_fixture()
        self.assertEqual(result["status"], "FINISHED")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn(b"GUARD=1", (target / "raw.bin").read_bytes())
        self.assertGreater(result["peak_job_commit_bytes"], 0)
        self.assertFalse(result["model_launched"])

    def test_timeout_drains_owned_job(self):
        result, _ = self.run_fixture("sleep", timeout_s=0.3)
        self.assertEqual(result["status"], "TIMEOUT")

    def test_output_is_strictly_bounded(self):
        result, _ = self.run_fixture("flood", output_bytes=4096)
        self.assertEqual(result["status"], "OUTPUT_LIMIT")
        self.assertEqual(result["written_bytes"], 4096)
        self.assertGreater(result["observed_dropped_bytes"], 0)

    def test_grandchild_and_unrelated_same_executable(self):
        sentinel = subprocess.Popen([sys.executable, "-I", "-B", "-c", "import time; time.sleep(20)"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            result, _ = self.run_fixture("grandchild")
            self.assertEqual(result["status"], "FINISHED")
            self.assertGreaterEqual(result["job_total_processes"], 2)
            self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate()  # This test created this exact handle; never match by PID/name.
            sentinel.wait(timeout=5)

    def test_job_commit_limit_denies_fixture_allocation(self):
        result, target = self.run_fixture("memory", job_commit_bytes=96 * 1024**2)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn(b"MEM_DENIED", (target / "raw.bin").read_bytes())
        self.assertEqual(result["configured_job_commit_bytes"], 96 * 1024**2)
        exceeded = result["peak_job_commit_bytes"] > result["configured_job_commit_bytes"]
        self.assertEqual(result["commit_budget_exceeded"], exceeded)
        if exceeded:
            self.assertEqual(result["status"], "COMMIT_BUDGET_EXCEEDED")

    def test_process_count_limit_denies_grandchild(self):
        result, target = self.run_fixture("grandchild", max_processes=1)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn(b"PROCESS_DENIED", (target / "raw.bin").read_bytes())
        # TotalProcesses is lifetime accounting, including a denied creation;
        # it is not the simultaneous active-process limit.
        self.assertGreaterEqual(result["job_total_processes"], 1)

    def test_resource_query_failure_aborts_without_resume(self):
        from unittest.mock import patch
        target = destination()
        with patch.object(pilot, "system_resources", side_effect=OSError("fixture telemetry unavailable")):
            result = pilot.run_cpu_fixture(configuration(), target, pilot.Budget(), "ok")
        self.assertEqual(result["status"], "PREFLIGHT_BLOCKED")
        self.assertFalse(result["contained_before_resume"])
        self.assertIsNone(result["owned_root_pid"])

    def test_post_run_source_read_failure_cannot_report_success(self):
        from unittest.mock import patch
        actual_open = Path.open
        reads = 0
        def fail_second_fixture_read(path, *args, **kwargs):
            nonlocal reads
            if path.name == "cpu_child.py" and args and args[0] == "rb":
                reads += 1
                if reads == 2:
                    raise OSError("injected post-run read failure")
            return actual_open(path, *args, **kwargs)
        target = destination()
        with patch.object(Path, "open", fail_second_fixture_read):
            result = pilot.run_cpu_fixture(configuration(), target, pilot.Budget())
        self.assertEqual(reads, 2)
        self.assertEqual(result["status"], "SOURCE_VERIFICATION_FAILED")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertFalse(result["output_complete"])
        self.assertTrue(result["termination_confirmed"])
        self.assertEqual(result["job_active_after"], 0)

    def test_preflight_pressure_cannot_start_child(self):
        from unittest.mock import patch
        target = destination()
        with patch.object(pilot, "system_resources", return_value={"available_ram": 0, "commit_headroom": 0, "H_free": 0}):
            result = pilot.run_cpu_fixture(configuration(), target, pilot.Budget())
        self.assertEqual(result["status"], "PREFLIGHT_BLOCKED")
        self.assertIsNone(result["owned_root_pid"])
        self.assertFalse((target / "raw.bin").exists())

    def test_monitor_failure_and_pressure_terminate_only_owned_child(self):
        from unittest.mock import patch
        real = pilot.system_resources
        for failure in (OSError("fixture"), {"available_ram": 0, "commit_headroom": 0, "H_free": 0}):
            target = destination()
            with patch.object(pilot, "system_resources", side_effect=[real(target.parent), failure]):
                result = pilot.run_cpu_fixture(configuration(), target, pilot.Budget(), "sleep")
            self.assertIn(result["status"], ("CONTROL_ERROR", "RESOURCE_PRESSURE"))
            self.assertTrue(result["contained_before_resume"])
            self.assertTrue(result["termination_confirmed"])
            self.assertEqual(result["job_active_after"], 0)

    def test_membership_or_resume_failure_still_drains_suspended_child(self):
        from tools.perf131k import windows_job
        from unittest.mock import patch
        real_api = windows_job.api
        for failing in ("IsProcessInJob", "ResumeThread"):
            kernel = real_api()
            setattr(kernel, failing, lambda *args: 0 if failing == "IsProcessInJob" else 0xFFFFFFFF)
            target = destination()
            with patch.object(windows_job, "api", return_value=kernel):
                result = pilot.run_cpu_fixture(configuration(), target, pilot.Budget())
            self.assertEqual(result["status"], "CONTROL_ERROR")
            self.assertEqual(result["job_active_after"], 0)
            self.assertTrue(result["termination_confirmed"])
            self.assertEqual((target / "raw.bin").read_bytes(), b"")

    def test_cli_is_connected_to_cpu_only_consumer(self):
        folder = destination()
        folder.mkdir(parents=True)
        cfg = folder / "config.json"
        cfg.write_text(json.dumps(configuration()), encoding="utf-8")
        command = [sys.executable, "-B", "-m", "tools.perf131k.pilot", str(cfg), str(folder / "run"),
                   "--cpu-fixture", "ok"]
        done = subprocess.run(command, timeout=10, capture_output=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(done.returncode, 0, done.stderr.decode(errors="replace"))
        receipt = json.loads((folder / "run/receipt.json").read_text())
        self.assertEqual(receipt["status"], "FINISHED")
        self.assertEqual(receipt["job_active_after"], 0)
        self.assertFalse(receipt["model_launched"])
        # The same CLI has no real-model selector; unknown scenarios fail before creating a run.
        command[-1] = "real-model"
        done = subprocess.run(command, timeout=10, capture_output=True)
        self.assertNotEqual(done.returncode, 0)


if __name__ == "__main__":
    unittest.main()
