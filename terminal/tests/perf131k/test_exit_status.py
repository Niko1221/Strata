"""Owned CPU children only; distinguish observed exit from controller classification/cleanup."""
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from tools.perf131k import windows_job as job

if os.environ.get("P01_JOB_UNDER_TEST"):
    spec = importlib.util.spec_from_file_location("exit_job_under_test", os.environ["P01_JOB_UNDER_TEST"])
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)


class ExitStatusTests(unittest.TestCase):
    def child(self, code=7, *, scenario="exit", peak_over=False):
        folder = Path(os.environ["P01_DIAGNOSTICS_ARTIFACTS"]) / uuid.uuid4().hex
        folder.mkdir()
        budget = SimpleNamespace(timeout_s=.06 if scenario=="timeout" else 2,
            job_commit_bytes=128*1024**2, max_processes=1, output_bytes=4096,
            min_available_ram=0, min_commit_headroom=0, min_H_free=0)
        program = "import ctypes; ctypes.windll.kernel32.ExitProcess(ctypes.c_uint(%d))" % code
        if scenario in ("timeout", "pressure"):
            program = "import time; time.sleep(10)"
        kernel = job.api()
        original_query = kernel.QueryInformationJobObject
        def query(handle, kind, output, length, returned):
            ok = original_query(handle, kind, output, length, returned)
            if ok and kind==9 and peak_over:
                C.cast(output, C.POINTER(job.Limits)).contents.peak_job = budget.job_commit_bytes+1
            if scenario=="cleanup_failure" and kind==1:
                C.set_last_error(5)
                return 0
            return ok
        kernel.QueryInformationJobObject = query
        reads = []
        original_exit = kernel.GetExitCodeProcess
        def exit_query(*args):
            reads.append(1)
            if scenario=="exit_query_failure":
                C.set_last_error(5)
                return 0
            return original_exit(*args)
        kernel.GetExitCodeProcess = exit_query
        resources = {"available_ram": 1024**3, "commit_headroom": 2*1024**3, "H_free": 8*1024**3}
        if scenario=="pressure":
            resources["available_ram"] = -1  # Explicit fixture to exercise the existing guard.
        with patch.object(job, "api", return_value=kernel):
            result = job.run_owned([sys.executable, "-I", "-c", program], folder, dict(os.environ),
                                   folder/"raw.bin", budget, lambda _: resources)
        (folder/"child-result.json").write_text(json.dumps({"test":self.id(),"result":result,
            "budget":vars(budget),"resource_kind":"scripted_CPU_fixture",
            "peak_counter_kind":"INJECTED_OVER_LIMIT" if peak_over else "REAL_WIN32",
            "exit_query_calls":len(reads)},indent=2),encoding="utf-8")
        if scenario!="cleanup_failure":
            self.assertTrue(result["termination_confirmed"])
            self.assertEqual(result["job_active_after"],0)
        return result,len(reads)

    def test_observed_nonzero_survives_commit_classification(self):
        result, calls = self.child(7, peak_over=True)
        self.assertEqual((result["status"],result["exit_code"]),("COMMIT_BUDGET_EXCEEDED",125))
        self.assertEqual(result["native_exit_code"],7)
        self.assertEqual(result["native_exit_code_observation"],"OBSERVED_BEFORE_CLEANUP")
        self.assertEqual(calls,1)

    def test_zero_and_unsigned_exit_are_not_treated_as_unknown(self):
        for code in (0,125,259,0xC0000005):
            with self.subTest(code=code):
                result,calls = self.child(code)
                self.assertEqual(result["native_exit_code"],code)
                self.assertEqual((result["status"],result["exit_code"]),("FINISHED",code))
                self.assertEqual(result["native_exit_code_observation"],"OBSERVED_BEFORE_CLEANUP")
                self.assertEqual(calls,1)

    def test_timeout_and_pressure_do_not_report_termination_code_as_native(self):
        for scenario,status,code in (("timeout","TIMEOUT",124),("pressure","RESOURCE_PRESSURE",125)):
            with self.subTest(scenario=scenario):
                result,calls = self.child(scenario=scenario)
                self.assertEqual((result["status"],result["exit_code"]),(status,code))
                self.assertIsNone(result["native_exit_code"])
                self.assertEqual(result["native_exit_code_observation"],"NOT_OBSERVED")
                self.assertEqual(calls,0)

    def test_query_failure_stays_unknown(self):
        result,calls = self.child(scenario="exit_query_failure")
        self.assertEqual(result["status"],"CONTROL_ERROR")
        self.assertIsNone(result["native_exit_code"])
        self.assertEqual(result["native_exit_code_observation"],"QUERY_FAILED")
        self.assertEqual(calls,1)

    def test_observed_exit_survives_cleanup_failure(self):
        result,_ = self.child(11,scenario="cleanup_failure")
        self.assertEqual((result["status"],result["exit_code"]),("CLEANUP_UNCONFIRMED",126))
        self.assertEqual(result["native_exit_code"],11)
        self.assertEqual(result["native_exit_code_observation"],"OBSERVED_BEFORE_CLEANUP")

    def test_real_cpu_commit_denial_keeps_fixture_exit(self):
        from tests.perf131k.test_pilot import LifecycleTests
        result,_ = LifecycleTests().run_fixture("memory",job_commit_bytes=96*1024**2)
        self.assertEqual(result["native_exit_code"],23)  # The CPU fixture reports its caught MemoryError.
        self.assertEqual(result["native_exit_code_observation"],"OBSERVED_BEFORE_CLEANUP")
        self.assertEqual(result["exit_code"],125 if result["commit_budget_exceeded"] else 23)

    def test_native_consumer_receipt_keeps_both_codes(self):
        from tests.perf131k.test_native_pilot import NativePilotTests
        code,result,calls = NativePilotTests().fake_cell(job_result={
            "status":"COMMIT_BUDGET_EXCEEDED","exit_code":125,"native_exit_code":7,
            "native_exit_code_observation":"OBSERVED_BEFORE_CLEANUP"})
        self.assertEqual((code,calls),(125,1))
        self.assertEqual(result["native_exit_code"],7)
        self.assertEqual(result["execution_exit_code"],125)
        self.assertEqual(result["native_exit_code_observation"],"OBSERVED_BEFORE_CLEANUP")
        self.assertFalse(result["measurement_valid"])


if __name__=="__main__":
    unittest.main()
