"""CPU diagnostics verification, including real contained fake children; no model/CUDA imports."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from tools.perf131k import windows_job as job
from tools.perf131k import native_pilot as native

ROOT = Path(__file__).resolve().parents[2]


def load_override(name, module, key):
    if not os.environ.get(key):
        return module
    spec = importlib.util.spec_from_file_location(name, os.environ[key])
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


job = load_override("job_diagnostics_under_test", job, "P01_JOB_UNDER_TEST")
native = load_override("native_diagnostics_under_test", native, "P01_NATIVE_UNDER_TEST")


def artifacts():
    base = Path(os.environ["P01_DIAGNOSTICS_ARTIFACTS"]) / uuid.uuid4().hex
    base.mkdir()
    return base


class DiagnosticsTests(unittest.TestCase):
    def child(self, enabled=False, reader=None, *, reader_kind="injected", timeout_s=2, expected_status="FINISHED"):
        base = artifacts()
        budget = SimpleNamespace(timeout_s=timeout_s, job_commit_bytes=128 * 1024**2, max_processes=1,
                                 output_bytes=4096, min_available_ram=0, min_commit_headroom=0, min_H_free=0)
        n = 0
        def sample(_):
            nonlocal n
            n += 1
            return {"available_ram": 10000-n, "commit_headroom": 20000-n, "H_free": 30000-n}
        with patch.object(job, "process_memory_reader", return_value=reader) as factory:
            result = job.run_owned([sys.executable, "-I", "-c", "import time; time.sleep(.15)"],
                                   base, dict(os.environ), base / "raw.bin", budget, sample,
                                   memory_diagnostics=enabled)
        (base / "child-result.json").write_text(json.dumps({"test": self.id(), "budget": vars(budget),
            "reader_kind": reader_kind if enabled else "disabled",
            "system_resources_kind": "scripted_CPU_fixture_not_machine_telemetry",
            "result": result}, indent=2), encoding="utf-8")
        expected_exit = 0 if expected_status == "FINISHED" else 124
        self.assertEqual((result["status"], result["exit_code"], result["job_active_after"]), (expected_status, expected_exit, 0))
        self.assertTrue(result["termination_confirmed"])
        self.assertEqual(factory.call_count, int(enabled))
        return result

    def test_default_off_never_constructs_reader(self):
        result = self.child()
        self.assertNotIn("memory_diagnostics", result)

    def test_opt_in_uses_owned_handle_and_bounded_aggregates(self):
        handles = []
        def reader(handle):
            handles.append(handle)
            n = len(handles)
            return {"private_bytes": 1000+n, "working_set_bytes": 2000+n, "error_code": None}
        result = self.child(True, reader)
        d = result["memory_diagnostics"]
        self.assertGreater(d["samples"], 1)
        self.assertEqual(len(set(handles)), 1)
        self.assertEqual(len(handles), result["resource_samples"])
        self.assertEqual(d["scope"], "owned_root_process_sampled_not_job_peak")
        self.assertEqual(d["status"], "AVAILABLE")
        self.assertEqual(d["max_private_bytes"], 1000+len(handles))
        self.assertEqual(d["max_working_set_bytes"], 2000+len(handles))
        self.assertEqual(d["min_available_ram"], 10000-len(handles))
        self.assertEqual(d["min_commit_headroom"], 20000-len(handles))
        self.assertEqual(d["min_H_free"], 30000-len(handles))
        self.assertLess(len(json.dumps(d)), 1500)

    def test_failed_counters_remain_unknown_and_do_not_abort_child(self):
        result = self.child(True, lambda _: {"private_bytes": None, "working_set_bytes": None, "error_code": 5})
        d = result["memory_diagnostics"]
        self.assertEqual(d["status"], "UNKNOWN")
        self.assertIsNone(d["max_private_bytes"])
        self.assertIsNone(d["max_working_set_bytes"])
        self.assertEqual(d["failed_samples"], d["samples"])
        self.assertEqual(d["last_error_code"], 5)

    def test_partial_samples_keep_zero_as_known_and_failures_as_unknown(self):
        count = 0
        def reader(_):
            nonlocal count
            count += 1
            if count == 1:
                return {"private_bytes": 0, "working_set_bytes": 0, "error_code": None}
            raise OSError("injected unavailable memory query")
        d = self.child(True, reader)["memory_diagnostics"]
        self.assertEqual(d["status"], "PARTIAL")
        self.assertEqual(d["max_private_bytes"], 0)
        self.assertEqual(d["max_working_set_bytes"], 0)
        self.assertEqual(d["failed_samples"], d["samples"]-1)
        self.assertIsNone(d["last_error_code"])

    def test_unsupported_query_and_failed_win32_call_report_unknown(self):
        with patch.object(job.C, "WinDLL", side_effect=OSError("unavailable")):
            counters = job.process_memory_reader()(None)
        self.assertEqual(counters, {"private_bytes": None, "working_set_bytes": None, "error_code": None})
        # This call cannot affect another process: NULL is not an owned process handle.
        counters = job.process_memory_reader()(None)
        self.assertIsNone(counters["private_bytes"])
        self.assertIsNone(counters["working_set_bytes"])
        self.assertNotEqual(counters["error_code"], 0)

    def test_invalid_diagnostic_option_is_rejected_before_process_creation(self):
        for value in (1, "true", None):
            with self.subTest(value=value), patch.object(job, "api") as init, self.assertRaises(ValueError):
                job.run_owned(None, None, None, None, None, None, memory_diagnostics=value)
            init.assert_not_called()

    def test_real_process_memory_reader_with_owned_child(self):
        reader = job.process_memory_reader()
        d = self.child(True, reader, reader_kind="real_win32")["memory_diagnostics"]
        self.assertEqual(d["status"], "AVAILABLE")
        self.assertGreater(d["max_private_bytes"], 0)
        self.assertGreater(d["max_working_set_bytes"], 0)

    def test_enabled_diagnostics_preserve_timeout_cleanup(self):
        result = self.child(True, job.process_memory_reader(), reader_kind="real_win32",
                            timeout_s=.03, expected_status="TIMEOUT")
        self.assertGreater(result["memory_diagnostics"]["samples"], 0)

    def test_native_consumer_option_reaches_owned_runner(self):
        # Reuse the existing bound-token fixture and scripted native protocol, never the model.
        from tests.perf131k.test_native_pilot import NativePilotTests
        harness = NativePilotTests()
        for enabled in (False, True):
            observed = {}
            code, _, calls = harness.fake_cell(memory_diagnostics=enabled, observed=observed)
            self.assertEqual((code, calls), (0, 1))
            self.assertEqual(observed["diagnostics"].get("memory_diagnostics", False), enabled)
            self.assertEqual(observed["launch"]["memory_diagnostics"], enabled)

    def test_cli_explicit_flag_reaches_consumer_without_launch(self):
        with patch.object(native, "run_cell", return_value=0) as consume, patch.object(sys, "argv", [
                "pilot", "--execute-authorized", "--tokens", "4096", "--trace", "OFF", "--binding-sha", "a"*64,
                "--memory-diagnostics"]):
            self.assertEqual(native.main(), 0)
        self.assertIs(consume.call_args.kwargs["memory_diagnostics"], True)

    def test_actual_cpp_failure_paths_with_cpu_stubs(self):
        base = artifacts()
        source = Path(os.environ.get("P01_CACHE_UNDER_TEST") or ROOT / "src/core/expert_cache.cpp").read_text(encoding="utf-8")
        def function(text, signature):
            start = text.index(signature)
            opening = text.index("{", start)
            depth = 1
            end = opening+1
            while depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            return text[start:end]
        functions = "\n".join(function(source, signature) for signature in (
            "bool map_segment(", "bool ExpertCache::open_segmented(", "void ExpertCache::release_segmented(",
            "int64_t ExpertCache::mapped_bytes(", "bool ExpertCache::shrink(", "bool ExpertCache::grow(",
            "uint8_t* ExpertCache::device_slot(", "const uint8_t* ExpertCache::device_slot("))
        generate = Path(os.environ.get("P01_GENERATE_UNDER_TEST") or ROOT / "src/program/generate.cpp").read_text(encoding="utf-8")
        start = generate.index("const uint8_t* b = srcp->blob_stable(profile[(size_t) i].first")
        end = generate.index("        fill_bytes +=", start)
        template = (ROOT / "tests/perf131k/diagnostics_cpu.cpp").read_text(encoding="utf-8")
        cpp = base / "diagnostics.cpp"
        valid = function((ROOT / "include/strata/core/expert_cache.hpp").read_text(encoding="utf-8"), "bool valid() const")
        shrink_start = generate.index("const bool shrunk = xcache.shrink(keep, e);")
        shrink_end = generate.index("            } else if (mapped < xcache.full_bytes())", shrink_start)
        guard = "if (xcache.mapped_bytes() < 0) return 1;"
        self.assertIn(guard, generate)
        cpp.write_text(template.replace("// @CACHE_FUNCTIONS@", functions).replace("// @CACHE_VALID@", valid)
                       .replace("// @SHRINK_CALLER@", generate[shrink_start:shrink_end])
                       .replace("// @REQUEST_GUARD@", guard)
                       .replace("// @GENERATE_BLOCK@", generate[start:end]), encoding="utf-8")
        receipt = json.loads(Path("H:/Strata/evidence/p01-b0-20261025/build/compile-expert_source.json").read_text())
        compiler = Path(receipt["command"][0])
        env = dict(os.environ, **{k: receipt["environment"][k] for k in ("INCLUDE", "LIB")})
        env["PATH"] = str(compiler.parent) + os.pathsep + env["PATH"]
        exe = base / "diagnostics.exe"
        command = [str(compiler), "/nologo", "/std:c++17", "/EHsc", "/Od", "/I"+str(ROOT / "include"), str(cpp),
                   "/Fe:"+str(exe), "/Fo:"+str(base / "diagnostics.obj"), "/link", "/INCREMENTAL:NO"]
        (base / "command.json").write_text(json.dumps(command), encoding="utf-8")
        import hashlib
        (base / "toolchain.json").write_text(json.dumps({
            "compiler": str(compiler), "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
            "source_receipt": "H:/Strata/evidence/p01-b0-20261025/build/compile-expert_source.json",
            "INCLUDE": env["INCLUDE"], "LIB": env["LIB"],
            "scope": "extracted consumer functions with CPU stubs; not product compilation"}), encoding="utf-8")
        with (base / "compile.txt").open("xb") as out:
            code = subprocess.run(command, cwd=base, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=20).returncode
        self.assertEqual(code, 0, str(base / "compile.txt"))
        codes = []
        for label, args in (("vmm", []), ("blob", ["blob-only"])):
            with (base / f"run-{label}.txt").open("xb") as out:
                codes.append(subprocess.run([str(exe), *args], cwd=base, stdout=out,
                                           stderr=subprocess.STDOUT, timeout=5).returncode)
        self.assertEqual(codes, [0, 0], str(base))


if __name__ == "__main__":
    unittest.main()
