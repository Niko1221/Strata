"""CPU allocation fault tests using verbatim production consumers; no GPU/model load."""
import json
import os
from pathlib import Path
import subprocess
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[2]


def function(text, signature):
    start = text.index(signature)
    opening = text.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@unittest.skipUnless(os.environ.get("P01_DIAGNOSTICS_ARTIFACTS"), "local CPU build receipt and output directory required")
class StartupLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(os.environ["P01_DIAGNOSTICS_ARTIFACTS"]) / uuid.uuid4().hex
        cls.base.mkdir()
        source = (ROOT / "src/core/expert_source.cpp").read_text(encoding="utf-8")
        claim = function(source, "bool FileExpertSource::claim_stage(")
        todo = function(source, "void FileExpertSource::claim_todo(")
        workers = function(source, "template<class Work> void run_fill_workers(")
        complement_workers = function(source, "template<class Work> void run_complement_workers(")
        complement_failure = function(source, "template<class Release> bool complement_copy_failed(")
        fill = function(source, "bool FileExpertSource::fill_stage(")
        publish = function(source, "void FileExpertSource::publish_stage(")
        stage_free = function(source, "void FileExpertSource::StageBufFree::operator()(")
        direct = function(source, "bool FileExpertSource::read_direct(")
        direct = direct[:direct.index("#elif defined(__linux__)")] + "#endif\n}"
        generate = (ROOT / "src/program/generate.cpp").read_text(encoding="utf-8")
        grow_guard = function(generate, "if (!xcache.grow(")
        terminal_guard = next(line.strip() for line in generate.splitlines() if "if (xcache.mapped_bytes() < 0) return 1;" in line)
        template = (ROOT / "tests/perf131k/startup_lifetime_cpu.cpp").read_text(encoding="utf-8")
        cpp = cls.base / "startup.cpp"
        cpp.write_text(template.replace("// @CLAIM@", claim).replace("// @DIRECT@", direct).replace("// @WORKERS@", workers).replace("// @COMP_WORKERS@", complement_workers).replace("// @COMP_FAILURE@", complement_failure).replace("// @FILL@", fill).replace("// @PUBLISH@", publish).replace("// @STAGE_FREE@", stage_free).replace("// @TODO@", todo).replace("// @VRAM_GROW@", grow_guard).replace("// @VRAM_TERMINAL@", terminal_guard), encoding="utf-8")
        receipt_path = Path(os.environ.get("P01_CPU_COMPILE_RECEIPT", "H:/Strata/evidence/p01-b0-20261025/build/compile-expert_source.json"))
        receipt = json.loads(receipt_path.read_text())
        compiler = Path(receipt["command"][0])
        env = dict(os.environ, **{k: receipt["environment"][k] for k in ("INCLUDE", "LIB")})
        env["PATH"] = str(compiler.parent) + os.pathsep + env["PATH"]
        cls.exe = cls.base / "startup.exe"
        command = [str(compiler), "/nologo", "/std:c++20", "/EHsc", "/Od", "/I" + str(ROOT / "include"), str(cpp),
                   "/Fe:" + str(cls.exe), "/Fo:" + str(cls.base / "startup.obj"), "/link", "/INCREMENTAL:NO"]
        (cls.base / "command.json").write_text(json.dumps({"command": command, "scope": "CPU extracted consumers with fake APIs"}), encoding="utf-8")
        result = subprocess.run(command, cwd=cls.base, env=env, capture_output=True, timeout=30)
        (cls.base / "compile.txt").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise AssertionError(str(cls.base / "compile.txt"))
        cls.joined_exe = cls.base / "joined_prefetch.exe"
        joined_source = ROOT / "tests/perf131k/joined_prefetch_cpu.cpp"
        command = [str(compiler), "/nologo", "/std:c++20", "/EHsc", "/Od", "/I" + str(ROOT / "include"),
                   str(joined_source), "/Fe:" + str(cls.joined_exe),
                   "/Fo:" + str(cls.base / "joined_prefetch.obj"), "/link", "/INCREMENTAL:NO"]
        (cls.base / "joined-command.json").write_text(json.dumps({"command": command,
            "scope": "real CPU threads and production JoinedPrefetch; no model/GPU"}), encoding="utf-8")
        result = subprocess.run(command, cwd=cls.base, env=env, capture_output=True, timeout=30)
        (cls.base / "joined-compile.txt").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise AssertionError(str(cls.base / "joined-compile.txt"))

    def test_joined_prefetch_releases_tls_and_handles_launch_and_task_failures(self):
        result = subprocess.run([str(self.joined_exe)], capture_output=True, timeout=10)
        (self.base / "joined-run.txt").write_bytes(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        generate = (ROOT / "src/program/generate.cpp").read_text(encoding="utf-8")
        profile = generate.split("// ---- R4.2e: fill the tier from the profile.", 1)[1].split(
            "int64_t prefilled", 1)[1].split("const double fill_s", 1)[0]
        self.assertIn("strata::program::JoinedPrefetch ahead;", profile)
        self.assertNotIn("std::async", profile)
        self.assertLess(profile.index("auto read_batch"), profile.index("JoinedPrefetch ahead"))

    def run_case(self, args, pin):
        env = dict(os.environ, STRATA_STAGE_PIN=str(pin))
        result = subprocess.run([str(self.exe), *args], env=env, capture_output=True, timeout=5)
        (self.base / ("-".join(args) + "-pin" + str(pin) + ".txt")).write_bytes(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertIn(b"PASS", result.stdout)

    def test_stage_growth_and_reuse_survive_each_allocation_failure(self):
        for pin in (0, 1):
            with self.subTest(pin=pin):
                self.run_case(["stage"], pin)

    def test_direct_reads_never_allocate_with_requests_in_flight(self):
        for point in range(16):
            with self.subTest(point=point):
                self.run_case(["direct", str(point)], 0)
        self.run_case(["direct"], 0)

    def test_failed_fills_and_busy_slots_retain_capacity_until_destruction(self):
        for pin in (0, 1):
            with self.subTest(pin=pin):
                self.run_case(["retention"], pin)

    def test_partial_worker_creation_joins_and_failed_fills_publish(self):
        self.run_case(["workers"], 0)

    def test_complement_worker_exception_joins_before_failure(self):
        self.run_case(["complement"], 0)

    def test_prefetch_preclaim_allocation_failure_keeps_demand_retry_available(self):
        self.run_case(["todo"], 0)

    def test_vram_cleanup_pending_stops_publication_and_next_request(self):
        self.run_case(["vram"], 0)
