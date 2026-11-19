"""CPU-only checks for missing evidence, privacy and passive collector boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/qwen35/q36_collect_environment.py"
SPEC = importlib.util.spec_from_file_location("collector", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def test_streamed_file_hash_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.exe"
            path.write_bytes(b"binary fixture")
            record = collector.file_identity(path)["value"]
            self.assertEqual(record["sha256"], hashlib.sha256(b"binary fixture").hexdigest())
            self.assertTrue(Path(record["path"]).is_absolute())
            self.assertIsNone(collector.file_identity(path.with_suffix(".missing"))["value"])

    def test_cache_omits_secrets_and_retains_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "CMakeCache.txt"
            path.write_text("API_TOKEN:STRING=never-output-this\nCMAKE_CUDA_FLAGS:STRING=-lineinfo\nCMAKE_CUDA_ARCHITECTURES:STRING=120\n", encoding="utf-8")
            result = collector.cmake_cache(path)
            self.assertNotIn("never-output-this", json.dumps(result))
            self.assertEqual(result["settings"]["value"]["CMAKE_CUDA_ARCHITECTURES"], "120")
            self.assertEqual(result["settings"]["value"]["CMAKE_CUDA_FLAGS"], "-lineinfo")

    def test_no_gpu_commands_or_invented_device_properties(self):
        calls = []

        def fake_run(argv, cwd=None, hash_output=False):
            calls.append(argv)
            return collector.observation(reason="fixture unavailable")

        with tempfile.TemporaryDirectory() as folder, patch.object(collector, "run", fake_run), patch.object(collector, "windows_inventory", return_value=collector.observation(reason="fixture")), patch.object(collector.shutil, "which", return_value=None):
            result = collector.collect_environment(folder)
        self.assertEqual(result["schema_version"], 1)
        for field in result["cuda"].values():
            self.assertIsNone(field["value"])
            self.assertEqual(field["status"], "NOT_RUN")
            self.assertTrue(field["reason"])
        self.assertEqual(result["loaded_dlls"]["status"], "NOT_RUN")
        self.assertTrue(all(argv[0] == "git" for argv in calls))

    def test_source_diff_is_hashed_never_serialised(self):
        digest = hashlib.sha256(b"secret diff text").hexdigest()
        with patch.object(collector, "run", side_effect=[collector.observation("a" * 40), collector.observation(" M file"), collector.observation(digest)]):
            result = collector.source_identity(Path.cwd())
        self.assertTrue(result["tracked_dirty"]["value"])
        self.assertEqual(result["tracked_diff_sha256"]["value"], digest)
        self.assertNotIn("secret diff text", json.dumps(result))

    def test_static_cuda_default_is_read_from_compiler_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            meta = Path(folder) / "CMakeFiles" / "4.4.0"
            meta.mkdir(parents=True)
            (meta / "CMakeCUDACompiler.cmake").write_text('set(CMAKE_CUDA_COMPILER_VERSION "13.5.37")\nset(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")\n')
            value = collector.compiler_metadata(folder)["value"]
        self.assertEqual(value["CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT"], "STATIC")
        self.assertEqual(value["CMAKE_CUDA_COMPILER_VERSION"], "13.5.37")

    def test_no_pid_is_not_empty_loaded_dll_success(self):
        self.assertIsNone(collector.loaded_modules(None)["value"])
        with patch.object(collector.platform, "system", return_value="Windows"):
            with self.assertRaises(ValueError):
                collector.loaded_modules("1; anything")


if __name__ == "__main__":
    unittest.main()
