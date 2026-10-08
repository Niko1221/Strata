"""ARM CPU detection regression: no GPU or downloads required."""
import io
import json
import tempfile
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class CpuDetection(unittest.TestCase):
    def test_arm_build_command_limits_jobs_by_available_ram(self):
        calls = []
        with mock.patch.object(setup, "WIN", False), \
             mock.patch.object(setup, "arm64_host", return_value=True), \
             mock.patch.object(setup.os, "cpu_count", return_value=12), \
             mock.patch("builtins.open", return_value=io.StringIO("MemAvailable: 2097152 kB\n")), \
             mock.patch.object(setup, "find_tool", side_effect=lambda name: name), \
             mock.patch.object(setup, "run", side_effect=lambda cmd, **kw: calls.append(cmd) or mock.Mock(returncode=0)):
            setup.cmake_build("source", "build", "strata", [], None, "unused.bat")
        self.assertEqual(calls[1][-2:], ["-j", "1"])

    def test_arm_build_memory_and_cpu_limits(self):
        with mock.patch.object(setup, "WIN", False), \
             mock.patch.object(setup, "arm64_host", return_value=True), \
             mock.patch.object(setup.os, "cpu_count", return_value=12):
            for text, want in [("MemAvailable: 8388608 kB\n", 3),
                               ("MemAvailable: 29360128 kB\n", 6),
                               ("MemAvailable: 0 kB\n", 1),
                               ("MemTotal: 33554432 kB\n", 1)]:
                with self.subTest(text=text), mock.patch("builtins.open", return_value=io.StringIO(text)):
                    self.assertEqual(setup.build_jobs(), want)

    def test_desktop_build_parallelism_is_preserved(self):
        with mock.patch.object(setup, "WIN", False), \
             mock.patch.object(setup, "arm64_host", return_value=False), \
             mock.patch.object(setup.os, "cpu_count", return_value=12):
            self.assertEqual(setup.build_jobs(), 6)

    def test_arm_has_a_native_backend(self):
        with mock.patch.object(setup.platform, "machine", return_value="aarch64"):
            self.assertEqual(setup.cpu_floor(False), "")

    def test_arm_never_downloads_an_x86_engine(self):
        with mock.patch.object(setup, "arm64_host", return_value=True), \
             mock.patch.object(setup, "download", side_effect=AssertionError("downloaded x86 engine")):
            self.assertIsNone(setup.get_prebuilt("https://example.invalid/", {}, "none", toolkit=12))

    def test_arm_keeps_jetpack_libraries(self):
        with mock.patch.object(setup, "arm64_host", return_value=True), \
             mock.patch.object(setup, "pip_install", side_effect=AssertionError("installed CUDA wheels")):
            setup.pip_cuda_libs(12)

    def test_arm_features_are_not_x86_flags(self):
        with mock.patch.object(setup, "WIN", False), \
             mock.patch.object(setup.platform, "processor", return_value="aarch64"), \
             mock.patch("builtins.open", return_value=io.StringIO("processor : 0\nFeatures : fp asimd atomics\n")):
            self.assertEqual(setup.cpu_info(), ("aarch64", False, False))

    def test_x86_detection_is_preserved(self):
        with mock.patch.object(setup, "WIN", False), \
             mock.patch("builtins.open", return_value=io.StringIO("model name : Test CPU\nflags : avx2 avx512f avx512bw avx512vl avx512_vnni avx512vbmi\n")):
            self.assertEqual(setup.cpu_info(), ("Test CPU", True, True))


class Function:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args):
        return self.fn(*args)


class DriverDiscovery(unittest.TestCase):
    def driver(self, integrated=1, failure=False):
        def put(out, value):
            out._obj.value = value
            return 0
        attrs = {18: integrated, 19: 1, 75: 8, 76: 7, 83: 1, 89: 0, 99: 1}
        driver = mock.Mock()
        driver.cuInit = Function(lambda flags: 1 if failure else 0)
        driver.cuDeviceGetCount = Function(lambda p: put(p, 1))
        driver.cuDriverGetVersion = Function(lambda p: put(p, 12060))
        driver.cuDeviceGet = Function(lambda p, i: put(p, i))
        def name(buf, size, dev):
            buf.value = b"Orin"
            return 0
        driver.cuDeviceGetName = Function(name)
        driver.cuDeviceTotalMem_v2 = Function(lambda p, dev: put(p, 32 << 30))
        driver.cuDeviceGetAttribute = Function(lambda p, key, dev: put(p, attrs[key]))
        return driver

    def discover(self, **kwargs):
        with mock.patch.object(setup, "arm64_host", return_value=True), \
             mock.patch.object(setup.ctypes, "CDLL", return_value=self.driver(**kwargs)), \
             mock.patch.object(setup, "ram_gb", return_value=30):
            return setup.cuda_driver_gpus()

    def test_orin_uses_one_physical_pool(self):
        g = self.discover()[0]
        self.assertEqual(g["arch"], "87")
        self.assertEqual(g["driver_api"], 12060)
        self.assertTrue(g["jetson"])
        self.assertEqual(g["shared_gb"], 30)
        self.assertEqual(g["vram_gb"], 24)
        self.assertEqual(setup.low_ram_vram(g), 0)
        self.assertFalse(g["concurrent_managed_access"])

    def test_discrete_gpu_keeps_its_memory(self):
        g = self.discover(integrated=0)[0]
        self.assertFalse(g.get("uma", False))
        self.assertEqual(g["vram_gb"], 32)

    def test_failed_driver_probe_returns_no_devices(self):
        self.assertEqual(self.discover(failure=True), [])

    def test_na_memory_is_not_invented_on_x86(self):
        with mock.patch.object(setup, "arm64_host", return_value=False), \
             mock.patch.object(setup, "out", return_value="0, GPU, [N/A], 8.7, 540.0"):
            self.assertEqual(setup.gpus(), [])


class LocalEngineArchitecture(unittest.TestCase):
    def test_gpu_vision_request_rebuilds_cpu_encoder(self):
        for stored, requested, expected in [("cpu", "gpu", ["strata-vision"]), ("gpu", "cpu", [])]:
            with self.subTest(stored=stored), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                eng = root / "engine-cuda12"
                eng.mkdir()
                (eng / setup.EXE).write_bytes(b"engine")
                (eng / setup.VEXE).write_bytes(b"encoder")
                (eng / "BUILD.json").write_text(json.dumps({"source": "local", "src": "same", "vision_src": "same",
                    "archs": [87], "cpu_arch": "arm64", "vision": stored, "toolkit": 12}))
                calls = []
                def compile_vision(source, build, target, defs, *args):
                    calls.append(target)
                    self.assertIn("-DSTRATA_VISION_CUDA=ON", defs)
                    self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=87", defs)
                    (build / "bin").mkdir(parents=True)
                    (build / "bin" / setup.VEXE).write_bytes(b"gpu encoder")
                with mock.patch.object(setup, "ROOT", root), \
                     mock.patch.object(setup, "arm64_host", return_value=True), \
                     mock.patch.object(setup, "source_hash", return_value="same"), \
                     mock.patch.object(setup, "source_version", return_value="test"), \
                     mock.patch.object(setup, "install_build_tools", return_value=("/usr/local/cuda-12/bin/nvcc", None)), \
                     mock.patch.object(setup, "cmake_build", side_effect=compile_vision):
                    setup.build_engine({"arch": 87, "jetson": True}, requested, True, root / "llama")
                self.assertEqual(calls, expected)
                self.assertEqual(json.loads((eng / "BUILD.json").read_text())["vision"], "gpu")

    def test_arm_rebuilds_an_engine_with_missing_or_x86_metadata(self):
        for recorded in (None, "x86_64", "arm64"):
            with self.subTest(recorded=recorded), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                eng = root / "engine-cuda12"
                eng.mkdir()
                (eng / setup.EXE).write_bytes(b"old")
                meta = {"source": "local", "src": "same", "archs": [87]}
                if recorded:
                    meta["cpu_arch"] = recorded
                (eng / "BUILD.json").write_text(json.dumps(meta))
                calls = []
                def compile_engine(source, build, target, defs, *args):
                    calls.append(target)
                    build.mkdir(exist_ok=True)
                    (build / setup.EXE).write_bytes(b"arm engine")
                with mock.patch.object(setup, "ROOT", root), \
                     mock.patch.object(setup, "arm64_host", return_value=True), \
                     mock.patch.object(setup, "source_hash", return_value="same"), \
                     mock.patch.object(setup, "source_version", return_value="test"), \
                     mock.patch.object(setup, "install_build_tools", return_value=("/usr/local/cuda-12/bin/nvcc", None)), \
                     mock.patch.object(setup, "cmake_build", side_effect=compile_engine), \
                     mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"):
                    setup.build_engine({"arch": "87", "jetson": True}, "none", True, root)
                self.assertEqual(calls, [] if recorded == "arm64" else ["strata"])
                self.assertEqual(json.loads((eng / "BUILD.json").read_text())["cpu_arch"], "arm64")


class Installation(unittest.TestCase):
    def test_orin_configuration_uses_local_cuda12_and_file_backing(self):
        from tools.test_setup_golden import install, argv_for
        gpu = {"index": 0, "name": "Orin", "arch": "87", "driver": "CUDA API 12.6", "driver_api": 12060,
               "vram_gb": 24, "uma": True, "dedicated_gb": 0, "shared_gb": 30, "jetson": True}
        builds = []
        def build(gpu, vision, yes, llama, toolkit=None):
            builds.append((gpu, vision, toolkit))
            eng = setup.engine_dir(toolkit)
            eng.mkdir(parents=True, exist_ok=True)
            (eng / setup.EXE).write_bytes(b"")
            (eng / "BUILD.json").write_text('{"source":"local","toolkit":12,"archs":[87]}')
            return eng
        for vision, reserve in [(None, None), ("gpu", None), ("gpu", 1024)]:
            builds.clear()
            argv = argv_for("coder", "IQ1_M")
            if vision: argv += ["--vision", vision]
            if reserve: argv += ["--vram-reserve-mib", str(reserve)]
            with self.subTest(vision=vision, reserve=reserve), mock.patch.dict(os.environ, {"STRATA_CUDA": ""}):
                code, out, cfg, _ = install(30, [gpu], argv, extra=[
                    mock.patch.object(setup, "arm64_host", return_value=True),
                    mock.patch.object(setup, "build_engine", side_effect=build),
                    mock.patch.object(setup, "pcie_link", return_value=None),
                ])
            self.assertEqual(code, 0, out)
            self.assertEqual(builds[0][2], 12)
            self.assertEqual(cfg["cuda"], 12)
            self.assertIn("--mmap-experts", cfg["args"])
            args = cfg["args"]
            self.assertLessEqual(int(args[args.index("--max-context") + 1]), 32768)
            if vision:
                self.assertEqual(int(args[args.index("--vram-reserve-mib") + 1]), reserve or 3072)


if __name__ == "__main__":
    unittest.main()
