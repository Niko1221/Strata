"""GLM installation contract, with no GPU, build, downloads or server required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup as S
from glm_synth_gguf import write_gguf, kv_str
FIND_VCVARS = S.find_vcvars


def args(**values):
    defaults = dict(backend=None, model=S.GLM_MODEL, host=None, api_key=None, gpu=None, gpus=None,
                    check=False, context=None, data_dir=None, gguf_dir=None, models_dir=None,
                    yes=True, download_model=False, no_vision=True, port=None, browser=False, no_start=True)
    return SimpleNamespace(**(defaults | values))


class GlmSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (("ROOT", self.root), ("WIN", False)):
            self.stack.enter_context(patch.object(S, name, value))
        self.stack.enter_context(patch.object(sys, "platform", "linux"))
        self.stack.enter_context(patch.object(S, "is_wsl", return_value=False))
        self.stack.enter_context(patch.object(S, "cpu_info", return_value=("test CPU", True, False)))
        self.stack.enter_context(patch.object(S.platform, "machine", return_value="x86_64"))
        self.stack.enter_context(patch.object(S, "ram_gb", return_value=64))
        self.stack.enter_context(patch.object(S, "find_nvcc", return_value=("nvcc", (12, 8))))
        self.stack.enter_context(patch.object(S, "find_vcvars", return_value=self.root / "vcvars64.bat"))
        self.stack.enter_context(patch.object(S, "page_file_gb", return_value=16))
        self.stack.enter_context(patch.object(S.shutil, "which", return_value="g++"))
        self.stack.enter_context(patch.object(S.shutil, "disk_usage", return_value=SimpleNamespace(free=200e9)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(S, "gpus", return_value=[
            dict(index=0, arch=86, name="test NVIDIA", vram_gb=24, driver="580.95")]))
        self.stack.enter_context(patch.object(S, "data_folder", return_value=(self.root, None)))
        self.download = self.stack.enter_context(patch.object(S, "download"))
        self.ask = self.stack.enter_context(patch.object(S, "ask", return_value="n"))

    def test_yes_does_not_authorize_model_download(self):
        for windows in (False, True):
            with patch.object(S, "WIN", windows):
                self.assertEqual(S.setup_glm(args()), 0)
        self.download.assert_not_called()
        self.ask.assert_not_called()

    def test_check_is_read_only(self):
        self.assertEqual(S.setup_glm(args(check=True, download_model=True)), 0)
        self.download.assert_not_called()
        S.data_folder.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_platform_gpu_and_external_host_validation(self):
        for overrides in (dict(backend="hip"), dict(backend="sycl"), dict(gpus="0,0"), dict(host="0.0.0.0"),
                          dict(host="::", api_key="  ")):
            with self.assertRaises(SystemExit):
                S.setup_glm(args(**overrides))
        with patch.object(sys, "platform", "darwin"), self.assertRaises(SystemExit):
            S.setup_glm(args())
        with patch.object(S, "is_wsl", return_value=True), self.assertRaises(SystemExit):
            S.setup_glm(args())
        with patch.object(S, "gpus", return_value=[]), self.assertRaises(SystemExit):
            S.setup_glm(args())
        with patch.object(S, "cpu_info", return_value=("old CPU", False, False)), self.assertRaises(SystemExit):
            S.setup_glm(args())
        with patch.object(S.platform, "machine", return_value="arm64"), self.assertRaises(SystemExit):
            S.setup_glm(args())
        self.download.assert_not_called()

    def test_windows_check_requires_compatible_vs_and_is_read_only(self):
        with patch.object(S, "WIN", True), patch.object(sys, "platform", "win32"), \
             patch.object(S.shutil, "which", return_value=None):
            self.assertEqual(S.setup_glm(args(check=True, download_model=True)), 0)
            S.find_vcvars.assert_called_with((12, 8))
            with patch.object(S, "find_vcvars", return_value=None), self.assertRaises(SystemExit):
                S.setup_glm(args(check=True))
        S.data_folder.assert_not_called()
        self.download.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_windows_commit_warning(self):
        with patch.object(S, "WIN", True), patch.object(S, "page_file_gb", return_value=0), \
             patch.object(S, "warn") as warning:
            self.assertEqual(S.setup_glm(args(check=True)), 0)
        self.assertTrue(any("system commit" in call.args[0] for call in warning.call_args_list))

    def test_vs_discovery_excludes_incompatible_versions(self):
        with patch.object(Path, "exists", return_value=True), patch.object(S, "out", return_value="") as out:
            self.assertIsNone(FIND_VCVARS((12, 8)))
            self.assertIn("[16.0,18.0)", out.call_args.args[0])
            out.return_value = str(self.root / "VS2022")
            self.assertEqual(FIND_VCVARS((12, 8)), self.root / "VS2022/VC/Auxiliary/Build/vcvars64.bat")
            self.assertIsNotNone(FIND_VCVARS((13, 3)))
            self.assertIn("[16.0,19.0)", out.call_args.args[0])

    def test_external_config_requires_key(self):
        with self.assertRaises(SystemExit):
            S.write_config(self.root / "config.json", dict(host="0.0.0.0", api_key=" "))
        S.write_config(self.root / "config.json", dict(host="0.0.0.0", api_key="secret"))
        self.assertEqual(json.loads((self.root / "config.json").read_text())["host"], "0.0.0.0")

    def test_check_rejects_insufficient_ram_toolkit_compiler_and_driver(self):
        with patch.object(S, "ram_gb", return_value=29.9), self.assertRaises(SystemExit):
            S.setup_glm(args(check=True))
        with patch.object(S, "ram_gb", return_value=30):
            self.assertEqual(S.setup_glm(args(check=True)), 0)
        with patch.object(S, "find_nvcc", return_value=(None, None)), self.assertRaises(SystemExit):
            S.setup_glm(args(check=True))
        with patch.object(S.shutil, "which", return_value=None), self.assertRaises(SystemExit):
            S.setup_glm(args(check=True))
        with patch.object(S, "driver_major", return_value=500), self.assertRaises(SystemExit):
            S.setup_glm(args(check=True))
        self.download.assert_not_called()

    def test_volta_selects_cuda12_and_blackwell_needs_cuda128(self):
        with patch.object(S, "gpus", return_value=[dict(index=0, arch=70, name="V100", vram_gb=32, driver="580")]), \
             patch.object(S, "find_nvcc", return_value=("nvcc", (12, 8))) as find:
            self.assertEqual(S.setup_glm(args(check=True)), 0)
            find.assert_called_once_with(below=(13, 0))
        with patch.object(S, "gpus", return_value=[dict(index=0, arch=100, name="B200", vram_gb=96, driver="580")]), \
             patch.object(S, "find_nvcc", return_value=("nvcc", (12, 7))), self.assertRaises(SystemExit):
            S.setup_glm(args(check=True))

    def test_disk_shortage_stops_before_download(self):
        with patch.object(S.shutil, "disk_usage", return_value=SimpleNamespace(free=100e9)), \
             self.assertRaises(SystemExit):
            S.setup_glm(args(download_model=True))
        self.download.assert_not_called()

    def install_stubs(self):
        folder = self.root / "gguf"
        folder.mkdir()
        for name in S.GLM_FILES:
            write_gguf(folder / name, [("general.architecture", kv_str("glm5-next"))], [])
        eng = self.root / "engine-glm"
        eng.mkdir()
        (eng / "BUILD.json").write_text('{"cuda_dirs": []}')
        for name in ("glm_sha256_ok", "check_shards", "pip_install", "get_llama_cpp", "run"):
            value = True if name == "glm_sha256_ok" else self.root / "llama" if name == "get_llama_cpp" else None
            self.stack.enter_context(patch.object(S, name, return_value=value))
        self.stack.enter_context(patch.object(S, "build_engine", return_value=eng))
        return folder.resolve(), eng

    def test_config_and_launcher_keep_qwen_separate(self):
        folder, eng = self.install_stubs()
        qwen = self.root / "strata-IQ2_XXS.json"
        qwen.write_text("unchanged")
        self.assertEqual(S.setup_glm(args(gguf_dir=str(folder))), 0)
        cfg = json.loads((self.root / "strata-glm-maya-s-v2-iq2_xxs.json").read_text())
        self.assertEqual(cfg["exe"], str(eng / S.EXE))
        self.assertEqual(cfg["args"], ["--glm-pack", str(folder / "pack"), "--max-context", "32768"])
        self.assertEqual(cfg["tokenizer"], str(folder / "pack" / "tokenizer"))
        self.assertEqual(cfg["gpu"], [0])
        self.assertNotIn("vision", cfg)
        self.assertTrue((self.root / "run-glm-maya-s-v2-iq2_xxs.sh").exists())
        self.assertEqual(qwen.read_text(), "unchanged")
        self.download.assert_not_called()

    def test_yes_does_not_authorize_vision_download(self):
        folder, _ = self.install_stubs()
        for windows in (False, True):
            with patch.object(S, "WIN", windows), self.assertRaises(SystemExit):
                S.setup_glm(args(gguf_dir=str(folder), no_vision=False))
        self.download.assert_not_called()
        self.ask.assert_not_called()

    def test_windows_launcher_and_vision_paths(self):
        folder, eng = self.install_stubs()
        vision = self.root / "models" / "glm-maya-s-v2-vision"
        vision.mkdir(parents=True)
        for name in S.GLM_VISION_FILES:
            (vision / name).write_bytes(b"synthetic vision")
        with patch.object(S, "WIN", True), patch.object(S, "EXE", "strata.exe"), \
             patch.object(S, "VEXE", "strata-vision.exe"):
            self.assertEqual(S.setup_glm(args(gguf_dir=str(folder), no_vision=False)), 0)
        cfg_path = self.root / "strata-glm-maya-s-v2-iq2_xxs.json"
        cfg = json.loads(cfg_path.read_text())
        self.assertEqual(cfg["exe"], str(eng / "strata.exe"))
        self.assertEqual(cfg["vision"]["exe"], str(eng / "strata-vision.exe"))
        self.assertEqual(cfg["vision"]["model"], str(vision / "GLM-5.3-Flash-vocab.gguf"))
        launcher = (self.root / "run-glm-maya-s-v2-iq2_xxs.bat").read_text()
        self.assertIn(f'"{cfg_path}"', launcher)
        self.assertIn(f'"{self.root / "serve" / "server.py"}"', launcher)
        self.assertNotIn('--open', launcher)
        S.build_engine.assert_called_once_with(S.gpus()[0] | {"archs": [86]}, "gpu", True,
                                               self.root / "llama", toolkit=12, glm=True)
        self.download.assert_not_called()

    def test_explicit_download_and_vision_contract(self):
        folder, eng = self.install_stubs()
        def download_stub(url, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.name in S.GLM_FILES:
                write_gguf(target, [("general.architecture", kv_str("glm5-next"))], [])
            else:
                target.write_bytes(b"synthetic vision")
        self.download.side_effect = download_stub
        self.assertEqual(S.setup_glm(args(download_model=True, no_vision=False)), 0)
        self.assertEqual(self.download.call_count, 5)
        cfg = json.loads((self.root / "strata-glm-maya-s-v2-iq2_xxs.json").read_text())
        self.assertEqual(cfg["vision"], dict(exe=str(eng / S.VEXE),
            mmproj=str(self.root / "models" / "glm-maya-s-v2-vision" / "mmproj-GLM-5.3-Flash-F16.gguf"),
            model=str(self.root / "models" / "glm-maya-s-v2-vision" / "GLM-5.3-Flash-vocab.gguf"),
            gpu=True, no_flash_attn=True, max_tokens=4096))
        self.ask.assert_not_called()

    def test_published_hash_is_checked(self):
        folder, _ = self.install_stubs()
        with patch.object(S, "glm_sha256_ok", return_value=False), self.assertRaises(SystemExit):
            S.setup_glm(args(gguf_dir=str(folder)))

    def test_source_build_enables_glm_native_experts(self):
        def compile_stub(source, build, target, defs, *rest):
            build.mkdir(exist_ok=True)
            (build / S.EXE).write_text("test engine")
            self.assertIn("-DSTRATA_ENABLE_GLM=ON", defs)
            self.assertIn("-DSTRATA_NATIVE_EXPERTS=ON", defs)
        with patch.object(S, "source_hash", return_value="hash"), \
             patch.object(S, "source_version", return_value="test"), \
             patch.object(S, "cpu_info", return_value=(None, None)), \
             patch.object(S, "cpu_floor", return_value=""), \
             patch.object(S, "install_build_tools", return_value=(str(self.root / "bin" / "nvcc"), None)), \
             patch.object(S, "cmake_build", side_effect=compile_stub):
            eng = S.build_engine(dict(arch=86), "none", True, self.root / "llama", glm=True)
        self.assertEqual(eng, self.root / "engine-glm")
        self.assertTrue((self.root / "build-glm" / S.EXE).exists())


if __name__ == "__main__":
    unittest.main()
