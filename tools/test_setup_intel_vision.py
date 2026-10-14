"""CPU-only checks for the Intel image encoder's setup branch (sycl/setup_intel.py).

The branch decides whether `--vision` reaches the engine at all, so the cases that matter are: off stays off,
on is refused when the encoder was never built, and the config's `vision` section survives `to_sycl()` - the
port's config rewrite used to drop it.  The encoder's launcher is checked here too: setup names it in the
config, so a launcher that cannot find the oneAPI runtime fails at the first image rather than here."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sycl"))

import setup_intel as I  # noqa: E402

SH = shutil.which("bash") or shutil.which("sh") or ""
ENV = shutil.which("env") or ""     # a real binary that prints its environment with no arguments
LDD = shutil.which("ldd") or ""     # the launcher's own preflight


class VisionBranchTests(unittest.TestCase):
    def test_asked_off_stays_off(self):
        for asked in (None, "no", "none"):
            self.assertEqual(I.vision_intel(asked), "none", f"asked={asked!r}")

    def test_on_needs_a_built_encoder(self):
        with patch.object(I, "sycl_vision_encoder", return_value=(None, "not built")):
            for asked in ("yes", "gpu", "cpu"):
                self.assertEqual(I.vision_intel(asked), "none", f"asked={asked!r}")

    def test_yes_and_gpu_place_the_graph_on_the_card(self):
        with patch.object(I, "sycl_vision_encoder", return_value=(Path("/nowhere/strata-vision"), None)):
            self.assertEqual(I.vision_intel("yes"), "gpu")
            self.assertEqual(I.vision_intel("gpu"), "gpu")

    def test_cpu_is_the_same_binary_without_gpu(self):
        with patch.object(I, "sycl_vision_encoder", return_value=(Path("/nowhere/strata-vision"), None)):
            self.assertEqual(I.vision_intel("cpu"), "cpu")

    def test_to_sycl_keeps_the_vision_section(self):
        # setup adds --vision to the engine's args itself (when images are on); to_sycl must not drop it, and
        # must name the encoder's launcher in the config's vision section - not the bare binary, which cannot
        # resolve libsycl.so / oneMKL from the loader's default path and dies at the first image
        cfg = {"args": ["--max-context", "32768", "--vision", "--vram-reserve-mib", "700"],
               "vision": {"exe": "temporary/stub/strata-vision", "gpu": True,
                          "mmproj": "m.gguf", "model": "t.gguf"}}
        with patch.object(I, "VISION_ENCODER", Path("/nowhere/strata-vision")):
            out = I.to_sycl(cfg, ROOT / "build-sycl-aot" / "strata", 64.0, {}, 16.0, "xe", "56a0")
        self.assertIn("vision", out)
        self.assertEqual(out["vision"]["exe"], str(I.VISION_LAUNCHER))
        self.assertTrue(out["vision"]["gpu"])
        self.assertIn("--vision", out["args"])

    def test_the_launcher_the_config_names_is_there(self):
        # a config that names a launcher the checkout does not have fails at the first image (or at the
        # server's start), not here, so check it here
        self.assertTrue(I.VISION_LAUNCHER.is_file(), I.VISION_LAUNCHER)
        self.assertTrue(os.access(I.VISION_LAUNCHER, os.X_OK), I.VISION_LAUNCHER)

    def test_the_launcher_keeps_the_encoders_arguments(self):
        # the server passes --mmproj/--model/--gpu/... through the launcher; `exec "$bin" "$@"` is what keeps
        # them, and the oneAPI runtime has to be on the loader path before the binary runs
        text = I.VISION_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('exec "$bin" "$@"', text)
        self.assertIn("LD_LIBRARY_PATH", text)

    def test_to_sycl_leaves_the_reserve_alone_without_images(self):
        cfg = {"args": ["--max-context", "32768"]}
        out = I.to_sycl(cfg, ROOT / "build-sycl-aot" / "strata", 64.0, {}, 16.0, "xe", "56a0")
        self.assertNotIn("vision", out)
        reserve = out["args"][out["args"].index("--vram-reserve-mib") + 1]
        self.assertEqual(int(reserve), 1024)

    def test_to_sycl_raises_the_reserve_for_the_encoder(self):
        cfg = {"args": ["--max-context", "32768"],
               "vision": {"exe": "e", "gpu": True, "mmproj": "m", "model": "t"}}
        out = I.to_sycl(cfg, ROOT / "build-sycl-aot" / "strata", 64.0, {}, 16.0, "xe", "56a0")
        reserve = out["args"][out["args"].index("--vram-reserve-mib") + 1]
        self.assertGreaterEqual(int(reserve), I.VISION_RESERVE_MIB)


@unittest.skipIf(os.name == "nt" or not (SH and ENV and LDD), "runs the launcher (a POSIX shell and ldd)")
class LauncherRuntimeTests(unittest.TestCase):
    """The launcher composes the oneAPI runtime path itself, so it has to find the install: STRATA_ONEAPI_ROOT,
    else the oneAPI whose icpx is on PATH, else Intel's own default /opt/intel/oneapi.  A root one level too
    deep adds no directory at all, and then the encoder dies at the first image on a missing libsycl.so.

    The binary is `env`: with no arguments it prints the environment, so the launcher reaches its `exec` and
    the test reads the LD_LIBRARY_PATH it composed."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "oneapi"
        for d in ("compiler/2026.1/lib", "umf/0.11/lib", "mkl/2026.1/lib"):
            (self.root / d).mkdir(parents=True)

    def run_launcher(self, icpx_on_path=False, root=None) -> str:
        env = dict(os.environ)
        env.pop("STRATA_ONEAPI_ROOT", None)
        env.pop("LD_LIBRARY_PATH", None)                 # the launcher's own composition is what is read
        env["STRATA_VISION_BIN"] = ENV
        if root is not None:
            env["STRATA_ONEAPI_ROOT"] = str(root)
        if icpx_on_path:
            bin_dir = self.root / "compiler" / "2026.1" / "bin"
            bin_dir.mkdir(parents=True)
            icpx = bin_dir / "icpx"
            icpx.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            icpx.chmod(0o755)
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        else:                                           # a real icpx on this host must not be picked up
            env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                          if not (Path(p) / "icpx").exists())
        done = subprocess.run([SH, str(I.VISION_LAUNCHER)], env=env, capture_output=True, text=True)
        for line in done.stdout.splitlines():
            if line.startswith("LD_LIBRARY_PATH="):
                return line[len("LD_LIBRARY_PATH="):]
        self.fail(f"the launcher added no runtime directory (rc={done.returncode}): {done.stderr}")

    def test_the_root_is_the_one_the_icpx_on_path_belongs_to(self):
        # <root>/compiler/<version>/bin/icpx: the root is three levels above the bin directory
        libs = self.run_launcher(icpx_on_path=True)
        self.assertIn(str(self.root / "compiler" / "2026.1" / "lib"), libs)
        self.assertIn(str(self.root / "mkl" / "2026.1" / "lib"), libs)

    def test_strata_oneapi_root_names_the_install(self):
        libs = self.run_launcher(root=self.root)
        self.assertIn(str(self.root / "compiler" / "2026.1" / "lib"), libs)
        self.assertIn(str(self.root / "umf" / "0.11" / "lib"), libs)


if __name__ == "__main__":
    unittest.main()
