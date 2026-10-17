"""Tests for the SYCL port's image support (docs/INTEL.md, "Images (SYCL port)"): sycl/setup_intel.py maps --vision and
checks the CPU encoder before setup runs, to_sycl() keeps the config's vision section on the host and the --vision
flag exactly when images are on, and the --vram-reserve-mib rule for the encoder; sycl/serve/server_intel.py puts the
embeddings files under two directories; sycl/serve/strata-sycl.sh mounts those read-only; sycl/tools/build-vision.sh
parses and stops clearly without llama.cpp.  No setup run, no container, no GPU, no network.  The tests use temporary
directories: /dev/shm, the live server's data and the live config are never touched.

    python -m unittest tools.test_setup_sycl_vision
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402


def _load(name: str, path: Path):
    """The sycl/ modules by path: the bare names "serve" and "tools" are the repo's own packages, not sycl's."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SI = _load("setup_intel", ROOT / "sycl" / "setup_intel.py")
SV = _load("server_intel", ROOT / "sycl" / "serve" / "server_intel.py")
SRV = SV.S                                  # serve/server.py, which server_intel.py extends

DATA = ROOT.parent / "Strata-data"          # the models, packs and tokenizer folder beside the checkout
CARD = {"index": 0, "name": "Intel Arc Pro B60", "vram_gb": 24.0, "arch": "xe", "driver": "xe", "devid": "e211"}


def work(path) -> str:
    """A host path under the folder above the checkout, as the engine's container sees it (/work/...)."""
    return "/work/" + Path(path).relative_to(ROOT.parent).as_posix()


def base_args(vision: bool, data: Path = DATA, checkout: Path = ROOT) -> list:
    """The args setup.py writes for Coder IQ1_M on the CUDA/HIP path (the port's to_sycl() input), and with images
    setup's GPU-encoder reserve beside --vision."""
    a = ["--pack", str(data / "packs/coder-iq1_m"),
         "--native", str(data / "models/coder-IQ1_M/Q-00001-of-00002.gguf"),
         "--ple-gguf", str(data / "models/coder-IQ1_M/Q-00002-of-00002.gguf"),
         "--expert-profile", str(checkout / "data/expert-profile-coder.bin"),
         "--expert-cache", "auto", "--spec", "4", "--spec-min-p", "0.5",
         "--mtp", str(data / "mtp/rt"), "--max-context", "131072", "--kv", "int8",
         "--prefill", "4096", "--resident-experts"]
    if vision:
        a += ["--vision", "--vram-reserve-mib", "700"]
    return a


def setup_cfg(vision: bool, reserve=None, data: Path = DATA, checkout: Path = ROOT) -> dict:
    """A config as setup.py writes it before the SYCL step; with images, its vision section."""
    args = base_args(vision, data, checkout)
    if reserve is not None:
        if "--vram-reserve-mib" in args:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
    cfg = {"exe": str(checkout / "build-sycl-aot" / "strata"), "args": args, "cwd": str(checkout.parent),
           "tokenizer": str(data / "tok"), "model_name": "coder", "log": "x.log", "port": 8085, "backend": "hip",
           "gpu": 0, "lib_dirs": [], "env": {}, "host": "127.0.0.1", "api_key": "test-key-not-real"}
    if vision:
        cfg["vision"] = {"exe": str(checkout / "build-vision" / "strata-vision.sh"),
                         "mmproj": str(data / "models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
                         "model": str(data / "models/coder-IQ1_M/Q-00001-of-00002.gguf"),
                         "gpu": False, "max_tokens": 300, "threads": 4}
    return cfg


def expected_args(reserve: str, vision: bool = False) -> list:
    """to_sycl()'s args for setup_cfg(): the port's --stream-experts and KV streaming (the context is 131072), the
    prompt chunk of 4096, and --vision last when images are on."""
    a = ["--pack", work(DATA / "packs/coder-iq1_m"),
         "--native", work(DATA / "models/coder-IQ1_M/Q-00001-of-00002.gguf"),
         "--ple-gguf", work(DATA / "models/coder-IQ1_M/Q-00002-of-00002.gguf"),
         "--expert-profile", work(ROOT / "data/expert-profile-coder.bin"),
         "--expert-cache", "auto", "--spec", "4", "--spec-min-p", "0.5",
         "--mtp", work(DATA / "mtp/rt"), "--max-context", "131072", "--kv", "int8",
         "--kv-resident", "32768", "--stream-experts", "--vram-reserve-mib", reserve,
         "--prefill", "4096"]
    return a + (["--vision"] if vision else [])


def expected_plain(args: list) -> dict:
    """to_sycl()'s output for a config without a vision section: the keys setup wrote, with the SYCL port's exe,
    backend, args and sycl_root; lib_dirs and env are gone (no env: nothing else needs setting)."""
    return {"exe": str(SI.SYCL_WRAPPER), "args": args, "cwd": str(ROOT.parent),
            "tokenizer": str(DATA / "tok"), "model_name": "coder", "log": "x.log", "port": 8085,
            "backend": "sycl", "gpu": 0, "host": "127.0.0.1", "api_key": "test-key-not-real",
            "sycl_root": str(ROOT.parent)}


def to_sycl(cfg: dict) -> dict:
    exe = ROOT / "build-sycl-aot" / "strata"
    return SI.to_sycl(json.loads(json.dumps(cfg)), exe, 96.0, {}, 24.0, "xe", "e211")


class VisionMapping(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(S, "warn")
        self.warn = p.start()
        self.addCleanup(p.stop)

    def test_cpu_is_the_cpu_encoder_without_a_warning(self):
        self.assertEqual(SI.intel_vision("cpu"), "cpu")
        self.warn.assert_not_called()

    def test_yes_and_gpu_read_on_the_cpu_with_one_warning(self):
        for asked in ("yes", "gpu"):
            with self.subTest(asked=asked):
                self.warn.reset_mock()
                self.assertEqual(SI.intel_vision(asked), "cpu")
                self.assertEqual(self.warn.call_count, 1)
                self.assertIn("no GPU image encoder", self.warn.call_args[0][0])

    def test_everything_else_leaves_images_off_without_a_warning(self):
        for asked in ("none", "no", None, "bogus"):
            with self.subTest(asked=asked):
                self.assertEqual(SI.intel_vision(asked), "none")
        self.warn.assert_not_called()


class ToSycl(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(SI, "MOUNT", ROOT.parent)     # what STRATA_SYCL_ROOT would change
        p.start()
        self.addCleanup(p.stop)

    def test_no_vision_section_gives_the_pre_vision_output(self):
        """The output for a config without images, as the code before the vision work produced it (run once at
        HEAD and written out here)."""
        out = to_sycl(setup_cfg(False))
        self.assertEqual(out, expected_plain(expected_args("2048")))

    def test_no_vision_section_keeps_a_reserve_the_user_gave(self):
        out = to_sycl(setup_cfg(False, reserve=1500))
        self.assertEqual(out, expected_plain(expected_args("1500")))

    def test_a_stray_vision_flag_without_a_section_is_dropped(self):
        cfg = setup_cfg(False)
        cfg["args"] += ["--vision"]
        out = to_sycl(cfg)
        self.assertNotIn("--vision", out["args"])
        self.assertNotIn("vision", out)

    def test_vision_section_goes_to_the_host_launcher(self):
        cfg = setup_cfg(True)
        out = to_sycl(cfg)
        self.assertEqual(out["exe"], str(SI.SYCL_WRAPPER))
        self.assertEqual(out["vision"], {"exe": str(SI.VISION_LAUNCHER), "mmproj": cfg["vision"]["mmproj"],
                                         "model": cfg["vision"]["model"], "gpu": False, "max_tokens": 300,
                                         "threads": 4})
        self.assertEqual(out["args"], expected_args("700", vision=True))

    def test_the_launcher_is_the_build_vision_folder_beside_the_checkout(self):
        self.assertEqual(SI.VISION_LAUNCHER, ROOT / "build-vision" / "strata-vision.sh")

    def test_vision_flag_is_kept_exactly_once(self):
        cfg = setup_cfg(True)
        cfg["args"] += ["--vision"]
        self.assertEqual(to_sycl(cfg)["args"].count("--vision"), 1)


class InstallBase(unittest.TestCase):
    """install() with the hardware and the run stubbed: a temporary checkout (install() reads the folder's configs
    and writes its run script there), and setup's main() replaced so that the arguments it would get are recorded."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vision-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.mount = self.tmp                          # the folder above the checkout (what strata-sycl.sh mounts)
        self.root = self.tmp / "Checkout"
        self.root.mkdir()
        self.launcher = self.root / "build-vision" / "strata-vision.sh"
        self.argv_seen = None
        self.hip_vision_seen = None
        self.saved_setup = dict(vars(S))
        self.saved_argv = list(sys.argv)
        self.addCleanup(self._restore)
        self.addCleanup(mock.patch.stopall)
        patches = [
            mock.patch.object(SI, "ROOT", self.root),
            mock.patch.object(SI, "MOUNT", self.mount),
            mock.patch.object(SI, "VISION_LAUNCHER", self.launcher),
            mock.patch.object(SI, "intel_gpus", return_value=[CARD]),
            mock.patch.object(SI, "sycl_engine", return_value=(self.root / "build-sycl-aot" / "strata", None)),
            mock.patch.object(SI.atexit, "register"),
            mock.patch.object(tempfile, "tempdir", str(self.tmp)),
            mock.patch.object(S, "warn"),
            mock.patch.object(S, "say"),
            mock.patch.object(S, "ram_gb", return_value=64.0),
            mock.patch.object(S, "fail", side_effect=self._fail),
            mock.patch.object(S, "main", side_effect=self._main),
            mock.patch.object(S, "write_run_script", new=self._fake_run_script),
        ]
        for p in patches:
            p.start()

    def _restore(self):
        for k in list(vars(S)):
            if k not in self.saved_setup:
                delattr(S, k)
        for k, v in self.saved_setup.items():
            if getattr(S, k, None) is not v:
                setattr(S, k, v)
        sys.argv = self.saved_argv

    @staticmethod
    def _fail(msg, hint=None):
        raise Fail(msg + " | " + (hint or ""))

    def _main(self):
        self.argv_seen = list(sys.argv)
        self.hip_vision_seen = S.hip_vision
        return 0

    def _fake_run_script(self, model, cfg_path, port, open_browser=True):
        """setup.write_run_script's stand-in: a run script that names setup's server.py, as the real one does."""
        p = self.root / "run-coder.sh"
        p.write_text("server.py: " + str(self.root / "serve" / "server.py") + "\n")
        return p

    def install(self, argv):
        try:
            SI.install(list(argv))
        except SystemExit:
            pass

    def run_writer(self, cfg: dict, argv: list) -> dict:
        """install() with argv, then the run-script writer it installs, on cfg; returns the config it wrote."""
        cfg_path = self.root / "strata-coder.json"
        cfg_path.write_text(json.dumps(cfg, indent=1))
        S.write_run_script = self._fake_run_script      # install() wraps whatever writer is there: start from this
        self.install(argv)
        S.write_run_script("coder", str(cfg_path), 8085, False)
        return json.loads(cfg_path.read_text())


class Fail(Exception):
    pass


class InstallEncoderCheck(InstallBase):
    def test_images_asked_without_the_encoder_stop_before_setup(self):
        for argv in (["--vision", "cpu"], ["--vision", "yes"], ["--vision", "gpu"],
                     ["--vision=cpu"], ["--vision=yes"], ["--vision=gpu"]):
            with self.subTest(argv=argv):
                self.argv_seen = None
                with self.assertRaises(Fail) as ctx:
                    SI.install(list(argv))
                msg = str(ctx.exception)
                self.assertIn("not built", msg)
                self.assertIn("build-vision/strata-vision.sh", msg)
                self.assertIn(f"REPO=/work/{self.root.name}", msg)
                order = [msg.index(s) for s in ("1) ./setup.sh --backend sycl", "2) build the encoder",
                                                "3) ./setup.sh --backend sycl --vision cpu")]
                self.assertEqual(order, sorted(order))
                self.assertIsNone(self.argv_seen)           # setup's main() was not reached

    def test_images_off_does_not_need_the_encoder(self):
        self.install(["--vision", "none"])
        self.assertIsNotNone(self.argv_seen)
        self.assertEqual(self.argv_seen.count("--vision"), 1)
        self.assertEqual(self.argv_seen[self.argv_seen.index("--vision") + 1], "none")

    def test_setup_gets_the_vision_flag_once_when_images_are_on(self):
        self.launcher.parent.mkdir()
        self.launcher.write_text("#!/bin/sh\n")
        self.install(["--vision", "cpu"])
        self.assertEqual(self.argv_seen.count("--vision"), 1)
        self.assertEqual(self.argv_seen[self.argv_seen.index("--vision") + 1], "cpu")
        self.assertIs(self.hip_vision_seen, SI.intel_vision)

    def test_images_off_by_default_setup_gets_vision_none(self):
        self.install([])
        self.assertEqual(self.argv_seen.count("--vision"), 1)
        self.assertEqual(self.argv_seen[self.argv_seen.index("--vision") + 1], "none")

    def test_equals_form_passes_once_and_no_injected_none(self):
        self.launcher.parent.mkdir()
        self.launcher.write_text("#!/bin/sh\n")
        for argv in (["--vision=cpu"], ["--vision=yes"]):
            with self.subTest(argv=argv):
                self.install(argv)
                flags = [a for a in self.argv_seen if a == "--vision" or a.startswith("--vision=")]
                self.assertEqual(flags, argv)
                self.assertNotIn("none", self.argv_seen)


class InstallWriter(InstallBase):
    """The run-script writer install() installs: the config setup wrote, turned into the port's config."""

    def setUp(self):
        super().setUp()
        self.launcher.parent.mkdir()
        self.launcher.write_text("#!/bin/sh\n")

    def test_images_config_points_at_the_launcher_and_keeps_the_flag_once(self):
        got = self.run_writer(setup_cfg(True, data=self.mount / "Strata-data", checkout=self.root), ["--vision", "cpu"])
        self.assertEqual(got["exe"], str(SI.SYCL_WRAPPER))
        self.assertEqual(got["vision"]["exe"], str(self.launcher))
        self.assertIs(got["vision"]["gpu"], False)
        self.assertEqual(got["args"].count("--vision"), 1)

    def test_setups_gpu_reserve_is_not_kept_when_the_user_did_not_type_one(self):
        got = self.run_writer(setup_cfg(True, data=self.mount / "Strata-data", checkout=self.root), ["--vision", "cpu"])
        self.assertEqual(got["args"][got["args"].index("--vram-reserve-mib") + 1], "2048")
        self.assertEqual(got["args"].count("--vram-reserve-mib"), 1)

    def test_reserve_rule(self):
        """(name, reserve setup wrote, argv the user typed, the reserve the config must end with)."""
        cases = (
            ("typed in the = form is kept", 700, ["--vision", "cpu", "--vram-reserve-mib=700"], "700"),
            ("typed in the space form is kept", 700, ["--vision", "cpu", "--vram-reserve-mib", "700"], "700"),
            ("not typed, setup's 700 is stripped", 700, ["--vision", "cpu"], "2048"),
            ("not typed, a reserve from an earlier config is kept", 1500, ["--vision", "cpu"], "1500"),
            ("--vision= form, reserve from an earlier config is kept", 1500, ["--vision=cpu"], "1500"),
        )
        for name, wrote, argv, want in cases:
            with self.subTest(name):
                got = self.run_writer(setup_cfg(True, reserve=wrote, data=self.mount / "Strata-data",
                                                checkout=self.root), argv)
                self.assertEqual(got["args"].count("--vram-reserve-mib"), 1)
                self.assertEqual(got["args"][got["args"].index("--vram-reserve-mib") + 1], want)

    def test_images_off_writes_no_vision_flag_and_no_section(self):
        got = self.run_writer(setup_cfg(False, data=self.mount / "Strata-data", checkout=self.root), ["--vision", "no"])
        self.assertNotIn("--vision", got["args"])
        self.assertNotIn("vision", got)


class InstallVisionMapping(InstallBase):
    def test_hip_vision_is_the_intel_mapping_while_setup_runs(self):
        self.install([])
        self.assertIs(self.hip_vision_seen, SI.intel_vision)


class ServerIntelVisionPaths(unittest.TestCase):
    """server_intel.install_vision_paths(): with a vision section the embeddings files go under two directories the
    wrapper mounts; without one nothing changes.  /dev/shm is replaced by a temporary directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vision-test-srv-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shm = self.tmp / "shm"                       # stands in for /dev/shm/strata-sycl-vision
        self.root = self.tmp / "root"                     # the data root (sycl_root)
        self.root.mkdir()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ).start()
        for k in ("STRATA_SYCL_VISION_DIRS", "TMPDIR"):
            os.environ.pop(k, None)
        mock.patch.object(tempfile, "tempdir", tempfile.tempdir).start()     # restored afterwards
        mock.patch.object(SRV, "combined_embeddings_path", SRV.combined_embeddings_path).start()
        mock.patch.object(SV, "VISION_SHM", self.shm).start()

    def write_cfg(self, name: str, cfg: dict) -> Path:
        p = self.tmp / name
        p.write_text(json.dumps(cfg))
        return p

    def test_no_vision_section_changes_nothing(self):
        before = SRV.combined_embeddings_path
        tmpdir = tempfile.tempdir
        SV.install_vision_paths(["--config", str(self.write_cfg("plain.json", {"sycl_root": str(self.root)}))])
        self.assertNotIn("STRATA_SYCL_VISION_DIRS", os.environ)
        self.assertNotIn("TMPDIR", os.environ)
        self.assertIs(SRV.combined_embeddings_path, before)
        self.assertEqual(tempfile.tempdir, tmpdir)
        self.assertFalse((self.root / "vision-tmp").exists())

    def test_missing_or_unreadable_config_is_ignored(self):
        bad = self.tmp / "bad.json"
        bad.write_text("{not json")
        for argv in (["--config", str(self.tmp / "nope.json")], [], ["--config"], ["--config", str(bad)]):
            with self.subTest(argv=argv):
                SV.install_vision_paths(argv)
                self.assertNotIn("STRATA_SYCL_VISION_DIRS", os.environ)

    def test_vision_section_exports_the_two_directories(self):
        cfg = {"sycl_root": str(self.root), "vision": {"exe": "x"}}
        SV.install_vision_paths(["--config", str(self.write_cfg("withv.json", cfg))])
        dirs = os.environ["STRATA_SYCL_VISION_DIRS"].split(":")
        self.assertEqual(dirs, [str(self.shm), str(self.root / "vision-tmp")])
        self.assertEqual(os.environ["TMPDIR"], str(self.root / "vision-tmp"))
        self.assertEqual(tempfile.tempdir, str(self.root / "vision-tmp"))
        for d in (self.shm, self.root / "vision-tmp"):
            self.assertEqual(os.stat(d).st_mode & 0o777, 0o700, str(d))

    def test_a_small_embeddings_file_goes_to_ram_a_huge_one_to_vision_tmp(self):
        SV.install_vision_paths(["--config", str(self.write_cfg(
            "withv.json", {"sycl_root": str(self.root), "vision": {"exe": "x"}}))])
        small = SRV.combined_embeddings_path(self.root / "vision-tmp", 1000)
        self.assertEqual(small.parent, self.shm)
        self.assertTrue(small.name.startswith("strata-vision-"))
        big = SRV.combined_embeddings_path(self.root / "vision-tmp", 10 ** 15)     # no room in RAM
        self.assertEqual(big.parent, self.root / "vision-tmp")

    def test_an_unusable_ram_directory_leaves_only_vision_tmp(self):
        afile = self.tmp / "afile"
        afile.write_text("not a directory")
        mock.patch.object(SV, "VISION_SHM", afile / "shm").start()
        SV.install_vision_paths(["--config", str(self.write_cfg(
            "withv.json", {"sycl_root": str(self.root), "vision": {"exe": "x"}}))])
        self.assertEqual(os.environ["STRATA_SYCL_VISION_DIRS"], str(self.root / "vision-tmp"))
        self.assertEqual(SRV.combined_embeddings_path(self.root / "vision-tmp", 1000).parent,
                         self.root / "vision-tmp")

    def test_private_dir_creates_mode_0700(self):
        fresh = self.tmp / "fresh"
        SV.private_dir(fresh)
        self.assertEqual(os.stat(fresh).st_mode & 0o777, 0o700)

    def test_private_dir_sets_an_existing_dir_we_own_to_0700(self):
        loose = self.tmp / "loose"
        loose.mkdir(mode=0o755)
        os.chmod(loose, 0o755)
        SV.private_dir(loose)
        self.assertEqual(os.stat(loose).st_mode & 0o777, 0o700)


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bash"), "needs Linux and bash")
class WrapperMounts(unittest.TestCase):
    """sycl/serve/strata-sycl.sh with a fake `docker` on PATH that prints its arguments: no container is started."""

    WRAPPER = str(ROOT / "sycl" / "serve" / "strata-sycl.sh")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vision-test-wrap-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        fake = self.tmp / "bin"
        fake.mkdir()
        (fake / "docker").write_text('#!/bin/sh\nfor a in "$@"; do printf "ARG %s\\n" "$a"; done\n')
        (fake / "docker").chmod(0o755)
        self.fake = fake
        self.dir_a = self.tmp / "ram"
        self.dir_a.mkdir()
        self.dir_b = self.tmp / "vision-tmp"
        self.dir_b.mkdir()

    def run_wrapper(self, extra_env: dict, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_SYCL_VISION")}
        env["PATH"] = f"{self.fake}{os.pathsep}{env.get('PATH', '')}"
        env.update(extra_env)
        r = subprocess.run(["bash", self.WRAPPER, *args], capture_output=True, text=True, env=env)
        return r.returncode, r.stdout.splitlines(), r.stderr

    @staticmethod
    def mounts(lines):
        """the read-only mounts: the value after each -v that ends in :ro (the /work mount is not read-only)"""
        vals = [lines[i + 1].split("ARG ", 1)[1] for i in range(len(lines) - 1) if lines[i] == "ARG -v"]
        return [v for v in vals if v.endswith(":ro")]

    def test_vision_mounts_the_existing_dirs_read_only_at_their_paths(self):
        dirs = f"{self.dir_a}:{self.tmp / 'missing'}:{self.dir_b}"
        rc, out, err = self.run_wrapper({"STRATA_SYCL_VISION_DIRS": dirs}, "--config", "/x.json", "--vision", "cpu")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.mounts(out), [f"{self.dir_a}:{self.dir_a}:ro", f"{self.dir_b}:{self.dir_b}:ro"])

    def test_no_vision_means_no_mounts_even_with_the_dirs_set(self):
        dirs = f"{self.dir_a}:{self.dir_b}"
        rc, out, err = self.run_wrapper({"STRATA_SYCL_VISION_DIRS": dirs}, "--config", "/x.json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.mounts(out), [])

    def test_vision_without_the_dirs_warns_and_mounts_nothing(self):
        rc, out, err = self.run_wrapper({}, "--vision", "cpu")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.mounts(out), [])
        self.assertIn("without STRATA_SYCL_VISION_DIRS", err)

    def test_no_vision_and_no_dirs_is_silent(self):
        rc, out, err = self.run_wrapper({}, "--config", "/x.json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.mounts(out), [])
        self.assertEqual(err, "")


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class ShellScripts(unittest.TestCase):
    def test_build_vision_and_wrapper_parse(self):
        for script in (ROOT / "sycl" / "tools" / "build-vision.sh", ROOT / "sycl" / "serve" / "strata-sycl.sh"):
            with self.subTest(script=script.name):
                r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_build_vision_is_executable(self):
        self.assertTrue(os.access(ROOT / "sycl" / "tools" / "build-vision.sh", os.X_OK))

    def test_build_vision_without_llama_cpp_stops_with_the_fetch_step(self):
        """Stops before any compile: no third_party/llama.cpp in the checkout given."""
        tmp = Path(tempfile.mkdtemp(prefix="vision-test-bv-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        env = {**os.environ, "REPO": str(tmp), "BUILD_DIR": str(tmp / "build")}
        r = subprocess.run(["bash", str(ROOT / "sycl" / "tools" / "build-vision.sh")], capture_output=True, text=True,
                           env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("llama.cpp is not in", r.stdout)
        self.assertIn("./setup.sh --backend sycl", r.stdout)
        self.assertIn("BUILD EXIT 1", r.stdout)


if __name__ == "__main__":
    unittest.main()
