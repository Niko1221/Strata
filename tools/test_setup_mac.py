"""Tests for setup on a Mac (metal/setup_mac.py and setup.py's hand-over): the Mac check, the pack recommended by
unified memory, what a bigger choice risks, the engine arguments with the Metal defaults and the config. Runs on any
PC: sysctl / sw_vers / xcode-select are mocked. No GPU, no network.

    python -m unittest tools.test_setup_mac
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metal"))
import setup  # noqa: E402
import setup_mac as M  # noqa: E402


def fake_run(values: dict, clt_rc=0):
    def run(cmd, **kw):
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(cmd, 0, values.get(cmd[2], ""), "")
        if cmd[0] == "sw_vers":
            return subprocess.CompletedProcess(cmd, 0, values.get("macos", ""), "")
        if cmd[0] == "xcode-select":
            return subprocess.CompletedProcess(cmd, clt_rc, "/Library/Developer/CommandLineTools" if clt_rc == 0 else "",
                                               "")
        raise AssertionError(cmd)
    return run


M5 = {"machdep.cpu.brand_string": "Apple M5 Max", "hw.memsize": str(128 * 2 ** 30), "macos": "27.2",
      "hw.perflevel0.physicalcpu": "12"}


class MacCheck(unittest.TestCase):
    def test_reads_chip_memory_macos_and_tools(self):
        with mock.patch.object(M.platform, "machine", return_value="arm64"):
            info = M.mac_info(fake_run(M5))
        self.assertEqual(info["chip"], "Apple M5 Max")
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["macos"], "27.2")
        self.assertTrue(info["arm64"])
        self.assertTrue(info["clt"])
        self.assertEqual(M.mac_problems(info), [])

    def test_intel_old_macos_and_missing_tools_are_named(self):
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "macos": "14.6", "machdep.cpu.brand_string": "Intel Core i9"}, clt_rc=2))
        p = " | ".join(M.mac_problems(info))
        self.assertIn("Apple Silicon", p)
        self.assertIn("macOS 15", p)
        self.assertIn("xcode-select --install", p)


class Recommendation(unittest.TestCase):
    def test_by_memory(self):
        # IQ3_S and UD-Q4_K_XL are the recommended sizes; IQ3_S is the default wherever it stays resident
        self.assertEqual(M.RECOMMENDED, ("IQ3_S", "UD-Q4_K_XL"))
        rec, notes = M.recommend_model(128)
        self.assertEqual(rec, "IQ3_S")
        self.assertTrue(any("UD-Q4_K_XL" in n and "higher quality" in n for n in notes))   # named next to it
        rec96, notes96 = M.recommend_model(96)
        self.assertEqual(rec96, "IQ3_S")
        self.assertFalse(notes96)                       # UD-Q4_K_XL does not fit 96 GB resident: no note
        self.assertEqual(M.recommend_model(64)[0], "Q2_0")      # no recommended size fits: the largest that does
        self.assertEqual(M.recommend_model(80)[0], "IQ3_XXS")
        rec, notes = M.recommend_model(32)
        self.assertEqual(rec, "Q2_0")
        self.assertTrue(notes and "stream" in notes[0] and "~14 tok/s" in notes[0])
        self.assertTrue(M.streams("Q2_0", 32) and M.streams("Q2_0", 48) and not M.streams("Q2_0", 64))

    def test_a_bigger_choice_is_kept_with_a_note(self):
        notes = M.model_note("IQ3_XXS", 64)
        self.assertTrue(any("stream from the SSD" in n for n in notes))
        self.assertEqual(M.model_note("Q2_0", 128), [])
        self.assertEqual(M.model_note("IQ3_S", 128), [])

    def test_context(self):
        self.assertEqual(M.recommend_context(64), 32768)
        self.assertEqual(M.recommend_context(128), 65536)


class Config(unittest.TestCase):
    def test_engine_args_carry_the_metal_defaults(self):
        args = M.engine_args(Path("/p/packs/q2_0"), [Path("/m/a-00001.gguf"), Path("/m/a-00002.gguf")],
                             Path("/d/mtp/rt"), 32768)
        self.assertEqual(args[args.index("--spec") + 1], "2")
        self.assertEqual(args[args.index("--native") + 1], str(Path("/m/a-00001.gguf")))
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(Path("/m/a-00002.gguf")))
        self.assertEqual(args[args.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertIn("--expert-profile", args)
        self.assertNotIn("--hc-q8", args)                         # Q8_0 HC is the engine's own default on Metal
        st = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True)
        self.assertNotIn("--mtp", st)                             # streaming: one token per step, no draft layer
        self.assertEqual(st[st.index("--suffix-draft") + 1], "0")
        self.assertEqual(st[st.index("--spec") + 1], "2")
        # a big Mac that streams (UD-Q5/Q6 on 128 GB) keeps the draft layer, with a longer window
        big = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 65536, stream=True,
                            ram_gb=128)
        self.assertEqual(big[big.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertEqual(big[big.index("--spec") + 1], "3")
        self.assertEqual(big[big.index("--suffix-draft") + 1], "0")
        small = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True,
                              ram_gb=32)
        self.assertNotIn("--mtp", small)
        self.assertTrue(M.stream_mtp(False, 16) and M.stream_mtp(True, 128) and not M.stream_mtp(True, 64))

    def test_config_is_a_model_config_with_the_metal_backend(self):
        cfg = M.mac_config(Path("/r/build-metal/strata"), ["--pack", "x"], Path("/p"), "qwen3.8-flash-next-q2_0",
                           "Q2_0", 8080)
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["tokenizer"], str(Path("/p") / "tokenizer"))
        self.assertTrue(cfg["exe"] and isinstance(cfg["args"], list))


class Vision(unittest.TestCase):
    def test_the_projector_is_pinned_and_the_encoder_builds_with_metal(self):
        self.assertIn("mmproj-Qwen3.8-Flash-Next-BF16.gguf", M.VISION_SHA256)
        src = (ROOT / "metal" / "setup_mac.py").read_text(encoding="utf-8")
        self.assertIn('"-DSTRATA_VISION_CUDA=OFF"', src)
        self.assertIn('cfg["args"] += ["--vision"]', src)


class ExpertsFile(unittest.TestCase):
    """The streaming Mac's <model>.experts: written only with room on the disk, else a warning and the GGUF."""

    def run_it(self, free_gb, done=False):
        import tempfile
        sys.path.insert(0, str(ROOT / "metal" / "tools"))
        import stream_experts
        d = Path(tempfile.mkdtemp())
        native = d / "m.gguf"
        native.write_bytes(b"")
        if done:
            (d / "m.gguf.experts").write_bytes(b"")
            (d / "m.gguf.experts.done").write_text("x")
        with mock.patch.object(stream_experts, "size_bytes", return_value=40e9), \
                mock.patch.object(M.S, "free_gb", return_value=free_gb), \
                mock.patch.object(M.S, "run") as run, mock.patch.object(M.S, "warn") as warn, \
                mock.patch.object(M.S, "mark") as mark, mock.patch.object(M.S, "say"), mock.patch.object(M.S, "ok"):
            M.experts_file(native)
        return run, warn, mark

    def test_fits_needs_the_margin(self):
        self.assertFalse(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB - 0.1))
        self.assertTrue(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB))

    def test_written_and_verified_with_room(self):
        run, warn, mark = self.run_it(100)
        self.assertIn("--verify", run.call_args[0][0])
        self.assertTrue(str(run.call_args[0][0][1]).endswith("stream_experts.py"))
        mark.assert_called_once()
        warn.assert_not_called()

    def test_too_little_disk_warns_and_goes_on(self):
        run, warn, mark = self.run_it(45)
        run.assert_not_called()
        mark.assert_not_called()
        self.assertIn("not written", warn.call_args[0][0])

    def test_done_is_not_written_again(self):
        run, warn, mark = self.run_it(100, done=True)
        run.assert_not_called()


class HandOver(unittest.TestCase):
    def test_darwin_hands_the_run_to_setup_mac(self):
        with mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--model", "Q2_0", "--yes"]):
            self.assertEqual(setup.main(), 0)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("metal") / "setup_mac.py")))
        self.assertEqual(cmd[2:], ["--model", "Q2_0", "--yes"])

    def test_other_platforms_never_reach_it(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin":', src)
        with mock.patch.object(setup, "mac_setup", side_effect=AssertionError("called")), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--inspect", "x"]), \
                mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(SystemExit):
                setup.main()

    def test_setup_sh_fetches_a_pinned_python_on_macos_only(self):
        sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertIn('[ "$(uname -s)" = Darwin ]', sh)
        self.assertIn("83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8", sh)


class RepackGuard(unittest.TestCase):
    """No engine beside a repack: the locks, and the resident rule that matches the engine's arena_budget.h."""

    def test_resident_rule_matches_the_engine(self):
        # the resident cap: Metal's working set (3/4), and on a Mac of 64 GB or more the RAM less 30%
        self.assertAlmostEqual(M.gpu_budget_gb(128), 128 * 1.073741824 * 0.70, places=3)    # 96.2 GB, as arena_budget.h
        self.assertAlmostEqual(M.gpu_budget_gb(96), 96 * 1.073741824 * 0.70, places=3)
        self.assertAlmostEqual(M.gpu_budget_gb(32), 32 * 1.073741824 * 0.75, places=3)      # a small Mac: the working set
        self.assertFalse(M.streams("UD-Q4_K_XL", 128))          # 82.6 + 8 GB: resident (measured: nothing compressed)
        self.assertTrue(M.streams("UD-Q5_K_XL", 128))
        self.assertTrue(M.streams("UD-Q4_K_XL", 96))

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is a probe on macOS and Linux only")
    def test_a_live_lock_is_seen_a_stale_one_is_dropped(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(M.repack_running(d), [])
            (d / "m-metal.gguf.repack.lock").write_text(str(os.getppid()))          # a live process that is not us
            self.assertEqual(M.repack_running(d), [os.getppid()])
            (d / "m-metal.gguf.repack.lock").write_text("999999999")                # no such process
            self.assertEqual(M.repack_running(d), [])
            self.assertFalse((d / "m-metal.gguf.repack.lock").exists())             # the stale lock was removed
            (d / M.PREPARE_LOCK).write_text(str(os.getppid()))
            self.assertEqual(M.repack_running(d), [os.getppid()])
            (d / M.PREPARE_LOCK).write_text(str(os.getpid()))                       # our own setup's lock does not stop us
            self.assertEqual(M.repack_running(d), [])

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is a probe on macOS and Linux only")
    def test_start_refuses_while_a_repack_runs(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gguf, exe = d / "m-metal.gguf", d / "strata"
            gguf.write_bytes(b"x")
            exe.write_bytes(b"x")
            cfg = d / "strata-x.json"
            cfg.write_text(json.dumps({"exe": str(exe), "args": ["--native", str(gguf)], "port": 8080}))
            (d / "m-metal.gguf.repack.lock").write_text(str(os.getppid()))
            with mock.patch.object(M.subprocess, "call", side_effect=AssertionError("started")), \
                    mock.patch.object(M.S, "fail", side_effect=SystemExit("refused")) as fail:
                with self.assertRaises(SystemExit):
                    M.start(cfg, None, False)
            self.assertIn("repack is still running", fail.call_args[0][0])

    def test_the_repack_writes_in_bounded_pieces(self):
        src = (ROOT / "metal" / "tools" / "metal_repack.py").read_text(encoding="utf-8")
        self.assertIn("SYNC_EVERY = 1 << 30", src)                  # flushed every GiB
        self.assertIn("nocache(o)", src)                            # F_NOCACHE on the output
        self.assertIn("buffering=0", src)                           # write() is a write
        self.assertNotIn("mmap", src)                               # no dirty mapping of the output


class Source(unittest.TestCase):
    """setup.py hands a Mac over before its own parser runs, so the Mac parser has to know --source:
    source_hint() recommends it on every platform (#2363)."""

    def test_the_source_flag_reaches_strata_source(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_SOURCE", None)
        warnings = []
        # setup.data_folder() is mocked: the real one moves the model files of this checkout and of the data folder in
        # the user's settings.json into --data-dir, and saves that dir as the new data folder; with a temp dir that
        # deletes the models on exit and leaves settings.json pointing at a deleted directory
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(setup, "data_folder", lambda requested: (Path(tmp), [])), \
                mock.patch("subprocess.run", fake_run(M5)), \
                mock.patch.object(setup, "warn", lambda *a, **k: warnings.append(" ".join(str(x) for x in a))), \
                mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "ok", lambda *a, **k: None), \
                mock.patch.object(setup, "step", lambda *a, **k: None):
            M.main(["--source", "modelscope", "--yes", "--check", "--data-dir", tmp])
        self.assertEqual(os.environ.get("STRATA_SOURCE"), "modelscope")
        self.assertFalse([w for w in warnings if "--source" in w],
                         f"--source was not used: {warnings}")


if __name__ == "__main__":
    unittest.main()
