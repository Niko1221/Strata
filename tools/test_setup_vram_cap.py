"""Optional total-VRAM cap: flag/config writing, persistence, override and default-off regression.

CPU only: PC probes, downloads, builds and process launches are mocked. GPU validation (peak VRAM <= 80%
on a real run) is a SEPARATE task after 04:00; these tests do not establish a measured GPU peak.

    python tools/test_setup_vram_cap.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import GOLDEN, PROFILES, install  # noqa: E402
from test_setup_qwen36 import Base, F_IQ4, nvidia  # noqa: E402


class FlagAndPaths(Base):
    def setUp(self):
        super().setUp()
        self.offline = contextlib.ExitStack()
        self.addCleanup(self.offline.close)
        self.offline.enter_context(mock.patch.object(setup, "out", return_value=""))
        self.offline.enter_context(mock.patch.object(setup.subprocess, "Popen",
                                  side_effect=AssertionError("external process blocked during CPU tests")))

    def test_config_writes_engine_flag_and_windows_paths(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--vram-cap", "0.8"])
        self.assertEqual(code, 0, out)
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "0.8")
        self.assertEqual(cfg["args"].count("--vram-frac"), 1)
        self.assertNotIn("--vram-cap", cfg["args"])
        self.assertNotIn("vram_cap", cfg)   # stored as an engine argument, not an ignored top-level key
        self.assertEqual(Path(cfg["exe"]), self.t / "engine" / setup.EXE)
        self.assertEqual(Path(setup.flag_value(cfg["args"], "--native")),
                         self.t / "data" / "models" / "qwen36-UD-IQ4_XS" / F_IQ4)
        self.assertEqual(Path(cfg["cwd"]), self.t)
        self.assertEqual(self.cfg_path, self.t / "strata-qwen36-ud-iq4_xs.json")

    def test_reserve_is_not_replaced_by_setup_math(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--vram-cap", "0.8",
                                   "--vram-reserve-mib", "3000"])
        self.assertEqual(code, 0, out)
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "0.8")
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-reserve-mib"), "3000")

    def test_explicit_off_and_default_off(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS", "--vram-cap", "1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "1.0")  # overrides an environment cap
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-IQ4_XS"])
        self.assertEqual(code, 0, out)
        self.assertNotIn("--vram-frac", cfg["args"])
        self.assertNotIn("--vram-reserve-mib", cfg["args"])


class Parsing(unittest.TestCase):
    def invoke(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["setup.py", *argv]), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err), mock.patch.object(setup, "data_folder") as data, \
                mock.patch.object(setup, "gpus") as gpu:
            with self.assertRaises(SystemExit) as raised:
                setup.main()
            data.assert_not_called()
            gpu.assert_not_called()
        return raised.exception.code, out.getvalue(), err.getvalue()

    def test_invalid_fractions_stop_before_any_pc_probe(self):
        for v in ("0", "-0.1", "1.1", "nan", "inf", "-inf", "", "0.8garbage"):
            with self.subTest(v=v):
                code, _, err = self.invoke(["--vram-cap=" + v])
                self.assertEqual(code, 2)
                self.assertIn("--vram-cap", err)

    def test_help_explains_fraction_and_off_default(self):
        code, out, _ = self.invoke(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("--vram-cap F", out)
        self.assertIn("80%", out)
        self.assertIn("cap off", out)
        self.assertIn("--vram-frac", "".join(out.split()))  # argparse may wrap at the flag's hyphen

    def test_sycl_does_not_silently_write_an_unsupported_flag(self):
        code, _, err = self.invoke(["--backend", "sycl", "--vram-cap", "0.8"])
        self.assertEqual(code, 2)
        self.assertIn("separate SYCL engine", err)


class Persistence(unittest.TestCase):
    def install(self, *extra, configs=()):
        ram, found = PROFILES["64GB-1x32GB"]
        with mock.patch.object(setup, "out", return_value=""), \
                mock.patch.object(setup.subprocess, "Popen",
                                  side_effect=AssertionError("external process blocked during CPU tests")):
            return install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *extra], configs=configs)

    def test_generated_default_matches_existing_golden(self):
        code, out, cfg, _ = self.install()
        self.assertEqual(code, 0, out)
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["64GB-1x32GB qwen Q2_0"]["config"]
        self.assertEqual(cfg, golden)

    def test_rerun_keeps_cap_and_explicit_one_opts_out(self):
        code, out, first, _ = self.install("--vram-cap", "0.8")
        self.assertEqual(code, 0, out)
        configs = [("strata-q2_0.json", first)]
        code, out, cfg, _ = self.install("--context", "65536", configs=configs)
        self.assertEqual(code, 0, out)
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "0.8")
        self.assertIn("args --vram-frac", out)
        code, out, cfg, _ = self.install("--vram-cap", "1", configs=configs)
        self.assertEqual(code, 0, out)
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "1.0")
        self.assertEqual(cfg["args"].count("--vram-frac"), 1)

    def test_config_choices_round_trip_fraction(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            setup.write_config(p, {"exe": str(Path(d) / setup.EXE),
                                   "args": ["--vram-frac", "0.875", "--vram-reserve-mib", "3000"]})
            ch = setup.choices_from_config(p)
            self.assertEqual(ch["vram_cap"], 0.875)
            self.assertEqual(ch["vram_reserve_mib"], 3000)
            setup.write_config(p, {"args": ["--vram-frac", "0.95", "--vram-frac", "0.8"]})
            self.assertEqual(setup.choices_from_config(p)["vram_cap"], 0.8)  # last wins, never relax on adoption
            new = {"args": ["--kv", "int8"]}
            setup.carry_over(json.loads(p.read_text(encoding="utf-8")), new)
            self.assertEqual(setup.last_flag_value(new["args"], "--vram-frac"), "0.8")
            setup.write_config(p, {"args": []})
            self.assertIsNone(setup.choices_from_config(p)["vram_cap"])

    def start(self, args, keep):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / setup.EXE
            exe.write_bytes(b"")
            p = Path(d) / "strata-q2_0.json"
            setup.write_config(p, {"exe": str(exe), "args": args, "gpu": 0, "gpus_asked": True})
            before = p.read_bytes()
            with mock.patch.object(setup, "gpus", return_value=nvidia()), \
                    mock.patch.object(setup, "out", return_value=""), \
                    mock.patch.object(setup, "is_wsl", return_value=False), \
                    mock.patch.object(setup, "ensure_engine_for", side_effect=lambda cards, path, cfg, yes: cfg), \
                    mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.start(p, None, yes=True, keep=keep), 0)
            self.assertEqual(Path(call.call_args[0][0][1]), ROOT / "serve" / "server.py")
            return json.loads(p.read_text(encoding="utf-8")), before == p.read_bytes()

    def test_start_updates_all_cap_occurrences_without_starting_a_process(self):
        cfg, unchanged = self.start(["--kv", "int8", "--vram-frac", "0.6", "--vram-frac", "0.9"],
                                    {"vram_cap": 0.8})
        self.assertFalse(unchanged)
        self.assertEqual(cfg["args"], ["--kv", "int8", "--vram-frac", "0.8"])
        self.assertNotIn("vram_cap", cfg)
        cfg, unchanged = self.start(["--kv", "int8"], {"vram_cap": None})
        self.assertTrue(unchanged)
        self.assertEqual(cfg["args"], ["--kv", "int8"])

    def installed_main(self, extra, calibrate=None):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            setup.write_config(p, {"exe": str(Path(d) / setup.EXE), "args": ["--kv", "int8"]})
            with mock.patch.object(sys, "argv", ["setup.py", "--vram-cap", "0.8", *extra]), \
                    mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                    mock.patch.object(setup, "installed_configs", return_value=[p]), \
                    mock.patch.object(setup, "update_installed_engine"), \
                    mock.patch.object(setup, "gpus") as probe, \
                    mock.patch.object(setup, "start", return_value=0) as start, \
                    mock.patch.object(setup, "calibrate_config", side_effect=calibrate or AssertionError("calibrated")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
                probe.assert_not_called()
            return start.call_args, json.loads(p.read_text(encoding="utf-8"))

    def test_cli_for_installed_model_passes_cap_to_start(self):
        call, _ = self.installed_main([])
        self.assertEqual(call.kwargs["keep"]["vram_cap"], 0.8)

    def test_calibration_sees_cap_before_any_engine_can_start(self):
        def calibrate(path):
            cfg = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "0.8")
            return True
        call, cfg = self.installed_main(["--calibrate", "--no-start"], calibrate)
        self.assertIsNone(call)
        self.assertEqual(setup.flag_value(cfg["args"], "--vram-frac"), "0.8")

    def test_server_forwards_engine_flag_unchanged(self):
        from serve.server import engine_args
        args = ["--pack", str(ROOT / "packs" / "test"), "--vram-frac", "0.8"]
        self.assertEqual(engine_args({"args": args}), args)


if __name__ == "__main__":
    unittest.main()
