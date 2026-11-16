"""The release gate for a feed model's install, as a test in `python tools/check` (py-tools): a REAL synthetic GGUF installed through setup.py from a signed
test feed (throw-away key) and a local Hugging Face stand-in. See tools/feed_model_install_gate.py for what it does and checks; it catches the Q4-P8 rehearsal's
F2 (KeyError 'mmproj'), F3 (the Flash-Next draft layer fetched for `drafter: none`), F4 (the shipped expert profile of another shape) and F5 (no RAM budget, a
second copy of the experts).

    python -m pytest tools/test_feed_model_install_realfile.py

Needs numpy (no llama.cpp checkout and no STRATA_TEST_LLAMA_CPP any more: the preparation tools use tools/strata_gguf.py, not gguf-py); without it the test is SKIPPED
with that reason, which `tools/check` prints. The engine step (one request) is the gate script's own --engine option and needs a box with an engine built for the file's shape.
"""
from __future__ import annotations

import shutil
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import feed_model_install_gate as gate  # noqa: E402


class InstallFixtureContracts(unittest.TestCase):
    def test_mac_tools_answer_only_the_explicit_hardware_contract(self):
        from strata_control import mac_fit
        with tempfile.TemporaryDirectory() as td:
            env = {"PATH": "original-child-path"}
            real_path = os.environ.get("PATH")
            driver = gate.mac_fixture_tools(env, Path(td), lambda s: None)
            self.assertEqual(os.environ.get("PATH"), real_path)
            self.assertTrue(env["PATH"].endswith(os.pathsep + "original-child-path"))
            def run(args, **kwargs):
                return subprocess.run([sys.executable, str(driver), *args], **kwargs)
            for query, (code, output) in gate.MAC_QUERIES.items():
                result = run(list(query), capture_output=True, text=True, timeout=10)
                self.assertEqual((result.returncode, result.stdout), (code, output))
            for query in (("sysctl", "-n", "unknown"), ("sysctl", "-a"), ("xcode-select", "--install")):
                result = run(list(query), capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
            # Exercise the genuine detector through its subprocess boundary.
            # machine() is the Python build; existing Intel/Rosetta tests stay intact.
            info = mac_fit.mac_info(run=run)
            self.assertTrue(info["arm64"])
            self.assertEqual(info["ram_gb"], 32)
            self.assertEqual(info["clt"], "")
            self.assertEqual(mac_fit.mac_problems(info, need_compiler=False), [])
            self.assertTrue(mac_fit.mac_problems(info, need_compiler=True))

    def test_empty_components_raise_real_engine_missing_without_starting_a_process(self):
        from strata_control import engine_component
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(engine_component.subprocess, "run", side_effect=AssertionError("unexpected engine/probe launch")):
                with self.assertRaises(engine_component.EngineMissing):
                    engine_component.resolve(root=Path(td), full=False)

    def test_backend_specific_f5_and_complete_neutral_f4(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            model = folder / "tiny.gguf"
            model.write_bytes(b"model fixture")
            profile = folder / "expert-profile-1x8.bin"
            blob = b"STRP" + struct.pack("<5I", 1, 1, 8, 8, 8)
            blob += b"".join(struct.pack("<HH", 0, e) for e in range(8)) + struct.pack("<8i", *range(8))
            profile.write_bytes(blob)
            self.assertEqual(gate.profile_problems(profile, (1, 8)), [])
            config = folder / "run.json"
            common = ["--native", str(model), "--expert-profile", str(profile)]
            cfg = dict(exe="component:strata-engine", args=common + ["--resident-budget-gib", "8"])
            config.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(gate.config_problems(config, folder, folder, "", geometry=(1, 8)), [])
            cfg.update(backend="metal", args=common + ["--ple-gguf", str(model), "--mmap-experts"])
            config.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(gate.config_problems(config, folder, folder, "", backend="metal", geometry=(1, 8)), [])
            self.assertIn("--resident-budget-gib", " ".join(gate.config_problems(config, folder, folder, "")))
            for corrupt in (blob[:-1], blob[:24] + blob[28:32] + blob[28:], blob[:-4] + struct.pack("<i", 0)):
                profile.write_bytes(corrupt)
                self.assertTrue(gate.profile_problems(profile, (1, 8)))
            profile.write_bytes(blob)
            for bad_args in (common, common + ["--ple-gguf", "other.gguf", "--mmap-experts"],
                             common + ["--ple-gguf", str(model), "--mmap-experts", "--mtp", "draft"]):
                config.write_text(json.dumps(dict(cfg, args=bad_args)), encoding="utf-8")
                self.assertTrue(gate.config_problems(config, folder, folder, "", backend="metal", geometry=(1, 8)))

    def test_idempotency_snapshot_detects_content_mtime_and_semantic_changes(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            model, profile, config = (folder / name for name in ("m.gguf", "profile.bin", "run.json"))
            model.write_bytes(b"original")
            profile.write_bytes(b"profile")
            cfg = dict(exe="component:strata-engine", args=["--native", str(model), "--expert-profile", str(profile)])
            config.write_text(json.dumps(cfg), encoding="utf-8")
            original = gate.installed_snapshot(config)
            config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            self.assertEqual(gate.installed_snapshot(config), original)
            stat = profile.stat()
            os.utime(profile, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
            self.assertNotEqual(gate.installed_snapshot(config), original)
            os.utime(profile, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            profile.write_bytes(b"changed")
            os.utime(profile, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(gate.installed_snapshot(config), original)
            profile.write_bytes(b"profile")
            os.utime(profile, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            config.write_text(json.dumps(dict(cfg, backend="metal")), encoding="utf-8")
            self.assertNotEqual(gate.installed_snapshot(config), original)


class FeedModelInstallFromRealFile(unittest.TestCase):
    def test_a_small_synthetic_model_installs_through_setup(self):
        try:
            gate.check_gguf_tools()
            import numpy  # noqa: F401
        except gate.NotRun as e:
            self.skipTest(str(e))
        except ImportError:
            self.skipTest("numpy is not installed (tools/q4geo_synth.py needs it)")
        work = Path(tempfile.mkdtemp(prefix="strata-feed-install-"))
        self.addCleanup(shutil.rmtree, work, True)
        try:
            res = gate.check(work, None, gate.TINY, lambda s: None)
        except gate.NotRun as e:
            self.skipTest(str(e))
        self.assertEqual(res["problems"], [], "\n".join(res["out"].splitlines()[-30:]))


if __name__ == "__main__":
    unittest.main()
