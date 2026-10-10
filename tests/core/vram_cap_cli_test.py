"""CUDA CLI-only CPU checks. All calls return during option validation / missing token input, before GPU calls.

python tests/core/vram_cap_cli_test.py --exe build-capmode/strata.exe --baseline build-vram/strata.exe
The optional baseline compares the no-mode CLI byte-for-byte, except the new help line. No model runs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import unittest

EXE: Path
BASELINE: Path | None = None


class CapModeCLI(unittest.TestCase):
    def run_cli(self, args, *, exe=None, fraction=None):
        env = dict(os.environ)
        env.pop("STRATA_VRAM_FRAC", None)
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        if fraction is not None:
            env["STRATA_VRAM_FRAC"] = fraction
        # This test intentionally requires CUDA builds: HIP's existing arch-defaults query precedes
        # the missing-token guard. Do not run it on a HIP binary and call that CPU-only validation.
        program = exe or EXE
        cache = program.parent / "CMakeCache.txt"
        self.assertIn("STRATA_ENABLE_CUDA:BOOL=ON", cache.read_text(encoding="utf-8"))
        result = subprocess.run([str(program), *args], env=env, capture_output=True, timeout=15)
        return result.returncode, result.stdout, result.stderr

    def test_help(self):
        code, _, err = self.run_cli(["--help"])
        self.assertEqual(code, 0)
        self.assertIn(b"--vram-cap-mode M", err)
        self.assertIn(b"fast (default)", err)
        self.assertIn(b"even cap off", err)

    def test_invalid_modes_and_missing_value(self):
        for value in ("", "slow", "QUALITY", "quality "):
            with self.subTest(value=value):
                code, _, err = self.run_cli(["--vram-cap-mode", value])
                self.assertEqual(code, 2)
                self.assertIn(b"--vram-cap-mode needs fast or quality", err)
                self.assertNotIn(b"--tokens is required", err)
        code, _, err = self.run_cli(["--vram-cap-mode"])
        self.assertEqual(code, 2)
        self.assertIn(b"--vram-cap-mode needs a value", err)

    def test_quality_rejects_unsupported_options_before_gpu(self):
        for extra in (["--mmap-experts"], ["--peer-device", "1"], ["--layer-split", "auto"],
                      ["--expert-cache-device1", "100"], ["--no-pool"], ["--expert-profile", ""],
                      ["--spec", "1"]):
            with self.subTest(extra=extra):
                args = ["--vram-frac", "0.8", "--vram-cap-mode", "quality", "--spec", "4",
                        "--expert-profile", "not-opened.bin", "--expert-cache", "auto", *extra]
                code, _, err = self.run_cli(args)
                self.assertEqual(code, 2)
                self.assertIn(b"--vram-cap-mode quality:", err)
                self.assertNotIn(b"--tokens is required", err)

    def test_quality_overrides_pcie_in_both_orders_even_uncapped(self):
        for fraction in ("1", "0.8"):
            for tail in (["--pcie-frac", "0", "--vram-cap-mode", "quality"],
                         ["--vram-cap-mode", "quality", "--pcie-frac", "0"]):
                with self.subTest(fraction=fraction, tail=tail):
                    code, _, err = self.run_cli(["--vram-frac", fraction, "--spec", "4",
                                                "--expert-profile", "not-opened.bin", *tail])
                    self.assertEqual(code, 2)  # no token input; no model/source/GPU opened
                    self.assertIn(b"--pcie-frac 1, overriding CLI/link/request shares", err)
                    self.assertIn(b"prefill CPU share off", err)
                    self.assertIn(b"--tokens is required", err)

    def test_fraction_environment_and_cli_precedence(self):
        common = ["--vram-cap-mode", "quality", "--spec", "4", "--expert-profile", "not-opened.bin"]
        code, _, err = self.run_cli(common, fraction="0.8")
        self.assertEqual(code, 2)
        self.assertIn(b"--tokens is required", err)
        code, _, err = self.run_cli([*common, "--vram-frac", "1"], fraction="bad")
        self.assertEqual(code, 2)
        self.assertIn(b"--tokens is required", err)  # explicit cap off still permits quality reference
        code, _, err = self.run_cli(common, fraction="bad")
        self.assertEqual(code, 2)
        self.assertIn(b"STRATA_VRAM_FRAC needs a finite fraction", err)

    def test_no_mode_cli_matches_pre_change(self):
        if BASELINE is None:
            self.skipTest("no pre-change CUDA baseline supplied")
        def normalize(result):
            code, out, err = result
            return (code, b"".join(l for l in out.splitlines(keepends=True) if b"--vram-cap-mode M" not in l),
                    b"".join(l for l in err.splitlines(keepends=True) if b"--vram-cap-mode M" not in l))
        for args, env in (([], None), (["--help"], None), (["--version"], None),
                          (["--vram-frac", "1"], None), (["--vram-frac", "0.8"], None),
                          (["--pcie-frac", "0"], None), (["--pcie-frac", "1"], None),
                          ([], "0.8"), ([], "bad"), (["--vram-frac", "1"], "bad")):
            with self.subTest(args=args, env=env):
                before = self.run_cli(args, exe=BASELINE, fraction=env)
                after = self.run_cli(args, fraction=env)
                self.assertEqual(normalize(after), normalize(before))
                if "--help" not in args:
                    fast = self.run_cli([*args, "--vram-cap-mode", "fast"], fraction=env)
                    self.assertEqual(normalize(fast), normalize(before))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    options, remaining = parser.parse_known_args()
    EXE = options.exe.resolve()
    BASELINE = options.baseline.resolve() if options.baseline else None
    unittest.main(argv=[sys.argv[0], *remaining])
