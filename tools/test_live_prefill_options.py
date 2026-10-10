"""Native CLI rejection checks; no pack exists and no model is loaded."""
import argparse
import os
import pathlib
import subprocess
import unittest


class Options(unittest.TestCase):
    def run_native(self, *args, **environment):
        env = dict(os.environ)
        for key in ("STRATA_KV_GROW", "STRATA_KV_STAGE_OWN", "STRATA_FS_SLOTS"):
            env.pop(key, None)
        env.update(environment)
        return subprocess.run([str(self.binary), "--pack", str(self.missing), *args],
                              env=env, capture_output=True, text=True, timeout=20)

    def test_invalid_integer(self):
        for text in ("-1", "129", "1.5", "junk", "+0", "", "99999999999999999999"):
            with self.subTest(text=text):
                result = self.run_native("--live-prefill-min-retained", text)
                self.assertEqual(result.returncode, 2)
                self.assertIn("requires an integer from 0 to 128", result.stderr)

    def test_unsupported_scope(self):
        for extra, env in (([], {}), (["--live-memory", "--no-prefill-borrow"], {}),
                           (["--live-memory", "--prefill", "0"], {}),
                           (["--live-memory", "--prefill", "256", "--kv-grow"], {}),
                           (["--live-memory", "--prefill", "256"], {"STRATA_KV_GROW": "1"}),
                           (["--live-memory", "--prefill", "256"], {"STRATA_KV_STAGE_OWN": "0"})):
            with self.subTest(extra=extra, env=env):
                result = self.run_native(*extra, "--live-prefill-min-retained", "0", **env)
                self.assertEqual(result.returncode, 2)
                self.assertIn("requires --live-memory, borrowed prefill", result.stderr)

    def test_valid_explicit_and_effective_fixed_kv(self):
        common = ["--serve", "--live-memory", "--mmap-experts", "--expert-profile", "missing-profile.csv",
                  "--expert-cache", "512", "--prefill", "256"]
        for value, extra, env in (("0", [], {}), ("16", [], {}), ("32", [], {}), ("128", [], {}),
                                  ("0", ["--kv-grow"], {"STRATA_KV_GROW": "0"})):
            with self.subTest(value=value, env=env):
                result = self.run_native(*common, *extra, "--live-prefill-min-retained", value, **env)
                self.assertNotEqual(result.returncode, 0)  # deliberately nonexistent model
                self.assertNotIn("--live-prefill-min-retained requires", result.stderr)
                self.assertIn("guarded prompt borrowing", result.stderr)

    def test_omitted_option_keeps_legacy_scope(self):
        result = self.run_native("--kv-grow", STRATA_KV_STAGE_OWN="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--live-prefill-min-retained requires", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    args, remaining = parser.parse_known_args()
    Options.binary = args.binary.resolve(strict=True)
    Options.missing = Options.binary.parent / "pressure-option-nonexistent-pack"
    if Options.missing.exists():
        raise SystemExit("refuse: the deliberately missing model path unexpectedly exists")
    unittest.main(argv=[__file__, *remaining])
