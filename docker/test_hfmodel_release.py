"""Release axis of docker/hfmodel.py (offline: fake cache in a tmpdir, no GPU, no downloads).

run.sh gained a --release axis (plans/run-default-swift-15-iq3xxs-2026-10.md §5.2): hfmodel is
the single owner of release facts - repo, pack-dir tag, advertised model name, license, PLE
shard. These tests pin those keys; the launcher contract pins the docker line that consumes
them. Run: python -m unittest discover -s docker -p 'test_*.py'."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

SWIFT = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
QWEN = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
CODER = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"


class HfmodelRelease(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)
        self.make_shards(QWEN, ["IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                                "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"])
        self.make_shards(SWIFT, ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                                 "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"])

    def tearDown(self):
        self.tmp.cleanup()

    def make_shards(self, repo, names):
        snap = self.cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abcd1234"
        snap.mkdir(parents=True, exist_ok=True)
        for name in names:
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            (snap / name).write_bytes(b"gguf")

    def run_cli(self, *args, expect_rc=0):
        """stdout only (eval-able); warnings go to stderr."""
        cmd = [sys.executable, str(HERE / "hfmodel.py"), "--cache", str(self.cache)] + list(args)
        p = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(p.returncode, expect_rc, p.stdout + p.stderr)
        return p.stdout

    def run_cli_all(self, *args, expect_rc=0):
        cmd = [sys.executable, str(HERE / "hfmodel.py"), "--cache", str(self.cache)] + list(args)
        p = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(p.returncode, expect_rc, p.stdout + p.stderr)
        return p.stdout + p.stderr

    def test_release_of_selections(self):
        self.assertEqual(self.run_cli("--model", "IQ3_S", "--print", "release").strip(), "qwen")
        self.assertEqual(self.run_cli("--model", "IQ3_S", "--repo", SWIFT,
                                      "--print", "release").strip(), "swift")
        self.assertEqual(self.run_cli("--model", "IQ1_M", "--print", "release").strip(), "coder")
        self.assertEqual(self.run_cli("--model", "IQ3_XXS", "--release", "swift",
                                      "--print", "release").strip(), "swift")
        self.assertEqual(self.run_cli("--model", "IQ3_S", "--repo", "someone/New-GGUF",
                                      "--print", "release").strip(), "")

    def test_quant_conflicts_with_named_release(self):
        out = self.run_cli_all("--release", "qwen", "--model", "IQ1_M", "--print", "shell",
                               expect_rc=1)
        self.assertIn("only released for the expert-pruned 'coder' release", out)
        # shared quant names under another named release: allowed, warning on stderr, stdout clean
        self.assertEqual(self.run_cli("--model", "IQ3_XXS", "--release", "swift",
                                      "--print", "release").strip(), "swift")
        self.assertIn("catalogued for 'qwen'",
                      self.run_cli_all("--model", "IQ3_XXS", "--release", "swift",
                                       "--print", "release"))
        # an unknown quant under a named release is allowed (a new quant of that repo)
        self.assertEqual(self.run_cli("--release", "swift", "--model", "IQ9_NEW",
                                      "--print", "release").strip(), "swift")

    def test_unknown_release_named_with_known_ones(self):
        out = self.run_cli_all("--release", "nope", "--model", "IQ3_S", "--print", "release",
                               expect_rc=2)
        self.assertIn("coder", out)      # argparse choices message names the valid set

    def test_shell_keys_per_release(self):
        qwen = self.run_cli("--model", "IQ3_S", "--print", "shell", "--allow-missing")
        self.assertIn("STRATA_FAMILY=qwen", qwen)
        self.assertIn("STRATA_PACK_TAG=''", qwen)
        self.assertIn("STRATA_MODEL_NAME_DEFAULT=qwen3.8-flash-next-iq3_s", qwen)
        swift = self.run_cli("--model", "IQ3_XXS", "--repo", SWIFT, "--print", "shell")
        self.assertIn("STRATA_FAMILY=swift", swift)
        self.assertIn("STRATA_PACK_TAG=swift-", swift)
        self.assertIn("STRATA_MODEL_NAME_DEFAULT=swift-1.5-iq3_xxs", swift)
        self.assertIn("Swift Open License 1.0", swift)

    def test_release_flag_resolves_shards_and_ple_without_repo(self):
        ple = self.run_cli("--model", "IQ3_XXS", "--release", "swift", "--print", "ple",
                           "--rev", "abcd1234").strip()
        self.assertTrue(ple.endswith("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
                        ple)
        shard1 = self.run_cli("--model", "IQ3_XXS", "--release", "swift", "--print", "shard1",
                              "--rev", "abcd1234").strip()
        self.assertIn("models--ukisai--", shard1)

    def test_a_quant_absent_from_a_release_is_never_a_sibling_file(self):
        """Measured 2026-10-05: asking for a quant the release does not have used to return
        ANOTHER quant's shard through a name-blind catch-all glob, so the launcher said 'cached'
        and the engine would have loaded a different quantization than the model id advertised."""
        for model, repo in (("IQ2_XS", SWIFT), ("IQ2_XS", "")):
            args = ("--model", model, "--print", "shard1", "--allow-missing")
            if repo:
                args += ("--repo", repo)
            out = self.run_cli(*args).strip()
            self.assertEqual(out, "", f"{model} under {repo or 'qwen'} resolved a sibling shard")
            keys = self.run_cli("--model", model, *(('--release', 'swift') if not repo else
                                                ('--repo', repo)), "--print", "shell",
                                "--allow-missing")
            self.assertIn("STRATA_CACHED=0", keys)

    def test_json_carries_release(self):
        import json
        out = json.loads(self.run_cli("--model", "IQ3_XXS", "--repo", SWIFT, "--print", "json"))
        self.assertEqual(out["release"], "swift")


if __name__ == "__main__":
    unittest.main()
