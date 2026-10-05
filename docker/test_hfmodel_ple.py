"""Per-family PLE file resolution in docker/hfmodel.py (offline: a fake cache in a tmpdir).

Why this exists: the container picked the PLE table as shard 2 for every release, but Swift 1.5
packs per_layer_token_embd.weight into file 1 (measured on Swift IQ3_XXS on the gfx1101 host,
2026-10-04, bench/results/2026-10-04-iq3s-tuning/swift-launch.log: the engine fatal-errors that
the tensor "is not in" 00002 and reports the table in 00001).  Run:
python -m unittest discover -s docker -p 'test_*.py'."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class HfmodelPle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_shards(self, repo: str, names) -> Path:
        snap = self.cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abcd1234"
        snap.mkdir(parents=True)
        for name in names:
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            (snap / name).write_bytes(b"gguf")
        return snap

    def print_ple(self, model: str, repo: str = "") -> str:
        cmd = [sys.executable, str(HERE / "hfmodel.py"), "--cache", str(self.cache),
               "--model", model, "--print", "ple", "--rev", "abcd1234"]
        if repo:
            cmd += ["--repo", repo]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    def test_qwen_ple_is_shard_two(self):
        self.make_shards("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                        ["IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                         "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"])
        self.assertTrue(self.print_ple("IQ3_S").endswith("00002-of-00002.gguf"))

    def test_swift_ple_is_shard_one(self):
        snap = self.make_shards("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                                ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                                 "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"])
        self.assertEqual(self.print_ple("IQ3_XXS", "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                         str(snap / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"))

    def test_shell_output_carries_ple_file(self):
        self.make_shards("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                        ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                         "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"])
        out = subprocess.run([sys.executable, str(HERE / "hfmodel.py"), "--cache", str(self.cache),
                              "--model", "IQ3_XXS", "--repo",
                              "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                              "--rev", "abcd1234", "--print", "shell"],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("STRATA_PLE_FILE=", out)
        self.assertIn("00001-of-00002.gguf", out.split("STRATA_PLE_FILE=")[1].splitlines()[0])


if __name__ == "__main__":
    unittest.main()
