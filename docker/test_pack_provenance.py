"""Pack provenance guard in docker/bootstrap-model.sh (offline: fake cache and work dirs).

The quant vocabulary is shared between releases (Qwen and Swift both ship IQ3_XXS), so a pack
built from one release's shards could otherwise be loaded by the other - silent expert
corruption. bootstrap-model.sh compares experts.bin.src.json against the shards resolved for
the selection and dies on mismatch. Run: python -m unittest discover -s docker -p 'test_*.py'."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWIFT = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
SWIFT_SHARDS = ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]
QWEN_SHARDS = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
               "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"]


class PackProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.cache = base / "cache"
        self.work = base / "work"
        snap = self.cache / ("models--" + SWIFT.replace("/", "--")) / "snapshots" / "abcd1234"
        snap.mkdir(parents=True)
        for name in SWIFT_SHARDS:
            (snap / name).write_bytes(b"gguf")
        self.pack = self.work / "packs" / "iq3_xxs"       # deliberately untagged

    def tearDown(self):
        self.tmp.cleanup()

    def complete_pack(self, shard_names):
        (self.pack / "tokenizer").mkdir(parents=True)
        (self.pack / "tokenizer" / "vocab.json").write_text("{}")
        (self.pack / "experts.bin").write_bytes(b"x")
        (self.pack / "native_experts.txt").write_text("x\n")
        if shard_names is not None:
            (self.pack / "experts.bin.src.json").write_text(json.dumps(
                {"schema": 1, "shards": [{"name": n, "size": 1} for n in shard_names],
                 "native_experts_sha256": "0" * 64, "bytes": 1}))

    def bootstrap_check(self):
        env = {**os.environ, "STRATA_MODEL": "IQ3_XXS", "STRATA_HF_REPO": SWIFT,
               "STRATA_HF_CACHE": str(self.cache), "STRATA_WORK": str(self.work),
               "STRATA_PACK_DIR": str(self.pack)}
        env.pop("STRATA_HF_REV", None)
        return subprocess.run(["bash", str(HERE / "bootstrap-model.sh"), "check"],
                              capture_output=True, text=True, env=env)

    def test_foreign_pack_is_refused(self):
        self.complete_pack(QWEN_SHARDS)                    # a Qwen pack under a Swift launch
        p = self.bootstrap_check()
        self.assertEqual(p.returncode, 1)
        self.assertIn("another release's shards", p.stderr)
        self.assertIn(QWEN_SHARDS[0], p.stderr)            # names the offending shard

    def test_matching_pack_passes(self):
        self.complete_pack(SWIFT_SHARDS)
        p = self.bootstrap_check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("pack provenance: ok", p.stdout)

    def test_legacy_pack_without_src_warns_and_passes(self):
        self.complete_pack(None)                           # pre-src.json pack
        p = self.bootstrap_check()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("provenance is its directory name only", p.stderr)
        self.assertNotIn("provenance: ok", p.stdout)       # warned, not certified


if __name__ == "__main__":
    unittest.main()
