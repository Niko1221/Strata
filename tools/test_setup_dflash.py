"""Tests for setup.py's DFlash choice (docs/DFLASH.md): the artifact's metadata note reads a real
dflash-architecture GGUF (written in-memory by tools/gguf_writer.py) and refuses a non-dflash one; the
config's setup-owned keys carry "dflash".  Pure file work in a temporary folder - no GPU, no downloads,
no prompts.

    python -m unittest tools.test_setup_dflash
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402


def write_artifact(path: Path, arch: str = "dflash") -> Path:
    """A minimal drafter artifact: the metadata the note reads plus one real tensor."""
    w = GGUFWriter()
    w.add("general.architecture", arch)
    w.add("general.name", "DFlash drafter" if arch == "dflash" else "something else")
    w.add("dflash.embedding_length", 2560)
    w.add("dflash.block_count", 5)
    w.add("dflash.attention.head_count", 24)
    w.add("dflash.attention.head_count_kv", 2)
    w.add("dflash.attention.key_length", 256)
    w.add("dflash.feed_forward_length", 7680)
    w.add("dflash.vocab_size", 248320)
    w.add("dflash.block_size", 7)
    w.add("dflash.mask_token_id", 248077)
    w.add("dflash.target_layers", [3, 15, 23, 35, 43])
    w.add_bf16("blk.0.attn_norm.weight", np.ones((1, 2560)), shape=[2560])
    return w.write(path)


class ArtifactNote(unittest.TestCase):
    def test_reads_the_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            note = setup.dflash_artifact_note(write_artifact(Path(d) / "dflash.gguf"))
            self.assertIn("5 draft layers", note)
            self.assertIn("trained block 7", note)
            self.assertIn("mask 248077", note)
            self.assertIn("taps [3, 15, 23, 35, 43]", note)

    def test_refuses_a_non_dflash_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "not a DFlash artifact"):
                setup.dflash_artifact_note(write_artifact(Path(d) / "other.gguf", arch="llama"))

    def test_refuses_a_non_gguf_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "junk.gguf"
            p.write_bytes(b"not a gguf file at all")
            with self.assertRaisesRegex(ValueError, "not readable as GGUF"):
                setup.dflash_artifact_note(p)

    def test_tolerates_missing_optional_keys(self):
        with tempfile.TemporaryDirectory() as d:
            w = GGUFWriter()
            w.add("general.architecture", "dflash")
            w.write(p := Path(d) / "bare.gguf")
            note = setup.dflash_artifact_note(p)
            self.assertIn("? draft layers", note)
            self.assertNotIn("mask", note)


class ConfigKey(unittest.TestCase):
    def test_setup_owns_the_key(self):
        # carry_over keeps setup-written keys across a re-run's config rewrite: "dflash" must be setup's
        self.assertIn("dflash", setup.SETUP_KEYS)


if __name__ == "__main__":
    unittest.main()
