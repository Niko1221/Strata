"""Tests for setup.py's DFlash choice (docs/DFLASH.md): the artifact's metadata note reads a real
dflash-architecture GGUF (written in-memory by tools/gguf_writer.py) and refuses a non-dflash one; the
config's setup-owned keys carry "dflash".  Pure file work in a temporary folder - no GPU, no downloads,
no prompts.

    python -m unittest tools.test_setup_dflash
"""
from __future__ import annotations

import sys
import json
import hashlib
from unittest.mock import patch
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


class DrafterChoice(unittest.TestCase):
    def test_defaults_and_explicit_choices(self):
        with patch.object(setup, "say"), patch.object(setup, "ask", return_value="1"):
            self.assertEqual(setup.choose_drafter(None, None, None, True), ("mtp", "original"))
            self.assertEqual(setup.choose_drafter(None, "auto", None, True), ("dflash", "original"))
            for q in setup.DFLASH_QUANTS:
                self.assertEqual(setup.choose_drafter("dflash", None, q, True), ("dflash", q))
        with patch.object(setup, "say"), patch.object(setup, "ask", side_effect=["2", "3"]):
            self.assertEqual(setup.choose_drafter(None, None, None, False), ("dflash", "q5"))
        with patch.object(setup, "fail", side_effect=ValueError):
            with self.assertRaises(ValueError): setup.choose_drafter("mtp", None, "q4", True)

    def test_exclusive_arguments_and_saved_choice(self):
        a = setup.drafter_args(Path("drafter-Q4.gguf"), Path("mtp/rt"))
        self.assertIn("--dflash", a)
        self.assertNotIn("--mtp", a)
        self.assertEqual(a[a.index("--dflash-window")+1], "0")
        self.assertNotIn("--dflash", setup.drafter_args(None, Path("mtp/rt")))
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder)/"strata-iq3_xxs.json"
            cfg.write_text(json.dumps({"args": a + ["--max-context", "8192"],
                                      "dflash_quant": "q4", "dflash_source": "auto"}))
            saved = setup.choices_from_config(cfg)
            self.assertEqual((saved["model"], saved["drafter"], saved["dflash"], saved["dflash_quant"]),
                             ("IQ3_XXS", "dflash", "auto", "q4"))
            new = {"args": setup.drafter_args(None, "rt"), "drafter": "mtp"}
            setup.carry_over(json.loads(cfg.read_text()), new)
            self.assertNotIn("dflash_quant", new)
            self.assertNotIn("dflash_source", new)

    def test_model_drafter_off_keeps_lookup_and_saved_configuration(self):
        with patch.object(setup, "say"), patch.object(setup, "ask", return_value="3") as ask:
            self.assertEqual(setup.choose_drafter(None, None, None, False), ("none", "original"))
            self.assertEqual(ask.call_count, 1)
        args = setup.drafter_args(None, None)
        for flag in ("--mtp", "--dflash"):
            self.assertNotIn(flag, args)
        for flag in ("--suffix-draft", "--lookup-chain"):
            self.assertNotIn(flag, args)   # preserve the engine's prompt-lookup defaults
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder)/"strata-iq3_xxs.json"
            cfg.write_text(json.dumps({"args": args, "drafter": "none"}))
            saved = setup.choices_from_config(cfg)
            self.assertEqual(saved["drafter"], "none")
            self.assertIsNone(saved["dflash"])
            self.assertIsNone(saved["dflash_quant"])
            self.assertEqual(setup.choose_drafter(saved["drafter"], saved["dflash"], saved["dflash_quant"], True),
                             ("none", "original"))
            cfg.write_text(json.dumps({"args": setup.drafter_args(None, "mtp/rt")}))
            saved = setup.choices_from_config(cfg)
            self.assertEqual(saved["drafter"], "mtp")
            self.assertIsNone(saved["dflash_quant"])
            self.assertEqual(setup.choose_drafter(saved["drafter"], saved["dflash"], saved["dflash_quant"], True),
                             ("mtp", "original"))
        for path, quant in (("auto", None), (None, "q4")):
            with patch.object(setup, "fail", side_effect=ValueError):
                with self.assertRaises(ValueError): setup.choose_drafter("none", path, quant, True)
        old = {"dflash": "draft.gguf", "dflash_source": "auto", "dflash_quant": "q4", "draft_vocab": "en"}
        new = {"drafter": "none", "args": args}
        setup.carry_over(old, new)
        for key in old: self.assertNotIn(key, new)

    def test_manual_conversion_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            original = write_artifact(data/"custom.gguf")
            self.assertEqual(setup.prepare_dflash(data, [data], str(original), "original", {}), original)
            def convert(cmd, **kwargs):
                from dflash_quantize import export
                export(cmd[2], cmd[cmd.index("-o")+1], cmd[cmd.index("--type")+1])
            with patch.object(setup, "download", side_effect=AssertionError("manual source must not download")), \
                    patch.object(setup, "run", side_effect=convert) as run:
                output = setup.prepare_dflash(data, [data], str(original), "q5", {})
                self.assertTrue(output.is_file())
                self.assertEqual(setup.prepare_dflash(data, [data], str(original), "q5", {}), output)
                self.assertEqual(run.call_count, 1)
                # A readable but wrong tensor directory must also invalidate the cache.
                w = GGUFWriter(); w.add("general.architecture", "dflash")
                w.add("strata.dflash.source_sha256", hashlib.sha256(original.read_bytes()).hexdigest())
                w.add("strata.dflash.quantization", "Q5_0")
                w.add_bf16("wrong-norm.weight", np.ones((2560,), np.float32), shape=[2560]); w.write(output)
                setup.prepare_dflash(data, [data], str(original), "q5", {})
                self.assertEqual(run.call_count, 2)
                output.write_bytes(output.read_bytes()[:-32])
                setup.prepare_dflash(data, [data], str(original), "q5", {})
                self.assertEqual(run.call_count, 3)
                output.write_bytes(b"broken cached GGUF")
                setup.prepare_dflash(data, [data], str(original), "q5", {})
                self.assertEqual(run.call_count, 4)

    def test_pinned_download_and_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder); fetched = []
            checkpoint = b"fake checkpoint for installer control flow"
            def fetch(url, path, what=None):
                fetched.append(url)
                path.write_bytes(checkpoint if path.name == "model.safetensors" else b"{}")
            def convert(cmd, **kwargs):
                output = Path(cmd[cmd.index("-o")+1])
                w = GGUFWriter(); w.add("general.architecture", "dflash")
                w.add("dflash.source.sha256", hashlib.sha256(checkpoint).hexdigest())
                for i in range(58): w.add_bf16(f"norm-{i}", np.ones((32,),np.float32), shape=[32])
                w.write(output)
            with patch.object(setup, "download", side_effect=fetch), patch.object(setup, "run", side_effect=convert), \
                    patch.object(setup, "DFLASH_SHA256", hashlib.sha256(checkpoint).hexdigest()):
                output = setup.prepare_dflash(data, [data], "auto", "original", {})
                self.assertTrue(output.is_file())
                self.assertEqual(len(fetched), 2)
                self.assertTrue(all(setup.HF_REVISIONS[setup.DFLASH_REPO] in u for u in fetched))
                self.assertEqual(setup.prepare_dflash(data, [data], "auto", "original", {}), output)
                self.assertEqual(len(fetched), 2)
            output.unlink()
            with patch.object(setup, "download", side_effect=fetch), \
                    patch.object(setup, "fail", side_effect=ValueError) as fail:
                with self.assertRaises(ValueError): setup.prepare_dflash(data, [data], "auto", "original", {})
                self.assertIn("SHA-256", fail.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
