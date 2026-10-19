"""Installer tests for the official DFlash2 option and its shared output head.

These use tiny synthetic GGUFs and temporary folders; they need no GPU or downloads.
Run with: .venv/bin/python tools/test_setup_dflash2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402


def write_artifact(path: Path, *, dflash2: bool = True, tensors: int = 3) -> Path:
    w = GGUFWriter()
    w.add("general.architecture", "dflash")
    w.add("general.name", "Official DFlash2 drafter")
    w.add("dflash.embedding_length", 2560)
    w.add("dflash.block_count", 5)
    w.add("dflash.attention.head_count", 32)
    w.add("dflash.attention.head_count_kv", 8)
    w.add("dflash.attention.key_length", 128)
    w.add("dflash.feed_forward_length", 7680)
    w.add("dflash.vocab_size", 248320)
    w.add("dflash.block_size", 8)
    w.add("dflash.mask_token_id", 248077)
    w.add("dflash.target_layers", [5, 15, 25, 35, 45])
    if dflash2:
        w.add("dflash.sample_from_anchor", "false")
        w.add("dflash.conv_kernel_size", 4)
        w.add("dflash.conv_group_size", 32)
        w.add("dflash.selector_rank", 64)
        w.add("dflash.selector_top_k", 16)
        w.add("dflash.attention.sliding_window", 2048)
    w.add_bf16("blk.0.attn_norm.weight", np.ones((1, 32)), shape=[32])
    w.add_bf16("selector_hidden.weight", np.ones((32, 32)))
    w.add_bf16("fc.weight", np.ones((32, 32)))
    for i in range(3, tensors):
        w.add_bf16(f"norm-{i}.weight", np.ones((32,)), shape=[32])
    return w.write(path)


class ArtifactNote(unittest.TestCase):
    def test_reads_dflash2_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            note = setup.dflash2_artifact_note(write_artifact(Path(d) / "dflash2.gguf"))
        self.assertIn("DFlash2", note)
        self.assertIn("trained block 8", note)
        self.assertIn("selector top-16", note)
        self.assertIn("local window 2048", note)
        self.assertIn("taps [5, 15, 25, 35, 45]", note)

    def test_refuses_plain_dflash(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "not a DFlash2 artifact"):
                setup.dflash2_artifact_note(write_artifact(Path(d) / "plain.gguf", dflash2=False))

    def test_refuses_unreadable_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "broken.gguf"
            path.write_bytes(b"not GGUF")
            with self.assertRaisesRegex(ValueError, "not readable as GGUF"):
                setup.dflash2_artifact_note(path)


class DrafterChoice(unittest.TestCase):
    def test_setup_owns_new_keys_and_drops_old_keys(self):
        self.assertIn("dflash2", setup.SETUP_KEYS)
        self.assertIn("dflash2_quant", setup.SETUP_KEYS)
        self.assertIn("dflash", setup.SETUP_KEYS)  # old config keys are consumed during migration
        old = {"dflash": "old.gguf", "dflash_quant": "q4", "dflash_head": "original"}
        new = {"drafter": "mtp", "args": []}
        setup.carry_over(old, new)
        self.assertNotIn("dflash", new)
        self.assertNotIn("dflash_quant", new)
        self.assertNotIn("dflash_head", new)

    def test_q8_recommended_and_quant_options(self):
        with patch.object(setup, "say"), patch.object(setup, "ask", return_value="1"):
            self.assertEqual(setup.choose_drafter(None, "auto", None, True), ("dflash2", "q8"))
        with patch.object(setup, "say"), patch.object(setup, "ask", return_value="2"):
            self.assertEqual(setup.choose_drafter("dflash2", None, None, False), ("dflash2", "q4"))
        for quant in setup.DFLASH2_QUANTS:
            self.assertEqual(setup.choose_drafter("dflash2", None, quant, True), ("dflash2", quant))
        with patch.object(setup, "fail", side_effect=ValueError):
            with self.assertRaises(ValueError):
                setup.choose_drafter("mtp", None, "q4", True)

    def test_dflash2_uses_shared_target_head_and_no_dflash_head_flags(self):
        args = setup.drafter_args(Path("draft-q8.gguf"), None)
        self.assertIn("--dflash", args)  # engine-level DFlash runtime option
        self.assertIn("--dflash-block", args)
        self.assertEqual(args[args.index("--dflash-block") + 1], "3")
        self.assertEqual(args[args.index("--dflash-window") + 1], "0")
        self.assertNotIn("--dflash-head", args)
        self.assertNotIn("--dflash-vocab", args)
        self.assertNotIn("--mtp", args)
        self.assertNotIn("--dflash", setup.drafter_args(None, Path("mtp/rt")))

    def test_vram_estimate_scales_with_context_and_weight_quant(self):
        short = setup.dflash2_vram_estimate_gb(32768, "q8")
        long = setup.dflash2_vram_estimate_gb(131072, "q8")
        bf16 = setup.dflash2_vram_estimate_gb(131072, "bf16")
        self.assertGreater(long, short)
        self.assertGreater(bf16, long)
        self.assertLess(setup.low_ram_gpu_gb("IQ3_XXS", 16, 131072, "int8", "q8"),
                        setup.low_ram_gpu_gb("IQ3_XXS", 16, 131072, "int8"))

    def test_vram_estimate_covers_observed_128k_allocations(self):
        # RTX 4070 Ti SUPER, CUDA13.4, eager pools: weights+pools+scratch,
        # recorded by the runtime in the 2026-10-19 DFlash2 VRAM audit.
        for quant, observed_mib in {"bf16": 4237.2, "q8": 3827.5, "q4": 3608.8}.items():
            with self.subTest(quant=quant):
                self.assertGreaterEqual(setup.dflash2_vram_estimate_gb(131072, quant) * 1e9,
                                        observed_mib * 1024 * 1024)

    def test_saved_dflash2_choice_and_legacy_migration(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "strata-iq3_xxs.json"
            cfg.write_text(json.dumps({"args": ["--dflash", "draft-q8.gguf"], "drafter": "dflash2",
                                      "dflash2": "draft-q8.gguf", "dflash2_quant": "q8", "dflash2_source": "auto"}))
            saved = setup.choices_from_config(cfg)
            self.assertEqual((saved["drafter"], saved["dflash2"], saved["dflash2_quant"]),
                             ("dflash2", "auto", "q8"))
            cfg.write_text(json.dumps({"args": ["--dflash", "old-dflash.gguf"], "drafter": "dflash"}))
            self.assertEqual(setup.choices_from_config(cfg)["drafter"], "mtp")

    def test_off_and_mtp_keep_their_paths(self):
        self.assertEqual(setup.choose_drafter("none", None, None, True), ("none", None))
        self.assertEqual(setup.choose_drafter("mtp", None, None, True), ("mtp", None))
        self.assertNotIn("--dflash", setup.drafter_args(None, None))
        self.assertNotIn("--dflash", setup.drafter_args(None, Path("mtp/rt")))


class Preparation(unittest.TestCase):
    def test_manual_bf16_and_q8_conversion_cache(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            source = write_artifact(data / "custom-dflash2.gguf")
            self.assertEqual(setup.prepare_dflash2(data, [data], str(source), "bf16", {}), source)

            def convert(cmd, **kwargs):
                from dflash_quantize import export
                export(cmd[2], cmd[cmd.index("-o") + 1], cmd[cmd.index("--type") + 1])

            with patch.object(setup, "download", side_effect=AssertionError("manual source must not download")), \
                    patch.object(setup, "run", side_effect=convert) as run:
                output = setup.prepare_dflash2(data, [data], str(source), "q8", {})
                self.assertTrue(output.is_file())
                self.assertEqual(setup.prepare_dflash2(data, [data], str(source), "q8", {}), output)
                self.assertEqual(run.call_count, 1)

    def test_auto_download_uses_pinned_revision_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            checkpoint = b"synthetic DFlash2 checkpoint"
            digest = hashlib.sha256(checkpoint).hexdigest()
            fetched = []

            def fetch(url, path, what=None):
                fetched.append(url)
                path.write_bytes(checkpoint if path.name == "model.safetensors" else b"{}")

            def convert(cmd, **kwargs):
                out = Path(cmd[cmd.index("-o") + 1])
                w = GGUFWriter()
                w.add("general.architecture", "dflash")
                w.add("general.name", "DFlash2")
                for key, value in (("embedding_length", 2560), ("block_count", 5), ("attention.head_count", 32),
                                   ("attention.head_count_kv", 8), ("attention.key_length", 128),
                                   ("feed_forward_length", 7680), ("vocab_size", 248320), ("block_size", 8),
                                   ("mask_token_id", 248077), ("conv_kernel_size", 4), ("conv_group_size", 32),
                                   ("selector_rank", 64), ("selector_top_k", 16), ("attention.sliding_window", 2048)):
                    w.add("dflash." + key, value)
                w.add("dflash.sample_from_anchor", "false")
                w.add("dflash.target_layers", [5, 15, 25, 35, 45])
                w.add("dflash.source.sha256", digest)
                for i in range(81):
                    w.add_bf16(f"norm-{i}.weight", np.ones((32,), np.float32), shape=[32])
                w.write(out)

            with patch.object(setup, "download", side_effect=fetch), patch.object(setup, "run", side_effect=convert), \
                    patch.object(setup, "DFLASH2_SHA256", digest):
                output = setup.prepare_dflash2(data, [data], "auto", "bf16", {})
                self.assertTrue(output.is_file())
                self.assertEqual(len(fetched), 2)
                revision = setup.HF_REVISIONS[setup.DFLASH2_REPO]
                self.assertTrue(all(revision in url for url in fetched))
                self.assertEqual(setup.prepare_dflash2(data, [data], "auto", "bf16", {}), output)
                self.assertEqual(len(fetched), 2)


if __name__ == "__main__":
    unittest.main()
