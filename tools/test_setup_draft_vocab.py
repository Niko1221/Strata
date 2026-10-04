"""Tests for setup.py's draft-vocabulary choice (#287): a setup run again without --draft-vocab keeps the subset the
model's config chose before (cyrillic, en), and refresh_draft_vocab copies the chosen shipped subset.  Pure file work
in a temporary folder - no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_draft_vocab
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class SavedChoice(unittest.TestCase):
    def test_a_config_keeps_its_choice(self):
        with tempfile.TemporaryDirectory() as d:
            for choice in setup.DRAFT_VOCABS:
                p = Path(d) / "strata-x.json"
                p.write_text(json.dumps({"args": [], "draft_vocab": choice}), encoding="utf-8")
                self.assertEqual(setup.saved_draft_vocab(p), choice)

    def test_no_choice_without_a_valid_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(setup.saved_draft_vocab(Path(d) / "missing.json"))
            for text in ("{}", "[1]", "not json", json.dumps({"draft_vocab": "klingon"})):
                p = Path(d) / "strata-y.json"
                p.write_text(text, encoding="utf-8")
                self.assertIsNone(setup.saved_draft_vocab(p), text)


class Refresh(unittest.TestCase):
    def test_cyrillic_replaces_a_shipped_subset_and_stays(self):
        if not (ROOT / "data" / setup.DRAFT_VOCABS["cyrillic"]).exists():
            self.skipTest("data/draft_vocab_cyrillic.bin is not in this checkout")
        want = hashlib.sha256((ROOT / "data" / setup.DRAFT_VOCABS["cyrillic"]).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            (rt / "draft_vocab.bin").write_bytes((ROOT / "data" / setup.DRAFT_VOCABS["cjk"]).read_bytes())
            setup.refresh_draft_vocab(rt, "cyrillic")
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)
            setup.refresh_draft_vocab(rt, "cyrillic")                       # again: nothing changes
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)


class SmallCardNote(unittest.TestCase):
    """#474: a card under 14 GB is told about a smaller draft subset - a note only, and not when one was chosen."""

    def test_note_on_a_small_card_only(self):
        note = setup.draft_vocab_note(12.0, None)
        self.assertTrue(note)
        self.assertIn("--draft-vocab en", " ".join(note))
        self.assertIn("the draft head does not fit", " ".join(note))
        for vram, chosen in ((16.0, None), (24.0, None), (12.0, "en"), (12.0, "cyrillic"), (12.0, "cjk"), (0.0, None)):
            self.assertEqual(setup.draft_vocab_note(vram, chosen), [], (vram, chosen))

    def test_sizes_follow_the_shipped_subsets(self):
        # the MiB the note gives scale with the subsets' token counts (4 bytes per token id in data/)
        sizes = {}
        for choice, name in setup.DRAFT_VOCABS.items():
            p = ROOT / "data" / name
            if not p.exists():
                self.skipTest(f"data/{name} is not in this checkout")
            sizes[choice] = p.stat().st_size // 4
        for choice in ("en", "cyrillic"):
            want = setup.DRAFT_VOCAB_MIB["cjk"] * sizes[choice] / sizes["cjk"]
            self.assertAlmostEqual(setup.DRAFT_VOCAB_MIB[choice], want, delta=3)


if __name__ == "__main__":
    unittest.main()
