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


if __name__ == "__main__":
    unittest.main()
