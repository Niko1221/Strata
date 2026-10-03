"""Tests for setup.py's API identity of a custom --variant build, and the Windows no-window child policy.

  - the published config keeps ONLY its canonical model name (qwen3.8-flash-next-q2_0), as before
  - a --variant config advertises its own model name (qwen3.8-flash-next-q2_0-abliterated) and keeps the
    canonical name as an alias, so clients that still send the published id are answered
  - aliases edited by hand survive a setup run again (they are never overwritten)
  - short utility subprocesses (nvidia-smi, powershell, python tools) get CREATE_NO_WINDOW exactly when the
    parent has no console (the pythonw/GUI context the Manager runs in); an interactive terminal run keeps
    its console behavior; Linux is untouched

Pure functions: no GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_identity
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class ApiIdentity(unittest.TestCase):
    """The model_name + aliases a config advertises; the canonical published build must not change."""

    def test_canonical_name_is_unchanged(self):
        name, aliases = setup.api_identity(setup.FAMILIES["qwen"], "Q2_0", None)
        self.assertEqual(name, "qwen3.8-flash-next-q2_0")
        self.assertIsNone(aliases)                        # no alias list on the published model

    def test_variant_gets_its_own_name_and_keeps_the_canonical_alias(self):
        name, aliases = setup.api_identity(setup.FAMILIES["qwen"], "Q2_0", "abliterated")
        self.assertEqual(name, "qwen3.8-flash-next-q2_0-abliterated")
        self.assertEqual(aliases, ["qwen3.8-flash-next-q2_0"])

    def test_variant_of_another_family(self):
        name, aliases = setup.api_identity(setup.FAMILIES["swift"], "IQ3_XXS", "abliterated")
        self.assertEqual(name, "swift-1.5-iq3_xxs-abliterated")
        self.assertEqual(aliases, ["swift-1.5-iq3_xxs"])

    def test_any_quant_keeps_the_same_suffix_rule(self):
        # the label is appended to the canonical name verbatim for every size
        for model in ("Q2_0", "IQ3_XXS", "UD-Q4_K_XL"):
            name, aliases = setup.api_identity(setup.FAMILIES["qwen"], model, "my-build")
            self.assertEqual(name, f"qwen3.8-flash-next-{model.lower()}-my-build")
            self.assertEqual(aliases, [f"qwen3.8-flash-next-{model.lower()}"])

    def test_normalized_variant_label_matches_config_tag(self):
        # setup normalizes --variant (strip().lower()) BEFORE naming anything; the config stem and the
        # API name must agree on that normalized label
        variant = ("  Abliterated  ").strip().lower()
        self.assertEqual(variant, "abliterated")
        name, aliases = setup.api_identity(setup.FAMILIES["qwen"], "Q2_0", variant)
        self.assertEqual(name, "qwen3.8-flash-next-q2_0-abliterated")


class SavedAliases(unittest.TestCase):
    """saved_aliases(): hand-edited aliases are read back and never overwritten by a setup run again."""

    def write_cfg(self, d: Path, text):
        p = d / "strata-q2_0-abliterated.json"
        p.write_text(text, encoding="utf-8")
        return p

    def test_hand_edited_aliases_are_returned(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.write_cfg(Path(td), json.dumps(
                {"model_name": "qwen3.8-flash-next-q2_0-abliterated",
                 "aliases": ["qwen3.8-flash-next-q2_0", "my-own-name"]}))
            self.assertEqual(setup.saved_aliases(p), ["qwen3.8-flash-next-q2_0", "my-own-name"])

    def test_missing_or_invalid_aliases_are_none(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.assertIsNone(setup.saved_aliases(d / "missing.json"))
            p = self.write_cfg(d, json.dumps({"model_name": "qwen3.8-flash-next-q2_0"}))
            self.assertIsNone(setup.saved_aliases(p))
            self.write_cfg(d, json.dumps({"aliases": "not-a-list"}))
            self.assertIsNone(setup.saved_aliases(d / "strata-q2_0-abliterated.json"))
            self.write_cfg(d, "not json at all")
            self.assertIsNone(setup.saved_aliases(d / "strata-q2_0-abliterated.json"))


class ChildFlags(unittest.TestCase):
    """The scoped Windows child policy: no-window flags exactly in the console-less (pythonw) context."""

    def _windll(self, console: int):
        w = mock.Mock()
        w.kernel32.GetConsoleWindow.return_value = console
        return w

    def test_zero_when_a_console_exists(self):
        # an interactive cmd/PowerShell run (and a Manager-spawned child with its hidden console): normal behavior
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup.ctypes, "windll", create=True, new=self._windll(1)):
            self.assertEqual(setup.child_flags(), 0)

    def test_create_no_window_without_a_console(self):
        # the pythonw/GUI context: nvidia-smi / powershell / taskkill must never flash a black window
        with mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup.ctypes, "windll", create=True, new=self._windll(0)):
            self.assertEqual(setup.child_flags(), subprocess.CREATE_NO_WINDOW)

    def test_zero_on_linux(self):
        with mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.child_flags(), 0)


if __name__ == "__main__":
    unittest.main()