"""Optional Arabic output for the `strata` command (opt-in; default path stays English)."""
from __future__ import annotations

import builtins
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from strata_cli import i18n


class ArabicCli(unittest.TestCase):
    def test_default_language_is_english_and_output_is_untouched(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(i18n.enabled())
            self.assertEqual(i18n.tr("Nothing to download."), "Nothing to download.")
            self.assertEqual(i18n.tr("trials    {} of {}"), "trials    {} of {}")

    def test_language_detection_from_supported_variables(self):
        for value in ("ar", "ar-SA", "ar_EG"):
            with patch.dict(os.environ, {"LANG": value}, clear=True):
                self.assertTrue(i18n.enabled())
        with patch.dict(os.environ, {"STRATA_CLI_LANG": "en", "LANG": "ar"}, clear=True):
            self.assertFalse(i18n.enabled())  # an explicit choice overrides the locale
        with patch.dict(os.environ, {"LANG": "de_DE"}, clear=True):
            self.assertFalse(i18n.enabled())

    def test_catalog_keeps_slots_and_does_not_translate_structured_output(self):
        with patch.dict(os.environ, {"STRATA_CLI_LANG": "ar"}, clear=True):
            self.assertNotEqual(i18n.tr("Nothing to download."), "Nothing to download.")
            text = i18n.tr("trials    {} of {}")
            self.assertEqual(text.count("{}"), 2, text)
            self.assertEqual(i18n.tr('{"key": "Nothing to download."}'), '{"key": "Nothing to download."}')
            self.assertEqual(i18n.tr("You said: Nothing to download."), "You said: Nothing to download.")
            self.assertEqual(i18n.tr(42), 42)
            self.assertTrue(all(k.count("{}") == v.count("{}") for k, v in i18n.CATALOG.items()))

    def test_formatted_line_is_translated_with_its_values(self):
        with patch.dict(os.environ, {"STRATA_CLI_LANG": "ar"}, clear=True):
            self.assertEqual(i18n.tr("trials    3 of 9"), "التجارب   3 من 9")
            self.assertEqual(i18n.tr("job abc -> done"), "المهمة abc -> done")

    def test_help_text_is_translated_only_when_requested(self):
        from strata_cli.cli import parser
        with patch.dict(os.environ, {"STRATA_CLI_LANG": "ar"}, clear=True):
            self.assertIn("عملاء خدمة Strata", parser().format_help())
            # Subcommand option descriptions are translated too, not just the top-level one.
            for english in ("requested concurrent engine slots", "show changes without saving",
                            "server root; remote connections require HTTPS", "write Markdown to a new file"):
                self.assertNotEqual(i18n.tr(english), english, english)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("Strata serving and terminal clients", parser().format_help())

    def test_install_changes_print_only_when_arabic_is_requested(self):
        original = builtins.print
        try:
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(i18n.install())
                buf = io.StringIO()
                with redirect_stdout(buf):
                    print("Nothing to download.")
                self.assertEqual(buf.getvalue(), "Nothing to download.\n")
            with patch.dict(os.environ, {"STRATA_CLI_LANG": "ar"}, clear=True):
                self.assertTrue(i18n.install())
                self.assertFalse(i18n.install())  # idempotent
                buf = io.StringIO()
                with redirect_stdout(buf):
                    print("Nothing to download.")
                self.assertIn("لا شيء للتنزيل", buf.getvalue())
        finally:
            builtins.print = original


if __name__ == "__main__":
    unittest.main()
