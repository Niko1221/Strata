"""#7119: when the engine package of a model shape (Inkling's `strata-inkling`) cannot be installed, the app showed "The engine for Inkling-2-Mini could not be
installed - See the line above. Nothing was downloaded ..." and there was no line above it (the app shows the stopping line and its hint only).

The hint now carries the real reason: the last line `shape_engine.ensure` said (download error, checksum, no package for this PC, the feed unreachable), and, when a
package was unpacked by hand into a wrong place, one sentence about the folder with the right place
(`<components>/shapes/<shape>/engine/<package>/`, `engine-component.json` directly inside).

    python -m pytest tests/terminal/test_shape_engine_failure_reason_7119.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import engine_component as ec, shape_engine  # noqa: E402

SHAPE = "inkling2"
COMP = "strata-inkling"


class Folders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.comp = Path(self.tmp.name)
        self.place = ec.engine_dir(shape_engine.root(SHAPE, self.comp))

    def note(self):
        return shape_engine.hand_install_note(SHAPE, self.comp)

    def test_nothing_unpacked_says_nothing(self):
        self.assertEqual(self.note(), "")
        self.place.mkdir(parents=True)
        self.assertEqual(self.note(), "")

    def test_a_package_folder_without_a_manifest(self):
        (self.place / "0.1.66-windows-x86_64-cuda").mkdir(parents=True)
        text = self.note()
        self.assertIn("0.1.66-windows-x86_64-cuda cannot be used", text)
        self.assertIn("engine-component.json is missing", text)
        self.assertIn(str(self.place), text)

    def test_one_folder_too_deep(self):
        deep = self.place / "download" / "0.1.66-windows-x86_64-cuda"
        deep.mkdir(parents=True)
        (deep / ec.MANIFEST).write_text("{}", encoding="utf-8")
        text = self.note()
        self.assertIn("one folder too deep", text)
        self.assertIn("0.1.66-windows-x86_64-cuda", text)

    def test_the_shape_engine_in_the_strata_engine_folder(self):
        wrong = ec.engine_dir(self.comp) / "0.1.66-windows-x86_64-cuda"
        wrong.mkdir(parents=True)
        (wrong / ec.MANIFEST).write_text(json.dumps({"schema_version": 1, "id": COMP, "shape": SHAPE}), encoding="utf-8")
        text = self.note()
        self.assertIn("Strata engine's folder", text)
        self.assertIn(str(self.place), text)

    def test_only_our_own_folders_are_read(self):
        """The Strata engine folder is read for its package manifests only; an unrelated folder or file in it is not an error."""
        (ec.engine_dir(self.comp) / "other").mkdir(parents=True)
        (ec.engine_dir(self.comp) / "other" / ec.MANIFEST).write_text(json.dumps({"id": "strata-engine"}), encoding="utf-8")
        self.assertEqual(self.note(), "")


class Hint(unittest.TestCase):
    """setup.shape_engine_fetch: the stopping hint says why, not "See the line above"."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patches = [mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": self.tmp.name}),
                   mock.patch.dict(os.environ, {"STRATA_ALLOW_NO_ENGINE": "0"}),
                   mock.patch.object(setup, "SHAPE_ENGINE", {"shape": SHAPE, "title": "Inkling-2-Mini"}),
                   mock.patch.object(setup, "say", lambda *a, **k: None),
                   mock.patch("strata_cli.update_cmd._feed_source", lambda *a, **k: "https://feed.invalid/")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.events = []
        p = mock.patch.object(setup, "event", lambda name, **f: self.events.append((name, f)))
        p.start()
        self.addCleanup(p.stop)

    def stop_with(self, said):
        def ensure(shape, source, *, emit, **kw):
            for line in said:
                emit(line)
            return None
        with mock.patch.object(shape_engine, "ensure", ensure):
            with self.assertRaises(SystemExit):
                setup.shape_engine_fetch(mock.Mock(feed=None))
        errors = [f for n, f in self.events if n == "error"]
        self.assertEqual(len(errors), 1)
        return errors[0]

    def test_the_download_error_is_in_the_hint(self):
        err = self.stop_with(["Inkling-2-Mini needs its own engine. It downloads separately, and the Strata engine stays as it is.",
                              "The engine for Inkling-2-Mini could not be downloaded: the checksum of the download does not match"])
        self.assertEqual(err["message"], "the engine for Inkling-2-Mini could not be installed")
        self.assertNotIn("See the line above", err["hint"])
        self.assertIn("could not be downloaded: the checksum of the download does not match.", err["hint"])
        self.assertIn("Nothing was downloaded for the model.", err["hint"])

    def test_the_first_line_is_not_taken_for_the_reason(self):
        err = self.stop_with(["Inkling-2-Mini needs its own engine. It downloads separately, and the Strata engine stays as it is.",
                              "The update feed has no engine for Inkling-2-Mini for this computer."])
        self.assertIn("no engine for Inkling-2-Mini for this computer.", err["hint"])
        self.assertNotIn("downloads separately", err["hint"])

    def test_a_wrong_folder_adds_the_right_one(self):
        place = ec.engine_dir(shape_engine.root(SHAPE, Path(self.tmp.name)))
        (place / "0.1.66-windows-x86_64-cuda").mkdir(parents=True)
        err = self.stop_with(["The engine for Inkling-2-Mini could not be downloaded: no network"])
        self.assertIn("no network.", err["hint"])
        self.assertIn("A package unpacked by hand", err["hint"])
        self.assertIn(str(place), err["hint"])

    def test_no_line_at_all_keeps_the_old_hint(self):
        err = self.stop_with([])
        self.assertEqual(err["hint"], shape_engine.WORDS["setup_failed_hint"])


if __name__ == "__main__":
    unittest.main()
