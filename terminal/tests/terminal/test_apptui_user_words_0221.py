"""0221 APPTUI: four things a user saw and could not make sense of (found in the end-to-end pass).

a. `strata launch --plan` for a model with its own engine (Inkling-2-Mini, Hy4-Flash) said nothing of the engine download when none was installed (the
   missing-engine error was swallowed and counted as "an engine is here"), and nothing of the free disk.
b. The picture reader was "on (the reader runs on the processor)" in the plan and "pictures gpu" in the running-server line.
c. The picture reader's last figure came back as "Downloading the picture reader" after its check.
d. `strata update` printed "Control ? -> 0.1.66" and a bare "Engine none".
Nothing is started, probed or downloaded."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import update_cmd
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import PictureOrder, Runner
from tests.terminal.launch_support import HW_12, Home, catalog, not_intel


class PlanWords(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        not_intel(self)
        t = tempfile.TemporaryDirectory(prefix="strata-apptui-")
        self.addCleanup(t.cleanup)
        self.base = Path(t.name)

    def runner(self, **kw) -> Runner:
        r = Runner(S.LaunchSettings(family="inkling-2-mini", model="SQ-10", vision="off", context=32768, data_dir=str(self.base / "data"), **kw),
                   catalog=catalog(HW_12), components_dir=str(self.base / "comp"))
        for p in (mock.patch.object(Runner, "installed_state", lambda self: (None, None)),):
            p.start()
            self.addCleanup(p.stop)
        return r

    def test_the_plan_names_the_engine_download_and_the_free_disk(self):
        with mock.patch.object(Catalog, "disk_free", lambda self, path: 54.0):
            plan = self.runner().download_plan()
        self.assertTrue(plan["engine_missing"], plan)
        self.assertIn("The engine for your graphics card is downloaded first", plan["text"])
        self.assertIn("54 GB are free where it goes", plan["text"])
        self.assertNotIn("Not enough room", plan["text"])

    def test_a_drive_that_is_too_small_says_so(self):
        with mock.patch.object(Catalog, "disk_free", lambda self, path: 3.0):
            plan = self.runner().download_plan()
        self.assertIn("Not enough room: 3 GB are free where it goes", plan["text"])
        self.assertIn("--data-dir", plan["text"])


class PictureWords(unittest.TestCase):
    def test_the_plan_and_the_running_line_say_the_same(self):
        self.assertEqual(S.PICTURE_WORDS["cpu"], "on (the picture reader runs on the processor)")
        self.assertEqual(S.PICTURE_WORDS["gpu"], "on (the picture reader runs on the graphics card)")
        for key in ("cpu", "gpu"):
            line = Runner.describe_running({"name": "e2e-main", "port": 18780, "choices": {"context": 32768, "vision": key, "access": "local"}})
            self.assertNotIn("pictures gpu", line)
            self.assertNotIn("pictures cpu", line)
            self.assertIn("pictures on, read by the " + ("processor" if key == "cpu" else "graphics card"), line)

    def test_the_last_reader_figure_is_not_shown_again_after_its_check(self):
        order = PictureOrder()
        full = {"percent": 100.0}
        part = {"percent": 40.0}
        self.assertFalse(order.stale("download", {"percent": 100.0}))
        self.assertFalse(order.stale("verify", full))                # the model's check
        self.assertFalse(order.stale("pictures", part))
        self.assertFalse(order.stale("pictures", full))              # the reader is complete: shown once
        self.assertFalse(order.stale("pictures", full))              # (still before its check)
        self.assertFalse(order.stale("verify", full))                # the reader's check
        self.assertTrue(order.stale("pictures", full))               # the same figure again: not news
        self.assertFalse(order.stale("pictures", part))              # a new, partial download line is shown


class UpdateWords(unittest.TestCase):
    def test_control_has_no_question_mark(self):
        with mock.patch.object(update_cmd, "_source_version", return_value="0.1.66"):
            text = update_cmd._line("Control", {"component": "control", "status": "updated", "installed_before": None, "installed_after": "0.1.66"})
        self.assertNotIn("?", text)
        self.assertIn("0.1.66 installed as a component", text)
        self.assertIn("version 0.1.66", text)
        text = update_cmd._line("Control", {"component": "control", "status": "updated", "installed_before": "0.1.65", "installed_after": "0.1.66"})
        self.assertIn("0.1.65 -> 0.1.66   updated", text)

    def test_an_engine_line_with_nothing_installed_says_what_engine_it_is(self):
        text = update_cmd._line("Engine", {"component": "engine", "status": "skipped", "installed_before": None, "note": "no build for this card"})
        self.assertIn("none installed (the Strata engine, which runs the Qwen models)", text)
        self.assertIn("no build for this card", text)


if __name__ == "__main__":
    unittest.main()
