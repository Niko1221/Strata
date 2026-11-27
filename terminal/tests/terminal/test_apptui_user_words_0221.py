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


class DiskBeforeStopping(PlanWords):
    """Found on the NVIDIA Linux box: `launch --model SQ-15 --restart` stopped the serving SQ-10 first, then setup failed for lack of disk, and nothing served."""

    def runner_with_a_server(self, free_gb):
        r = self.runner()
        r._stop_first = ["strata-e2e-main"]
        for p in (mock.patch.object(Runner, "download_plan", lambda self: {"gb": 15.0, "parts": [{"kind": "weights", "label": "the model", "gb": 14.3}]}),
                  mock.patch.object(Catalog, "disk_free", lambda self, path: free_gb)):
            p.start()
            self.addCleanup(p.stop)
        return r

    def test_nothing_is_stopped_when_the_model_cannot_be_downloaded(self):
        from strata_cli.launch.runner import LaunchError
        r = self.runner_with_a_server(7.7)
        with mock.patch("strata_control.runtime.stop") as stop:
            with self.assertRaises(LaunchError) as caught:
                r.stop_before_setup()
        stop.assert_not_called()
        self.assertIn("Nothing was stopped: the running model keeps serving.", str(caught.exception))
        self.assertIn("14.3 GB", str(caught.exception))
        self.assertIn("7.7 GB are free", str(caught.exception))

    def test_with_room_the_running_server_is_stopped_as_before(self):
        r = self.runner_with_a_server(60.0)
        with mock.patch("strata_control.runtime.stop") as stop, mock.patch("strata_cli.launch.access.stop_tunnel"):
            r.stop_before_setup()
        stop.assert_called_once()


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


class InklingPictureReader(unittest.TestCase):
    """GRID: an Unsloth GGUF with mmproj-BF16.gguf beside it. strata-inkling reads only its own Vision.strata (`--info mmproj-BF16.gguf`: "GGUF architecture is
    'clip', expected 'inkling2'"), so the projector must never be passed as --vision, and a launch with pictures on and no reader fetches the right file or says plainly
    that images will not work."""

    def test_the_projector_beside_a_gguf_is_not_a_reader(self):
        import setup
        with tempfile.TemporaryDirectory() as td:
            folder, data = Path(td) / "g", Path(td) / "d"
            folder.mkdir()
            data.mkdir()
            (folder / "mmproj-BF16.gguf").write_bytes(b"GGUF")
            self.assertIsNone(setup.inkling_find_local_vision(folder, data, True))
            (folder / "Inkling-2-Mini-Vision.strata").write_bytes(b"x")
            self.assertEqual(setup.inkling_find_local_vision(folder, data, True), folder / "Inkling-2-Mini-Vision.strata")

    def test_adopting_such_a_folder_says_so_and_passes_no_vision(self):
        from strata_cli.launch import adopt
        from strata_control import inkling_family
        from tests.terminal.test_inkling_gguf_import import make_inkling_gguf, write_fake_o200k_files, ROOT
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            folder, data = base / "g", base / "d"
            folder.mkdir()
            gguf = folder / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
            gguf.write_bytes(make_inkling_gguf(architecture="inkling2", expert_type=12))
            (folder / "mmproj-BF16.gguf").write_bytes(b"GGUF")
            tok = data / "packs" / "inkling-2-mini-ud-q4_k_xl" / "tokenizer"
            tok.mkdir(parents=True)
            (tok / "vocab.json").write_text("{}", encoding="utf-8")
            notes: list[str] = []
            cfg, problems = adopt.plan(family="inkling-2-mini", model="UD-Q4_K_XL", model_dir=folder, data_dir=data, src=ROOT, context=32768, kv=None,
                                       model_name="inkling-2-mini-ud-q4_k_xl", family_def={}, log=base / "ink.log", backend="cuda", notes=notes)
            self.assertEqual(problems, [], problems)
            self.assertNotIn("--vision", cfg["args"])
            self.assertTrue(any("mmproj-BF16.gguf beside the model is llama.cpp's picture projector" in n and inkling_family.VISION_FILE in n for n in notes), notes)

    def test_launch_says_the_real_reason_when_the_installed_model_has_no_reader(self):
        from strata_cli.launch.runner import Runner
        from strata_control import inkling_family
        from tests.terminal.launch_support import HW_12, Home, catalog
        with Home():
            r = Runner(S.LaunchSettings(family="inkling-2-mini", model="SQ-10", vision="on", context=32768), catalog=catalog(HW_12))
            r.catalog.strata.describe_config = lambda path: {"images": "off"}
            r.check_pictures(Path("x.json"))
        self.assertTrue(any(f"Images will not work: no picture reader file ({inkling_family.VISION_FILE}, 0.64 GB)" in n for n in r.notices), r.notices)


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
