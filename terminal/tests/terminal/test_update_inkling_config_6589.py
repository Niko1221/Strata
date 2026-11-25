"""#6589 (0.2.20): `strata update` test-started the Qwen engine with an Inkling-2-Mini run config ("unknown argument: --vram-experts-gb").
Setup writes `exe: component:strata-engine` into an Inkling config (serve finds the engine by its `arch`), and the classifier took that reference as proof.
The engine family the config names (arch, shape-only arguments, the .strata file) decides; a reference to a shape engine still wins."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec, updater


INK_ARGS = ["--native", "/data/models/inkling-2-mini/Inkling-2-Mini-SQ-20.strata", "--max-context", "32768", "--vram-experts-gb", "12.49",
            "--ram-experts-gb", "20.00", "--draft", "dspark"]


class ConfigComponent(unittest.TestCase):
    def test_the_reporters_inkling_config_with_the_general_engine_reference_is_an_inkling_config(self):
        cfg = {"exe": "component:strata-engine", "args": INK_ARGS, "arch": "inkling2_loop", "model_name": "Inkling-2-Mini-sq-20"}
        self.assertEqual(updater._config_component_id(cfg), "strata-inkling")
        self.assertTrue(updater._is_shape_engine_config(cfg))

    def test_without_the_arch_the_arguments_still_say_so(self):
        for args in (INK_ARGS, INK_ARGS[:2] + ["--vram-experts-gb", "8"], ["--native", "C:\\m\\Inkling-2-Mini-SQ-10.strata"]):
            cfg = {"exe": "component:strata-engine", "args": args}
            self.assertEqual(updater._config_component_id(cfg), "strata-inkling", args)

    def test_an_exe_path_to_the_general_engine_does_not_hide_an_inkling_config(self):
        cfg = {"exe": "/home/u/.local/share/strata/components/engine/0.1.65-linux-x86_64-cuda-cuda13/strata", "args": INK_ARGS, "arch": "inkling2_loop"}
        self.assertEqual(updater._config_component_id(cfg), "strata-inkling")

    def test_hy4_and_qwen_are_as_before(self):
        hy4 = {"exe": "component:strata-engine", "args": ["--native", "/m/Hy4-Flash-SQ-64.strata", "--engram", "/m/Hy4-Flash-Engram-Q8.strata"]}
        self.assertEqual(updater._config_component_id(hy4), "strata-hy4")
        hy4_flags = {"exe": "component:strata-engine", "args": ["--vram-experts-gb", "8", "--native", "/m/Hy4-Flash-SQ-64.strata"]}
        self.assertEqual(updater._config_component_id(hy4_flags), "strata-hy4")
        qwen = {"exe": "component:strata-engine", "arch": "qwen4exp", "args": ["--pack", "/p", "--native", "/m/Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf",
                                                                              "--expert-cache", "auto", "--max-context", "32768"]}
        self.assertEqual(updater._config_component_id(qwen), ec.COMPONENT_ID)
        self.assertEqual(updater._config_component_id({"exe": "/e/strata", "args": ["--native", "/m/q.gguf"]}), ec.COMPONENT_ID)
        self.assertEqual(updater._config_component_id({}), ec.COMPONENT_ID)

    def test_a_reference_to_a_shape_engine_still_decides(self):
        self.assertEqual(updater._config_component_id({"exe": "component:strata-inkling", "args": []}), "strata-inkling")
        self.assertEqual(updater._config_component_id({"exe": "component:strata-hy4", "args": []}), "strata-hy4")


class DiscoverForTheGeneralEngine(unittest.TestCase):
    def test_the_qwen_update_test_never_picks_the_inkling_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ink = root / "strata-inkling-2-mini-sq-20.json"
            qwen = root / "strata-qwen.json"
            ink.write_text(json.dumps({"exe": "component:strata-engine", "arch": "inkling2_loop", "args": INK_ARGS}), encoding="utf-8")
            qwen.write_text(json.dumps({"exe": "component:strata-engine", "args": ["--native", "/m/q.gguf"]}), encoding="utf-8")
            listed = {"models": [
                {"id": "a-inkling", "component_status": "present", "source_config": str(ink)},
                {"id": "b-qwen", "component_status": "present", "source_config": str(qwen)},
            ]}
            with mock.patch("strata_control.models.list_installed", return_value=listed), \
                    mock.patch("strata_control.retired.config_file_reason", return_value=None), \
                    mock.patch("strata_control.profiles.list_profiles", return_value=[]):
                self.assertEqual(updater.discover_run_config(component_id=ec.COMPONENT_ID), qwen)
                self.assertEqual(updater.discover_run_config(component_id="strata-inkling"), ink)
                # only the Inkling model installed: the Qwen engine has nothing to test with (a version-only start), not the Inkling config
                listed["models"] = listed["models"][:1]
                self.assertIsNone(updater.discover_run_config(component_id=ec.COMPONENT_ID))


if __name__ == "__main__":
    unittest.main()
