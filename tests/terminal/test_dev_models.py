"""Strata Dev: dev/models overlays on a run config (strata_control/dev_models.py) and their place in presets.overlay."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import dev_models, presets

CFG = {"exe": "engine", "tokenizer": "tok", "model_name": "qwen3.8-flash-next-q2_0",
       "args": ["--max-context", "131072", "--kv", "int8", "--mtp"], "sampling": {"temperature": 1.0, "top_k": 20}}


class DevModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.models = root / "dev" / "models"
        self.models.mkdir(parents=True)
        self.home = root / "home"
        self.home.mkdir()
        self.env = {"STRATA_DEV": "1", "STRATA_DEV_CONFIG": str(root / "dev.config.json"), "STRATA_HOME": str(self.home)}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        (self.models / name).write_text(json.dumps(data), encoding="utf-8")

    def test_outside_dev_nothing_changes(self):
        self.write("q.json", {"model": "qwen*", "context": 8192})
        out, info = dev_models.apply(CFG, env={**self.env, "STRATA_DEV": ""})
        self.assertIs(out, CFG)
        self.assertEqual(info, {"applied": False})

    def test_overlay_and_profile(self):
        self.write("q.json", {"model": "qwen3.8-*", "context": 65536, "args": {"--kv": "q8_0", "--mtp": None},
                              "env": {"A": "1"}, "vision": False, "sampling": {"top_p": 0.9}, "profile": "coder"})
        self.write("_off.json", {"model": "*", "context": 1})
        self.write("broken.json", ["not", "an", "object"])
        out, info = dev_models.apply(CFG, env=self.env)
        self.assertTrue(info["applied"])
        self.assertEqual(info["overlays"], [{"overlay": "q", "profile": "coder"}])
        self.assertIn("broken.json", info["warnings"][0])
        args = out["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "65536")
        self.assertEqual(args[args.index("--kv") + 1], "q8_0")
        self.assertNotIn("--mtp", args)
        self.assertEqual(out["env"], {"A": "1"})
        self.assertFalse(out["vision"])
        self.assertEqual(out["reasoning_effort"], "high")            # the coder profile
        self.assertEqual(out["sampling"], {"temperature": 0.2, "top_k": 20, "top_p": 0.95})
        self.assertEqual(CFG["args"], ["--max-context", "131072", "--kv", "int8", "--mtp"], "the run config itself is not changed")

    def test_the_apps_profile_choice_wins_and_other_models_are_untouched(self):
        self.write("q.json", {"model": "qwen*", "profile": "coder", "profiles": {"schnell": {"thinking": "none"}}})
        (self.home / "dev-model-profiles.json").write_text(json.dumps({"q": "schnell"}), encoding="utf-8")
        out, info = dev_models.apply(CFG, env=self.env)
        self.assertEqual(out["reasoning_effort"], "none", "the file's own schnell replaces the built-in one")
        other, info2 = dev_models.apply({**CFG, "model_name": "deepseek-v4.1-flash"}, env=self.env)
        self.assertFalse(info2["applied"])

    def test_chat_template_uses_a_copy_of_the_tokenizer_folder(self):
        tok = Path(self.tmp.name) / "tok"
        tok.mkdir()
        (tok / "chat_template.jinja").write_text("original", encoding="utf-8")
        (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.models / "t.jinja").write_text("mine", encoding="utf-8")
        self.write("q.json", {"model": "qwen*", "chat_template": "t.jinja"})
        with mock.patch.dict(os.environ, self.env):
            out, info = dev_models.apply({**CFG, "tokenizer": str(tok)}, env=self.env)
        self.assertNotEqual(out["tokenizer"], str(tok))
        self.assertEqual((Path(out["tokenizer"]) / "chat_template.jinja").read_text(encoding="utf-8"), "mine")
        self.assertEqual((tok / "chat_template.jinja").read_text(encoding="utf-8"), "original")

    def test_presets_overlay_runs_it_first_only_in_dev(self):
        self.write("q.json", {"model": "qwen*", "context": 4096})
        with mock.patch.dict(os.environ, self.env), mock.patch.object(presets, "preset_dir", return_value=Path(self.tmp.name) / "none"):
            out, info = presets.overlay(CFG)
        self.assertEqual(out["args"][out["args"].index("--max-context") + 1], "4096")
        self.assertTrue(info["dev"]["applied"])
        self.assertFalse(info["applied"], "no saved preset was applied")
        with mock.patch.dict(os.environ, {**self.env, "STRATA_DEV": "0"}), mock.patch.object(presets, "preset_dir", return_value=Path(self.tmp.name) / "none"):
            out, info = presets.overlay(CFG)
        self.assertIs(out, CFG)
        self.assertNotIn("dev", info)


if __name__ == "__main__":
    unittest.main()
