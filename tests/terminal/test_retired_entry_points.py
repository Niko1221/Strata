"""Fork DeepSeek compatibility boundaries preserve local files and report missing engines honestly.

The upstream retirement recognizer remains tested separately; this fork retains the explicit adapters. No model is loaded or downloaded."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import cfg_paths, models, retired, runtime  # noqa: E402
from strata_control.profiles import ProfileError  # noqa: E402

D1_HEAD = "is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started."


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ds_install(d: Path, *, model_bytes: int = 3_000_000) -> tuple[Path, Path, Path]:
    """(config, model folder, pack) of a fake DeepSeek install as setup 0.2.13 wrote it."""
    models_dir = d / "data" / "models" / "deepseek-v4.1-flash-ist-das-2.00bpw"
    models_dir.mkdir(parents=True)
    (models_dir / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").write_bytes(b"\0" * model_bytes)
    pack = d / "data" / "packs" / "deepseek"
    pack.mkdir(parents=True)
    (pack / "ds4.txt").write_text(f"format gguf\nmodel_dir {models_dir}\n", encoding="utf-8")
    eng = d / "engine-dsv41" / "0.1.59-linux-x86_64-cuda-cuda13"
    eng.mkdir(parents=True)
    exe = eng / "strata-dsv41"
    exe.write_text("#!/bin/sh\necho started > \"$0.ran\"\n", encoding="utf-8")
    cfg = {"exe": str(exe), "args": ["--pack", str(pack), "--max-context", "131072", "--eos", "1"], "cwd": str(d),
           "tokenizer": str(pack / "tokenizer"), "model_name": "deepseek-v4.1-flash", "arch": "deepseek41", "port": 8080,
           "lib_dirs": [str(d / "components" / "engine" / "0.1.50-linux-x86_64-cuda" / "lib")]}
    path = d / "strata-deepseek-v4.1-flash-ist-das-2.00bpw.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path, models_dir, pack


class ForkCompatibility(unittest.TestCase):
    """Compatibility and availability are independent; all fixtures stay private."""

    def test_missing_engine_is_not_substituted_with_qwen(self):
        import setup
        from strata_control import dsv41_engine
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            qwen = root / "engine" / "strata.exe"
            qwen.parent.mkdir()
            qwen.write_bytes(b"fixture Qwen engine")
            before = sha(qwen)
            with mock.patch.dict(setup.os.environ, {"STRATA_DSV41_EXE": ""}), \
                    mock.patch.object(setup, "dsv41_engine_dir", return_value=root / "engine-dsv41"):
                self.assertIsNone(setup.dsv41_exe(qwen.parent))
                self.assertIsNone(dsv41_engine.find(root / "engine-dsv41"))
            self.assertEqual(sha(qwen), before)
            self.assertFalse((root / "engine-dsv41").exists())

    def test_config_list_preserves_explicit_deepseek_entries(self):
        import setup
        with tempfile.TemporaryDirectory() as d:
            path, model_dir, pack = ds_install(Path(d))
            before = {p: sha(p) for p in (path, *model_dir.iterdir(), *pack.iterdir()) if p.is_file()}
            with mock.patch.object(setup, "home", return_value=Path(d)):
                self.assertIn(path, setup.installed_configs())
            self.assertEqual({p: sha(p) for p in before}, before)

    def test_model_inspection_reports_missing_components_without_writes(self):
        with tempfile.TemporaryDirectory() as d:
            path, model_dir, pack = ds_install(Path(d))
            before = {p: sha(p) for p in (path, *model_dir.iterdir(), *pack.iterdir()) if p.is_file()}
            report = {"ok": False, "issues": [{"code": "engine_missing"}], "components": []}
            with mock.patch.object(models, "inspect_components", return_value=report):
                out = models.inspect(path.stem.removeprefix("strata-"), override=d)["model"]
            self.assertEqual(out["component_status"], "missing_or_invalid")
            self.assertEqual(out["execution_compatibility"], "unverified")
            self.assertNotIn("retired", out)
            self.assertEqual({p: sha(p) for p in before}, before)

    def test_server_keeps_deepseek_protocol_without_qwen_drafter(self):
        from serve import server
        cfg = {"arch": "deepseek41", "exe": "missing/strata-dsv41.exe",
               "args": ["--pack", "/private/pack", "--max-context", "131072", "--eos", "1"]}
        with mock.patch.object(server.subprocess, "Popen", side_effect=AssertionError("started process")):
            args = server.engine_args(cfg, cfg["exe"])
        self.assertIn("--pack", args)
        self.assertNotIn("--draft-costs", args)
        self.assertNotIn("--mtp", args)

    def test_saved_preset_keeps_deepseek_choice_without_starting(self):
        from strata_cli.launch import cmd, store
        from strata_cli.launch.settings import LaunchSettings
        saved = LaunchSettings()
        saved.family, saved.model = "deepseek41", "IST-DAS-2.00bpw"
        parser = argparse.ArgumentParser()
        cmd.add_parsers(parser.add_subparsers(dest="command"))
        with mock.patch.object(store, "load", return_value=saved):
            chosen, _ = cmd.settings_from_args(parser.parse_args(["launch", "--preset", "big"]), catalog=object())
        self.assertEqual((chosen.family, chosen.model), ("deepseek41", "IST-DAS-2.00bpw"))

    def test_display_name_does_not_override_qwen35_metadata(self):
        from tests.terminal.test_retired_deepseek import gguf
        name = "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"
        with tempfile.TemporaryDirectory() as d:
            path = gguf(Path(d) / name, "qwen35")
            before = sha(path)
            with mock.patch.object(models, "_already_set_up", return_value=None):
                plan = models.plan_import(str(path))
            self.assertEqual(plan["metadata"]["architecture"], "qwen35")
            self.assertFalse(plan["would_write"])
            self.assertNotEqual((plan.get("candidate_selection") or {}).get("family"), "deepseek41")
            self.assertEqual(sha(path), before)


if __name__ == "__main__":
    unittest.main()
