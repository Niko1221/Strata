"""`strata models explain NAME`: the folder chain behind Catalog.model_folder() (#3996).

setup.py writes `models/IQ3_S` (the model name's case kept); the run-config tag is lower case (`iq3_s`). The launch lookup
tries setup's spelling, then the lower-case tag, then a folder that differs only in case, and the report must say which one
it takes and why, and count the .gguf files in it. The tests make the disk case-sensitive on any system by answering
`Path.is_dir` from the names the folder really lists, so the letter-case cases mean the same on Windows as on Linux.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_cli.launch.catalog import Catalog  # noqa: E402
from tests.terminal.launch_support import Home, run_cli  # noqa: E402

_real_is_dir = Path.is_dir


def _case_sensitive_is_dir(self):
    try:
        listed = {entry.name for entry in self.parent.iterdir()}
    except OSError:
        return False
    return self.name in listed and _real_is_dir(self)


class StubCatalog:
    """Just what the folder walk and its report use; the real tables need the installer files."""
    tables = lambda self: ({}, {"qwen": {"tag": ""}}, None, None)          # noqa: E731
    tag_of = Catalog.tag_of
    model_folder = Catalog.model_folder
    model_folder_chain = Catalog.model_folder_chain
    explain_model = Catalog.explain_model


class FolderReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        self.models = self.data / "models"
        self.models.mkdir(parents=True)
        patch = mock.patch.object(Path, "is_dir", _case_sensitive_is_dir)
        patch.start()
        self.addCleanup(patch.stop)
        self.cat = StubCatalog()

    def ggufs(self, folder: Path, *sizes: int) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for number, size in enumerate(sizes, 1):
            (folder / f"m-{number:05d}-of-{len(sizes):05d}.gguf").write_bytes(b"x" * size)

    def spellings(self, report: dict) -> list[tuple[str, bool]]:
        return [(Path(c["path"]).name, c["exists"]) for c in report["candidates"]]

    def test_the_folder_setup_wrote_wins_and_its_gguf_files_are_counted(self):
        self.ggufs(self.models / "IQ3_S", 1000, 250)
        (self.models / "IQ3_S" / "notes.txt").write_text("not a shard")
        report = self.cat.explain_model(self.data, "qwen", "IQ3_S")
        self.assertEqual((report["family"], report["model"], report["tag"]), ("qwen", "IQ3_S", "iq3_s"))
        self.assertTrue(report["found"])
        self.assertEqual(self.spellings(report), [("IQ3_S", True), ("iq3_s", False)])
        self.assertEqual((Path(report["chosen"]["path"]).name, report["chosen"]["reason"]), ("IQ3_S", "exact spelling"))
        self.assertEqual(report["gguf"], {"shard_count": 2, "bytes": 1250})
        self.assertEqual(report["nearby_folders"], [])
        self.assertEqual(self.cat.model_folder(self.data, "qwen", "IQ3_S").name, "IQ3_S")

    def test_a_lower_case_folder_is_found_and_counted(self):
        self.ggufs(self.models / "iq3_s", 4096)
        report = self.cat.explain_model(self.data, "qwen", "IQ3_S")
        self.assertEqual(self.spellings(report), [("IQ3_S", False), ("iq3_s", True)])
        self.assertEqual((Path(report["chosen"]["path"]).name, report["chosen"]["reason"]), ("iq3_s", "lower-case tag"))
        self.assertEqual(report["gguf"], {"shard_count": 1, "bytes": 4096})

    def test_a_folder_that_differs_only_in_letter_case_is_found(self):
        self.ggufs(self.models / "Iq3_S", 512)
        report = self.cat.explain_model(self.data, "qwen", "IQ3_S")
        self.assertEqual(self.spellings(report), [("IQ3_S", False), ("iq3_s", False), ("Iq3_S", True)])
        self.assertEqual((Path(report["chosen"]["path"]).name, report["chosen"]["reason"]), ("Iq3_S", "case-insensitive match"))
        self.assertEqual(report["gguf"], {"shard_count": 1, "bytes": 512})

    def test_nothing_found_falls_back_and_names_the_folders_that_are_there(self):
        (self.models / "IQ3_XXS").mkdir()
        (self.models / "q3_k").mkdir()
        report = self.cat.explain_model(self.data, "qwen", "IQ3_S")
        self.assertFalse(report["found"])
        self.assertEqual(self.spellings(report), [("IQ3_S", False), ("iq3_s", False)])
        self.assertEqual((Path(report["chosen"]["path"]).name, report["chosen"]["reason"]), ("IQ3_S", "fallback"))
        self.assertEqual(report["gguf"], {"shard_count": 0, "bytes": 0})
        self.assertEqual(set(report["nearby_folders"]), {"IQ3_XXS", "q3_k"})
        self.assertEqual(report["nearby_folders"][0], "IQ3_XXS")
        self.assertEqual(self.cat.model_folder(self.data, "qwen", "IQ3_S").name, "IQ3_S")
        self.assertEqual(self.cat.explain_model(self.data / "empty", "qwen", "IQ3_S")["nearby_folders"], [])


class ExplainCommand(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.data = Path(self.home.base) / "data"
        (self.data / "models").mkdir(parents=True)

    def explain(self, *args):
        code, out, err = run_cli(["models", "explain", *args, "--data-dir", str(self.data), "--output", "json"])
        return code, (json.loads(out) if out.strip() else None), err

    def snapshot(self) -> dict:
        return {str(p.relative_to(self.data)): (p.is_dir(), p.stat().st_size if p.is_file() else 0)
                for p in sorted(self.data.rglob("*"))}

    def test_json_reports_the_chain_and_accepts_any_case_or_a_family_prefix(self):
        folder = self.data / "models" / "IQ3_S"
        folder.mkdir()
        (folder / "m-00001-of-00002.gguf").write_bytes(b"x" * 1000)
        for name in ("iq3_s", "qwen/IQ3_S"):
            code, report, err = self.explain(name)
            self.assertEqual((code, err), (0, ""), name)
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual((report["family"], report["model"]), ("qwen", "IQ3_S"))
            self.assertEqual(report["chosen"]["reason"], "exact spelling")
            self.assertEqual(report["gguf"], {"shard_count": 1, "bytes": 1000})
        self.assertEqual(report["candidates"][0], {"path": str(folder), "spelling": "setup's spelling", "exists": True})
        self.assertEqual(Path(report["candidates"][1]["path"]).name, "iq3_s")

    def test_a_missing_model_says_so_and_names_the_folders_that_are_there(self):
        (self.data / "models" / "IQ3_XXS").mkdir()
        code, report, err = self.explain("Q2_0")
        self.assertEqual((code, err), (0, ""))
        self.assertFalse(report["found"])
        self.assertEqual(report["chosen"]["reason"], "fallback")
        self.assertEqual(report["nearby_folders"], ["IQ3_XXS"])

    def test_human_output_says_what_was_tried_and_chosen(self):
        folder = self.data / "models" / "IQ3_S"
        folder.mkdir()
        (folder / "m-00001-of-00002.gguf").write_bytes(b"x" * 70000)
        code, out, err = run_cli(["models", "explain", "IQ3_S", "--data-dir", str(self.data)])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("qwen IQ3_S (folder tag: iq3_s)", out)
        self.assertIn("exists: yes", out)
        self.assertIn("Chosen: " + str(folder) + " (the exact (setup) spelling)", out)
        self.assertIn("1 .gguf file(s), 70,000 bytes in total", out)
        code, out, err = run_cli(["models", "explain", "Q2_0", "--data-dir", str(self.data)])
        self.assertIn("No model folder found", out)

    def test_an_unknown_name_is_refused_with_the_closest_models(self):
        code, out, err = run_cli(["models", "explain", "IQ3_Z", "--data-dir", str(self.data)])
        self.assertEqual(code, 2)
        self.assertIn("There is no model called 'IQ3_Z'", err)
        code, report, err = self.explain("IQ3_Z")
        self.assertEqual(code, 2)
        self.assertIs(report["ok"], False)
        self.assertIn("IQ3_Z", report["error"]["message"])

    def test_the_command_writes_nothing(self):
        (self.data / "models" / "IQ3_S").mkdir()
        (self.data / "models" / "IQ3_S" / "m-00001-of-00002.gguf").write_bytes(b"x" * 10)
        before = self.snapshot()
        self.explain("IQ3_S")
        self.explain("Q2_0")
        run_cli(["models", "explain", "IQ3_S", "--data-dir", str(self.data)])
        self.assertEqual(self.snapshot(), before)
        from strata_control import discovery
        mine, app = discovery.own_settings_file(), discovery.app_settings_file()
        old, new = Path(self.home.base) / "old-data", Path(self.home.base) / "new-data"
        old.mkdir(), new.mkdir()
        mine.parent.mkdir(parents=True, exist_ok=True)
        app.parent.mkdir(parents=True, exist_ok=True)
        mine.write_text(json.dumps({"data_dir": str(old), "data_dir_at": 1000}), encoding="utf-8")
        app.write_text(json.dumps({"data_dir": str(new), "data_dir_at": 2000}), encoding="utf-8")
        code, out, err = run_cli(["models", "explain", "IQ3_S"])
        self.assertEqual((code, err), (0, ""))
        self.assertIn(str(new), out)                                        # the folder changed last is the one launch uses
        self.assertEqual(json.loads(mine.read_text(encoding="utf-8"))["data_dir"], str(old))  # nothing written


if __name__ == "__main__":
    unittest.main()
