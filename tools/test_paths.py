"""gguf-py directory discovery, with no packages, models or downloads.

    python tools/test_paths.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("strata_tool_paths", Path(__file__).with_name("_paths.py"))
paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paths)


class GgufPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.engine = root / "source" / "strata"
        self.install = root / "installed strata"
        self.source_gguf = self.engine / "third_party" / "llama.cpp" / "gguf-py"
        self.install_gguf = self.install / "third_party" / "llama.cpp" / "gguf-py"
        self.dev_gguf = root / ".ref" / "llama.cpp" / "gguf-py"
        self.override = root / "explicit gguf-py"
        for replacement in (patch.object(paths, "ENGINE", self.engine), patch.dict(os.environ, {}, clear=True)):
            replacement.start()
            self.addCleanup(replacement.stop)

    def package_dir(self, directory):
        (directory / "gguf").mkdir(parents=True)
        return str(directory)

    def test_install_dir_without_gguf_override(self):
        expected = self.package_dir(self.install_gguf)
        os.environ["STRATA_INSTALL_DIR"] = str(self.install)
        self.assertEqual(paths.gguf_py(), expected)

    def test_explicit_gguf_override_wins(self):
        for directory in (self.install_gguf, self.source_gguf, self.dev_gguf):
            self.package_dir(directory)
        expected = self.package_dir(self.override)
        os.environ.update(STRATA_GGUF_PY=expected, STRATA_INSTALL_DIR=str(self.install))
        self.assertEqual(paths.gguf_py(), expected)

    def test_missing_gguf_override_uses_install_dir(self):
        expected = self.package_dir(self.install_gguf)
        os.environ.update(STRATA_GGUF_PY=str(self.override), STRATA_INSTALL_DIR=str(self.install))
        self.assertEqual(paths.gguf_py(), expected)

    def test_install_dir_precedes_source_and_development_trees(self):
        for directory in (self.source_gguf, self.dev_gguf):
            self.package_dir(directory)
        expected = self.package_dir(self.install_gguf)
        os.environ["STRATA_INSTALL_DIR"] = str(self.install)
        self.assertEqual(paths.gguf_py(), expected)

    def test_source_tree_precedes_development_tree(self):
        self.package_dir(self.dev_gguf)
        expected = self.package_dir(self.source_gguf)
        self.assertEqual(paths.gguf_py(), expected)

    def test_development_tree_fallback(self):
        expected = self.package_dir(self.dev_gguf)
        self.assertEqual(paths.gguf_py(), expected)

    def test_missing_overrides_fall_back_to_source(self):
        expected = self.package_dir(self.source_gguf)
        os.environ.update(STRATA_GGUF_PY=str(self.override), STRATA_INSTALL_DIR=str(self.install))
        self.assertEqual(paths.gguf_py(), expected)

    def test_install_candidate_requires_gguf_directory(self):
        self.install_gguf.mkdir(parents=True)
        (self.install_gguf / "gguf").touch()
        os.environ["STRATA_INSTALL_DIR"] = str(self.install)
        expected = self.package_dir(self.dev_gguf)
        self.assertEqual(paths.gguf_py(), expected)

    def test_empty_install_dir_falls_back_to_source(self):
        os.environ["STRATA_INSTALL_DIR"] = ""
        expected = self.package_dir(self.source_gguf)
        self.assertEqual(paths.gguf_py(), expected)

    def test_missing_paths_report_override_hint(self):
        with self.assertRaisesRegex(SystemExit, "or set STRATA_GGUF_PY"):
            paths.gguf_py()


if __name__ == "__main__":
    unittest.main()
