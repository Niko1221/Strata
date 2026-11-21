"""A self-built engine that already IS engine/ (setup's own place) installs without a copy.

Copying strata.exe onto itself fails on Windows with PermissionError (WinError 32), and setup reported that as
"a model is running from it" while nothing was running at all - the engine is simply already where it belongs.

    python -m unittest tools.test_setup_local_engine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class SelfInstall(unittest.TestCase):
    """install_local_engine with the engine already in engine/ (local_installed_engine's third case)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.eng = self.root / "engine"
        self.eng.mkdir()
        (self.eng / setup.EXE).write_bytes(b"the engine")

    def install(self, *, vision="none", with_reader=True):
        if with_reader:
            (self.eng / setup.VEXE).write_bytes(b"the reader")
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(setup, "binary_engine_version", return_value=(0, 1, 63)), \
                mock.patch.object(setup, "binary_engine_flavor", return_value="q4u32"), \
                mock.patch.object(setup, "binary_engine_label", return_value="0.1.63-q4u32"), \
                mock.patch.object(setup, "local_rocm_lib_dirs", return_value=[]), \
                mock.patch.object(setup, "source_hash", return_value="deadbeef"):
            return setup.install_local_engine(self.eng, {"arch": "gfx1200"}, hip=True, vision=vision)

    def test_the_engine_already_in_place_is_not_copied_onto_itself(self):
        """The case that failed: setup copied engine/strata.exe onto itself and read the error as a running model."""
        eng = self.install()
        self.assertEqual(eng, self.eng)
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"the engine")
        meta = json.loads((self.eng / "BUILD.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["source"], "local")
        self.assertEqual(meta["flavor"], "q4u32")

    def test_the_image_reader_already_beside_the_engine_is_kept(self):
        eng = self.install(vision="cpu")
        self.assertEqual(eng, self.eng)
        self.assertEqual((self.eng / setup.VEXE).read_bytes(), b"the reader")
        meta = json.loads((self.eng / "BUILD.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["vision"], "cpu")
        self.assertEqual(meta["vision_src"], "deadbeef")

    def test_a_reader_from_the_build_folder_is_still_copied(self):
        """The normal path is unchanged: a reader from another folder still goes beside the engine."""
        (self.root / "build-vision" / "bin").mkdir(parents=True)
        (self.root / "build-vision" / "bin" / setup.VEXE).write_bytes(b"built reader")
        eng = self.install(vision="cpu", with_reader=False)
        self.assertEqual(eng, self.eng)
        self.assertEqual((self.eng / setup.VEXE).read_bytes(), b"built reader")
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"the engine")

    def test_an_engine_from_another_folder_is_still_copied(self):
        """The other normal path: a build folder's engine goes into engine/."""
        build = self.root / "build-hip-port"
        build.mkdir()
        (build / setup.EXE).write_bytes(b"built engine")
        with mock.patch.object(setup, "ROOT", self.root), \
                mock.patch.object(setup, "binary_engine_version", return_value=(0, 1, 63)), \
                mock.patch.object(setup, "binary_engine_flavor", return_value=""), \
                mock.patch.object(setup, "binary_engine_label", return_value=""), \
                mock.patch.object(setup, "local_rocm_lib_dirs", return_value=[]), \
                mock.patch.object(setup, "source_hash", return_value="deadbeef"):
            eng = setup.install_local_engine(build, {"arch": "gfx1200"}, hip=True)
        self.assertEqual(eng, self.eng)
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"built engine")


class OptIn(unittest.TestCase):
    """A build that sits in the tree is used only when the person asked for it, and never over --prebuilt."""

    @staticmethod
    def args(**kw):
        return argparse.Namespace(**kw)

    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in ("STRATA_PREFER_SELF_BUILD", "STRATA_SELF_BUILD_DIR")}

        def restore():
            for key, value in self.saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)

    def test_a_build_in_the_tree_is_left_alone_without_the_opt_in(self):
        self.assertFalse(setup.wants_self_build(self.args()))
        self.assertTrue(setup.wants_self_build(self.args(prefer_self_build=True)))

    def test_the_environment_asks_the_same_way(self):
        with mock.patch.dict(os.environ, {"STRATA_PREFER_SELF_BUILD": "1"}):
            self.assertTrue(setup.wants_self_build(self.args()))
        with mock.patch.dict(os.environ, {"STRATA_SELF_BUILD_DIR": "D:/build"}):
            self.assertTrue(setup.wants_self_build(self.args()))

    def test_an_explicit_prebuilt_wins_over_the_self_build(self):
        self.assertFalse(setup.wants_self_build(self.args(prefer_self_build=True, prebuilt_given=True)))


class BuildFolders(unittest.TestCase):
    """local_engine_build finds the folder the documented build makes, at the top of the checkout."""

    def test_a_build_at_the_top_of_the_checkout_is_found(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "terminal"
        root.mkdir()
        build = Path(tmp.name) / "build-hip"
        build.mkdir()
        (build / setup.EXE).write_bytes(b"built")
        with mock.patch.object(setup, "ROOT", root):
            self.assertEqual(setup.local_engine_build(), build)

    def test_a_build_under_terminal_is_found_too(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "terminal"
        build = root / "build-hip"
        build.mkdir(parents=True)
        (build / setup.EXE).write_bytes(b"built")
        with mock.patch.object(setup, "ROOT", root):
            self.assertEqual(setup.local_engine_build(), build)


class NamedFormats(unittest.TestCase):
    """A format of this tree's engine is written only for an engine that has it: an upstream build refuses it at start."""

    def test_q4u32_and_its_window_need_this_trees_engine(self):
        self.assertFalse(setup.kv_needs_own_engine("q4u32", "q4u32"))
        self.assertFalse(setup.kv_needs_own_engine("q4u32w", "q4u32"))
        self.assertFalse(setup.kv_needs_own_engine("q4lm8v", "q4u32"))
        self.assertTrue(setup.kv_needs_own_engine("q4u32", ""))
        self.assertTrue(setup.kv_needs_own_engine("q4u32", None))
        self.assertTrue(setup.kv_needs_own_engine("q4u32w", ""))
        self.assertTrue(setup.kv_needs_own_engine("q4lm8v", "upstream"))

    def test_every_other_format_is_written_for_any_engine(self):
        for kv in ("int8", "q4_0", "k8v4", "fp16", None, ""):
            self.assertFalse(setup.kv_needs_own_engine(kv, ""), kv)


if __name__ == "__main__":
    unittest.main()
