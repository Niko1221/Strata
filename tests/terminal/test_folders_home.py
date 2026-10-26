"""strata_control/folders.problem: a system folder is refused, except inside the running user's own home (root's /root included)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import folders


class OwnHome(unittest.TestCase):
    def test_a_system_folder_is_still_refused(self):
        root = folders._system_roots()[0]
        with mock.patch.dict(os.environ, {"HOME": tempfile.gettempdir(), "USERPROFILE": tempfile.gettempdir()}):
            p = folders.problem(str(Path(root) / "strata-models"))
        self.assertIsNotNone(p)
        self.assertIn("system folder", p[0])

    def test_the_users_own_home_is_allowed_even_when_it_is_under_a_system_root(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "rootlike"
            home.mkdir()
            with mock.patch.object(folders, "_system_roots", lambda: [d]), \
                    mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}):
                self.assertIsNone(folders.problem(str(home / "models")))
                other = Path(d) / "elsewhere"
                other.mkdir()
                self.assertIn("system folder", folders.problem(str(other / "models"))[0])

    @unittest.skipUnless(os.name != "nt", "POSIX home symlinks")
    def test_symlinked_home_target_under_var_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            home_target = Path(d) / "var" / "home" / "user"
            home_target.mkdir(parents=True)
            home_link = Path(d) / "home" / "user"
            home_link.parent.mkdir()
            try:
                home_link.symlink_to(home_target, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable")
            with (mock.patch.object(folders, "_system_roots", lambda: [str(Path(d) / "var")]),
                  mock.patch.dict(os.environ, {"HOME": str(home_link)})):
                self.assertIsNone(folders.problem(str(home_target / "Strata-data")))
                elsewhere = Path(d) / "var" / "lib" / "models"
                elsewhere.parent.mkdir()
                self.assertIn("system folder", folders.problem(str(elsewhere))[0])

    @unittest.skipUnless(os.name != "nt", "POSIX external media paths")
    def test_run_media_is_allowed_but_the_run_root_remains_protected(self):
        with tempfile.TemporaryDirectory() as d:
            media = Path(d) / "media"
            mounted = media / "user" / "external_SSD"
            mounted.mkdir(parents=True)
            with (mock.patch.object(folders, "_system_roots", lambda: [d]),
                  mock.patch.object(folders, "_POSIX_EXTERNAL_MEDIA", str(media))):
                self.assertIsNone(folders.problem(str(mounted / "Strata-data")))
                self.assertIn("system folder", folders.problem(str(Path(d) / "other" / "models"))[0])
            with (mock.patch.object(folders, "_system_roots", lambda: ["/run"]),
                  mock.patch.object(folders, "_POSIX_EXTERNAL_MEDIA", "/run/media")):
                self.assertIn("system folder", folders.problem("/run/models")[0])

    @unittest.skipUnless(os.name != "nt", "POSIX path rules")
    def test_external_media_exception_does_not_match_similarly_named_paths(self):
        with (mock.patch.object(folders, "_system_roots", lambda: ["/run"]),
              mock.patch.object(folders, "_POSIX_EXTERNAL_MEDIA", "/run/media")):
            self.assertIn("system folder", folders.problem("/run/mediadrive/models")[0])


if __name__ == "__main__":
    unittest.main()
