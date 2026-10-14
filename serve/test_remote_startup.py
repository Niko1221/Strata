"""Public API startup is an explicit local-config opt-in, not a side effect of cloning."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.server import start_remote_access


class RemoteStartup(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        helpers = self.root / "remote-access"
        helpers.mkdir()
        (helpers / "Start-Remote.ps1").touch()

    def test_default_never_starts_public_access_even_when_helpers_exist(self):
        with mock.patch("serve.server.os.name", "nt"), mock.patch("serve.remote_access.start_remote_access") as start:
            self.assertIsNone(start_remote_access(self.root))
        start.assert_not_called()

    def test_explicit_opt_in_delegates_to_monitor(self):
        with mock.patch("serve.server.os.name", "nt"), mock.patch("serve.remote_access.start_remote_access") as start:
            worker = start_remote_access(self.root, enabled=True)
        start.assert_called_once_with(self.root)
        self.assertIs(worker, start.return_value)

    def test_non_windows_is_unchanged_when_enabled(self):
        with mock.patch("serve.server.os.name", "posix"), mock.patch("serve.remote_access.start_remote_access") as start:
            self.assertIsNone(start_remote_access(self.root, enabled=True))
        start.assert_not_called()

    def test_missing_helpers_do_not_start_monitor(self):
        with mock.patch("serve.server.os.name", "nt"), mock.patch("serve.remote_access.start_remote_access") as start:
            self.assertIsNone(start_remote_access(self.root / "absent", enabled=True))
        start.assert_not_called()
