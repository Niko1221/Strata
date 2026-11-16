"""Startup must not inspect command lines of unrelated processes."""
import unittest
from unittest import mock

from strata_control import procs


class OrphanScan(unittest.TestCase):
    def test_foreign_executable_is_filtered_before_command_line_access(self):
        foreign = mock.Mock(pid=70001)
        foreign.exe.return_value = "/other/application"
        foreign.cmdline.side_effect = AssertionError("unrelated command-line read")
        fake = mock.Mock()
        fake.Error = RuntimeError
        fake.NoSuchProcess = ProcessLookupError
        fake.Process.return_value.parents.return_value = []
        fake.process_iter.return_value = [foreign]
        with mock.patch.object(procs, "psutil", fake), mock.patch.object(procs, "WINDOWS", False), \
                mock.patch.object(procs, "same_file", return_value=False):
            self.assertEqual(procs.orphans_of("/wanted/engine"), [])
        fake.process_iter.assert_called_once_with(["pid"])
        foreign.cmdline.assert_not_called()
        foreign.ppid.assert_not_called()
        foreign.as_dict.assert_not_called()

    def test_posix_candidate_uses_one_ownership_snapshot_after_executable_match(self):
        candidate = mock.Mock(pid=70001, info={"pid": 70001})
        candidate.exe.return_value = "/wanted/engine"
        snapshot = {"pid": 70001, "ppid": 1, "cmdline": ["engine", "--serve"], "create_time": 100.0}
        candidate.as_dict.return_value = snapshot
        fake = mock.Mock(Error=RuntimeError)
        fake.Process.return_value.parents.return_value = []
        fake.process_iter.return_value = [candidate]
        with mock.patch.object(procs, "psutil", fake), mock.patch.object(procs, "WINDOWS", False), \
                mock.patch.object(procs, "same_file", return_value=True), \
                mock.patch.object(procs, "identify", return_value="synthetic-ident"):
            self.assertEqual(procs.orphans_of("/wanted/engine"), ["synthetic-ident"])
        fake.process_iter.assert_called_once_with(["pid"])
        candidate.as_dict.assert_called_once_with(attrs=["pid", "ppid", "cmdline", "create_time"])
        candidate.cmdline.assert_not_called()
        candidate.ppid.assert_not_called()
        candidate.create_time.assert_not_called()
