"""Windows process snapshots used to recognise a venv redirector's child."""
import ctypes
import os
import subprocess
import sys
import unittest
from unittest import mock

from strata_control import winproc


@unittest.skipUnless(os.name == "nt", "Windows process snapshots")
class ParentPidTests(unittest.TestCase):
    def test_current_process_parent(self):
        self.assertEqual(winproc.parent_pid(os.getpid()), os.getppid())

    def test_venv_interpreter_is_a_direct_child_of_its_redirector(self):
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", "import os,sys; print(os.getpid(), flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            interpreter_pid = int(child.stdout.readline())
            if interpreter_pid == child.pid:
                self.skipTest("this interpreter does not use a redirector")
            self.assertEqual(winproc.parent_pid(interpreter_pid), child.pid)
        finally:
            child.communicate("\n", timeout=10)

    def test_invalid_pid_is_not_queried(self):
        with mock.patch.object(winproc._k32, "CreateToolhelp32Snapshot") as snapshot:
            for pid in (None, "123", True, -1, 0, 1 << 32):
                with self.subTest(pid=pid):
                    self.assertIsNone(winproc.parent_pid(pid))
            snapshot.assert_not_called()

    def test_snapshot_failure_is_not_a_parent_match(self):
        with mock.patch.object(winproc._k32, "CreateToolhelp32Snapshot",
                               return_value=ctypes.c_void_p(-1).value), \
                mock.patch.object(winproc._k32, "CloseHandle") as close:
            self.assertIsNone(winproc.parent_pid(os.getpid()))
            close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
