"""#4975: `host_platform()` answers without `platform.uname()` on Windows, whose WMI call has no timeout.

`platform.system()` and `platform.machine()` are both `platform.uname()`; on Windows Python 3.12 `uname()` reads
WMI (`platform._wmi_query`), so a stuck `winmgmt` service hung every Control command that resolves the engine
component.  The OS and the architecture are known without it.

A real WMI stall cannot be produced from a test, so the mock below stands in for the stuck service: it makes every
`platform` call the Windows path would use fail.  The tests also keep the answers of the other systems as they were.

    python -m pytest tests/terminal/test_host_platform.py
"""
import os
import sys
import unittest
from unittest import mock

from strata_control import targets


def _no_wmi():
    """The platform calls the Windows path used to make, all failing like a stuck WMI service would."""
    return (mock.patch.object(targets.platform, "uname", side_effect=AssertionError("WMI is stuck")),
            mock.patch.object(targets.platform, "system", side_effect=AssertionError("WMI is stuck")),
            mock.patch.object(targets.platform, "machine", side_effect=AssertionError("WMI is stuck")))


class HostPlatform(unittest.TestCase):
    def test_windows_is_known_while_wmi_is_stuck(self):
        a, b, c = _no_wmi()
        with a, b, c, \
                mock.patch.object(sys, "platform", "win32"), \
                mock.patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": ""}):
            self.assertEqual(targets.host_platform(), "windows-x86_64")

    def test_windows_arm64_comes_from_the_same_source(self):
        a, b, c = _no_wmi()
        with a, b, c, \
                mock.patch.object(sys, "platform", "win32"), \
                mock.patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "ARM64", "PROCESSOR_ARCHITEW6432": ""}):
            self.assertEqual(targets.host_platform(), "windows-arm64")

    def test_macos_and_linux_keep_their_answers(self):
        with mock.patch.object(sys, "platform", "darwin"), \
                mock.patch.object(targets.platform, "system", return_value="Darwin"), \
                mock.patch.object(targets.platform, "machine", return_value="arm64"):
            self.assertEqual(targets.host_platform(), "macos-arm64")
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(targets.platform, "system", return_value="Linux"), \
                mock.patch.object(targets.platform, "machine", return_value="aarch64"):
            self.assertEqual(targets.host_platform(), "linux-aarch64")

    def test_the_answer_is_a_key_of_the_backend_matrix(self):
        self.assertIn(targets.host_platform(), {target for targets_ in targets.BACKEND_MATRIX.values()
                                                 for target in targets_})


if __name__ == "__main__":
    unittest.main()
