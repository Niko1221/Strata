"""pytest, tools/: the tests here describe Linux and Windows PCs on x86 (NVIDIA, AMD and Intel cards; the Spark and the Mac tests say
so themselves by patching the platform or the CPU). On a Mac every one of them ran the Mac or ARM branch instead: setup.main() goes to
the Metal installer (`sys.platform == "darwin"`), the hardware tools take the unified-memory path, and an Apple CPU reads as ARM64.
On macOS only, each test runs as on GitHub's Linux runner: sys.platform "linux", platform.system "Linux" and an x86_64 CPU
(platform.machine and
strata_control.sysinfo.machine). Linux and Windows are not touched. A test that wants darwin or arm64 patches it on top, as before."""
import platform
import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _a_linux_x86_pc_on_macos():
    if sys.platform != "darwin":
        yield
        return
    from strata_control import sysinfo
    with mock.patch.object(sys, "platform", "linux"), \
            mock.patch.object(platform, "machine", lambda: "x86_64"), \
            mock.patch.object(platform, "system", lambda: "Linux"), \
            mock.patch.object(sysinfo, "machine", lambda: "x86_64"):
        yield
