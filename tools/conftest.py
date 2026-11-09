"""pytest, tools/: setup.py's tests describe Linux and Windows PCs (NVIDIA, AMD, Intel cards). On a Mac, setup.main() goes to the Metal
installer (`sys.platform == "darwin"`), so every such test ran the Mac flow there instead. On macOS only, setup.py sees "linux" for the
duration of each test; the Mac tests (test_setup_mac.py) set "darwin" themselves, as they do on Linux and Windows. Only setup.py's own
view changes (its module-level `sys` is a view of the real one with another platform); the rest of the process still sees darwin."""
import sys
from unittest import mock

import pytest


class _SysView:
    """The real sys module with its own `platform`; every other read and write goes to sys itself."""

    def __init__(self, real, platform):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "platform", platform)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name == "platform":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)

    def __delattr__(self, name):
        if name == "platform":
            object.__setattr__(self, name, "linux")
        else:
            delattr(object.__getattribute__(self, "_real"), name)


@pytest.fixture(autouse=True)
def _setup_sees_a_pc_on_macos():
    setup = sys.modules.get("setup")
    if sys.platform != "darwin" or setup is None or getattr(setup, "sys", None) is not sys:
        yield
        return
    with mock.patch.object(setup, "sys", _SysView(sys, "linux")):
        yield
