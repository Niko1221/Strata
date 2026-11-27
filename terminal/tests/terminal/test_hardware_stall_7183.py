"""#7183: `strata launch` froze for 12 s on a Windows PC (Ryzen 7 5700X + Radeon AI PRO R9700) at the first key press that needed the computer's
hardware after ten idle minutes.  The shared hardware reader (tools/strata_mcp.py Strata.hardware) keeps its answer for 600 s and then read the PC
on the caller's thread: seven child processes, four of them a fresh PowerShell start (the CPU name and three Win32_VideoController queries).

Three changes, each pinned here:
  - a screen that lives for hours (the Terminal's Catalog) gets the old answer at once and the re-read runs in a background thread (stale_ok);
  - the Windows display adapters are read ONCE for one hardware read (setup.win_video_scope), and feed all three consumers;
  - the CPU name comes from the registry, not from PowerShell.
No Windows PC and no PowerShell here: the child process and the registry are fakes."""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest import mock

HW1 = {"ram_gb": 64.0, "os": "Windows 11", "cpu": {"name": "first"}, "gpus": []}
HW2 = {"ram_gb": 64.0, "os": "Windows 11", "cpu": {"name": "second"}, "gpus": []}


def strata():
    from tests.terminal.launch_support import Home
    from strata_cli.launch import catalog as C
    home = Home()
    home.__enter__()
    cat = C.Catalog()
    return home, cat


class StaleAnswer(unittest.TestCase):
    def setUp(self):
        self.home, self.cat = strata()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.s = self.cat.strata
        self.gate = threading.Event()
        self.reads = []

        def read():
            self.reads.append(time.monotonic())
            if len(self.reads) > 1:
                self.gate.wait(10)              # the second read is the slow one (12 s on the reporter's PC)
            return dict(HW1 if len(self.reads) == 1 else HW2)
        p = mock.patch.object(self.s, "_hardware", read)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.gate.set)

    def age(self, seconds):
        stamp, hw = self.s._hw_cache
        self.s._hw_cache = (stamp - seconds, hw)

    def test_the_first_read_is_waited_for(self):
        self.assertEqual(self.s.hardware(stale_ok=True)["cpu"]["name"], "first")
        self.assertEqual(len(self.reads), 1)

    def test_an_old_answer_is_returned_at_once_and_read_again_behind_the_screen(self):
        self.s.hardware()
        self.age(self.s.HW_TTL_S + 5)
        began = time.monotonic()
        hw = self.s.hardware(stale_ok=True)
        self.assertLess(time.monotonic() - began, 2.0, "the key press must not wait for the read")
        self.assertEqual(hw["cpu"]["name"], "first")
        self.assertIsNotNone(self.s._hw_thread)
        self.gate.set()
        self.s._hw_thread.join(10)
        self.assertEqual(len(self.reads), 2)
        self.assertEqual(self.s.hardware(stale_ok=True)["cpu"]["name"], "second")     # the next call sees the new answer
        self.assertEqual(len(self.reads), 2)                                          # and it is fresh again: no third read

    def test_two_key_presses_start_one_read(self):
        self.s.hardware()
        self.age(self.s.HW_TTL_S + 5)
        self.s.hardware(stale_ok=True)
        self.s.hardware(stale_ok=True)
        self.s.hardware(stale_ok=True)
        self.gate.set()
        self.s._hw_thread.join(10)
        self.assertEqual(len(self.reads), 2)

    def test_a_read_that_fails_keeps_the_old_answer_and_tries_again_later(self):
        self.s.hardware()
        self.age(self.s.HW_TTL_S + 5)
        with mock.patch.object(self.s, "_hardware", side_effect=RuntimeError("PowerShell is gone")):
            self.s.hardware(stale_ok=True)
            self.s._hw_thread.join(10)
        self.assertEqual(self.s.hardware(stale_ok=True)["cpu"]["name"], "first")
        self.gate.set()
        self.s._hw_thread.join(10)
        self.assertFalse(self.s._hw_refreshing)

    def test_without_stale_ok_an_old_answer_is_still_read_on_the_spot(self):
        # the app's own readers keep the rule they had: an answer older than the limit is not served
        self.s.hardware()
        self.age(self.s.HW_TTL_S + 5)
        self.gate.set()
        self.assertEqual(self.s.hardware()["cpu"]["name"], "second")
        self.assertIsNone(self.s._hw_thread)

    def test_fresh_reads_now_even_with_stale_ok(self):
        self.s.hardware()
        self.gate.set()
        self.assertEqual(self.s.hardware(fresh=True, stale_ok=True)["cpu"]["name"], "second")

    def test_the_catalog_asks_for_the_stale_answer(self):
        seen = {}

        def fake(fresh=False, stale_ok=False):
            seen.update(fresh=fresh, stale_ok=stale_ok)
            return {"ram_gb": 1.0, "os": "Windows 11", "cpu": {}, "gpus": []}
        with mock.patch.object(self.s, "hardware", fake):
            self.cat.hardware()
        self.assertEqual(seen, {"fresh": False, "stale_ok": True})


ROWS = ("AMD Radeon AI PRO R9700|PCI\\VEN_1002&DEV_7551&SUBSYS_0000&REV_00\\4&1|4293918720|32.0.13031.3015|3840\r\n"
        "Microsoft Basic Display Adapter|PCI\\VEN_1002&DEV_164E&SUBSYS_0000&REV_00\\4&2|536870912|10.0.19041.1|0\r\n"
        "Remote Display||0||0\r\n")


class OneDisplayRead(unittest.TestCase):
    def setUp(self):
        import setup
        self.setup = setup
        self.calls = []

        def out(cmd):
            self.calls.append(cmd)
            return ROWS
        p = mock.patch.object(setup, "out", out)
        p.start()
        self.addCleanup(p.stop)

    def test_the_rows_carry_what_each_consumer_wants(self):
        rows = self.setup.win_video_rows()
        self.assertEqual([r["name"] for r in rows], ["AMD Radeon AI PRO R9700", "Microsoft Basic Display Adapter", "Remote Display"])
        self.assertEqual((rows[0]["ram"], rows[0]["driver"], rows[0]["res"]), (4293918720, "32.0.13031.3015", 3840))
        self.assertEqual(rows[1]["res"], 0)

    def test_the_adapter_list_keeps_its_shape_and_leaves_out_a_row_with_no_pci_id(self):
        found = self.setup._win_display_adapters()
        self.assertEqual([a["name"] for a in found], ["AMD Radeon AI PRO R9700", "Microsoft Basic Display Adapter"])
        self.assertEqual(set(found[0]), {"name", "pnp", "ram", "driver"})

    def test_outside_a_scope_every_call_asks_again(self):
        self.setup.win_video_rows()
        self.setup.win_video_rows()
        self.assertEqual(len(self.calls), 2)

    def test_inside_a_scope_the_adapters_are_read_once(self):
        with self.setup.win_video_scope():
            self.setup.win_video_rows()
            self.setup._win_display_adapters()
            with self.setup.win_video_scope():                      # a nested scope shares the outer read
                self.setup.win_video_rows()
        self.assertEqual(len(self.calls), 1)
        self.setup.win_video_rows()                                 # and the scope is gone after it
        self.assertEqual(len(self.calls), 2)

    def test_a_scope_hands_out_copies(self):
        with self.setup.win_video_scope():
            self.setup.win_video_rows()[0]["name"] = "changed"
            self.assertEqual(self.setup.win_video_rows()[0]["name"], "AMD Radeon AI PRO R9700")

    def test_the_old_three_column_answer_still_parses(self):
        with mock.patch.object(self.setup, "out", return_value="Some GPU|PCI\\VEN_10DE&DEV_0A60&SUBSYS|0\r\n"):
            found = self.setup._win_display_adapters()
        self.assertEqual(len(found), 1)
        self.assertIn("VEN_10DE", found[0]["pnp"])


class WindowsHardwareRead(unittest.TestCase):
    """One hardware read on a Windows PC starts ONE process for the display adapters, not three."""

    @mock.patch("strata_control.engine_component.status", lambda: {"usable": False})
    def test_the_three_consumers_share_one_read(self):
        home, cat = strata()
        self.addCleanup(home.__exit__, None, None, None)
        s = cat.strata
        M = sys.modules[type(s).__module__]               # the Catalog loads strata_mcp under its own name
        S = s.setup_module()
        calls = []

        def out(cmd):
            calls.append(cmd)
            return ROWS
        card = {"index": 0, "name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "arch": "gfx1201", "driver": "32.0", "pci_id": 0x7551}

        def amd_gpus():
            S._win_display_adapters()                       # what the real one does first on Windows (amd_gpus_windows)
            return [dict(card)]
        # the registry and the HIP runtime are not the point here; the display adapters are
        with mock.patch.object(M, "WIN", True), mock.patch.object(S, "out", out), mock.patch.object(M, "run_quiet", lambda cmd, t=60: out(cmd)), \
                mock.patch.object(S, "gpus", lambda: []), mock.patch.object(S, "amd_gpus", amd_gpus), \
                mock.patch.object(S, "ram_gb", lambda: 63.6), mock.patch.object(S, "page_file_gb", lambda: 10.0), \
                mock.patch.object(S, "cpu_info", lambda: ("AMD Ryzen 7 5700X", True, False)), \
                mock.patch.object(S, "amd_problem", lambda g: None):
            hw = s._hardware()
        self.assertEqual(len([c for c in calls if "Win32_VideoController" in " ".join(map(str, c))]), 1, calls)
        self.assertTrue(hw["gpus"][0]["drives_display"])                  # the desktop resolution came from the same read
        self.assertEqual([d["name"] for d in hw["display_adapters"]],
                         ["AMD Radeon AI PRO R9700", "Microsoft Basic Display Adapter", "Remote Display"])
        # the display answer is gone after the read: no stale rows for the next caller
        self.assertIsNone(M._hardware_video_rows())

    def test_outside_a_read_the_old_readers_run_their_own_command(self):
        import strata_mcp as M
        with mock.patch.object(M, "run_quiet", return_value="{PNP}|1920\n") as rq:
            self.assertEqual(M.windows_desktop_adapters(), ["{PNP}"])
        self.assertEqual(rq.call_count, 1)


class CpuNameFromRegistry(unittest.TestCase):
    def fake_winreg(self, name):
        class Key:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        mod = types.ModuleType("winreg")
        mod.HKEY_LOCAL_MACHINE = 1
        mod.OpenKey = lambda *a: Key()
        if name is None:
            def query(key, value):
                raise OSError("no such value")
        else:
            def query(key, value):
                assert value == "ProcessorNameString"
                return (name, 1)
        mod.QueryValueEx = query
        return mod

    def test_the_name_is_read_without_a_process(self):
        import setup
        with mock.patch.dict(sys.modules, {"winreg": self.fake_winreg("  AMD Ryzen 7 5700X   8-Core Processor           ")}):
            self.assertEqual(setup._win_cpu_name_registry(), "AMD Ryzen 7 5700X 8-Core Processor")

    def test_an_unreadable_key_gives_an_empty_name_so_the_old_reader_runs(self):
        import setup
        with mock.patch.dict(sys.modules, {"winreg": self.fake_winreg(None)}):
            self.assertEqual(setup._win_cpu_name_registry(), "")

    def test_no_winreg_gives_an_empty_name(self):
        import setup
        with mock.patch.dict(sys.modules, {"winreg": None}):
            self.assertEqual(setup._win_cpu_name_registry(), "")


if __name__ == "__main__":
    unittest.main()
