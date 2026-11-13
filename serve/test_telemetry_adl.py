"""The Monitor's AMD-on-Windows readings (ADL2 PMLog) degrade safely and map to the gpu_* keys the web UI uses.

No GPU or download is required for the safety half: on a box without atiadlxx.dll every reader must report
"unavailable" (ok() False, read() {}) and never raise, so the server keeps running.  On a box that does have the
driver (the R9700 / RX 9070 XT the Monitor is meant for), the same reader must fill gpu_util / gpu_temp /
gpu_mem_used / gpu_power, and each card's own PCIe link (PMLog's negotiated BUS_SPEED / BUS_LANES) so a
motherboard's PCI bifurcation shows the real width per slot.  A reader also holds an ADL context, so a one-off
read has to give it back - the handle-growth finding reported on #1434."""
import gc
import os
import unittest

from serve.telemetry import _AmdWin, amd_card_count, free_vram_mib, gpu_reader


class AdlDegrades(unittest.TestCase):
    """Nothing in the Windows AMD path may raise: an absent driver, a blocked call, or an unreadable sensor all
    leave the reader off, and the server still runs."""

    def test_reader_never_raises(self):
        for index in (0, 1, 5):
            r = gpu_reader(index, amd=True)
            self.assertTrue(hasattr(r, "ok") and hasattr(r, "read"))
            self.assertIsInstance(r.read(), dict)
            self.assertIsInstance(r.name(), str)

    def test_free_vram_never_raises(self):
        for index in (0, 1):
            free = free_vram_mib(index, amd=True)
            self.assertTrue(free is None or isinstance(free, int))

    def test_card_count_never_raises(self):
        n = amd_card_count()
        self.assertTrue(n is None or isinstance(n, int))

    def test_unavailable_off(self):
        if _AmdWin(0).lib is None:  # no atiadlxx.dll here: the reader must be fully off
            r = _AmdWin(0)
            self.assertFalse(r.ok())
            self.assertEqual(r.read(), {})
            self.assertEqual(amd_card_count(), 0)


class AdlReadings(unittest.TestCase):
    """On a machine that has the driver, the reader fills the gpu_* keys the Monitor tab renders."""

    def test_readings_when_driver_present(self):
        r = _AmdWin(0)
        if not r.ok():
            return  # no AMD driver on this box; the degrade test above covers it
        out = r.read()
        self.assertIn("util", out)
        self.assertIn("mem_used", out)
        self.assertIn("mem_total", out)
        self.assertGreaterEqual(out["mem_used"], 0)
        self.assertLessEqual(out["mem_used"], out["mem_total"])
        if out.get("temp") is not None:
            self.assertTrue(0 < out["temp"] < 120)  # degrees C, not millidegrees
        if out.get("power") is not None:
            self.assertTrue(0 < out["power"] < 1000)  # watts (BOARD_POWER factor 1.0), not milliwatts
        # PCIe is the per-card negotiated link, not a device-wide max
        for k in ("pcie_gen_max", "pcie_width"):
            self.assertTrue(out.get(k) is None or isinstance(out[k], int))

    def test_every_card_enumerates(self):
        """The Monitor shows a tile per card, so every present discrete card (index < card_count) must read."""
        n = amd_card_count()
        if n < 1:
            return
        for i in range(n):
            self.assertTrue(_AmdWin(i).ok())


class AdlReaderCleanup(unittest.TestCase):
    """A reader holds an ADL context and a PMLog device handle, so a one-off read has to give them back: #1434
    measured +2 process handles per free_vram_mib() call on a 7900 XTX before its close(), and the server calls
    that in a 0.5 s retry loop while it waits for VRAM."""

    def test_stalled_sensors_restart_the_stream_once(self):
        """The driver tears PMLog down in deep idle while QueryPMLogData_Get keeps answering with the frozen
        buffer: identical util+power STALL_LIMIT samples in a row must restart the stream exactly once."""
        r = _AmdWin(0)          # works without a driver: the stall watch never touches the library
        restarts = []
        r._restart_pmlog = lambda: restarts.append(1)
        for _ in range(r.STALL_LIMIT - 1):
            r._watch_stall({"util": 0.0, "power": 5.0})
        self.assertEqual(restarts, [])
        r._watch_stall({"util": 0.0, "power": 5.0})   # the STALL_LIMIT-th identical sample
        self.assertEqual(len(restarts), 1)
        r._watch_stall({"util": 0.0, "power": 5.0})   # counter reset: no second restart right away
        self.assertEqual(len(restarts), 1)
        r._watch_stall({"util": 3.0, "power": 6.0})   # a moving reading: live stream, nothing to do
        for _ in range(r.STALL_LIMIT):
            r._watch_stall({"util": None, "power": None})   # no sensors at all: never restart on dead keys
        self.assertEqual(len(restarts), 1)

    def test_close_is_safe_and_idempotent(self):
        r = _AmdWin(0)          # on a box without atiadlxx.dll this reader never opened anything at all
        r.close()
        r.close()               # closing twice must not fault on handles that are already gone
        self.assertFalse(r.ok())
        self.assertEqual(r.read(), {})
        self.assertEqual(r.card_count, 0)

    @unittest.skipIf(os.name != "nt", "the ADL reader is Windows only")
    def test_one_off_reads_do_not_grow_handles(self):
        if _AmdWin(0).lib is None:
            self.skipTest("no atiadlxx.dll on this box")
        if self.handles() is None:
            self.skipTest("kernel32 GetProcessHandleCount is unavailable")
        for _ in range(3):      # warm up: the first read starts PMLog and caches the Device Manager query
            free_vram_mib(0, amd=True)
        gc.collect()
        before = self.handles()
        for _ in range(40):
            free_vram_mib(0, amd=True)
        gc.collect()
        after = self.handles()
        self.assertLessEqual(after, before + 4, f"40 one-off reads grew the handle count {before} -> {after}")

    @staticmethod
    def handles():
        """The process handle count, without a dependency: kernel32's GetProcessHandleCount, the same figure
        psutil reports and the one #1434's reproduction measured.  None when the call is unavailable."""
        import ctypes
        count = ctypes.c_uint(0)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessHandleCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        if not kernel32.GetProcessHandleCount(kernel32.GetCurrentProcess(), ctypes.byref(count)):
            return None
        return count.value


@unittest.skipIf(os.name != "nt", "the Device Manager link query is a Windows PowerShell call")
class DmLinkMap(unittest.TestCase):
    """The PCIe link map (PR #4884 review): the PowerShell query must never show a console window on a
    server started without one, must not block the first read() (the free-VRAM admission check), and a
    failed query must be retried at most every 60 s, not on every 1-second sample."""

    def setUp(self):
        import serve.telemetry as t
        for name, value in (("_DM_LINK", None), ("_DM_LINK_RETRY", 0.0), ("_DM_LINK_STARTED", False)):
            setattr(t, name, value)

    def test_query_never_shows_a_console(self):
        import subprocess
        from unittest import mock
        import serve.telemetry as t
        with mock.patch.object(t.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="")
            t._dm_link_query()
        self.assertEqual(run.call_args.kwargs["creationflags"], subprocess.CREATE_NO_WINDOW)

    def test_first_call_returns_empty_and_starts_the_query(self):
        from unittest import mock
        import serve.telemetry as t
        with mock.patch.object(t.threading, "Thread") as thread:
            self.assertEqual(t._dm_link_map(), {})  # the first read() is not blocked by the seconds-long query
        thread.return_value.start.assert_called_once()
        self.assertTrue(t._DM_LINK_STARTED)

    def test_failure_retries_after_60_s_not_every_sample(self):
        import time
        from unittest import mock
        import serve.telemetry as t
        with mock.patch.object(t.subprocess, "run", side_effect=OSError("no powershell")):
            t._dm_link_query()
        self.assertIsNone(t._DM_LINK)
        self.assertFalse(t._DM_LINK_STARTED)
        self.assertGreater(t._DM_LINK_RETRY, time.monotonic())  # retried later, not on the next sample
        with mock.patch.object(t.threading, "Thread") as thread:
            self.assertEqual(t._dm_link_map(), {})
        thread.return_value.start.assert_not_called()  # inside the retry window: no relaunch

    def test_success_caches(self):
        from unittest import mock
        import serve.telemetry as t
        with mock.patch.object(t.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="DEV_1234 4 16 4 16\nDEV_5678 5 8 5 16\njunk\n")
            t._dm_link_query()
        self.assertEqual(t._DM_LINK, {"1234": (4, 16, 4, 16), "5678": (5, 8, 5, 16)})
        self.assertEqual(t._dm_link_map(), t._DM_LINK)  # cached: no second query


if __name__ == "__main__":
    unittest.main()
