"""#7060: the engine's READY deadline (config "engine_ready_s", STRATA_ENGINE_READY_S, default 900 s) counts from the last sign of
life, not from the start: a model read from a slow disk that keeps writing to its log (or keeps reading files) is not ended after 15
minutes; an engine that goes quiet still is.  The override keeps its meaning, 0 waits for ever.

    python -m pytest serve/test_ready_deadline.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import serve.server as server  # noqa: E402
from serve.server import ReadyDeadline, StrataEngine, apply_ready_timeout  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class DeadlineMoves(unittest.TestCase):
    def make(self, window=900.0, ceiling=7200.0):
        self.clock = Clock()
        self.signs = [0, 0]
        self.dl = ReadyDeadline(window, lambda: tuple(self.signs), now=self.clock, ceiling_s=ceiling)

    def at(self, seconds):
        self.clock.t = 1000.0 + seconds
        return self.dl.left()

    def test_a_silent_start_ends_at_the_window_as_before(self):
        self.make()
        self.assertAlmostEqual(self.at(0), 900.0)
        self.assertGreater(self.at(899), 0)
        self.assertLessEqual(self.at(900), 0)
        self.assertFalse(self.dl.moved)

    def test_a_growing_log_moves_the_deadline(self):
        self.make()
        self.at(800)
        self.signs[0] = 10                              # a line before the look at 850 s: the deadline is now 850 + 900
        self.assertAlmostEqual(self.at(850), 900.0)
        self.assertTrue(self.dl.moved)
        self.assertGreater(self.at(1749), 0)
        self.assertLessEqual(self.at(1750), 0)          # quiet since 850: ended 900 s after the last sign

    def test_reading_from_disk_is_a_sign_of_life_too(self):
        self.make()
        self.at(600)
        self.signs[1] = 5_000_000
        self.assertAlmostEqual(self.at(1000), 900.0)     # the read seen at 1000 s: 900 s more

    def test_nothing_new_moves_nothing(self):
        self.make()
        self.signs[:] = [5, 5]
        self.at(0)                                      # (the first look after the start sees these as new: the deadline is 900 s from now)
        self.at(100)
        self.at(500)                                    # the same numbers again: no movement
        self.assertLessEqual(self.at(900), 0)

    def test_the_ceiling_ends_a_start_that_keeps_writing(self):
        self.make(window=900.0, ceiling=7200.0)
        t = 0
        while t < 6900:
            t += 300
            self.signs[0] += 1
            self.assertGreater(self.at(t), 0, t)
        t += 300                                        # 7300 s: past the ceiling, whatever it printed
        self.signs[0] += 1
        self.assertLessEqual(self.at(7300), 0)

    def test_a_window_above_a_quarter_of_the_ceiling_gets_four_windows(self):
        self.make(window=3600.0, ceiling=7200.0)        # (an engine_ready_s of 3600: up to 4 h in all)
        self.signs[0] = 1
        self.at(3500)
        self.signs[0] = 2
        self.assertGreater(self.at(10000), 0)           # more than the 7200 s ceiling: four windows win
        self.signs[0] = 3
        self.assertLessEqual(self.at(14400), 0)

    def test_a_probe_that_fails_changes_nothing(self):
        clock = Clock()
        dl = ReadyDeadline(900.0, mock.Mock(side_effect=RuntimeError("no psutil")), now=clock)
        clock.t += 899
        self.assertGreater(dl.left(), 0)
        clock.t += 2
        self.assertLessEqual(dl.left(), 0)


# a fake engine: a line to its log every 0.25 s for `busy` seconds (stderr is the engine's log), then READY (or never)
FAKE = ("import sys, time\n"
        "busy = float(sys.argv[sys.argv.index('--busy') + 1])\n"
        "end = time.time() + busy\n"
        "while time.time() < end:\n"
        "    sys.stderr.write('loading ...\\n'); sys.stderr.flush(); time.sleep(0.25)\n"
        "if '--never' in sys.argv:\n"
        "    time.sleep(600)\n"
        "print('READY 4096 stop', flush=True)\n"
        "for line in sys.stdin:\n"
        "    if line.startswith('QUIT'):\n"
        "        break\n")


class LiveStart(unittest.TestCase):
    def start(self, busy, window, never=False):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        script = Path(d.name) / "fake_strata.py"
        script.write_text(FAKE, encoding="utf-8")
        log = str(Path(d.name) / "engine.log")
        real = server.subprocess.Popen
        procs = []

        def popen(cmd, **kw):
            procs.append(real([sys.executable, str(script), *cmd[2:]], **kw))
            return procs[-1]
        extra = ["--busy", str(busy)] + (["--never"] if never else [])
        with mock.patch.object(server.subprocess, "Popen", popen), mock.patch.object(server, "ENGINE_READY_S", window), \
                mock.patch.object(server, "READY_POLL_S", 0.2), mock.patch.object(server, "narrate_start", lambda *a, **k: None), \
                mock.patch.object(server, "end_leftover_engines", lambda *a, **k: None):
            try:
                eng = StrataEngine("strata", extra, log=log)
            except RuntimeError as e:
                return None, str(e), procs
            self.addCleanup(eng.close)
            return eng, "", procs

    def test_a_slow_start_that_keeps_writing_is_not_ended(self):
        t0 = time.monotonic()
        eng, err, _ = self.start(busy=3.5, window=1.2)      # 3.5 s of lines, a window of 1.2 s: the old rule ended it at 1.2 s
        self.assertEqual(err, "")
        self.assertEqual(eng.max_context, 4096)
        self.assertGreater(time.monotonic() - t0, 3.0)

    def test_a_start_that_goes_quiet_is_ended_after_the_window(self):
        t0 = time.monotonic()
        eng, err, procs = self.start(busy=1.0, window=1.5, never=True)     # lines for 1 s, then nothing for ever
        self.assertIsNone(eng)
        self.assertIn("did not report READY within 2 s", err)               # 1.5 s rounds to 2
        self.assertIn("last sign of life", err)
        self.assertIn("engine_ready_s", err)
        self.assertLess(time.monotonic() - t0, 20)
        procs[0].wait(10)
        self.assertIsNotNone(procs[0].poll())


class OverrideStaysAnOverride(unittest.TestCase):
    def test_the_config_sets_the_window(self):
        with mock.patch.object(server, "ENGINE_READY_S", 900.0):
            apply_ready_timeout({"engine_ready_s": 3600})
            self.assertEqual(server.ENGINE_READY_S, 3600.0)
        with mock.patch.object(server, "ENGINE_READY_S", 900.0):
            apply_ready_timeout({})
            self.assertEqual(server.ENGINE_READY_S, 900.0)
        with self.assertRaises(ValueError):
            apply_ready_timeout({"engine_ready_s": 0})


if __name__ == "__main__":
    unittest.main()
