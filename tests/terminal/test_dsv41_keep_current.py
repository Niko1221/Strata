"""The app starts a saved profile without going through `strata launch`: before `runtime start` the DeepSeek engine in that profile's run config is
kept current (strata_control.dsv41_engine.keep_current).  A newer package is fetched by a background process (the start waits for it a little, then goes
on), the run configs of every saved profile are pointed at the newest installed package, a folder a running engine uses is never removed, and nothing
here ever fails a start."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import dsv41_engine  # noqa: E402
from tests.terminal import test_dsv41_upgrade as _up  # noqa: E402

EXE = dsv41_engine.exe_name()


class FakeChild:
    """The background fetch: runs the upgrade in this process when waited for (the test box has no GPU to detect), or never ends."""
    def __init__(self, work=None, hang=False):
        self.work, self.hang = work, hang

    def wait(self, timeout=None):
        if self.hang:
            raise subprocess.TimeoutExpired("fetch", timeout)
        if self.work:
            self.work()
        return 0


class KeepCurrent(_up.Upgrade):
    def setUp(self):
        super().setUp()
        self.cfg = self.root / "run" / "app-1.json"
        self.cfg2 = self.root / "run" / "app-2.json"
        self.cfg.parent.mkdir()
        for c in (self.cfg, self.cfg2):
            self.write_cfg(c, self.old)
        self.lines = []

    def write_cfg(self, path, exe):
        path.write_text(json.dumps({"exe": str(exe), "args": ["--pack", "p"], "tokenizer": "t", "lib_dirs": ["C:/other", str(Path(exe).parent)]}), encoding="utf-8")

    def exe_of(self, path):
        return Path(json.loads(Path(path).read_text(encoding="utf-8"))["exe"])

    def keep(self, child=None, **kw):
        spawn = mock.patch.object(dsv41_engine, "_spawn_fetch", return_value=child or FakeChild())
        with spawn as sp, mock.patch.object(dsv41_engine, "in_use", return_value=kw.pop("busy", False)):
            out = dsv41_engine.keep_current(self.cfg, source=self.url, log=self.lines.append, configs=[self.cfg2], wait=5, **kw)
        return out, sp

    def fetch_now(self):
        return lambda: self.upgrade()

    def test_a_due_check_fetches_waits_and_every_saved_profile_points_at_the_new_engine(self):
        self.publish("0.1.49.1")
        out, sp = self.keep(FakeChild(self.fetch_now()))
        sp.assert_called_once()
        new = dsv41_engine.find(self.dest)
        self.assertEqual(out, new)
        self.assertIn("0.1.49.1", str(new))
        self.assertEqual(self.exe_of(self.cfg), new)
        self.assertEqual(self.exe_of(self.cfg2), new, "the other saved profile must not be left on a folder that goes")
        libs = json.loads(self.cfg.read_text(encoding="utf-8"))["lib_dirs"]
        self.assertEqual(libs[0], "C:/other")
        self.assertIn(str(new.parent), libs)
        self.assertEqual(self.folders(), ["0.1.49.1"])
        self.assertIn("used from now on", " ".join(self.lines))

    def test_a_fresh_check_is_not_repeated_and_costs_no_network(self):
        self.publish("0.1.49.1")
        dsv41_engine._write_check(self.dest, at=time.time() - 60, ok=True)
        out, sp = self.keep()
        sp.assert_not_called()
        self.assertEqual(out, self.old)
        self.assertEqual(self.exe_of(self.cfg), self.old)

    def test_the_check_is_due_after_six_hours_or_half_an_hour_after_a_failure(self):
        now = time.time()
        self.assertTrue(dsv41_engine.fetch_due(self.dest, now))                       # never checked
        dsv41_engine._write_check(self.dest, at=now - 3600, ok=True)
        self.assertFalse(dsv41_engine.fetch_due(self.dest, now))
        self.assertTrue(dsv41_engine.fetch_due(self.dest, now + 6 * 3600))
        dsv41_engine._write_check(self.dest, at=now - 600, ok=False)
        self.assertFalse(dsv41_engine.fetch_due(self.dest, now))
        self.assertTrue(dsv41_engine.fetch_due(self.dest, now + 1800))

    def test_a_failed_or_offline_fetch_keeps_the_installed_engine_and_the_next_start_does_not_try_again_at_once(self):
        out, sp = self.keep(FakeChild())                                              # (the child ends and installs nothing)
        self.assertEqual(out, self.old)
        self.assertEqual(self.exe_of(self.cfg), self.old)
        out, sp2 = self.keep()
        sp2.assert_not_called()                                                       # (_spawn_fetch wrote the attempt before it ran)
        self.assertEqual(out, self.old)

    def test_a_slow_download_does_not_hold_the_start_and_the_next_one_uses_the_new_engine(self):
        self.publish("0.1.49.1")
        out, _ = self.keep(FakeChild(hang=True))
        self.assertEqual(out, self.old)
        self.assertEqual(self.exe_of(self.cfg), self.old)
        self.assertIn("downloading in the background", " ".join(self.lines))
        self.upgrade()                                                                 # (the background fetch ends)
        out, sp = self.keep()
        sp.assert_not_called()
        self.assertEqual(self.exe_of(self.cfg), dsv41_engine.find(self.dest))
        self.assertIn("0.1.49.1", str(out))

    def test_a_folder_a_running_engine_uses_is_not_removed(self):
        self.publish("0.1.49.1")
        with mock.patch.object(dsv41_engine, "in_use", return_value=True):
            self.assertIsNotNone(self.upgrade())
        self.assertEqual(self.folders(), ["0.1.49", "0.1.49.1"])
        out, _ = self.keep(busy=True)
        self.assertEqual(self.folders(), ["0.1.49", "0.1.49.1"], "still in use: kept")
        out, _ = self.keep(busy=False)
        self.assertEqual(self.folders(), ["0.1.49.1"], "free now: removed")

    def test_a_config_whose_folder_was_removed_by_a_launch_is_pointed_at_the_installed_one(self):
        self.publish("0.1.49.1")
        self.assertIsNotNone(self.upgrade())                                           # `strata launch` did this; the app's configs still name 0.1.49
        self.assertFalse(self.old.exists())
        dsv41_engine._write_check(self.dest, at=time.time(), ok=True)
        out, sp = self.keep()
        sp.assert_not_called()
        self.assertEqual(self.exe_of(self.cfg), dsv41_engine.find(self.dest))
        self.assertEqual(self.exe_of(self.cfg2), dsv41_engine.find(self.dest))

    def test_an_engine_of_your_own_or_a_component_reference_is_left_alone(self):
        for exe in ("@component:strata-engine", str(self.root / "mine" / EXE)):
            c = self.root / "run" / "own.json"
            c.write_text(json.dumps({"exe": exe, "args": []}), encoding="utf-8")
            with mock.patch.object(dsv41_engine, "_spawn_fetch") as sp:
                self.assertIsNone(dsv41_engine.keep_current(c, source=self.url, log=self.lines.append))
            sp.assert_not_called()
            self.assertEqual(json.loads(c.read_text(encoding="utf-8"))["exe"], exe)
        with mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": str(self.old)}), mock.patch.object(dsv41_engine, "_spawn_fetch") as sp:
            self.assertIsNone(dsv41_engine.keep_current(self.cfg, source=self.url, log=self.lines.append))
        sp.assert_not_called()

    def test_nothing_in_the_check_can_fail_a_start(self):
        with mock.patch.object(dsv41_engine, "_spawn_fetch", side_effect=OSError("no process")):
            out = dsv41_engine.keep_current(self.cfg, source=self.url, log=self.lines.append, configs=[self.cfg2], wait=1)
        self.assertIsNone(out)
        self.assertIn("skipped", " ".join(self.lines))
        self.cfg.write_text("{not json", encoding="utf-8")
        self.assertIsNone(dsv41_engine.keep_current(self.cfg, source=self.url, log=self.lines.append))

    def test_a_package_of_another_backend_is_not_an_update_of_this_one(self):
        self.publish("0.1.49.1")
        with mock.patch.object(dsv41_engine, "kind_of", side_effect=lambda n: "other" if n.startswith("0.1.49.1") else "cuda"):
            with mock.patch.object(dsv41_engine, "_test_start", return_value=(True, "strata-dsv41 0.1.49.1")):
                self.assertIsNone(dsv41_engine.upgrade(self.dest, self.url, installed=self.old, hardware=self.hw, emit=self.lines.append, same_kind=True))
        self.assertEqual(self.folders(), ["0.1.49"])

    def test_two_fetches_do_not_run_at_once(self):
        self.publish("0.1.49.1")
        with dsv41_engine._fetch_lock(self.dest.parent) as first:
            self.assertTrue(first)
            self.assertIsNone(self.upgrade())                                          # a second caller gets None at once
        self.assertIsNotNone(self.upgrade())


for _name in dir(_up.Upgrade):                       # (the parent's own tests are not repeated here)
    if _name.startswith("test_") and _name not in KeepCurrent.__dict__:
        setattr(KeepCurrent, _name, None)


if __name__ == "__main__":
    unittest.main()
