"""Portable watcher identity survives process-state changes, but never PID reuse."""
import builtins
import signal
import unittest
from types import SimpleNamespace
from unittest import mock

from serve import lifeline


class WatcherToken(unittest.TestCase):
    birth = "Mon Oct 26 10:25:00 2026"
    pid = 424242

    def token(self, output):
        with mock.patch.object(lifeline.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            return lifeline._start_token(self.pid)

    def watcher(self, token, observations, *, released=False):
        # Run the actual standalone watcher with simulated pipe/process observations.
        # No process is launched, no OS signal is delivered, and no real handlers change.
        fake_os = SimpleNamespace(set_blocking=mock.Mock(), read=mock.Mock(return_value=b"x" if released else b""),
                                  kill=mock.Mock())
        modules = {
            "os": fake_os,
            "signal": SimpleNamespace(signal=mock.Mock(), SIGINT=signal.SIGINT, SIGHUP=1, SIG_IGN=signal.SIG_IGN,
                                      SIGTERM=signal.SIGTERM, SIGKILL=9),
            "subprocess": SimpleNamespace(run=mock.Mock(side_effect=[SimpleNamespace(stdout=o) for o in observations])),
            "sys": SimpleNamespace(argv=["watcher", str(self.pid), token], exit=mock.Mock(side_effect=SystemExit)),
            "time": SimpleNamespace(sleep=mock.Mock()),
        }
        namespace = {"__builtins__": {**vars(builtins), "__import__": lambda name, *a, **kw: modules[name]}}
        try:
            exec(lifeline._WATCHER, namespace)
        except SystemExit:
            pass
        return fake_os.kill

    def test_running_sleeping_and_state_flags_keep_the_same_identity(self):
        for state in ("R", "S", "Ss", "Sl", "R+", "D", "I"):
            with self.subTest(state=state):
                self.assertEqual(self.token(f"{self.birth} {state}\n"), "s:" + self.birth)

    def test_localized_start_time_keeps_its_text(self):
        birth = "火 10月 20 16:11:52 2026"
        self.assertEqual(self.token(f"{birth} S\n"), "s:" + birth)

    def test_zombies_and_missing_processes_have_no_identity(self):
        for output in ("", f"{self.birth} Z", f"{self.birth} Zs", f"{self.birth} Z+"):
            with self.subTest(output=output):
                self.assertEqual(self.token(output), "")

    def test_parent_death_still_terminates_a_child_that_started_running_then_slept(self):
        token = self.token(f"{self.birth} R\n")
        kill = self.watcher(token, [f"{self.birth} S\n", f"{self.birth} Z\n"])
        kill.assert_called_once_with(self.pid, signal.SIGTERM)

    def test_reused_pid_is_never_signalled(self):
        token = self.token(f"{self.birth} R\n")
        kill = self.watcher(token, ["Mon Oct 26 10:25:01 2026 S\n"])
        kill.assert_not_called()

    def test_released_child_is_never_signalled(self):
        token = self.token(f"{self.birth} R\n")
        kill = self.watcher(token, [], released=True)
        kill.assert_not_called()

    def test_existing_zombie_with_state_flags_is_never_signalled(self):
        token = self.token(f"{self.birth} S\n")
        kill = self.watcher(token, [f"{self.birth} Zs\n"])
        kill.assert_not_called()
