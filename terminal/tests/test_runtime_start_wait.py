"""Startup keeps its overall deadline while a busy owner misses a status reply."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_control import runtime
from strata_control.profiles import ProfileError


class StartupStatusWait(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.proc = mock.Mock()
        self.proc.poll.return_value = None
        self.enterContext(mock.patch.object(runtime.time, "monotonic", side_effect=lambda: self.now))
        self.enterContext(mock.patch.object(runtime.time, "sleep", side_effect=self.advance))

    def advance(self, seconds):
        self.now += seconds

    def test_live_owner_status_timeout_can_recover_inside_the_original_start_budget(self):
        ready = {"ok": True, "state": "ready", "owner_id": "synthetic-owner"}
        calls = []

        def reply(name, command, *, timeout=5):
            calls.append(timeout)
            if len(calls) == 1:
                self.advance(timeout)
                raise ProfileError("runtime operation wait expired; inspect runtime status", 8)
            return ready

        with mock.patch.object(runtime, "_rpc", side_effect=reply):
            result = runtime._wait_for_start("local", self.proc, 10)
        self.assertEqual(result, {**ready, "exit_code": 0})
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(self.now, 10)
        self.assertLessEqual(calls[1], 10 - 5)

    def test_repeated_status_timeouts_do_not_extend_the_start_deadline(self):
        calls = []

        def reply(name, command, *, timeout=5):
            calls.append(timeout)
            self.advance(timeout)
            raise ProfileError("runtime operation wait expired; inspect runtime status", 8)

        with mock.patch.object(runtime, "_rpc", side_effect=reply):
            result = runtime._wait_for_start("local", self.proc, 12)
        self.assertEqual(result["error"], "startup_wait_expired")
        self.assertEqual(result["exit_code"], 8)
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(self.now, 12)
        self.assertLess(calls[-1], 5)

    def test_exited_owner_timeout_is_not_retried(self):
        self.proc.poll.return_value = 1
        error = ProfileError("runtime operation wait expired; inspect runtime status", 8)
        with mock.patch.object(runtime, "_rpc", side_effect=error) as rpc:
            with self.assertRaises(ProfileError) as caught:
                runtime._wait_for_start("local", self.proc, 30)
        self.assertIs(caught.exception, error)
        self.assertEqual(rpc.call_count, 1)

    def test_non_timeout_errors_are_not_hidden_or_retried(self):
        error = ProfileError("invalid private control response", 6)
        with mock.patch.object(runtime, "_rpc", side_effect=error) as rpc:
            with self.assertRaises(ProfileError) as caught:
                runtime._wait_for_start("local", self.proc, 30)
        self.assertIs(caught.exception, error)
        self.assertEqual(rpc.call_count, 1)
        self.proc.poll.assert_not_called()

    def test_success_and_progress_callbacks_keep_the_existing_result(self):
        loading = {"ok": True, "state": "starting", "phase": "loading"}
        ready = {"ok": True, "state": "ready"}
        progress = []
        with mock.patch.object(runtime, "_rpc", side_effect=[loading, ready]):
            result = runtime._wait_for_start("local", self.proc, 10, progress.append)
        self.assertEqual(progress, [loading, ready])
        self.assertEqual(result, {**ready, "exit_code": 0})
