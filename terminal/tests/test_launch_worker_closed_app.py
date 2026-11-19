"""A worker posts a request-local RunOutcome even after the real Textual app has closed.

Closing the UI must not update its removed screen, leak a coroutine or raise in the worker.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

from strata_cli.launch.app import LaunchApp, RunOutcome
from strata_cli.launch.runner import Cancelled, LaunchError
from tests.terminal.launch_support import Home
from tests.terminal.test_launch_tui import make_ctx


class WorkerAfterQuit(unittest.TestCase):
    def setUp(self):
        home = Home().__enter__()
        self.addCleanup(home.__exit__, None, None, None)

    def check(self, run, *, result=None, error=None, cancelled=False):
        app = LaunchApp(make_ctx())

        async def open_and_close():
            async with app.run_test() as pilot:
                await pilot.pause()

        asyncio.run(open_and_close())
        self.assertTrue(app._closed)
        self.assertTrue(app._loop.is_closed())
        app.run_screen = mock.Mock()
        app.runner = SimpleNamespace(run=run)
        failures = []

        def finish():
            try:
                app._work()
            except Exception as exc:
                failures.append(type(exc).__name__)

        with warnings.catch_warnings(record=True) as seen, mock.patch.object(app, "post_message", wraps=app.post_message) as post:
            warnings.simplefilter("always")
            worker = threading.Thread(target=finish)
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            gc.collect()
        self.assertEqual(failures, [])
        post.assert_called_once()
        outcome = post.call_args.args[0]
        self.assertIsInstance(outcome, RunOutcome)
        self.assertEqual((outcome.result, outcome.error, outcome.cancelled), (result, error, cancelled))
        self.assertIsNone(app.result)
        app.run_screen.on_finished.assert_not_called()
        app.run_screen.on_failure.assert_not_called()
        self.assertEqual([str(w.message) for w in seen if "never awaited" in str(w.message)], [])

    def test_finished(self):
        self.check(lambda: "result", result="result")

    def test_cancelled(self):
        def run():
            raise Cancelled()
        self.check(run, cancelled=True)

    def test_launch_error(self):
        def run():
            raise LaunchError("could not start", "try again")
        self.check(run, error="could not start try again")

    def test_unexpected_error(self):
        def run():
            raise KeyError("x")
        self.check(run, error="Something unexpected went wrong: KeyError: 'x'")

    def test_a_live_app_handles_the_message_on_its_event_loop(self):
        app = LaunchApp(make_ctx())
        app.runner = SimpleNamespace(run=lambda: "result")
        app.run_screen = mock.Mock()
        app._save_pending = mock.Mock()

        async def go():
            async with app.run_test() as pilot:
                app.running = True
                await asyncio.to_thread(app._work)
                await pilot.pause()
                self.assertFalse(app.running)
                self.assertEqual(app.result, "result")
                app.run_screen.on_finished.assert_called_once()
                app._save_pending.assert_called_once()

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
