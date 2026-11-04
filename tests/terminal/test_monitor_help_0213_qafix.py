"""F13 (QATUI 0213): `?` in the monitor shows the key list on the Overview as on the other views, and it stays up past the next poll."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_overview import Source, metrics  # noqa: E402
from strata_cli.monitor import MonitorApp  # noqa: E402

KEYS = "Space freeze  r refresh  t theme  q detach"


def last_line(app) -> str:
    return [s.text for s in app.screen._compositor.render_strips()][-1]


class Help(unittest.TestCase):
    def _run(self, view_key: str):
        async def go():
            # a one-second poll in real use; 0.2 s here so several polls pass while the help is up
            app = MonitorApp(Source(metrics("generating")), None, interval=0.2)
            async with app.run_test(size=(140, 36)) as pilot:
                await pilot.pause(0.5)
                await pilot.press(view_key)
                await pilot.pause(0.3)
                self.assertNotIn("Space freeze  r refresh", last_line(app))
                await pilot.press("question_mark")
                await pilot.pause(1.0)               # several polls later
                self.assertIn(KEYS, last_line(app))
                await pilot.press("question_mark")   # again: closes it
                await pilot.pause(0.3)
                self.assertNotIn("Space freeze  r refresh", last_line(app))
                await pilot.press("question_mark")
                await pilot.pause(0.3)
                self.assertIn(KEYS, last_line(app))
                await pilot.press("escape")
                await pilot.pause(0.3)
                self.assertNotIn("Space freeze  r refresh", last_line(app))
        asyncio.run(go())

    def test_overview(self):
        self._run("1")

    def test_serving_view(self):
        self._run("2")

    def test_changing_view_closes_it(self):
        async def go():
            app = MonitorApp(Source(metrics("generating")), None, interval=0.2)
            async with app.run_test(size=(140, 36)) as pilot:
                await pilot.pause(0.5)
                await pilot.press("question_mark")
                await pilot.pause(0.3)
                self.assertIn(KEYS, last_line(app))
                await pilot.press("3")
                await pilot.pause(0.3)
                self.assertNotIn("Space freeze  r refresh", last_line(app))
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
