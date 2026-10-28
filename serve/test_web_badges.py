"""serve/test_web_badges.py - the Recent requests table has a badge for every way a request can end (#3991),
and a gate badge for the cvec auto domain.

A request that waits past the queue limit ends with finish "timeout"; the Status cell showed the raw word. Reads the page source.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"
SERVER = Path(__file__).resolve().parent / "server.py"


class FinishBadges(unittest.TestCase):
    def test_a_timed_out_request_has_a_badge(self):
        src = APP.read_text(encoding="utf-8")
        table = src[src.index("const badge = {stop:"):]
        table = table[:table.index("};")]
        self.assertRegex(table, r'timeout: \["st-badge--queued", "Timeout"\]')

    def test_every_finish_the_server_stores_has_a_badge(self):
        src = APP.read_text(encoding="utf-8")
        table = src[src.index("const badge = {stop:"):]
        table = table[:table.index("};")]
        for finish in ("stop", "length", "cancel", "disconnect", "error", "timeout"):
            self.assertRegex(table, rf"\b{finish}:", finish)
        self.assertIn('"timeout"', SERVER.read_text(encoding="utf-8"))


class GateBadges(unittest.TestCase):
    """The cvec auto gate badge (the Recent requests table): an auto request the gate fired
    shows the green ESP; one that stayed stock (or was never measured) shows a neutral auto;
    bool requests keep the plain on/off badge."""

    def _auto_branch(self):
        src = APP.read_text(encoding="utf-8")
        start = src.index('r.projection === "auto"')
        return src[start:src.index("const hit =")]

    def test_a_steered_auto_request_shows_a_green_esp(self):
        seg = self._auto_branch()
        steer = seg[: seg.index('class="st-badge" title=')]
        self.assertIn('r.gate === "steer"', steer)
        self.assertIn("st-badge--reading", steer)
        self.assertIn(">ESP<", steer)

    def test_an_unsteered_auto_request_shows_a_neutral_auto(self):
        seg = self._auto_branch()
        neutral = seg[seg.index('class="st-badge" title='):]
        self.assertIn('r.gate === "stock"', neutral)
        self.assertIn(">auto<", neutral)


if __name__ == "__main__":
    unittest.main()
