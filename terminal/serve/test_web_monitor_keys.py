"""The Monitor tab shows the "api_keys" gate (#5004): the card names each configured key with what it holds and
the ceiling that holds it, and hides entirely without the block.  serve/web/app.js is run in node with stand-ins
for the page's helpers (no browser), as in test_web_monitor_notes.py."""
from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")

STUBS = """
const els = {};
const $ = (id) => (els[id] ||= {id, textContent: "", innerHTML: "", hidden: false});
const esc = (s) => String(s);
const icon = () => "";
"""


def function_source(text: str, name: str) -> str:
    start = text.index(f"function {name}(")
    return text[start:text.index("\n}\n", start) + 3]


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class KeyLimitsCard(unittest.TestCase):
    def render(self, limits):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        script = (STUBS + function_source(app, "facts") + function_source(app, "renderKeyLimits") +
                  f"renderKeyLimits({json.dumps(limits)});\n"
                  "console.log(JSON.stringify({hidden: $('kl-card').hidden, sum: $('kl-sum').textContent,"
                  " facts: $('kl-facts').innerHTML, note: $('kl-note').textContent}));")
        done = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8",
                              timeout=60, check=True)
        return json.loads(done.stdout)

    SNAP = {"slots": 2,
            "keys": [{"name": "hermes", "inflight": 1, "max_concurrent": 1, "reserved_capacity": 1},
                     {"name": "home assistant", "inflight": 0, "max_concurrent": 1, "reserved_capacity": 0}],
            "unlisted": {"inflight": 1, "max": 1}}

    def test_no_block_no_card(self):
        out = self.render(None)
        self.assertTrue(out["hidden"])
        self.assertEqual(out["facts"], "")

    def test_named_keys_held_now_and_the_shared_bucket(self):
        out = self.render(self.SNAP)
        self.assertFalse(out["hidden"])
        self.assertEqual(out["sum"], "2 slots on the server, 2 held now")
        self.assertIn("<dt>hermes</dt><dd>1 of 1 · 1 reserved</dd>", out["facts"])
        self.assertIn("<dt>home assistant</dt><dd>0 of 1</dd>", out["facts"])
        self.assertIn("<dt>(other keys)</dt><dd>1 of 1</dd>", out["facts"],
                      "the bucket of unnamed keys is a row of its own, with the reach the reservations leave")

    def test_a_max_beyond_its_reach_shows_the_ceiling_that_holds(self):
        snap = {"slots": 2,
                "keys": [{"name": "busy", "inflight": 1, "max_concurrent": 3, "reserved_capacity": 0},
                         {"name": "mine", "inflight": 0, "max_concurrent": None, "reserved_capacity": 1}],
                "unlisted": {"inflight": 0, "max": 1}}
        facts = self.render(snap)["facts"]
        self.assertIn("<dt>busy</dt><dd>1 of 1 · capped by the other keys' reservations</dd>", facts,
                      "the ceiling shown is the reach, as at the server's start")
        self.assertIn("<dt>mine</dt><dd>0 of 2 · 1 reserved</dd>", facts,
                      "a reservation is not a ceiling: its own key reaches every slot no other key reserves")

    def test_the_note_points_at_the_config_not_at_a_key(self):
        out = self.render(self.SNAP)
        self.assertIn("api_keys", out["note"])
        self.assertNotIn("hermes-key", out["note"] + out["facts"], "the card shows names, never keys")


if __name__ == "__main__":
    unittest.main()
