"""serve/test_web_live_row.py - a running request appears in the table while the engine works on it: one synthetic row on top with live streaming counters, replaced by the real DONE row the moment the history entry is written.

Reads the page source, as test_web_badges does; the row builder itself is extracted from app.js and run in node (test_web_live_row.js).
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
APP = WEB / "app.js"


class LiveRequestRow(unittest.TestCase):
    def src(self):
        return APP.read_text(encoding="utf-8")

    def test_the_row_is_built_from_the_fsm_and_the_servers_own_clock(self):
        src = self.src()
        self.assertIn("function liveRequestRow(live, fsm, serverTime) {", src)
        self.assertIn("const start = ((serverTime || Date.now() / 1000) - (live.elapsed_s || 0)) * 1000;", src,
                      "the start time is the server's sample instant minus the elapsed time the server reported")
        self.assertIn('const reading = fsm.isReading;', src)

    def test_the_row_is_prepended_and_the_empty_guard_counts_it(self):
        src = self.src()
        self.assertIn('const liveRow = (fsm.isReading || fsm.isGenerating) ? liveRequestRow(live, fsm, serverTime) : "";', src)
        self.assertIn("if (!requests.length && !liveRow) {", src,
                      "a working engine with no finished requests yet must not show \"No requests yet\"")
        self.assertIn("body.innerHTML = liveRow + requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {", src)
        self.assertIn("function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept, serverTime) {", src)
        self.assertIn("renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept, m.time);", src,
                      "the /metrics payload carries \"time\": the row is timed from the server, not from browser memory")

    def test_phase_words_follow_the_engines_own_vocabulary(self):
        src = self.src()
        for needle in ('live.phase.startsWith("writing a tool call") ? "Writing"',
                       'live.phase.startsWith("tool call complete") ? "Running"',
                       '({thinking: "Thinking", answering: "Answering"})[live.phase] || "Generating"',
                       'const text = reading ? "Reading…" : word + "…";',
                       'const cls = reading ? "st-badge--reading" : "st-badge--generating"'):
            self.assertIn(needle, src, needle)

    def test_an_unreported_value_stays_a_dash_never_a_fake_zero(self):
        src = self.src()
        for needle in ('const reused = live.reused > 0 ? fmt(live.reused) : "–";',
                       'const prefill = live.prefill_tok_s_mean != null ? fmt(live.prefill_tok_s_mean) : "–";',
                       'const decode = !reading && live.tok_s != null ? fmt(live.tok_s, 1) : "–";',
                       'const dur = live.elapsed_s == null ? "–" : `${fmt(live.elapsed_s, 1)} s`;',
                       '<td class="num">${prefill}</td><td class="num">${decode}</td><td class="num">–</td><td class="num">${dur}</td></tr>'):
            self.assertIn(needle, src, needle)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_row_renders_both_phases_in_node(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_live_row.js"))],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
