"""serve/test_web_kv_phase.py - the phase machinery: one FSM descriptor in the monitor, the kvLamp legend lamp, ksplit token math, arc hover titles, the resume latch.

Reads the page source, as test_web_badges does; the lamp line and the token math are cut out of app.js verbatim and executed in node (test_web_kv_phase.js).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
APP = WEB / "app.js"
CSS = WEB / "app.css"


def ring_of(src: str) -> str:
    return src[src.index("function renderKvDonut("):src.index("function renderMonitor(")]


class PhaseScript(unittest.TestCase):
    def test_rendermonitor_builds_one_fsm_descriptor(self):
        src = APP.read_text(encoding="utf-8")
        monitor = src[src.index("function renderMonitor("):]
        fsm = monitor[:monitor.index("// model state")]
        self.assertIn("const fsm = {", fsm)
        for line in ('isIdle:       live.state === "idle" && !!last,',
                     'isReading:    live.state === "reading",',
                     'isGenerating: live.state === "generating",',
                     'isQueued:     live.state === "queued" || (live.queued > 0),',
                     'state:        live.state || "idle"'):
            self.assertIn(line, fsm, line)
        self.assertIn("renderKvDonut(fsm, live, eng, last, requests);", monitor)

    def test_the_drawing_obeys_the_fsm_not_live_state(self):
        src = APP.read_text(encoding="utf-8")
        ring = ring_of(src)
        for gone in ('live.state === "reading"', 'live.state === "generating"', 'live.state === "idle"'):
            self.assertNotIn(gone, ring, "the donut answers to fsm.isReading / isGenerating / isIdle, never live.state")
        for keep in ("fsm.isReading", "fsm.isGenerating", "fsm.isIdle"):
            self.assertIn(keep, ring, keep)

    def test_arcs_carry_hover_titles_with_the_exact_counts(self):
        src = APP.read_text(encoding="utf-8")
        ring = ring_of(src)
        for call in ('setSvgTitle($("kv-seg-reuse"), ctx ? `Cached (prefix): ${segReuse.toLocaleString()} tokens',
                     'setSvgTitle($("kv-seg-prefill"), ctx ? `Reading (prefill): ${segRead.toLocaleString()} tokens`',
                     'setSvgTitle($("kv-seg-decode"), ctx ? `Generating (decode): ${segGen.toLocaleString()} tokens`',
                     'setSvgTitle($("kv-seg-free"), ctx ? `Free headroom: ${free.toLocaleString()} tokens`'):
            self.assertIn(call, ring, call)
        self.assertIn('t = document.createElementNS("http://www.w3.org/2000/svg", "title");', src,
                      "the title is created in the SVG namespace")

    def test_the_cached_row_keeps_one_fixed_name_and_one_honest_percent(self):
        src = APP.read_text(encoding="utf-8")
        ring = ring_of(src)
        self.assertNotIn("kv-leg-reuse-label", ring,
                         "the REUSE row keeps a fixed static name in the page; the donut never rewrites it")
        self.assertIn('const effPct = (segRead === 0 && eff >= 1) ? "100%"', ring,
                      "100 % only for a prefix that covers the whole prompt, never before any read progress")
        self.assertNotIn('(segRead === 0 && segReuse > 0)', ring,
                         "B-1: a cached prefix reported before read progress must not read as a full hit")
        self.assertIn('const cachedHit = requests.find(r => (r.reused || 0) > 0 && (r.prompt_tokens || 0) > 0);', ring,
                      "the quiet percent quotes the newest history record that really got a cached prefix")
        self.assertIn('$("kv-leg-reuse-pct").textContent = segReuse > 0 ? (merged ? quietPct : effPct) : "";', ring)
        self.assertIn("if (!merged && segReuse > 0) lastReusePct = effPct;", ring)
        self.assertIn('if (segReuse === 0) lastReusePct = "";', ring)

    def test_the_resume_flash_latches_on_the_rising_edge_of_reused(self):
        src = APP.read_text(encoding="utf-8")
        self.assertIn("let resumeWasAnnounced = false, resumeFlashUntil = 0;", src)
        self.assertIn("if (resumedVal > 0 && !resumeWasAnnounced) resumeFlashUntil = Date.now() + 1500;", src)
        self.assertIn("resumeWasAnnounced = resumedVal > 0;", src)

    def test_the_legend_rows_dim_to_one_lamp(self):
        src = APP.read_text(encoding="utf-8")
        ring = ring_of(src)
        self.assertIn("const lamp = kvLamp(fsm, segReuse > 0, !!live.prompt_read, resumeFlash);", ring)
        self.assertIn('for (const k of ["reuse", "prefill", "decode", "free"]) $(`kv-leg-${k}`).closest("li").classList.toggle("dim", k !== lamp);', ring)
        self.assertIn('.kv-legend li.dim span:not(.kv-key), .kv-leg-head .kv-col {', CSS.read_text(encoding="utf-8"),
                      "the dim style already belongs to the ring's own rules")

    def test_the_cached_label_carries_no_colour_rule_of_its_own(self):
        css = CSS.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"#kv-leg-reuse-label\b", css),
                          "the Cached row dims with its row, so it must not get its own colour rule")


class KvLampTruthTable(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_lamp_and_the_token_math_run_verbatim_in_node(self):
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("test_web_kv_phase.js"))],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
