"""serve/test_web_speed_card.py - the speed card and the palette: two counters under the FSM (sv boxes, lit only by their own phase, dimmed by colour never opacity), the totals line naming Prefill and Decode as its own columns, the WCAG-solved scoping blocks.

Reads the page source, as test_web_badges does.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
APP = WEB / "app.js"
CSS = WEB / "app.css"


def monitor_of(src: str) -> str:
    return src[src.index("function renderMonitor("):]


def totals_of(src: str) -> str:
    return src[src.index("function renderTotals("):src.index("function liveRequestRow")]


class SpeedCardScript(unittest.TestCase):
    def src(self):
        return APP.read_text(encoding="utf-8")

    def test_each_counter_lives_in_its_own_sv_box(self):
        src = self.src()
        for needle in ('<div class="sv" id="sv-speed"><span class="st-metric__value" id="mv-speed">',
                       '<div class="sv" id="sv-prefill"><span class="st-metric__value" id="mv-prefill">'):
            self.assertIn(needle, src, needle)

    def test_the_speed_and_prefill_conditions_are_fsm_conditions(self):
        monitor = monitor_of(self.src())
        for line in ('const speed = fsm.isGenerating ? live.tok_s : last ? last.decode_tok_s : null;',
                     '"t/s", fsm.isGenerating ? "Generating now" : last ? "Last decode" : "Decode");',
                     'const prefill = (fsm.isReading || fsm.isGenerating) ? live.prefill_tok_s_mean',
                     'fsm.isReading ? "Reading now" : fsm.isGenerating ? "Prefill this request" : last ? "Last prefill" : "Prefill");'):
            self.assertIn(line, monitor, line)
        for gone in ('live.state === "generating" ? live.tok_s', 'live.state !== "idle" ? live.prefill_tok_s_mean',
                     '"Decode now"', '"Prefill now"', '"Decode last request"', '"Prefill last request"'):
            self.assertNotIn(gone, monitor, "the counters answer to fsm, never to live.state on its own: " + gone)

    def test_the_phase_classes_toggle_with_the_fsm(self):
        monitor = monitor_of(self.src())
        for line in ('$("sv-speed").classList.toggle("sv--generating", fsm.isGenerating);',
                     '$("sv-prefill").classList.toggle("sv--reading", fsm.isReading);',
                     '$("sv-speed").classList.toggle("dim", !fsm.isGenerating);',
                     '$("sv-prefill").classList.toggle("dim", !fsm.isReading);'):
            self.assertIn(line, monitor, line)

    def test_the_totals_line_names_each_speed_as_its_own(self):
        src = self.src()
        self.assertIn('} Prefill t/s` : "";', src)
        self.assertIn('} Decode t/s` : "";', src)
        self.assertNotIn("tok/s", totals_of(src), "a totals line must not carry an undivided tok/s")


class SpeedCardStyle(unittest.TestCase):
    def css(self):
        return CSS.read_text(encoding="utf-8")

    def test_inactive_pills_are_transparent_on_the_card_not_faded(self):
        css = self.css()
        self.assertIn('.badges .st-badge:not(.on) { background: transparent; border-color: var(--st-line); }', css)
        self.assertNotIn(".st-badge { opacity:", css, "38 % opacity on an inactive pill is unreadable text")
        self.assertNotIn(".st-badge.on { opacity: 1; }", css)

    def test_dim_counters_keep_their_hue_and_never_mix_to_white(self):
        css = self.css()
        block = css[css.index(".speed-values {"):css.index('.st-metric__spark[data-tone="warn"]')]
        for line in ('#sv-speed.dim .st-metric__value { color: color-mix(in srgb, var(--st-accent-text) var(--dim-accent-value), var(--st-surface)); }',
                     '#sv-prefill.dim .st-metric__value { color: color-mix(in srgb, var(--st-info-text) var(--dim-info-value), var(--st-surface)); }',
                     '#sv-speed.dim .st-metric__sub { color: color-mix(in srgb, var(--st-accent-text) var(--dim-accent-caption), var(--dim-caption-partner)); }',
                     '#sv-prefill.dim .st-metric__sub { color: color-mix(in srgb, var(--st-info-text) var(--dim-info-caption), var(--st-surface)); }',
                     '.sv--reading .st-metric__value { color: var(--st-info-text); }',
                     '.sv--generating .st-metric__value { color: var(--st-accent-text); }'):
            self.assertIn(line, block, line)
        self.assertNotIn("opacity:", block, "dimming here is colour work, not opacity")
        self.assertIn('.st-metric__spark .speed-prefill { color: var(--st-info); }', css,
                      "the Reading spark line is blue, the Decode line keeps the card's green")


class PaletteScoping(unittest.TestCase):
    def theme_block(self, theme):
        m = re.search(rf'\[data-theme="{theme}"\]\s*\{{.*?\n\}}', CSS.read_text(encoding="utf-8"), re.DOTALL)
        self.assertIsNotNone(m, f'[data-theme="{theme}"] block')
        return m.group(0)

    def test_the_dark_theme_ships_the_solved_accent_and_per_family_keeps(self):
        dark = self.theme_block("dark")
        self.assertIn("--st-accent: #6AD09D;", dark)
        self.assertIn("--st-accent-text: #6AD09D;", dark)
        for tok in ("--dim-accent-value", "--dim-accent-caption", "--dim-info-value",
                    "--dim-info-caption", "--dim-unit", "--dim-legend"):
            self.assertIn(tok + ":", dark, tok)
        self.assertIn("--dim-caption-partner: var(--st-surface);", dark)

    def test_the_light_theme_solves_only_dim_and_untouches_the_kit_tokens(self):
        light = self.theme_block("light")
        for tok in ("--st-accent", "--st-info"):
            self.assertNotIn(tok, light, "the light theme keeps the kit's own colour tokens")
        for tok in ("--dim-accent-value", "--dim-accent-caption", "--dim-info-value",
                    "--dim-info-caption", "--dim-unit", "--dim-legend"):
            self.assertIn(tok + ":", light, tok)
        self.assertIn("--dim-caption-partner: var(--st-ink);", light,
                      "the accepted colour decision: the light green caption mixes toward --st-ink")

    def test_inactive_state_pills_carry_calm_dark_literals(self):
        css = CSS.read_text(encoding="utf-8")
        for state, colour in (("idle", "#7E868D"), ("reading", "#6C8AA8"), ("queued", "#98875A"),
                             ("error", "#A67B82"), ("generating", "#5D8D75")):
            self.assertIn(f'[data-theme="dark"] .badges .st-badge:not(.on)[data-s="{state}"] {{ color: {colour}; }}', css,
                          f"the {state} pill keeps a readable text in its own hue")


if __name__ == "__main__":
    unittest.main()
