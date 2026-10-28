"""serve/test_web_context_ring.py - the Context card is one card: a KV ring (Cached | Reading | Generating | Free) with its legend, the native fill bars (Experts in VRAM / System RAM / GPU temperature) inside the same card below the legend — no second container, no .ctx-col (operator ruling 2026-10-28, option A: two stacked cards overflowed a 1080p screen and left a dead gap; the accepted single-card geometry is restored).

Reads the page source, as test_web_badges does: the ring's parts exist in the page, free is painted under the live arcs so its edge can never composite over a colored arc at the junctions,
the script cuts the arcs at shared quantized seams, the styles carry no colour literals of their own,
and no arc transitions its dash geometry — a re-interpolating dash transition desyncs adjacent arcs mid-animation and sparks the shared seams (operator report 2026-10-28, A/B-proven).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
PAGE = WEB / "index.html"
APP = WEB / "app.js"
CSS = WEB / "app.css"


class ContextRingMarkup(unittest.TestCase):
    def test_the_ring_and_its_legend_exist_in_the_page(self):
        page = PAGE.read_text(encoding="utf-8")
        for needle in ("class=\"kv-wrap\"", "class=\"kv-ring\"", "class=\"kv-center\"", 'id="kv-pct"', 'id="kv-sub"',
                       "class=\"kv-leg-head\"", "class=\"kv-legend\"", "<span>Cached (prefix)</span>", 'id="kv-leg-reuse-pct"',
                       'id="kv-leg-reuse"', 'id="kv-leg-prefill"', 'id="kv-leg-decode"', 'id="kv-leg-free"'):
            self.assertIn(needle, page, needle)

    def test_the_native_fill_card_keeps_its_progress_bars(self):
        page = PAGE.read_text(encoding="utf-8")
        for keep in ('id="ram-progress"', 'id="ram-bar"', 'id="slots-bar"', 'id="temp-bar"'):
            self.assertIn(keep, page, keep)

    def test_free_is_painted_first_under_the_live_arcs(self):
        page = PAGE.read_text(encoding="utf-8")
        arcs = re.findall(r'class="kv-seg kv-seg--(\w+)"', page)
        self.assertEqual(arcs, ["free", "reuse", "prefill", "decode"],
                         "the free arc's antialiased edge must never composite over a colored arc at the junctions")

    def test_the_ring_replaces_the_single_share_gauge(self):
        page = PAGE.read_text(encoding="utf-8")
        for gone in ("ctx-gauge", 'id="ctx-fill"', 'id="ctx-pct"', 'id="ctx-sub"'):
            self.assertNotIn(gone, page, gone)

    def test_the_context_card_stays_single_with_the_native_bars_below_the_legend(self):
        page = PAGE.read_text(encoding="utf-8")
        self.assertNotIn("ctx-col", page, "the two stacked containers overflowed 1080p — the column is gone")
        self.assertNotIn("Context usage", page, "one card: the ring and the fill live under one title")
        self.assertEqual(page.count('class="st-card ctx-card"'), 1,
                         "exactly one context card — the dead gap under the legend is filled by the native bars")
        title = page.index('<span class="card-title">Context fill</span>')
        self.assertLess(title, page.index('class="kv-wrap"'), "the ring opens the card")
        self.assertLess(page.index('id="kv-leg-free"'), page.index('id="slots-bar"'),
                        "the legend closes the donut block; the native bars follow it inside the same card")
        for bar in ('id="slots-bar"', 'id="ram-bar"', 'id="temp-bar"'):
            self.assertLess(page.index(bar), page.index('class="st-card req-card"'),
                            "the bars stay inside the context card, before the request card begins")


class ContextRingScript(unittest.TestCase):
    def test_the_monitor_draws_the_ring(self):
        src = APP.read_text(encoding="utf-8")
        monitor = src[src.index("function renderMonitor("):]
        self.assertIn("renderKvDonut(live, eng, last);", monitor)

    def test_arcs_are_cut_at_shared_quantized_seams_in_growth_order(self):
        src = APP.read_text(encoding="utf-8")
        loop = src[src.index("let edge = 0;"):]
        loop = loop[:loop.index("const usedPct")]
        self.assertEqual(re.findall(r'\["kv-seg-\w+", \w+\]', loop),
                         ['["kv-seg-reuse", segReuse]', '["kv-seg-prefill", segRead]',
                          '["kv-seg-decode", segGen]', '["kv-seg-free", free]'],
                         "clockwise growth from 12 o'clock: the free arc is always the trailing remainder")
        self.assertIn("const seam = Math.round(edge * 1000) / 1000;", loop)
        self.assertIn("const end = Math.round(edge * 1000) / 1000;", loop)
        self.assertIn('el.setAttribute("stroke-dashoffset", `${(-seam).toFixed(3)}`)', loop)

    def test_the_center_and_the_legend_quote_one_plain_formatting(self):
        src = APP.read_text(encoding="utf-8")
        ring = src[src.index("function renderKvDonut("):src.index("function renderMonitor(")]
        self.assertIn('$("kv-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";', ring,
                      "the total carries the page's own context format, as the removed gauge rendered it")
        for var in ("segReuse", "segRead", "segGen", "free"):
            cell = {"segReuse": "reuse", "segRead": "prefill", "segGen": "decode", "free": "free"}[var]
            self.assertIn(f'$("kv-leg-{cell}").textContent = kfmt({var})', ring)


class ContextRingStyle(unittest.TestCase):
    def test_the_ring_styles_carry_no_colour_literals_of_their_own(self):
        css = CSS.read_text(encoding="utf-8")
        block = css[css.index(".kv-wrap"):css.index(".bar-row")]
        self.assertIsNone(re.search(r"#[0-9a-fA-F]{3,8}", block), "every colour comes from the design system's tokens")
        for tok in ("var(--kv-cached)", "var(--st-info)", "var(--st-accent)", "var(--st-surface)", "var(--st-line)"):
            self.assertIn(tok, block, tok)

    def test_the_arcs_never_transition_their_dash_geometry(self):
        css = CSS.read_text(encoding="utf-8")
        rule = re.search(r"\.kv-seg \{[^}]*\}", css)
        self.assertIsNotNone(rule, ".kv-seg base rule")
        self.assertNotIn("transition", rule.group(0),
                         "four adjacent arcs share quantized seams; a dash transition re-interpolates each arc independently and opens the junctions mid-animation (measured 4.4-9.5 units with it, zero without)")
        self.assertIn("stroke-linecap: butt", rule.group(0))

    def test_the_cached_and_legend_tokens_exist_in_the_theme_blocks(self):
        css = CSS.read_text(encoding="utf-8")
        for theme in ("dark", "light"):
            block = re.search(rf'\[data-theme="{theme}"\]\s*\{{[^}}]*\}}', css)
            self.assertIsNotNone(block, f"[data-theme=\"{theme}\"] block")
            self.assertIn("--kv-cached: var(--st-line);", block.group(0))   # resolves per theme through --st-line itself
            self.assertIn("--dim-legend:", block.group(0))

    def test_the_metrics_grid_keeps_the_upstream_auto_fit(self):
        css = CSS.read_text(encoding="utf-8")
        self.assertIn(".metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));", css,
                      "#3176's auto-fit row is not re-flowed by this PR")


if __name__ == "__main__":
    unittest.main()
