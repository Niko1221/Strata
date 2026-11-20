"""Tests for ab_report: the HTML report and the PNG card.  No network, no browser."""
import copy
import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_report  # noqa: E402
from ab_report import build_html, card_pair, render_png, suite_sections, write_report  # noqa: E402

REQS = ["chat0", "chat1", "chat2", "long"]


def arm(version, decode, prompt, setup=120.0, start=(20.0, 22.0)):
    rows = [{"round": r, "req": q, "decode": decode[i] + r * 0.1, "prompt": prompt[i] + r, "prompt_n": 40 if q != "long" else 9000,
             "n": 256} for r in range(1, 4) for i, q in enumerate(REQS)]
    return {"version": version, "status": "ran", "reason": "", "setup_s": setup, "start_s": list(start), "rows": rows}


RESULTS = {
    "created": "2026-11-16T12:00:00", "model": "Qwen3.6-35B-A3B <UD-Q4_K_XL> & co", "gpu": "NVIDIA GeForce RTX 3090",
    "tune": "quick", "rounds": 3, "requests": REQS,
    "arms": [arm("0.1.55", [100, 102, 98, 90], [400, 410, 405, 1500], setup=150.0, start=(25.0, 24.0)),
             arm("0.1.56", [118, 120, 110, 99], [420, 430, 425, 1800], setup=130.0, start=(21.0, 20.0))],
    "untouched": True, "changed": [],
}


def card(h):
    return h[h.index('id="card"'):h.index("</section>")]


class CardPair(unittest.TestCase):
    def test_card_compares_oldest_and_newest_that_ran(self):
        r = copy.deepcopy(RESULTS)
        r["arms"] = [{"version": "0.1.54", "status": "skipped", "reason": "no package", "setup_s": None, "start_s": [], "rows": []},
                     arm("0.1.57", [1] * 4, [1] * 4), arm("0.1.55", [1] * 4, [1] * 4), arm("0.1.56", [1] * 4, [1] * 4)]
        old, new = card_pair(r)
        self.assertEqual((old["version"], new["version"]), ("0.1.55", "0.1.57"))

    def test_card_none_with_one_version(self):
        r = copy.deepcopy(RESULTS)
        r["arms"] = r["arms"][:1]
        self.assertIsNone(card_pair(r))


class CardMedian(unittest.TestCase):
    def test_card_median_skips_round_0_when_there_are_more_rounds(self):
        rows = [{"round": i, "req": "chat0", "decode": d} for i, d in enumerate((10.0, 50.0, 52.0))]
        self.assertEqual(ab_report._median({"rows": rows}, "chat0", "decode"), 51.0)
        self.assertEqual(ab_report._median({"rows": rows[:1]}, "chat0", "decode"), 10.0)


class Html(unittest.TestCase):
    def test_html_has_title_rows_and_no_external_urls(self):
        h = build_html(RESULTS)
        self.assertIn("Strata 0.1.55 → 0.1.56", h)
        self.assertEqual(h.count('<tr data-version='), 8)  # 2 versions x 4 requests
        self.assertNotRegex(h, r'(src|href)="https?://')
        self.assertIn("github.com/Niko1221/Strata", h)
        self.assertIn("&lt;UD-Q4_K_XL&gt; &amp; co", h)
        self.assertNotIn("<UD-Q4_K_XL>", h)

    def test_skipped_version_appears_as_chip_and_row(self):
        r = copy.deepcopy(RESULTS)
        r["arms"].insert(0, {"version": "0.1.54", "status": "skipped", "reason": "served engine 0.1.53, expected 0.1.54",
                             "setup_s": None, "start_s": [], "rows": []})
        h = build_html(r)
        self.assertIn("0.1.54 did not run", card(h))
        self.assertIn('<tr data-version="0.1.54"', h)
        self.assertIn("served engine 0.1.53, expected 0.1.54", h)
        self.assertIn("Strata 0.1.55 → 0.1.56", re.sub(r"<[^>]+>", "", h))

    def test_bar_widths_are_proportional(self):
        r = copy.deepcopy(RESULTS)
        r["arms"] = [arm("0.1.55", [60, 30, 0, 50], [1] * 4), arm("0.1.56", [120, 110, 90, 100], [1] * 4)]
        w = [float(x) for x in re.findall(r"width:calc\(\(100% - 80px\) \* ([0-9.]+)\)", card(build_html(r)))]
        self.assertEqual(len(w), 8)  # old, new per request
        top = 120.2  # largest median (120 + round-2 offset 0.2)
        for got, want in zip(w, [60.2, 120.2, 30.2, 110.2, 0.2, 90.2, 50.2, 100.2]):
            self.assertAlmostEqual(got, want / top, places=3)
        self.assertNotEqual(w[1], w[3])  # 120 vs 110 differ
        self.assertNotIn("max-width:calc(100% - 80px)", build_html(r))

    def test_subtitle_without_gpu_has_no_leading_dot(self):
        r = copy.deepcopy(RESULTS)
        r["gpu"] = ""
        self.assertIn('<p class="sub">Qwen3.6-35B-A3B', card(build_html(r)))

    def test_isolation_result_under_the_table_not_on_the_card(self):
        h = build_html(RESULTS)
        self.assertIn("untouched", h[h.index("</table>"):])
        self.assertNotIn("untouched", card(h))
        r = dict(RESULTS, untouched=False, changed=["~C:/Users/x/AppData/Local/Strata/<a>.json"])
        h = build_html(r)
        self.assertIn("<code>~C:/Users/x/AppData/Local/Strata/&lt;a&gt;.json</code>", h[h.index("</table>"):])
        self.assertNotIn("AppData", card(h))

    def test_regression_shown_negative(self):
        r = copy.deepcopy(RESULTS)
        r["arms"][1] = arm("0.1.56", [90, 95, 88, 80], [300, 310, 305, 1200])
        c = card(build_html(r))
        self.assertRegex(c, r'class="big">-\d+%')
        self.assertRegex(c, r'-\d+%</div>')  # long prompt tile


class Png(unittest.TestCase):
    def test_no_browser_writes_html_only(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ab_report, "find_browser", return_value=None):
            html, png = write_report(RESULTS, Path(d))
            self.assertTrue(html.is_file())
            self.assertIsNone(png)

    def test_render_png_uses_list_args_and_card_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            html, png = Path(d) / "report.html", Path(d) / "report.png"
            html.write_text("x", encoding="utf-8")

            def fake(cmd, **kw):
                self.assertIsInstance(cmd, list)
                self.assertNotIn("shell", kw)
                self.assertIn("timeout", kw)
                self.assertTrue(cmd[-1].endswith("#card") and cmd[-1].startswith("file:"))
                self.assertIn(f"--screenshot={png}", cmd)
                profile = [c for c in cmd if c.startswith("--user-data-dir=")]
                self.assertEqual(len(profile), 1)
                self.assertTrue(Path(profile[0].split("=", 1)[1]).is_dir())
                png.write_bytes(b"png")
                return mock.Mock(returncode=0)
            with mock.patch.object(ab_report.subprocess, "run", side_effect=fake):
                self.assertTrue(render_png("edge", html, png))

    def test_render_png_failure_returns_false(self):
        with tempfile.TemporaryDirectory() as d:
            html, png = Path(d) / "report.html", Path(d) / "report.png"
            with mock.patch.object(ab_report.subprocess, "run", side_effect=OSError("nope")):
                self.assertFalse(render_png("edge", html, png))
            with mock.patch.object(ab_report.subprocess, "run",
                                   side_effect=ab_report.subprocess.TimeoutExpired("edge", 60)):
                self.assertFalse(render_png("edge", html, png))


def srow(suite, req, rnd, ok, decode=50.0, prompt=1000.0, prompt_n=8000, predicted_n=300, wall_s=12.0):
    return {"round": rnd, "req": req, "suite": suite, "decode": decode, "prompt": prompt, "prompt_n": prompt_n, "n": 256,
            "ok": ok, "check": "x", "predicted_n": predicted_n, "wall_s": wall_s}


def suite_arm(version, rows, n_ctx=262144, **kw):
    a = arm(version, [100, 102, 98, 90], [400, 410, 405, 1500])
    a.update(n_ctx=n_ctx, suites_status="ran", suites_reason="", left_out=[])
    a["rows"] += rows
    a.update(kw)
    return a


def suite_results(suites, arms, context=262144):
    r = copy.deepcopy(RESULTS)
    r.update(suites=suites, context=context, corpus="v0.2.12", run_code=False, arms=arms)
    return r


class Suites(unittest.TestCase):
    def test_no_suites_no_section(self):
        self.assertEqual(suite_sections({"suites": []}), "")
        self.assertEqual(suite_sections({}), "")
        # sha256 of build_html(RESULTS) before the suites existed
        self.assertEqual(hashlib.sha256(build_html(RESULTS).encode()).hexdigest(),
                         "8dfef416c9f494961cf68cbb5ed6b7f8ec367435380f64f950d099a4f39ddb63")
        self.assertEqual(build_html(dict(RESULTS, suites=[], context=None, corpus=None, run_code=False)), build_html(RESULTS))

    def test_ladder_section_has_charts_and_counts(self):
        rows = [srow("ladder", f"ladder.{k}k", 1, ok, prompt_n=k * 1000) for k, ok in ((8, True), (16, True), (32, True), (64, False))]
        r = suite_results(["ladder"], [suite_arm("0.1.59", rows), suite_arm("0.1.60", [])])
        h = build_html(r)
        self.assertEqual(h.count("<svg"), 2)
        self.assertIn("richtig 3/4", h)
        self.assertIn('id="ladder"', h)
        self.assertIn("<title>v0.1.59 · 8K: 50.0 tok/s (rounds 1)</title>", h)
        self.assertNotIn("v0.1.60 ·", h)  # no rows, no markers

    def test_ladder_charts_use_fixed_colours_and_grey_after_four(self):
        rows = lambda: [srow("ladder", f"ladder.{k}k", 1, True, prompt_n=k * 1000) for k in (8, 16)]
        r = suite_results(["ladder"], [suite_arm(f"0.1.{50 + i}", rows()) for i in range(5)])
        h = build_html(r)
        for c in ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"):
            self.assertIn(c, h)
        self.assertIn("no own colour", h)

    def test_suite_rows_never_reach_base_numbers(self):
        rows = [srow("ladder", "ladder.8k", 1, True, decode=999.0)]
        base = suite_arm("0.1.59", [])
        self.assertEqual(ab_report._median(base, "chat0", "decode"), ab_report._median(suite_arm("0.1.59", rows), "chat0", "decode"))
        self.assertIsNone(ab_report._median(suite_arm("0.1.59", rows), "ladder.8k", "decode"))
        self.assertEqual(ab_report._median(suite_arm("0.1.59", rows), "ladder.8k", "decode", "ladder"), 999.0)

    def test_worse_correctness_is_marked(self):
        tasks = ["math1", "math2", "math3", "math4", "read1", "read2"]
        good = [srow("think", f"think.{t}", 1, True, decode=70.0) for t in tasks]
        bad = [srow("think", f"think.{t}", 1, t != "read2", decode=90.0) for t in tasks]
        h = build_html(suite_results(["think"], [suite_arm("0.1.59", good), suite_arm("0.1.60", bad)]))
        worse = re.findall(r'<tr class="worse"[^>]*>.*?</tr>', h, re.S)
        self.assertEqual(len(worse), 6)  # one per task, only the second version
        self.assertTrue(all("0.1.60" in w and "weniger richtig als v0.1.59" in w for w in worse))
        self.assertNotIn("weniger richtig als v0.1.60", h)
        self.assertIn("richtig 5/6", h)
        self.assertNotIn('class="fast"', " ".join(worse))  # no faster highlight on a worse version

    def test_faster_is_highlighted_only_among_equal_or_better(self):
        a = [srow("think", "think.math1", 1, True, decode=70.0)]
        b = [srow("think", "think.math1", 1, True, decode=90.0)]
        h = build_html(suite_results(["think"], [suite_arm("0.1.59", a), suite_arm("0.1.60", b)]))
        self.assertEqual(h.count('class="fast"'), 1)
        self.assertRegex(h, r'class="fast">90\.0')

    def test_other_context_marked(self):
        rows = [srow("agent", "agent.s.0", 1, True)]
        r = suite_results(["agent"], [suite_arm("0.1.59", rows), suite_arm("0.1.60", rows, n_ctx=131072)])
        h = build_html(r)
        self.assertEqual(h.count("anderer Kontext"), 1)
        self.assertIn("131072", h)
        self.assertIn("262144", h)

    def test_agent_section_ratio_and_left_out(self):
        a = [srow("agent", f"agent.s.{i}", 1, True, decode=d) for i, d in enumerate((50.0, 40.0, 30.0))]
        b = [srow("agent", f"agent.s.{i}", 1, True, decode=d) for i, d in enumerate((60.0, 40.0, 15.0))]
        r = suite_results(["agent"], [suite_arm("0.1.59", a, left_out=["agent.s.3"]), suite_arm("0.1.60", b, left_out=["agent.s.3"])])
        h = suite_sections(r)
        self.assertIn("1/3", h)  # turns won by 0.1.60
        self.assertIn("agent.s.3", h)
        self.assertRegex(h, r"x1\.000")  # median of 1.2, 1.0, 0.5

    def test_think_section_columns_and_escaping(self):
        rows = [srow("think", "think.math1", 1, True, predicted_n=1234, wall_s=45.5)]
        r = suite_results(["think"], [suite_arm("0.1.59", rows, suites_status="ran")])
        h = suite_sections(r)
        for want in ("1234", "45.5", "think.math1"):
            self.assertIn(want, h)
        r["arms"][0]["version"] = "0.1.59<x>"
        self.assertNotIn("<x>", suite_sections(r))

    def test_header_names_context_corpus_and_sampling(self):
        rows = [srow("agent", "agent.s.0", 1, True)]
        h = build_html(suite_results(["agent"], [suite_arm("0.1.59", rows)]))
        self.assertIn("v0.2.12", h)
        self.assertIn("temperature 0", h)
        self.assertIn("262144", h)

    def test_skipped_suites_reason_is_shown(self):
        r = suite_results(["think"], [suite_arm("0.1.59", [srow("think", "think.math1", 1, True)]),
                                      suite_arm("0.1.60", [], n_ctx=None, suites_status="skipped", suites_reason="cannot start <ctx>")])
        h = suite_sections(r)
        self.assertIn("cannot start &lt;ctx&gt;", h)


class LadderCard(unittest.TestCase):
    def test_render_png_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            html, png = Path(d) / "report.html", Path(d) / "x.png"
            html.write_text("x", encoding="utf-8")

            def fake(cmd, **kw):
                self.assertTrue(cmd[-1].endswith("#ladder"))
                png.write_bytes(b"png")
                return mock.Mock(returncode=0)
            with mock.patch.object(ab_report.subprocess, "run", side_effect=fake):
                self.assertTrue(render_png("edge", html, png, anchor="ladder"))

    def test_write_report_renders_ladder_png_when_chosen(self):
        rows = [srow("ladder", "ladder.8k", 1, True)]
        r = suite_results(["ladder"], [suite_arm("0.1.59", rows), suite_arm("0.1.60", rows)])
        calls = []

        def fake(browser, html, png, anchor="card"):
            calls.append((anchor, png.name))
            return True
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ab_report, "find_browser", return_value="edge"), \
                mock.patch.object(ab_report, "render_png", side_effect=fake):
            html, png = write_report(r, Path(d))
            self.assertEqual(png.name, "report.png")
        self.assertEqual(calls, [("card", "report.png"), ("ladder", "report-ladder.png")])
        calls.clear()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ab_report, "find_browser", return_value="edge"), \
                mock.patch.object(ab_report, "render_png", side_effect=fake):
            write_report(dict(r, suites=["think"]), Path(d))
        self.assertEqual(calls, [("card", "report.png")])


if __name__ == "__main__":
    unittest.main()
