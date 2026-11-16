"""Tests for ab_report: the HTML report and the PNG card.  No network, no browser."""
import copy
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_report  # noqa: E402
from ab_report import build_html, card_pair, render_png, write_report  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
