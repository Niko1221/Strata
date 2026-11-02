"""`judge_limits.py` 的单测：三条判据各自能独立报错。

判据松了会漏掉真 bug（降速被当成"长上下文本来慢"）；紧了会把正常格子判死（显存顶到上限时
变慢是正常的）。所以这里两个方向都测。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import judge_limits as J  # noqa: E402


def cell(T, tok_s, peak=20.0, status="ok", algo="ce", mode="mezo", synthetic=False, **kw):
    d = {"algo": algo, "mode": mode, "T": T, "toks": T, "status": status,
         "synthetic": synthetic, "peak": peak, "peak_rss": 30.0}
    if status == "ok":
        d.update(tok_s=tok_s, s=T / tok_s if tok_s else 0.0)
    d.update(kw)
    return d


def judge(rows):
    return {k: J.judge_cell_series(v, J.budgets_from({"rows": rows}, 48.0))
            for k, v in J.group(rows).items()}


class Limit(unittest.TestCase):
    def test_limit_is_max_ok(self):
        rep = judge([cell(8192, 4000.0), cell(16384, 4100.0), cell(32768, 4200.0),
                     cell(65536, 0, status="oom")])
        got = rep[("ce", "mezo")]
        self.assertEqual(got["limit_T"], 32768)
        self.assertEqual(got["at_limit"]["T"], 32768)
        self.assertEqual([b["T"] for b in got["breaks"]], [65536])

    def test_hang_and_budget_are_breaks(self):
        rep = judge([cell(8192, 4000.0), cell(16384, 0, status="hang"),
                     cell(32768, 0, status="budget")])
        got = rep[("ce", "mezo")]
        self.assertEqual(got["limit_T"], 8192)
        self.assertEqual({b["status"] for b in got["breaks"]}, {"hang", "budget"})


class Slowdown(unittest.TestCase):
    def test_drop_with_headroom_is_a_violation(self):
        rep = judge([cell(8192, 4000.0, peak=20.0), cell(16384, 2000.0, peak=24.0)])
        viol = rep[("ce", "mezo")]["slowdown"]
        self.assertEqual(len(viol), 1)
        self.assertAlmostEqual(viol[0]["drop"], 0.5, places=2)

    def test_drop_at_capacity_is_not_a_violation(self):
        # 峰值贴到预算（48×0.95×0.9 = 41.0）⇒ 变慢是显存逼的，不是退化
        rep = judge([cell(8192, 4000.0, peak=20.0), cell(16384, 2000.0, peak=45.0)])
        self.assertEqual(rep[("ce", "mezo")]["slowdown"], [])

    def test_small_drop_is_fine(self):
        rep = judge([cell(8192, 4000.0), cell(16384, 3800.0)])   # 掉 5%
        self.assertEqual(rep[("ce", "mezo")]["slowdown"], [])

    def test_synthetic_cells_are_not_judged(self):
        rep = judge([cell(8192, 4000.0, synthetic=True), cell(16384, 100.0, synthetic=True)])
        self.assertEqual(rep[("ce", "mezo")]["slowdown"], [])
        self.assertEqual(rep[("ce", "mezo")]["limit_T"], 16384)   # 极限仍然有效

    def test_monotonic_increase_is_fine(self):
        rep = judge([cell(8192, 3600.0), cell(32768, 4300.0)])    # 越长越快（专家去重后正常）
        self.assertEqual(rep[("ce", "mezo")]["slowdown"], [])


class Formula(unittest.TestCase):
    def test_within_tolerance_passes(self):
        rep = judge([cell(8192, 4000.0)])
        b = {"rows": [{"profile": "hf-ce_mezo", "max_seq": 9000}]}
        got = J.compare_formula(rep, b)
        self.assertTrue(got["ce/mezo"]["pass"])

    def test_optimistic_formula_fails(self):
        rep = judge([cell(8192, 4000.0)])
        b = {"rows": [{"profile": "hf-ce_mezo", "max_seq": 9984}]}
        got = J.compare_formula(rep, b)
        self.assertEqual(got["ce/mezo"]["ratio"], round(8192 / 9984, 3))
        self.assertFalse(got["ce/mezo"]["pass"])        # 0.82 < 0.85

    def test_zero_measured_still_fails(self):
        rep = judge([cell(8192, 0, status="oom", mode="bwd")])
        b = {"rows": [{"profile": "hf-ce_bwd", "max_seq": 9984}]}
        got = J.compare_formula(rep, b)
        self.assertFalse(got["ce/bwd"]["pass"])

    def test_engine_profile_matches_mezo_rows(self):
        rep = judge([cell(131072, 4300.0)])
        b = {"rows": [{"profile": "engine-all", "max_seq": 131072}]}
        got = J.compare_formula(rep, b)
        self.assertTrue(got["ce/mezo"]["pass"])


class Cli(unittest.TestCase):
    def _run(self, rows, budget=None, extra=()):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "bench.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"config": {}, "rows": rows}, f)
        argv = [p] + list(extra)
        if budget is not None:
            bp = os.path.join(d, "budget.json")
            with open(bp, "w", encoding="utf-8") as f:
                json.dump(budget, f)
            argv += ["--budget", bp]
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        old = sys.argv
        sys.argv = ["judge_limits.py"] + argv
        try:
            with redirect_stdout(buf):
                code = J.main()
        finally:
            sys.argv = old
        return code, buf.getvalue()

    def test_clean_run_exits_0(self):
        code, out = self._run([cell(8192, 3600.0), cell(32768, 4300.0)])
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)

    def test_slowdown_exits_1(self):
        code, out = self._run([cell(8192, 4000.0, peak=20.0), cell(16384, 1500.0, peak=22.0)])
        self.assertEqual(code, 1)
        self.assertIn("降速", out)

    def test_synthetic_only_warns(self):
        code, out = self._run([cell(8192, 4000.0, synthetic=True),
                               cell(16384, 3000.0, synthetic=True)])
        self.assertEqual(code, 0)
        self.assertIn("synthetic", out)


if __name__ == "__main__":
    unittest.main()
