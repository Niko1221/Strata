"""极限模式（阶段 B）的三件基建的单测：阶梯反推 / 超时算法 / 看门狗落盘 + 续跑。

为什么用子进程：看门狗命中时是 `os._exit(3)`（挂住的是 CUDA kernel，Python 侧抛不出去），
在同进程里测会把测试进程一起杀掉。所以「挂死 → 记 hang → 落盘 → 退出码 3」只能在子进程里验。

跑法（Windows 上必须先关掉 PYTHONUTF8 的注入，否则 inductor 用 GBK 读模板）：
    set PYTHONUTF8=0 && python -m unittest test_bench_limits -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_fullctx as B  # noqa: E402


class SeqlensFromBudget(unittest.TestCase):
    def _budget(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"device": {"vram_gib": 48.0}, "rows": rows}, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_takes_half_and_full_per_profile(self):
        p = self._budget([{"profile": "hf-ce_mezo", "max_seq": 55808},
                          {"profile": "hf-ce_bwd", "max_seq": 9984}])
        got = B.seqlens_from_budget(p)
        # 一律向下取到 512 对齐：9984→9728、27904→27648，宁可少测一点也不能超极限
        self.assertIn(9728, got)
        self.assertIn(27648, got)
        self.assertIn(55808, got)
        self.assertEqual(got, sorted(set(got)))

    def test_no_fit_profile_is_ignored(self):
        p = self._budget([{"profile": "hf-ce_bwd(refetch)", "max_seq": 0},
                          {"profile": "engine-all", "max_seq": 131072}])
        self.assertEqual(B.seqlens_from_budget(p), [65536, 131072])

    def test_cap_and_floor(self):
        p = self._budget([{"profile": "x", "max_seq": 10 ** 9},
                          {"profile": "y", "max_seq": 100}])
        got = B.seqlens_from_budget(p, cap=131072)
        self.assertEqual(max(got), 131072)             # 不超过模型上下文
        self.assertTrue(all(v >= 512 for v in got))    # 太小没有测量意义


class CellsFromBudget(unittest.TestCase):
    """按方案分格：每个 algo_mode 只测它自己那一行的两点，而不是所有组合一起爬公共阶梯。"""

    def _budget(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"device": {"vram_gib": 48.0}, "rows": rows}, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def _rows(self):
        return [{"profile": "engine-all", "max_seq": 131072},
                {"profile": "hf-ce_mezo", "max_seq": 55808},
                {"profile": "hf-grpo_mezo(batch K=4)", "max_seq": 44032},
                {"profile": "hf-ce_bwd(ckpt)", "max_seq": 9984},
                {"profile": "hf-dpo_bwd(ckpt,rows=2)", "max_seq": 5120},
                {"profile": "hf-grpo_bwd(ckpt,batch K=4)", "max_seq": 2560},
                {"profile": "hf-ce_bwd(refetch)", "max_seq": 0}]

    def test_each_combination_gets_its_own_two_points(self):
        got = B.cells_from_budget(self._budget(self._rows()), ["ce", "dpo", "grpo"], ["mezo", "bwd"])
        self.assertEqual(got["ce_mezo"], [27648, 55808])
        self.assertEqual(got["grpo_mezo"], [22016, 44032])
        self.assertEqual(got["dpo_bwd"], [2560, 5120])
        self.assertEqual(got["grpo_bwd"], [1024, 2560])

    def test_the_engine_row_is_not_a_cell(self):
        # engine-all 是引擎侧那一行，bench_fullctx 是 HF 侧，不该把 131072 拉进来
        got = B.cells_from_budget(self._budget(self._rows()), ["ce"], ["mezo"])
        self.assertNotIn(131072, got["ce_mezo"])

    def test_a_parenthetical_suffix_still_matches(self):
        got = B.cells_from_budget(self._budget(self._rows()), ["grpo"], ["bwd"])
        self.assertEqual(got["grpo_bwd"], [1024, 2560])          # 来自 "(ckpt,batch K=4)" 那行

    def test_two_rows_with_one_name_do_not_overwrite_the_fitting_one(self):
        # hf-ce_bwd(ckpt)=9984 与 hf-ce_bwd(refetch)=0 同名：取大的，否则会被覆盖成「跑不了」
        got = B.cells_from_budget(self._budget(self._rows()), ["ce"], ["bwd"])
        self.assertEqual(got["ce_bwd"], [4608, 9728])

    def test_dpo_mezo_falls_back_to_the_shared_forward_row(self):
        got = B.cells_from_budget(self._budget(self._rows()), ["dpo"], ["mezo"])
        self.assertEqual(got["dpo_mezo"], [27648, 55808])

    def test_a_combination_with_no_row_is_absent_not_zero(self):
        got = B.cells_from_budget(self._budget(self._rows()), ["ce", "grpo"], ["mezo"])
        self.assertNotIn("dpo_mezo", got)
        self.assertEqual(sum(len(v) for v in got.values()), 4)   # 2 组合 × 2 点


class TimeoutPolicy(unittest.TestCase):
    def _args(self, v):
        return SimpleNamespace(per_cell_timeout=v)

    def test_auto_has_60s_floor(self):
        # 上一档 4 s（引擎 ce）⇒ 3×60 = 180，不是 3×10
        self.assertEqual(B.cell_timeout_seconds(self._args("auto"), 4.0), 180.0)

    def test_auto_scales_with_prev_step(self):
        # 上一档 120 s（GRPO）⇒ 3×300 = 900
        self.assertEqual(B.cell_timeout_seconds(self._args("auto"), 120.0), 900.0)

    def test_first_cell_uses_floor(self):
        self.assertEqual(B.cell_timeout_seconds(self._args("auto"), None), 180.0)

    def test_explicit_and_off(self):
        self.assertEqual(B.cell_timeout_seconds(self._args("45"), 4.0), 45.0)
        self.assertEqual(B.cell_timeout_seconds(self._args("off"), 4.0), 0.0)


class EstimateStepSeconds(unittest.TestCase):
    """上一档的步时必须按 T 放大再用：4608 → 27648 是 6 倍，直接用会误杀（实测 dpo 就被记成 hang）。"""

    def test_no_previous_cell_has_no_estimate(self):
        self.assertIsNone(B.estimate_step_seconds(None, 27648))

    def test_same_length_is_the_same_step(self):
        self.assertAlmostEqual(B.estimate_step_seconds((20.0, 4608), 4608), 20.0)

    def test_a_six_times_longer_cell_scales_super_linearly(self):
        # 指数 1.25：6 倍长度 ⇒ 约 9.4 倍步时（实测 ce 是 7.0 倍，所以这个估计偏保守）
        got = B.estimate_step_seconds((20.0, 4608), 27648)
        self.assertGreater(got, 6 * 20.0)
        self.assertAlmostEqual(got, 20.0 * 6 ** 1.25, places=4)

    def test_a_shorter_cell_is_not_scaled_up(self):
        # T 只可能比上一档大（阶梯升序），真给了更小的就按原步时，别把它放大成误杀
        self.assertAlmostEqual(B.estimate_step_seconds((200.0, 27648), 4608), 200.0 * (4608 / 27648) ** 1.25)

    def test_a_zero_length_reference_is_passed_through(self):
        self.assertEqual(B.estimate_step_seconds((20.0, 0), 27648), 20.0)

    def test_the_timeout_is_built_from_the_estimate(self):
        est = B.estimate_step_seconds((26.37, 4608), 27648)
        self.assertGreater(B.cell_timeout_seconds(SimpleNamespace(per_cell_timeout="auto"), est), 180.0)


class EstimateForCell(unittest.TestCase):
    """按方案分格后，一个组合可能没有更小的档——不能因此顶着 180 s 的空白上限。"""

    def test_its_own_previous_cell_wins(self):
        prev = {("ce", "mezo"): (146.6, 27648), ("dpo", "mezo"): (26.4, 4608)}
        self.assertAlmostEqual(B.estimate_for_cell(prev, "dpo", "mezo", 27648),
                               26.4 * 6 ** 1.25, places=4)

    def test_a_combination_with_no_history_borrows_the_priciest_estimate(self):
        # dpo_mezo 只测 27648/55808：拿 ce@27648 当参考，比顶着 180 s 上限把真步时记成 hang 强
        prev = {("ce", "mezo"): (146.6, 27648)}
        self.assertEqual(B.estimate_for_cell(prev, "dpo", "mezo", 27648), 146.6)
        self.assertGreater(B.cell_timeout_seconds(SimpleNamespace(per_cell_timeout="auto"),
                                                  B.estimate_for_cell(prev, "dpo", "mezo", 27648)), 180.0)

    def test_nothing_measured_yet_is_still_none(self):
        self.assertIsNone(B.estimate_for_cell({}, "dpo", "mezo", 27648))


_DRIVER = r'''
import os, sys, time, types, json
sys.path.insert(0, r"{here}")
import bench_fullctx as B

B.timed = lambda fn, w, r: (fn(), (1.0, 0.0))[1]
B._preflight = lambda *a, **k: True

args = types.SimpleNamespace(per_cell_timeout="{timeout}", out=r"{out}",
                             warmup=0, reps=1, on_oom="skip")
rows, done, prev = [], set(), {{}}
sampler = B.RSSSampler(0.05); sampler.start()

def slow():
    time.sleep(30)

B.run_cell(args, rows, done, sampler, prev, "mezo", "ce", slow, 4096, "drv", 8192)
print("SHOULD_NOT_REACH")
'''


class Watchdog(unittest.TestCase):
    def test_hang_is_recorded_then_exits_3(self):
        out = os.path.join(tempfile.mkdtemp(), "lim.json")
        code = _DRIVER.format(here=HERE, timeout="0.4", out=out)
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(r.returncode, 3, r.stderr[-800:])
        self.assertNotIn("SHOULD_NOT_REACH", r.stdout)
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(len(d["rows"]), 1)
        row = d["rows"][0]
        self.assertEqual(row["status"], "hang")
        self.assertEqual((row["algo"], row["mode"], row["T"]), ("ce", "mezo", 8192))
        self.assertIn("watchdog", row["note"])
        self.assertIn("hang", r.stderr)

    def test_resume_skips_measured_cells(self):
        out = os.path.join(tempfile.mkdtemp(), "lim.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"config": {}, "rows": [
                {"algo": "ce", "mode": "mezo", "T": 8192, "toks": 8192, "s": 1.0,
                 "tok_s": 8192.0, "peak": 1.0, "peak_rss": 1.0, "status": "ok"}]}, f)
        code = _DRIVER.format(here=HERE, timeout="0.4", out=out).replace(
            "rows, done, prev = [], set(), {}", '''rows, done, prev = [], set(), {}
old = json.load(open(args.out, encoding="utf-8"))
rows.extend(old["rows"])
done = {(r["algo"], r["mode"], r["T"]) for r in old["rows"]}''')
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("跳过", r.stdout)                 # 已测过的格子不再跑（也就不会挂死）
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        self.assertEqual(len(d["rows"]), 1)
        self.assertEqual(d["rows"][0]["status"], "ok")


class SyntheticFlag(unittest.TestCase):
    def test_default_is_synthetic(self):
        self.assertIsNone(B._CORPUS)

    def test_real_corpus_marks_repeat(self):
        import numpy as np
        f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        np.save(f.name, np.arange(100, dtype=np.int32))     # 只有 100 个 token
        self.addCleanup(setattr, B, "_CORPUS", None)
        n = B.load_corpus(f.name)
        self.assertEqual(n, 100)
        ids, att = B.synth(64, 4, 32, "cpu", seed=0)
        self.assertEqual(tuple(ids.shape), (4, 64))
        self.assertEqual(tuple(att.shape), (4, 64))
        # 4×64=256 > 100 ⇒ 至少要平铺 3 遍，报告里 repeat=3
        self.assertEqual(B._REPEAT[64], 3)
        self.assertTrue(bool((ids < 100).all()))        # 全是语料里的 id，不是 synth 的随机 id


if __name__ == "__main__":
    unittest.main()
