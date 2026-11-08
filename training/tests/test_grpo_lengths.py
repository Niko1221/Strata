"""GRPO 的长度门：不可训练必须报错，不能静默筛样本（审核 §5.4.0 第二条）。

    python -m unittest test_grpo_lengths

要守住的：
  1. `--max-seq` 放不下任何一条 ⇒ 报错，且错误信息给出两条改法（调大 max_seq / 查 max_new）；
  2. 超限比例 > `--max-skip-frac`（默认 0）⇒ 报错 —— 默认行为是「一条都不许丢」，
     因为丢掉的样本会让训练集变成「短轨迹子集」，而结果会被当成目标任务的结果；
  3. 显式放宽到 0.3、实际 25% ⇒ 放行并打印比例（长尾真实存在，但不是静默）；
  4. 空 completion 单独计数、不进长度门（采样失败 ≠ 长度不可训练）；
  5. 缺 `prompt_ids` 且没给 tokenizer ⇒ **不猜**，返回 known=False 交训练循环再判。

这条与 `sft.py:798` 的 fail-fast 是同一条判据：「能不能训」是前置问题，
「训完会不会退化」才是后验评测（后者见 RETENTION-EVAL.zh.md §3.7）。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import unittest

import grpo


def traj(p=10, c=10, **kw):
    t = {"prompt": "q", "prompt_ids": list(range(p)), "token_ids": list(range(c)),
         "content": "a", "reward": 1.0}
    t.update(kw)
    return t


class FakeTok:
    """只实现 apply_chat_template：返回固定长度的假 prompt ids。"""

    def __init__(self, n=7):
        self.n = n

    def apply_chat_template(self, msgs, add_generation_prompt=True, return_dict=False):
        return list(range(self.n))


def quiet(*a, **k):
    pass


class TestLengthGate(unittest.TestCase):

    def test_all_fit(self):
        st = grpo.check_trainable_lengths(
            [traj(10, 10), traj(20, 20), traj(5, 5)], max_seq=768, log=quiet)
        self.assertEqual(st["n"], 3)
        self.assertEqual(st["n_toolong"], 0)
        self.assertEqual(st["max"], 40)
        self.assertTrue(st["known"])

    def test_nothing_fits_raises(self):
        with self.assertRaises(SystemExit) as cm:
            grpo.check_trainable_lengths(
                [traj(600, 600), traj(700, 700)], max_seq=768, log=quiet)
        msg = str(cm.exception)
        self.assertIn("放不下任何一条轨迹", msg)
        self.assertIn("--max-seq", msg)
        self.assertIn("--max-new", msg)   # 第二条改法：采样上限与训练上限矛盾

    def test_default_is_zero_tolerance(self):
        # 4 条里 1 条超限 = 25% > 默认 0 ⇒ 必须报错，不能静默丢
        trajs = [traj(10, 10), traj(10, 10), traj(10, 10), traj(500, 500)]
        with self.assertRaises(SystemExit) as cm:
            grpo.check_trainable_lengths(trajs, max_seq=768, log=quiet)
        msg = str(cm.exception)
        self.assertIn("条轨迹", msg)
        self.assertIn("25.0%", msg)
        self.assertIn("--max-skip-frac", msg)   # 提示显式放宽的写法

    def test_tolerance_allows_and_reports(self):
        trajs = [traj(10, 10), traj(10, 10), traj(10, 10), traj(500, 500)]
        st = grpo.check_trainable_lengths(trajs, max_seq=768, max_skip_frac=0.3, log=quiet)
        self.assertEqual(st["n_toolong"], 1)
        self.assertEqual(st["n"], 4)

    def test_just_at_limit_passes(self):
        # 恰好等于 max_seq 不算超（判据是 > max_seq）
        st = grpo.check_trainable_lengths([traj(380, 388)], max_seq=768, log=quiet)
        self.assertEqual(st["n_toolong"], 0)

    def test_empty_completion_not_length_gated(self):
        trajs = [traj(10, 10), traj(10, 0)]      # 第二条 token_ids 为空 = 采样失败
        st = grpo.check_trainable_lengths(trajs, max_seq=768, log=quiet)
        self.assertEqual(st["n_empty"], 1)
        self.assertEqual(st["n"], 1)             # 只有 1 条参与长度判定
        self.assertEqual(st["n_toolong"], 0)

    def test_all_empty_does_not_raise(self):
        st = grpo.check_trainable_lengths([traj(10, 0), traj(10, 0)], max_seq=768, log=quiet)
        self.assertEqual(st["n"], 0)
        self.assertFalse(st["known"])            # 判不了就说判不了，不猜

    def test_missing_prompt_ids_without_tokenizer_is_unknown(self):
        t = traj(0, 500)
        t.pop("prompt_ids")
        st = grpo.check_trainable_lengths([t], max_seq=768, log=quiet)
        self.assertFalse(st["known"])            # 没有 tokenizer ⇒ 不判、不报错
        self.assertEqual(st["n_toolong"], 0)

    def test_missing_prompt_ids_with_tokenizer_is_exact(self):
        t = traj(0, 500)
        t.pop("prompt_ids")
        # 给了 tokenizer ⇒ prompt 部分按本地模板编码（300）+ completion 500 = 800，是精确值
        st = grpo.check_trainable_lengths([t], max_seq=1000, tok=FakeTok(300), log=quiet)
        self.assertTrue(st["known"])
        self.assertEqual(st["max"], 800)


if __name__ == "__main__":
    unittest.main()
