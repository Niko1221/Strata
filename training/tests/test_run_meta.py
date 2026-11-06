"""超参落盘 + 训练一步闭环（审核 §5.4.1 第 1 项 / §9 Q7）。

    python -m unittest test_run_meta

要守住的：
  1. 每次训练都留下 `<out>.meta.json`，且记的是 **vars(args) 全量** —— 以后新增参数
     自动进 meta，不会再出现「−31 项那次不知道 lr/steps/rank」的情况；
  2. `phase=start` 在加载模型之前就落盘，崩了也查得到；`phase=done` 合并而不覆盖；
  3. 写 meta 失败绝不把训练搞挂；
  4. §9 Q7 补的那 1 项端到端：**项目自己的 `objectives.sft_loss` + 项目自己的
     `update_rules.BackwardRule` 真跑一步，answer CE 必须下降**。

第 4 条的边界要写清楚：这是**组件级最小闭环**（torch autograd + AdamW + clip 都是真的），
不含 125B 模型加载与引擎 `SCORE`；full e2e 需要 GPU，走 §6 的 V5/V6。别把这条
当成「125B 训得动」的证据，它守的是「训练循环的接线没断」。
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


import argparse
import json
import tempfile
import unittest
from unittest import mock

import run_meta as RM

try:
    import torch
except ImportError:                      # 零阶那条树没有 torch 也能跑前 3 组
    torch = None


def _args(**kw) -> argparse.Namespace:
    base = dict(lr=1e-4, steps=40, rank=16, alpha=32, seed=0, out="final.pt",
                data="", dpo_data="", gguf=["a.gguf"], ckpt=True, epochs=None,
                dpo_beta=0.1, cap_tol=0.5, cap_stop=1.5, eval_every=4)
    base.update(kw)
    return argparse.Namespace(**base)


class RunMeta(unittest.TestCase):
    def test_full_args_recorded_and_json_serializable(self):
        a = _args(a_path=_p.Path("somewhere"), extra_obj=object())
        rec = RM.run_meta(a, phase="start")
        json.dumps(rec)                                   # 非标量不许让记录崩掉
        for k in ("lr", "steps", "rank", "alpha", "seed", "out", "ckpt",
                  "dpo_beta", "cap_tol", "eval_every"):
            self.assertIn(k, rec, f"复现必需的参数 {k} 丢了")
        self.assertEqual(rec["phase"], "start")
        self.assertEqual(rec["schema"], RM.SCHEMA)
        self.assertEqual(rec["gguf"], ["a.gguf"])
        self.assertEqual(rec["a_path"], "somewhere")     # Path 降级成字符串
        self.assertIn("recorded_at", rec)
        self.assertIn("python", rec)
        self.assertIn("git_head", rec)

    def test_new_argument_automatically_enters_meta(self):
        # 治根判据：不是「手列几个字段」，而是全量 vars(args)。
        a = _args(lr_schedule="cosine", some_future_knob=7)
        rec = RM.run_meta(a)
        self.assertEqual(rec["lr_schedule"], "cosine")
        self.assertEqual(rec["some_future_knob"], 7)

    def test_data_digest_handles_jsonl_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            f = _p.Path(d) / "train.jsonl"
            f.write_text('{"prompt":"a","answer":"b"}\n\n{"prompt":"c","answer":"d"}\n',
                         encoding="utf-8")
            rec = RM.run_meta(_args(data="jsonl:" + str(f)))
            self.assertTrue(rec["data_present"])
            self.assertEqual(rec["data_rows"], 2)        # 空行不算
            self.assertEqual(rec["data_sha256"], __import__("hashlib").sha256(
                f.read_bytes()).hexdigest())

    def test_missing_data_is_reported_not_silently_dropped(self):
        rec = RM.run_meta(_args(data="jsonl:/does/not/exist.jsonl"))
        self.assertFalse(rec["data_present"])

    def test_start_then_done_merge_without_losing_start_fields(self):
        with tempfile.TemporaryDirectory() as d:
            out = _p.Path(d) / "nested" / "final.pt"     # 父目录还不存在
            p1 = RM.write_run_meta(str(out), _args(), phase="start", steps_done=0)
            self.assertTrue(_p.Path(p1).is_file(), "start 必须先落盘")
            p2 = RM.write_run_meta(str(out), _args(), phase="done", steps_done=40,
                                   peak_gib=41.1, best_step=4)
            rec = json.loads(_p.Path(p2).read_text(encoding="utf-8"))
            self.assertEqual(rec["phase"], "done")
            self.assertEqual(rec["steps_done"], 40)
            self.assertEqual(rec["peak_gib"], 41.1)
            self.assertEqual(rec["best_step"], 4)
            self.assertEqual(rec["lr"], 1e-4)            # start 的参数没被覆盖掉
            self.assertEqual(rec["rank"], 16)

    def test_extra_may_contain_out_without_colliding_with_the_path_arg(self):
        # 2026-11-07 真机重训踩到：done_extra 里带了 out=...，而 write_run_meta 的
        # 首参当时也叫 out ⇒ TypeError（且是在函数调用时抛，函数体无法自救）。
        # 调用方常把 vars(args) 派生的 dict 直接 ** 展开，这个形状必须能跑。
        with tempfile.TemporaryDirectory() as d:
            out = _p.Path(d) / "final.pt"
            p = RM.write_run_meta(str(out), _args(), phase="done",
                                  out=str(out), steps_done=8, peak_gib=28.0)
            rec = json.loads(_p.Path(p).read_text(encoding="utf-8"))
            self.assertEqual(rec["phase"], "done")
            self.assertEqual(rec["steps_done"], 8)
            self.assertEqual(rec["peak_gib"], 28.0)

    def test_write_failure_warns_but_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            out = _p.Path(d) / "final.pt"
            with mock.patch("run_meta.pathlib.Path.replace",
                            side_effect=OSError("disk full")):
                self.assertEqual(RM.write_run_meta(str(out), _args(), phase="start"), "")


@unittest.skipUnless(torch, "torch not installed")
class TrainStepE2E(unittest.TestCase):
    """§9 Q7：真跑一步再断言 CE 下降。边界见本文件 docstring。"""

    def test_one_backward_step_reduces_answer_ce(self):
        import objectives as O
        from update_rules import BackwardRule

        V, target = 8, 3
        w = torch.nn.Parameter(torch.zeros(V))           # 可训练 logits（LoRA 的角色）
        rule = BackwardRule([w], lr=0.5, clip=1.0)

        def loss_fn():
            logp = torch.log_softmax(w, dim=0)
            return O.sft_loss(logp[target:target + 1])    # answer 段（1 个 token）

        torch.manual_seed(0)
        before = float(loss_fn().detach())
        m = rule.step(loss_fn)
        after = float(loss_fn().detach())
        self.assertGreater(m["grad_norm"], 0.0, "梯度不该是 0")
        self.assertLess(after, before, f"跑一步后 CE 没降：{before:.4f} → {after:.4f}")

        # 多跑几步应当单调改善，而不是碰巧一个方向对。
        for _ in range(20):
            rule.step(loss_fn)
        self.assertLess(float(loss_fn().detach()), 0.5)

    def test_numpy_and_torch_paths_agree_on_the_same_logp(self):
        import numpy as np
        import objectives as O
        lp = np.array([-1.0, -2.0, -0.5])
        mask = np.array([True, False, True])
        self.assertAlmostEqual(float(O.sft_loss(lp, mask)),
                               float(O.sft_loss(torch.tensor(lp), torch.tensor(mask))),
                               places=6)


if __name__ == "__main__":
    unittest.main()
