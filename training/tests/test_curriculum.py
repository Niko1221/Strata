"""课程续训（`--init-adapter`）的单测：只用小模块，不加载 125B 模型。

这条路径的风险不在「load 会不会成功」，而在**静默错配**：
rank / alpha / targets 变了之后，键名还能对上、形状对不上；或者只 load 进一半，
D 阶段的「加了数据集，身份掉了多少」就永远解释不清。所以这里重点测三件事：
  ① 正常往返：存出来的 θ 装回去，逐位相等（A 与 B 都验，B 初始为 0 时要能看出变化）；
  ② 形状不符 → 报错，且**不**部分写入；
  ③ 键不齐（少 module / 多 module）→ 报错。
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


import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from lora import (inject_lora, load_lora_state, lora_state_dict,  # noqa: E402
                  trainable_summary)


class Tiny(nn.Module):
    """只有几层 nn.Linear 的壳子；叶子名 o_proj 命中 DEFAULT_TARGETS。"""

    def __init__(self, d: int = 8):
        super().__init__()
        self.o_proj = nn.Linear(d, d)
        self.up_proj = nn.Linear(d, d)
        self.other = nn.Linear(d, d)


def build(d: int = 8, r: int = 4, alpha: int = 8) -> Tiny:
    m = Tiny(d)
    n = inject_lora(m, targets=("o_proj", "up_proj"), r=r, alpha=alpha)
    assert n == 2
    return m


def _save(model, path: str) -> None:
    torch.save(lora_state_dict(model), path)


class RoundTrip(unittest.TestCase):
    def test_state_survives_save_and_load(self):
        src = build()
        with torch.no_grad():                      # 把 B 从 0 挪开，否则「载入成功」与「没载入」看不出差别
            src.o_proj.B.normal_()
            src.up_proj.B.normal_()
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        _save(src, f.name)

        dst = build()
        self.assertEqual(trainable_summary(dst)["trainable"], trainable_summary(src)["trainable"])
        k = load_lora_state(dst, f.name)
        self.assertEqual(k, 4)                     # 2 modules × (A, B)
        for name in ("o_proj", "up_proj"):
            self.assertTrue(torch.equal(getattr(src, name).B, getattr(dst, name).B))
            self.assertTrue(torch.equal(getattr(src, name).A, getattr(dst, name).A))

    def test_returns_tensor_count(self):
        m = build()
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        _save(m, f.name)
        self.assertEqual(load_lora_state(build(), f.name), 4)


class Mismatch(unittest.TestCase):
    def _file(self, model):
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        _save(model, f.name)
        return f.name

    def test_rank_change_is_fatal(self):
        p = self._file(build(r=4, alpha=8))
        with self.assertRaises(SystemExit) as cm:
            load_lora_state(build(r=8, alpha=16), p)     # 同 alpha/r 比，但形状变了
        self.assertIn("形状", str(cm.exception))

    def test_missing_module_is_fatal(self):
        p = self._file(build())                          # 两个 module 都注入了
        m = Tiny(8)
        inject_lora(m, targets=("o_proj",), r=4, alpha=8)   # 只注入一个 ⇒ 存的那份多出 up_proj 的键
        with self.assertRaises(SystemExit) as cm:
            load_lora_state(m, p)
        self.assertIn("多", str(cm.exception))

    def test_no_lora_at_all_is_fatal(self):
        p = self._file(build())
        with self.assertRaises(SystemExit):
            load_lora_state(Tiny(8), p)                  # 还没 inject
        # 注意 args 名：错误里应当说清「先 inject_lora」

    def test_not_a_state_dict_is_fatal(self):
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        torch.save([1, 2, 3], f.name)
        with self.assertRaises(SystemExit):
            load_lora_state(build(), f.name)

    def test_failure_does_not_half_write(self):
        """报错时目标模型的 B 必须还是 0——半写入会让后续训练假装「续上了」。"""
        p = self._file(build(r=4, alpha=8))
        dst = build(r=8, alpha=16)
        before = [t.clone() for t in lora_state_dict(dst).values()]
        with self.assertRaises(SystemExit):
            load_lora_state(dst, p)
        after = list(lora_state_dict(dst).values())
        for b, a in zip(before, after):
            self.assertTrue(torch.equal(b, a))


if __name__ == "__main__":
    unittest.main()
