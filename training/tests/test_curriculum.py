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


class InitDeterminism(unittest.TestCase):
    """2026-11-07 真机重训复现：`--seed 0` + 同一数据 + 同一份代码，两次权重 sha256 不同、
    相对 L2 差 1.41（跟权重本身同量级）。根因是 `LoRALinear.A` 用
    `nn.init.kaiming_uniform_` 吃 **torch 全局 RNG**，而训练入口只固定了数据顺序那个
    Generator；进程启动时全局 RNG 由系统熵播种，于是每次都是不同的随机投影。

    这里守住两件事：同 seed 必须逐位相同、不同 seed 必须不同；以及换成局部生成器后
    分布仍是 kaiming_uniform_(a=√5) 的闭式界 U(−1/√fan_in, +1/√fan_in)。
    """

    def test_same_seed_is_bit_identical_and_other_seed_differs(self):
        a, b = build(), build()
        self.assertTrue(torch.equal(a.o_proj.A, b.o_proj.A))
        self.assertTrue(torch.equal(a.up_proj.A, b.up_proj.A))
        m = Tiny(8)
        inject_lora(m, targets=("o_proj",), r=4, alpha=8, seed=1)
        self.assertFalse(torch.equal(a.o_proj.A, m.o_proj.A))
        # 入口契约：backward/sft.py 必须把 --seed 传进来，否则 --seed 只管数据顺序，
        # 初始 adapter 会永远是同一个（差距看不出来，但 A/B 多 seed 实验就失效）。
        sft_py = os.path.join(os.path.dirname(HERE), "backward", "sft.py")
        with open(sft_py, encoding="utf-8") as f:
            self.assertIn("seed=args.seed", f.read())

    def test_A_still_matches_kaiming_uniform_closed_form(self):
        m = build()
        bound = 1.0 / (8 ** 0.5)
        g = torch.Generator().manual_seed(0)   # build() 里 o_proj 是第 0 个 ⇒ init_seed=0
        ref = torch.empty(4, 8)
        ref.uniform_(-bound, bound, generator=g)
        self.assertTrue(torch.allclose(m.o_proj.A.detach(), ref))
        self.assertLessEqual(float(m.o_proj.A.detach().abs().max()), bound + 1e-7)


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
