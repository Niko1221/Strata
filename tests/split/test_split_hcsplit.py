"""The NVIDIA split's prompt-path switches (src/split): pictures of what each default is, from the sources (no GPU).

FNX_PF_HCSPLIT (0.2.10.1): the hyper-connection read of a prompt chunk runs on one card's rows each instead of on both cards identically. Mode 1 (the default)
is bit-identical to the unsplit read: it splits only passes of whole 4096-row chunks and runs the down GEMM through the cublasLt algorithm that start-up
calibration found to give the same rows at 4096 and 2048 rows (and cublasGemmEx's at 4096); no such algorithm on a card: the read stays on both cards.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "split" / "src"


def text(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


class HcSplit(unittest.TestCase):
    def test_default_is_on_and_exact_mode_needs_the_calibrated_algorithm(self):
        s = text("prefill.cu")
        self.assertRegex(s, r'p->hcs_mode = \(int\) env_int\("FNX_PF_HCSPLIT", 1\)')
        # the split runs on every card or on none; mode 1 only with the calibrated algorithm on every card
        self.assertRegex(s, r"pf_\[c\]->hcs_mode != 1 \|\| pf_\[c\]->down_lt")
        # a pass is split only when every chunk is a whole 4096 rows (mode 1): the sizes the algorithm was calibrated at
        self.assertRegex(s, r"hcs_mode == 1 && \(!pf_\[0\]->down_lt \|\| \(int\) ck\.tokens\.size\(\) != 4096\)")
        self.assertIn("if (n != 4096 && n != 2048) return false;", s)

    def test_nvidia_only(self):
        s = text("prefill.cu")
        i = s.index('p->hcs_mode = (int) env_int("FNX_PF_HCSPLIT"')
        self.assertIn("#if !FNX_HIP", s[i - 200:i])   # the AMD build never splits the read
        self.assertIn("NCARD == 2", s[i:i + 200])
        cm = (ROOT / "src" / "split" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("CUDA::cublasLt", cm)

    def test_the_sum_before_a_split_read_is_a_reduce_scatter_only(self):
        s = text("pf_common.cuh")
        self.assertIn("bool own_out = false", s)
        self.assertRegex(s, r"o\.k == PfRingOp::PASS \|\| o\.k == PfRingOp::SIGNALA \|\| o\.k == PfRingOp::WAITA")

    def test_four_chunks_a_pass_only_while_nothing_else_decodes(self):
        sf = (ROOT / "src" / "split" / "tools" / "sfnx.cpp").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"e\.set_pass_cap\(any_decoding\(\) \? 2 : 0\)", sf)), 2)   # solo and batch admissions
        self.assertIn("void set_pass_cap(int n)", (SRC / "engine.hpp").read_text(encoding="utf-8"))

    def test_other_switches_keep_their_defaults(self):
        k = text("kernels.cu")
        self.assertRegex(k, r'env_int\("FNX_PFX_ATTN_I8", 0\)')   # off: the owner's decision is pending (agent-A10)
        self.assertRegex(k, r'env_int\("FNX_PFX_GDN_TC", 1\)')
        self.assertRegex(text("engine.cu"), r'env_int\("FNX_ONEHOP", 1\) == 2')   # =2 is a screen switch for two cards, not a default


if __name__ == "__main__":
    unittest.main()
