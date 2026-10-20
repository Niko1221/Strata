"""Tests for tools/check_isa_guard.py's policy (#795): which functions may hold wider-ISA instructions.  No binary, no GPU.

    python -m unittest tools.test_check_isa_guard
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_isa_guard as G  # noqa: E402


class WideInstruction(unittest.TestCase):
    def test_evex_prefix_is_wide(self):
        self.assertTrue(G.is_wide(0x62, "vmovups", "zmm0, zmmword ptr [rcx]"))
        self.assertTrue(G.is_wide(0x62, "vpmullq", "xmm0, xmm0, xmm6"))      # EVEX with xmm0-15 operands only

    def test_vex_avx2_is_not(self):
        self.assertFalse(G.is_wide(0xC4, "vpinsrd", "xmm1, xmm1, dword ptr [r13 + r9*4 + 0x405fc0], 3"))
        self.assertFalse(G.is_wide(0xC5, "vmovdqu", "ymm0, ymmword ptr [r14]"))

    def test_avx_vnni_and_registers_above_15(self):
        self.assertTrue(G.is_wide(0xC4, "vpdpbusd", "ymm0, ymm1, ymm2"))
        self.assertTrue(G.is_wide(0xC5, "vpxor", "ymm17, ymm17, ymm17"))
        self.assertTrue(G.is_wide(0xC5, "kmovb", "k1, eax"))


class Policy(unittest.TestCase):
    def test_kernel_in_an_anonymous_namespace_of_a_wide_file_is_fine(self):
        self.assertIsNone(G.judge("??$gu_rows@$0BA@$00@?A0xa1d599f5@cpu@kernels@strata@@YAXXZ",
                                  "strata_kernels_cpu:iq_avx512.cpp.obj", "", False, False, False))
        self.assertIsNone(G.judge("strata::kernels::cpu::(anonymous namespace)::gu_rows<18, 1>(int)", None, "", False,
                                  False, True))

    def test_listed_public_entry_point_is_fine(self):
        self.assertIsNone(G.judge("?act_quant_q8_1@cpu@kernels@strata@@YAXPEBMHAEAUActQ@123@@Z",
                                  "strata_kernels_cpu:expert.cpp.obj", "", False, False, False))

    def test_trellis_avx512_rows_are_wide_on_purpose(self):
        self.assertIsNone(G.judge("strata::kernels::cpu::tq_detail::tq_rows_avx512(int, unsigned char const*, "
                                  "unsigned long, int, strata::kernels::cpu::tq_detail::TqAct const* const*, int, "
                                  "float* const*, int, int)", "tq_avx512.cpp.o", "", False, False, True))
        # a synthetic MSVC-style decoration: judge() reads only the leading name, not the signature
        self.assertIsNone(G.judge("?tq_rows_avx512@tq_detail@cpu@kernels@strata@@YAXHPEBE_KH@Z",
                                  "strata_kernels_cpu:tq_avx512.cpp.obj", "", False, False, False))
        # the shared tq_states.hpp helpers must never serve other units as a wide COMDAT copy
        why = G.judge("strata::kernels::cpu::tq_detail::tq_half_to_float(unsigned short)", "tq_avx512.cpp.o", "",
                      False, True, True)
        self.assertIn("weak", why)

    def test_the_probe_must_not_be_wide(self):
        self.assertIsNotNone(G.judge("?cpu_features@cpu@kernels@strata@@YA?AUCpuFeatures@123@XZ",
                                     "strata_kernels_cpu:expert.cpp.obj", "", False, False, False))
        self.assertIsNotNone(G.judge("strata::kernels::cpu::cpu_features()", None, "", False, False, True))
        self.assertIsNotNone(G.judge("strata::kernels::cpu::iq512_supported(int)", None, "", False, False, True))

    def test_weak_symbol_is_a_violation_even_in_a_wide_file(self):
        why = G.judge("std::vector<float, std::allocator<float> >::_M_realloc_insert(float const&)", None, "", False,
                      True, True)
        self.assertIn("weak", why)

    def test_wide_code_in_another_object_is_a_violation(self):
        self.assertIsNotNone(G.judge("?f@@YAXXZ", "strata_kernels_cpu:pool.cpp.obj", "", False, False, False))

    def test_msvc_vectorizer_guard_is_accepted(self):
        self.assertIsNone(G.judge("?bench@@YAHXZ", "iq_avx2_parity.cpp.obj", "", True, False, False))


class InitialiserRule(unittest.TestCase):
    """run_elf's #2324 check: a static initialiser holding AVX-class code in a listed object is a violation."""

    NM = "tq_avx2.cpp.o:\n0000000000000000 T tq_rows_avx2\n"
    DIS = ("tq_avx2.cpp.o:     file format elf64-x86-64\n"
           "<_GLOBAL__sub_I_tq_table>:\n"
           "\tc5 f8 10 05 00 00 00 00\tvmovups 0x0(%rip),%xmm0\n")

    def run_guard(self, nm, dis):
        import contextlib, io, unittest.mock as mock
        def fake_run(cmd, **kw):
            return mock.Mock(stdout=nm if cmd[0] == "nm" else dis)
        out = io.StringIO()
        with mock.patch.object(G.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(G.shutil, "which", return_value="/usr/bin/true"), \
                contextlib.redirect_stdout(out):
            rc = G.run_elf("fake.a")
        return rc, out.getvalue()

    def test_initialiser_in_tq_avx2_is_flagged(self):
        rc, out = self.run_guard(self.NM, self.DIS)
        self.assertNotEqual(rc, 0)
        self.assertIn("#2324", out)
        self.assertIn("tq_avx2.cpp.o", out)

    def test_plain_function_in_tq_avx2_is_not(self):
        dis = self.DIS.replace("_GLOBAL__sub_I_tq_table", "tq_detail::tq_rows_avx2(int)")
        rc, out = self.run_guard(self.NM, dis)
        self.assertEqual(rc, 0)
        self.assertNotIn("#2324", out)


if __name__ == "__main__":
    unittest.main()
