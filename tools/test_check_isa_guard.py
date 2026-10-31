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

    def test_vex_marker_is_not_read_as_the_mnemonic(self):
        """#795: binutils 2.42 prints AVX-VNNI as "{vex} vpdpbusd ..." - the marker is not the instruction.

        Read naively, the check called a binary holding vpdpbusd clean, which is the one thing it exists to catch.
        """
        self.assertEqual(G.split_instr("{vex} vpdpbusd %ymm8,%ymm10,%ymm6"), ("vpdpbusd", "%ymm8,%ymm10,%ymm6"))
        self.assertTrue(G.is_wide(0xC4, *G.split_instr("{vex} vpdpbusd %ymm0, %ymm1, %ymm2")))
        self.assertEqual(G.split_instr("{evex} vpmullq %zmm0,%zmm1,%zmm2")[0], "vpmullq")
        self.assertEqual(G.split_instr("lock xadd %eax, (%rbx)"), ("lock", "xadd %eax, (%rbx)"))
        self.assertEqual(G.split_instr("{vex}"), ("", ""))


class Policy(unittest.TestCase):
    def test_kernel_in_an_anonymous_namespace_of_a_wide_file_is_fine(self):
        self.assertIsNone(G.judge("??$gu_rows@$0BA@$00@?A0xa1d599f5@cpu@kernels@strata@@YAXXZ",
                                  "strata_kernels_cpu:iq_avx512.cpp.obj", "", False, False, False))
        self.assertIsNone(G.judge("strata::kernels::cpu::(anonymous namespace)::gu_rows<18, 1>(int)", None, "", False,
                                  False, True))

    def test_trellis_rows_entry_point_is_fine(self):
        """#3358: tq_avx512.cpp's rows function is behind cpu_avx512_ok() like the other wide kernels."""
        name = "strata::kernels::cpu::tq_detail::tq_rows_avx512(int, unsigned char const*, unsigned long, int)"
        self.assertIsNone(G.judge(name, "tq_avx512.cpp.o", "", False, False, True))
        self.assertIsNotNone(G.judge(name, "tq_cpu.cpp.o", "", False, False, True))      # the same code elsewhere is not

    def test_object_names_match_themselves_only(self):
        """'native_expert.cpp.o' ends with 'expert.cpp.o' and is not it; a substring test said otherwise."""
        name = "strata::kernels::cpu::(anonymous namespace)::sneak(int)"
        elf = "CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/{}.cpp.o"
        self.assertIsNone(G.judge(name, elf.format("expert"), "", False, False, True))
        self.assertIsNotNone(G.judge(name, elf.format("native_expert"), "", False, False, True))
        # the map writes an internal name with ?A0x in it, which is what makes it internal to the check
        self.assertIsNone(G.judge("?$sneak@$00@?A0x12345678@cpu@kernels@strata@@YAXXZ",
                                  "strata_kernels_cpu:expert.cpp.obj", "", False, False, False))
        self.assertIsNotNone(G.judge("?$sneak@$00@?A0x12345678@cpu@kernels@strata@@YAXXZ",
                                     "strata_kernels_cpu:native_expert.cpp.obj", "", False, False, False))

    def test_nvfp4_avx2_vnni_copy_is_fine(self):
        """the NVFP4 AVX-2 rows' vpdpbusd copy (nvfp4_avx2_rows.inl): internal, in a file with a VNNI arm."""
        name = "void strata::kernels::cpu::(anonymous namespace)::vnni::row_dot<8>(unsigned char const*, int, float*)"
        obj = "CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/nvfp4_avx2.cpp.o"
        self.assertIsNone(G.judge(name, obj, "", False, False, True))
        self.assertIsNotNone(G.judge(name, obj.replace("nvfp4_avx2", "native_expert"), "", False, False, True))

    def test_listed_public_entry_point_is_fine(self):
        self.assertIsNone(G.judge("?act_quant_q8_1@cpu@kernels@strata@@YAXPEBMHAEAUActQ@123@@Z",
                                  "strata_kernels_cpu:expert.cpp.obj", "", False, False, False))

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


if __name__ == "__main__":
    unittest.main()
