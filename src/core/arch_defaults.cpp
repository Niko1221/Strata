// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): only the PROMPT half of the gfx1151 table.  Reported exact (identical answers) on an R9700
        // in #1478 and re-checked bitwise here (notes-142-amd.md); the decode half measured slower on that card, so it stays out
        // (a layer split over gfx12 GPUs can opt in to most of it: gfx12_split_decode below).
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
    return t;
}

namespace {
// Sets each entry of `t` the user has not set (to anything) and returns the names it set.
std::vector<std::string> put_missing(const std::vector<std::pair<std::string, std::string>>& t) {
    std::vector<std::string> set;
    for (const auto& kv : t) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    return set;
}
void print_names(const std::vector<std::string>& set) {
    for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
    std::fprintf(stderr, "\n");
}
}  // namespace

std::vector<std::string> apply_arch_defaults(const char* gcn_arch) {
    const std::vector<std::string> set = put_missing(arch_default_env(gcn_arch));
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        print_names(set);
    }
    return set;
}

// gfx12 layer split, opt-in (STRATA_GFX12_SPLIT_DECODE=1; strata/core/arch_defaults.hpp).  The decode half of the gfx1151
// table measured slower on one R9700 (#1478).  Split over two R9700s, where every expert is resident and the CPU computes no
// misses, these nine measured faster: 2x R9700 32 GB, Linux, ROCm 7.2.4, Qwen3.8-Flash-Next IQ3_S, --layer-split 26 with
// every expert resident; greedy decode 115.0 tok/s against 108.9 / 108.5 for two stock 0.1.43 sessions (+5.8%), sampled
// 106.2 against 101.5 / 101.0 (+4.9%), 3 repetitions per prompt.  On gfx12 only the generated output was checked: greedy
// output identical to stock (4 prompts x 3 runs), seeded sampled output 11 of 12 identical (stock against stock 12 of 12
// in that session; in a later one stock itself gave two answers for one seeded request).  So they are speed switches on
// gfx12, not shown to be bitwise there; docs/AMD_HIP.md has the rest and the report.
// Not taken: STRATA_Q8_PACKED (the measured pack has no Q8_0 matrix it packs, so its effect on gfx12 is unknown) and
// STRATA_SH_STREAM (not measured on a gfx12 split; off on other AMD cards, #816).
Gfx12SplitDecode gfx12_split_decode(const std::vector<std::string>& gpu_archs, bool measured_os, std::string* why) {
    auto off = [&](const std::string& w) { if (why) *why = w; return Gfx12SplitDecode::off; };
    if (why) why->clear();
    const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
    if (off12 != nullptr && off12[0] == '0') return off("STRATA_GFX12_DEFAULTS=0");
    if (gpu_archs.size() < 2) return off("the split runs on one GPU");
    for (const auto& a : gpu_archs)
        if (!strata::kernels::gfx_arch_is_gfx12(a.c_str())) return off("not every GPU of the split is gfx12 (one is " + a + ")");
    if (!measured_os) return off("measured on Linux only");
    const char* v = std::getenv("STRATA_GFX12_SPLIT_DECODE");
    if (v == nullptr) return Gfx12SplitDecode::hint;
    if (std::strcmp(v, "1") == 0) return Gfx12SplitDecode::on;   // exactly "1": "10" or "1x" is another value
    return off(std::string("STRATA_GFX12_SPLIT_DECODE=") + v);
}

std::vector<std::pair<std::string, std::string>> gfx12_split_decode_env() {
    return {
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (a second copy of each output head in VRAM)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one KV cell per thread
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once
    };
}

std::vector<std::string> apply_gfx12_split_decode() {
    const std::vector<std::string> set = put_missing(gfx12_split_decode_env());
    if (!set.empty()) {
        std::fprintf(stderr, "strata: gfx12 layer split: %zu decode speed switches on (STRATA_GFX12_SPLIT_DECODE=1; on gfx12 "
                             "only the generated output was checked, see docs/AMD_HIP.md; a switch you set is kept): ",
                     set.size());
        print_names(set);
    }
    return set;
}

}  // namespace strata::core
