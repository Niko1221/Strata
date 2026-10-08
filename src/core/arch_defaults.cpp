// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

namespace {
// gfx1150 (Strix Point) takes the gfx1151 table less STRATA_HCD_EXACT: that kernel copies gfx1151's hipBLASLt solution
// 1176 / 1177, and the gfx1150 table (tools/hip/gfx1150-hipblaslt-100401.txt) picks other solutions, so there it would
// only say that it cannot run.  The rest gave the same greedy answers with and without them on one 890M
// (bench/results/2026-10-07-community-gfx1150).
bool point(const char* gcn_arch) { return strata::kernels::gfx_arch_is_gfx1150(gcn_arch); }
const char* opt_out(const char* gcn_arch) { return point(gcn_arch) ? "STRATA_GFX1150_DEFAULTS" : "STRATA_GFX1151_DEFAULTS"; }
}  // namespace

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch) && !point(gcn_arch)) return t;
    const char* off = std::getenv(opt_out(gcn_arch));
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
    if (point(gcn_arch))
        t.erase(std::remove_if(t.begin(), t.end(), [](const auto& kv) { return kv.first == "STRATA_HCD_EXACT"; }), t.end());
    return t;
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch) {
    std::vector<std::string> set;
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns them off; a switch you set is "
                             "kept): ", point(gcn_arch) ? "gfx1150 (Strix Point)" : "gfx1151 (Strix Halo)", set.size(),
                     opt_out(gcn_arch));
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

}  // namespace strata::core
