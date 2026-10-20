// src/core/nv_defaults.cpp - see include/strata/core/nv_defaults.hpp.
#include "strata/core/nv_defaults.hpp"

#include "strata/core/emulate.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {
bool off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}
}  // namespace

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor) {
    std::vector<std::pair<std::string, std::string>> t;
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    if (mj == 8 && mn == 6) {
        if (off("STRATA_SM86_PERF")) return t;
        t = {
            // 148-nv, RTX 3060 + RTX A4000 (IQ3_XXS, default flags, 8 pairs): the stack +9.6% / +14.6% (8/8 each), identity gate IDENTICAL.
            // Each is in because removing it from the full stack cost speed on a card (drop-one, 4 rounds) and saved none on the other.
            {"STRATA_ADAPT_NOWAIT", "1"},        // A4000 -6.3 points when removed (with ADAPT_LAG), 3060 -0.8
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_CPU_IQ_ALL_EXACT1", "1"},   // A4000 -3.3, 3060 -0.6
            {"STRATA_SIDE_MIXER", "2"},          // A4000 -1.2, 3060 -2.3
            {"STRATA_ATTN_DAILY", "1"},          // -1.2, -1.3
            {"STRATA_GR_UP75", "1"},             // -0.9, -1.8
            {"STRATA_GU_SWIGLU", "1"},           // -0.8, -1.4
            {"STRATA_QSA_ROWS", "1"},            // -0.5, -1.2
            {"STRATA_B6_MMVQ_ROWS", "1"},        // A4000 0, 3060 -1.7
        };
    } else if (mj == 7 && mn == 0) {
        if (off("STRATA_SM70_PERF")) return t;
        t = {
            // Dual V100 PCIe 32 GB, IQ3_S, MTP4: initial six-switch stack +5.8%/+7.6% decode;
            // seven additions +4.7%/+4.8% against that stack. Ten decoded texts matched.
            // Combined-stack evidence, not a token-id gate: bench/results/2026-10-20-v100-sm70-defaults/.
            {"STRATA_SIDE_MIXER", "2"},
            {"STRATA_GR_UP75", "1"},
            {"STRATA_B6_MMVQ_ROWS", "1"},
            {"STRATA_GU_SWIGLU", "1"},
            {"STRATA_QSA_ROWS", "1"},
            {"STRATA_ATTN_DAILY", "1"},
            {"STRATA_SM70_TABLE", "1"},
            {"STRATA_GR_NORM_FAST", "1"},
            {"STRATA_ARGMAX_ROWS_SM70", "1"},
            {"STRATA_QIL", "1"},
            {"STRATA_EXPDEC", "1"},
            {"STRATA_GR_DOWN_TG", "1"},
            {"STRATA_DEQUANT_WIDE", "1"},
        };
    } else if (mj == 6 && mn == 0) {
        if (off("STRATA_SM60_PERF")) return t;
        t = {
            // 148-nv, Tesla P100 (IQ3_XXS, default flags, 8 pairs): the stack +17.8% (8/8), identity gate IDENTICAL.
            {"STRATA_ADAPT_NOWAIT", "1"},        // -10 points when removed (with ADAPT_LAG)
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_SIDE_MIXER", "2"},          // -1.8
            {"STRATA_FUSE_WAIT", "2"},           // -1.6
            {"STRATA_B6_MMVQ_ROWS", "1"},        // -1.3
            {"STRATA_FUSE_DECODE", "1"},         // -1.3
            {"STRATA_GR_UP75", "1"},             // -1.2
        };
    }
    return t;
}

std::vector<std::string> apply_nv_defaults(int major, int minor) {
    std::vector<std::string> set;
    for (const auto& kv : nv_default_env(major, minor)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const int mj = strata::cc_major_of(major);
        const char* arch = mj == 8 ? "sm_86 (Ampere)" : mj == 7 ? "sm_70 (Volta)" : "sm_60 (Pascal)";
        const char* opt_out = mj == 8 ? "STRATA_SM86_PERF" : mj == 7 ? "STRATA_SM70_PERF" : "STRATA_SM60_PERF";
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns them off; a switch you set is kept): ",
                     arch, set.size(), opt_out);
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

}  // namespace strata::core
