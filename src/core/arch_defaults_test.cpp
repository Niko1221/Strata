// src/core/arch_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/arch_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// Every variable the gfx12 checks below read or set, so that an inherited setting cannot change a result.
static const char* const kGfx12Vars[] = {
    "STRATA_GFX12_DEFAULTS", "STRATA_GFX12_SPLIT_DECODE", "STRATA_GFX1151_DEFAULTS",
    "STRATA_GDN_HEAD", "STRATA_GDN_PP", "STRATA_GDN_CONVL2", "STRATA_GDN_NOY", "STRATA_CVEC_FUSE",
    "STRATA_Q6_PACKED", "STRATA_MMVF_ROWS", "STRATA_ATTN_LANECELL", "STRATA_EXPERT_V2", "STRATA_TSUM", "STRATA_LFUSE",
    "STRATA_GDN_SPLIT", "STRATA_QFUSE", "STRATA_PLE_BATCH",
    "STRATA_Q8_PACKED", "STRATA_SH_STREAM", "STRATA_HCD_EXACT", "STRATA_PF_PAD",
};
static void clean() { for (const char* k : kGfx12Vars) unset(k); }
// How many of kGfx12Vars are set in the environment.
static int n_set() {
    int n = 0;
    for (const char* k : kGfx12Vars) n += std::getenv(k) != nullptr;
    return n;
}

// What generate does at start-up on a HIP build: the arch table for GPU 0, then the opt-in decode set when the split
// qualifies.  Returns the names set.
static std::vector<std::string> start_up(const std::vector<std::string>& archs, bool measured_os) {
    using namespace strata::core;
    std::vector<std::string> set = apply_arch_defaults(archs.empty() ? "gfx1201" : archs[0].c_str());
    if (gfx12_split_decode(archs, measured_os) == Gfx12SplitDecode::on)
        for (const auto& n : apply_gfx12_split_decode()) set.push_back(n);
    return set;
}

int main() {
    using namespace strata::core;
    clean();
    // other architectures (and CUDA, which passes an empty name): nothing
    CHECK(arch_default_env("gfx1100").empty());
    CHECK(arch_default_env("gfx1150").empty());
    CHECK(arch_default_env("gfx1152").empty());
    CHECK(arch_default_env("gfx11510").empty());   // not a prefix match
    CHECK(arch_default_env("").empty());
    CHECK(arch_default_env(nullptr).empty());
    // gfx1151, plain and with its feature suffix: the table, and no bit-changing switch in it
    for (const char* a : {"gfx1151", "gfx1151:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() >= 10);
        for (const auto& kv : t) {
            CHECK(kv.first.rfind("STRATA_", 0) == 0);
            for (const char* bits : {"STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                     "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != bits);
        }
    }
    // gfx12: the five prompt switches only, no decode switch, no bit-changing one
    unset("STRATA_GFX12_DEFAULTS");
    for (const char* a : {"gfx1200", "gfx1201", "gfx1201:sramecc-:xnack-"}) {
        const auto t = arch_default_env(a);
        CHECK(t.size() == 5);
        for (const auto& kv : t) CHECK(kv.first != "STRATA_HCD_EXACT");
        for (const auto& kv : t) {
            CHECK(kv.first != "STRATA_Q8_PACKED" && kv.first != "STRATA_SH_STREAM" && kv.first != "STRATA_PF_GEMM" &&
                  kv.first != "STRATA_PF_FUSED");
        }
    }
    CHECK(arch_default_env("gfx12010").empty());
    put("STRATA_GFX12_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1201").empty());
    unset("STRATA_GFX12_DEFAULTS");
    // gfx12 layer split, opt-in.  The decision: every GPU of the resolved split gfx12, two or more of them, a measured OS
    {
        using V = std::vector<std::string>;
        using D = Gfx12SplitDecode;
        const V two{"gfx1201", "gfx1201"};
        std::string why;
        clean();
        // no flag: a qualifying split only gets the hint; the table is the five prompt switches either way
        CHECK(gfx12_split_decode(two, true, &why) == D::hint && why.empty());
        CHECK(gfx12_split_decode(V{"gfx1200", "gfx1201:sramecc-:xnack-"}, true) == D::hint);
        CHECK(gfx12_split_decode(V{"gfx1201", "gfx1201", "gfx1201", "gfx1201"}, true) == D::hint);
        // no hint where the flag would do nothing: one GPU, mixed architectures, an unmeasured OS
        CHECK(gfx12_split_decode(V{}, true, &why) == D::off && !why.empty());
        CHECK(gfx12_split_decode(V{"gfx1201"}, true) == D::off);
        CHECK(gfx12_split_decode(V{"gfx1201", "gfx1100"}, true, &why) == D::off && why.find("gfx1100") != std::string::npos);
        CHECK(gfx12_split_decode(V{"gfx1100", "gfx1201"}, true) == D::off);
        CHECK(gfx12_split_decode(V{"gfx1201", "gfx1151"}, true) == D::off);
        CHECK(gfx12_split_decode(V{"gfx1201", "gfx1201", "gfx1030"}, true) == D::off);
        CHECK(gfx12_split_decode(V{"gfx1201", "gfx12010"}, true) == D::off);   // not a prefix match
        CHECK(gfx12_split_decode(V{"gfx1201", "unknown"}, true) == D::off);    // a GPU whose name could not be read
        CHECK(gfx12_split_decode(V{"gfx1201", ""}, true) == D::off);
        CHECK(gfx12_split_decode(two, false, &why) == D::off && !why.empty());
        // the flag: on for a qualifying split only
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(gfx12_split_decode(two, true, &why) == D::on && why.empty());
        CHECK(gfx12_split_decode(V{"gfx1200", "gfx1201"}, true) == D::on);
        CHECK(gfx12_split_decode(V{}, true) == D::off);
        CHECK(gfx12_split_decode(V{"gfx1201"}, true) == D::off);
        CHECK(gfx12_split_decode(V{"gfx1201", "gfx1100"}, true) == D::off);
        CHECK(gfx12_split_decode(two, false) == D::off);                       // Windows
        put("STRATA_GFX12_DEFAULTS", "0");                                       // the whole gfx12 table off wins
        CHECK(gfx12_split_decode(two, true, &why) == D::off && why == "STRATA_GFX12_DEFAULTS=0");
        clean();
        // any value other than exactly "1": off, and no hint (the user chose)
        for (const char* other : {"0", "", "10", "1garbage", " 1", "true"}) {
            put("STRATA_GFX12_SPLIT_DECODE", other);
#if defined(_WIN32)
            if (other[0] == '\0') {   // _putenv_s with an empty value removes the variable: unset, so the hint
                CHECK(gfx12_split_decode(two, true, &why) == D::hint);
                continue;
            }
#endif
            CHECK(gfx12_split_decode(two, true, &why) == D::off &&
                  why == std::string("STRATA_GFX12_SPLIT_DECODE=") + other);
        }
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(gfx12_split_decode(two, true, &why) == D::on && why.empty());
        clean();
        put("STRATA_GFX12_DEFAULTS", "0");
        CHECK(gfx12_split_decode(two, true) == D::off);
        clean();
        // the flag does not change the arch table itself: still the five prompt switches
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(arch_default_env("gfx1201").size() == 5);
        clean();
    }
    // the decode set: nine switches of the gfx1151 table, not STRATA_Q8_PACKED (inert on the measured pack), not
    // STRATA_SH_STREAM (not measured on a gfx12 split), none of the prompt five, none of the opt-in rounding-level ones
    {
        const auto dec = gfx12_split_decode_env();
        const auto g1151 = arch_default_env("gfx1151");
        CHECK(dec.size() == 9);
        for (const auto& kv : dec) {
            CHECK(kv.second == "1");
            bool in1151 = false;
            for (const auto& g : g1151) in1151 = in1151 || g == kv;
            CHECK(in1151);
            for (const auto& p : arch_default_env("gfx1201")) CHECK(kv.first != p.first);
            for (const char* no : {"STRATA_Q8_PACKED", "STRATA_SH_STREAM", "STRATA_VERIFY_QDEDUP", "STRATA_HCD_EXACT", "STRATA_PF_PAD",
                                   "STRATA_PF_GEMM", "STRATA_PF_FUSED", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_HIP_WMMA",
                                   "STRATA_SELECT_WMMA", "STRATA_HC_Q8", "STRATA_PF_HCDOWN", "STRATA_WMMA_GEMM"})
                CHECK(kv.first != no);
        }
    }
    // start-up, from a clean environment each time: what ends up set
    {
        using V = std::vector<std::string>;
        const V two{"gfx1201", "gfx1201"};
        clean();
        CHECK(start_up(two, true).size() == 5 && n_set() == 5);                   // gfx12 split, no flag: the 0.1.43 table
        clean();
        CHECK(start_up(V{"gfx1201"}, true).size() == 5 && n_set() == 5);          // one GPU, no flag
        clean();
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(start_up(two, true).size() == 14 && n_set() == 15);                 // the flag: 14 switches (+ the flag)
        CHECK(std::getenv("STRATA_Q8_PACKED") == nullptr && std::getenv("STRATA_SH_STREAM") == nullptr);
        clean();
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(start_up(V{"gfx1201"}, true).size() == 5 && n_set() == 6);          // the flag on one GPU
        clean();
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(start_up(V{"gfx1201", "gfx1100"}, true).size() == 5 && n_set() == 6);   // the flag, mixed architectures
        clean();
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        CHECK(start_up(two, false).size() == 5 && n_set() == 6);                  // the flag on Windows
        clean();
        put("STRATA_GFX12_SPLIT_DECODE", "1");
        put("STRATA_GFX12_DEFAULTS", "0");
        CHECK(start_up(two, true).empty() && n_set() == 2);                       // STRATA_GFX12_DEFAULTS=0: nothing
        clean();
    }
    // the decode set keeps a user's setting, whatever it is, and sets the rest
    {
        clean();
        put("STRATA_EXPERT_V2", "0");
        put("STRATA_TSUM", "x");
        const auto set = apply_gfx12_split_decode();
        CHECK(set.size() == 7);
        CHECK(std::getenv("STRATA_EXPERT_V2") != nullptr && std::string(std::getenv("STRATA_EXPERT_V2")) == "0");
        CHECK(std::getenv("STRATA_PLE_BATCH") != nullptr && std::string(std::getenv("STRATA_PLE_BATCH")) == "1");
        for (const auto& n : set) CHECK(n != "STRATA_EXPERT_V2" && n != "STRATA_TSUM");
        clean();
        // all of them set by the user: nothing set, nothing printed
        for (const auto& kv : gfx12_split_decode_env()) put(kv.first.c_str(), "0");
        CHECK(apply_gfx12_split_decode().empty());
        clean();
    }
    // a user's setting wins, whatever it is; the rest are set
    put("STRATA_TSUM", "0");
    unset("STRATA_QFUSE");
    const auto set = apply_arch_defaults("gfx1151");
    CHECK(std::getenv("STRATA_TSUM") != nullptr && std::string(std::getenv("STRATA_TSUM")) == "0");
    CHECK(std::getenv("STRATA_QFUSE") != nullptr && std::string(std::getenv("STRATA_QFUSE")) == "1");
    for (const auto& n : set) CHECK(n != "STRATA_TSUM");
    // the opt-out
    put("STRATA_GFX1151_DEFAULTS", "0");
    CHECK(arch_default_env("gfx1151").empty());
    CHECK(apply_arch_defaults("gfx1151").empty());
    std::printf(fails ? "arch_defaults_test: %d FAILED\n" : "arch_defaults_test: OK\n", fails);
    return fails ? 1 : 0;
}
