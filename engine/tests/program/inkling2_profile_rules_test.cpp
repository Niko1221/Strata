// Ported from strata-inkling2 by kuraneko1 (_ryu15_)
// https://github.com/kuraneko1/Strata (MIT License)
// See docs/third-party/inkling2-kuraneko/CREDITS.md
// tests/program/inkling2_profile_rules_test.cpp - CPU-only unit tests for strata::inkling2::resolve_inkling2_profile.
// Locks down V100 (sm_70 32GB), RTX 4070 (sm_89 12GB @ 65K/131K/262K), unmeasured Generic GPUs, and explicit CLI overrides.
#include "strata/inkling2/profile_rules.hpp"

#include <cstdio>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++fails;
    }
}

constexpr size_t GiB = (size_t) 1 << 30;
constexpr size_t MiB = (size_t) 1 << 20;

// Inkling-2-Mini-UD-Q4_K_M exact expert footprint: 12 core layers = 6.45 GiB, all 24 layers = 13.76 GiB.
constexpr size_t kTier1Bytes = (size_t) (6.45 * (double) GiB);
constexpr size_t kAllExpBytes = (size_t) (13.76 * (double) GiB);

strata::inkling2::ProfileInput v100_32g(int max_ctx, bool draft = true) {
    strata::inkling2::ProfileInput in;
    in.cc_major = 7;
    in.cc_minor = 0;
    in.sm_count = 80;
    in.vram_total_bytes = 32 * GiB;
    in.vram_free_bytes = draft ? (size_t) (29.1 * (double) GiB) : (size_t) (29.8 * (double) GiB);
    in.max_ctx = max_ctx;
    in.kv_q8 = true;
    in.has_draft = draft;
    in.tier1_experts_bytes = kTier1Bytes;
    in.all_experts_bytes = kAllExpBytes;
    return in;
}

strata::inkling2::ProfileInput rtx4070_12g(int max_ctx, bool draft = false) {
    strata::inkling2::ProfileInput in;
    in.cc_major = 8;
    in.cc_minor = 9;
    in.sm_count = 46;
    in.vram_total_bytes = 12 * GiB;
    // Measured free VRAM on RTX 4070 after dense model (~2.0 GiB) + desktop (~0.2 GiB) load: ~9.85 GiB.
    in.vram_free_bytes = (size_t) (9.85 * (double) GiB);
    in.max_ctx = max_ctx;
    in.kv_q8 = true;
    in.has_draft = draft;
    in.tier1_experts_bytes = kTier1Bytes;
    in.all_experts_bytes = kAllExpBytes;
    return in;
}
}  // namespace

int main() {
    using namespace strata::inkling2;

    // 1. V100 32GB @ 819,200 ctx (DSpark ON): must stay sm70 + full_vram with want_sp=7 (6 exact waves on 80 SMs) & d_hits=off!
    {
        const auto in = v100_32g(819200, true);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.arch.id == ArchId::Sm70Volta, "V100 selects Sm70Volta arch");
        check(p.regime == VramRegime::FullVram, "V100 819K selects FullVram");
        check(p.chunk == 1024, "V100 819K keeps chunk=1024");
        check(p.full_kv_init == 819200, "V100 819K allocates full 819K KV upfront");
        check(p.want_sp == 7, "V100 819K preserves want_sp=7 (best_sp=5 on 80 SMs)");
        check(!p.use_d_hits, "V100 FullVram skips d_hits atomics");
        check(p.pp_stage_cnt_pref == 0, "V100 FullVram allocates no pp_stage");
        check(p.force_bf16_x_f16, "V100 enables force_bf16_x_f16");
    }

    // 2. V100 32GB @ 1,048,576 ctx (1M, DSpark OFF): overflows FullVram -> drops cleanly to CoreResidentHybrid
    {
        const auto in = v100_32g(1048576, false);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.regime == VramRegime::CoreResidentHybrid, "V100 1M selects CoreResidentHybrid");
        check(p.chunk == 1024, "V100 1M keeps chunk=1024");
        check(p.full_kv_init == 1048576, "V100 1M allocates 1M KV upfront while keeping core 128/128");
        check(p.use_d_hits, "V100 1M hybrid enables d_hits");
    }

    // 3. RTX 4070 12GB @ 65,536 ctx: CoreResidentHybrid (core 128/128 resident, upfront 64K KV, chunk=2048 + 2x64MiB stage)
    {
        const auto in = rtx4070_12g(65536);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.arch.id == ArchId::Sm89Ada, "RTX 4070 selects Sm89Ada arch");
        check(p.regime == VramRegime::CoreResidentHybrid, "RTX 4070 65K selects CoreResidentHybrid");
        check(p.chunk == 2048, "RTX 4070 65K promotes to chunk=2048 (+20% PP while keeping tier1=128/128)");
        check(p.full_kv_init == 65536, "RTX 4070 65K allocates 65K KV upfront (zero mid-run expansion pause)");
        check(p.want_sp == 4, "RTX 4070 65K reserves 4 splits at T=2048");
        check(p.pp_stage_cnt_pref == 2 && p.pp_stage_mb == 64, "RTX 4070 65K uses 2x64 MiB staging");
        check(p.use_d_hits, "RTX 4070 65K enables d_hits");
        check(!p.force_bf16_x_f16, "RTX 4070 uses native BF16 TC");
    }

    // 4. RTX 4070 12GB @ 131,072 ctx: CoreResidentHybrid (promotes to chunk=2048 when free VRAM fits tier1=128/128,
    //    and safely falls back to chunk=1024 when desktop eats extra VRAM)
    {
        const auto in = rtx4070_12g(131072);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.regime == VramRegime::CoreResidentHybrid, "RTX 4070 131K selects CoreResidentHybrid");
        check(p.chunk == 2048, "RTX 4070 131K promotes to chunk=2048 when tier1=128/128 fits");
        check(p.full_kv_init == 131072, "RTX 4070 131K allocates 131K KV upfront (zero mid-run pause)");
        check(p.want_sp == 4, "RTX 4070 131K reserves 4 splits at T=2048");
        check(p.pp_stage_cnt_pref == 2, "RTX 4070 131K uses 2x64 MiB staging");

        auto in_tight_desk = rtx4070_12g(131072);
        in_tight_desk.vram_free_bytes = (size_t) (9.50 * (double) GiB);
        const auto p_fb = resolve_inkling2_profile(in_tight_desk);
        check(p_fb.regime == VramRegime::CoreResidentHybrid, "RTX 4070 131K tight desktop stays CoreResidentHybrid");
        check(p_fb.chunk == 1024 && p_fb.want_sp == 5, "RTX 4070 131K tight desktop falls back to chunk=1024 to protect tier1=128/128");
    }

    // 5. RTX 4070 12GB @ 262,144 ctx: TightHybrid (auto chunk=2048 halves PCIe streaming, dynamic 32K->262K KV)
    {
        const auto in = rtx4070_12g(262144);
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.regime == VramRegime::TightHybrid, "RTX 4070 262K selects TightHybrid");
        check(p.chunk == 2048, "RTX 4070 262K auto-selects chunk=2048");
        check(p.full_kv_init == 32768, "RTX 4070 262K starts KV at 32K for fast short prompts");
        check(p.want_sp == 4, "RTX 4070 262K reserves 4 splits at T=2048");
        check(p.pp_stage_cnt_pref == 3 && p.pp_stage_mb == 64, "RTX 4070 262K uses 3x64 MiB ring");
    }

    // 6. Explicit CLI --chunk override ALWAYS wins (e.g. user wants Prefill-focused --chunk 2048 at 65K, or --chunk 1024 at 262K)
    {
        auto in65 = rtx4070_12g(65536);
        in65.requested_chunk = 2048;
        const auto p65 = resolve_inkling2_profile(in65);
        check(p65.chunk == 2048 && p65.want_sp == 4, "explicit --chunk 2048 at 65K wins and adjusts want_sp=4");

        auto in262 = rtx4070_12g(262144);
        in262.requested_chunk = 1024;
        const auto p262 = resolve_inkling2_profile(in262);
        check(p262.chunk == 1024 && p262.want_sp == 5, "explicit --chunk 1024 at 262K wins and adjusts want_sp=5");
    }

    // 7. Unmeasured GPU (e.g. RTX 3090 24GB sm_86 @ 262K) uses conservative Generic profile (never guesses unmeasured T=2048)
    {
        ProfileInput in;
        in.cc_major = 8;
        in.cc_minor = 6;
        in.sm_count = 82;
        in.vram_total_bytes = 24 * GiB;
        in.vram_free_bytes = 21 * GiB;
        in.max_ctx = 262144;
        in.tier1_experts_bytes = kTier1Bytes;
        in.all_experts_bytes = kAllExpBytes;
        const auto p = resolve_inkling2_profile(in);
        std::printf("%s\n", format_profile_line(in, p).c_str());
        check(p.arch.id == ArchId::Generic, "sm_86 selects Generic arch profile");
        check(p.regime == VramRegime::FullVram, "24GB @ 262K fits FullVram");
        check(p.chunk == 1024 && p.want_sp == 7 && !p.use_d_hits, "24GB FullVram uses clean 1024/7/off");

        in.vram_total_bytes = 12 * GiB;
        in.vram_free_bytes = (size_t) (9.5 * (double) GiB);
        const auto p_tight = resolve_inkling2_profile(in);
        check(p_tight.regime == VramRegime::TightHybrid, "unmeasured 12GB @ 262K selects TightHybrid");
        check(p_tight.chunk == 1024, "unmeasured sm_86 TightHybrid stays conservatively on chunk=1024");
    }

    // 8. Forced --expert-vram below all_experts_bytes demotes FullVram -> Hybrid cleanly (for parity testing)
    {
        auto in = rtx4070_12g(16384);
        in.expert_vram_forced = true;
        in.expert_vram_cap_bytes = (size_t) (4.46 * (double) GiB);
        const auto p = resolve_inkling2_profile(in);
        check(p.regime == VramRegime::TightHybrid, "forced --expert-vram 4.46 GiB selects TightHybrid");
        check(p.use_d_hits && p.pp_stage_cnt_pref == 3, "forced hybrid keeps d_hits & staging active");
        check(p.full_kv_init == 16384, "16K ctx still allocates 16K KV upfront");
    }

    if (fails > 0) {
        std::printf("FAILED: %d check(s)\n", fails);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
