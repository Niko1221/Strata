// Unit tests for strata-inkling CUDA automatic layer split (strata/core/ink_layer_split.hpp, #7030).
// Covers asymmetric 2-GPU (both orders, P2P on/off), single-card fit, equal 2-GPU, 3-GPU, 4-GPU,
// per-stage Stage-0 KV sizing (HF-NVFP4 18,6 vs 17,7), and tight-VRAM host overflow minimization.
#include "strata/core/ink_layer_split.hpp"

#include <cstdio>
#include <vector>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr size_t MiB = 1ull << 20;
constexpr size_t GiB = 1ull << 30;

InkLayerSplitIn make_inkling_base(size_t exp_per_layer, size_t trk_per_layer, int max_ctx = 32768) {
    InkLayerSplitIn in;
    in.n_phys = 24;
    in.core_start = 6;
    in.core_end = 18;
    in.num_passes = 3;
    in.is_swa.assign(24, 1);
    for (int l = 5; l < 24; l += 6) in.is_swa[l] = 0;  // global layers: 5, 11, 17, 23 (8 visits total)
    in.layer_exp_bytes.assign(24, exp_per_layer);
    in.layer_trunk_bytes.assign(24, trk_per_layer);
    // FP8 KV: 2048 B / pos / visit (1024 K + 1024 V), swa_ring = 1536, ShortConv = 3 * 6656 * 4 = 79872 B / visit
    in.stage0_swa_kv_per_visit  = 1536ull * 2048ull;
    in.stage0_glob_kv_per_visit = static_cast<size_t>(max_ctx) * 2048ull;
    in.stage0_sconv_per_visit   = 3ull * 6656ull * 4ull;
    in.stage0_ws_reserve_bytes  = 640 * MiB + 128 * MiB;
    in.stage0_wddm_floor_bytes  = 512 * MiB;
    in.stage0_base_reserve_bytes = 1280 * MiB;
    in.stage0_extra_reserve_bytes = 960 * MiB;
    return in;
}
}  // namespace

int main() {
    const double spd_5060ti = ink_stage_rel_speed(14000000, 128, 120);  // ~448 GB/s * 1.15
    const double spd_3060   = ink_stage_rel_speed(7500000, 192, 86);    // ~360 GB/s * 1.00

    // 1. Asymmetric 2-GPU (5060 Ti 16 GB + 3060 12 GB, SQ-15 32K, ~11.4 GiB experts):
    //    Stage 0 holds L0..22 (all 12 looped core layers L6..17!), Stage 1 holds tail coda layers only; 0 overflow, 1 hop.
    {
        auto in = make_inkling_base(485 * MiB, 85 * MiB, 32768);
        in.n_stages = 2;
        in.stage_speed = {spd_5060ti, spd_3060};
        in.stage_p2p[0][1] = in.stage_p2p[1][0] = true;
        in.stage0_free_bytes = 13200 * MiB;  // ~13.2 GiB free after trunk upload
        in.peer_room_bytes = {0, 10200 * MiB};
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.counts[0] >= 18 && out.counts[0] < 24);
        CHECK(out.counts[0] + out.counts[1] == 24);
        CHECK(out.hops == 1);
        CHECK(out.visits[0] >= 42 && out.visits[1] <= 6);
        std::printf("case 1 (5060Ti+3060 SQ-15 P2P=1): %d,%d visits=%d+%d hops=%d ovf=%zu\n",
                    out.counts[0], out.counts[1], out.visits[0], out.visits[1], out.hops, out.overflow_bytes);

        // Same with P2P=0: still keeps core loop on 5060 Ti with 1 hop.
        in.stage_p2p[0][1] = in.stage_p2p[1][0] = false;
        const auto out_nop2p = ink_plan_layer_split(in);
        CHECK(out_nop2p.overflow_bytes == 0);
        CHECK(out_nop2p.counts[0] >= 18 && out_nop2p.hops == 1);
    }

    // 2. Reversed asymmetric 2-GPU (3060 12 GB first + 5060 Ti 16 GB second, SQ-15 32K):
    //    Keeps looped core L6..17 on the faster peer (5060 Ti), only prelude layers on 3060; 0 overflow, 1 hop.
    {
        auto in = make_inkling_base(485 * MiB, 85 * MiB, 32768);
        in.n_stages = 2;
        in.stage_speed = {spd_3060, spd_5060ti};
        in.stage_p2p[0][1] = in.stage_p2p[1][0] = true;
        in.stage0_free_bytes = 9300 * MiB;
        in.peer_room_bytes = {0, 14000 * MiB};
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.counts[0] <= 6 && out.counts[1] >= 18);
        CHECK(out.hops == 1);
        CHECK(out.visits[1] >= 42);
        std::printf("case 2 (3060+5060Ti SQ-15 rev): %d,%d visits=%d+%d hops=%d ovf=%zu\n",
                    out.counts[0], out.counts[1], out.visits[0], out.visits[1], out.hops, out.overflow_bytes);
    }

    // 3. Single-card fit (5060 Ti 16 GB + 3060 12 GB, SQ-10 32K, ~8.0 GiB experts):
    //    Whole model + full 32K KV fits on Stage 0 -> chooses single card (24,0), 0 hops.
    {
        auto in = make_inkling_base(340 * MiB, 75 * MiB, 32768);
        in.n_stages = 2;
        in.stage_speed = {spd_5060ti, spd_3060};
        in.stage0_free_bytes = 13500 * MiB;
        in.peer_room_bytes = {0, 10200 * MiB};
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.counts[0] == 24 && out.counts[1] == 0);
        CHECK(out.hops == 0);
        std::printf("case 3 (5060Ti+3060 SQ-10 single): %d,%d hops=%d\n", out.counts[0], out.counts[1], out.hops);
    }

    // 4. Equal 2-GPU pair (2x RTX 3060 12 GB, SQ-15 32K):
    //    Does not fit on one card; chooses a 1-hop boundary outside L6..17 (e.g. 6,18) instead of 3 hops inside the loop.
    {
        auto in = make_inkling_base(485 * MiB, 85 * MiB, 32768);
        in.n_stages = 2;
        in.stage_speed = {spd_3060, spd_3060};
        in.stage_p2p[0][1] = in.stage_p2p[1][0] = false;
        in.stage0_free_bytes = 9300 * MiB;
        in.peer_room_bytes = {0, 10600 * MiB};  // fits 18 layers (18 * 570 + 864 MiB ~ 11.1 GiB -> try 11.2 GiB)
        in.peer_room_bytes[1] = 11300 * MiB;
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.hops == 1);
        CHECK(out.counts[0] == 6 && out.counts[1] == 18);
        std::printf("case 4 (equal 2x 12GB SQ-15): %d,%d hops=%d ovf=%zu\n",
                    out.counts[0], out.counts[1], out.hops, out.overflow_bytes);
    }

    // 5. Stage-0 per-visit KV awareness (HF-NVFP4 ~600 MiB/layer on 5060 Ti + 3060):
    //    With Stage-0-only KV (35 SWA + 7 global for L0..17 instead of 40 + 8), 18,6 (1 hop) fits where a 48-visit reserve would force 17,7 (3 hops).
    {
        auto in = make_inkling_base(600 * MiB, 140 * MiB, 32768);
        in.n_stages = 2;
        in.stage_speed = {spd_5060ti, spd_3060};
        in.stage_p2p[0][1] = in.stage_p2p[1][0] = true;
        // 18 layers experts = 10,800 MiB. For 18 layers: 35*3 MiB + 7*64 MiB + 42*0.076 MiB = 556 MiB KV + 768 MiB ws + 512 MiB wddm = 1,836 MiB.
        // Total needed for 18 layers = 12,636 MiB. Full 48-visit KV would need +79 MiB (12,715 MiB).
        // Set free VRAM = 12,680 MiB so per-stage KV fits 18,6 (1 hop), whereas 48-visit KV would fail 18 and pick 17,7 (3 hops).
        in.stage0_free_bytes = 12680 * MiB;
        in.peer_room_bytes = {0, 10200 * MiB};
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.counts[0] == 18 && out.counts[1] == 6);
        CHECK(out.hops == 1);
        std::printf("case 5 (per-stage KV 18,6 vs 17,7): %d,%d hops=%d bud=%.2f GiB\n",
                    out.counts[0], out.counts[1], out.hops, static_cast<double>(out.stage0_budget_bytes) / GiB);
    }

    // 6. 3-GPU case (8 GB + 8 GB + 8 GB, SQ-15 ~11.4 GiB): spreads across 3 cards with 0 overflow and minimal hops.
    {
        auto in = make_inkling_base(485 * MiB, 85 * MiB, 16384);
        in.n_stages = 3;
        in.stage_speed = {300.0, 300.0, 300.0};
        in.stage0_free_bytes = 5600 * MiB;
        in.peer_room_bytes = {0, 6200 * MiB, 6200 * MiB};
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.counts[0] + out.counts[1] + out.counts[2] == 24);
        CHECK(out.counts[1] > 0 && out.counts[2] > 0);
        std::printf("case 6 (3x 8GB SQ-15): %d,%d,%d hops=%d ovf=%zu\n",
                    out.counts[0], out.counts[1], out.counts[2], out.hops, out.overflow_bytes);
    }

    // 7. 4-GPU case with mixed speeds (fastest card at stage 1): puts looped core on fastest stage if room allows.
    {
        auto in = make_inkling_base(485 * MiB, 85 * MiB, 16384);
        in.n_stages = 4;
        in.stage_speed = {250.0, 600.0, 250.0, 250.0};
        in.stage0_free_bytes = 5000 * MiB;
        in.peer_room_bytes = {0, 8000 * MiB, 5000 * MiB, 5000 * MiB};
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes == 0);
        CHECK(out.counts[0] + out.counts[1] + out.counts[2] + out.counts[3] == 24);
        CHECK(out.visits[1] >= 36);  // stage 1 (fastest) gets the entire looped core
        CHECK(out.counts[2] >= out.counts[3]);  // earlier peer stage filled before later tied stage
        std::printf("case 7 (4-GPU fast stage1): %d,%d,%d,%d visits=%d+%d+%d+%d hops=%d\n",
                    out.counts[0], out.counts[1], out.counts[2], out.counts[3],
                    out.visits[0], out.visits[1], out.visits[2], out.visits[3], out.hops);
    }

    // 8. Tight VRAM overflow case: when all cards combined cannot hold every expert, minimizes Stage 0 host overflow.
    {
        auto in = make_inkling_base(600 * MiB, 100 * MiB, 32768);
        in.n_stages = 2;
        in.stage_speed = {spd_5060ti, spd_3060};
        in.stage0_free_bytes = 8000 * MiB;
        in.peer_room_bytes = {0, 5000 * MiB};  // fits at most 6 layers (~4.5 GiB)
        const auto out = ink_plan_layer_split(in);
        CHECK(out.overflow_bytes > 0);
        CHECK(out.counts[1] == 6 && out.counts[0] == 18);  // maximizes peer usage to minimize host overflow
        std::printf("case 8 (tight overflow): %d,%d ovf=%.0f MiB hops=%d\n",
                    out.counts[0], out.counts[1], static_cast<double>(out.overflow_bytes) / MiB, out.hops);
    }

    if (failed) {
        std::fprintf(stderr, "ink_layer_split_test: %d failed\n", failed);
        return 1;
    }
    std::printf("ink_layer_split_test: ok\n");
    return 0;
}
