// engine/metal/src/hy4flash_gguf/msl/hy4_flash_kernels.h - the host-side facts about the MSL beside it: which files make the hy4-flash library,
// in which order, and the one frozen tile choice for the dense GEMV.
//
// One list, used by the engine (engine/metal/src/hy4flash_gguf/*.cpp) and by the kernel parity tests, so the two cannot drift.  The
// tests point STRATA_METAL_HY4_FLASH_KERNELS at the source tree.
//
// Order matters: hy4_flash_fp.metal comes FIRST and turns FMA contraction off for the whole translation unit, including
// the shared kernels - units.metal's unit_combine ends every GEMV-shaped kernel here and its reduction order has to
// be the source's, not the compiler's.  hy4_flash_common.metal repeats the pragma so it stays self-contained.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata_hy4_flash_metal {

// Where the hy4-flash MSL sources are: STRATA_METAL_HY4_FLASH_KERNELS, else the path compiled in.
inline std::string kernel_dir() {
    if (const char* e = std::getenv("STRATA_METAL_HY4_FLASH_KERNELS")) return e;
#ifdef STRATA_METAL_HY4_FLASH_KERNELS_DEFAULT
    return STRATA_METAL_HY4_FLASH_KERNELS_DEFAULT;
#else
    return "";
#endif
}

// The library's sources in order.  The shared ones are relative (mtl::Context::kernel_dir resolves them), hy4-flash's own
// are absolute.
/// HY4_FLASH_PREFILL_MOE_MM resolved to a name: "0", "f16", "f32" (the default) or "f32x2".  THE one place the string is
/// parsed, so the kernel list, forward.cpp's mode and the prefill snapshot's binding cannot disagree.  An
/// unrecognised value resolves to the default here and is rejected with a message in forward.cpp (which runs
/// first), so a typo never silently selects something.
inline const char* prefill_moe_mm_name() {
    static const std::string v = [] {
        const char* e = std::getenv("HY4_FLASH_PREFILL_MOE_MM");
        const std::string s = e != nullptr ? e : "f32";
        if (s == "0" || s == "off" || s == "gemv") return std::string("0");
        if (s == "f16") return std::string("f16");
        if (s == "f32x2") return std::string("f32x2");
        return std::string("f32");
    }();
    return v.c_str();
}
inline bool prefill_moe_mm_on() { return std::string(prefill_moe_mm_name()) != "0"; }

inline std::vector<std::string> hy4_flash_kernel_files() {
    const std::string d = kernel_dir() + "/";
    std::vector<std::string> f{d + "hy4_flash_fp.metal", "quant.metal", "units.metal", "gemv.metal",
                               d + "hy4_flash_common.metal", d + "hy4_flash_norm.metal", d + "hy4_flash_rope.metal",
                               d + "hy4_flash_act.metal", d + "hy4_flash_hc.metal", d + "hy4_flash_moe.metal",
                               d + "hy4_flash_router.metal", d + "hy4_flash_indexer.metal", d + "hy4_flash_attn.metal",
                               d + "hy4_flash_attn_pf.metal", d + "hy4_flash_engram.metal", d + "hy4_flash_gemv_q8.metal",
                               d + "hy4_flash_mm.metal",
                               // UNCONDITIONAL, and that is the point: a file that is only compiled when a
                               // feature is on is a file whose gate can pass while it is absent.  These
                               // kernels use plain MSL and simdgroup only - no MetalPerformancePrimitives -
                               // so there is no runtime capability to gate them on.
                               d + "hy4_flash_vision.metal"};
    // The prefill grouped GEMM is the only hy4-flash kernel file that needs Metal 4's MetalPerformancePrimitives (the
    // dense prefill MMA in hy4_flash_mm.metal uses simdgroup_matrix, which needs no framework), and the whole list is
    // compiled as ONE library at Forward::open - so on a runtime without MPP it would take DECODE down with it.
    // HY4_FLASH_PREFILL_MOE_MM=0 leaves it out entirely, which is that runtime's escape hatch.  With the GEMM selected,
    // a missing MPP is a loud compile error, as it should be.
    if (prefill_moe_mm_on()) f.push_back(d + "hy4_flash_moe_mm.metal");
    return f;
}

// Prepended to the source.  Empty: ggml-common.h is wanted (quant.metal's grids), and nothing is configured by a
// define - the window size is a function constant and the shapes are arguments.
inline std::string hy4_flash_prelude() { return ""; }

// ---- the dense GEMV, f32 activations ------------------------------------------------------------------------------
// gemv.metal holds acc[NR][8], so one dispatch covers at most 8 tokens; prefill therefore runs ceil(T / 8) of them,
// which is arithmetically what decode does for the same position.
inline constexpr int kHy4FlashGemvMaxTok = 8;

// GGML type ids of the dense weights hy4-flash reads with f32 activations.  Q8_0 (attn projections, token_embd), F32
// (hc_fn in this checkpoint) and BF16 (the output head) all read units.metal's RAW x plane, so no xprep pass is
// needed and the activation stays f32 end to end.  Q3_K / Q4_K need the prepared planes instead.
inline bool gemv_f32_raw_x(int ggml_type) { return ggml_type == 8 || ggml_type == 0 || ggml_type == 30; }
inline int gemv_unit(int ggml_type) { return (ggml_type == 0 || ggml_type == 30) ? 16 : 32; }

struct GemvTile {
    int lpr, nr, sg, ks;   // gemv<F, LPR, NR, SG, KS>
    int rows_per_tg() const { return (sg / ks) * (32 / lpr) * nr; }
    std::string name(int ggml_type) const {
        return "gemv_" + std::to_string(ggml_type) + "_" + std::to_string(lpr) + "_" + std::to_string(nr) + "_" +
               std::to_string(sg) + "_" + std::to_string(ks);
    }
    // The thread group shape the engine dispatches for this (logical) tile.  Q8_0's wide tile {32, 2, 8, 2} runs
    // with SG 16: with LPR and KS held, SG only sets how many row slots a thread group has (RS = SG / KS) - a row's
    // K halves keep ks = 0 / 1, the same u0 / u1 and the same k-ordered merge - so the rows' bits are the logical
    // tile's.  hy4_kernel_parity's gemv_q8 cases compare it with gemv on the logical tile.
    GemvTile engine_tile(int ggml_type) const {
        if (ggml_type == 8 && lpr == 32 && nr == 2 && sg == 8 && ks == 2) return {32, 2, 16, 2};
        return *this;
    }
    // The kernel the engine runs for this tile: hy4_flash_gemv_q8.metal for Q8_0 (bitwise with gemv<Q8_0, ...>),
    // gemv.metal's gemv for every other format.  Named by engine_tile's shape.
    std::string engine_name(int ggml_type) const {
        if (ggml_type == 8) {
            const GemvTile e = engine_tile(ggml_type);
            return "hy4_flash_gemv_q8_" + std::to_string(e.lpr) + "_" + std::to_string(e.nr) + "_" + std::to_string(e.sg) +
                   "_" + std::to_string(e.ks);
        }
        return name(ggml_type);
    }
};

// Frozen: a function of the format, n_in and (for the mid band's NR) n_out ONLY, never of the window size (window.cpp
// picks by ntok and by STRATA_METAL_GEMV_NR; hy4-flash rejects that env switch and must not have the reduction order move
// with T).  The bands
// are the ones window.cpp's parity test uses for the same shapes; the first candidate that the kernel's own
// constraints allow (the K split must divide the units, and a lane must get at least LPR / 2 of them) wins.
//
// These tiles run gemv.metal, whose lane reduction is quant.metal's lane_sum<LPR>, and lane_sum<32> is simd_sum - so
// the LPR = 32 tiles (every n_in >= 6144 shape, and the mid band's top choice) reduce with an order Metal does not
// specify.  T invariance across window sizes is therefore a property the parity tests check on this device, not a
// structural guarantee; hy4-flash's own kernels use hy4_flash_common.metal's written-out butterfly instead.
inline constexpr uint32_t kGemvMidNr2Rows = 6144;
inline constexpr uint32_t kGemvWideNr2Rows = 1024;
inline GemvTile gemv_tile(int ggml_type, uint32_t n_in, uint32_t n_out) {
    // NR = 2 on the wide tile's first choice is BITWISE SAFE because a lane's unit sequence `u = u0 + li; u += LPR`
    // is a function of (li, LPR, KS) only - NR decides which ROWS a lane owns, never the order in which one row's
    // partial products accumulate, and `lane_sum<LPR>`'s tree and the K-split merge are unchanged.  LPR / KS / SG do
    // change the order and must stay frozen.
    // NR above 2 is not used: it shrinks the grid below what fills the GPU.  This is not a thread-group limit -
    // `maxTotalThreadsPerThreadgroup` is a flat 1024 over every tile x NTOK = 1..8.
    // NR 2 also needs a matrix tall enough to keep the grid: a shape like hc_attn_fn / hc_ffn_fn (12288 -> 8) or
    // output_hc_fn (-> 4) gets ONE thread group at NR 2 and two at NR 1, so those keep NR 1.  Same bitwise argument.
    const GemvTile wide[] = {{32, n_out >= kGemvWideNr2Rows ? 2 : 1, 8, 2}, {32, 1, 4, 1}, {16, 1, 4, 1}, {8, 1, 4, 1},
                             {1, 1, 4, 1}};
    // NR = 2 on the mid band's first choice applies to TALL matrices only, by the same argument as the wide tile's:
    // LPR 16, SG 4 and KS 1 are held, so every row's unit sequence and its lane_sum<16> tree are unchanged and the
    // output is bitwise the NR = 1 tile's.  The height condition keeps the grid large: 16 rows a thread group, so
    // n_out >= kGemvMidNr2Rows still leaves hundreds of thread groups.  Shorter matrices keep NR = 1 - halving an
    // already small grid would idle cores.
    const bool tall = n_out >= kGemvMidNr2Rows;
    const GemvTile mid[] = {{16, tall ? 2 : 1, 4, 1}, {8, 1, 4, 1}, {4, 1, 4, 1}, {2, 1, 4, 1}, {1, 1, 4, 1}};
    const GemvTile narrow[] = {{8, 1, 4, 1}, {4, 1, 4, 1}, {2, 1, 4, 1}, {1, 1, 4, 1}};
    const GemvTile* c = n_in >= 6144 ? wide : n_in <= 640 ? narrow : mid;
    const int n = n_in >= 6144 ? 5 : n_in <= 640 ? 4 : 5;
    const int u = (int)n_in / gemv_unit(ggml_type);
    for (int i = 0; i < n; ++i)
        if (u % c[i].ks == 0 && u / c[i].ks >= c[i].lpr / 2) return c[i];
    return {1, 1, 2, 1};
}

// The tile for a PER-HEAD weight run through DenseGemvF32::append_heads (grid = {row groups, H}).  Same arguments as
// gemv_tile and still a function of format and width alone - H is a property of the weight's layout, not of the
// window, so this returns the same tile for every T.
//
// WHY IT DIFFERS AT ALL.  gemv_tile's three bands were chosen on PLAIN matrices, where the whole grid is grid.x, and
// two of them already give a TALL matrix NR 2 (kGemvWideNr2Rows, kGemvMidNr2Rows): `gemv` holds q[NR][UNIT] and
// re-reads x per token, so a unit costs NR*D + NTOK*(L + NR*C) and NR divides the per-token activation load L.  The
// narrow band is the only one with no height condition, and a per-head dispatch is exactly where that costs: grid.y
// = H multiplies the thread-group count by 32 or 64, so halving grid.x cannot idle the GPU the way it does for a
// plain matrix of the same height.
//
// T = 1 IS REACHABLE, so the tile may not be conditioned on it.  A tile may not be a function of the window, so
// there is no "NR 2 only at T >= 2" to reach for, and all three of these paths do hit T = 1: generation without
// --mtp calls Forward::window(..., 1) (main.cpp), MTP clips its verify window to the drafts left so a short tail
// window can be 1 or 2, and a prefill chunk's <= 8-row tail reaches the GEMV.
//
// BF16 ONLY.  Two formats can reach a per-head dispatch at all (gemv_f32_raw_x admits F32, BF16 and Q8_0):
//   * Q8_0 must NOT take it: Q8_0 does not run gemv.metal but hy4_flash_gemv_q8.metal, whose HY4_FLASH_GQ8 list has {8,1,4,1}
//     and no {8,2,4,1}, so selecting this tile aborts the engine at the first decode token with "no function
//     hy4_flash_gemv_q8_8_2_4_1".  (Its per-head use is the MTP's requantised k_b / v_b, `wr[i] = WRef{&qdense, at, 8}`
//     in mtp.cpp, dispatched at n_tok = 1.)
//   * F32 is deliberately left out even though GV_SET instantiates {8,2,4,1} for it and nothing in the mechanism is
//     format-specific.  A checkpoint whose per-head k_b is F32 therefore keeps gemv_tile's choice.
//
// BOTH BOUNDS ARE PART OF THE GUARD: few units a row AND tall.  Height alone is not the discriminator - the two
// per-head shapes the guard was built around have opposite signs at equal thread-group count, and they differ in
// n_in as well as n_out, so n_out cannot be what separates them.  A per-head BF16 512 -> 512 therefore does NOT get
// NR 2: under an n_out-only rule it would have, on the strength of a shape whose n_in disagreed.  Nothing is known
// about the shapes between the two, which is why this is a guard and not a formula - widen it only on evidence.
//
// The occupancy half also assumes a real head count: at n_out 512 this tile gives ceil(512/32) * H thread groups,
// but only 16 at H = 1 - and DenseGemvF32::append() will run a heads-configured shape with n_heads = 1, so a caller
// that did that would halve an already tiny grid.  No engine path does: all four per-head sites pass the block's H.
//
// BITWISE.  Only NR changes, and only in the narrow band.  With KS = 1 a row's unit sequence is `u = li; u += LPR`
// and its reduction is `lane_sum<LPR>`'s XOR tree over the same aligned LPR lanes, both functions of (li, LPR)
// alone, and there is no K-split merge; NR decides which lane and which thread group own a row, never the order in
// which one row's partial products accumulate.
inline constexpr uint32_t kGemvHeadNr2Rows = 512;   ///< per-head NR 2 guard: the minimum n_out
inline constexpr int kGemvHeadNr2Units = 12;       ///< and the maximum units a row (192 / 16 for BF16)
inline GemvTile gemv_head_tile(int ggml_type, uint32_t n_in, uint32_t n_out) {
    const GemvTile t = gemv_tile(ggml_type, n_in, n_out);
    // A format with a BESPOKE engine kernel has its own instantiation list, so a tile gemv.metal provides may not
    // exist there - that is how Q8_0 would abort the engine.  Asking engine_name() rather than testing
    // `ggml_type == 8` keeps this true if another format ever gets its own kernel.
    if (t.engine_name(ggml_type) != t.name(ggml_type)) return t;
    if (ggml_type != 30) return t;   // BF16 is the only format this guard covers (see above)
    if (t.lpr == 8 && t.nr == 1 && t.sg == 4 && t.ks == 1 && n_out >= kGemvHeadNr2Rows &&
        (int) n_in / gemv_unit(ggml_type) <= kGemvHeadNr2Units)
        return {8, 2, 4, 1};
    return t;
}

// ---- the prepared activation plane (units.metal xprep) --------------------------------------------------------------
// xprep writes, per token row of n floats, four planes - raw, x * 4^-(i%8), x * 2^-(8 (i%4)), x * 2^-(8 (i%2)) - and
// then the per-16 sums; XP = [4][n_tok][n] floats followed by S16 = [n_tok][n / 16].  The scalings are exact powers
// of two, so the activation is still f32 and nothing is quantized.  A format reads the plane its decoder's mask
// trick needs, so the kind must match units.metal's XKind<F>.
enum { kXkRaw = 0, kXkQ2 = 1, kXkP4 = 2, kXkP2 = 3 };
inline int xprep_kind(int ggml_type) {
    switch (ggml_type) {
        case 42: return kXkQ2;             // Q2_0
        case 12: case 13: return kXkP4;    // Q4_K, Q5_K
        case 11: case 14: return kXkP2;    // Q3_K, Q6_K
        default: return kXkRaw;            // Q8_0, F32, BF16, IQ2_XXS, the i-quants
    }
}
// Floats xprep writes for a window of n_tok rows of n values.  The raw plane equals x, so a format whose kind is
// kXkRaw can be given the activation buffer directly and no xprep pass is needed.
inline size_t xprep_floats(uint32_t n, uint32_t n_tok) { return (size_t)n_tok * n * 4 + (size_t)n_tok * (n / 16); }
inline bool xprep_needed(int ggml_type) { return xprep_kind(ggml_type) != kXkRaw; }

// ---- the routed experts' GEMVs --------------------------------------------------------------------------------------
// Frozen, like gemv_tile: a function of the format and the width, never of the window or of the group's entry count
// (hy4_flash_moe.metal's T-invariance note).  Gate/up: one choice for every format and every width the engine uses - 8
// lanes a row, one row a lane, 4 simdgroups, no K split; a 1536-wide row gives each lane 6 of its 48 units, and the
// narrowest row the engine has (256, one K-quant block) still gives it one.
//
// The down (moe_down_tile) differs in ONE place: the Q4_K down takes TWO rows a lane.  The bits do not move - LPR
// (8) fixes a row's unit sequence u = li, li + 8, ... and its butterfly, and NR only decides how many rows one lane
// carries - but every entry's prepared x unit (32 floats + 2 sums, 136 B for 18 B of weights) is then loaded once
// for two rows instead of once a row, and at T >= 2 that x traffic is what the down waits on.  Only Q4_K takes it;
// Q6_K / BF16 / Q8_0 keep the gate/up tile.  A down tile's rows a threadgroup therefore depend on the format, so
// anything that sizes a down grid must ask for the DOWN's tile (MoeF32::down_rows_per_tg; forward.cpp keeps the
// indirect grid widths per ordinal for the same reason).  hy4_flash_moe.metal instantiates exactly these tiles.
inline GemvTile moe_tile(int /*ggml_type*/, uint32_t /*n_in*/) { return {8, 1, 4, 1}; }
inline GemvTile moe_down_tile(int ggml_type, uint32_t n_in) {
    return ggml_type == 12 /* Q4_K */ ? GemvTile{8, 2, 4, 1} : moe_tile(ggml_type, n_in);
}
inline std::string moe_name(const char* kind, int ggml_type, const GemvTile& t) {
    return std::string("hy4_flash_moe_") + kind + "_" + std::to_string(ggml_type) + "_" + std::to_string(t.lpr) + "_" +
           std::to_string(t.nr) + "_" + std::to_string(t.sg);
}
inline std::string moe_gu_name(int ggml_type, uint32_t n_in) {
    return moe_name("gu", ggml_type, moe_tile(ggml_type, n_in));
}
inline std::string moe_down_name(int ggml_type, uint32_t n_in) {
    return moe_name("down", ggml_type, moe_down_tile(ggml_type, n_in));
}

}  // namespace strata_hy4_flash_metal