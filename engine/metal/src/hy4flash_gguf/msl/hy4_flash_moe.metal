// engine/metal/src/hy4flash_gguf/msl/hy4_flash_moe.metal - the routed experts of one MoE block with f32 activations.
//
// Why not moe.metal's moe_gu / moe_down: those are reusable for f32 x and are (gate/up for Q4_K, down for Q8_0 and
// friends), but three things here are different and all three are in the frozen spec:
//
//   * the SwiGLU carries the clamp - g' = min(g, limit), u' = clamp(u, +-limit) - which moe.metal does not have
//     (hy4_block.py latent_moe, semantics.md §7c).  moe.metal's h = silu(g) * u is a different function;
//   * the reduction across the lanes of a row must be a WRITTEN-OUT butterfly.  quant.metal's lane_sum<32> calls
//     simd_sum(), whose order Metal leaves open, and T invariance is a bitwise requirement;
//   * the plan's sections are addressed by OFFSETS in the args, not by moe.metal's compile-time P_MAXE = 128, so a
//     prefill block of 1024 positions x 16 experts fits the same kernels as a decode window of 8.
//
// What IS reused, deliberately: quant.metal's Dec<F> decoders, units.metal's X<F> prepared-x loaders and
// unit_combine, and units.metal's xprep.  The half-input mm_gu3 / mm_down3 / moe_gu16 / moe_down16 are NOT used:
// they are outside the f32 whitelist and their reduction order differs from the per-row one, which breaks T
// invariance.
//
// Expert widths are the LATENT ones - 1536 in and 1536 out, not the hidden 3072 (semantics.md §7c: the router reads
// the full-width cur, the experts work on ffn_routed_down . cur).
//
// T INVARIANCE.  A group's entry count NE changes with the window (an expert used by three of eight tokens has
// NE = 3; the same position alone has NE = 1), so NE must not touch the arithmetic.  It does not: every entry owns
// its own accumulator and the unit loop `for (u = li; u < U; u += LPR)` is a function of the row width alone, so NE
// only decides how many accumulators are live at once.  The frozen tiles (moe_tile / moe_down_tile in
// hy4_flash_kernels.h) are a function of the format and the width, never of the window.
//
// Formats: gate/up Q3_K or Q4_K (the q3k_new_mtp checkpoint) and IQ2_XXS (the IQ2_AH checkpoint); down Q4_K in both,
// Q6_K for the type semantics.md records.  Q3_K / Q4_K / Q6_K take a prepared x plane from xprep (the mask trick
// pre-scales x by exact powers of two, so the activation is still f32); IQ2_XXS reads the raw plane, which is x
// itself, and ignores the per-16 sums.

// ---- the plan ------------------------------------------------------------------------------------------------------
// One int buffer, sections at the offsets the args carry:
//   gexp   [n_group]            the expert id of each group
//   gstart [n_group + 1]        the first entry of each group (gstart[n_group] = n_entry)
//   etok   [n_entry]            the token an entry belongs to
//   ew     [n_entry]            the routing weight of that entry, as float bits
//   tokent [n_tok * topk]       the entry index of (token, slot), so the combine can add in SLOT order
// Entries of one group are contiguous and the groups partition the window's n_tok * topk selections.
struct Hy4FlashMoeArgs {
    ulong gate_off;     // bytes from an expert's base to its gate matrix
    ulong up_off;
    ulong down_off;
    ulong gu_stride;    // bytes between experts' gate/up blocks
    ulong d_stride;     // bytes between experts' down blocks
    uint gu_row;        // bytes per gate/up row
    uint d_row;         // bytes per down row
    uint n_latent;      // the experts' input and output width (1536)
    uint n_ff;          // the experts' hidden width (1536)
    uint n_tok;
    uint topk;
    uint n_group;
    uint off_gexp;      // ints into the plan buffer
    uint off_gstart;
    uint off_etok;
    uint off_ew;
    uint off_tokent;
    float limit;        // swiglu_clamp_exp (10 here); +inf disables the clamp
};

// Test hook, in the style of gemv.metal's GEMV_PROBE and moe.metal's MOE_PROBE: 1 = when a group carries more than
// one entry, reduce the row with the butterfly run the other way round.  That is a reduction order that follows the
// window - exactly what T invariance forbids - and the T-invariance tests use it as their negative control.  Unset
// in production, and the correct path is the one with no probe.
constant int HY4_FLASH_MOE_PROBE [[function_constant(52)]];
constant int hy4_flash_moe_probe = is_function_constant_defined(HY4_FLASH_MOE_PROBE) ? HY4_FLASH_MOE_PROBE : 0;

// ---- gate + up + the clamped SwiGLU --------------------------------------------------------------------------------
template <int F, int LPR, int NR, int NE>
static inline void hy4_flash_gu_body(device const uchar* g0, device const uchar* u0, device const float* x,
                               device const float* xs16, device float* h, device const int* etok, int c, int row0,
                               int li, constant Hy4FlashMoeArgs& a, threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    const bool rev = hy4_flash_moe_probe == 1 && NE > 1;
    float ag[NR][NE], au[NR][NE];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) ag[r][t] = au[r][t] = 0.0f;
    const int U = int(a.n_latent) / UNIT;
    // a row past the end reads the last one and has its result dropped on the write, so n_ff need not be a multiple
    // of the rows a threadgroup covers
    device const uchar* gp[NR];
    device const uchar* up[NR];
    for (int r = 0; r < NR; ++r) {
        const ulong rc = (ulong)min(row0 + r, int(a.n_ff) - 1);
        gp[r] = g0 + rc * a.gu_row;
        up[r] = u0 + rc * a.gu_row;
    }
    for (int u = li; u < U; u += LPR) {
        float qg[NR][UNIT], qu[NR][UNIT];
        float4 cg[NR], cu[NR];
        for (int r = 0; r < NR; ++r) {
            Dec<F>::run(gp[r], u, qg[r], cg[r], tab);
            Dec<F>::run(up[r], u, qu[r], cu[r], tab);
        }
        for (int t = 0; t < NE; ++t) {
            const int tok = etok[t];
            float xs[UNIT], xsum[2];
            X<F>::load(x + tok * int(a.n_latent), xs16 + tok * int(a.n_latent / 16), u, xs, xsum);
            for (int r = 0; r < NR; ++r) {
                ag[r][t] += unit_combine<UNIT>(qg[r], cg[r], xs, xsum);
                au[r][t] += unit_combine<UNIT>(qu[r], cu[r], xs, xsum);
            }
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) {
            const float gv = hy4_flash_lane_sum<LPR>(ag[r][t], rev), uv = hy4_flash_lane_sum<LPR>(au[r][t], rev);
            if (li == 0 && row0 + r < int(a.n_ff)) {
                const float g2 = min(gv, a.limit), u2 = clamp(uv, -a.limit, a.limit);
                h[(c + t) * int(a.n_ff) + row0 + r] = hy4_flash_silu(g2) * u2;
            }
        }
}

// grid (ceil(n_ff / rows per threadgroup), n_group); threads SG * 32.
template <int F, int LPR, int NR, int SG>
kernel void hy4_flash_moe_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* h [[buffer(2)]], device const int* plan [[buffer(3)]],
                       constant Hy4FlashMoeArgs& a [[buffer(4)]], device const float* xs16 [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR;
    // An EMPTY group slot (gstart[g] == gstart[g + 1]) leaves before the table load and its barrier: g is the
    // threadgroup's, so the whole threadgroup returns together, and an empty group writes nothing either way.
    const int g = int(tg.y);
    if (g >= int(a.n_group) || plan[a.off_gstart + g + 1] == plan[a.off_gstart + g]) return;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const ulong ex = (ulong)plan[a.off_gexp + g];
    const int c = plan[a.off_gstart + g], ne = plan[a.off_gstart + g + 1] - c;
    device const int* etok = plan + a.off_etok + c;
    device const uchar* E = W + ex * a.gu_stride;
    device const uchar* g0 = E + a.gate_off;
    device const uchar* u0 = E + a.up_off;
    // NE blocks of at most 4, so the entry count only decides register blocking (see the T-invariance note above)
    for (int k = 0; k < ne; k += 4) {
        const int m = min(4, ne - k);
        if (m == 4) hy4_flash_gu_body<F, LPR, NR, 4>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
        else if (m == 3) hy4_flash_gu_body<F, LPR, NR, 3>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
        else if (m == 2) hy4_flash_gu_body<F, LPR, NR, 2>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
        else hy4_flash_gu_body<F, LPR, NR, 1>(g0, u0, x, xs16, h, etok + k, c + k, row0, li, a, tab);
    }
}

// ---- down, with the routing weight folded in -----------------------------------------------------------------------
// The weight is applied in the down, then the experts are added in the latent space.  So ye holds the
// already-weighted expert output and hy4_flash_moe_combine_latent is a plain sum.
template <int F, int LPR, int NR, int NE>
static inline void hy4_flash_dn_body(device const uchar* d0, device const float* h, device const float* hs16,
                               device float* ye, device const int* plan, int c, int row0, int li,
                               constant Hy4FlashMoeArgs& a, threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    const bool rev = hy4_flash_moe_probe == 1 && NE > 1;
    float ac[NR][NE];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) ac[r][t] = 0.0f;
    const int U = int(a.n_ff) / UNIT;
    device const uchar* dp[NR];
    for (int r = 0; r < NR; ++r) dp[r] = d0 + (ulong)min(row0 + r, int(a.n_latent) - 1) * a.d_row;
    for (int u = li; u < U; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(dp[r], u, q[r], sc[r], tab);
        for (int t = 0; t < NE; ++t) {
            float xs[UNIT], xsum[2];
            X<F>::load(h + (c + t) * int(a.n_ff), hs16 + (c + t) * int(a.n_ff / 16), u, xs, xsum);
            for (int r = 0; r < NR; ++r) ac[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NE; ++t) {
            const float v = hy4_flash_lane_sum<LPR>(ac[r][t], rev);
            if (li == 0 && row0 + r < int(a.n_latent))
                ye[(c + t) * int(a.n_latent) + row0 + r] = v * as_type<float>(plan[a.off_ew + c + t]);
        }
}

template <int F, int LPR, int NR, int SG>
kernel void hy4_flash_moe_down(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                         device float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                         constant Hy4FlashMoeArgs& a [[buffer(4)]], device const float* hs16 [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int RG = 32 / LPR;
    // An EMPTY group slot (gstart[g] == gstart[g + 1]) leaves before the table load and its barrier: g is the
    // threadgroup's, so the whole threadgroup returns together, and an empty group writes nothing either way.
    const int g = int(tg.y);
    if (g >= int(a.n_group) || plan[a.off_gstart + g + 1] == plan[a.off_gstart + g]) return;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    const ulong ex = (ulong)plan[a.off_gexp + g];
    const int c = plan[a.off_gstart + g], ne = plan[a.off_gstart + g + 1] - c;
    device const uchar* d0 = W + ex * a.d_stride + a.down_off;
    for (int k = 0; k < ne; k += 4) {
        const int m = min(4, ne - k);
        if (m == 4) hy4_flash_dn_body<F, LPR, NR, 4>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
        else if (m == 3) hy4_flash_dn_body<F, LPR, NR, 3>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
        else if (m == 2) hy4_flash_dn_body<F, LPR, NR, 2>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
        else hy4_flash_dn_body<F, LPR, NR, 1>(d0, h, hs16, ye, plan, c + k, row0, li, a, tab);
    }
}

// ---- the combine, in the latent space ------------------------------------------------------------------------------
// The top-k outputs are added ONE AT A TIME IN SELECTION ORDER over the 1536-wide latent, with FMA contraction off
// (hy4_flash_common.metal's pragma covers this file).  tokent gives the entry of (token, slot), so the order is the slot
// order whatever the plan's grouping did - that is what makes the sum independent of the window.
// grid (ceil(n_latent / HY4_FLASH_TG) * HY4_FLASH_TG, n_tok).
kernel void hy4_flash_moe_combine_latent(device const float* ye [[buffer(0)]], device float* y [[buffer(1)]],
                                   device const int* plan [[buffer(2)]], constant Hy4FlashMoeArgs& a [[buffer(3)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n_latent || gid.y >= a.n_tok) return;
    device const int* tt = plan + a.off_tokent + gid.y * a.topk;
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += ye[(ulong)tt[j] * a.n_latent + gid.x];
    y[(ulong)gid.y * a.n_latent + gid.x] = s;
}

// ---- instantiations ------------------------------------------------------------------------------------------------
// The tiles are frozen in hy4_flash_kernels.h (moe_tile, moe_down_tile): LPR 8, SG 4 for every format and every width this
// engine uses (48 units of a 1536-wide row over 8 lanes a row; a 256-wide row still gives each lane one unit), NR 1
// except the Q4_K down, NR 2.  One name per format and kernel; the Q4_K down's NR 1 instance is kept as the bitwise
// reference its NR 2 tile is checked against (tmp/tests/hy4/metal/hy4_moe_dn_tile_test.cpp).
// The HY4_FLASH_GU / HY4_FLASH_GU_ pair is the indirection the shared kernels use too (gemv.metal's GV_SET): the outer macro
// expands T_Q4_K to its number before the inner one stringifies it, so the host name is hy4_flash_moe_gu_12_8_1_4.
#define HY4_FLASH_GU_(F, L, R, S) template [[host_name("hy4_flash_moe_gu_" #F "_" #L "_" #R "_" #S)]] \
                           kernel decltype(hy4_flash_moe_gu<F, L, R, S>) hy4_flash_moe_gu<F, L, R, S>;
#define HY4_FLASH_DN_(F, L, R, S) template [[host_name("hy4_flash_moe_down_" #F "_" #L "_" #R "_" #S)]] \
                           kernel decltype(hy4_flash_moe_down<F, L, R, S>) hy4_flash_moe_down<F, L, R, S>;
#define HY4_FLASH_GU(F, L, R, S) HY4_FLASH_GU_(F, L, R, S)
#define HY4_FLASH_DN(F, L, R, S) HY4_FLASH_DN_(F, L, R, S)
HY4_FLASH_GU(T_Q4_K, 8, 1, 4) HY4_FLASH_GU(T_Q3_K, 8, 1, 4) HY4_FLASH_GU(T_IQ2_XXS, 8, 1, 4)
HY4_FLASH_DN(T_Q4_K, 8, 2, 4) HY4_FLASH_DN(T_Q4_K, 8, 1, 4) HY4_FLASH_DN(T_Q6_K, 8, 1, 4)
// the MTP block's experts: BF16 gate / up / down, read from the raw f32 plane like every raw format
HY4_FLASH_GU(T_BF16, 8, 1, 4) HY4_FLASH_DN(T_BF16, 8, 1, 4)
// and the same block with HY4_FLASH_MTP_QUANT=q8_0 (mtp.cpp): Q8_0 gate / up / down, raw plane as well
HY4_FLASH_GU(T_Q8_0, 8, 1, 4) HY4_FLASH_DN(T_Q8_0, 8, 1, 4)
