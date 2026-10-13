// src/core/ocl_peer.cpp - see include/strata/core/ocl_peer.hpp.
//
// The MI50 as a second expert tier, in OpenCL.  Every kernel below is a transcription of the CUDA one
// in src/kernels/cuda/iq_kernels.cu, and the transcription is what makes this worth doing at all: an
// OpenCL C kernel can be checked against the CUDA source operation by operation, which a SPIR-V shader
// could not.
//
// Three things had to be got exactly right, and each was settled against the CUDA source, not guessed:
//
//  1. The reduction tree.  `row_dot` walks `k = lane; k < nb*ipb; k += 32` - lane l owns calls l, l+32,
//     l+64, ... - and finishes with `for (o = 16; o; o >>= 1) v += shfl_xor(v, o)`, an xor butterfly.
//     An OpenCL work-group whose local-memory tree runs off = WG/2, WG/4 ... 1 builds the *same* tree
//     when WG is 32, so 32 lanes is what reproduces the result bit for bit.  IQ4_NL has 40 calls per
//     row, and there a 64-lane group (whose first tree step is off = 32, i.e. lane l += lane l+32)
//     splits the calls the same way and is 1.7x faster - the one format that does not use 32.
//
//  2. Which q8_1 activation block a call multiplies.  It is not uniform across formats: IQ3_XXS,
//     IQ3_S, IQ2_S, IQ4_XS and Q2_0 all use `bq8_1[kbx * (qk/32) + iqs/step]` = kbx*ipb + m, but
//     IQ4_NL takes the whole block `bq8_1 + kbx` and picks four of its eight words.  Reading the
//     wrong block leaves every dot product uncorrelated while still producing plausible numbers.
//
//  3. The float expression.  The engine builds with --use_fast_math (CMakeLists.txt), so nvcc turns
//     `acc += d * sumi` into `mul.f32` + `fma.rn.f32` - the accumulate rounds ONCE.  `FP_CONTRACT OFF`
//     plus an explicit fma() reproduces exactly that, and `dw * dx` still rounds on its own.
//
// And one real bug was found on the way: block_q2_0 is {ggml_half d; uint8_t qs[16]}, so its `d` is a
// 2-byte half at offset 0 and the 16 qs bytes start at 2.  The probe and its Python reference both read
// `d` as a 4-byte float - self-consistent, so the cross-check could not see it, but wrong.
#define _CRT_SECURE_NO_WARNINGS
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include "strata/core/ocl_peer.hpp"

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace strata::core {
namespace {

// ---------------------------------------------------------------- the format table

struct KFmt {
    int type;
    int qk, bb, ipb, step, wg;
    const char* name;
};

const KFmt kFmts[] = {
    {18, 256,  98, 8, 2, 32, "IQ3_XXS"},
    {20,  32,  18, 2, 2, 64, "IQ4_NL"},
    {21, 256, 110, 8, 2, 32, "IQ3_S"},
    {22, 256,  82, 8, 2, 32, "IQ2_S"},
    {23, 256, 136, 8, 4, 32, "IQ4_XS"},
    {42,  64,  18, 2, 1, 32, "Q2_0"},
};

const KFmt* fmt_of(int type) {
    for (const KFmt& f : kFmts)
        if (f.type == type) return &f;
    return nullptr;
}

/// Calls per row: (values / qk) blocks, `ipb` calls each.
int calls_per_row(const KFmt& f, int64_t row_vals) { return (int) ((row_vals / f.qk) * f.ipb); }
/// Bytes of one weight row of this format for `n` values.
uint64_t row_bytes(const KFmt& f, int64_t n) { return (uint64_t) (n / f.qk) * (uint64_t) f.bb; }

/// Rows one work-group owns.  Free: every row's tree is the same 32-lane tree whatever R is, so this only
/// trades register pressure against how many rows share one pass over the activation.
constexpr int kRows = 4;
/// The same number at run time (STRATA_OCL_ROWS, 1..32), because it is worth sweeping and costs nothing to
/// change: the reduction - IQ_WG/2 log steps of barrier() - is paid once per work-GROUP, so R rows per
/// group is R rows per reduction.  R=4, the shape that was verified bit-exact, pays six barriers for four
/// rows; nothing about a single result depends on R, only on the per-row loop order (lane l takes calls
/// l, l+IQ_WG, l+2*IQ_WG...), which R does not touch.
int g_rows = kRows;
/// Lane count of the IQ4_NL kernel: 8 sub-groups of the reference's 8-lane sub-warp, one row each.  Its
/// rows per group are IQ_WG/8, and this is the width that was verified bit-exact.
constexpr int kIq4nlWg = 64;
/// The IQ4_NL kernel's work-group width (STRATA_OCL_IQ4NL_WG, a multiple of 8 in 8..256).  Its rows per
/// group are IQ_WG/8 - one per sub-warp - and there is no barrier to amortise (the reference's 8-lane
/// sub-warp reduces with an xor ladder), so the reason to raise it is different and simpler: an IQ4_NL
/// layer's down pass is 2560 rows, and at 8 rows/group that is 320 work-groups per entry.  What a small
/// window pays for is the WAVE COUNT, not the arithmetic.
int g_iq4nl_wg = kIq4nlWg;
/// Rows one work-group covers, by format: IQ4_NL takes its shape from row_dot_40_sub8, everyone else from
/// g_rows.
int grp_rows(int type) { return type == 20 ? g_iq4nl_wg / 8 : g_rows; }

/// The work-group width a format's row kernel is built - and launched - with.
int fmt_wg(const KFmt& f) { return f.type == 20 ? g_iq4nl_wg : f.wg; }

/// Work-group width for one row program, by its call count.  f.wg mirrors the reference's 32-lane tree,
/// and on NVIDIA that is a full warp.  On a wave64 GPU (gfx906) a 32-wide group is HALF a wave: two of
/// them share one hardware wave, so their barriers synchronise each other and the scheduler can only
/// start whole waves.  A row of >= 64 calls also fills 64 lanes with work (chunks stride across lanes),
/// so wide rows go wide: gate/up is 80 calls on IQ3_S, and at 32 lanes every lane walked it in 2-3
/// iterations where 64 lanes walk it in 1-2 on native waves.  Narrow rows (the down pass, 20 calls)
/// stay at 32 - widening them would idle 44 of every 64 lanes.
///
/// The 64-lane tree over >= 64 chunks groups the partial sums differently from the reference's 32-lane
/// tree, so rows through this path can differ from the CUDA peer in the last ulp - the same rounding
/// class the expert cache already accepts (bench/results/2026-09-27-cache-parity).  STRATA_OCL_WG=32
/// restores the exact tree; the default follows the wave shape of the card.
int row_wg(const KFmt& f, int nchunk) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_OCL_WG");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return (n == 32 || n == 64) ? n : 0;
    }();
    if (forced != 0) return forced;
    return nchunk >= 64 ? 64 : f.wg;
}


/// Reads one of the two above from the environment, clamped.  Called from OclPeer::open, before any
/// program is built (the generated source carries the number as a #define).
int rows_from_env(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr) return dflt;
    const int n = std::atoi(v);
    return (n >= 1 && n <= 32) ? n : dflt;
}
/// Entries (expert-tokens) one layer's window can hold.  The engine's own CAP is MAXT*10 = 80; twice
/// that costs a few MB and removes the question.
constexpr int64_t kCapEntries = 256;
/// Tokens one activation upload can carry.
constexpr int64_t kCapTokens = 256;

// ---------------------------------------------------------------- kernel text

const char* kHelpers = R"CLC(
// block_q8_1 is 36 bytes = 9 uint: word 0 is {fp16 d, fp16 sum}, words 1..8 are the 32 int8.  So the
// activation buffer is read through one u32 view - the d is the low half of the first word - and the
// whole thing stays a single cl_mem bound to one argument.
inline float f16_to_f(uint h) {
    const uint s = (h >> 15u) & 1u;
    const uint e = (h >> 10u) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 31u) return as_float((s << 31u) | 0x7F800000u | (m << 13u));
    if (e == 0u) {
        if (m == 0u) return as_float(s << 31u);
        const float v = (float) m * 5.9604644775390625e-08f;
        return s ? -v : v;
    }
    return as_float((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// f32 -> f16, round to nearest even, exactly as __float2half.  Written out rather than using the
// fp16 extension so the rounding is visible: q8_1_ds stores both the scale and the block sum this way.
inline uint f32_to_f16(float v) {
    const uint u = as_uint(v);
    const uint s = (u >> 16) & 0x8000u;
    const uint e = (u >> 23) & 0xFFu;
    const uint m = u & 0x7FFFFFu;
    if (e == 255u) return s | 0x7C00u | (m != 0u ? 0x200u : 0u);
    const int ne = (int) e - 127 + 15;
    if (ne >= 31) return s | 0x7C00u;
    if (ne <= -11) return s;
    if (ne <= 0) {
        const uint mm = m | 0x800000u;
        const uint sh = (uint) (14 - ne);
        uint r = mm >> sh;
        const uint rem = mm & ((1u << sh) - 1u);
        const uint halfway = 1u << (sh - 1u);      // not `half`: that is a type name in OpenCL
        if (rem > halfway || (rem == halfway && (r & 1u) != 0u)) ++r;
        return s | r;
    }
    uint r = m >> 13;
    const uint rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) != 0u)) {
        if (++r == 0x400u) { r = 0u; if (ne + 1 >= 31) return s | 0x7C00u; return s | ((uint) (ne + 1) << 10); }
    }
    return s | ((uint) ne << 10) | r;
}

inline uint rd16(__global const uchar* p, uint o) {
    return (uint) p[o] | ((uint) p[o + 1u] << 8);
}
inline uint rd32(__global const uchar* p, uint o) {
    return (uint) p[o] | ((uint) p[o + 1u] << 8) | ((uint) p[o + 2u] << 16) | ((uint) p[o + 3u] << 24);
}

// Eight CONTIGUOUS bytes (the qa+qb of one iq4_nl call, at o = block*18 + 2 + 8*m) as one read.  rd32
// assembles each dword from four byte loads - ten LSU instructions per call - and this kernel is
// weight-bound, so the load-issue rate IS the streaming bandwidth: measured, the row passes moved
// weights at ~61 GB/s against a 633 GB/s bus.  Two (o 4-aligned) or three (o 2-mod-4) aligned dword
// loads replace the eight; `p` is 4-byte aligned (row strides and expert blobs are multiples of 4)
// and o's misalignment is only ever 0 or 2 (18*kbx + 2 + 8m mod 4).  Pure loads: bit-identical.
inline uint2 rd64(__global const uchar* p, uint o) {
    const uint mis = o & 3u;
    const __global uint* q = (const __global uint*) (p + (o - mis));
    const uint a = q[0], b = q[1];
    if (mis == 0u) return (uint2) (a, b);
    const uint c = q[2];
    return (uint2) ((a >> 16u) | (b << 16u), (b >> 16u) | (c << 16u));
}

// Same treatment for the two scale bytes at o = block*18 (mis 0 or 2): one dword instead of two bytes.
inline uint rd16a(__global const uchar* p, uint o) {
    const uint mis = o & 3u;
    const uint a = *(const __global uint*) (p + (o - mis));
    return mis == 0u ? (a & 0xFFFFu) : (a >> 16u);
}

inline uint unpack_ksigns(uint v) {
    const uint p = (uint) popcount(v & 0xFFu) & 1u;
    return (v & 0xFFu) ^ (p << 7u);
}

// vcmpne4/vsub4 as explicit byte loops: this driver mis-lowers the branchless bit trick in bit 31 of the
// mask (ocl_iq3.cpp), which turns a negative weight positive.  Measured bit-exact against __vcmpne4.
inline int vcmpne4(int a, int b) {
    const uint t = as_uint(a) ^ as_uint(b);
    uint r = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (((t >> (8 * i)) & 0xFFu) != 0u) r |= 0xFFu << (8 * i);
    return as_int(r);
}

inline int vsub4(int a, int b) {
    const uint ua = as_uint(a), ub = as_uint(b);
    const uint kHigh = 0x80808080u;
    return as_int(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}

// __dp4a without cl_khr_integer_dot_product (this device lacks it): mad24 x4 with sign-extended bytes,
// bit-exact against __dp4a and the fastest form the driver has (ocl_dot.cpp, 0/65536 mismatches).
inline int dp4x8(int a, int b, int c) {
    int s = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        s = mad24((a << (24 - 8 * i)) >> 24, (b << (24 - 8 * i)) >> 24, s);
    return s;
}

inline int pack4(int b0, int b1, int b2, int b3) {
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

// q8_1_finite: the block scale and sum are clamped to the largest finite half before being rounded to
// one, so a massive-activation dimension gives an approximate product instead of inf * 0 = NaN (#606).
inline float f16_max_clamp(float v) {
    return fabs(v) > 65504.0f ? copysign(65504.0f, v) : v;
}

// q = round(x/d) clamped to +-127, 0 when the block is all zeros.  OpenCL's round() is ties-away-from-zero,
// which is roundf on the CUDA side; rint() would be ties-to-even and differs on an exact half.
inline int q8_1_quant(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = round(xi / d);
    return (int) (q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q));
}

// q8_1_store (iq_kernels.cu): amax and sum come from an xor butterfly over the warp, offsets 16,8,4,2,1.
// The max is order-insensitive; the sum is not, and it is stored, so the tree has to be this one.  Every
// lane ends up with the same value.
inline void q8_1_store(float h, uint lane, __global uint* ds, __global uchar* qs) {
    __local float lmem[64];
    __local float lsc[64];
    lmem[lane] = fabs(h);
    lsc[lane] = h;
    for (uint o = 16u; o > 0u; o >>= 1u) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float va = lmem[lane ^ o];
        const float vs = lsc[lane ^ o];
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[lane] = fmax(lmem[lane], va);
        lsc[lane] = lsc[lane] + vs;
    }
    const float d = f16_max_clamp(lmem[lane] / 127.0f);
    qs[lane] = (uchar) q8_1_quant(h, d, lmem[lane]);
    if (lane == 0u) ds[0] = f32_to_f16(d) | (f32_to_f16(f16_max_clamp(lsc[lane])) << 16u);
}
)CLC";

/// The row kernels take one weight pointer per arena segment, appended after their existing arguments.
///
/// The arena cannot be one buffer: this device reports a 13.60 GiB maximum single allocation while holding
/// ~28 GiB it will actually keep resident, so a single buffer wastes more than half the card.  With the
/// arena split, a slot's segment rides in the top byte of the `ent_off` value the kernel already reads, and
/// picking the base is a few compares per work-group rather than a second pass over the entries.
const char* kSegArgs =
    ",\n            __global const uchar* wt1, __global const uchar* wt2, __global const uchar* wt3,\n"
    "            __global const uchar* wt4, __global const uchar* wt5, __global const uchar* wt6,\n"
    "            __global const uchar* wt7"
    ",\n            __global const float* gu_src, const uint n_ff";

/// `ent_off[e]` is `segment << 56 | offset within that segment`.  Unused segments are bound to segment 0's
/// buffer, so only the compares that can actually match are reachable.
const char* kSegPick = R"CLC(
    const ulong eo  = ent_off[e];
    const uint  seg = (uint) (eo >> 56);
    const ulong loc = eo & 0x00FFFFFFFFFFFFFFul;
    __global const uchar* base = wt;
    if (seg == 1u) base = wt1;
    else if (seg == 2u) base = wt2;
    else if (seg == 3u) base = wt3;
    else if (seg == 4u) base = wt4;
    else if (seg == 5u) base = wt5;
    else if (seg == 6u) base = wt6;
    else if (seg >= 7u) base = wt7;
)CLC";

/// The weight-side half of one call, in the format's own terms.  Returns a string because the text is
/// the transcription: each block below names the CUDA function it came from.
std::string fmt_decode(int type) {
    switch (type) {
        case 18:   // vec_dot_iq3_xxs_q8_1; block: d(2) qs[64]@2 aux[32]@66
            return R"CLC(
            const uint qlo = rd32(wr, base + 2u + 8u * m);
            const uint qhi = rd32(wr, base + 2u + 8u * m + 4u);
            const uint a32 = rd32(wr, base + 66u + 4u * m);
            int g[8];
            #pragma unroll
            for (int li = 0; li < 4; ++li) {
                const int l0 = li * 2;
                const uint wv = (li < 2) ? qlo : qhi;
                const uint sh = (uint) (8 * l0 - 32 * (li / 2));
                const uint gx = t2[(wv >> sh) & 0xFFu];
                const uint gy = t2[(wv >> (sh + 8u)) & 0xFFu];
                const uint signs = unpack_ksigns(a32 >> (uint) (7 * l0 / 2)) * 0x01010101u;
                const int n0 = vcmpne4((int) (signs & 0x08040201u), 0);
                const int n1 = vcmpne4((int) (signs & 0x80402010u), 0);
                g[l0 + 0] = vsub4((int) gx ^ n0, n0);
                g[l0 + 1] = vsub4((int) gy ^ n1, n1);
            }
            int si = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) si = dp4x8(g[j], (int) xw[j], si);
            si = (((int) (a32 >> 28u)) * si + si / 2) / 2;      // ls = aux32 >> 28
            const float dw = f16_to_f(rd16(wr, base));
)CLC";
        case 20:   // vec_dot_iq4_nl_q8_1; block: d(2) qs[16]@2.  THIS format uses the whole q8_1 block.
            // `get_int_from_table_16` picks values[n0],values[n2],values[n4],values[n6] - nibbles, so the
            // top index is `(q >> 24) & 0x0F`, NOT `q >> 24`.  kvalues_iq4nl has sixteen entries; the
            // unmasked form indexes 0..255 and read whatever followed the table on the device.
            // STRATA_OCL_RD8=1 restores the byte-assembled loads (A/B: the dword form moved gate/up
            // 0.17 -> 0.15 and down 0.10 -> 0.07 ms/layer of pure device time).
            if (std::getenv("STRATA_OCL_RD8") != nullptr)
                return R"CLC(
            const uint qa = rd32(wr, base + 2u + 8u * m);
            const uint qb = rd32(wr, base + 2u + 8u * m + 4u);
            int g[4];
            g[0] = pack4((int) (char) kv[qa & 0x0Fu], (int) (char) kv[(qa >> 8) & 0x0Fu],
                         (int) (char) kv[(qa >> 16) & 0x0Fu], (int) (char) kv[(qa >> 24) & 0x0Fu]);
            g[1] = pack4((int) (char) kv[(qa >> 4) & 0x0Fu], (int) (char) kv[(qa >> 12) & 0x0Fu],
                         (int) (char) kv[(qa >> 20) & 0x0Fu], (int) (char) kv[(qa >> 28) & 0x0Fu]);
            g[2] = pack4((int) (char) kv[qb & 0x0Fu], (int) (char) kv[(qb >> 8) & 0x0Fu],
                         (int) (char) kv[(qb >> 16) & 0x0Fu], (int) (char) kv[(qb >> 24) & 0x0Fu]);
            g[3] = pack4((int) (char) kv[(qb >> 4) & 0x0Fu], (int) (char) kv[(qb >> 12) & 0x0Fu],
                         (int) (char) kv[(qb >> 20) & 0x0Fu], (int) (char) kv[(qb >> 28) & 0x0Fu]);
            int si = 0;
            si = dp4x8(g[0], (int) xw0, si);
            si = dp4x8(g[1], (int) xw4, si);
            si = dp4x8(g[2], (int) xw1, si);
            si = dp4x8(g[3], (int) xw5, si);
            const float dw = f16_to_f(rd16(wr, base));
)CLC";
            return R"CLC(
            const uint2 qq = rd64(wr, base + 2u + 8u * m);
            const uint qa = qq.x, qb = qq.y;
            int g[4];
            g[0] = pack4((int) (char) kv[qa & 0x0Fu], (int) (char) kv[(qa >> 8) & 0x0Fu],
                         (int) (char) kv[(qa >> 16) & 0x0Fu], (int) (char) kv[(qa >> 24) & 0x0Fu]);
            g[1] = pack4((int) (char) kv[(qa >> 4) & 0x0Fu], (int) (char) kv[(qa >> 12) & 0x0Fu],
                         (int) (char) kv[(qa >> 20) & 0x0Fu], (int) (char) kv[(qa >> 28) & 0x0Fu]);
            g[2] = pack4((int) (char) kv[qb & 0x0Fu], (int) (char) kv[(qb >> 8) & 0x0Fu],
                         (int) (char) kv[(qb >> 16) & 0x0Fu], (int) (char) kv[(qb >> 24) & 0x0Fu]);
            g[3] = pack4((int) (char) kv[(qb >> 4) & 0x0Fu], (int) (char) kv[(qb >> 12) & 0x0Fu],
                         (int) (char) kv[(qb >> 20) & 0x0Fu], (int) (char) kv[(qb >> 28) & 0x0Fu]);
            int si = 0;
            si = dp4x8(g[0], (int) xw0, si);   // l = 0, low nibbles  <-> q8[iqs + 0]
            si = dp4x8(g[1], (int) xw4, si);   // l = 0, high nibbles <-> q8[iqs + 4]
            si = dp4x8(g[2], (int) xw1, si);   // l = 1, low nibbles  <-> q8[iqs + 1]
            si = dp4x8(g[3], (int) xw5, si);   // l = 1, high nibbles <-> q8[iqs + 5]
            const float dw = f16_to_f(rd16a(wr, base));
)CLC";
        case 21:   // vec_dot_iq3_s_q8_1; block: d(2) qs[64]@2 qh[8]@66 signs[32]@74 scales[4]@106
            return R"CLC(
            const uint qs0 = rd32(wr, base + 2u + 8u * m);       // qs[0..3] of the 8-byte packed pair
            const uint qs1 = rd32(wr, base + 2u + 8u * m + 4u);  // qs[4..7]
            const uint qh  = wr[base + 2u + 64u + m];
            const uint sgb = rd32(wr, base + 2u + 72u + 4u * m);
            int g[8];
            #pragma unroll
            for (int il = 0; il < 4; ++il) {
                // The CUDA walks the packed pair as `qs[l0 + 0]` and `qs[l0 + 1]` with l0 = 2*il, so the
                // eight bytes are taken in order: il 0/1 read the first word, il 2/3 the second, and the
                // low bit of il picks the half.  Reading qs1 for il 1 would land on qs[4] instead of qs[2].
                const uint wq = (il < 2) ? qs0 : qs1;
                const uint shx = 16u * ((uint) il & 1u);
                const uint u1 = ((wq >> shx) & 0xFFu) | ((qh << (8u - 2u * (uint) il)) & 0x100u);
                const uint u2 = ((wq >> (shx + 8u)) & 0xFFu) | ((qh << (7u - 2u * (uint) il)) & 0x100u);
                const uint sg = (sgb >> (8u * (uint) il)) & 0xFFu;
                const int n0 = vcmpne4((int) (((sg & 0x03u) << 7) | ((sg & 0x0Cu) << 21)), 0);
                const int n1 = vcmpne4((int) (((sg & 0x30u) << 3) | ((sg & 0xC0u) << 17)), 0);
                g[2 * il + 0] = vsub4((int) t3[u1] ^ n0, n0);
                g[2 * il + 1] = vsub4((int) t3[u2] ^ n1, n1);
            }
            const int ls = 1 + 2 * (int) ((wr[base + 106u + m / 2u] >> (4u * (m & 1u))) & 0x0Fu);
            int si = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) si = dp4x8(g[j], (int) xw[j], si);
            si *= ls;
            const float dw = f16_to_f(rd16(wr, base));
)CLC";
        case 22:   // vec_dot_iq2_s_q8_1 (SplitLs2::apply); block: d(2) qs[64]@2 qh[8]@66 scales[8]@74
            return R"CLC(
            const uint qs0 = rd32(wr, base + 2u + 4u * m);
            const uint qh  = wr[base + 2u + 64u + m];
            const uint sgb = rd32(wr, base + 2u + 32u + 4u * m);   // signs = qs + QK_K/8
            int g[8];
            #pragma unroll
            for (int il = 0; il < 4; ++il) {
                const uint idx = ((qs0 >> (8u * (uint) il)) & 0xFFu) | ((qh << (8u - 2u * (uint) il)) & 0x300u);
                const uint lo = t2[2u * idx + 0u];
                const uint hi = t2[2u * idx + 1u];
                const uint sg = (sgb >> (8u * (uint) il)) & 0xFFu;
                const int n0 = vcmpne4((int) (((sg & 0x03u) << 7) | ((sg & 0x0Cu) << 21)), 0);
                const int n1 = vcmpne4((int) (((sg & 0x30u) << 3) | ((sg & 0xC0u) << 17)), 0);
                g[2 * il + 0] = vsub4((int) lo ^ n0, n0);
                g[2 * il + 1] = vsub4((int) hi ^ n1, n1);
            }
            const uint sc = wr[base + 74u + m];
            const int ls0 = (int) (sc & 0x0Fu), ls1 = (int) (sc >> 4);
            int s0 = 0, s1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) s0 = dp4x8(g[j], (int) xw[j], s0);
            #pragma unroll
            for (int j = 4; j < 8; ++j) s1 = dp4x8(g[j], (int) xw[j], s1);
            const int si = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
            const float dw = f16_to_f(rd16(wr, base));
)CLC";
        case 23:   // vec_dot_iq4_xs_q8_1; block: d(2) scales_h(2)@2 scales_l[4]@4 qs[128]@8
            return R"CLC(
            int g[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint wv = rd32(wr, base + 8u + 16u * m + 4u * (uint) j);
                g[j + 0] = pack4((int) (char) kv[wv & 0x0Fu], (int) (char) kv[(wv >> 8) & 0x0Fu],
                                 (int) (char) kv[(wv >> 16) & 0x0Fu], (int) (char) kv[(wv >> 24) & 0x0Fu]);
                g[j + 4] = pack4((int) (char) kv[(wv >> 4) & 0x0Fu], (int) (char) kv[(wv >> 12) & 0x0Fu],
                                 (int) (char) kv[(wv >> 20) & 0x0Fu], (int) (char) kv[wv >> 28]);
            }
            const uint sh = rd16(wr, base + 2u);
            const int ls = (int) (((wr[base + 4u + m / 2u] >> (4u * (m & 1u))) & 0x0Fu) |
                                   (((sh >> (2u * m)) & 3u) << 4));
            int si = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                si = dp4x8(g[j], (int) xw[j], si);
                si = dp4x8(g[j + 4], (int) xw[j + 4], si);
            }
            si *= (ls - 32);
            const float dw = f16_to_f(rd16(wr, base));
)CLC";
        default:   // 42: vec_dot_q2_0_q8_1.  block_q2_0: fp16 d(2) qs[16]@2 - NOT a 4-byte float
            return R"CLC(
            const uint q0 = rd16(wr, base + 2u + 8u * m + 0u);
            const uint q1 = rd16(wr, base + 2u + 8u * m + 2u);
            const uint q2 = rd16(wr, base + 2u + 8u * m + 4u);
            const uint q3 = rd16(wr, base + 2u + 8u * m + 6u);
            int g[8];
            g[0] = pack4((int) ((q0 >> 0u) & 3u) - 1, (int) ((q0 >> 2u) & 3u) - 1,
                         (int) ((q0 >> 4u) & 3u) - 1, (int) ((q0 >> 6u) & 3u) - 1);
            g[1] = pack4((int) ((q0 >> 8u) & 3u) - 1, (int) ((q0 >> 10u) & 3u) - 1,
                         (int) ((q0 >> 12u) & 3u) - 1, (int) ((q0 >> 14u) & 3u) - 1);
            g[2] = pack4((int) ((q1 >> 0u) & 3u) - 1, (int) ((q1 >> 2u) & 3u) - 1,
                         (int) ((q1 >> 4u) & 3u) - 1, (int) ((q1 >> 6u) & 3u) - 1);
            g[3] = pack4((int) ((q1 >> 8u) & 3u) - 1, (int) ((q1 >> 10u) & 3u) - 1,
                         (int) ((q1 >> 12u) & 3u) - 1, (int) ((q1 >> 14u) & 3u) - 1);
            g[4] = pack4((int) ((q2 >> 0u) & 3u) - 1, (int) ((q2 >> 2u) & 3u) - 1,
                         (int) ((q2 >> 4u) & 3u) - 1, (int) ((q2 >> 6u) & 3u) - 1);
            g[5] = pack4((int) ((q2 >> 8u) & 3u) - 1, (int) ((q2 >> 10u) & 3u) - 1,
                         (int) ((q2 >> 12u) & 3u) - 1, (int) ((q2 >> 14u) & 3u) - 1);
            g[6] = pack4((int) ((q3 >> 0u) & 3u) - 1, (int) ((q3 >> 2u) & 3u) - 1,
                         (int) ((q3 >> 4u) & 3u) - 1, (int) ((q3 >> 6u) & 3u) - 1);
            g[7] = pack4((int) ((q3 >> 8u) & 3u) - 1, (int) ((q3 >> 10u) & 3u) - 1,
                         (int) ((q3 >> 12u) & 3u) - 1, (int) ((q3 >> 14u) & 3u) - 1);
            int si = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) si = dp4x8(g[j], (int) xw[j], si);
            const float dw = f16_to_f(rd16(wr, base));
)CLC";
    }
}

/// Which q8_1 words this format's call multiplies, and where the block is.
std::string fmt_activation(int type) {
    switch (type) {
        case 20:   // bq8_1 + kbx: the whole block, four of its eight words per call
            return R"CLC(
        const uint kbx = k >> 1u, m = k & 1u;
        const uint xo = kbx * 9u;
        const float dx = f16_to_f(XW[xo] & 0xFFFFu);
        const uint xw0 = XW[xo + 1u + 2u * m];
        const uint xw1 = XW[xo + 2u + 2u * m];
        const uint xw4 = XW[xo + 5u + 2u * m];
        const uint xw5 = XW[xo + 6u + 2u * m];
)CLC";
        case 42:   // bq8_1 + kbx*2 + iqs: one 64-value block covers two q8_1 blocks
            return R"CLC(
        const uint kbx = k >> 1u, m = k & 1u;
        const uint xo = (kbx * 2u + m) * 9u;
        const float dx = f16_to_f(XW[xo] & 0xFFFFu);
        uint xw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xw[j] = XW[xo + 1u + (uint) j];
)CLC";
        default:   // kbx*ipb + m: the block is numbered by 32-value position along the whole row
            return R"CLC(
        const uint kbx = k >> 3u, m = k & 7u;
        const uint xo = (kbx * 8u + m) * 9u;
        const float dx = f16_to_f(XW[xo] & 0xFFFFu);
        uint xw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xw[j] = XW[xo + 1u + (uint) j];
)CLC";
    }
}

/// Calls per row for IQ4_NL at n_ff == 640: (640 / 32) blocks * 2 calls each.  This is the only shape the
/// peer ships a dedicated kernel for, because the generic "one lane per call, then add" template cannot
/// reproduce its arithmetic (see row_source_iq4nl).
constexpr int kIq4nlCalls = 40;
/// `row_dot_40_sub8` for IQ4_NL, reproduced call for call.
///
/// The engine's 8-lane sub-warp dot is NOT the arithmetic of "one lane per call, then add them up".  Its
/// loop body is `s[c] += S::apply(...)`, which nvcc contracts into one `fma` per call, so the two calls
/// lane t owns (k = t and k = t + 32) chain through a single register with one rounding each.  Giving each
/// of those calls its own lane and adding them in the reduction tree rounds twice instead, which lands one
/// ulp away on about 12% of rows - measured, not guessed (opencl/probe/tree_variants.py).
///
/// So the shape here is the engine's own: 8 lanes carry one row (the reference's sub-warp), the loop's two
/// calls chain through `fma`, and the three remaining calls (t+16, t+8, t+24) come in as the plain adds the
/// reference's `__fadd_rn` makes them.  Then the 3-step xor butterfly `row_dot_40_sub8` ends with.
std::string row_source_iq4nl(int nchunk, bool act_identity, bool fold) {
    std::string s;
    s += "#pragma OPENCL FP_CONTRACT OFF\n";
    s += "#define IQ_WG " + std::to_string(g_iq4nl_wg) + "\n";
    s += "#define IQ_NCHUNK " + std::to_string(nchunk) + "\n";
    s += "#define IQ_BB 18\n";
    s += kHelpers;
    s += R"CLC(
// `wt` and its siblings are the arena's segments; `ent_off[e]` locates entry e's expert in one of them
// (segment in the top byte, offset below), `ent_act[e]` the activation row that entry multiplies (the
// token's q8_1 for gate/up, e itself for the down pass, where the activation is the entry's own SwiGLU
// output).  out row = e * out_stride + row.
//
// One call of `S::apply`, as `float2(scale, sumi)`: the scale has to stay separately rounded because the
// reference multiplies `r.dw * __low2float(bq8_1->ds)` first and only then folds it into the accumulator.
// The folded down pass reads its q8_1 straight out of the gate/up results instead of a buffer: each call
// re-quantises its own block in-lane, which is bit-identical (amax is order-insensitive, the block sum is
// never read) and takes the standalone SwiGLU kernel, its enqueue, and the gap behind it off the layer.
)CLC";
    if (!fold) {
        s += R"CLC(
inline float2 iq20_call(__global const uchar* wr, __global const uint* XW, __global const uchar* kv,
                        const uint k)
{
)CLC";
        s += fmt_activation(20);
        s += "        const uint base = kbx * (uint) IQ_BB;\n";
    } else {
        // The folded down pass: the call's activation IS q8_1 block kbx of the entry's own SwiGLU output,
        // recomputed from the gate/up results and quantised in-lane.  Same arithmetic as swiglu_q8_1
        // (FP_CONTRACT OFF, native_exp, ties-away round), bit-identical because the amax is
        // order-insensitive and the block sum is never read by the dot.  A row's five calls land in five
        // different blocks, so each re-quantises its own; the redundancy reads L1-resident floats and is
        // noise next to the weight traffic the call already moves.
        s += R"CLC(
inline float2 iq20_call(__global const uchar* wr, __global const uchar* kv,
                        const uint k, __global const float* gu_src, const uint n_ff, const uint e)
{
        const uint kbx = k >> 1u, m = k & 1u;
        __global const float* gp = gu_src + (size_t) e * 2u * (size_t) n_ff + (size_t) kbx * 32u;
        float amax = 0.0f;
        for (uint j = 0u; j < 32u; ++j) {
            const float g = gp[j];
            const float u = gp[(size_t) n_ff + j];
            amax = fmax(amax, fabs((g / (1.0f + native_exp(-g))) * u));
        }
        float dx = 0.0f;
        uint xw0 = 0u, xw1 = 0u, xw4 = 0u, xw5 = 0u;
        if (amax > 0.0f) {
            const float d = f16_max_clamp(amax / 127.0f);
            dx = f16_to_f(f32_to_f16(d));
            int q[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float g = gp[j];
                const float u = gp[(size_t) n_ff + j];
                q[j] = q8_1_quant((g / (1.0f + native_exp(-g))) * u, d, amax);
            }
            // the call takes words 1+2m / 2+2m / 5+2m / 6+2m of its block - values 8m..8m+7 and
            // 16+8m..16+8m+7, half the block; its partner call (k ^ 1) takes the other half
            if (m == 0u) {
                xw0 = pack4(q[0],  q[1],  q[2],  q[3]);  xw1 = pack4(q[4],  q[5],  q[6],  q[7]);
                xw4 = pack4(q[16], q[17], q[18], q[19]); xw5 = pack4(q[20], q[21], q[22], q[23]);
            } else {
                xw0 = pack4(q[8],  q[9],  q[10], q[11]); xw1 = pack4(q[12], q[13], q[14], q[15]);
                xw4 = pack4(q[24], q[25], q[26], q[27]); xw5 = pack4(q[28], q[29], q[30], q[31]);
            }
        }
        const uint base = kbx * (uint) IQ_BB;
)CLC";
    }
    s += fmt_decode(20);
    s += R"CLC(
    return (float2)(dw * dx, (float) si);
}

__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iq_row(__global const uchar* wt, __global const uint* xq,
            __global const uint* t2, __global const uint* t3, __global const uchar* kv,
            __global float* outv, __global const ulong* ent_off, __global const uint* ent_act,
            const uint n_entries, const uint bpe, const uint n_row_e,
            const uint wrow_bytes, const uint w_extra, const uint act_blk_row, const uint out_stride)CLC";
    s += kSegArgs;
    s += R"CLC()
{
    const uint lane = get_local_id(0);
    const uint e    = get_group_id(0) / bpe;
    if (e >= n_entries) return;                      // the grid is exact; the guard is per-group uniform
    const uint row0 = (get_group_id(0) - e * bpe) * (IQ_WG / 8u);
    const uint t    = lane & 7u;                     // the reference's sub-warp lane
    const uint row  = row0 + (lane >> 3u);
    const bool ok   = row < n_row_e;

)CLC";
    s += kSegPick;
    s += R"CLC(
    __global const uchar* W  = base + loc + w_extra;
)CLC";
    if (!fold)
        s += act_identity ? "    __global const uint*  XW = xq + (size_t) e * act_blk_row * 9u;\n"
                          : "    __global const uint*  XW = xq + (size_t) ent_act[e] * act_blk_row * 9u;\n";
    s += R"CLC(
    __global const uchar* wr = W + (size_t) (ok ? row : 0u) * (size_t) wrow_bytes;

    //   for (int k = t; k < 40; k += 32) s[c] += S::apply(w, ...);   -> one register, two fmAs
)CLC";
    if (fold) {
        s += "    const float2 wa = iq20_call(wr, kv, t, gu_src, n_ff, e);\n"
             "    const float2 wb = iq20_call(wr, kv, t + 32u, gu_src, n_ff, e);\n";
    } else {
        s += "    const float2 wa = iq20_call(wr, XW, kv, t);\n"
             "    const float2 wb = iq20_call(wr, XW, kv, t + 32u);\n";
    }
    s += R"CLC(
    float s = fma(wa.x, wa.y, 0.0f);
    s = fma(wb.x, wb.y, s);

    //   float s16 = 0; s16 += apply(...); s[c] = __fadd_rn(s[c], s16);
)CLC";
    s += fold ? "    const float2 wc = iq20_call(wr, kv, t + 16u, gu_src, n_ff, e);\n"
              : "    const float2 wc = iq20_call(wr, XW, kv, t + 16u);\n";
    s += R"CLC(
    s = s + (wc.x * wc.y);

    //   s8 and s24 likewise, then s[c] = __fadd_rn(s[c], __fadd_rn(s8[c], s24[c]));
)CLC";
    if (fold) {
        s += "    const float2 wd = iq20_call(wr, kv, t + 8u, gu_src, n_ff, e);\n"
             "    const float2 we = iq20_call(wr, kv, t + 24u, gu_src, n_ff, e);\n";
    } else {
        s += "    const float2 wd = iq20_call(wr, XW, kv, t + 8u);\n"
             "    const float2 we = iq20_call(wr, XW, kv, t + 24u);\n";
    }
    s += R"CLC(
    s = s + ((wd.x * wd.y) + (we.x * we.y));

    // the reference's closing butterfly is __shfl_xor over 4, 2, 1 - three steps inside the 8-lane group
    __local float lred[IQ_WG];
    lred[lane] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = 4u; off > 0u; off >>= 1u) {
        if (t < off) lred[lane] = lred[lane] + lred[lane + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0u && ok)
        outv[(size_t) e * out_stride + row] = lred[lane];
}
)CLC";
    return s;
}

/// The row kernel: one work-group = IQ_R rows of one entry, all IQ_WG lanes walking that entry's
/// activation together.  Rows of one entry share the activation load, which is what the R is for: at
/// 2-3.5 bits per weight the activation is several times the weight traffic.
std::string row_source(const KFmt& f, int nchunk, int rows, bool act_identity, int wg, bool fold) {
    if (f.type == 20 && nchunk == kIq4nlCalls) return row_source_iq4nl(nchunk, act_identity, fold);
    std::string s;
    s += "#pragma OPENCL FP_CONTRACT OFF\n";
    s += "#define IQ_WG " + std::to_string(wg) + "\n";
    s += "#define IQ_R " + std::to_string(rows) + "\n";
    s += "#define IQ_NCHUNK " + std::to_string(nchunk) + "\n";
    s += "#define IQ_BB " + std::to_string(f.bb) + "\n";
    s += kHelpers;
    s += R"CLC(
// `wt` and its siblings are the arena's segments; `ent_off[e]` locates entry e's expert in one of them
// (segment in the top byte, offset below), `ent_act[e]` the activation row that entry multiplies (the
// token's q8_1 for gate/up, e itself for the down pass, where the activation is the entry's own SwiGLU
// output).  out row = e * out_stride + row.
__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iq_row(__global const uchar* wt, __global const uint* xq,
            __global const uint* t2, __global const uint* t3, __global const uchar* kv,
            __global float* outv, __global const ulong* ent_off, __global const uint* ent_act,
            const uint n_entries, const uint bpe, const uint n_row_e,
            const uint wrow_bytes, const uint w_extra, const uint act_blk_row, const uint out_stride)CLC";
    s += kSegArgs;
    s += R"CLC()
{
    const uint lane = get_local_id(0);
    const uint e    = get_group_id(0) / bpe;
    if (e >= n_entries) return;                      // the grid is exact; the guard is per-group uniform
    const uint row0 = (get_group_id(0) - e * bpe) * (uint) IQ_R;
)CLC";
    s += kSegPick;
    s += R"CLC(
    __global const uchar* W  = base + loc + w_extra;
)CLC";
    if (!fold)
        s += act_identity ? "    __global const uint*  XW = xq + (size_t) e * act_blk_row * 9u;\n"
                          : "    __global const uint*  XW = xq + (size_t) ent_act[e] * act_blk_row * 9u;\n";
    s += R"CLC(
    float acc[IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) acc[r] = 0.0f;

    for (uint k = lane; k < (uint) IQ_NCHUNK; k += (uint) IQ_WG) {
)CLC";
    if (fold) {
        // The SwiGLU, folded into the down pass: chunk k IS one q8_1 block of the entry's own activation,
        // so its 32 values are recomputed from the gate/up results and quantised in-lane - the same
        // arithmetic as swiglu_q8_1 (FP_CONTRACT OFF, native_exp, ties-away round), minus the second
        // kernel, its enqueue, and the device-side gap that came between them.  The amax is
        // order-insensitive so a running max replaces the butterfly, and the block sum the standalone
        // kernel also stores is never read by the dot, so this is bit-identical to the two-kernel form.
        // The 32 h values are evaluated twice (once for the amax, once for the quant) rather than staged:
        // exp is cheaper than the local-memory round trip and the register pressure of 32 live floats.
        s += R"CLC(
        const uint kbx = k >> 3u, m = k & 7u;
        __global const float* gp = gu_src + (size_t) e * 2u * (size_t) n_ff + (size_t) k * 32u;
        float amax = 0.0f;
        for (uint j = 0u; j < 32u; ++j) {
            const float g = gp[j];
            const float u = gp[(size_t) n_ff + j];
            amax = fmax(amax, fabs((g / (1.0f + native_exp(-g))) * u));
        }
        float dx = 0.0f;
        uint xw[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) xw[j] = 0u;
        if (amax > 0.0f) {
            const float d = f16_max_clamp(amax / 127.0f);
            dx = f16_to_f(f32_to_f16(d));
            for (uint j = 0u; j < 32u; ++j) {
                const float g = gp[j];
                const float u = gp[(size_t) n_ff + j];
                const int q = q8_1_quant((g / (1.0f + native_exp(-g))) * u, d, amax);
                xw[j >> 2u] |= ((uint) q & 0xFFu) << (8u * (j & 3u));
            }
        }
)CLC";
    } else {
        s += fmt_activation(f.type);
    }
    s += R"CLC(
        #pragma unroll
        for (int r = 0; r < IQ_R; ++r) {
            const uint row = min(row0 + (uint) r, n_row_e - 1u);
            __global const uchar* wr = W + (size_t) row * (size_t) wrow_bytes;
            const uint base = kbx * (uint) IQ_BB;
)CLC";
    s += fmt_decode(f.type);
    s += R"CLC(
            // --use_fast_math makes the CUDA accumulation a mul.f32 then an fma.rn.f32; dw*dx rounds on
            // its own and the accumulate rounds once.  FP_CONTRACT OFF keeps that split.
            acc[r] = fma(dw * dx, (float) si, acc[r]);
        }
    }

    // The reference tree, spelled out: IQ_WG/2 .. 4, 2, 1.  Every row is reduced at once so the
    // barriers are paid once per work-group instead of once per row.
    __local float lred[IQ_WG * IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) lred[lane * IQ_R + r] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint off = IQ_WG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            #pragma unroll
            for (int r = 0; r < IQ_R; ++r)
                lred[lane * IQ_R + r] = lred[lane * IQ_R + r] + lred[(lane + off) * IQ_R + r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0u) {
        #pragma unroll
        for (int r = 0; r < IQ_R; ++r)
            if (row0 + (uint) r < n_row_e)
                outv[(size_t) e * out_stride + row0 + (uint) r] = lred[r];
    }
}
)CLC";
    return s;
}

/// x (n_tok rows of n_embd floats, row-major and contiguous) -> q8_1, one work-group per 32-value block.
std::string quant_source() {
    std::string s;
    s += "#pragma OPENCL FP_CONTRACT OFF\n";
    s += kHelpers;
    s += R"CLC(
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void quant_rows(__global const float* x, __global uchar* xq)
{
    const uint b = get_group_id(0);              // the flat q8_1 block == the flat 32 values / 32
    const uint lane = get_local_id(0);
    q8_1_store(x[b * 32u + lane], lane, (__global uint*) (xq + (size_t) b * 36u), xq + (size_t) b * 36u + 4u);
}
)CLC";
    return s;
}

/// SwiGLU + q8_1: gu[e] is [gate n_ff | up n_ff] for entry e; the output is n_ff/32 q8_1 blocks.
std::string swiglu_source() {
    std::string s;
    s += "#pragma OPENCL FP_CONTRACT OFF\n";     // the product must round on its own - __fmul_rn on CUDA
    s += kHelpers;
    s += R"CLC(
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void swiglu_q8_1(__global const float* gu, __global uchar* hq, const uint n_ff)
{
    const uint b    = get_group_id(0);           // flat q8_1 block over every entry
    const uint lane = get_local_id(0);
    const uint bpe  = n_ff >> 5u;
    const uint e    = b / bpe;
    const uint blk  = b - e * bpe;
    __global const float* gp = gu + (size_t) e * 2u * n_ff + blk * 32u;
    const float g = gp[lane];
    const float u = gp[n_ff + lane];
    // __expf is native_exp's counterpart; the quantiser is insensitive to which of the two is used
    // (ocl_swiglu.cpp: 0 code differences over 8 reduction/exp combinations), so the fast one is right.
    const float h = (g / (1.0f + native_exp(-g))) * u;
    q8_1_store(h, lane, (__global uint*) (hq + (size_t) b * 36u), hq + (size_t) b * 36u + 4u);
}
)CLC";
    return s;
}

// ---------------------------------------------------------------- small OpenCL plumbing

std::string cl_err(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        default: break;
    }
    return "cl error " + std::to_string((int) e);
}

struct Prog {
    cl_program p = nullptr;
    cl_kernel k = nullptr;
    void release() {
        if (k) clReleaseKernel(k);
        if (p) clReleaseProgram(p);
        k = nullptr;
        p = nullptr;
    }
};

/// clSetKernelArg, once per argument, stopping at the first failure so a launch reads as a list of binds.
struct Args {
    cl_kernel k = nullptr;
    cl_int rc = CL_SUCCESS;
    explicit Args(cl_kernel kk) : k(kk) {}
    Args& mem(cl_uint i, cl_mem m) { return raw(i, sizeof(cl_mem), &m); }
    Args& u32(cl_uint i, cl_uint v) { return raw(i, sizeof(cl_uint), &v); }
    Args& raw(cl_uint i, size_t sz, const void* p) {
        if (rc == CL_SUCCESS) rc = clSetKernelArg(k, i, sz, p);
        return *this;
    }
    bool done(const char* what, std::string& err) const {
        if (rc == CL_SUCCESS) return true;
        err = std::string("peer: binding ") + what + ": " + cl_err(rc);
        return false;
    }
};

cl_command_queue make_queue(cl_context ctx, cl_device_id dev) {
    cl_int rc = CL_SUCCESS;
    // PROFILING, because STRATA_OCL_STATS can say that a layer's GPU time is 0.9 ms but not which of the
    // three kernels owns it, and the three want completely different fixes: the row kernel wants a wider
    // grid, the quantiser wants fewer work-groups, and the SwiGLU wants to disappear into the row kernel.
    // The properties array is name/value pairs, not a bitfield: CL_QUEUE_PROPERTIES is the name.
    const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &rc);
    return rc == CL_SUCCESS ? q : nullptr;
}

}  // namespace

// ---------------------------------------------------------------- Impl

struct OclPeer::Impl {
    int index = -1;
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;        // the layer's own pipeline, in order
    cl_command_queue refill = nullptr;   // adapt()'s copies, so they never stall the pipeline
    cl_event refill_ev = nullptr;

    // The arena, and where each slot lives in it.
    //
    // NOT one allocation.  The device reports a 13.60 GiB maximum single allocation and the tier used to
    // believe it, which threw away more than half of the card: this gfx906 streams ~28 GiB at 633 GB/s and
    // only past that does the driver spill into system RAM (24 GB/s, measured in ocl_alloc.cpp).  So the
    // arena is a SET of segments, each inside the reported maximum, and a slot is (segment, offset) with the
    // segment in the top byte of the value the kernel reads from `ent_off`.
    static constexpr int kSegs = 8;
    cl_mem arena = nullptr;                      // == arenas[0]
    cl_mem arenas[kSegs] = {};
    int n_seg = 0;
    uint64_t seg_bytes = 0;                      // each segment is the same size
    uint64_t arena_bytes = 0, arena_used = 0;
    std::vector<uint64_t> slot_off;              // offset WITHIN its segment
    std::vector<int32_t> slot_seg;
    Prog touch;                                  // the residency check's streaming kernel

    // per-layer formats and sizes
    std::vector<OclLayerInfo> layers;
    int64_t n_layers = 0, n_expert = 0;
    std::vector<int64_t> gu_chunk, d_chunk;          // calls per row
    std::vector<uint64_t> gu_row, d_row, down_off;   // row bytes, and the down offset
    std::vector<uint64_t> blob_bytes;                // == layers[l].blob_bytes by construction
    int64_t max_embd = 0, max_ff = 0;

    // work buffers
    cl_mem xqbuf = nullptr, gubuf = nullptr, hqbuf = nullptr, dobuf = nullptr;
    // one packed input upload per layer: [ent_off | ent_act | x] in a single device buffer, with the first
    // three regions bound to the kernels as sub-buffer views so the kernel sources do not change at all
    cl_mem inbuf = nullptr, xsub = nullptr, ent_off = nullptr, ent_act = nullptr;
    unsigned char* in_host = nullptr;     // the staging the packed upload is written from
    cl_uint off_cap = 0, act_cap = 0, x_base = 0, x_cap = 0;
    cl_mem outstg = nullptr;
    float* out_host = nullptr;            // outstg, mapped: a page-locked read-back target
    cl_mem t_iq3xxs = nullptr, t_iq2s = nullptr, t_iq3s = nullptr, t_kv = nullptr;

    std::map<uint64_t, Prog> progs;      // key = type * 2^20 + nchunk
    Prog quant, swiglu;

    // per-launch scratch on the host
    std::vector<uint64_t> h_off;
    std::vector<uint32_t> h_act, h_ident;
    std::vector<int32_t> h_tok, h_seen;
    int64_t launched_rows = 0;
    int64_t last_layer = -1;
    int64_t last_act_d = 0;              // q8_1 blocks per entry in the last down pass (STRATA_OCL_DUMP_HQ)
    /// STRATA_OCL_STATS: where the tier's per-layer time goes.  `ms_enq` is the host side of `launch`
    /// (everything the caller pays before the GPU is even asked), `ms_read` the read-back in `finish`.
    /// The two add up to what the engine's `dispatch run` phase attributes to the tier.
    double ms_enq = 0, ms_read = 0;
    int64_t stat_layers = 0, stat_rows = 0, stat_entries = 0;
    /// The result read-back, ISSUED AT THE END OF launch.  It has to be issued there and not in `finish`
    /// for the tier to be worth anything: `launch` runs before the engine computes its own CPU rows for the
    /// same layer, so a copy issued there overlaps that work, while one issued in `finish` runs after it and
    /// puts the whole device->host transfer - and the driver's queue drain around it, measured at ~0.75
    /// ms/layer against ~0.03 ms of host-side enqueue - on the critical path.
    cl_event read_ev = nullptr;

    /// STRATA_OCL_STATS, part two: one profiling event per kernel so the device's time can be split by
    /// pass.  `finish` waits on `read_ev`, which is ordered after all of them, so by the time it resolves
    /// these the timestamps are final.
    cl_event ev_quant = nullptr, ev_gu = nullptr, ev_swi = nullptr, ev_down = nullptr;
    double ms_quant = 0, ms_gu = 0, ms_swi = 0, ms_down = 0;
    /// And the parts of `finish` that `ms_read` lumps together: how long the host actually blocked, how
    /// long the device->host copy was in flight, and how much of the wait was the queue draining before
    /// the copy could even start.  Those three want different fixes, so they get different counters.
    double ms_wait = 0, ms_copy_wait = 0, ms_gap = 0, ms_host_gap = 0, ms_prof = 0, ms_copy = 0, ms_seg1 = 0;
    int64_t stat_kern = 0;
    /// When `launch` finished submitting, so `finish` can say how long the host had of its own work before
    /// it came back to collect the rows.  That number against the device's own duration is the whole story:
    /// the tier is free if it is longer than the device's time, and the residual wait if it is shorter.
    std::chrono::steady_clock::time_point t_enq;
    bool have_enq = false;

    // residency, as PeerExperts keeps it
    std::vector<int32_t> res;                                  // [n_layers * n_expert] -> slot or -1
    std::vector<std::pair<int32_t, int32_t>> pending;          // (res index, slot) once the copies land
    BlobFn blob = nullptr;
    void* blob_ctx = nullptr;

    Impl() = default;
    ~Impl() { release(); }                  // every open()'s failure path can then just return false
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void release() {
        for (auto& kv : progs) kv.second.release();
        progs.clear();
        quant.release();
        swiglu.release();
        touch.release();
        if (refill_ev) clReleaseEvent(refill_ev);
        refill_ev = nullptr;
        if (read_ev) clReleaseEvent(read_ev);
        read_ev = nullptr;
        cl_event* evs[] = {&ev_quant, &ev_gu, &ev_swi, &ev_down};
        for (cl_event* e : evs)
            if (*e) { clReleaseEvent(*e); *e = nullptr; }
        cl_mem* bufs[] = {&xsub, &inbuf, &xqbuf, &gubuf, &hqbuf, &dobuf, &ent_off, &ent_act, &outstg,
                          &t_iq3xxs, &t_iq2s, &t_iq3s, &t_kv};
        for (cl_mem* b : bufs)
            if (*b) { clReleaseMemObject(*b); *b = nullptr; }
        if (in_host) { std::free(in_host); in_host = nullptr; }
        for (int i = 0; i < kSegs; ++i)
            if (arenas[i]) { clReleaseMemObject(arenas[i]); arenas[i] = nullptr; }
        arena = nullptr;
        n_seg = 0;
        if (q) clReleaseCommandQueue(q);
        if (refill) clReleaseCommandQueue(refill);
        if (ctx) clReleaseContext(ctx);
        q = refill = nullptr;
        ctx = nullptr;
        dev = nullptr;
        plat = nullptr;
        out_host = nullptr;
    }

    /// Compiles one program and takes its kernel.  The build options are exactly the probe's: no
    /// fast-relaxed-math (round() and the half conversions have to stay IEEE) and no -cl-mad-enable
    /// (the accumulate is an explicit fma under FP_CONTRACT OFF, and nothing else may be contracted).
    bool build(Prog& pr, const std::string& src, const char* kname, const char* what, std::string& err) {
        if (std::getenv("STRATA_OCL_DUMP_SRC") != nullptr) {   // STRATA_OCL_DUMP_SRC=1 writes what the driver sees
            std::string fn = std::string("ocl_src_") + what + ".cl";
            for (char& c : fn)
                if (c == '/' || c == '\\' || c == ':' || c == ' ') c = '_';
            if (FILE* fh = std::fopen(fn.c_str(), "wb")) {
                std::fwrite(src.data(), 1, src.size(), fh);
                std::fclose(fh);
            }
        }
        const char* sp = src.c_str();
        cl_int rc = CL_SUCCESS;
        pr.p = clCreateProgramWithSource(ctx, 1, &sp, nullptr, &rc);
        if (rc != CL_SUCCESS) {
            err = std::string("peer: clCreateProgramWithSource for ") + what + ": " + cl_err(rc);
            return false;
        }
        const cl_int br = clBuildProgram(pr.p, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
        if (br != CL_SUCCESS) {
            size_t n = 0;
            clGetProgramBuildInfo(pr.p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::vector<char> log(n + 1, 0);
            if (n > 0) clGetProgramBuildInfo(pr.p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            err = std::string("peer: building ") + what + ": " + cl_err(br) + "\n" + log.data();
            pr.release();
            return false;
        }
        pr.k = clCreateKernel(pr.p, kname, &rc);
        if (rc != CL_SUCCESS) {
            err = std::string("peer: the ") + kname + " kernel: " + cl_err(rc);
            pr.release();
            return false;
        }
        return true;
    }

    /// The program (and kernel) for `type` at `nchunk` calls per row, built on first use.  One program
    /// per (format, row length) pair: the loop bound is a compile-time constant, which is what lets the
    /// unroll below see all of it.
    Prog* row_prog(int type, int nchunk, int rows, bool act_identity, bool fold, std::string& err) {
        const uint64_t key = ((uint64_t) (uint32_t) type << 32) | ((uint64_t) (uint32_t) rows << 24) |
                             ((uint64_t) (act_identity ? 1u : 0u) << 23) |
                             ((uint64_t) (fold ? 1u : 0u) << 22) | (uint64_t) (uint32_t) nchunk;
        auto it = progs.find(key);
        if (it != progs.end()) return it->second.k ? &it->second : nullptr;
        const KFmt* f = fmt_of(type);
        if (f == nullptr) { err = "peer: no kernel for format " + std::to_string(type); return nullptr; }
        if (fold && !act_identity) {
            err = "peer: the folded SwiGLU is a down-pass feature";
            return nullptr;
        }
        Prog& pr = progs[key];
        // the width is a pure function of (format, nchunk) - the cache key above needs no extra field
        const int wg = row_wg(*f, nchunk);
        const std::string what = std::string(f->name) + " at " + std::to_string(nchunk) + " calls/row, " +
                                 std::to_string(rows) + " rows/group, WG " + std::to_string(wg) +
                                 (fold ? ", folded SwiGLU" : "");
        if (!build(pr, row_source(*f, nchunk, rows, act_identity, wg, fold), "iq_row", what.c_str(), err)) {
            progs.erase(key);
            return nullptr;
        }
        return &pr;
    }

    /// The weight-side codebook a format's decode indexes through the kernel's `t2` argument.  IQ3_S reads
    /// `t3` instead and IQ4_NL / IQ4_XS read `kv`, so those three get any valid buffer here.
    cl_mem table_for(int type) const {
        if (type == 18) return t_iq3xxs;
        if (type == 22) return t_iq2s;
        return t_iq3s;
    }

    /// One pass of the row kernel for every pending entry.  The full grid is n_entries * ceil(n_row_e/kRows)
    /// work-groups of the format's width; each work-group owns kRows rows of one entry and walks that
    /// entry's activation, which is why the activation is read once per kRows rows instead of once per row.
    struct RowJob {
        int type = 0, nchunk = 0;
        cl_mem xq = nullptr, outv = nullptr;
        uint64_t w_extra = 0;
        uint32_t n_row_e = 0, wrow_bytes = 0, act_blk_row = 0, out_stride = 0, n_entries = 0;
        cl_event* ev = nullptr;          /// profiling handle, STRATA_OCL_STATS
        /// The down pass multiplies each entry by its own SwiGLU row, i.e. `ent_act[e] == e`, so the kernel
        /// can derive the index instead of reading a table.  That is one of the nine enqueues a layer's
        /// launch costs, and on this driver an enqueue is ~60 us of host time - the tier's whole host budget
        /// is 48 layers of them.
        bool act_identity = false;
        /// Fold the SwiGLU into the row pass: the down kernel recomputes its own activation from the
        /// gate/up results and quantises it in-lane, instead of reading a separate kernel's output.  One
        /// enqueue and one device-side gap fewer per layer, bit-identical (the amax is order-insensitive
        /// and the block sum the standalone kernel stores is never read by the dot).
        bool fold_swiglu = false;
        cl_mem gu = nullptr;             /// the gate/up results, read only by the folded down pass
        uint32_t n_ff = 0;               /// n_ff of the folded down pass
    };

    bool run_row(const RowJob& J, std::string& err) {
        if (J.n_entries == 0) return true;
        const cl_uint gr = (cl_uint) grp_rows(J.type);
        Prog* pr = row_prog(J.type, J.nchunk, (int) gr, J.act_identity, J.fold_swiglu, err);
        if (pr == nullptr) return false;
        const KFmt& f = *fmt_of(J.type);
        const cl_mem t2 = table_for(J.type);
        Args A(pr->k);
        A.mem(0, arenas[0]).mem(1, J.xq).mem(2, t2).mem(3, t_iq3s).mem(4, t_kv)
            .mem(5, J.outv).mem(6, ent_off).mem(7, ent_act);
        A.u32(8, J.n_entries);
        A.u32(9, (J.n_row_e + gr - 1u) / gr);
        A.u32(10, J.n_row_e);
        A.u32(11, J.wrow_bytes);
        A.u32(12, (cl_uint) J.w_extra);
        A.u32(13, J.act_blk_row);
        A.u32(14, J.out_stride);
        // one pointer per extra segment, appended so the arguments above keep their indices; a segment the
        // arena does not have is bound to segment 0's buffer so the argument is never null
        for (int i = 1; i < kSegs; ++i) A.mem(14 + (cl_uint) i, arenas[i] != nullptr ? arenas[i] : arenas[0]);
        // the folded down pass reads the gate/up results through these; every row kernel has the two
        // arguments, so the binding is uniform and an unfolded kernel simply never touches them
        A.mem(22, J.gu != nullptr ? J.gu : gubuf).u32(23, J.n_ff);
        if (!A.done(("the row kernel (" + std::string(f.name) + ")").c_str(), err)) return false;
        const cl_uint bpe = (J.n_row_e + gr - 1u) / gr;
        // the launch width MUST be the width the program was built at, which is row_wg's decision (the
        // cache key above makes it a pure function of this call's format and call count)
        const size_t wg = (size_t) row_wg(f, J.nchunk);
        const size_t gsz = (size_t) (J.n_entries * bpe) * wg, lsz = wg;
        const cl_int kr = clEnqueueNDRangeKernel(q, pr->k, 1, nullptr, &gsz, &lsz, 0, nullptr, J.ev);
        if (kr != CL_SUCCESS) {
            err = "peer: launching the row kernel (" + std::string(f.name) + "): " + cl_err(kr);
            return false;
        }
        return true;
    }

    /// The activation quantiser and the SwiGLU, whose buffers never change; only n_ff does.
    bool bind_quant(std::string& err) {
        return Args(quant.k).mem(0, xsub).mem(1, xqbuf).done("quant_rows", err);
    }
    bool bind_swiglu(cl_uint n_ff, std::string& err) {
        return Args(swiglu.k).mem(0, gubuf).mem(1, hqbuf).u32(2, n_ff).done("swiglu_q8_1", err);
    }
};

// ---------------------------------------------------------------- static probing

namespace {

/// The AMD platform's devices, or nothing.  A platform whose vendor is not AMD is skipped: the tier is
/// for the MI50, and picking up an Intel or Microsoft runtime by accident would be worse than declining.
cl_platform_id amd_platform() {
    cl_uint n = 0;
    if (clGetPlatformIDs(0, nullptr, &n) != CL_SUCCESS || n == 0) return nullptr;
    std::vector<cl_platform_id> plats(n);
    clGetPlatformIDs(n, plats.data(), nullptr);
    for (cl_uint i = 0; i < n; ++i) {
        size_t sz = 0;
        if (clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) continue;
        std::vector<char> v(sz + 1, 0);
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, sz, v.data(), nullptr);
        if (std::strstr(v.data(), "Advanced Micro") != nullptr) return plats[i];
    }
    return nullptr;
}

std::vector<cl_device_id> amd_devices(cl_platform_id p) {
    std::vector<cl_device_id> out;
    if (p == nullptr) return out;
    cl_uint n = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &n) != CL_SUCCESS || n == 0) return out;
    out.resize(n);
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, n, out.data(), nullptr) != CL_SUCCESS) out.clear();
    return out;
}

std::string dev_str(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::vector<char> v(n + 1, 0);
    clGetDeviceInfo(d, what, n, v.data(), nullptr);
    return std::string(v.data());
}

}  // namespace

int OclPeer::count() { return (int) amd_devices(amd_platform()).size(); }

bool OclPeer::probe(int index, std::string& name, double& gib, double& max_alloc_gib) {
    const auto devs = amd_devices(amd_platform());
    if (devs.empty()) return false;
    const int i = index < 0 ? 0 : index;
    if (i >= (int) devs.size()) return false;
    cl_device_id d = devs[(size_t) i];
    name = dev_str(d, CL_DEVICE_NAME);
    cl_ulong g = 0, m = 0;
    clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(g), &g, nullptr);
    clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(m), &m, nullptr);
    gib = (double) g / 1073741824.0;
    max_alloc_gib = (double) m / 1073741824.0;
    return true;
}

bool OclPeer::layer_supported(const OclLayerInfo& L, std::string& why) {
    for (int role = 0; role < 2; ++role) {
        const int t = role == 0 ? L.gu_type : L.d_type;
        const int64_t vals = role == 0 ? L.n_embd : L.n_ff;
        const KFmt* f = fmt_of(t);
        if (f == nullptr) {
            why = "the peer has no " + std::string(role == 0 ? "gate/up" : "down") + " kernel for format " +
                  std::to_string(t);
            return false;
        }
        if (vals % f->qk != 0 || vals % 32 != 0) {
            why = std::string(f->name) + " does not divide " + std::to_string(vals) + " values per row";
            return false;
        }
        (void) row_bytes(*f, vals);
    }
    return true;
}

// ---------------------------------------------------------------- lifecycle

OclPeer::OclPeer() = default;
OclPeer::~OclPeer() { close(); }

bool OclPeer::has(int64_t layer, int64_t expert) const {
    if (!p_ || layer < 0 || layer >= p_->n_layers || expert < 0 || expert >= p_->n_expert) return false;
    return p_->res[(size_t) (layer * p_->n_expert + expert)] >= 0;
}

// ---------------------------------------------------------------- open

bool OclPeer::open(int index, const std::vector<OclLayerInfo>& layers, int64_t n_expert, BlobFn blob, void* ctx,
                   const int8_t* primary_slot, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                   int reserve_mib, int64_t max_slots, std::string& err) {
    close();
    if (layers.empty()) { err = "peer: no layers"; return false; }
    if (n_expert <= 0 || n_expert > (int64_t) 1 << 20) { err = "peer: bad expert count"; return false; }
    for (size_t l = 0; l < layers.size(); ++l)
        if (!layer_supported(layers[l], err)) return false;

    auto im = std::make_unique<Impl>();
    im->index = index;
    im->layers = layers;
    im->n_layers = (int64_t) layers.size();
    im->n_expert = n_expert;
    im->blob = blob;
    im->blob_ctx = ctx;
    // Before a single program is built (the numbers go into the generated source as #defines).
    g_rows = rows_from_env("STRATA_OCL_ROWS", kRows);
    {
        const int wg = rows_from_env("STRATA_OCL_IQ4NL_WG", kIq4nlWg);
        g_iq4nl_wg = (wg % 8 == 0 && wg >= 8 && wg <= 256) ? wg : kIq4nlWg;
    }

    // ---- the platform, the device, and the two queues
    im->plat = amd_platform();
    if (im->plat == nullptr) { err = "peer: no AMD OpenCL platform (is the driver installed?)"; return false; }
    const auto devs = amd_devices(im->plat);
    const int di = index < 0 ? 0 : index;
    if (di >= (int) devs.size()) {
        err = "peer: OpenCL device " + std::to_string(di) + " is not there (" + std::to_string(devs.size()) +
              " found)";
        return false;
    }
    im->dev = devs[(size_t) di];
    index_ = di;
    name_ = dev_str(im->dev, CL_DEVICE_NAME);
    cl_int rc = CL_SUCCESS;
    im->ctx = clCreateContext(nullptr, 1, &im->dev, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS) { err = "peer: clCreateContext: " + cl_err(rc); return false; }
    im->q = make_queue(im->ctx, im->dev);
    im->refill = make_queue(im->ctx, im->dev);
    if (im->q == nullptr || im->refill == nullptr) { err = "peer: creating the command queues"; return false; }

    // ---- per-layer sizes, and the check that they agree with the pack's own blob size.  This is the one
    // place where a wrong idea about the layout can still be caught cheaply: the engine hands over the pack's
    // blob_bytes, and this file's arithmetic has to land on the same number.
    im->gu_chunk.resize((size_t) im->n_layers);
    im->d_chunk.resize((size_t) im->n_layers);
    im->gu_row.resize((size_t) im->n_layers);
    im->d_row.resize((size_t) im->n_layers);
    im->down_off.resize((size_t) im->n_layers);
    im->blob_bytes.resize((size_t) im->n_layers);
    for (int64_t l = 0; l < im->n_layers; ++l) {
        const OclLayerInfo& L = layers[(size_t) l];
        const KFmt& g = *fmt_of(L.gu_type);
        const KFmt& d = *fmt_of(L.d_type);
        const size_t li = (size_t) l;
        im->gu_chunk[li] = calls_per_row(g, L.n_embd);
        im->d_chunk[li] = calls_per_row(d, L.n_ff);
        im->gu_row[li] = row_bytes(g, L.n_embd);
        im->d_row[li] = row_bytes(d, L.n_ff);
        im->down_off[li] = 2ull * (uint64_t) L.n_ff * im->gu_row[li];
        im->blob_bytes[li] = im->down_off[li] + (uint64_t) L.n_embd * im->d_row[li];
        im->max_embd = std::max(im->max_embd, L.n_embd);
        im->max_ff = std::max(im->max_ff, L.n_ff);
        if (im->blob_bytes[li] != L.blob_bytes) {
            err = "peer: layer " + std::to_string(l) + " blob is " + std::to_string(L.blob_bytes) +
                  " bytes in the pack but " + std::to_string(im->blob_bytes[li]) +
                  " by this file's arithmetic (" + g.name + "/" + d.name + " at n_embd " +
                  std::to_string(L.n_embd) + ", n_ff " + std::to_string(L.n_ff) + ")";
            return false;
        }
        if ((L.n_embd / 32) * 32 != L.n_embd || (L.n_ff / 32) * 32 != L.n_ff) {
            err = "peer: layer " + std::to_string(l) + " has dimensions the q8_1 blocks cannot cover";
            return false;
        }
    }

    // ---- the arena.  A SET of allocations, not one.
    //
    // The device reports a 13.60 GiB maximum single allocation, and this code used to believe it, which threw
    // away more than half of the card: ocl_alloc.cpp streams ~28 GiB of this gfx906 at 633 GB/s and only past
    // that does the driver start spilling into system RAM - the buffer is still readable, it just reads at 24
    // GB/s, which would quietly make the tier worse than the CPU it is meant to relieve.  So the arena is
    // split into segments, each inside the reported maximum, and how many are kept is decided by MEASURING
    // how fast each one streams rather than by trusting CL_DEVICE_GLOBAL_MEM_SIZE.
    cl_ulong max_alloc = 0, global = 0;
    clGetDeviceInfo(im->dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr);
    clGetDeviceInfo(im->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global), &global, nullptr);
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    uint64_t budget = 0;
    {
        const char* env = std::getenv("STRATA_OCL_ARENA_GIB");   // the TOTAL the tier may hold, not one segment
        const double want = env != nullptr ? std::atof(env) : 0.0;
        budget = want > 0.0 ? (uint64_t) (want * 1073741824.0) : (uint64_t) global;
    }
    budget = budget > reserve ? budget - reserve : 0;
    // Segments the size of an even share of the budget: small enough that the last one is not thrown away
    // whole when the resident limit falls inside it, and never larger than one allocation may be.
    uint64_t seg_want = std::min(budget / (uint64_t) Impl::kSegs, (uint64_t) max_alloc);
    seg_want &= ~(uint64_t) 0xFF;                     // 256-byte slots
    if (seg_want < (uint64_t) 64 << 20) {
        err = "peer: only " + std::to_string(budget >> 20) + " MiB fits (" + name_ + ": " +
              std::to_string(max_alloc >> 20) + " MiB max allocation, " + std::to_string(global >> 20) +
              " MiB total)";
        return false;
    }

    // ---- how many segments are really resident
    //
    // The streaming kernel is trivial on purpose: it touches every 4-byte word, so what it measures is the
    // memory and not the kernel.  A segment that comes back at hundreds of GB/s is on the card; one that
    // comes back at tens has been spilled to system RAM and is dropped, together with everything after it.
    {
        static const char* touch_src = R"CLC(
__kernel void peer_touch(__global float* p, const ulong n) {
    const ulong i = get_global_id(0);
    if (i < n) p[i] = p[i] * 1.5f + 1.0f;
}
)CLC";
        if (!im->build(im->touch, touch_src, "peer_touch", "peer_touch", err)) return false;
    }
    double slowest = 1e9;
    for (int i = 0; i < Impl::kSegs; ++i) {
        cl_int r = CL_SUCCESS;
        cl_mem m = clCreateBuffer(im->ctx, CL_MEM_READ_WRITE, (size_t) seg_want, nullptr, &r);
        if (r != CL_SUCCESS) break;                  // no more room: the segments already taken are the budget
        const size_t n = (size_t) seg_want / sizeof(float);
        clSetKernelArg(im->touch.k, 0, sizeof(cl_mem), &m);
        clSetKernelArg(im->touch.k, 1, sizeof(cl_ulong), &n);
        const size_t gsz = (n + 255) & ~(size_t) 255;
        cl_event ev = nullptr;
        if (clEnqueueNDRangeKernel(im->q, im->touch.k, 1, nullptr, &gsz, nullptr, 0, nullptr, &ev) != CL_SUCCESS ||
            clWaitForEvents(1, &ev) != CL_SUCCESS) {
            if (ev != nullptr) clReleaseEvent(ev);
            clReleaseMemObject(m);
            break;
        }
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        const double sec = (double) (t1 - t0) * 1e-9;
        const double gbs = sec > 0 ? (double) seg_want * 2.0 / sec / 1e9 : 0.0;
        if (std::getenv("STRATA_OCL_ARENA_VERBOSE") != nullptr)
            std::fprintf(stderr, "strata peer:   segment %d of %.2f GiB streams at %.1f GB/s\n", i,
                         (double) seg_want / 1073741824.0, gbs);
        if (gbs < 150.0) {                           // resident VRAM is hundreds; spilt memory is tens
            clReleaseMemObject(m);
            break;
        }
        im->arenas[i] = m;
        im->n_seg = i + 1;
        slowest = std::min(slowest, gbs);
    }
    if (im->n_seg == 0) { err = "peer: not one arena segment fits on " + name_; return false; }
    im->arena = im->arenas[0];
    im->seg_bytes = seg_want;
    im->arena_bytes = seg_want * (uint64_t) im->n_seg;
    budget = im->arena_bytes;
    (void) slowest;

    // ---- the pairs, in rank order, that the primary does not hold and that fit.  `res` doubles as the
    // "already taken" map here, so a pair that appears twice in `ranked` is only picked once.
    struct Pick { int32_t layer, expert; uint64_t off, slot; int32_t seg; };
    std::vector<Pick> picks;
    im->res.assign((size_t) (im->n_layers * im->n_expert), -1);
    int32_t cur_seg = 0;
    uint64_t cur_off = 0;
    for (const auto& pr : ranked) {
        if (pr.first < 0 || pr.first >= im->n_layers || pr.second < 0 || pr.second >= im->n_expert) continue;
        int32_t& rs = im->res[(size_t) pr.first * (size_t) im->n_expert + (size_t) pr.second];
        if (rs >= 0) continue;                                     // already picked
        if (primary_slot != nullptr &&
            primary_slot[(size_t) pr.first * (size_t) im->n_expert + (size_t) pr.second] >= 0)
            continue;                                              // the primary's VRAM tier has it
        const uint64_t b = im->blob_bytes[(size_t) pr.first];
        const uint64_t b256 = (b + 255u) & ~(uint64_t) 255;
        if (b256 > im->seg_bytes) continue;                        // one blob larger than a whole segment
        if (cur_off + b256 > im->seg_bytes) { cur_seg = cur_seg + 1; cur_off = 0; }   // on to the next segment
        if (cur_seg >= im->n_seg) break;                           // every segment is full
        if (max_slots > 0 && (int64_t) picks.size() >= max_slots) break;
        rs = (int32_t) picks.size();                                // the slot is the pick's index
        picks.push_back(Pick{pr.first, pr.second, cur_off, b256, cur_seg});
        cur_off += b256;
    }
    if (picks.empty()) { err = "peer: no room or no expert left for the peer"; return false; }

    // A blob never straddles a segment, so there is nothing left to fit: the packing above already stopped at
    // the first segment with no room.  Record where each slot landed and how full the segments got.
    im->slot_off.resize(picks.size());
    im->slot_seg.resize(picks.size());
    uint64_t used = 0;
    for (size_t i = 0; i < picks.size(); ++i) {
        im->slot_off[i] = picks[i].off;
        im->slot_seg[i] = picks[i].seg;
        used = std::max(used, picks[i].off + picks[i].slot);
    }
    im->arena_used = used;

    // ---- fill it, then read the first slot back: the OpenCL upload path is worth one check
    if (blob == nullptr) { err = "peer: no blob reader"; return false; }
    for (const Pick& pk : picks) {
        const uint8_t* b = blob(ctx, pk.layer, pk.expert);
        if (b == nullptr) {
            err = "peer: the blob of (layer " + std::to_string(pk.layer) + ", expert " +
                  std::to_string(pk.expert) + ") is not there";
            return false;
        }
        const cl_int wr = clEnqueueWriteBuffer(im->q, im->arenas[pk.seg], CL_FALSE, pk.off,
                                              (size_t) im->blob_bytes[(size_t) pk.layer], b, 0, nullptr, nullptr);
        if (wr != CL_SUCCESS) {
            err = "peer: filling slot (layer " + std::to_string(pk.layer) + ", expert " +
                  std::to_string(pk.expert) + "): " + cl_err(wr);
            return false;
        }
    }
    if (clFinish(im->q) != CL_SUCCESS) { err = "peer: waiting for the initial fills"; return false; }
    {
        const Pick& pk = picks.front();
        const uint8_t* b = blob(ctx, pk.layer, pk.expert);
        const size_t nb = (size_t) im->blob_bytes[(size_t) pk.layer];
        std::vector<uint8_t> back(nb, 0);
        const cl_int rd =
            clEnqueueReadBuffer(im->q, im->arenas[pk.seg], CL_TRUE, pk.off, nb, back.data(), 0, nullptr, nullptr);
        if (rd != CL_SUCCESS) { err = "peer: reading slot 0 back: " + cl_err(rd); return false; }
        if (b == nullptr || std::memcmp(back.data(), b, nb) != 0) {
            err = "peer: slot 0 does not read back what was written to it";
            return false;
        }
    }

    // ---- the work buffers.  Sized for the widest layer and the largest window the engine can hand over.
    struct Buf { cl_mem* h; uint64_t bytes; const char* what; };
    const uint64_t cE = (uint64_t) kCapEntries, cT = (uint64_t) kCapTokens;
    const Buf bufs[] = {
        {&im->xqbuf,   cT * (uint64_t) (im->max_embd / 32) * 36,                "the quantised activations"},
        {&im->gubuf,   cE * 2u * (uint64_t) im->max_ff * 4,                     "the gate/up results"},
        {&im->hqbuf,   cE * (uint64_t) (im->max_ff / 32) * 36,                  "the SwiGLU activations"},
        {&im->dobuf,   cE * (uint64_t) im->max_embd * 4,                        "the down results"},
    };
    for (const Buf& b : bufs) {
        *b.h = clCreateBuffer(im->ctx, CL_MEM_READ_WRITE, (size_t) b.bytes, nullptr, &rc);
        if (rc != CL_SUCCESS) {
            err = "peer: allocating " + std::string(b.what) + " (" + std::to_string(b.bytes >> 10) + " KiB): " +
                  cl_err(rc);
            return false;
        }
    }
    // One packed input upload per layer.  On this driver every clEnqueue* costs the host ~60 us and a
    // launch used to pay for three separate writes of the same small class (the activations, the entry
    // offsets, the entry activation indices).  Packed into one buffer they leave in one write; the
    // kernels are bound to sub-buffer views, so the kernel sources and their argument lists do not move.
    // Layout: [ent_off (cE*8) | pad | ent_act (cE*4) | pad | x (cT*n_embd*4)], every region start on the
    // device's sub-buffer alignment.
    {
        cl_uint align_bits = 0;
        clGetDeviceInfo(im->dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits, nullptr);
        const cl_uint al = align_bits == 0 || align_bits / 8 < 256u ? 256u : align_bits / 8;
        const auto up = [al](uint64_t v) { return (v + al - 1u) / al * al; };
        im->off_cap = (cl_uint) up(cE * 8);
        im->act_cap = (cl_uint) up(cE * 4);
        im->x_base = im->off_cap + im->act_cap;
        im->x_cap = (cl_uint) (cT * (uint64_t) im->max_embd * 4);
        const uint64_t in_bytes = (uint64_t) im->x_base + im->x_cap;
        im->inbuf = clCreateBuffer(im->ctx, CL_MEM_READ_WRITE, (size_t) in_bytes, nullptr, &rc);
        if (rc != CL_SUCCESS) { err = "peer: allocating the packed input buffer: " + cl_err(rc); return false; }
        const cl_buffer_region r_off{0, (size_t) im->off_cap},
                              r_act{im->off_cap, (size_t) im->act_cap},
                              r_x{im->x_base, (size_t) im->x_cap};
        im->ent_off = clCreateSubBuffer(im->inbuf, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r_off, &rc);
        if (rc != CL_SUCCESS) { err = "peer: viewing the entry offsets: " + cl_err(rc); return false; }
        im->ent_act = clCreateSubBuffer(im->inbuf, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r_act, &rc);
        if (rc != CL_SUCCESS) { err = "peer: viewing the entry activations: " + cl_err(rc); return false; }
        im->xsub = clCreateSubBuffer(im->inbuf, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r_x, &rc);
        if (rc != CL_SUCCESS) { err = "peer: viewing the activations: " + cl_err(rc); return false; }
        im->in_host = (unsigned char*) std::malloc((size_t) in_bytes);
        if (im->in_host == nullptr) { err = "peer: allocating the packed input staging"; return false; }
    }
    // The read-back target: page-locked, mapped once, so the D2H is one DMA into memory the driver does not
    // have to stage through.  `finish` reads into `out_host` and then copies the rows where they belong.
    im->outstg = clCreateBuffer(im->ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               (size_t) (cE * (uint64_t) im->max_embd * 4), nullptr, &rc);
    if (rc != CL_SUCCESS) { err = "peer: allocating the read-back staging: " + cl_err(rc); return false; }
    im->out_host = (float*) clEnqueueMapBuffer(im->q, im->outstg, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                                              (size_t) (cE * (uint64_t) im->max_embd * 4), 0, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS || im->out_host == nullptr) {
        err = "peer: mapping the read-back staging: " + cl_err(rc);
        return false;
    }

    // ---- the four codebooks the decodes index.  From ggml-common.h, not a copy: the tables are the same
    // ones the CUDA kernels read, so there is nothing to keep in step.
    {
        // IQ3_XXS (type 18) reads iq3xxs_grid, NOT iq2xxs_grid: the two 256-entry grids are different
        // tables and the names differ by one character.  `iq2xxs_grid` is a uint64[256] whose .x half the
        // CUDA reads for IQ2_XXS (type 16), a format this pack does not use.
        std::vector<uint32_t> t(iq3xxs_grid, iq3xxs_grid + 256);
        im->t_iq3xxs = clCreateBuffer(im->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, t.size() * 4, t.data(), &rc);
        if (rc != CL_SUCCESS) { err = "peer: the IQ3_XXS codebook: " + cl_err(rc); return false; }
    }
    {
        std::vector<uint32_t> t((size_t) 1024 * 2);
        std::memcpy(t.data(), iq2s_grid, t.size() * 4);                          // uint64 entries, read as int2
        im->t_iq2s = clCreateBuffer(im->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, t.size() * 4, t.data(), &rc);
        if (rc != CL_SUCCESS) { err = "peer: the IQ2_S codebook: " + cl_err(rc); return false; }
    }
    {
        std::vector<uint32_t> t(iq3s_grid, iq3s_grid + 512);
        im->t_iq3s = clCreateBuffer(im->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, t.size() * 4, t.data(), &rc);
        if (rc != CL_SUCCESS) { err = "peer: the IQ3_S codebook: " + cl_err(rc); return false; }
    }
    {
        uint8_t kv[16];
        for (int i = 0; i < 16; ++i) kv[i] = (uint8_t) (int8_t) kvalues_iq4nl[i];
        im->t_kv = clCreateBuffer(im->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16, kv, &rc);
        if (rc != CL_SUCCESS) { err = "peer: the IQ4_NL codebook: " + cl_err(rc); return false; }
    }

    if (!im->build(im->quant, quant_source(), "quant_rows", "quant_rows", err)) return false;
    if (!im->build(im->swiglu, swiglu_source(), "swiglu_q8_1", "swiglu_q8_1", err)) return false;
    // Every layer's two kernels here rather than on the first token: each build is a few hundred ms on this
    // driver, and the first token is exactly where that would be noticed.
    for (int64_t l = 0; l < im->n_layers; ++l) {
        if (im->row_prog(layers[(size_t) l].gu_type, (int) im->gu_chunk[(size_t) l], grp_rows(layers[(size_t) l].gu_type), false, false, err) == nullptr) return false;
        // both down variants: the fold is a runtime switch (STRATA_OCL_FOLD), and a program built at the
        // first launch costs ~1.6 s on this driver - paid inside round one, where it looks like the whole
        // pipeline stalled
        if (im->row_prog(layers[(size_t) l].d_type, (int) im->d_chunk[(size_t) l], grp_rows(layers[(size_t) l].d_type), true, false, err) == nullptr) return false;
        if (im->row_prog(layers[(size_t) l].d_type, (int) im->d_chunk[(size_t) l], grp_rows(layers[(size_t) l].d_type), true, true, err) == nullptr) return false;
    }

    im->h_off.reserve((size_t) kCapEntries);
    im->h_act.reserve((size_t) kCapEntries);
    im->h_tok.reserve((size_t) kCapEntries);
    im->h_seen.reserve((size_t) kCapEntries);

    resident_ = (int64_t) picks.size();
    entries_ = experts_ = swaps_ = 0;
    gib_ = (double) im->arena_bytes / 1073741824.0;
    p_ = std::move(im);
    open_ = true;
    return true;
}

// ---------------------------------------------------------------- the per-layer run

bool OclPeer::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                     const int32_t* kind, std::string& err, float* out) {
    (void) out;      // no CUDA-visible address exists for an OpenCL buffer, so the rows come back in finish()
    if (!open_ || !p_) { err = "peer: not open"; return false; }
    Impl& im = *p_;
    struct Enq {                       // STRATA_OCL_STATS: the host side of one launch
        std::chrono::steady_clock::time_point t0;
        double* acc;
        explicit Enq(double* a) : t0(std::chrono::steady_clock::now()), acc(a) {}
        ~Enq() { *acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } _enq(&im.ms_enq);
    im.launched_rows = 0;
    if (im.read_ev != nullptr) { clReleaseEvent(im.read_ev); im.read_ev = nullptr; }   // this layer's read is next
    if (layer < 0 || layer >= im.n_layers) {
        err = "peer: layer " + std::to_string(layer) + " out of range";
        return false;
    }
    const int64_t n = n_tok * k;
    if (n_tok <= 0 || n_tok > kCapTokens || n > kCapEntries || k <= 0) {
        err = "peer: a window of " + std::to_string(n_tok) + " x " + std::to_string(k) + " entries is too large";
        return false;
    }
    const OclLayerInfo& L = im.layers[(size_t) layer];
    const int64_t neb = L.n_embd, nff = L.n_ff;
    const int64_t act_gu = neb / 32, act_d = nff / 32;

    // the entries the planner gave the peer, in routing order
    uint32_t m = 0;
    im.h_off.clear();
    im.h_act.clear();
    im.h_tok.clear();
    im.h_seen.clear();
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        const int32_t e = ids[i];
        if (e < 0 || e >= im.n_expert) {
            err = "peer: an entry was planned for the peer with an expert id out of range";
            return false;
        }
        const int32_t sl = im.res[(size_t) (layer * im.n_expert + e)];
        if (sl < 0) { err = "peer: an entry was planned for the peer but is not resident"; return false; }
        // segment in the top byte, offset below: the row kernel reads this one value and picks its base
        im.h_off.push_back((((uint64_t) im.slot_seg[(size_t) sl]) << 56) | im.slot_off[(size_t) sl]);
        im.h_act.push_back((uint32_t) (i / k));          // the token whose q8_1 the gate/up rows read
        im.h_tok.push_back((int32_t) i);                 // the row of `out` this entry belongs to
        if (std::find(im.h_seen.begin(), im.h_seen.end(), e) == im.h_seen.end()) {
            im.h_seen.push_back(e);
            ++experts_;
        }
        ++m;
    }
    if (m == 0) return true;
    entries_ += (int64_t) m;
    im.stat_entries += (int64_t) m;
    im.h_ident.clear();

    const size_t mb8 = (size_t) m * 8, mb4 = (size_t) m * 4;
    // ONE packed upload instead of three enqueues (see the buffer's comment at open): the entry tables
    // and the activations leave the host in a single write, the hole between them rides along for free.
    std::memcpy(im.in_host, im.h_off.data(), mb8);
    std::memcpy(im.in_host + im.off_cap, im.h_act.data(), mb4);
    std::memcpy(im.in_host + im.x_base, x, (size_t) (n_tok * neb) * sizeof(float));
    cl_int rc = clEnqueueWriteBuffer(im.q, im.inbuf, CL_FALSE, 0,
                                     (size_t) im.x_base + (size_t) (n_tok * neb) * sizeof(float), im.in_host,
                                     0, nullptr, nullptr);
    if (rc != CL_SUCCESS) { err = "peer: uploading the packed input: " + cl_err(rc); return false; }

    // the activation q8_1: one 32-lane group per block of 32 values, over every token
    {
        if (!im.bind_quant(err)) return false;
        const size_t gsz = (size_t) (n_tok * act_gu) * 32, lsz = 32;
        rc = clEnqueueNDRangeKernel(im.q, im.quant.k, 1, nullptr, &gsz, &lsz, 0, nullptr, &im.ev_quant);
        if (rc != CL_SUCCESS) { err = "peer: quantising the activations: " + cl_err(rc); return false; }
    }

    // gate/up: 2*n_ff rows per entry, every one of them dotted with its token's activation
    {
        Impl::RowJob J;
        J.type = L.gu_type;
        J.nchunk = (int) im.gu_chunk[(size_t) layer];
        J.xq = im.xqbuf;
        J.outv = im.gubuf;
        J.w_extra = 0;
        J.n_row_e = (uint32_t) (2 * nff);
        J.wrow_bytes = (uint32_t) im.gu_row[(size_t) layer];
        J.act_blk_row = (uint32_t) act_gu;
        J.out_stride = (uint32_t) (2 * nff);
        J.n_entries = m;
        J.ev = &im.ev_gu;
        if (!im.run_row(J, err)) return false;
    }

    // The standalone SwiGLU, quantise included.  Folding it into the down pass was tried and REVERTED:
    // the in-lane re-quantisation (a serial 32-value exp/quant chain per call, five calls per row) made
    // the down kernel ~6x slower - 0.09 -> 0.56 ms/layer - by starving the weight-streaming loop that
    // used to hide everything behind memory.  STRATA_OCL_FOLD=1 puts it back for a driver or a shape
    // where an enqueue costs more than an exp chain.
    static const bool fold_dn = std::getenv("STRATA_OCL_FOLD") != nullptr;
    // STRATA_OCL_MAPOUT=1 puts the zero-copy form back (see the down pass below for why it is off).
    static const bool map_out = std::getenv("STRATA_OCL_MAPOUT") != nullptr;
    if (!fold_dn) {
        if (!im.bind_swiglu((cl_uint) nff, err)) return false;
        const size_t gsz = (size_t) (m * act_d) * 32, lsz = 32;
        const cl_int r2 =
            clEnqueueNDRangeKernel(im.q, im.swiglu.k, 1, nullptr, &gsz, &lsz, 0, nullptr, &im.ev_swi);
        if (r2 != CL_SUCCESS) { err = "peer: launching the SwiGLU kernel: " + cl_err(r2); return false; }
    }

    // down: n_embd rows per entry, dotted with the entry's own SwiGLU activation.
    {
        Impl::RowJob J;
        J.act_identity = true;
        J.fold_swiglu = fold_dn;
        J.gu = im.gubuf;
        J.n_ff = (uint32_t) nff;
        J.type = L.d_type;
        J.nchunk = (int) im.d_chunk[(size_t) layer];
        J.xq = im.hqbuf;
        // Writing the page-locked staging from the kernel (no read-back command) was tried and REVERTED:
        // enqueueing a kernel that targets mapped host memory costs the HOST ~0.76 ms/layer on this
        // driver - thirty times the whole rest of the launch.  The DMA read-back stays.
        J.outv = map_out ? im.outstg : im.dobuf;
        J.w_extra = im.down_off[(size_t) layer];
        J.n_row_e = (uint32_t) neb;
        J.wrow_bytes = (uint32_t) im.d_row[(size_t) layer];
        J.act_blk_row = (uint32_t) act_d;
        J.out_stride = (uint32_t) neb;
        J.n_entries = m;
        J.ev = &im.ev_down;
        if (!im.run_row(J, err)) return false;
    }

    im.launched_rows = (int64_t) m;
    im.last_layer = layer;
    im.last_act_d = act_d;
    // The rows are on their way back while the engine computes its own for this layer.  Unless the zero-
    // copy form is on, the D2H read-back follows the kernels IN-QUEUE, into the page-locked staging, so
    // `finish` only waits.  In-queue, deliberately.  A second queue ordered behind ev_down was tried and
    // REVERTED: the cross-queue event signal costs more than the buffer drain it replaces (gap
    // 0.147 -> 0.189 ms/layer).  (Measured on this card; revisit only with a different driver.)
    if (!map_out) {
        // STRATA_OCL_SYNCREAD=1 moves the D2H copy out of the device's timeline: the profiling shows the
        // in-queue read-back starting ~0.15 ms after the down kernel's END on this driver - the compute-
        // to-copy-engine signal, on a queue that has nothing else to do while it waits.  In the sync form
        // `finish` waits for the down kernel and THEN issues a blocking copy, paying one command
        // submission (~0.05 ms, hidden in the host's own wait) instead of the dead gap.
        static const bool sync_read = std::getenv("STRATA_OCL_SYNCREAD") != nullptr;
        if (!sync_read) {
            if (im.read_ev != nullptr) { clReleaseEvent(im.read_ev); im.read_ev = nullptr; }
            const cl_int rr = clEnqueueReadBuffer(im.q, im.dobuf, CL_FALSE, 0,
                                                  (size_t) (m * neb) * sizeof(float), im.out_host,
                                                  0, nullptr, &im.read_ev);
            if (rr != CL_SUCCESS) {
                err = "peer: starting the result read-back: " + cl_err(rr);
                im.launched_rows = 0;
                return false;
            }
        }
    }
    // AND SUBMIT IT.  AMD's Windows runtime buffers commands and hands them to the GPU when the buffer
    // fills or something blocking happens, so a CL_FALSE copy added here would sit in the queue until
    // `finish` calls clWaitForEvents - which is the very wait the early issue exists to remove.
    // Measured with the copy issued at the end of launch: 0.456 ms/layer in `finish` without this
    // flush, against 0.036 ms of host-side enqueue for the whole launch.  clFlush is the submit.
    clFlush(im.q);
    im.t_enq = std::chrono::steady_clock::now();
    im.have_enq = true;
    return true;
}

bool OclPeer::finish(float* out, std::string& err) {
    if (!open_ || !p_) { err = "peer: not open"; return false; }
    Impl& im = *p_;
    if (im.launched_rows == 0) return true;
    if (out == nullptr) { err = "peer: no result block to write into"; return false; }
    const std::chrono::steady_clock::time_point t_fin = std::chrono::steady_clock::now();
    if (im.have_enq) {
        im.ms_host_gap +=
            std::chrono::duration<double, std::milli>(t_fin - im.t_enq).count();
        im.have_enq = false;
    }
    const int64_t m = im.launched_rows;
    const int64_t neb = im.layers[(size_t) im.last_layer].n_embd;
    // Asked once, not once per layer.  `std::getenv` in the UCRT takes the environment lock and walks the
    // block, and on this machine - 21 expert-pool threads and a 20 KB environment - two calls per layer
    // measured at ~0.4 ms, which is twice what the layer's kernels cost and was several sessions' worth of
    // "the read-back is slow".  Nothing here reads the environment again.
    static const bool dump_w = std::getenv("STRATA_OCL_DUMP_W") != nullptr;
    static const bool dump_hq = std::getenv("STRATA_OCL_DUMP_HQ") != nullptr;
    static const bool stats = std::getenv("STRATA_OCL_STATS") != nullptr;
    // Per-segment wall trace of finish for a handful of layers.  The stats accumulators below sum to a
    // fraction of ms_read on this driver, so one of the segments the averages hide is the real cost - the
    // trace prints the raw deltas until it is found.
    static const bool trace = std::getenv("STRATA_OCL_TRACE") != nullptr;
    // What the kernel actually saw: the expert's bytes as they sit in the arena, and the geometry the host
    // derived for them.  A row that does not match the reference can then be re-run offline against the
    // real bytes instead of an assumed layout.
    if (dump_w && m == 1 && !im.h_off.empty()) {
        const int64_t L = im.last_layer;
        const size_t bb = (size_t) im.blob_bytes[(size_t) L];
        std::vector<uint8_t> w(bb, 0);
        if (clEnqueueReadBuffer(im.q, im.arenas[im.slot_seg[0]], CL_TRUE, (size_t) im.slot_off[0], bb, w.data(), 0,
                                nullptr, nullptr) == CL_SUCCESS) {
            const std::string fn = "peer_blob_l" + std::to_string(L) + ".bin";
            if (FILE* fh = std::fopen(fn.c_str(), "wb")) {
                const uint64_t hdr[8] = {im.slot_off[0] | ((uint64_t) im.slot_seg[0] << 56), im.down_off[(size_t) L], im.d_row[(size_t) L],
                                         im.gu_row[(size_t) L], (uint64_t) im.d_chunk[(size_t) L],
                                         (uint64_t) im.gu_chunk[(size_t) L], bb,
                                         (uint64_t) im.layers[(size_t) L].gu_type * 256 +
                                             (uint64_t) im.layers[(size_t) L].d_type};
                std::fwrite(hdr, 8, 8, fh);
                std::fwrite(w.data(), 1, bb, fh);
                std::fclose(fh);
            }
            std::printf("[DUMP_W] l%lld ent_off %llu down_off %llu d_row %llu gu_row %llu d_chunk %lld "
                        "gu_chunk %lld bytes %zu\n",
                        (long long) L,
                        (unsigned long long) (im.slot_off[0] | ((uint64_t) im.slot_seg[0] << 56)),
                        (unsigned long long) im.down_off[(size_t) L], (unsigned long long) im.d_row[(size_t) L],
                        (unsigned long long) im.gu_row[(size_t) L], (long long) im.d_chunk[(size_t) L],
                        (long long) im.gu_chunk[(size_t) L], bb);
        }
    }
    if (dump_hq && m == 1) {   // the q8_1 the down pass read (one entry)
        const size_t nb = (size_t) (m * im.last_act_d) * 36u;
        std::vector<uint8_t> hq(nb, 0);
        if (clEnqueueReadBuffer(im.q, im.hqbuf, CL_TRUE, 0, nb, hq.data(), 0, nullptr, nullptr) == CL_SUCCESS) {
            const std::string fn = "peer_hq_l" + std::to_string(im.last_layer) + ".bin";
            if (FILE* fh = std::fopen(fn.c_str(), "wb")) {
                std::fwrite(hq.data(), 1, nb, fh);
                std::fclose(fh);
            }
        }
    }
    // Which event `finish` waits on depends on the read-back form: the copy's event when the DMA read-
    // back is on (the queue is in order, so it covers the whole chain), otherwise the down kernel's own
    // completion event, which covers the kernels behind it and makes the mapped results host-visible.
    cl_event* const ev = im.read_ev != nullptr ? &im.read_ev
                        : (im.ev_down != nullptr ? &im.ev_down : nullptr);
    const bool wait_on_event = ev != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    im.ms_seg1 += std::chrono::duration<double, std::milli>(t0 - t_fin).count();
    const cl_int rc = ev != nullptr ? clWaitForEvents(1, ev) : clFinish(im.q);
    const auto t_w1 = std::chrono::steady_clock::now();
    const double waited = std::chrono::duration<double, std::milli>(t_w1 - t0).count();
    // im., NOT bare: there is a same-named member on OclPeer itself (hpp), and the bare spelling sent the
    // per-layer blocked time into a counter the stats never print - "blocked 0.000" while the trace showed
    // 0.63 ms of true blocking per layer.  That lie cost an hour of looking for time that was never missing.
    im.ms_wait += waited;
    if (rc != CL_SUCCESS) { err = "peer: waiting for the rows: " + cl_err(rc); return false; }
    // The sync read-back form: the kernels are done, NOW copy - one blocking command, no in-queue slot.
    static const bool sync_read = std::getenv("STRATA_OCL_SYNCREAD") != nullptr;
    if (sync_read && im.read_ev == nullptr) {
        const cl_int rr = clEnqueueReadBuffer(im.q, im.dobuf, CL_TRUE, 0,
                                              (size_t) (m * neb) * sizeof(float), im.out_host,
                                              0, nullptr, nullptr);
        if (rr != CL_SUCCESS) { err = "peer: the result read-back: " + cl_err(rr); return false; }
    }
    // STRATA_OCL_STATS, part two.  The queue has drained, so the four kernels' timestamps are final.  They
    // are serialised in one in-order queue, so their durations account for the device's whole time on the
    // layer and the split says which of the three passes to attack - the row kernel wants a wider grid, the
    // quantiser is a rounding error, and the SwiGLU is the one that could be folded into the row kernel.
    //
    // The read event is profiled too, and separately.  `waited` above is the host's blocked time, but that
    // lumps together two unrelated things: the queue finishing the kernels, and the device->host copy that
    // follows them.  Its START is when the DMA began (so START minus the down kernel's END is the queue's
    // own gap) and END minus START is the transfer itself.  On this driver the transfer is not the small
    // part, and knowing which of the two the host is actually waiting on decides the next change.
    double copy_ms = 0, gap_ms = 0;
    const auto t_prof = std::chrono::steady_clock::now();
    {
        auto rt = [](cl_event e, cl_ulong& a, cl_ulong& b) {
            a = b = 0;
            if (e == nullptr) return false;
            return clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(a), &a, nullptr) == CL_SUCCESS &&
                   clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr) == CL_SUCCESS;
        };
        auto take = [&](cl_event& e, double& acc, cl_ulong* end_out) {
            if (e == nullptr) return;
            cl_ulong a = 0, b = 0;
            if (rt(e, a, b)) {
                acc += (double) (b - a) * 1e-6;
                if (end_out != nullptr) *end_out = b;
            }
            clReleaseEvent(e);
            e = nullptr;
        };
        cl_ulong down_end = 0;
        take(im.ev_quant, im.ms_quant, nullptr);
        take(im.ev_gu, im.ms_gu, nullptr);
        take(im.ev_swi, im.ms_swi, nullptr);
        take(im.ev_down, im.ms_down, &down_end);
        cl_ulong rs = 0, re = 0;
        if (rt(im.read_ev, rs, re)) {
            copy_ms = (double) (re - rs) * 1e-6;
            if (down_end != 0 && rs > down_end) gap_ms = (double) (rs - down_end) * 1e-6;
        }
        ++im.stat_kern;
    }
    im.ms_copy_wait += copy_ms;
    im.ms_gap += gap_ms;
    const auto t_p1 = std::chrono::steady_clock::now();
    im.ms_prof += std::chrono::duration<double, std::milli>(t_p1 - t_prof).count();
    const auto t_cp = std::chrono::steady_clock::now();
    for (int64_t r = 0; r < m; ++r)
        std::memcpy(out + (size_t) im.h_tok[(size_t) r] * (size_t) neb,
                    im.out_host + (size_t) r * (size_t) neb, (size_t) neb * sizeof(float));
    const auto t_c1 = std::chrono::steady_clock::now();
    im.ms_copy += std::chrono::duration<double, std::milli>(t_c1 - t_cp).count();
    im.launched_rows = 0;
    im.ms_read += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fin).count();
    if (trace && (im.stat_layers < 4 || im.stat_layers % 512 == 0)) {
        const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fin).count();
        std::fprintf(stderr,
                     "[OCLT] layer %lld: m %lld ev %d | seg1 %.3f  wait %.3f  prof %.3f  copy %.3f  tail %.3f"
                     "  = %.3f ms\n",
                     (long long) im.last_layer, (long long) m, wait_on_event ? 1 : 0,
                     std::chrono::duration<double, std::milli>(t0 - t_fin).count(), waited,
                     std::chrono::duration<double, std::milli>(t_p1 - t_prof).count(),
                     std::chrono::duration<double, std::milli>(t_c1 - t_cp).count(),
                     total - std::chrono::duration<double, std::milli>(t_c1 - t_fin).count(), total);
    }
    if (stats && ++im.stat_layers % 48 == 0) {
        std::fprintf(stderr, "[OCL] %lld layers: launch %.1f ms (%.3f/layer), read+copy %.1f ms (%.3f/layer), "
                             "%.1f entries/layer, rows/group %d, iq4nl work-group %d\n",
                     (long long) im.stat_layers, im.ms_enq, im.ms_enq / (double) im.stat_layers, im.ms_read,
                     im.ms_read / (double) im.stat_layers, (double) im.stat_entries / (double) im.stat_layers,
                     g_rows, g_iq4nl_wg);
        const double nl = (double) std::max<int64_t>(1, im.stat_kern);
        const double sum = im.ms_quant + im.ms_gu + im.ms_swi + im.ms_down;
        std::fprintf(stderr, "[OCL]   the device's own time over %lld layers: quant %.2f  gate/up %.2f  "
                             "swiglu %.2f  down %.2f  = %.2f ms/layer (%.1f%% of it in the two row passes)\n",
                     (long long) im.stat_kern, im.ms_quant / nl, im.ms_gu / nl, im.ms_swi / nl,
                     im.ms_down / nl, sum / nl,
                     sum > 0 ? 100.0 * (im.ms_gu + im.ms_down) / sum : 0.0);
        std::fprintf(stderr, "[OCL]   finish %.3f ms/layer = blocked %.3f + profiling %.3f + copy-out %.3f; the "
                             "host had %.3f ms of its own work between submitting and finishing, the "
                             "device->host copy was %.3f in flight, and the queue's gap before it %.3f\n",
                     im.ms_read / (double) im.stat_layers, im.ms_wait / (double) im.stat_layers,
                     im.ms_prof / (double) im.stat_layers, im.ms_copy / (double) im.stat_layers,
                     im.ms_host_gap / (double) im.stat_layers, im.ms_copy_wait / nl, im.ms_gap / nl);
        // The raw totals, so the arithmetic can be checked without trusting the printed averages.  `seg1`
        // is the stretch between entering `finish` and reaching the wait, which nothing else accounts for.
        std::fprintf(stderr, "[OCL]   totals over %lld layers (ms): enq %.1f  finish %.1f  seg1 %.1f  wait %.1f  "
                             "prof %.1f  copy %.1f  |  host gap %.1f\n",
                     (long long) im.stat_layers, im.ms_enq, im.ms_read, im.ms_seg1, im.ms_wait, im.ms_prof,
                     im.ms_copy, im.ms_host_gap);
    }
    return true;
}

// ---------------------------------------------------------------- the adaptive tier

bool OclPeer::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!open_ || !p_) return true;
    Impl& im = *p_;
    if (!im.pending.empty() || max_swaps <= 0) return true;      // one swap set in flight at a time
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < im.n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * im.n_expert;
        const int32_t* r0 = res0 + l * im.n_expert;
        const int32_t* r1 = im.res.data() + l * im.n_expert;
        for (int32_t e = 0; e < (int32_t) im.n_expert; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;     // the same margin PeerExperts uses
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * im.n_expert + s.in);
        const size_t out = (size_t) (s.layer * im.n_expert + s.out);
        const int32_t slot = im.res[out];
        const uint8_t* b = im.blob != nullptr ? im.blob(im.blob_ctx, s.layer, s.in) : nullptr;
        if (slot < 0 || b == nullptr) continue;
        const cl_int rc = clEnqueueWriteBuffer(im.refill, im.arenas[im.slot_seg[(size_t) slot]], CL_FALSE,
                                              im.slot_off[(size_t) slot],
                                              (size_t) im.blob_bytes[(size_t) s.layer], b, 0, nullptr, nullptr);
        if (rc != CL_SUCCESS) { err = "peer: a refill copy: " + cl_err(rc); return false; }
        im.res[out] = -1;              // the incoming expert becomes resident at apply_pending; the slot's
                                       // old owner is out right now, so the CPU computes it meanwhile
        im.pending.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    if (im.refill_ev != nullptr) { clReleaseEvent(im.refill_ev); im.refill_ev = nullptr; }
    const cl_int rc = clEnqueueMarkerWithWaitList(im.refill, 0, nullptr, &im.refill_ev);
    if (rc != CL_SUCCESS) { err = "peer: marking the refill queue: " + cl_err(rc); return false; }
    return true;
}

void OclPeer::apply_pending(bool wait) {
    if (!open_ || !p_) return;
    Impl& im = *p_;
    if (im.pending.empty()) return;
    if (wait) {
        if (im.refill_ev != nullptr) clWaitForEvents(1, &im.refill_ev);
        else clFinish(im.refill);
    } else if (im.refill_ev != nullptr) {
        cl_int st = CL_COMPLETE;
        if (clGetEventInfo(im.refill_ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(st), &st, nullptr) != CL_SUCCESS ||
            st != CL_COMPLETE)
            return;
    }
    for (const auto& pr : im.pending) im.res[(size_t) pr.first] = pr.second;
    im.pending.clear();
}

void OclPeer::close() {
    if (p_) p_->release();
    p_.reset();
    open_ = false;
    index_ = -1;
    name_.clear();
    resident_ = 0;
    entries_ = experts_ = swaps_ = 0;
    gib_ = 0;
}

}  // namespace strata::core
