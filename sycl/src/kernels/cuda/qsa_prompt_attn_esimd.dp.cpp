// SYCL port (ARC-W): the QSA prompt attention on the int8 DPAS (ESIMD, Battlemage).
//
// The FP32 prompt kernel (qsa_decode_attn_batch, 64-cell chunks + a merge kernel) is compute bound at ~60% of the
// card's FP32 peak and takes 22-24% of a prompt.  This kernel runs both products on the s8 8x16x32 DPAS.
//
//  * One work-group of NTH = 4 hardware threads per (query, KV head).  Thread t owns the output dims [64 t, 64 t + 64),
//    which is also V's scale group t, and computes Q.K for 32 of every 128 selected cells (a "super-step").
//  * Q.K: A = 8 cells x 32 dims of K's int8 codes (a row of K is contiguous, so A is eight 32-byte slices), B = the
//    32 dims x 16 heads of Q (12 real heads, 4 zero lanes), D = 8 cells x 16 heads.  Q is quantized per head to
//    hi + lo int8 (x ~ sq (hi + lo / 254), error ~ 1.5e-5 of the head's largest value) so the product keeps ~15 bits;
//    K's per-64-dim scale is applied per cell row to each 64-dim group's int32 sum.  The scores of the thread's 32
//    cells go to local memory; one barrier later every thread has all 128.
//  * Softmax: every thread reads the 128 score rows (lane = head), takes the running max, rescales its output
//    accumulators and forms p = exp2(s - m).
//  * P.V: A = P (8 heads x 32 cells, int8 hi + lo of p * v_scale[cell][group t], scaled to 127 by the super-step's
//    largest v scale), B = the thread's 64 dims of 32 V rows packed four cells to a dword (a 4x4 byte transpose in
//    registers), D = 8 heads x 16 dims into float accumulators held in registers (12 heads in two 8-row tiles x 4
//    dim tiles).
//  * Not bitwise the FP32 kernel: Q and P are 16-bit (hi + lo int8) numbers, the sums are exact int32, the scales and
//    the softmax are FP32.  qsa_prompt_attn_parity bounds the difference against FP64.
// int8 KV only (the default); anything else keeps the caller's fallback.
#include "strata/kernels/qsa_prompt_attn_esimd.hpp"
#include "strata/kprof.hpp"
#include "strata/core/geometry.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#ifndef STRATA_I8_NO_ESIMD
namespace strata::kernels {
namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int HD = 256, G = 12, NTH = 4, SS = 128;
constexpr uint32_t P_STRIDE = SS + 4;
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_esimd is compiled for core/geometry.hpp");
constexpr uint32_t QT_OFF = 0;            // Q^T B tiles: [8 K steps][LV levels][128 dwords]

SYCL_ESIMD_FUNCTION inline simd<int, 128> dpas8(simd<int, 128> acc, simd<int, 128> b, simd<int, 64> a) {
    return xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(acc, b, a);
}

// row index of 32 selected cells (selection positions c_first ...), as the page table maps them; positions past n take `fallback`
SYCL_ESIMD_FUNCTION inline simd<int, 32> cell_rows(const int32_t* idsb, const int32_t* pt, int c_first, int n, int kvh, int nkv,
                                                   int pshift, int pmask, int fallback) {
    simd<uint32_t, 32> lane(0, 1);
    simd<int, 32> pos = simd<int, 32>(c_first) + simd<int, 32>(lane);
    simd_mask<32> ok = pos < n;
    simd<int, 32> cell = gather<int, 32>(idsb, lane * 4u, ok, simd<int, 32>(0));
    simd<uint32_t, 32> poff = simd<uint32_t, 32>(cell >> pshift) * 4u;
    simd<int, 32> page = gather<int, 32>(pt, poff, ok, simd<int, 32>(0));
    simd<int, 32> rows = ((page * nkv + kvh) << pshift) + (cell & pmask);
    rows.merge(simd<int, 32>(fallback), !ok);
    return rows;
}

template <int LV>
SYCL_ESIMD_FUNCTION constexpr float level_coef(int l) { return LV == 2 ? (l == 0 ? 254.0f : 1.0f) : (l == 0 ? 64516.0f : (l == 1 ? 254.0f : 1.0f)); }

// x in [-127, 127] as LV int8 levels: x ~ o[0] + o[1] / 254 + o[2] / 254^2
template <int N, int LV>
SYCL_ESIMD_FUNCTION inline void split_lv(simd<float, N> x, simd<int8_t, N>* o) {
    simd<float, N> r = x;
#pragma unroll
    for (int l = 0; l < LV; ++l) {
        simd<float, N> f = rnde<float>(r);   // |r| <= 127 at every level, so the rounded value fits an int8
        o[l] = f;
        r = (r - f) * 254.0f;
    }
}

constexpr unsigned SURF_H = (1u << 24) - 1u;   // 2D block loads: the surface height - 1 (no clamping)

// four consecutive pool rows starting at element i of a row-index vector
template <int N>
SYCL_ESIMD_FUNCTION inline bool contig4(const simd<int, N>& r, int i) {
    const int r0 = r[i];
    return r[i + 1] == r0 + 1 && r[i + 2] == r0 + 2 && r[i + 3] == r0 + 3;
}

struct Args {
    const float* q;
    const int8_t* kq;
    const uint16_t* ks;
    const int8_t* vq;
    const uint16_t* vs;
    const int32_t* pt;
    const int32_t* ids;
    const int32_t* steps;
    float* attn;
    int nkv, cap, pshift, pmask;
    float scale_log2;
    int dbg;
    QsaSplitAttnLayout split{};
};

// ESIMD's constant /96 still lowers to the general software integer divider.
SYCL_ESIMD_FUNCTION inline uint32_t div96(uint32_t value) {
    return (uint64_t(value) * 0xaaaaaaabull) >> 38;
}

template <int N>
SYCL_ESIMD_FUNCTION inline simd<uint32_t, N> div96(simd<uint32_t, N> value) {
    simd<uint64_t, N> product = value;
    product *= 0xaaaaaaabu;
    return simd<uint32_t, N>(product >> 38);
}

template <bool FIXED>
SYCL_ESIMD_FUNCTION inline int page_cell(const Args& a, int row) {
    if constexpr (FIXED) return row - 96 * div96(uint32_t(row));
    else return row % a.split.page_rows;
}

template <bool SPLIT, bool FIXED = false>
SYCL_ESIMD_FUNCTION inline size_t code_offset(const Args& a, int row) {
    if constexpr (!SPLIT) return (size_t) row * HD;
    if constexpr (FIXED) {
        const uint32_t group = div96(uint32_t(row)), cell = row - 96 * group;
        return (size_t)(group >> 1) * a.split.page_halves * 2 +
               (size_t)cell * 2 * HD + (group & 1u) * HD;
    }
    const int nr = a.split.page_rows;
    return (size_t) (row / (nr * a.nkv)) * a.split.page_halves * 2 +
           (size_t) (row % nr) * a.nkv * HD + (size_t) ((row / nr) % a.nkv) * HD;
}

template <bool FIXED = false>
SYCL_ESIMD_FUNCTION inline size_t split_scale_offset(const Args& a, int row) {
    if constexpr (FIXED) {
        const uint32_t group = div96(uint32_t(row)), cell = row - 96 * group;
        return (size_t)(group >> 1) * a.split.page_halves * 2 + 96 * 2 * HD +
               (cell * 2 + (group & 1u)) * 16;
    }
    const int nr = a.split.page_rows;
    return (size_t) (row / (nr * a.nkv)) * a.split.page_halves * 2 +
           (size_t) nr * a.nkv * HD +
           ((size_t) (row % nr) * a.nkv + (row / nr) % a.nkv) * 16;
}

template <bool SPLIT, bool FIXED = false>
SYCL_ESIMD_FUNCTION inline bool contiguous(const Args& a, const simd<int, 32>& rows, int i) {
    const bool consecutive = contig4(rows, i);
    if constexpr (!SPLIT) return consecutive;
    if constexpr (FIXED) return consecutive && page_cell<true>(a, rows[i]) <= 92;
    return consecutive && rows[i] / a.split.page_rows == rows[i + 3] / a.split.page_rows;
}

template <bool SPLIT, bool FIXED = false>
SYCL_ESIMD_FUNCTION inline simd<int, 32> selected_rows(
    const Args& a, const int32_t* ids, int first, int n, int kvh, int fallback, bool selected) {
    if constexpr (!SPLIT)
        return cell_rows(ids + first, a.pt, first, n, kvh, a.nkv, a.pshift, a.pmask, fallback);
    simd<int, 32> pos = simd<int, 32>(first) + simd<int, 32>(0, 1);
    simd_mask<32> ok = pos < n;
    simd<int, 32> cells = pos;
    if (selected)
        cells = gather<int, 32>(ids, simd<uint32_t, 32>(pos) * 4u, ok, simd<int, 32>(0));
    const int nr = a.split.page_rows;
    simd<uint32_t, 32> logical_pages;
    simd<int, 32> local_cells;
    if constexpr (FIXED) {
        logical_pages = div96(simd<uint32_t, 32>(cells));
        local_cells = cells - simd<int, 32>(logical_pages) * 96;
    } else {
        logical_pages = simd<uint32_t, 32>(cells / nr);
        local_cells = cells % nr;
    }
    simd<uint32_t, 32> offsets = logical_pages * 4u;
    simd<int, 32> pages = gather<int, 32>(a.pt, offsets, ok, simd<int, 32>(0));
    simd<int, 32> rows = (pages * a.nkv + kvh) * nr + local_cells;
    rows.merge(simd<int, 32>(fallback), !ok);
    return rows;
}

// ---- Q.K for one tile of 8 cells: S[cell][head] -> local memory (rows = cells, lanes = heads)
//      FAST: the 8 cells are two runs of four consecutive pool rows (the indexer selects blocks of four cells): 2D block loads
template <bool FAST, int QL, bool SPLIT, bool FIXED = false>
SYCL_ESIMD_FUNCTION inline void qk_tile(const Args& a, const simd<int, 8>& rows, int pos0, int n, const simd<float, 16>& cscale,
                                        uint32_t s_dst, simd<float, 16>* score_max) {
    simd<float, 128> sacc = 0.0f;
    simd<int, 512> kall;
    simd<uint8_t, 256> dA[4], dB[4];
    if constexpr (FAST) {
        const int rA = rows[0], rB = rows[4];
        const uint8_t* baseA = reinterpret_cast<const uint8_t*>(a.kq);
        const uint8_t* baseB = baseA;
        int yA = rA, yB = rB;
        uint32_t pitch = 255u, height = SURF_H;
        if constexpr (SPLIT) {
            yA = page_cell<FIXED>(a, rA);
            yB = page_cell<FIXED>(a, rB);
            baseA += code_offset<true, FIXED>(a, rA) - (size_t) yA * a.nkv * HD;
            baseB += code_offset<true, FIXED>(a, rB) - (size_t) yB * a.nkv * HD;
            pitch = a.nkv * HD - 1;
            height = a.split.page_rows - 1;
        }
#pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
            dA[pp] = sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 32, 4, 2, false, false>(
                baseA, 255u, height, pitch, 64 * pp, yA);
            dB[pp] = sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 32, 4, 2, false, false>(
                baseB, 255u, height, pitch, 64 * pp, yB);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const size_t off = code_offset<SPLIT, FIXED>(a, rows[j]);
            kall.template select<64, 1>(64 * j) = block_load<int, 64>(reinterpret_cast<const int*>(a.kq + off));
        }
    }
    // Independent KV32 scale groups for split; KV64 groups for PP.
    constexpr int NG = SPLIT ? 8 : 4;
    simd<float, 8 * NG> ksf;
    if constexpr (FIXED) {
        simd<uint64_t, 16> offsets;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint64_t offset = split_scale_offset<true>(a, rows[j]);
            offsets[2 * j] = offset;
            offsets[2 * j + 1] = offset + 8;
        }
        simd<uint64_t, 16> packed = gather<uint64_t, 16>(
            reinterpret_cast<const uint64_t*>(a.kq), offsets);
        ksf = simd<float, 64>(packed.template bit_cast_view<sycl::half>());
    } else if constexpr (SPLIT) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            simd<uint16_t, 8> scales = block_load<uint16_t, 8>(
                reinterpret_cast<const uint16_t*>(a.kq + split_scale_offset<FIXED>(a, rows[j])));
            simd<sycl::half, 8> halves = scales.template bit_cast_view<sycl::half>();
            ksf.template select<8, 1>(8 * j) = simd<float, 8>(halves);
        }
    } else {
        simd<uint32_t, 16> so;
#pragma unroll
        for (int j = 0; j < 8; ++j) { so[2 * j] = (uint32_t) rows[j] * 8u; so[2 * j + 1] = (uint32_t) rows[j] * 8u + 4u; }
        simd<uint32_t, 16> sd = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(a.ks), so);
        simd<sycl::half, 32> ksh = sd.template bit_cast_view<sycl::half>();
        ksf = ksh;
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        simd<int, 128> acc[QL];
#pragma unroll
        for (int l = 0; l < QL; ++l) acc[l] = 0;
#pragma unroll
        for (int ss = 0; ss < (SPLIT ? 1 : 2); ++ss) {
            const int s = SPLIT ? g : 2 * g + ss;
            simd<int, 64> A;
            if constexpr (FAST) {
                simd<int, 64> da = dA[s >> 1].template bit_cast_view<int>(), db = dB[s >> 1].template bit_cast_view<int>();
                A.template select<32, 1>(0) = da.template select<32, 1>((s & 1) * 32);
                A.template select<32, 1>(32) = db.template select<32, 1>((s & 1) * 32);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) A.template select<8, 1>(8 * j) = kall.template select<8, 1>(64 * j + 8 * s);
            }
#pragma unroll
            for (int l = 0; l < QL; ++l) {
                simd<int, 128> bq = slm_block_load<int, 128>(QT_OFF + (uint32_t) (s * QL + l) * 512u);
                if constexpr (FIXED && QL == 2) {
                    if (l == 0) acc[0] = dpas8(acc[0], bq, A);
                    else acc[1] = dpas8(acc[0] * 254, bq, A);
                } else acc[l] = dpas8(acc[l], bq, A);
            }
        }
        simd<float, 128> tf;
        if constexpr (QL == 2) {
            simd<int, 128> ti;
            if constexpr (FIXED) ti = acc[1];
            else ti = acc[0] * 254 + acc[1];
            tf = ti;
        } else {
            tf = 0.0f;
#pragma unroll
            for (int l = 0; l < QL; ++l) {
                simd<float, 128> fl = acc[l];
                tf += fl * level_coef<QL>(l);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float ksc = ksf[NG * j + g];
            sacc.template select<16, 1>(16 * j) += tf.template select<16, 1>(16 * j) * ksc;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        simd<float, 16> row = sacc.template select<16, 1>(16 * j) * cscale;
        const float madd = (pos0 + j >= n) ? -INFINITY : 0.0f;
        row += madd;
        if constexpr (FIXED) *score_max = max(*score_max, row);
        sacc.template select<16, 1>(16 * j) = row;
    }
    slm_block_store<float, 128>(s_dst, sacc);
}

SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> q_padding_offsets(int t) {
    const simd<uint32_t, 16> lane(0, 1);
    return simd<uint32_t, 16>(QT_OFF + (t >> 1) * 512u + (t & 1) * 256u + G * 4u) +
        (lane >> 2) * 64u + (lane & 3u) * 4u;
}

template <bool FIXED>
SYCL_ESIMD_FUNCTION constexpr int output_tile(int head, int nt) {
    return FIXED ? (head * 4 + nt) * 16 : ((head >> 3) * 4 + nt) * 128 + (head & 7) * 16;
}

template <bool FAST>
SYCL_ESIMD_FUNCTION inline simd<int, 512> split_v_tile(
    const Args& a, const simd<int, 32>& rows, int t) {
    simd<int, 512> values;
#pragma unroll
    for (int k4 = 0; k4 < 8; ++k4) {
        simd<int, 64> w;
        if constexpr (FAST) {
            const int row = rows[4 * k4], y = page_cell<true>(a, row);
            const auto* base = reinterpret_cast<const uint8_t*>(a.vq) +
                code_offset<true, true>(a, row) - (size_t)y * 2 * HD;
            simd<uint8_t, 256> packed =
                sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 16, 4, 4, false, true>(
                    base, 255u, 95u, 511u, 64 * t, y);
            w = packed.template bit_cast_view<int>();
        } else {
            auto bytes = w.template bit_cast_view<uint8_t>();
#pragma unroll
            for (int j = 0; j < 4; ++j)
                bytes.template select<64, 4>(j) = block_load<uint8_t, 64>(
                    reinterpret_cast<const uint8_t*>(a.vq) +
                    code_offset<true, true>(a, rows[4 * k4 + j]) + 64 * t);
        }
        values.template select<64, 1>(64 * k4) = w;
    }
    return values;
}

// ---- P.V for one step of 32 cells: O += P' V (this thread's 64 dims); lsum += sum p
template <bool FAST, int PL, bool SPLIT, int HALF = -1, bool FIXED = false>
SYCL_ESIMD_FUNCTION inline void pv_step(const Args& a, simd<float, FIXED ? G * 64 : 1024>& O, simd<float, 16>& lsum, const simd<float, 16>& m_new,
                                        uint32_t s_row0, const simd<float, 32>& vsf, const simd<int, 32>& rowsV, int t,
                                        const simd<float, G * 32>* shared_probabilities = nullptr,
                                        const simd<int, 512>* shared_values = nullptr,
                                        const simd<float, 32>* second_vsf = nullptr) {
    constexpr float COEF0P = PL == 2 ? 254.0f : 64516.0f;
    constexpr int PG = FIXED ? 4 : 2;
    simd<int8_t, 256> ap[PL][PG]; // [level][half * 2 + head tile] for the fused split path
    simd<float, 16> osv[FIXED ? 2 : 1];
    if constexpr (FIXED) {
        simd<float, G * 32> probabilities = *shared_probabilities;
        simd<float, G * 32> weighted[2];
        simd<float, 16> maxima[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) maxima[half] = 0.0f;
#pragma unroll
        for (int l = 0; l < PL; ++l)
#pragma unroll
            for (int group = 0; group < PG; ++group) ap[l][group] = 0;
        // Quantize contiguous cell vectors per head instead of scattering int8 lanes at stride32.
#pragma unroll
        for (int head = 0; head < G; ++head) {
            simd<float, 32> p = probabilities.template select<32, 1>(head * 32);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const simd<float, 32> w = p * (half == 0 ? vsf : *second_vsf);
                weighted[half].template select<32, 1>(head * 32) = w;
                maxima[half][head] = hmax<float>(w);
            }
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            simd<float, 16> pinv = 127.0f / maxima[half];
            pinv.merge(simd<float, 16>(0.0f), maxima[half] <= 0.0f);
            osv[half] = maxima[half] * (1.0f / (127.0f * COEF0P));
#pragma unroll
            for (int head = 0; head < G; ++head) {
                simd<float, 32> w = weighted[half].template select<32, 1>(head * 32);
                const float inv = pinv[head];
                simd<int8_t, 32> pb[PL];
                split_lv<32, PL>(w * inv, pb);
#pragma unroll
                for (int l = 0; l < PL; ++l)
                    ap[l][2 * half + head / 8].template select<32, 1>((head % 8) * 32) = pb[l];
            }
        }
    } else {
        simd<float, 16> prow[32];
        simd<float, 16> pm = 0.0f;
#pragma unroll
        for (int c = 0; c < 32; ++c) {
            simd<float, 16> srow = slm_block_load<float, 16>(s_row0 + (uint32_t) c * 64u);
            simd<float, 16> p = exp2(srow - m_new);
            if constexpr (HALF != 1) lsum += p;
            const float vsc = vsf[c];
            prow[c] = p * vsc;
            pm = max(pm, prow[c]);
        }
        simd<float, 16> pinv = 127.0f / pm;
        pinv.merge(simd<float, 16>(0.0f), pm <= 0.0f);
        osv[0] = pm * (1.0f / (127.0f * COEF0P));
#pragma unroll
        for (int c = 0; c < 32; ++c) {
            simd<float, 16> r = prow[c] * pinv;
            simd<int8_t, 16> pb[PL];
            split_lv<16, PL>(r, pb);
#pragma unroll
            for (int l = 0; l < PL; ++l) {
                ap[l][0].template select<8, 32>(c) = pb[l].template select<8, 1>(0);
                ap[l][1].template select<8, 32>(c) = pb[l].template select<8, 1>(8);
            }
        }
    }
    constexpr int VDIM = HALF < 0 ? 64 : 32;
    constexpr int VOFF = HALF < 0 ? 0 : 32 * HALF;
    simd<int, FIXED ? 512 : 8 * VDIM> vt;
    if constexpr (FIXED) vt = *shared_values;
    else {
#pragma unroll
        for (int k4 = 0; k4 < 8; ++k4) {
            simd<int, VDIM> w;
            if constexpr (FAST) {
                const uint8_t* base = reinterpret_cast<const uint8_t*>(a.vq);
                int y = rowsV[4 * k4];
                uint32_t pitch = 255u, height = SURF_H;
                if constexpr (SPLIT) {
                    const int row = y;
                    y = page_cell<FIXED>(a, row);
                    base += code_offset<true, FIXED>(a, row) - (size_t) y * a.nkv * HD;
                    pitch = a.nkv * HD - 1;
                    height = a.split.page_rows - 1;
                }
                simd<uint8_t, 4 * VDIM> v2 = sycl::ext::intel::experimental::esimd::lsc_load_2d<uint8_t, 16, 4, VDIM / 16, false, true>(
                    base, 255u, height, pitch, 64 * t + VOFF, y);
                w = v2.template bit_cast_view<int>();
            } else {
                auto wb = w.template bit_cast_view<uint8_t>();
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const size_t off = code_offset<SPLIT, FIXED>(a, rowsV[4 * k4 + j]);
                    wb.template select<VDIM, 4>(j) = block_load<uint8_t, VDIM>(reinterpret_cast<const uint8_t*>(a.vq) + off + 64 * t + VOFF);
                }
            }
            vt.template select<VDIM, 1>(VDIM * k4) = w;
        }
    }
    simd<int, 64> aq[PL][PG];
#pragma unroll
    for (int l = 0; l < PL; ++l) {
#pragma unroll
        for (int group = 0; group < PG; ++group)
            aq[l][group] = ap[l][group].template bit_cast_view<int>();
    }
#pragma unroll
    for (int nt = (HALF < 0 ? 0 : 2 * HALF); nt < (HALF < 0 ? 4 : 2 * HALF + 2); ++nt) {
        const int half = FIXED ? nt / 2 : 0;
        simd<int, 128> B;
#pragma unroll
        for (int k4 = 0; k4 < 8; ++k4)
            B.template select<16, 1>(16 * k4) =
                vt.template select<16, 1>((FIXED ? 64 : VDIM) * k4 +
                    16 * (nt - (FIXED || HALF < 0 ? 0 : 2 * HALF)));
        simd<int, 128> z = 0;
        simd<float, 128> f0 = 0.0f;
        simd<float, FIXED ? 64 : 128> f1 = 0.0f;
#pragma unroll
        for (int l = 0; l < PL; ++l) {
            simd<int, 128> d0 = dpas8(z, B, aq[l][2 * half]);
            simd<int, FIXED ? 64 : 128> d1;
            if constexpr (FIXED) {
                simd<int, 64> zero = 0;
                simd<int, 32> heads = aq[l][2 * half + 1].template select<32, 1>(0);
                d1 = xmx::dpas<8, 4, int, int, int, int,
                    xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(zero, B, heads);
            } else d1 = dpas8(z, B, aq[l][1]);
            simd<float, 128> e0 = d0;
            simd<float, FIXED ? 64 : 128> e1 = d1;
            f0 += e0 * level_coef<PL>(l);
            f1 += e1 * level_coef<PL>(l);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float s0 = osv[half][i], s1 = osv[half][8 + i];
            O.template select<16, 1>(output_tile<FIXED>(i, nt)) += f0.template select<16, 1>(16 * i) * s0;
            if (!FIXED || i < G - 8)
                O.template select<16, 1>(output_tile<FIXED>(8 + i, nt)) += f1.template select<16, 1>(16 * i) * s1;
        }
    }
}

template <int QL, int PL, bool SPLIT, bool FIXED> struct TagEsimdAttn;

template <int QL, int PL, bool SPLIT = false, bool FIXED = false>
void launch(sycl::queue& q, const Args input, int64_t n_q) {
    constexpr uint32_t S_OFF = 8u * QL * 512u;        // the scores of one super-step: [128 cells][16 heads] floats
    constexpr uint32_t SLM_BYTES = S_OFF + 8192u;
    constexpr uint32_t P_SUM_OFF = S_OFF + G * P_STRIDE * 4u;
    static_assert(P_SUM_OFF + NTH * 64u <= SLM_BYTES);
    static_assert(!FIXED || QL >= 2);
    constexpr float COEF0Q = QL == 2 ? 254.0f : 64516.0f;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    strata::kprof::kp(&q, __FILE__, __func__, __LINE__)->submit([&](sycl::handler& h) {
        h.parallel_for<TagEsimdAttn<QL, PL, SPLIT, FIXED>>(sycl::nd_range<3>(sycl::range<3>((size_t) n_q, (size_t) input.nkv, (size_t) NTH), sycl::range<3>(1, 1, NTH)), props,
                                     [=](sycl::nd_item<3> ndi) SYCL_ESIMD_KERNEL {
            Args a = input;
            if constexpr (FIXED) {
                a.nkv = 2;
                a.split.page_rows = 96;
                a.split.cards = 2;
            }
            slm_init<SLM_BYTES>();
            const int qi = (int) ndi.get_group(0) + (SPLIT ? a.split.first_query : 0);
            const int kvh = (int) ndi.get_group(1), t = (int) ndi.get_local_id(2);
            const int n_head = a.nkv * G;
            int n;
            bool selected = true;
            if constexpr (SPLIT) {
                selected = a.ids && a.split.counts && a.split.counts[qi] >= 0;
                const int p = *a.split.position + qi, block = p / 4;
                n = selected ? a.split.counts[qi] :
                    ((block + a.split.cards - 1 - a.split.card) / a.split.cards) * 4 +
                    (block % a.split.cards == a.split.card ? p % 4 + 1 : 0);
            } else n = a.steps[(size_t) qi * kStepCount + kStepWidth];
            const int32_t* idsb = a.ids ? a.ids + (size_t) qi * a.cap : nullptr;
            float* outp = a.attn + ((size_t) qi * n_head + (size_t) kvh * G) * HD + 64 * t;
            const float* qp = a.q + ((size_t) qi * n_head + (size_t) kvh * G) * HD * (SPLIT ? 2 : 1);

            // ---- Q: per-head int8 levels, the B tiles of this thread's two K steps (2t, 2t + 1)
            const simd<uint32_t, 16> lane16(0, 1);
            simd<float, 16> sqv = 0.0f;   // lane h: the head's scale (largest |q| / 127)
            simd<float, 16> query_maxima;
            if constexpr (FIXED) {
                simd<float, 4> owned = 0.0f;
#pragma unroll
                for (int i = 0; i < G / NTH; ++i) {
                    const float* row = qp + (size_t)(i * NTH + t) * 2 * HD;
                    simd<float, 64> r0 = block_load<float, 64>(row), r1 = block_load<float, 64>(row + 64);
                    simd<float, 64> r2 = block_load<float, 64>(row + 128), r3 = block_load<float, 64>(row + 192);
                    owned[i] = hmax<float>(max(max(abs(r0), abs(r1)), max(abs(r2), abs(r3))));
                }
                // Scores are not live yet; reuse their SLM for one maximum per real head.
                slm_scatter<float, 4>(simd<uint32_t, 4>(S_OFF + 4 * t, 4 * NTH), owned);
                barrier();
                query_maxima = slm_block_load<float, 16>(S_OFF);
                sqv = query_maxima * (1.0f / 127.0f);
            }
            {
                simd<int, 128> bq[QL][2];   // [level][step 2t, 2t + 1]
#pragma unroll
                for (int l = 0; l < QL; ++l) { bq[l][0] = 0; bq[l][1] = 0; }
#pragma unroll
                for (int hh = 0; hh < G; ++hh) {
                    const float* row = qp + (size_t) hh * HD * (SPLIT ? 2 : 1);
                    float amax;
                    if constexpr (FIXED) amax = query_maxima[hh];
                    else {
                        simd<float, 64> r0 = block_load<float, 64>(row), r1 = block_load<float, 64>(row + 64);
                        simd<float, 64> r2 = block_load<float, 64>(row + 128), r3 = block_load<float, 64>(row + 192);
                        simd<float, 64> mx = max(max(abs(r0), abs(r1)), max(abs(r2), abs(r3)));
                        amax = hmax<float>(mx);
                    }
                    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                    if constexpr (!FIXED)
                        sqv.merge(simd<float, 16>(amax * (1.0f / 127.0f)), lane16 == (uint32_t) hh);
                    simd<float, 64> mine = block_load<float, 64>(row + 64 * t);
                    simd<int8_t, 64> qb[QL];
                    split_lv<64, QL>(mine * inv, qb);
#pragma unroll
                    for (int l = 0; l < QL; ++l) {
                        simd<int, 16> qd = qb[l].template bit_cast_view<int>();
                        bq[l][0].template select<8, 16>(hh) = qd.template select<8, 1>(0);
                        bq[l][1].template select<8, 16>(hh) = qd.template select<8, 1>(8);
                    }
                }
                const uint32_t s0 = QT_OFF + (uint32_t) (2 * t) * (uint32_t) QL * 512u;
#pragma unroll
                for (int l = 0; l < QL; ++l) {
                    slm_block_store<int, 128>(s0 + (uint32_t) l * 512u, bq[l][0]);
                    slm_block_store<int, 128>(s0 + (uint32_t) (QL + l) * 512u, bq[l][1]);
                }
            }
            barrier();
            // S[cell][head] = sum_g ks[cell][g] * (sum of the levels) * sq[head] * scale_log2
            const simd<float, 16> cscale = sqv * (a.scale_log2 / COEF0Q);

            // the selection's first cell stands in for positions past n (finite scales, weight 0)
            int fallback = 0;
            if (n > 0) {
                const int c = selected ? idsb[0] : 0;
                if constexpr (FIXED) {
                    const int page = div96(uint32_t(c));
                    fallback = (a.pt[page] * 2 + kvh) * 96 + c - page * 96;
                } else if constexpr (SPLIT)
                    fallback = (a.pt[c / a.split.page_rows] * a.nkv + kvh) * a.split.page_rows + c % a.split.page_rows;
                else fallback = ((a.pt[c >> a.pshift] * a.nkv + kvh) << a.pshift) + (c & a.pmask);
            }

            simd<float, FIXED ? G * 64 : 1024> O = 0.0f;
            simd<float, 16> m = -INFINITY, l = 0.0f;

            for (int c0 = 0; c0 < n; c0 += SS) {
                // ---- Q.K for this thread's 32 cells
                simd<int, 32> rowsK = selected_rows<SPLIT, FIXED>(a, idsb, c0 + 32 * t, n, kvh, fallback, selected);
                simd<float, 16> local_max = -INFINITY;
#pragma unroll
                for (int tl = 0; tl < 4; ++tl) {
                    simd<int, 8> r8 = rowsK.template select<8, 1>(8 * tl);
                    const bool fast = !(a.dbg & 16) && contiguous<SPLIT, FIXED>(a, rowsK, 8 * tl) && contiguous<SPLIT, FIXED>(a, rowsK, 8 * tl + 4);
                    const uint32_t dst = S_OFF + (uint32_t) (32 * t + 8 * tl) * 64u;
                    if (a.dbg & 2) continue;
                    if (fast) qk_tile<true, QL, SPLIT, FIXED>(a, r8, c0 + 32 * t + 8 * tl, n, cscale, dst, &local_max);
                    else qk_tile<false, QL, SPLIT, FIXED>(a, r8, c0 + 32 * t + 8 * tl, n, cscale, dst, &local_max);
                }
                barrier();
                // ---- softmax over the 128 cells
                simd<float, 16> mx = -INFINITY;
                if constexpr (FIXED) {
                    // Q.K reads have finished. Only padded Q columns are reused; real heads stay intact.
                    slm_scatter<float, 16>(q_padding_offsets(t), local_max);
                    barrier();
#pragma unroll
                    for (int part = 0; part < NTH; ++part)
                        mx = max(mx, slm_gather<float, 16>(q_padding_offsets(part)));
                } else {
#pragma unroll 1
                    for (int b = 0; b < 16; ++b) {
                        simd<float, 128> blk = slm_block_load<float, 128>(S_OFF + (uint32_t) b * 512u);
#pragma unroll
                        for (int r = 0; r < 8; ++r) mx = max(mx, simd<float, 16>(blk.template select<16, 1>(16 * r)));
                    }
                }
                const simd<float, 16> m_new = max(m, mx);
                const simd<float, 16> alpha = exp2(m - m_new);
#pragma unroll
                for (int hh = 0; hh < (FIXED ? G : 16); ++hh) {
                    const float av = alpha[hh];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        auto tile = O.template select<16, 1>(output_tile<FIXED>(hh, nt));
                        tile *= av;
                    }
                }
                l *= alpha;
                simd<float, 16> lsum = 0.0f;
                if constexpr (FIXED) {
                    simd<float, 512> probabilities;
#pragma unroll
                    for (int b = 0; b < 4; ++b)
                        probabilities.template select<128, 1>(128 * b) =
                            slm_block_load<float, 128>(S_OFF + (uint32_t)(32 * t) * 64u + b * 512u);
                    simd<float, 16> local_sum = 0.0f;
#pragma unroll
                    for (int c = 0; c < 32; ++c) {
                        simd<float, 16> score = probabilities.template select<16, 1>(16 * c);
                        simd<float, 16> p = exp2(score - m_new);
                        probabilities.template select<16, 1>(16 * c) = p;
                        local_sum += p;
                    }
                    // Finish all score reads before replacing the table with aligned head-major P.
                    barrier();
#pragma unroll
                    for (int head = 0; head < G; ++head) {
                        simd<float, 32> p;
#pragma unroll
                        for (int b = 0; b < 4; ++b)
                            p.template select<8, 1>(8 * b) = probabilities.template select<8, 16>(128 * b + head);
                        slm_block_store<float, 32>(S_OFF + head * P_STRIDE * 4u + (uint32_t)t * 128u, p);
                    }
                    slm_block_store<float, 16>(P_SUM_OFF + (uint32_t)t * 64u, local_sum);
                    barrier();
                    // Only thread0 emits the denominator; combine four local FP32 sums once.
                    if (t == 0) {
#pragma unroll
                        for (int part = 0; part < NTH; ++part)
                            lsum += slm_block_load<float, 16>(P_SUM_OFF + (uint32_t)part * 64u);
                    }
                }
#pragma unroll (FIXED ? 1 : 4)
                for (int st = 0; st < 4; ++st) {
                    // V rows of the step's 32 cells, and their scales for this thread's group
                    const simd<int, 32> rowsV = selected_rows<SPLIT, FIXED>(a, idsb, c0 + 32 * st, n, kvh, fallback, selected);
                    simd<float, 32> vf, vf_second;
                    if constexpr (SPLIT) {
                        simd<uint64_t, 32> offsets;
#pragma unroll
                        for (int i = 0; i < 32; ++i)
                            offsets[i] = split_scale_offset<FIXED>(a, rowsV[i]) + 4 * t;
                        simd<uint32_t, 32> packed = gather<uint32_t, 32>(
                            reinterpret_cast<const uint32_t*>(a.vq), offsets);
                        simd<uint16_t, 32> low = packed & 0xffffu, high = packed >> 16;
                        vf = low.template bit_cast_view<sycl::half>();
                        vf_second = high.template bit_cast_view<sycl::half>();
                    } else {
                        simd<uint32_t, 32> vo = simd<uint32_t, 32>(rowsV) * 8u + (uint32_t) ((t >> 1) * 4);
                        simd<uint32_t, 32> vd = gather<uint32_t, 32>(reinterpret_cast<const uint32_t*>(a.vs), vo);
                        simd<uint32_t, 32> shv = (uint32_t) ((t & 1) * 16);
                        simd<uint16_t, 32> v16 = ((vd >> shv) & 0xffffu);
                        simd<sycl::half, 32> vh = v16.template bit_cast_view<sycl::half>();
                        vf = vh;
                    }
                    simd<uint32_t, 32> lane(0, 1);
                    simd_mask<32> ok = (simd<int, 32>(c0 + 32 * st) + simd<int, 32>(lane)) < n;
                    vf.merge(simd<float, 32>(0.0f), !ok);
                    if constexpr (SPLIT) vf_second.merge(simd<float, 32>(0.0f), !ok);
                    bool fastv = !(a.dbg & 16);
#pragma unroll
                    for (int g4 = 0; g4 < 8; ++g4) fastv = fastv && contiguous<SPLIT, FIXED>(a, rowsV, 4 * g4);
                    const uint32_t scores = S_OFF + (uint32_t) (32 * st) * 64u;
                    if constexpr (FIXED) {
                        simd<float, G * 32> probabilities;
#pragma unroll
                        for (int head = 0; head < G; ++head)
                            probabilities.template select<32, 1>(head * 32) =
                                slm_block_load<float, 32>(S_OFF + head * P_STRIDE * 4u + (uint32_t)st * 128u);
                        simd<int, 512> values;
                        if (fastv) values = split_v_tile<true>(a, rowsV, t);
                        else values = split_v_tile<false>(a, rowsV, t);
                        pv_step<true, PL, true, -1, true>(a, O, lsum, m_new, scores, vf, rowsV, t, &probabilities, &values, &vf_second);
                    } else if constexpr (SPLIT) {
                        if (fastv) {
                            pv_step<true, PL, true, 0, FIXED>(a, O, lsum, m_new, scores, vf, rowsV, t);
                            pv_step<true, PL, true, 1, FIXED>(a, O, lsum, m_new, scores, vf_second, rowsV, t);
                        } else {
                            pv_step<false, PL, true, 0, FIXED>(a, O, lsum, m_new, scores, vf, rowsV, t);
                            pv_step<false, PL, true, 1, FIXED>(a, O, lsum, m_new, scores, vf_second, rowsV, t);
                        }
                    } else {
                        if (fastv) pv_step<true, PL, false>(a, O, lsum, m_new, scores, vf, rowsV, t);
                        else pv_step<false, PL, false>(a, O, lsum, m_new, scores, vf, rowsV, t);
                    }
                }
                l += lsum;
                m = m_new;
                barrier();
            }
            // ---- out = O / l
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                const float lv = l[hh];
                const float il = lv > 0.0f ? 1.0f / lv : 0.0f;
                float* dst = outp + (size_t) hh * HD;
                if constexpr (SPLIT) {
                    const int head = kvh * G + hh, own = n_head / a.split.cards;
                    if constexpr (FIXED)
                        dst = a.attn + (((size_t)kvh * a.split.query_rows + qi) * G + hh) * (HD + 2) + 64 * t;
                    else
                        dst = a.attn + (((size_t) (head / own) * a.split.query_rows + qi) * own + head % own) * (HD + 2) + 64 * t;
                    if (t == 0) {
                        dst[HD] = m[hh] * 0.6931471805599453f;
                        dst[HD + 1] = lv;
                    }
                }
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    simd<float, 16> v = O.template select<16, 1>(output_tile<FIXED>(hh, nt)) * (SPLIT ? 1.0f : il);
                    if constexpr (SPLIT) {
                        // HD+2 partial rows are not uniformly 16-byte aligned.
                        scatter<float, 16>(dst + 16 * nt, simd<uint32_t, 16>(0, sizeof(float)), v);
                    } else block_store<float, 16>(dst + 16 * nt, v);
                }
            }
        });
    });
}
}  // namespace

bool qsa_prompt_attn_esimd(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    static const int mode = [] { const char* v = std::getenv("STRATA_PROMPT_ATTN_ESIMD"); return v ? std::atoi(v) : 1; }();   // default on (Battlemage); =0 off
    if (mode == 0 || n_q <= 0) return false;
    if (pools.k_q == nullptr || pools.v_q == nullptr || pools.k_scale == nullptr || pools.v_scale == nullptr || pools.k_q4 != nullptr ||
        pools.v_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !ids || !steps ||
        !pools.page_table)
        return false;
    const int64_t ps = s.page_size;
    if (ps <= 0 || (ps & (ps - 1)) != 0) return false;
    dpct::queue_ptr st = strata::q_of(stream);
    if (strata::intel_gpu_gen(st->get_device()) != strata::IntelGpuGen::Battlemage) return false;
    static bool told = false;
    if (!told) { told = true; std::fprintf(stderr, "strata: prompt attention on the int8 DPAS (ESIMD)\n"); }
    static const int lv = [] { const char* v = std::getenv("STRATA_ESIMD_LV"); return v ? std::atoi(v) : 2; }();
    int sh = 0;
    while ((1 << sh) < ps) ++sh;
    Args a{q, pools.k_q, pools.k_scale, pools.v_q, pools.v_scale, pools.page_table, ids, steps, attn,
           (int) s.n_head_kv, (int) cap, sh, (int) (ps - 1), 1.4426950408889634f / std::sqrt((float) HD),
           std::getenv("STRATA_ESIMD_DBG") != nullptr ? std::atoi(std::getenv("STRATA_ESIMD_DBG")) : 0};
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        Args b = a;
        b.q = q + q0 * s.n_head * HD;
        b.ids = ids + q0 * cap;
        b.steps = steps + q0 * kStepCount;
        b.attn = attn + q0 * s.n_head * HD;
        if (lv == 3) launch<3, 3>(*st, b, nb);
        else if (lv == 32) launch<3, 2>(*st, b, nb);
        else if (lv == 23) launch<2, 3>(*st, b, nb);
        else launch<2, 2>(*st, b, nb);
    }
    return true;
}

bool qsa_prompt_attn_esimd_split(const float* qg, const uint16_t* k, const uint16_t* v,
                                 const int32_t* cells, const QsaSplitAttnLayout& s,
                                 float* partial, void* stream) {
    if (!qg || !k || !v || !s.page_table || !s.position || !partial ||
        s.page_rows <= 0 || s.kv_heads <= 0 || s.cards <= 0 ||
        s.card < 0 || s.card >= s.cards || (s.kv_heads * G) % s.cards ||
        s.first_query < 0 || s.query_count <= 0 || s.first_query + s.query_count > s.query_rows ||
        s.selection_capacity <= 0 || s.page_halves * 2 < (uint64_t) s.page_rows * s.kv_heads * (HD + 16))
        return false;
    auto* st = strata::q_of(stream);
    if (strata::intel_gpu_gen(st->get_device()) != strata::IntelGpuGen::Battlemage) return false;
    Args a{};
    a.q = qg;
    a.kq = reinterpret_cast<const int8_t*>(k);
    a.vq = reinterpret_cast<const int8_t*>(v);
    a.pt = s.page_table;
    a.ids = cells;
    a.attn = partial;
    a.nkv = s.kv_heads;
    a.cap = s.selection_capacity;
    a.scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    a.split = s;
    const char* fixed = std::getenv("STRATA_SPLIT_ATTN_FIXED");
    if (s.page_rows == 96 && s.kv_heads == 2 && s.cards == 2 &&
        (!fixed || std::atoi(fixed) != 0))
        launch<2, 2, true, true>(*st, a, s.query_count);
    else
        launch<2, 2, true>(*st, a, s.query_count);
    return true;
}
}  // namespace strata::kernels
#else
namespace strata::kernels {
bool qsa_prompt_attn_esimd(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*, int64_t, void*) { return false; }
bool qsa_prompt_attn_esimd_split(const float*, const uint16_t*, const uint16_t*, const int32_t*,
                                 const QsaSplitAttnLayout&, float*, void*) { return false; }
}
#endif
