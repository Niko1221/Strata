// SYCL port: the prompt attention over 8-query union tiles on XMX (STRATA_PROMPT_ATTN_UNION=1) - see
// sycl/include/strata/kernels/qsa_prompt_attn_union.hpp.
//
// Pass 2's work-group is 12 sub-groups of 16 for the tile's 96 rows (row j * 12 + h = tile query j, head h). Sub-group
// s is (row group rg = s / 2 of 16 rows, half hf = s % 2). The scores are computed as S^T = K Q^T: the accumulator's
// columns are the tile's rows, so each lane holds its own row's cells and the online softmax needs no local copy of S
// and no shuffles. Each sub-group scores cells [32 hf, +32) of a 64-cell chunk for its 16 rows, the pair exchanges
// the row maxima and sums through local memory, and the P V product splits the head dimension: dims [128 hf, +128)
// of the same 16 rows (16 accumulator tiles in registers, 256 GRF). K (row-major) and V (packed in cell pairs) are
// staged as FP16 in a per-work-group global slot that stays in L1/L2, so every matrix tile is one 2D block load; Q^T
// is packed into the same slot once per work-group as the constant B operand; P goes through local memory once, as
// the A operand of P V.
#include "strata/kernels/qsa_prompt_attn_union.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::kernels {
class qsa_union_build;   // kernel names: namespace scope
class qsa_union_attn;
namespace {
namespace jm = sycl::ext::oneapi::experimental::matrix;
namespace jmi = sycl::ext::intel::experimental::matrix;
constexpr int HD = 256, G = 12, TQ = 8, ROWS = TQ * G;      // 96 rows: tile query j, head h -> row j * 12 + h
constexpr int SG = 16, NSG = ROWS / 8, THREADS = SG * NSG;  // 12 sub-groups
constexpr int CH = 64, RG = ROWS / 16, PS = CH + 8;          // union cells per chunk, row groups, P row stride
constexpr int RANGE = 16384;                                 // union build: cells per local bitmap pass
constexpr int BUILD_THREADS = 256;
constexpr int64_t TILES = 256;                               // tiles per (build, attention) launch pair
static_assert(HD == strata::core::kCompiledGeometry.head_dim &&
              G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv,
              "qsa_prompt_attn_union is compiled for core/geometry.hpp");

// a work-group's global staging slot, in halves: Q^T, then the chunk's K and V
constexpr size_t QT_H = (size_t) HD * ROWS, K_H = (size_t) CH * HD, V_H = (size_t) CH * HD;
constexpr size_t SLOT = QT_H + K_H + V_H;
// pass 2's local memory
struct Lay {
    static constexpr size_t p = (size_t) RG * 16 * PS * 2;   // P, FP16, rows x cells
    static constexpr size_t x = (size_t) RG * 2 * 16 * 4;    // the pair's row maxima / sums
    static constexpr size_t al = (size_t) ROWS * 4;          // per-row rescale, then the row sums
    static constexpr size_t rows = (size_t) 2 * CH * 8;      // [2][CH] KV rows of the chunk (double-buffered)
    static constexpr size_t mask = (size_t) 2 * CH * 4;      // [2][CH] which tile queries selected each cell
    static constexpr size_t bytes = p + 2 * x + 2 * al + rows + mask + 16 * 8;
};

template <typename T>
inline auto lptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}
template <typename T>
inline auto gptr(T* p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(p);
}

// pass 1: one work-group per tile; the union of the tile queries' selections, ascending: ulist = the cells, umask = bit
// j set when tile query j selected the cell
void union_build(const int32_t* __restrict__ ids, const int32_t* __restrict__ steps, int64_t cap, int64_t q0,
                 int64_t n_q, int32_t* __restrict__ ulist, uint8_t* __restrict__ umask, int32_t* __restrict__ ucount,
                 uint32_t* smask) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    auto grp = it.get_group();
    const int tile = (int) it.get_group(0), t = (int) it.get_local_id(0);
    const int64_t t0 = q0 + (int64_t) tile * TQ;
    const int nq = (int) (n_q - t0 < TQ ? n_q - t0 : TQ);
    int64_t nkv = 0;
    for (int j = 0; j < nq; ++j) { const int64_t v = steps[(t0 + j) * kStepCount + kStepNKv]; nkv = v > nkv ? v : nkv; }
    int32_t* out = ulist + (size_t) tile * TQ * cap;
    uint8_t* mout = umask + (size_t) tile * TQ * cap;
    int total = 0;
    for (int64_t r0 = 0; r0 < nkv; r0 += RANGE) {
        // 4 cells per word, one byte (the 8 queries' bits) per cell
        for (int w = t; w < RANGE / 4; w += BUILD_THREADS) smask[w] = 0u;
        it.barrier(sycl::access::fence_space::local_space);
        for (int64_t k = t; k < (int64_t) TQ * cap; k += BUILD_THREADS) {
            const int j = (int) (k / cap);
            const int64_t i = k - (int64_t) j * cap;
            if (j >= nq || i >= steps[(t0 + j) * kStepCount + kStepWidth]) continue;
            const int64_t id = ids[(t0 + j) * cap + i] - r0;
            if (id < 0 || id >= RANGE) continue;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                a(smask[id >> 2]);
            a.fetch_or((1u << j) << (8 * (id & 3)));
        }
        it.barrier(sycl::access::fence_space::local_space);
        constexpr int WPT = RANGE / 4 / BUILD_THREADS;   // words per thread (16 = 64 cells), contiguous
        int cnt = 0;
        for (int w = 0; w < WPT; ++w) {
            const uint32_t m = smask[t * WPT + w];
            cnt += (m & 0xFFu) != 0;
            cnt += (m & 0xFF00u) != 0;
            cnt += (m & 0xFF0000u) != 0;
            cnt += (m & 0xFF000000u) != 0;
        }
        const int base = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>());
        const int sum = sycl::reduce_over_group(grp, cnt, sycl::plus<int>());
        int o = total + base;
        for (int w = 0; w < WPT; ++w) {
            const uint32_t m = smask[t * WPT + w];
            if (m == 0u) continue;
            for (int b = 0; b < 4; ++b) {
                const uint32_t mb = (m >> (8 * b)) & 0xFFu;
                if (mb) { out[o] = (int32_t) (r0 + (t * WPT + w) * 4 + b); mout[o] = (uint8_t) mb; ++o; }
            }
        }
        total += sum;
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (t == 0) ucount[tile] = total;
}

using TA = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major>;
using TB = jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed>;
using TC = jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16>;

// pass 2: one work-group per (tile, KV head)
void union_attn(const float* __restrict__ q, QsaAttnPools p, const int32_t* __restrict__ ulist,
                const uint8_t* __restrict__ umask, const int32_t* __restrict__ ucount, int64_t cap, int64_t q0,
                int64_t n_q, int n_kv_heads, int page_size, float scale_log2, float* __restrict__ attn, uint8_t* smem,
                sycl::half* stage) {
    auto it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = it.get_sub_group();
    uint8_t* at = smem;
    auto take = [&](size_t b) { uint8_t* r = at; at += (b + 15) & ~(size_t) 15; return r; };
    sycl::half* sP = (sycl::half*) take(Lay::p);
    float* xmax = (float*) take(Lay::x);
    float* xsum = (float*) take(Lay::x);
    float* xal = (float*) take(Lay::al);
    float* xl = (float*) take(Lay::al);
    long long* srow2 = (long long*) take(Lay::rows);
    uint32_t* smk2 = (uint32_t*) take(Lay::mask);

    const int tile = (int) it.get_group(2), kvh = (int) it.get_group(1);
    const int t = (int) it.get_local_id(2), lane = (int) sg.get_local_id()[0], sgid = (int) sg.get_group_id()[0];
    const int rg = sgid >> 1, hf = sgid & 1;
    const int64_t t0 = q0 + (int64_t) tile * TQ;
    const int n_head = n_kv_heads * G;
    const int32_t* list = ulist + (size_t) tile * TQ * cap;
    const uint8_t* lmask = umask + (size_t) tile * TQ * cap;
    const int U = ucount[tile];
    sycl::half* qt = stage + ((size_t) tile * n_kv_heads + kvh) * SLOT;
    sycl::half* ks = qt + QT_H;
    sycl::half* vs = ks + K_H;
    // union entry c -> its KV row and query mask; a cell whose page is not resident (-1) is masked for every row, as
    // the FP32 kernel masks it
    auto entry = [&](int c, long long& r, uint32_t& mk) {
        r = -1; mk = 0u;
        if (c < U) {
            const int cell = list[c];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) { r = (page * n_kv_heads + kvh) * page_size + (cell % page_size); mk = lmask[c]; }
        }
    };
    // Q^T packed for B (dim pairs interleaved per row): word (k/2) * ROWS + n = q[n][k], q[n][k+1]
    for (int i = t; i < (HD / 2) * ROWS; i += THREADS) {
        const int n = i / (HD / 2), kp2 = i % (HD / 2);
        const int j = n / G, h = n % G;
        sycl::float2 v(0.0f);
        if (t0 + j < n_q) v = *reinterpret_cast<const sycl::float2*>(q + (size_t) (t0 + j) * n_head * HD + (size_t) (kvh * G + h) * HD + 2 * kp2);
        reinterpret_cast<uint32_t*>(qt)[(size_t) kp2 * ROWS + n] = sycl::bit_cast<uint32_t>(sycl::half2(v.x(), v.y()));
    }
    TC O[2][8];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int dd = 0; dd < 8; ++dd) jm::joint_matrix_fill(sg, O[rt][dd], 0.0f);
    float m_run = -INFINITY, l_run = 0.0f;   // this lane's row 16 rg + lane
    const int myrow = 16 * rg + lane;
    if (t < CH) { long long r; uint32_t mk; entry(t, r, mk); srow2[t] = r; smk2[t] = mk; }
    it.barrier(sycl::access::fence_space::global_and_local);

    for (int c0 = 0, buf = 0; c0 < U; c0 += CH, buf ^= 1) {
        const long long* srow = srow2 + buf * CH;
        const uint32_t* smk = smk2 + buf * CH;
        // staging: all of this thread's K and V loads issued first (raw 16-byte words + scales), then converted and
        // stored - one memory latency per chunk instead of one per item. K row-major (16 items per cell row), V packed.
        {
            constexpr int KI = (CH * (HD / 16) + THREADS - 1) / THREADS;          // 6
            constexpr int VI = ((CH / 2) * (HD / 16) + THREADS - 1) / THREADS;    // 3
            sycl::vec<uint32_t, 4> kr[KI];
            float ksc[KI];
            sycl::vec<uint32_t, 4> vr[VI][2];
            float vsc[VI][2];
#pragma unroll
            for (int a = 0; a < KI; ++a) {
                const int i = t + a * THREADS;
                ksc[a] = 0.0f;
                kr[a] = sycl::vec<uint32_t, 4>(0u);
                if (i < CH * (HD / 16)) {
                    const int n = i / (HD / 16), k0 = (i % (HD / 16)) * 16;
                    const long long r = srow[n];
                    if (r >= 0) {
                        kr[a] = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(p.k_q + r * HD + k0);
                        ksc[a] = (float) sycl::bit_cast<sycl::half>(p.k_scale[r * (HD / KV_Q8_GROUP) + k0 / KV_Q8_GROUP]);
                    }
                }
            }
#pragma unroll
            for (int a = 0; a < VI; ++a) {
                const int i = t + a * THREADS;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    vsc[a][e] = 0.0f;
                    vr[a][e] = sycl::vec<uint32_t, 4>(0u);
                    if (i < (CH / 2) * (HD / 16)) {
                        const int cp = i / (HD / 16), d0 = (i % (HD / 16)) * 16;
                        const long long r = srow[2 * cp + e];
                        if (r >= 0) {
                            vr[a][e] = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(p.v_q + r * HD + d0);
                            vsc[a][e] = (float) sycl::bit_cast<sycl::half>(p.v_scale[r * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP]);
                        }
                    }
                }
            }
            auto byte = [](const sycl::vec<uint32_t, 4>& w, int u) {
                return (float) (int8_t) ((w[u >> 2] >> (8 * (u & 3))) & 0xFFu);
            };
#pragma unroll
            for (int a = 0; a < KI; ++a) {
                const int i = t + a * THREADS;
                if (i >= CH * (HD / 16)) continue;
                const int n = i / (HD / 16), k0 = (i % (HD / 16)) * 16;
                sycl::vec<uint32_t, 8> w;
                const float sc = ksc[a];
#pragma unroll
                for (int u = 0; u < 8; ++u)
                    w[u] = sycl::bit_cast<uint32_t>(sycl::half2(byte(kr[a], 2 * u) * sc, byte(kr[a], 2 * u + 1) * sc));
                *reinterpret_cast<sycl::vec<uint32_t, 8>*>(ks + (size_t) n * HD + k0) = w;
            }
#pragma unroll
            for (int a = 0; a < VI; ++a) {
                const int i = t + a * THREADS;
                if (i >= (CH / 2) * (HD / 16)) continue;
                const int cp = i / (HD / 16), d0 = (i % (HD / 16)) * 16;
                sycl::vec<uint32_t, 8> o0, o1;
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    o0[u] = sycl::bit_cast<uint32_t>(sycl::half2(byte(vr[a][0], u) * vsc[a][0], byte(vr[a][1], u) * vsc[a][1]));
                    o1[u] = sycl::bit_cast<uint32_t>(sycl::half2(byte(vr[a][0], u + 8) * vsc[a][0], byte(vr[a][1], u + 8) * vsc[a][1]));
                }
                uint32_t* dst = reinterpret_cast<uint32_t*>(vs) + (size_t) cp * HD + d0;
                *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst) = o0;
                *reinterpret_cast<sycl::vec<uint32_t, 8>*>(dst + 8) = o1;
            }
        }
        it.barrier(sycl::access::fence_space::global_and_local);
        // the next chunk's rows: loads issued now, stored after this chunk's products (their latency overlaps them)
        long long nr = -1;
        uint32_t nmk = 0u;
        if (t < CH) entry(c0 + CH + t, nr, nmk);
        // S^T (32 cells of this half x the group's 16 rows) = K Q^T; 4 K tiles + 1 Q^T tile per k-step, double-buffered
        TC ST[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) jm::joint_matrix_fill(sg, ST[m], 0.0f);
        {
            const sycl::half* kb = ks + (size_t) (32 * hf) * HD;
            const sycl::half* qb = qt + (size_t) (16 * rg) * 2;
            TA Ac[4], An[4];
            TB Bc, Bn;
#pragma unroll
            for (int m = 0; m < 4; ++m) jm::joint_matrix_load(sg, Ac[m], gptr(kb + (size_t) (8 * m) * HD), (size_t) HD);
            jm::joint_matrix_load(sg, Bc, gptr(qb), (size_t) (ROWS * 2));
#pragma unroll
            for (int kk = 0; kk < HD / 16; kk += 2) {
#pragma unroll
                for (int m = 0; m < 4; ++m)
                    jm::joint_matrix_load(sg, An[m], gptr(kb + (size_t) (8 * m) * HD + (kk + 1) * 16), (size_t) HD);
                jm::joint_matrix_load(sg, Bn, gptr(qb + (size_t) ((kk + 1) * 8) * (ROWS * 2)), (size_t) (ROWS * 2));
#pragma unroll
                for (int m = 0; m < 4; ++m) jm::joint_matrix_mad(sg, ST[m], Ac[m], Bc, ST[m]);
                if (kk + 2 < HD / 16) {
#pragma unroll
                    for (int m = 0; m < 4; ++m)
                        jm::joint_matrix_load(sg, Ac[m], gptr(kb + (size_t) (8 * m) * HD + (kk + 2) * 16), (size_t) HD);
                    jm::joint_matrix_load(sg, Bc, gptr(qb + (size_t) ((kk + 2) * 8) * (ROWS * 2)), (size_t) (ROWS * 2));
                }
#pragma unroll
                for (int m = 0; m < 4; ++m) jm::joint_matrix_mad(sg, ST[m], An[m], Bn, ST[m]);
            }
        }
        // online softmax, the row in the lane: element (cell row, col) of ST[m] is cell 32 hf + 8 m + row of row col
        float mx = -INFINITY;
#pragma unroll
        for (int m = 0; m < 4; ++m)
            jmi::joint_matrix_apply(sg, ST[m], [&](float& v, size_t row, size_t col) {
                const int c = 32 * hf + 8 * m + (int) row, jq = (16 * rg + (int) col) / G;
                const bool ok = (smk[c] >> jq) & 1u;
                v = ok ? v * scale_log2 : -INFINITY;
                mx = sycl::fmax(mx, v);
            });
        xmax[(rg * 2 + hf) * 16 + lane] = mx;
        it.barrier(sycl::access::fence_space::local_space);
        const float m_new = sycl::fmax(m_run, sycl::fmax(mx, xmax[(rg * 2 + (hf ^ 1)) * 16 + lane]));
        const float alpha = m_new == -INFINITY ? 1.0f : (m_run == -INFINITY ? 0.0f : sycl::exp2(m_run - m_new));
        float sum = 0.0f;
#pragma unroll
        for (int m = 0; m < 4; ++m)
            jmi::joint_matrix_apply(sg, ST[m], [&](float& v, size_t row, size_t col) {
                const float e = (v == -INFINITY || m_new == -INFINITY) ? 0.0f : sycl::exp2(v - m_new);
                sum += e;
                sP[(size_t) (16 * rg + (int) col) * PS + 32 * hf + 8 * m + (int) row] = sycl::half(e);
            });
        xsum[(rg * 2 + hf) * 16 + lane] = sum;
        if (hf == 0) xal[myrow] = alpha;
        m_run = m_new;
        const bool rescale = sycl::any_of_group(sg, alpha != 1.0f);
        it.barrier(sycl::access::fence_space::local_space);
        l_run = l_run * alpha + sum + xsum[(rg * 2 + (hf ^ 1)) * 16 + lane];
        if (rescale) {
#pragma unroll
            for (int rt = 0; rt < 2; ++rt)
#pragma unroll
                for (int dd = 0; dd < 8; ++dd)
                    jmi::joint_matrix_apply(sg, O[rt][dd], [=](float& v, size_t row, size_t) { v *= xal[16 * rg + 8 * rt + row]; });
        }
        // O += P V for the group's 16 rows x this half's 128 dims
        {
            TA Pt[2][4];
#pragma unroll
            for (int rt = 0; rt < 2; ++rt)
#pragma unroll
                for (int kq = 0; kq < 4; ++kq)
                    jm::joint_matrix_load(sg, Pt[rt][kq], lptr(sP + (size_t) (16 * rg + 8 * rt) * PS + kq * 16), (size_t) PS);
            const sycl::half* vb = vs + (size_t) (128 * hf) * 2;
#pragma unroll
            for (int kq = 0; kq < 4; ++kq) {
                TB Vc, Vn;
                jm::joint_matrix_load(sg, Vc, gptr(vb + (size_t) (kq * 8) * (HD * 2)), (size_t) (HD * 2));
#pragma unroll
                for (int dd = 0; dd < 8; dd += 2) {
                    jm::joint_matrix_load(sg, Vn, gptr(vb + (size_t) (kq * 8) * (HD * 2) + (dd + 1) * 16 * 2), (size_t) (HD * 2));
                    jm::joint_matrix_mad(sg, O[0][dd], Pt[0][kq], Vc, O[0][dd]);
                    jm::joint_matrix_mad(sg, O[1][dd], Pt[1][kq], Vc, O[1][dd]);
                    if (dd + 2 < 8)
                        jm::joint_matrix_load(sg, Vc, gptr(vb + (size_t) (kq * 8) * (HD * 2) + (dd + 2) * 16 * 2), (size_t) (HD * 2));
                    jm::joint_matrix_mad(sg, O[0][dd + 1], Pt[0][kq], Vn, O[0][dd + 1]);
                    jm::joint_matrix_mad(sg, O[1][dd + 1], Pt[1][kq], Vn, O[1][dd + 1]);
                }
            }
        }
        if (t < CH) { srow2[(buf ^ 1) * CH + t] = nr; smk2[(buf ^ 1) * CH + t] = nmk; }
        it.barrier(sycl::access::fence_space::global_and_local);   // K, V of this chunk consumed; the next rows visible
    }
    if (hf == 0) xl[myrow] = l_run;
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int rt = 0; rt < 2; ++rt)
#pragma unroll
        for (int dd = 0; dd < 8; ++dd)
            jmi::joint_matrix_apply(sg, O[rt][dd], [=](float& v, size_t row, size_t col) {
                const int rr = 16 * rg + 8 * rt + (int) row, j = rr / G, h = rr % G;
                if (t0 + j < n_q) {
                    const float l = xl[rr];
                    attn[(size_t) (t0 + j) * n_head * HD + (size_t) (kvh * G + h) * HD + 128 * hf + 16 * dd + col] = l > 0.0f ? v / l : 0.0f;
                }
            });
}

// per queue (an engine stream): the union lists and the staging slots of TILES tiles, allocated on first use
struct Scratch {
    bool checked = false, ok = false;
    int32_t* ulist = nullptr;    // [TILES][TQ * cap]
    uint8_t* umask = nullptr;    // [TILES][TQ * cap]
    int32_t* ucount = nullptr;   // [TILES]
    sycl::half* stage = nullptr; // [TILES][n_kv_heads][SLOT]
    int64_t cap = 0, nkv = 0;
};
void release(Scratch& sc, sycl::queue& q) {
    if (sc.ulist) sycl::free(sc.ulist, q);
    if (sc.umask) sycl::free(sc.umask, q);
    if (sc.ucount) sycl::free(sc.ucount, q);
    if (sc.stage) sycl::free(sc.stage, q);
    sc.ulist = nullptr; sc.umask = nullptr; sc.ucount = nullptr; sc.stage = nullptr;
    sc.cap = 0; sc.nkv = 0;
}
// the queue's scratch for `cap` and `nkv`, or null (with one stderr line per queue) when the kernel does not run there
Scratch* scratch_for(sycl::queue* st, int64_t cap, int64_t nkv) {
    static std::mutex mu;
    static std::unordered_map<sycl::queue*, Scratch> all;
    std::lock_guard<std::mutex> lk(mu);
    Scratch& sc = all[st];
    if (!sc.checked) {
        sc.checked = true;
        const sycl::device dev = st->get_device();
        const char* why = nullptr;
        // the tiles are joint_matrix 8x16x16 on sub-group 16 and assume Xe2's accumulator layout (a lane holds a column)
        if (strata::intel_gpu_gen(dev) != strata::IntelGpuGen::Battlemage) why = "not a Battlemage card";
        else if (!st->is_in_order()) why = "the queue is not in order";
        else if (dev.get_info<sycl::info::device::local_mem_size>() < Lay::bytes) why = "too little local memory";
        sc.ok = why == nullptr;
        if (why)
            std::fprintf(stderr, "strata: STRATA_PROMPT_ATTN_UNION ignored on %s: %s\n",
                         dev.get_info<sycl::info::device::name>().c_str(), why);
        else
            std::fprintf(stderr, "strata: prompt attention on XMX, 8-query selection unions (STRATA_PROMPT_ATTN_UNION)\n");
    }
    if (!sc.ok) return nullptr;
    if (sc.cap < cap || sc.nkv != nkv) {
        st->wait();
        release(sc, *st);
        sc.ulist = sycl::malloc_device<int32_t>((size_t) TILES * TQ * cap, *st);
        sc.umask = sycl::malloc_device<uint8_t>((size_t) TILES * TQ * cap, *st);
        sc.ucount = sycl::malloc_device<int32_t>((size_t) TILES, *st);
        sc.stage = sycl::malloc_device<sycl::half>((size_t) TILES * nkv * SLOT, *st);
        if (!sc.ulist || !sc.umask || !sc.ucount || !sc.stage) {
            std::fprintf(stderr, "strata: STRATA_PROMPT_ATTN_UNION ignored: no device memory for its %.1f MiB of scratch\n",
                         (double) ((size_t) TILES * TQ * cap * 5 + (size_t) TILES * nkv * SLOT * 2) / 1048576.0);
            release(sc, *st);
            sc.ok = false;
            return nullptr;
        }
        sc.cap = cap; sc.nkv = nkv;
    }
    return &sc;
}
}  // namespace

bool qsa_prompt_attn_union_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PROMPT_ATTN_UNION"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

bool qsa_prompt_attn_union(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || s.n_head_kv <= 0 || s.page_size <= 0 || cap <= 0 ||
        cap > (1 << 20) || !ids || !steps || !pools.page_table || !pools.k_q || !pools.v_q || !pools.k_scale ||
        !pools.v_scale || pools.k_q4 || pools.v_q4)
        return false;
    sycl::queue* st = strata::q_of(stream);
    const int nkv = (int) s.n_head_kv, ps = (int) s.page_size;
    Scratch* sc = scratch_for(st, cap, nkv);
    if (sc == nullptr) return false;
    int32_t* ulist = sc->ulist;
    uint8_t* umask = sc->umask;
    int32_t* ucount = sc->ucount;
    sycl::half* stage = sc->stage;
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) HD);
    const int64_t n_tiles = (n_q + TQ - 1) / TQ;
    for (int64_t b0 = 0; b0 < n_tiles; b0 += TILES) {
        const int64_t nt = sycl::min<int64_t>(TILES, n_tiles - b0);
        const int64_t q0 = b0 * TQ;
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint32_t, 1> sm(sycl::range<1>(RANGE / 4), cgh);
            cgh.parallel_for<qsa_union_build>(
                sycl::nd_range<1>(sycl::range<1>((size_t) nt * BUILD_THREADS), sycl::range<1>(BUILD_THREADS)),
                [=](sycl::nd_item<1>) {
                    union_build(ids, steps, cap, q0, n_q, ulist, umask, ucount,
                                sm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
        st->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<uint8_t, 1> slm(sycl::range<1>(Lay::bytes), cgh);
            cgh.parallel_for<qsa_union_attn>(
                sycl::nd_range<3>(sycl::range<3>(1, (size_t) nkv, (size_t) nt * THREADS), sycl::range<3>(1, 1, THREADS)),
                sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(SG)]] {
                    union_attn(q, pools, ulist, umask, ucount, cap, q0, n_q, nkv, ps, scale_log2, attn,
                               slm.get_multi_ptr<sycl::access::decorated::no>().get(), stage);
                });
        });
    }
    return true;
}
}  // namespace strata::kernels
