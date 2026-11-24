#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// QSA: the attention layers' block selection (included by kernels.cu), as
// llama.cpp's qwen4exp graph builds it. Blocks of QSA_POOL positions in order;
// a block's key is the mean of its members' raw indexer keys, RMS-normed and
// rotated to its first member's position, kept in f16 (the cache type). A row
// at position p sees the complete blocks [0, (p + 1) / 4) and always attends
// its tail, the last (p + 1) % 4 positions; it scores the visible blocks with
// its indexer query (4 heads, each RMS-normed and rotated to p): sum over heads
// of relu(q_h . k) / sqrt(128), and keeps the QSA_TOP highest (the lower block
// on a tie; llama.cpp's CUDA top-k takes tied blocks in atomic order, so any
// choice among exact ties is one it can make). With QSA_TOP or fewer visible
// blocks every block is kept: the attention is dense. Every card computes the
// same selection from the same inputs.
namespace {

constexpr int QSA_POOL = 4, QSA_TOP = QSA_TOP_K / QSA_POOL, QSA_DIM = QSA_KEY, QSA_ROT = NROT;
static_assert(QSA_DIM % 16 == 0 && QSA_ROT <= QSA_DIM && QSA_TOP_K % QSA_POOL == 0,
              "the score tiles take QSA_DIM in sixteens, the rotary must fit a key, and top_k must divide into whole pools");
constexpr int QSA_ROWS = QSA_HEADS * QSA_DIM + QSA_DIM;   // the projection's rows: q (QSA_HEADS heads), then k

constexpr int QSA_MAXSEL = QSA_TOP * QSA_POOL + QSA_POOL - 1;

__dpct_inline__ float bf16f(uint16_t h) {
    return sycl::bit_cast<float>((unsigned)h << 16);
}

// y[t][r] = W_r . x[t] for the QSA_ROWS rows (r < QSA_HEADS * QSA_DIM: wq row r, else wk row r - that many), bf16 weights,
// f32 activations; a warp per row
template <int T>
void k_idx_proj(const uint16_t* __restrict__ wq, const uint16_t* __restrict__ wk,
                                                 const float* __restrict__ x, float* __restrict__ y) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31,
              r = item_ct1.get_group(2) * 8 + warp;
    if (r >= QSA_ROWS) return;
    const uint16_t* w = r < QSA_HEADS * QSA_DIM ? wq + (size_t) r * N : wk + (size_t) (r - QSA_HEADS * QSA_DIM) * N;
    float acc[T] = {};
#pragma unroll 4
    for (int c = lane * 2; c < N; c += 64) {
        const unsigned wv = *reinterpret_cast<const unsigned*>(w + c);
        const float w0 = bf16f((uint16_t) (wv & 0xffffu)), w1 = bf16f((uint16_t) (wv >> 16));
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const sycl::float2 xv =
                *reinterpret_cast<const sycl::float2 *>(x + (size_t)t * N + c);
            acc[t] = sycl::fma((float)w0, (float)(xv.x()),
                               sycl::fma((float)w1, (float)(xv.y()), acc[t]));
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * QSA_ROWS + r] = v;
    }
}

// x (QSA_DIM values, one per thread): RMS norm with w, then NEOX rotary on the first QSA_ROT dims at pos
__dpct_inline__ float idx_norm_rope(float v, const float *w, int pos, float eps,
                                    const Rope &base, float *sh, float *buf) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_local_id(2);
    const float ms = block_sum(v * v, sh) / (float) QSA_DIM;
    const float y = v * sycl::rsqrt(ms + eps) * w[i];
    buf[i] = y;
    
    item_ct1.barrier();
    float out = y;
    if (i < QSA_ROT) {
        const int k = i % (QSA_ROT / 2);
        const float theta = (float) rope_pos(base, pos, k) * rope_unit_freq(base, k, QSA_ROT);
        float sn, cs;
        sn = sycl::sincos(
            theta,
            sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::yes>(&cs));
        sn *= base.mscale; cs *= base.mscale;
        const float x0 = buf[k], x1 = buf[k + QSA_ROT / 2];
        out = i < QSA_ROT / 2 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    
    item_ct1.barrier();
    return out;
}

// block (row t, part): part < 4: indexer query head `part` normed and rotated into q; part 4: the raw key into the ring
void k_idx_prep(const float *__restrict__ y, const float *__restrict__ qn,
                sycl::half *ring, int ring_rows, float *__restrict__ q,
                const int *pos0p, float eps, Rope base, float *sh, float *buf) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2), part = item_ct1.get_group(1),
              i = item_ct1.get_local_id(2), p = *pos0p + t;

    const float* yt = y + (size_t) t * QSA_ROWS;
    if (part < QSA_HEADS) {
        q[(size_t) t * QSA_HEADS * QSA_DIM + part * QSA_DIM + i] =
            idx_norm_rope(yt[part * QSA_DIM + i], qn, p, eps, base, sh, buf);
    } else {
        ring[(size_t)(p % ring_rows) * QSA_DIM + i] =
            sycl::vec<float, 1>(yt[QSA_HEADS * QSA_DIM + i])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
}

// a row whose position completes a block pools it: the mean of the members' raw keys (f16), normed, rotated to the
// first member, stored f16
void k_idx_pool(const sycl::half *ring, int ring_rows,
                const float *__restrict__ kn, sycl::half *pooled, KvPages g,
                const int *pos0p, float eps, Rope base, float *sh, float *buf) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2), i = item_ct1.get_local_id(2),
              p = *pos0p + t;
    if ((p + 1) % QSA_POOL != 0) return;

    float v = 0.f;
#pragma unroll
    for (int m = 0; m < QSA_POOL; ++m) v +=
        sycl::vec<sycl::half, 1>(
            ring[(size_t)((p - (QSA_POOL - 1) + m) % ring_rows) * QSA_DIM + i])
            .convert<float, sycl::rounding_mode::automatic>()[0];
    v *= 1.0f / (float) QSA_POOL;
    const int j = (p + 1) / QSA_POOL - 1;
    pooled[kvp_block(g, j) + i] =
        sycl::vec<float, 1>(
            idx_norm_rope(v, kn, p - (QSA_POOL - 1), eps, base, sh, buf))
            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

// k_idx_prep and k_idx_pool in one launch: parts 0 .. QSA_HEADS as k_idx_prep's, part QSA_HEADS + 1 k_idx_pool's, which
// takes the window's own raw keys from y (rounded to f16 as the ring holds them) and the earlier ones from the ring
void k_idx_prep_pool(const float *__restrict__ y, const float *__restrict__ qn,
                     const float *__restrict__ kn, sycl::half *ring,
                     int ring_rows, float *__restrict__ q, sycl::half *pooled,
                     KvPages g, const int *pos0p, float eps, Rope base,
                     float *sh, float *buf) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2), part = item_ct1.get_group(1),
              i = item_ct1.get_local_id(2), pos0 = *pos0p, p = pos0 + t;

    const float* yt = y + (size_t) t * QSA_ROWS;
    if (part < QSA_HEADS) {
        q[(size_t) t * QSA_HEADS * QSA_DIM + part * QSA_DIM + i] = idx_norm_rope(yt[part * QSA_DIM + i], qn, p, eps, base, sh, buf);
    } else if (part == QSA_HEADS) {
        ring[(size_t)(p % ring_rows) * QSA_DIM + i] =
            sycl::vec<float, 1>(yt[QSA_HEADS * QSA_DIM + i])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    } else {
        if ((p + 1) % QSA_POOL != 0) return;
        float v = 0.f;
#pragma unroll
        for (int m = 0; m < QSA_POOL; ++m) {
            const int pm = p - (QSA_POOL - 1) + m;
            v +=
                sycl::vec<sycl::half, 1>(
                    pm >= pos0
                        ? sycl::vec<float, 1>(y[(size_t)(pm - pos0) * QSA_ROWS +
                                                QSA_HEADS * QSA_DIM + i])
                              .convert<sycl::half,
                                       sycl::rounding_mode::rte>()[0]
                        : ring[(size_t)(pm % ring_rows) * QSA_DIM + i])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        }
        v *= 1.0f / (float) QSA_POOL;
        const int j = (p + 1) / QSA_POOL - 1;
        pooled[kvp_block(g, j) + i] =
            sycl::vec<float, 1>(
                idx_norm_rope(v, kn, p - (QSA_POOL - 1), eps, base, sh, buf))
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
}

// scores[t][j] for the visible blocks j of rows with more than QSA_TOP of them: a thread per block
void k_idx_score(const float *__restrict__ q,
                 const sycl::half *__restrict__ pooled, KvPages g,
                 float *__restrict__ scores, int max_pools, const int *pos0p,
                 LaneRows lr, float *qs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1), p = row_pos(lr, pos0p, t),
              nv = (p + 1) / QSA_POOL;
    if (nv <= QSA_TOP) return;
    g = row_pages(g, lr, t);

    for (int k = item_ct1.get_local_id(2); k < QSA_HEADS * QSA_DIM;
         k += item_ct1.get_local_range(2))
        qs[k] = q[(size_t)t * QSA_HEADS * QSA_DIM + k];
    
    item_ct1.barrier();
    const int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (j >= nv) return;
    const sycl::uint4 *kr =
        reinterpret_cast<const sycl::uint4 *>(pooled + kvp_block(g, j));
    float d[QSA_HEADS] = {};
#pragma unroll 4
    for (int c = 0; c < QSA_DIM / 8; ++c) {
        const sycl::uint4 u = kr[c];
        const sycl::half2 *h2 = reinterpret_cast<const sycl::half2 *>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const sycl::float2 kv =
                h2[k].template convert<float, sycl::rounding_mode::automatic>();
            const int dim = c * 8 + 2 * k;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) d[h] = sycl::fma(
                qs[h * QSA_DIM + dim], (float)(kv.x()),
                sycl::fma(qs[h * QSA_DIM + dim + 1], (float)(kv.y()), d[h]));
        }
    }
    const float w = sycl::rsqrt((float)QSA_DIM);
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < QSA_HEADS; ++h) s += sycl::fmax(d[h], 0.f) * w;
    scores[(size_t) t * max_pools + j] = s;
}

// the same for a prompt's rows on the tensor cores: a block takes 16 rows (their 4 heads: 64 rows of q, in f16 high and
// low halves) against 64 blocks' keys (f16 already), f32 accumulation; each key is read once per 16 rows instead of once
// per row. Columns past a row's visible blocks get scores too (the select reads only the visible ones)
constexpr int QSC_ROWS = 16, QSC_KEYS = 64, QSC_LD = QSA_DIM + 8, QSC_SLD = 16 + 4;

void k_idx_score_tc(const float *__restrict__ q,
                    const sycl::half *__restrict__ pooled, KvPages g,
                    float *__restrict__ scores, int max_pools, const int *pos0p,
                    int T,
                    sycl::half qh[64 /*QSC_ROWS * QSA_HEADS*/][136 /*QSC_LD*/],
                    sycl::half ql[64 /*QSC_ROWS * QSA_HEADS*/][136 /*QSC_LD*/],
                    float st[8][16][20 /*QSC_SLD*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int t0 = item_ct1.get_group(1) * QSC_ROWS,
              j0 = item_ct1.get_group(2) * QSC_KEYS,
              tid = item_ct1.get_local_id(2), warp = tid >> 5, lane = tid & 31;
    const int tl = sycl::min(T, t0 + QSC_ROWS) -
                   1; // the block's last row sees the most blocks
    const int nv_max = (*pos0p + tl + 1) / QSA_POOL;
    if (nv_max <= QSA_TOP || j0 >= nv_max) return;

    for (int e = tid; e < QSC_ROWS * QSA_HEADS * QSA_DIM; e += 256) {
        const int r = e / QSA_DIM, d = e % QSA_DIM;
        const float x = t0 + r / QSA_HEADS < T ? q[(size_t) t0 * QSA_HEADS * QSA_DIM + e] : 0.f;
        const sycl::half h =
            sycl::vec<float, 1>(x)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        qh[r][d] = h;
        ql[r][d] =
            sycl::vec<float, 1>(
                x - sycl::vec<sycl::half, 1>(h)
                        .convert<float, sycl::rounding_mode::automatic>()[0])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    
    item_ct1.barrier();
    const float w = sycl::rsqrt((float)QSA_DIM);
    // 16 output tiles (4 of rows x 4 of keys); warp w takes row tile w / 2, key tiles 2 (w % 2) and 2 (w % 2) + 1
    const int rt = warp / 2;
    for (int kk = 0; kk < 2; ++kk) {
        const int kt = (warp % 2) * 2 + kk, jt = j0 + kt * 16;
        if (jt >= nv_max) break;
        
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::a, 16, 16, 16, sycl::half,
            dpct::experimental::matrix::row_major>
            fa;
        
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::b, 16, 16, 16, sycl::half,
            dpct::experimental::matrix::col_major>
            fb;
        
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::accumulator, 16, 16, 16, float>
            fs;
        sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(
            sycl::ext::oneapi::this_work_item::get_sub_group(), fs.get(), 0.f);
#pragma unroll
        for (int d0 = 0; d0 < QSA_DIM; d0 += 16) {
            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                sycl::ext::oneapi::this_work_item::get_sub_group(), fb.get(),
                sycl::address_space_cast<
                    sycl::access::address_space::generic_space,
                    sycl::access::decorated::no, const sycl::half>(
                    pooled + kvp_block(g, jt) + d0),
                QSA_DIM); // 16 blocks never cross a page (48)
            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                sycl::ext::oneapi::this_work_item::get_sub_group(), fa.get(),
                sycl::address_space_cast<
                    sycl::access::address_space::generic_space,
                    sycl::access::decorated::no, sycl::half>(&qh[rt * 16][d0]),
                QSC_LD);
            sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
                sycl::ext::oneapi::this_work_item::get_sub_group(), fs.get(),
                fa.get(), fb.get(), fs.get());
            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                sycl::ext::oneapi::this_work_item::get_sub_group(), fa.get(),
                sycl::address_space_cast<
                    sycl::access::address_space::generic_space,
                    sycl::access::decorated::no, sycl::half>(&ql[rt * 16][d0]),
                QSC_LD);
            sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
                sycl::ext::oneapi::this_work_item::get_sub_group(), fs.get(),
                fa.get(), fb.get(), fs.get());
        }
        sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
            sycl::ext::oneapi::this_work_item::get_sub_group(), fs.get(),
            sycl::address_space_cast<sycl::access::address_space::generic_space,
                                     sycl::access::decorated::no, float>(
                &st[warp][0][0]),
            QSC_SLD,
            sycl::ext::oneapi::experimental::matrix::layout::row_major);
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        // the tile's 4 rows x 16 keys: the heads' relus summed in k_idx_score's order
        for (int e = lane; e < 4 * 16; e += 32) {
            const int r = e / 16, c = e % 16, t = t0 + rt * 4 + r, j = jt + c;
            if (t >= T || j >= max_pools) continue;
            float s = 0.f;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) s +=
                sycl::fmax(st[warp][r * QSA_HEADS + h][c], 0.f) * w;
            scores[(size_t) t * max_pools + j] = s;
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    }
}

// row t's attended positions on this card (block j owned by card kv_card(j)), as local indices kv_local_block(j) * 4 + p % 4, ascending,
// into cells[t] (n_cells[t] of them), or n_cells[t] = -1 when every visible block is kept (dense). A block per row: a radix select of the QSA_TOP-th highest score (scores >= 0: their bits
// order as unsigned), then the blocks above it and the first ties by index, in index order, then the tail.
constexpr int QSA_SEL_THREADS = 1024;

template <int Threads = QSA_SEL_THREADS>
void k_idx_select(const float *__restrict__ scores, int *__restrict__ cells,
                  int *__restrict__ n_cells, int max_pools, const int *pos0p,
                  int card, LaneRows lr, unsigned *hist, unsigned &prefix_s,
                  unsigned &want_s, int *s_gt, int *s_eq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2), tid = item_ct1.get_local_id(2),
              p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL,
              n_tail = (p + 1) % QSA_POOL;
    if (nv <= QSA_TOP) {
        if (tid == 0) n_cells[t] = -1;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + (size_t) t * max_pools);

    unsigned prefix = 0, mask = 0, want = QSA_TOP;   // find the want-th largest among keys matching prefix on mask
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += Threads) hist[b] = 0;
        
        
        item_ct1.barrier();
        // a warp's keys of one bin counted with one atomic (scores share their high bits: per key, the first pass's
        // atomics all hit two or three bins and queue)
        for (int j0 = 0; j0 < nv; j0 += Threads) {
            const int j = j0 + tid;
            const unsigned k = j < nv ? key[j] : 0u;
            const bool in = j < nv && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#else
            const unsigned peers = dpct::match_any_over_sub_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), 0xffffffffu,
                bin);
#endif
            if (in && (tid & 31) == dpct::ffs<int>(peers) - 1)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hist[bin], (unsigned)sycl::popcount(peers));
        }
        
        
        item_ct1.barrier();
        if (tid < 32) {   // the highest bin b > 0 where the count from 255 down reaches want, else 0; acc: the count above b
            const int lane = tid, hi = 255 - 8 * lane;   // lane l: bins hi .. hi - 7
            unsigned own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) own += hist[hi - i];
            unsigned before = own;   // inclusive scan over the lanes (higher bins first), then exclusive
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = dpct::shift_sub_group_right(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), before,
                    o);
                if (lane >= o) before += v;
            }
            before -= own;
            const unsigned hit = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        before + own >= want
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>());
            const int first = hit ? dpct::ffs<int>(hit) - 1 : 31;
            if (lane == first) {
                unsigned acc = before;
                int b = hi - 7;
                for (int i = 0; i < 8; ++i) {
                    const int bb = hi - i;
                    if (bb == 0 || acc + hist[bb] >= want) {
                        b = bb;
                        break;
                    }
                    acc += hist[bb];
                }
                prefix_s = prefix | ((unsigned) b << shift);
                want_s = want - acc;
            }
        }
        
        
        item_ct1.barrier();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        
        
        item_ct1.barrier();
    }
    // prefix: the threshold key; `want` ties at it are kept (the first by index). Contiguous chunks per thread
    const unsigned thr = prefix;
    const int chunk = (nv + Threads - 1) / Threads,
              j0 = sycl::min(nv, tid * chunk), j1 = sycl::min(nv, j0 + chunk);
    int n_gt = 0, n_eq = 0;
    for (int j = j0; j < j1; ++j) {
        const unsigned k = key[j];
        n_gt += k > thr;
        n_eq += k == thr;
    }

    s_gt[tid] = n_gt;
    s_eq[tid] = n_eq;
    
    item_ct1.barrier();
    // exclusive scans (Hillis-Steele over the block)
    for (int o = 1; o < Threads; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0, e = tid >= o ? s_eq[tid - o] : 0;
        
        
        item_ct1.barrier();
        s_gt[tid] += a;
        s_eq[tid] += e;
        
        
        item_ct1.barrier();
    }
    // which blocks of the chunk are taken (ties: the first `want` by index), then how many of them this card owns
    const int eq_start = s_eq[tid] - n_eq;
    int own = 0;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            own += take && kv_card(j) == card;
        }
    }
    
    item_ct1.barrier();
    s_gt[tid] = own;
    
    item_ct1.barrier();
    for (int o = 1; o < Threads; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0;
        
        
        item_ct1.barrier();
        s_gt[tid] += a;
        
        
        item_ct1.barrier();
    }
    int out = s_gt[tid] - own;   // owned selected blocks before this chunk
    int* row = cells + (size_t) t * QSA_MAXSEL;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            if (take && kv_card(j) == card) {
#pragma unroll
                for (int m = 0; m < QSA_POOL; ++m) row[out * QSA_POOL + m] = kv_local_block(j) * QSA_POOL + m;
                ++out;
            }
        }
    }
    if (tid == Threads - 1) {   // the last chunk's thread: `out` is the card's selected-block count
        int n = out * QSA_POOL;
        const int jt = p / QSA_POOL;   // the tail's block (incomplete)
        if (kv_card(jt) == card)
            for (int m = 0; m < n_tail; ++m) row[n++] = kv_local_block(jt) * QSA_POOL + (p - n_tail + 1 + m) % QSA_POOL;
        n_cells[t] = n;
    }
}

}  // namespace
