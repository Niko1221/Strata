// The routed experts in three launches (included by kernels.cu, after k_mmvq's helpers):
//   router + route: the logits, top-10 per token, their weights, and the job list (one job per distinct expert);
//   gate|up: every job's stacked [gate; up] rows for its tokens;
//   down: each block makes its job's silu(gate) * up rows as q8_1 in shared memory, then the down rows.
// Output rows: pair t*TOPK+k. The shared expert runs beside them as dense launches into rows T*TOPK+t; the weighted
// sum happens in the ring.
namespace {

// The router and the routing in one launch. logits rows of NEXP + 1: the router's, then the shared expert's gate,
// a warp per row. The last block to finish selects each token's top 10 by logit (softmax is monotonic; ties: the lower
// index), a warp per token, the weights the selected softmax renormalized (exp(l - l_max) over the ten), and builds
// the jobs; with an ExpertCache (ec.n_slots) its first warp then plans the layer's slots (cache_plan).
constexpr int RR_THREADS = 256, RR_WPR = 2, RR_ROWS = RR_THREADS / 32 / RR_WPR;   // two warps per logit row

// one warp: the layer's jobs whose experts are in a slot keep it (ref 2: not to be taken); each other one takes the
// slot of the layer's pool that the pool's CLOCK hand reaches first with ref 0 (a ref 1 it passes drops to 0), its old
// expert pointing back at RAM. The hand goes round the pool's slots past its lent ones. list: how many took a slot,
// then their ids
void cache_plan(const Job* jobs, int nj, int layer, const ExpertCache& ec, int* list) {
    const int lane = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                         .get_local_id(2) &
                     31,
              base = layer * NEXP, p = ec.pool_of[layer], lent = ec.lent[p];
    // an arena's pools hold every expert until it lends, and then only its own pool's experts can be missing
    if (ec.arena_full && (p != ec.arena_pool || lent == 0)) {
        if (lane == 0) list[0] = 0;
        return;
    }
    const int s0 = ec.s0[p] + lent, n = ec.n[p] - lent;
    for (int j = lane; j < nj; j += 32)
        if (const int s = ec.slot_of[base + jobs[j].w]; s >= 0) ec.ref[s] = 2;
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
    int hand = ec.hand[p] - s0, k = 0;   // within the slots not lent
    if (hand < 0 || hand >= n) hand = 0;
    // the misses found 32 jobs at a time (a load each, side by side), then taken in job order
    for (int j0 = 0; j0 < nj; j0 += 32) {
        unsigned miss = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffffu &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                    j0 + lane < nj && ec.slot_of[base + jobs[j0 + lane].w] < 0
                ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                              .get_local_linear_id())
                : 0,
            sycl::ext::oneapi::plus<>());
        while (miss) {
            const int g = base + jobs[j0 + dpct::ffs<int>(miss) - 1].w;
            miss &= miss - 1;
            int victim = -1;
            while (victim < 0) {
                const int idx = s0 + (hand + lane) % n;
                const unsigned char r = ec.ref[idx];
                const unsigned free = sycl::reduce_over_group(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    (0xffffffffu &
                     (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id())) &&
                            r == 0
                        ? (0x1
                           << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                        : 0,
                    sycl::ext::oneapi::plus<>());
                const int f = free ? dpct::ffs<int>(free) - 1 : 32;
                if (lane < f && r == 1) ec.ref[idx] = 0;
                if (free) victim = s0 + (hand + f) % n;
                hand = (hand + (free ? f + 1 : 32)) % n;
            }
            if (lane == 0) {
                if (const int o = ec.owner[victim]; o >= 0) {
                    ec.slot_of[o] = -1;
                    ec.ep_gu[o] = ec.host_gu[o];
                    ec.ep_dn[o] = ec.host_dn[o];
                }
                ec.owner[victim] = g;
                ec.slot_of[g] = victim;
                ec.ref[victim] = 2;
                char* at = ec.pool[p] + (size_t) (victim - ec.s0[p]) * ec.slot_bytes[p];
                ec.ep_gu[g] = at;
                ec.ep_dn[g] = at + ec.dn_off[p];
                list[1 + k] = g;
            }
            ++k;
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
        }
    }
    for (int j = lane; j < nj; j += 32) ec.ref[ec.slot_of[base + jobs[j].w]] = 1;
    if (lane == 0) {
        ec.hand[p] = s0 + hand;
        list[0] = k;
    }
}

template <int T>

void k_router_route(const float *__restrict__ W, const float *__restrict__ x,
                    float *__restrict__ logits, int *__restrict__ topi,
                    float *__restrict__ topw, Job *__restrict__ gu,
                    Job *__restrict__ dn, int *__restrict__ njobs,
                    unsigned *__restrict__ counter, const ExpertCache ec,
                    int layer, int *list, const ShexpUp su,
                    float rsum[T][8 /*RR_THREADS / 32*/], bool &last, int *ex,
                    int *firstpair, int *slots, int *jobof,
                    unsigned char *first, int &nj, int phase = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(RR_THREADS == Q8S_THREADS,
                  "the shared expert's q8s blocks in the router's launch");
    constexpr int NC = N / 4 / 32 / RR_WPR, L = NEXP + 1;   // float4 chunks per lane
    const int nrr =
        (int)item_ct1.get_group_range(2) -
        su.blocks; // the router's own blocks; the shared expert's after them
    if (phase != 2 && (int)item_ct1.get_group(2) >= nrr) {
        shexp_up_block<T>(su, item_ct1.get_group(2) - nrr);
        return;
    }
    const int e = item_ct1.get_local_id(2), warp = e >> 5, lane = e & 31,
              rw = warp / RR_WPR, half = warp % RR_WPR;
    const int row = item_ct1.get_group(2) * RR_ROWS + rw;

    if (phase != 2) {
    if (row < L) {
        const sycl::float4 *w4 =
            reinterpret_cast<const sycl::float4 *>(W + (size_t)row * N) +
            half * NC * 32;
        sycl::float4 wv[NC];
#pragma unroll
        
        for (int i = 0; i < NC; ++i) wv[i] = *(w4 + i * 32 + lane);
        // unrolled over the tokens: each pair's x loads and sums run beside the others' (rolled, the pairs waited on
        // each other: one wave of blocks, so latency set the time). Each logit's summation order is unchanged
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const bool two = t + 1 < T;
            const sycl::float4 *x4 =
                reinterpret_cast<const sycl::float4 *>(x + (size_t)t * N) +
                half * NC * 32;
            const sycl::float4 *y4 = two ? x4 + N / 4 : x4;
            float a = 0.f, c = 0.f;
#pragma unroll
            for (int i = 0; i < NC; ++i) {
                const sycl::float4 b = x4[i * 32 + lane], d = y4[i * 32 + lane];
                a += wv[i].x() * b.x() + wv[i].y() * b.y() + wv[i].z() * b.z() +
                     wv[i].w() * b.w();
                c += wv[i].x() * d.x() + wv[i].y() * d.y() + wv[i].z() * d.z() +
                     wv[i].w() * d.w();
            }
            a = wsum(a);
            c = wsum(c);
            if (lane == 0) {
                rsum[t][warp] = a;
                if (two) rsum[t + 1][warp] = c;
            }
        }
    }
    
    item_ct1.barrier();
    for (int q = e; q < T * RR_ROWS; q += RR_THREADS) {
        const int t = q / RR_ROWS, r = q % RR_ROWS,
                  lrow = item_ct1.get_group(2) * RR_ROWS + r;
        if (lrow < L) logits[(size_t) t * L + lrow] = rsum[t][r * RR_WPR] + rsum[t][r * RR_WPR + 1];
    }

    
    if (phase == 1) return;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    
    item_ct1.barrier();
    if (e == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            counter, 1u) == (unsigned)nrr - 1;
    
    item_ct1.barrier();
    if (!last) return;
    
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    }
    // top 10 per token: warp w takes tokens w, w + 8, ...; lane l holds experts l + 32 i

    static_assert(T * TOPK < RR_THREADS, "a thread per pick and one more (the job count) below");
#pragma unroll 1
    for (int t = warp; t < T; t += RR_THREADS / 32) {
        float v[NEXP / 32];
#pragma unroll
        
        for (int i = 0; i < NEXP / 32; ++i)
            v[i] = *(logits + (size_t)t * L + lane + 32 * i);
        unsigned taken = 0;
        float top0 = 0.f, z = 0.f, wk = 0.f;
#pragma unroll 1
        for (int k = 0; k < TOPK; ++k) {
            float bv = -INFINITY;
            int bi = NEXP;
#pragma unroll
            for (int i = 0; i < NEXP / 32; ++i)
                if (!(taken >> i & 1u) && v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ov = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), bv, o);
                const int oi = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
            if (k == 0) top0 = bv;
            const float ek = sycl::native::exp(bv - top0);
            z += ek;
            if (lane == k) {
                wk = ek;
                topi[t * TOPK + k] = bi;
                ex[t * TOPK + k] = bi;
            }
        }
        if (lane < TOPK) topw[t * TOPK + lane] = wk / z;
    }
    
    item_ct1.barrier();
    // jobs: one per distinct expert, numbered in the order of each expert's first pick; a pair takes the next slot of
    // its expert's job (the order of a job's rows does not matter)
    constexpr int P = T * TOPK;

    for (int x2 = e; x2 < NEXP; x2 += RR_THREADS) {
        firstpair[x2] = P;
        slots[x2] = 0;
    }
    
    item_ct1.barrier();
    if (e < P)
        dpct::atomic_fetch_min<sycl::access::address_space::generic_space>(
            &firstpair[ex[e]], e);
    
    item_ct1.barrier();
    if (e < P) first[e] = firstpair[ex[e]] == e;
    
    item_ct1.barrier();
    if (e < P && first[e]) {
        int j = 0;
        for (int q = 0; q < e; ++q) j += first[q];
        jobof[ex[e]] = j;
        gu[j].w = dn[j].w = ex[e];
    }
    
    item_ct1.barrier();
    if (e < P) {
        const int j = jobof[ex[e]],
                  s = dpct::atomic_fetch_add<
                      sycl::access::address_space::generic_space>(&slots[ex[e]],
                                                                  1);
        gu[j].y[s] = e / TOPK;
        gu[j].out[s] = dn[j].y[s] = dn[j].out[s] = e;
    }
    
    item_ct1.barrier();
    if (e < P && first[e]) gu[jobof[ex[e]]].n = dn[jobof[ex[e]]].n = slots[ex[e]];

    if (e == P) {
        int n = 0;
        for (int q = 0; q < P; ++q) n += first[q];
        njobs[0] = nj = n;
        *counter = 0u;
    }
    if (!ec.n_slots) return;
    
    item_ct1.barrier();
    if (warp == 0) cache_plan(gu, nj, layer, ec, list);
}

// rows of one job: tmp[j] = W row . y[ys[j]] for j < n <= TT, G lanes per row
template <ggml_type type, int G, int TT>
__dpct_inline__ void
job_rows(const char *wr, bool live, int ncols, int n, const block_q8_1 *y,
         const int *ys, int y_stride, float *tmp, uint64_t *iq2s_grid,
         uint32_t *iq3xxs_grid, uint32_t *iq3s_grid, int8_t *kvalues_iq4nl) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = vdr_of<type>();
    constexpr int per_block = qi / vdr;
    static_assert(G >= per_block && G % per_block == 0, "G must cover a block");
    constexpr int bpi = G / per_block;
    const int gl = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                       .get_local_id(2) %
                   G,
              bpr = ncols / qk;
    if (live)
#pragma unroll 8
        for (int kbx = gl / per_block; kbx < bpr; kbx += bpi) {
            const int kby = kbx * (qk / QK8_1), kqs = vdr * (gl % per_block);
#pragma unroll
            for (int j = 0; j < TT; ++j)
                if (j < n) tmp[j] += vdot<type>(
                    wr, y + (size_t)ys[j] * y_stride + kby, kbx, kqs, iq2s_grid,
                    iq3xxs_grid, iq3s_grid, kvalues_iq4nl);
        }
}

// TT: the most rows a job has in this launch (the window's T): the loop unrolls that far, not to MAXT
template <int G, int TT = MAXT>
__dpct_inline__ void job_store(const float *tmp, int n, bool live, int row,
                               const int *outs, float *out, int out_stride) {
    const int gl =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % G;
#pragma unroll
    for (int j = 0; j < TT; ++j) {
        if (j >= n) break;
        float v = tmp[j];
#pragma unroll
        for (int o = G / 2; o; o >>= 1) v += dpct::permute_sub_group_by_xor(
            sycl::ext::oneapi::this_work_item::get_sub_group(), v, o);
        if (live && gl == 0) out[(size_t) outs[j] * out_stride + row] = v;
    }
}

// one thread's dot of U columns of a weight row (its unit `u`) with one q8_1 activation row
template <ggml_type type, int U>
__dpct_inline__ float unit_dot(const char *wrow, int u, const block_q8_1 *y,
                               uint64_t *iq2s_grid, uint32_t *iq3xxs_grid,
                               uint32_t *iq3s_grid, int8_t *kvalues_iq4nl) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = vdr_of<type>();
    float s = 0.f;
    if constexpr (U % qk == 0) {
#pragma unroll
        for (int b = 0; b < U / qk; ++b) {
            const int kbx = u * (U / qk) + b;
#pragma unroll
            for (int i = 0; i < qi / vdr; ++i) s +=
                vdot<type>(wrow, y + kbx * (qk / QK8_1), kbx, vdr * i,
                           iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl);
        }
    } else {
        assert(0); // a type whose blocks are wider than the unit: the launcher
                   // refuses it
    }
    return s;
}

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "moe_w.dp.hpp"

// gate|up: rows [0, 2 * ffs) of each job [0, *njobs)
template <ggml_type TA, int G, int TT>
void k_moe_gu(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                       const block_q8_1* __restrict__ y, float* __restrict__ out,
                                                       uint64_t *iq2s_grid,
                                                       uint32_t *iq3xxs_grid,
                                                       uint32_t *iq3s_grid,
                                                       int8_t *kvalues_iq4nl,
                                                       Job &jb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ji = item_ct1.get_group(1);
    if (ji >= *njobs) return;

    if (item_ct1.get_local_id(2) == 0) jb = jobs[ji];
    
    item_ct1.barrier();
    const int row = item_ct1.get_group(2) * (MMVQ_THREADS / G) +
                    item_ct1.get_local_id(2) / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
    float tmp[TT] = {};
    job_rows<TA, G, TT>(moe_expert(w, jb.w) + (size_t)r * w.row_a, live,
                        w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp, iq2s_grid,
                        iq3xxs_grid, iq3s_grid, kvalues_iq4nl);
    job_store<G, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

// down: one block per (job, rows_per_block rows). The block first makes its tokens' h = silu(gate) * up (gu rows
// jb.y[j] of [gate (ffs) | up (ffs)]) as q8_1 in shared memory; thread (row, u) then dots its 32-column unit u of the
// row with each token's h, and the block sums a row's units into out row jb.out[j]
constexpr int DN_THREADS = 256, DN_U = 32, DN_MAXB = FF_CARD_MAX / 32 > 8 ? FF_CARD_MAX / 32 : 8;   // DN_MAXB: ffs / 32 on any card

// every pair's h for k_moe_down, made once by k_moe_h (each of the ~70 down blocks of a job made its own). One MoE runs at
// a time on a card (its window's stream), so one buffer per card
inline dpct::global_memory<block_q8_1, 2> g_moe_hq(MAXT *TOPK, DN_MAXB);

// h = silu(gate) * up of pair row e of gu ([gate (ffs) | up (ffs)]) as q8_1, a warp per (pair, 32 columns)
void k_moe_h(const float* __restrict__ gu, int ffs, int npairs,
             dpct::accessor<block_q8_1, dpct::global, 2> g_moe_hq) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int nb = ffs / DN_U,
              q = item_ct1.get_group(2) * (item_ct1.get_local_range(2) / 32) +
                  (item_ct1.get_local_id(2) >> 5),
              lane = item_ct1.get_local_id(2) & 31;
    if (q >= npairs * nb) return;
    const int e = q / nb, b = q % nb;
    const float* g = gu + (size_t) e * 2 * ffs;
    const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
    const float amax = wmax(sycl::fabs(xi)), sm = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& h = g_moe_hq[e][b];
    h.qs[lane] = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / d);
    if (lane == 0) h.ds = sycl::half2(
        sycl::vec<float, 1>(d)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
        sycl::vec<float, 1>(sm)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
}

// PRE: the pairs' h made by k_moe_h (past 8 rows: kbench 16 rows 82.9 -> 70.7 us); else each block makes its own (the
// extra launch costs more than it saves: 5 rows 30.7 -> 31.7)
template <ggml_type TA, bool PRE>

void k_moe_down(MoeW w, const Job *__restrict__ jobs,
                const int *__restrict__ njobs, const float *__restrict__ gu,
                float *__restrict__ out, uint8_t *dpct_local,
                dpct::accessor<block_q8_1, dpct::global, 2> g_moe_hq,
                uint64_t *iq2s_grid, uint32_t *iq3xxs_grid, uint32_t *iq3s_grid,
                int8_t *kvalues_iq4nl, Job &jb,
                block_q8_1 hq[16 /*MAXT*/][10 /*DN_MAXB*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ji = item_ct1.get_group(1);
    if (ji >= *njobs) return;

    auto dn_part =
        (float *)dpct_local; // [the window's T][DN_THREADS] (launched with that
                             // much): MAXT's would halve the
    float(*part)[DN_THREADS] = reinterpret_cast<float(*)[DN_THREADS]>(dn_part);   // blocks an SM holds
    if (item_ct1.get_local_id(2) == 0) jb = jobs[ji];
    
    item_ct1.barrier();
    const int ffs = w.ncols, nb = ffs / DN_U, n = jb.n;
    if constexpr (PRE) {
        static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
        for (int q = item_ct1.get_local_id(2); q < n * nb * 9;
             q += DN_THREADS) {
            const int jb9 = q / 9, j = jb9 / nb, b = jb9 % nb;
            reinterpret_cast<int*>(&hq[j][b])[q % 9] = reinterpret_cast<const int*>(&g_moe_hq[jb.y[j]][b])[q % 9];
        }
    } else {
        const int warp = item_ct1.get_local_id(2) >> 5,
                  lane = item_ct1.get_local_id(2) & 31;
        for (int q = warp; q < n * nb; q += DN_THREADS / 32) {
            const int j = q / nb, b = q % nb;
            const float* g = gu + (size_t) jb.y[j] * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(sycl::fabs(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[j][b].qs[lane] = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / d);
            if (lane == 0) hq[j][b].ds = sycl::half2(
                sycl::vec<float, 1>(d)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
                sycl::vec<float, 1>(sm)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
        }
    }
    
    item_ct1.barrier();
    const int rpb = DN_THREADS / nb, r = item_ct1.get_local_id(2) / nb,
              u = item_ct1.get_local_id(2) % nb;
    const int row = item_ct1.get_group(2) * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    if (live) {
        const char* wr = moe_expert(w, jb.w) + (size_t) row * w.row_a;
        for (int j = 0; j < n; ++j) part[j][item_ct1.get_local_id(2)] =
            unit_dot<TA, DN_U>(wr, u, hq[j], iq2s_grid, iq3xxs_grid, iq3s_grid,
                               kvalues_iq4nl);
    }
    
    item_ct1.barrier();
    for (int e = item_ct1.get_local_id(2); e < n * rpb; e += DN_THREADS) {
        const int j = e / rpb, rr = e % rpb,
                  orow = item_ct1.get_group(2) * rpb + rr;
        if (orow >= w.nrows) continue;
        float s = 0.f;
        for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
        out[(size_t) jb.out[j] * w.nrows + orow] = s;
    }
}

template <ggml_type type> constexpr int per_block_of() { return ggml_cuda_type_traits<type>::qi / vdr_of<type>(); }
template <ggml_type type> constexpr bool unit_ok(int U) { return U % ggml_cuda_type_traits<type>::qk == 0; }

// lanes per row: enough to keep every block of a row busy (as mmvq picks them)
template <ggml_type TA> int moe_g(int ncols) {
    constexpr int pa = per_block_of<TA>();
    const int lanes = (ncols / ggml_cuda_type_traits<TA>::qk) * pa;
    int G = pa;
    while (G < lanes && G < 32) G *= 2;
    return G;
}

// k_moe_gu for BigJobs: the job's pairs MAXT at a time through the row
template <ggml_type TA, int G>
void k_moe_gu_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                           const block_q8_1* __restrict__ y, float* __restrict__ out,
                                                           uint64_t *iq2s_grid,
                                                           uint32_t *iq3xxs_grid,
                                                           uint32_t *iq3s_grid,
                                                           int8_t *kvalues_iq4nl,
                                                           BigJob &jb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ji = item_ct1.get_group(1);
    if (ji >= *njobs) return;

    load_bigjob(jb, jobs + ji);
    const int row = item_ct1.get_group(2) * (MMVQ_THREADS / G) +
                    item_ct1.get_local_id(2) / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int nb = sycl::min(MAXT, jb.n - b0);
        float tmp[MAXT] = {};
        job_rows<TA, G, MAXT>(moe_expert(w, jb.w) + (size_t)r * w.row_a, live,
                              w.ncols, nb, y, jb.y + b0, w.ncols / 32, tmp,
                              iq2s_grid, iq3xxs_grid, iq3s_grid, kvalues_iq4nl);
        job_store<G>(tmp, nb, live, row, jb.out + b0, out, w.nrows);
    }
}

// k_moe_down for BigJobs: per MAXT pairs, their h into shared memory, the dots, the sums
template <ggml_type TA>
void k_moe_down_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                            const float* __restrict__ gu, float* __restrict__ out,
                                                            uint64_t *iq2s_grid,
                                                            uint32_t *iq3xxs_grid,
                                                            uint32_t *iq3s_grid,
                                                            int8_t *kvalues_iq4nl,
                                                            BigJob &jb,
                                                            block_q8_1 hq[16/*MAXT*/][10/*DN_MAXB*/],
                                                            float part[16/*MAXT*/][256/*DN_THREADS*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ji = item_ct1.get_group(1);
    if (ji >= *njobs) return;

    load_bigjob(jb, jobs + ji);
    const int ffs = w.ncols, nb = ffs / DN_U,
              warp = item_ct1.get_local_id(2) >> 5,
              lane = item_ct1.get_local_id(2) & 31;
    const int rpb = DN_THREADS / nb, r = item_ct1.get_local_id(2) / nb,
              u = item_ct1.get_local_id(2) % nb;
    const int row = item_ct1.get_group(2) * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    const char* wr = moe_expert(w, jb.w) + (size_t) (live ? row : 0) * w.row_a;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int n = sycl::min(MAXT, jb.n - b0);
        for (int q = warp; q < n * nb; q += DN_THREADS / 32) {
            const int j = q / nb, b = q % nb;
            const float* g = gu + (size_t) jb.y[b0 + j] * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(sycl::fabs(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[j][b].qs[lane] = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / d);
            if (lane == 0) hq[j][b].ds = sycl::half2(
                sycl::vec<float, 1>(d)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
                sycl::vec<float, 1>(sm)
                    .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
        }
        
        
        item_ct1.barrier();
        if (live)
            for (int j = 0; j < n; ++j) part[j][item_ct1.get_local_id(2)] =
                unit_dot<TA, DN_U>(wr, u, hq[j], iq2s_grid, iq3xxs_grid,
                                   iq3s_grid, kvalues_iq4nl);
        
        
        item_ct1.barrier();
        for (int e = item_ct1.get_local_id(2); e < n * rpb; e += DN_THREADS) {
            const int j = e / rpb, rr = e % rpb,
                      orow = item_ct1.get_group(2) * rpb + rr;
            if (orow >= w.nrows) continue;
            float s = 0.f;
            for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
            out[(size_t) jb.out[b0 + j] * w.nrows + orow] = s;
        }
        
        
        item_ct1.barrier();
    }
}

template <ggml_type TA>
void moe_launch_big(bool down, const MoeW &w, const BigJob *jobs,
                    const int *njobs, int njobs_cap, const void *in, float *out,
                    dpct::queue_ptr s) {
    if (down) {
        if (!unit_ok<TA>(DN_U) || w.ncols / DN_U > DN_MAXB)
            throw std::runtime_error("moe_down: shape or type does not fit the units");
        const int rpb = DN_THREADS / (w.ncols / DN_U);
        {

            iq2s_grid.init(*s);
            iq3xxs_grid.init(*s);
            iq3s_grid.init(*s);
            kvalues_iq4nl.init(*s);

            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            s->submit([&](sycl::handler &cgh) {
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                sycl::local_accessor<BigJob, 0> jb_acc_ct1(cgh);
                
                
                sycl::local_accessor<block_q8_1[16 /*MAXT*/][10 /*DN_MAXB*/], 0>
                    hq_acc_ct1(cgh);
                
                
                sycl::local_accessor<float[16 /*MAXT*/][256 /*DN_THREADS*/], 0>
                    part_acc_ct1(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, njobs_cap,
                                       (w.nrows + rpb - 1) / rpb) *
                            sycl::range<3>(1, 1, DN_THREADS),
                        sycl::range<3>(1, 1, DN_THREADS)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            k_moe_down_big<TA>(
                                w, jobs, njobs, (const float *)in, out,
                                iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,
                                iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1,
                                jb_acc_ct1, hq_acc_ct1, part_acc_ct1);
                        });
            });
        }
        return;
    }
    const int G = moe_g<TA>(w.ncols);
#define FNX_MB(g)                                                              \
    {                                                                          \
        const dpct::dim3 grid(                                                 \
            (w.nrows + MMVQ_THREADS / g - 1) / (MMVQ_THREADS / g), njobs_cap); \
        {                                                                      \
                                                                               \
            iq2s_grid.init(*s);                                                \
            iq3xxs_grid.init(*s);                                              \
            iq3s_grid.init(*s);                                                \
            kvalues_iq4nl.init(*s);                                            \
                                                                               \
            dpct::has_capability_or_fail(s->get_device(),                      \
                                         {sycl::aspect::fp16});                \
                                                                               \
            s->submit([&](sycl::handler &cgh) {                                \
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();                  \
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();              \
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();                  \
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();          \
                                                                               \
                sycl::local_accessor<BigJob, 0> jb_acc_ct1(cgh);               \
                                                                               \
                auto w_ct0 = w;                                                \
                auto jobs_ct1 = jobs;                                          \
                auto njobs_ct2 = njobs;                                        \
                auto in_ct3 = (const block_q8_1 *)in;                          \
                auto out_ct4 = out;                                            \
                                                                               \
                cgh.parallel_for(                                              \
                    sycl::nd_range<3>(grid *                                   \
                                          sycl::range<3>(1, 1, MMVQ_THREADS),  \
                                      sycl::range<3>(1, 1, MMVQ_THREADS)),     \
                    [=](sycl::nd_item<3> item_ct1)                             \
                        [[sycl::reqd_sub_group_size(32)]] {                    \
                            k_moe_gu_big<TA, g>(                               \
                                w_ct0, jobs_ct1, njobs_ct2, in_ct3, out_ct4,   \
                                iq2s_grid_ptr_ct1, iq3xxs_grid_ptr_ct1,        \
                                iq3s_grid_ptr_ct1, kvalues_iq4nl_ptr_ct1,      \
                                jb_acc_ct1);                                   \
                        });                                                    \
            });                                                                \
        }                                                                      \
        return;                                                                \
    }
    if constexpr (per_block_of<TA>() <= 4) { if (G == 4) FNX_MB(4) }
    if constexpr (per_block_of<TA>() <= 8) { if (G == 8) FNX_MB(8) }
    if constexpr (per_block_of<TA>() <= 16) { if (G == 16) FNX_MB(16) }
    FNX_MB(32)
#undef FNX_MB
}

// the gate|up kernel for a window of T rows (a job holds at most T)
template <ggml_type TA, int G>
void gu_launch(int T, dpct::dim3 grid, dpct::queue_ptr s, const MoeW &w,
               const Job *jobs, const int *njobs, const block_q8_1 *y,
               float *out) {
    switch (T) {
    case 1: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 1>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 2: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 2>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 3: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 3>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 4: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 4>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 5: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 5>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 6: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 6>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 7: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 7>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 8: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 8>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 9: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 9>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 10: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 10>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 11: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 11>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 12: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 12>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 13: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 13>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 14: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 14>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 15: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 15>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
    case 16: {

        iq2s_grid.init(*s);
        iq3xxs_grid.init(*s);
        iq3s_grid.init(*s);
        kvalues_iq4nl.init(*s);

        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

        s->submit([&](sycl::handler &cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, MMVQ_THREADS),
                                  sycl::range<3>(1, 1, MMVQ_THREADS)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        k_moe_gu<TA, G, 16>(
                            w, jobs, njobs, y, out, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1);
                    });
        });
    } break;
        default: throw std::runtime_error("moe gate|up: T out of range");
    }
}

// gate|up with the shared expert's swiglu and down as extra rows of blocks (the other expert types' version of k_moe_gu_q4k's sd rows): 256-thread blocks (the shared
// expert's q8s blocks are Q8S_THREADS wide), G lanes a row as k_moe_gu has them, so every row sums its terms in the same order and the answers are the same bits
template <ggml_type TA, int G, int TT>
void k_moe_gu_sd(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                         const block_q8_1* __restrict__ y, float* __restrict__ out, const ShexpDown sd, int sd_rows,
                                                         uint64_t *iq2s_grid,
                                                         uint32_t *iq3xxs_grid,
                                                         uint32_t *iq3s_grid,
                                                         int8_t *kvalues_iq4nl,
                                                         int8_t *tq, float *td,
                                                         Job &jb) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if ((int)item_ct1.get_group(1) < sd_rows) {
        shexp_down_blocks<TT>(sd,
                              item_ct1.get_group(1) *
                                      item_ct1.get_group_range(2) +
                                  item_ct1.get_group(2),
                              sd_rows * item_ct1.get_group_range(2), tq, td);
        return;
    }
    const int ji = item_ct1.get_group(1) - sd_rows;
    if (ji >= *njobs) return;

    if (item_ct1.get_local_id(2) == 0) jb = jobs[ji];
    
    item_ct1.barrier();
    const int row = item_ct1.get_group(2) * (Q8S_THREADS / G) +
                    item_ct1.get_local_id(2) / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
    float tmp[TT] = {};
    job_rows<TA, G, TT>(moe_expert(w, jb.w) + (size_t)r * w.row_a, live,
                        w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp, iq2s_grid,
                        iq3xxs_grid, iq3s_grid, kvalues_iq4nl);
    job_store<G, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

template <ggml_type TA>
void gu_sd_launch(int T, const MoeW &w, const Job *jobs, const int *njobs,
                  int njobs_max, const block_q8_1 *y, float *out,
                  dpct::queue_ptr s, const ShexpDown &sd) {
    if (moe_g<TA>(w.ncols) != 32) throw std::runtime_error("moe gate|up: only 32 lanes a row are compiled (rows of N columns)");
    const int gx = (w.nrows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32), sd_blocks = (sd.rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
    const int sd_rows = sd.rows ? (sd_blocks + gx - 1) / gx : 0;
    const dpct::dim3 grid(gx, njobs_max + sd_rows);
#define FNX_GS(t)                                                              \
    case t: {                                                                  \
                                                                               \
        iq2s_grid.init(*s);                                                    \
        iq3xxs_grid.init(*s);                                                  \
        iq3s_grid.init(*s);                                                    \
        kvalues_iq4nl.init(*s);                                                \
                                                                               \
        dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});   \
                                                                               \
        strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) { \
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();                      \
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();                  \
            auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();                      \
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();              \
                                                                               \
            sycl::local_accessor<int8_t, 1> tq_acc_ct1(                        \
                sycl::range<1>(t * SH_MAXF), cgh);                             \
            sycl::local_accessor<float, 1> td_acc_ct1(                         \
                sycl::range<1>(t * (SH_MAXF / 32)), cgh);                      \
            sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);                      \
                                                                               \
            auto w_ct0 = w;                                                    \
            auto jobs_ct1 = jobs;                                              \
            auto njobs_ct2 = njobs;                                            \
            auto y_ct3 = y;                                                    \
            auto out_ct4 = out;                                                \
            auto sd_ct0 = sd;                                                  \
            auto sd_rows_ct6 = sd_rows;                                        \
                                                                               \
            cgh.parallel_for(                                                  \
                sycl::nd_range<3>(grid * sycl::range<3>(1, 1, Q8S_THREADS),    \
                                  sycl::range<3>(1, 1, Q8S_THREADS)),          \
                [=](sycl::nd_item<3> item_ct1)                                 \
                    [[sycl::reqd_sub_group_size(32)]] {                        \
                        k_moe_gu_sd<TA, 32, t>(                                \
                            w_ct0, jobs_ct1, njobs_ct2, y_ct3, out_ct4,        \
                            sd_ct0, sd_rows_ct6, iq2s_grid_ptr_ct1,            \
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,            \
                            kvalues_iq4nl_ptr_ct1,                             \
                            tq_acc_ct1                                         \
                                .get_multi_ptr<sycl::access::decorated::no>()  \
                                .get(),                                        \
                            td_acc_ct1                                         \
                                .get_multi_ptr<sycl::access::decorated::no>()  \
                                .get(),                                        \
                            jb_acc_ct1);                                       \
                    });                                                        \
        });                                                                    \
    } break;
    switch (T) {
        FNX_GS(1) FNX_GS(2) FNX_GS(3) FNX_GS(4) FNX_GS(5) FNX_GS(6) FNX_GS(7) FNX_GS(8)
        FNX_GS(9) FNX_GS(10) FNX_GS(11) FNX_GS(12) FNX_GS(13) FNX_GS(14) FNX_GS(15) FNX_GS(16)
        default: throw std::runtime_error("moe gate|up (shared expert): T out of range");
    }
#undef FNX_GS
}

template <ggml_type TA>
void moe_launch(bool down, const MoeW &w, const Job *jobs, const int *njobs,
                int njobs_max, const void *in, float *out, dpct::queue_ptr s,
                int T) {
    if (down) {
        if (!unit_ok<TA>(DN_U) || w.ncols / DN_U > DN_MAXB)
            throw std::runtime_error("moe_down: shape or type does not fit the units");
        const int rpb = DN_THREADS / (w.ncols / DN_U), nh = njobs_max * (w.ncols / DN_U);   // njobs_max: the window's pairs
        const dpct::dim3 grid((w.nrows + rpb - 1) / rpb, njobs_max);
        
        const size_t smem = sizeof(float) * T * DN_THREADS;
        if (T > 8) {
            if (njobs_max > MAXT * TOPK) throw std::runtime_error("moe_down: more pairs than g_moe_hq holds");
            {
                g_moe_hq.init(*s);

                strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                    auto g_moe_hq_acc_ct1 = g_moe_hq.get_access(cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, (nh + 7) / 8) *
                                              sycl::range<3>(1, 1, 256),
                                          sycl::range<3>(1, 1, 256)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_moe_h((const float *)in, w.ncols, njobs_max,
                                        g_moe_hq_acc_ct1);
                            });
                });
            }
            {

                g_moe_hq.init(*s);
                iq2s_grid.init(*s);
                iq3xxs_grid.init(*s);
                iq3s_grid.init(*s);
                kvalues_iq4nl.init(*s);

                dpct::has_capability_or_fail(s->get_device(),
                                             {sycl::aspect::fp16});

                strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                    auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                    auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                    auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                    auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(smem), cgh);
                    sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);
                    
                    
                    sycl::local_accessor<
                        block_q8_1[16 /*MAXT*/][10 /*DN_MAXB*/], 0>
                        hq_acc_ct1(cgh);
                    auto g_moe_hq_acc_ct1 = g_moe_hq.get_access(cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid *
                                              sycl::range<3>(1, 1, DN_THREADS),
                                          sycl::range<3>(1, 1, DN_THREADS)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_moe_down<TA, true>(
                                    w, jobs, njobs, (const float *)in, out,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get(),
                                    g_moe_hq_acc_ct1, iq2s_grid_ptr_ct1,
                                    iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                                    kvalues_iq4nl_ptr_ct1, jb_acc_ct1,
                                    hq_acc_ct1);
                            });
                });
            }
        } else {

            g_moe_hq.init(*s);
            iq2s_grid.init(*s);
            iq3xxs_grid.init(*s);
            iq3s_grid.init(*s);
            kvalues_iq4nl.init(*s);

            dpct::has_capability_or_fail(s->get_device(), {sycl::aspect::fp16});

            strata::kprof::kp(s, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
                auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
                auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
                auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(smem), cgh);
                sycl::local_accessor<Job, 0> jb_acc_ct1(cgh);
                
                
                sycl::local_accessor<block_q8_1[16 /*MAXT*/][10 /*DN_MAXB*/], 0>
                    hq_acc_ct1(cgh);
                auto g_moe_hq_acc_ct1 = g_moe_hq.get_access(cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, DN_THREADS),
                                      sycl::range<3>(1, 1, DN_THREADS)),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        k_moe_down<TA, false>(
                            w, jobs, njobs, (const float *)in, out,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            g_moe_hq_acc_ct1, iq2s_grid_ptr_ct1,
                            iq3xxs_grid_ptr_ct1, iq3s_grid_ptr_ct1,
                            kvalues_iq4nl_ptr_ct1, jb_acc_ct1, hq_acc_ct1);
                    });
            });
        }
        return;
    }
    // gate|up rows have N columns, where every type's lanes per row come to 32: only that variant is compiled (the
    // others, never used, were 192 of the file's kernels). Q4_K's gate|up has a kernel of its own (moe_q4k.cuh)
    if constexpr (TA == GGML_TYPE_Q4_K) {
        throw std::runtime_error("moe gate|up: Q4_K runs gu_q4k_launch");
    } else {
        if (moe_g<TA>(w.ncols) != 32) throw std::runtime_error("moe gate|up: only 32 lanes a row are compiled (rows of N columns)");
        const dpct::dim3 grid(
            (w.nrows + MMVQ_THREADS / 32 - 1) / (MMVQ_THREADS / 32), njobs_max);
        gu_launch<TA, 32>(T, grid, s, w, jobs, njobs, (const block_q8_1*) in, out);
    }
}


}  // namespace
