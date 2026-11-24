#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <time.h>
// The hyper-connection read (build_hc_mix) in one launch of H_NS * H_RG = 80
// blocks' work, with two grid barriers (included by kernels.cu). The Q8_0 mixes
// run on int8 activations with dp4a, as llama.cpp's CUDA path quantizes a Q8_0
// matmul's input to q8_1.
//   down:   block (slice s of 512 columns, row group): R' = R + bo * 2
//   sigmoid(inj / HC) for the slice, xs = R' * w_norm,
//           its square sums, xs as int8 per 32; partial dots of the slice for
//           the group's rows (the down rows, then the four inject rows in bf16
//           on xs itself). Then the block issues its up weights' loads.
//   reduce: (after the first barrier) one warp per 32 rows of lo and token: rs
//   per stream from the slices' square sums,
//           lo = silu(rs-weighted sum of the slices' partials / HC) as int8 per
//           32; the inject rows likewise.
//   up:     (after the second barrier) block b = 32 columns: every stream's up
//   row of each column . lo, then
//           mixed = mean_c xn_c * sigmoid(.), R <- R' when a write was folded
//           in, mixed as q8_1 when asked.
// The barriers need every block of the grid resident at once: the grid is one
// block per SM, up to 80 (hc_prepare), and a block of a card with fewer SMs (a
// 72-SM PG500-216) takes the down and up parts of 80 / SMs of them in turn,
// each the same sums in the same order, so the same bits. Nothing else may run
// on the card during a read (the engine's second-stream branches fork after
// it). A barrier that waits too long sets bit 4 of err.
namespace {

// row groups of 96 rows: lo's quantization blocks (32 rows) never straddle two groups
constexpr int H_SL = 512, H_NS = D / H_SL, H_SPS = N / H_SL, H_RPG = 96;
constexpr int H_ROWS = LR + HC, H_RG = (H_ROWS + H_RPG - 1) / H_RPG;
static_assert(H_RPG % 32 == 0, "lo blocks inside a group");
constexpr int H_TPRD = 8, H_DT = (H_RPG * H_TPRD + 31) / 32 * 32, H_BLOCKS = H_NS * H_RG;
// up: 32 columns per block, H_TPRU lanes per column; lane b < LR / 32 takes lo's 32-value block b
constexpr int H_UPC = 32, H_TPRU = 16, H_UB = LR / 32;
static_assert(H_BLOCKS * H_UPC == N, "one up block per down block");
static_assert(H_UPC * H_TPRU <= H_DT && H_UB <= H_TPRU, "the up lanes");
// the grid barrier's counter runs modulo hc_bar_mod(g), a multiple of the grid size g, so every barrier's g arrivals
// stay in one generation [k * g, (k + 1) * g) across the wrap
constexpr unsigned hc_bar_mod(unsigned g) { return 0xffffffffu / g * g; }

struct HcArgs {
    float* R;
    const float* bo;
    const float* inj_prev;
    const float* w_norm;
    Q8S w_down;
    Q8S w_up;
    const uint16_t* w_inj;
    float eps;
    float* mixed;
    float* inject;
    float* part;   // [H_NS][T][H_ROWS]
    float* ssp;    // [T][H_NS]: the slices' square sums
    int8_t* loq;   // [T][LR]: lo as int8 per 32
    float* lod;    // [T][LR / 32]: their scales
    unsigned* bar;       // the grid barrier's counter (modulo hc_bar_mod(grid))
    unsigned* err;       // bit 4: a barrier timed out
    block_q8_1* q8;      // mixed as q8_1 too, or null
    int8_t* tq = nullptr;   // and its split twin, or null
    float* td = nullptr;
};

__dpct_inline__ float rprime(const HcArgs &a, int t, int c, int i) {
    float r = a.R[(size_t) t * D + c * N + i];
    if (a.bo) r = sycl::fma((float)(a.bo[(size_t)t * N + i]),
                            2.0f * sigm(a.inj_prev[t * HC + c] / (float)HC), r);
    return r;
}

// 8 int8 weights (uint2) . 8 int8 activations (two ints)
__dpct_inline__ int dot8i(const sycl::uint2 w, const sycl::int2 x) {
    return dpct::dp4a((int)w.y(), x.y(), dpct::dp4a((int)w.x(), x.x(), 0));
}

// one warp quantizes one 32-value block: lane i's value v -> q[i], the block's scale -> *d (lane 0)
__dpct_inline__ void q8_block(float v, int8_t *q, float *d) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float amax = wmax(sycl::fabs(v));
    const float sc = amax / 127.0f;
    q[item_ct1.get_local_id(2) & 31] =
        amax == 0.0f ? 0 : (int8_t)sycl::round(v / sc);
    if ((item_ct1.get_local_id(2) & 31) == 0) *d = sc;
}

// every block of the grid gets here before any goes on (the counter grows by the grid size g per barrier, modulo
// hc_bar_mod(g)). The block's last thread arrives and waits until the counter has left its barrier's generation: it is
// then 1 to 2 * g - 1 past the generation's start, as no block can pass the next barrier before this one does. A fence
// waits for the thread's own loads too, and that thread loads no up weights
__dpct_inline__ void grid_sync(unsigned *bar, unsigned *err, unsigned g) {
    
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
        
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        const unsigned mod = hc_bar_mod(g);
        const unsigned old = dpct::atomic_fetch_compare_inc<
                           sycl::access::address_space::generic_space>(
                           bar, mod - 1u),
                       base = old - old % g;
        
        unsigned spins = 0;
        for (;;) {
            const unsigned cur = sycl::atomic_ref<unsigned, sycl::memory_order::acq_rel,
                sycl::memory_scope::device>(*bar).load(sycl::memory_order::acquire);
            if ((cur >= base ? cur - base : cur + (mod - base)) >= g) break;
            
            if (++spins >= sycl_port::spin_limit) {
                dpct::atomic_fetch_or<
                    sycl::access::address_space::generic_space>(err, 4u);
                break;
            }
            
        }
        
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
    }
    
    item_ct1.barrier();
}

// rs for token k, stream c: 1 / sqrt(mean of R'^2 over the stream + eps), from the slices' square sums
__dpct_inline__ float rs_of(const HcArgs &a, int k, int c) {
    float v = 0.f;
    
    for (int ss = 0; ss < H_SPS; ++ss) v +=
        *(a.ssp + (size_t)k * H_NS + c * H_SPS + ss);
    return sycl::rsqrt(v / (float)N + a.eps);
}

// the up part's arrays (after the second barrier) in the down part's xs (done before the first): 16 rows fit 48 KB
template <int T> struct HcUp {
    int8_t lq[T][LR];
    float ld[T][LR / 32];
    float rs[T][HC];
    float ms[T][H_UPC];
    float gs[T][HC][H_UPC];
};

// ONE: one grid barrier instead of two: after it every block makes the whole of lo itself (the same sums in the same
// order, so the same bits) from the slices' partials in L2, and block 0 the inject rows; ~T * 77 KB of L2 reads a block,
// so for small windows only
// WALK: the grid is short of H_BLOCKS and each block takes the shares vb = blockIdx.x + k * gridDim.x in turn; else
// vb = blockIdx.x alone, without the loops (the walk costs a register over the 79 of 80 SMs' form, and a stack slot at
// T = 15)
template <int T, bool ONE, bool WALK, int PHASE = 0>

void k_hc(HcArgs a, float xs[T][512 /*H_SL*/], int8_t xq[T][512 /*H_SL*/],
          float xd[T][16 /*H_SL / 32*/], float red[T][24 /*H_DT / 32*/]) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    static_assert(sizeof(HcUp<T>) <= sizeof(float) * T * H_SL && LR % 16 == 0,
                  "the up part's arrays inside xs, aligned");

    HcUp<T>& up = *reinterpret_cast<HcUp<T>*>(&xs[0][0]);
    auto& lq = up.lq;
    auto& ld = up.ld;
    auto& rs = up.rs;
    auto& ms = up.ms;
    auto& gs = up.gs;
    const int t = item_ct1.get_local_id(2), warp = t >> 5, lane = t & 31;
    const unsigned grid =
        WALK ? item_ct1.get_group_range(2) : (unsigned)H_BLOCKS;
    // ---------------- down, for each of the block's shares vb
    int vb = item_ct1.get_group(2);
    if constexpr (PHASE != 2) {
    do {
    
    
    if (vb != item_ct1.get_group(2))
        item_ct1.barrier(); // the last share's dots are done with xs / xq / xd
    const int s = vb / H_RG, grp = vb % H_RG;
    const int col0 = s * H_SL, c = col0 / N, i0 = col0 - c * N;
    // thread t < H_SL takes column t of every row, the rows' loads side by side (a row at a time, each cost a memory
    // latency: T of them on the group-0 blocks, which the others then waited for at the first barrier)
    static_assert(H_SL <= H_DT, "a thread per column");
    {
        float r[T];
        if (t < H_SL) {
            const float wn = a.w_norm[col0 + t];
#pragma unroll
            for (int k = 0; k < T; ++k) r[k] = rprime(a, k, c, i0 + t);
#pragma unroll
            for (int k = 0; k < T; ++k) xs[k][t] = r[k] * wn;
        }
        if (grp == 0) {
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float v = wsum(t < H_SL ? r[k] * r[k] : 0.f);
                if (lane == 0) red[k][warp] = v;
            }
        }
    }
    
    
    item_ct1.barrier();
    if (grp == 0 && t < T) {
        float v = 0.f;
        for (int w = 0; w < H_DT / 32; ++w) v += red[t][w];
        a.ssp[(size_t) t * H_NS + s] = v;
    }
    for (int q = warp; q < T * (H_SL / 32); q += H_DT / 32) {
        const int k = q / (H_SL / 32), blk = q % (H_SL / 32);
        q8_block(xs[k][blk * 32 + lane], &xq[k][blk * 32], &xd[k][blk]);
    }
    
    
    item_ct1.barrier();
    {
        constexpr int NQ = H_SL / 8 / H_TPRD;
        const int j = t % H_TPRD, row = grp * H_RPG + t / H_TPRD;
        const int nrows = a.w_inj ? H_ROWS : LR;
        const bool live = t / H_TPRD < H_RPG && row < nrows;
        const int rr = live ? row : 0;
        // the thread's weights to registers, then one pass per row t (rolled: unrolled over T the kernel is tens of KB
        // of code, fetched cold at every launch)
        if (rr < LR) {
            const sycl::uint2 *w2 = reinterpret_cast<const sycl::uint2 *>(
                a.w_down.q + (size_t)rr * D + col0);
            const uint16_t* wd = a.w_down.d + ((size_t) rr * D + col0) / 32;
            sycl::uint2 u[NQ];
            float sc[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                
                u[q] = *(w2 + q * H_TPRD + j);
                
                sc[q] = f16(*(wd + (q * H_TPRD + j) / 4));
            }
#pragma unroll 1
            for (int k = 0; k < T; ++k) {
                float acc = 0.f;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const int ch = q * H_TPRD + j;
                    acc = sycl::fma(
                        sc[q] * xd[k][ch / 4],
                        (float)dot8i(u[q],
                                     *reinterpret_cast<const sycl::int2 *>(
                                         &xq[k][ch * 8])),
                        acc);
                }
                for (int o = 1; o < H_TPRD; o <<= 1) acc +=
                    dpct::permute_sub_group_by_xor(
                        sycl::ext::oneapi::this_work_item::get_sub_group(), acc,
                        o);
                if (live && j == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
            }
        } else {
            const sycl::uint4 *w4 = reinterpret_cast<const sycl::uint4 *>(
                a.w_inj + (size_t)(rr - LR) * D + col0);
            sycl::uint4 u[NQ];
#pragma unroll
            
            for (int q = 0; q < NQ; ++q) u[q] = *(w4 + q * H_TPRD + j);
#pragma unroll 1
            for (int k = 0; k < T; ++k) {
                float acc = 0.f;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const int ch = q * H_TPRD + j;
                    const uint16_t* hh = reinterpret_cast<const uint16_t*>(&u[q]);
                    const sycl::float4 x0 =
                        *reinterpret_cast<const sycl::float4 *>(&xs[k][ch * 8]);
                    const sycl::float4 x1 =
                        *reinterpret_cast<const sycl::float4 *>(
                            &xs[k][ch * 8 + 4]);
                    acc += bf(hh[0]) * x0.x() + bf(hh[1]) * x0.y() +
                           bf(hh[2]) * x0.z() + bf(hh[3]) * x0.w() +
                           bf(hh[4]) * x1.x() + bf(hh[5]) * x1.y() +
                           bf(hh[6]) * x1.z() + bf(hh[7]) * x1.w();
                }
                for (int o = 1; o < H_TPRD; o <<= 1) acc +=
                    dpct::permute_sub_group_by_xor(
                        sycl::ext::oneapi::this_work_item::get_sub_group(), acc,
                        o);
                if (live && j == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
            }
        }
    }
    } while (WALK && (vb += grid) < H_BLOCKS);
    }
    if constexpr (PHASE == 1) return;
    // the up weights of share vb's 32 columns: lane uj < H_UB of column udd takes block uj of each stream's row. The
    // block's first share's are in flight through the reduction (nothing reads them before the barrier, so nothing
    // waits for them); a later share's load after the last share's up part
    const int uj = t % H_TPRU, udd = t / H_TPRU;
    const bool ulive = t < H_UPC * H_TPRU && uj < H_UB;
    static_assert(H_UPC * H_TPRU < H_DT, "the block's last thread has no up lane (it runs the grid barrier)");
    sycl::uint4 wq[HC][2] = {};
    uint16_t wsh[HC] = {};
    const auto load_up = [&](int vb) {
        if (ulive)
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                const size_t urow = (size_t) cc * N + vb * H_UPC + udd;
                const sycl::uint4 *wp = reinterpret_cast<const sycl::uint4 *>(
                    a.w_up.q + urow * LR + uj * 32);
                
                wq[cc][0] = *wp;
                
                wq[cc][1] = *(wp + 1);
                
                wsh[cc] = *(a.w_up.d + urow * (LR / 32) + uj);
            }
    };
    load_up(item_ct1.get_group(2));
    if constexpr (PHASE == 0) grid_sync(a.bar, a.err, grid);
    if constexpr (ONE) {
        // ---------------- reduce, in every block: rs, then lo per (token k, lo block b) a warp into lq / ld
        if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
        
        item_ct1.barrier();
        for (int it = warp; it < T * (LR / 32); it += H_DT / 32) {
            const int k = it / (LR / 32), r = (it % (LR / 32)) * 32 + lane;
            float pv[H_NS];
#pragma unroll
            
            for (int q = 0; q < H_NS; ++q)
                pv[q] = *(a.part + ((size_t)q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = sycl::fma(rs[k][cc], p, sum);
            }
            q8_block(silu_(sum / (float) HC), &lq[k][r & ~31], &ld[k][r / 32]);
        }
        if (item_ct1.get_group(2) == 0 && a.w_inj && warp < T && lane < HC) {
            const int k = warp, r = LR + lane;
            float pv[H_NS];
#pragma unroll
            
            for (int q = 0; q < H_NS; ++q)
                pv[q] = *(a.part + ((size_t)q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = sycl::fma(rs[k][cc], p, sum);
            }
            a.inject[k * HC + lane] = sum;
        }
        
        item_ct1.barrier();
    } else {
    // ---------------- reduce: item (token k, lo block b) per warp, then each token's inject rows
    {
        constexpr int ITEMS = T * (LR / 32) + T;
        for (int it = item_ct1.get_group(2) * (H_DT / 32) + warp; it < ITEMS;
             it += grid * (H_DT / 32)) {
            const bool inj_item = it >= T * (LR / 32);
            const int k = inj_item ? it - T * (LR / 32) : it / (LR / 32);
            // every lane runs the same path (q8_block shuffles across the warp); only an inject item's first HC lanes
            // have a row
            const bool active = !inj_item || (a.w_inj && lane < HC);
            const int r = !active ? 0 : inj_item ? LR + lane : (it % (LR / 32)) * 32 + lane;
            float pv[H_NS];
#pragma unroll
            
            for (int q = 0; q < H_NS; ++q)
                pv[q] = *(a.part + ((size_t)q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = sycl::fma(rs_of(a, k, cc), p, sum);
            }
            if (inj_item) {
                if (active) a.inject[k * HC + lane] = sum;
            } else q8_block(silu_(sum / (float) HC), a.loq + k * LR + (r & ~31), a.lod + k * (LR / 32) + r / 32);
        }
    }
    grid_sync(a.bar, a.err, grid);
    // ---------------- up
    if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
    
    for (int e = t; e < T * LR / 4; e += H_DT) reinterpret_cast<int *>(
        &lq[0][0])[e] = *(reinterpret_cast<const int *>(a.loq) + e);
    
    for (int e = t; e < T * (LR / 32); e += H_DT)
        ld[e / (LR / 32)][e % (LR / 32)] = *(a.lod + e);
    
    item_ct1.barrier();
    }
    // ---------------- up, share vb = 32 columns b
    vb = item_ct1.get_group(2);
    do {
    if (vb != item_ct1.get_group(2)) {
        
        
        item_ct1.barrier(); // the last share is done with gs / ms
        load_up(vb);
    }
    float wsc[HC];
#pragma unroll
    for (int cc = 0; cc < HC; ++cc) wsc[cc] = ulive ? f16(wsh[cc]) : 0.f;
    // one pass per row k, rolled (see the down part): the lane's block of lo against its block of each stream's row
#pragma unroll 1
    for (int k = 0; k < T; ++k) {
        const sycl::int4 *xp =
            reinterpret_cast<const sycl::int4 *>(&lq[k][(ulive ? uj : 0) * 32]);
        const sycl::int4 x0 = xp[0], x1 = xp[1];
        const float dl = ld[k][ulive ? uj : 0];
        float acc[HC];
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
            const sycl::uint4 w0 = wq[cc][0], w1 = wq[cc][1];
            int sacc = dpct::dp4a((int)w0.x(), x0.x(), 0);
            sacc = dpct::dp4a((int)w0.y(), x0.y(), sacc);
            sacc = dpct::dp4a((int)w0.z(), x0.z(), sacc);
            sacc = dpct::dp4a((int)w0.w(), x0.w(), sacc);
            sacc = dpct::dp4a((int)w1.x(), x1.x(), sacc);
            sacc = dpct::dp4a((int)w1.y(), x1.y(), sacc);
            sacc = dpct::dp4a((int)w1.z(), x1.z(), sacc);
            sacc = dpct::dp4a((int)w1.w(), x1.w(), sacc);
            acc[cc] = wsc[cc] * dl * (float) sacc;
        }
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
#pragma unroll
            for (int o = 1; o < H_TPRU; o <<= 1) acc[cc] +=
                dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), acc[cc],
                    o);
            if (t < H_UPC * H_TPRU && uj == 0) gs[k][cc][udd] = acc[cc];   // the sigmoid below, once per element on
        }                                                                      // every lane (here a warp issued it for 2)
    }
    
    
    item_ct1.barrier();
    // item (k, c, col): this block's 32 columns of stream c, row k, coalesced along the columns
    const int b = vb;
    for (int e = t; e < T * HC * H_UPC; e += H_DT) {
        const int col = e % H_UPC, cc = (e / H_UPC) % HC, k = e / (H_UPC * HC), dc = b * H_UPC + col;
        const float rv = rprime(a, k, cc, dc);
        if (a.bo) a.R[(size_t) k * D + cc * N + dc] = rv;
        gs[k][cc][col] = sigm(gs[k][cc][col]) * (rv * a.w_norm[cc * N + dc] * rs[k][cc]);
    }
    
    
    item_ct1.barrier();
    for (int e = t; e < T * H_UPC; e += H_DT) {
        const int col = e % H_UPC, k = e / H_UPC;
        const float sum = (gs[k][0][col] + gs[k][1][col] + gs[k][2][col] + gs[k][3][col]) / (float) HC;
        a.mixed[(size_t) k * N + b * H_UPC + col] = sum;
        ms[k][col] = sum;
    }
    if (!a.q8) continue;
    
    
    item_ct1.barrier();
    static_assert(H_UPC == 32, "one q8_1 block per token per up block");
    if (warp < T) {
        const float xi = ms[warp][lane];
        const float amax = wmax(sycl::fabs(xi)), sm = wsum(xi);
        const float dq = amax / 127.0f;
        block_q8_1& yq = a.q8[(size_t) warp * (N / 32) + b];
        const int8_t qv = amax == 0.0f ? 0 : (int8_t)sycl::round(xi / dq);
        yq.qs[lane] = qv;
        if (lane == 0) yq.ds = sycl::half2(
            sycl::vec<float, 1>(dq)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0],
            sycl::vec<float, 1>(sm)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0]);
        if (a.tq) {
            a.tq[(size_t) warp * N + b * 32 + lane] = qv;
            if (lane == 0) a.td[(size_t)warp * (N / 32) + b] =
                sycl::vec<sycl::half, 1>(
                    sycl::vec<float, 1>(dq)
                        .convert<sycl::half,
                                 sycl::rounding_mode::automatic>()[0])
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        }
    }
    } while (WALK && (vb += grid) < H_BLOCKS);
}

template <int T, int PHASE>
void hc_stage(const HcArgs& a, dpct::queue_ptr st) {
    strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float[T][H_SL], 0> xs(cgh);
        sycl::local_accessor<int8_t[T][H_SL], 0> xq(cgh);
        sycl::local_accessor<float[T][H_SL / 32], 0> xd(cgh);
        sycl::local_accessor<float[T][H_DT / 32], 0> red(cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, H_BLOCKS * H_DT),
                              sycl::range<3>(1, 1, H_DT)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                k_hc<T, true, false, PHASE>(a, xs, xq, xd, red);
            });
    });
}

// one: k_hc's one-barrier form (built for windows of up to 8 rows); grid: the card's, from hc_prepare
template <int T>
void hc_launch(const HcArgs &a, dpct::queue_ptr st, bool one, int grid) {
    if (env_flag("FNX_HC_STAGE_SPLIT")) {
        // The queue boundary publishes every down partial before any up group reads it.
        hc_stage<T, 1>(a, st);
        hc_stage<T, 2>(a, st);
        return;
    }
    if constexpr (T <= 8) {
        if (one) {
            
                if (grid < H_BLOCKS) strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                    
                    sycl::local_accessor<float[T][512 /*H_SL*/], 0> xs_acc_ct1(
                        cgh);
                    
                    sycl::local_accessor<int8_t[T][512 /*H_SL*/], 0> xq_acc_ct1(
                        cgh);
                    
                    sycl::local_accessor<float[T][16 /*H_SL / 32*/], 0>
                        xd_acc_ct1(cgh);
                    
                    sycl::local_accessor<float[T][24 /*H_DT / 32*/], 0>
                        red_acc_ct1(cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                              sycl::range<3>(1, 1, H_DT),
                                          sycl::range<3>(1, 1, H_DT)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_hc<T, true, true>(a, xs_acc_ct1, xq_acc_ct1,
                                                    xd_acc_ct1, red_acc_ct1);
                            });
                });
            
                else strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
                    
                    sycl::local_accessor<float[T][512 /*H_SL*/], 0> xs_acc_ct1(
                        cgh);
                    
                    sycl::local_accessor<int8_t[T][512 /*H_SL*/], 0> xq_acc_ct1(
                        cgh);
                    
                    sycl::local_accessor<float[T][16 /*H_SL / 32*/], 0>
                        xd_acc_ct1(cgh);
                    
                    sycl::local_accessor<float[T][24 /*H_DT / 32*/], 0>
                        red_acc_ct1(cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                              sycl::range<3>(1, 1, H_DT),
                                          sycl::range<3>(1, 1, H_DT)),
                        [=](sycl::nd_item<3> item_ct1)
                            [[sycl::reqd_sub_group_size(32)]] {
                                k_hc<T, true, false>(a, xs_acc_ct1, xq_acc_ct1,
                                                     xd_acc_ct1, red_acc_ct1);
                            });
                });
            return;
        }
    }
    
        if (grid < H_BLOCKS) strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            sycl::local_accessor<float[T][512 /*H_SL*/], 0> xs_acc_ct1(cgh);
            
            sycl::local_accessor<int8_t[T][512 /*H_SL*/], 0> xq_acc_ct1(cgh);
            
            sycl::local_accessor<float[T][16 /*H_SL / 32*/], 0> xd_acc_ct1(cgh);
            
            sycl::local_accessor<float[T][24 /*H_DT / 32*/], 0> red_acc_ct1(
                cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                   sycl::range<3>(1, 1, H_DT),
                                               sycl::range<3>(1, 1, H_DT)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     k_hc<T, false, true>(
                                         a, xs_acc_ct1, xq_acc_ct1, xd_acc_ct1,
                                         red_acc_ct1);
                                 });
        });
    
        else strata::kprof::kp(st, __FILE__, __func__, __LINE__)->submit([&](sycl::handler &cgh) {
            
            sycl::local_accessor<float[T][512 /*H_SL*/], 0> xs_acc_ct1(cgh);
            
            sycl::local_accessor<int8_t[T][512 /*H_SL*/], 0> xq_acc_ct1(cgh);
            
            sycl::local_accessor<float[T][16 /*H_SL / 32*/], 0> xd_acc_ct1(cgh);
            
            sycl::local_accessor<float[T][24 /*H_DT / 32*/], 0> red_acc_ct1(
                cgh);

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                   sycl::range<3>(1, 1, H_DT),
                                               sycl::range<3>(1, 1, H_DT)),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(32)]] {
                                     k_hc<T, false, false>(
                                         a, xs_acc_ct1, xq_acc_ct1, xd_acc_ct1,
                                         red_acc_ct1);
                                 });
        });
}

}  // namespace
