// src/kernels/qsa_select_parity.cu - the QSA block scores and top-k (qsa_select.cu) against the kernels they
// replaced, kept here as the reference: scores bitwise, selections identical, for decode windows (1-8 queries,
// the captured graph's fixed grid) and the prompt path's batches (up to 256 queries), from the identity case to 200K
// cells, with exact score ties, zero scores and every tail length.  --bench times both.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ref {

using namespace strata::kernels;
constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t last = n_bid < max_blocks - 1 ? n_bid : max_blocks - 1;
    const int lane = threadIdx.x & 31;
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float4 q4[IDX_HEADS];
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) q4[h] = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= last;
         b += (int64_t) gridDim.x * SCORE_WARPS) {
        const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
        const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
        float score = 0.0f;
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) {
            float d = k4.x * q4[h].x + k4.y * q4[h].y + k4.z * q4[h].z + k4.w * q4[h].w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
            score += d > 0.0f ? d : 0.0f;
        }
        if (lane == 0) {
            if (b == n_bid && n_kv % R != 0) score += 1e9f;
            out[qi * max_blocks + b] = score;
        }
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

void scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
            int64_t max_blocks, float* out, cudaStream_t s, int64_t grid_blocks) {
    const int64_t covered = grid_blocks < max_blocks ? grid_blocks : max_blocks;
    const dim3 grid((unsigned) ((covered + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, s>>>(pooled, dead, q_idx, steps, max_blocks, out);
}

void topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap, int32_t* ids,
          cudaStream_t s) {
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, s>>>(scores, steps, max_blocks, cap, ids);
}

}  // namespace ref

namespace {

using namespace strata::kernels;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}

constexpr int64_t kMaxCells = 200192;   // the configs' --max-context
constexpr int QD = 512;                 // a query's floats: 4 heads x 128

struct Data {
    QsaShapes s = qsa_real_shapes();
    int64_t max_blocks = kMaxCells / 4 + 2;
    int64_t cap = qsa_selection_width(kTopkMaxCells, qsa_real_shapes());
    float *pooled = nullptr, *dead = nullptr, *q = nullptr, *sc_ref = nullptr, *sc_new = nullptr;
    int32_t *steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    int64_t max_q = 256;
};

// Keys like the indexer's: most N(0,1); runs of copies (exact score ties, so the threshold often falls inside a
// run of equal keys and blocks are taken in part), all-zero keys (scores exactly 0) and -0.0 components.
void fill(Data& d, std::mt19937& rng) {
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> pooled((size_t) d.max_blocks * 128);
    for (int64_t b = 0; b < d.max_blocks; ++b) {
        float* row = pooled.data() + (size_t) b * 128;
        const int kind = (int) (rng() % 64);
        if (b > 0 && kind < 6) std::memcpy(row, row - 128, 128 * sizeof(float));   // a copy of the previous block
        else if (kind == 6) std::fill(row, row + 128, 0.0f);
        else
            for (int i = 0; i < 128; ++i) row[i] = (rng() % 97 == 0) ? -0.0f : n01(rng);
    }
    std::vector<float> dead(128);
    for (auto& x : dead) x = n01(rng);
    std::vector<float> q((size_t) d.max_q * QD);
    for (auto& x : q) x = 0.3f * n01(rng);
    d.pooled = dev<float>(pooled.size());
    d.dead = dev<float>(128);
    d.q = dev<float>(q.size());
    check(cudaMemcpy(d.pooled, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice), "up pooled");
    check(cudaMemcpy(d.dead, dead.data(), 128 * 4, cudaMemcpyHostToDevice), "up dead");
    check(cudaMemcpy(d.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "up q");
    d.sc_ref = dev<float>((size_t) d.max_q * d.max_blocks);
    d.sc_new = dev<float>((size_t) d.max_q * d.max_blocks);
    d.steps = dev<int32_t>((size_t) d.max_q * kStepCount);
    d.ids_ref = dev<int32_t>((size_t) d.max_q * d.cap);
    d.ids_new = dev<int32_t>((size_t) d.max_q * d.cap);
}

std::vector<int32_t> steps_from(const Data& d, int64_t pos0, int64_t nq) {
    std::vector<int32_t> st((size_t) nq * kStepCount);
    for (int64_t i = 0; i < nq; ++i) qsa_step_fill(st.data() + i * kStepCount, pos0 + i, d.s);
    check(cudaMemcpy(d.steps, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "up steps");
    return st;
}

// both implementations on queries at positions pos0.., then scores bitwise and selections equal
int compare(Data& d, int64_t pos0, int64_t nq, int64_t grid_blocks, cudaStream_t s, const char* what) {
    const std::vector<int32_t> st = steps_from(d, pos0, nq);
    check(cudaMemset(d.sc_ref, 0xff, (size_t) nq * d.max_blocks * 4), "clear");
    check(cudaMemset(d.sc_new, 0xff, (size_t) nq * d.max_blocks * 4), "clear");
    check(cudaMemset(d.ids_ref, 0xff, (size_t) nq * d.cap * 4), "clear");
    check(cudaMemset(d.ids_new, 0xff, (size_t) nq * d.cap * 4), "clear");
    ref::scores(d.pooled, d.dead, d.q, d.steps, nq, d.max_blocks, d.sc_ref, s, grid_blocks);
    ref::topk(d.sc_ref, d.steps, nq, d.max_blocks, d.cap, d.ids_ref, s);
    qsa_block_scores(d.pooled, d.dead, d.q, d.steps, nq, d.max_blocks, d.s, d.sc_new, s, grid_blocks);
    qsa_block_topk(d.sc_new, d.steps, nq, d.max_blocks, d.cap, d.s, d.ids_new, s);
    check(cudaStreamSynchronize(s), what);
    std::vector<uint32_t> a((size_t) nq * d.max_blocks), b(a.size());
    std::vector<int32_t> ia((size_t) nq * d.cap), ib(ia.size());
    check(cudaMemcpy(a.data(), d.sc_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down");
    check(cudaMemcpy(b.data(), d.sc_new, b.size() * 4, cudaMemcpyDeviceToHost), "down");
    check(cudaMemcpy(ia.data(), d.ids_ref, ia.size() * 4, cudaMemcpyDeviceToHost), "down");
    check(cudaMemcpy(ib.data(), d.ids_new, ib.size() * 4, cudaMemcpyDeviceToHost), "down");
    int bad = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t n_bid = st[(size_t) (i * kStepCount + kStepNBid)];
        const int64_t width = st[(size_t) (i * kStepCount + kStepWidth)];
        const int64_t last = std::min(n_bid, d.max_blocks - 1);
        for (int64_t k = 0; k <= last; ++k) {
            const size_t o = (size_t) (i * d.max_blocks + k);
            if (a[o] != b[o]) {
                if (bad < 5) {
                    float fa, fb;
                    std::memcpy(&fa, &a[o], 4);
                    std::memcpy(&fb, &b[o], 4);
                    std::fprintf(stderr, "%s: pos %lld, block %lld: score %.9g (0x%08x) vs %.9g (0x%08x)\n", what,
                                 (long long) (pos0 + i), (long long) k, fa, a[o], fb, b[o]);
                }
                ++bad;
            }
        }
        for (int64_t k = 0; k < width; ++k) {
            const size_t o = (size_t) (i * d.cap + k);
            if (ia[o] != ib[o]) {
                if (bad < 5)
                    std::fprintf(stderr, "%s: pos %lld, selection entry %lld: cell %d vs %d\n", what,
                                 (long long) (pos0 + i), (long long) k, ia[o], ib[o]);
                ++bad;
            }
        }
    }
    return bad;
}

int selftest(Data& d, cudaStream_t s) {
    int bad = 0, cases = 0;
    // decode windows: the captured graph's fixed grid, 1-8 queries at consecutive positions
    const int64_t pos_dec[] = {0, 3, 2046, 2047, 2050, 2051, 2052, 8190, 16381, 32766, 65533, 131070, 200180};
    for (int64_t p : pos_dec)
        for (int64_t nq : {1, 2, 3, 4, 8}) {
            if (p + nq > kMaxCells) continue;
            bad += compare(d, p, nq, qsa_score_grid_blocks, s, "decode");
            ++cases;
        }
    // prompt batches: 256 queries and a sub-chunk's last, shorter batch (11 go to the decode windows' kernel), the grid
    // covering the batch's last block
    const int64_t pos_pre[][2] = {{0, 256},          {1900, 256},        {16384, 256}, {65536 - 100, 256},
                                  {131072 - 256, 256}, {199936 - 1, 256}, {5000, 12},   {70001, 37},
                                  {131072 - 100, 100}, {199936 - 255, 255}, {9000, 11}};
    for (const auto& pq : pos_pre) {
        const int64_t p = pq[0], nq = pq[1], last = p + nq - 1;
        bad += compare(d, p, nq, (last + 1) / 4 + 1, s, "prompt");
        ++cases;
    }
    std::printf("qsa_select: %s (%d cases: scores bitwise, selections equal)\n", bad ? "MISMATCH" : "PASS", cases);
    return bad;
}

// GPU time per call: `reps` calls captured in a graph, as the engine runs them (a launch outside a graph costs more
// than these kernels on Windows)
template <typename F> float time_ms(cudaStream_t s, int reps, const F& fn) {
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < reps; ++i) fn();
    check(cudaStreamEndCapture(s, &g), "capture");
    check(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    check(cudaGraphLaunch(ge, s), "launch");
    cudaEventRecord(a, s);
    for (int i = 0; i < 3; ++i) check(cudaGraphLaunch(ge, s), "launch");
    cudaEventRecord(b, s);
    check(cudaEventSynchronize(b), "bench");
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms / (float) (3 * reps);
}

void bench(Data& d, cudaStream_t s) {
    std::printf("%-26s %12s %12s %12s %12s\n", "case", "ref scores", "new scores", "ref top-k", "new top-k");
    struct Case { const char* name; int64_t pos; int64_t nq; bool prompt; };
    const Case cs[] = {{"decode 4 q, 1K cells", 1024, 4, false},      {"decode 4 q, 2K cells", 2048, 4, false},
                       {"decode 4 q, 16K cells", 16384, 4, false},
                       {"decode 4 q, 64K cells", 65536, 4, false},    {"decode 4 q, 131K cells", 131072, 4, false},
                       {"decode 1 q, 131K cells", 131072, 1, false},  {"prompt 256 q at 16K", 16384 - 256, 256, true},
                       {"prompt 256 q at 64K", 65536 - 256, 256, true},
                       {"prompt 256 q at 131K", 131072 - 256, 256, true},
                       {"prompt 16 q at 131K", 131072 - 16, 16, true},  {"prompt 64 q at 131K", 131072 - 64, 64, true}};
    for (const Case& c : cs) {
        steps_from(d, c.pos, c.nq);
        const int64_t grid = c.prompt ? (c.pos + c.nq) / 4 + 1 : qsa_score_grid_blocks;
        const int reps = c.prompt ? 5 : 50;
        const float rs = time_ms(s, reps, [&] {
            ref::scores(d.pooled, d.dead, d.q, d.steps, c.nq, d.max_blocks, d.sc_ref, s, grid);
        });
        const float ns = time_ms(s, reps, [&] {
            qsa_block_scores(d.pooled, d.dead, d.q, d.steps, c.nq, d.max_blocks, d.s, d.sc_new, s, grid);
        });
        const float rt = time_ms(s, reps, [&] {
            ref::topk(d.sc_ref, d.steps, c.nq, d.max_blocks, d.cap, d.ids_ref, s);
        });
        const float nt = time_ms(s, reps, [&] {
            qsa_block_topk(d.sc_new, d.steps, c.nq, d.max_blocks, d.cap, d.s, d.ids_new, s);
        });
        std::printf("%-26s %9.1f us %9.1f us %9.1f us %9.1f us\n", c.name, rs * 1e3f, ns * 1e3f, rt * 1e3f, nt * 1e3f);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") do_bench = true;
        else {
            std::fprintf(stderr, "usage: qsa_select_parity [--bench]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260926);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    Data d;
    fill(d, rng);
    const int bad = selftest(d, s);
    if (do_bench) bench(d, s);
    return bad ? 1 : 0;
}
