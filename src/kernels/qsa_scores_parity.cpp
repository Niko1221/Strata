// The prompt path's block scores on the thread-per-head kernel (an active-block count) against block_scores_kernel
// (no count, more than 8 queries) and an independent pre-PR baseline scorer, every score of blocks 0..n_bid bitwise;
// plus qsa_block_scores_rows and qsa_block_topk parity against qsa_block_scores (multi) for nq = 1..8 (including
// differing n_bid values and partial tail blocks).  GPU, synthetic.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}

#if defined(__HIPCC__)
// Independent pre-PR baseline warp scorer: unmodified k4.x*q4.x + k4.y*q4.y + k4.z*q4.z + k4.w*q4.w and __shfl_xor_sync
// reduction tree (no rows_dot4, no sel_warp_sum_f32), so any shared score-bit drift in qsa_select.cu is caught.
__global__ void __launch_bounds__(256) pre_pr_block_scores_kernel(const float* __restrict__ pooled,
                                                                  const float* __restrict__ dead,
                                                                  const float* __restrict__ q_idx,
                                                                  const int32_t* __restrict__ steps,
                                                                  int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * k::kStepCount;
    const int64_t n_kv = st[k::kStepNKv], n_bid = st[k::kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * 128;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * 4 * 128 + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * 128);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % 4 != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}
#endif

// ctx: the last query's position + 1; queries: a prompt chunk ending there; mode 0 normal values, 1 wide magnitudes
// and exact zeros, 2 small integers (many ties and cancellations)
struct Case { int64_t ctx, queries; int mode; bool time; };
bool check(Case c) {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head, max_blocks = c.ctx / s.idx_block + 2;
    std::mt19937 rng(91731 + (unsigned) c.ctx + 7 * (unsigned) c.queries + 101 * (unsigned) c.mode);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> small(-3, 3), expo(-20, 20), coin(0, 9);
    auto value = [&] {
        if (c.mode == 1) return coin(rng) == 0 ? 0.f : std::ldexp(nd(rng), expo(rng));
        if (c.mode == 2) return (float) small(rng);
        return nd(rng);
    };
    std::vector<float> pooled((size_t) (max_blocks * D)), dead((size_t) D), q((size_t) (c.queries * H * D));
    for (auto& v : pooled) v = value();
    for (auto& v : dead) v = value();
    for (auto& v : q) v = value();
    std::vector<int32_t> steps((size_t) (c.queries * k::kStepCount));
    for (int64_t i = 0; i < c.queries; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = c.ctx - c.queries + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const int64_t active = steps[(size_t) ((c.queries - 1) * k::kStepCount + k::kStepNBid)] + 1;
    float *dp = nullptr, *dd = nullptr, *dq = nullptr, *da = nullptr, *db = nullptr;
    int32_t* dt = nullptr;
    const size_t n_out = (size_t) (c.queries * max_blocks);
    ck(cudaMalloc(&dp, pooled.size() * 4)); ck(cudaMalloc(&dd, dead.size() * 4)); ck(cudaMalloc(&dq, q.size() * 4));
    ck(cudaMalloc(&dt, steps.size() * 4)); ck(cudaMalloc(&da, n_out * 4)); ck(cudaMalloc(&db, n_out * 4));
    ck(cudaMemcpy(dp, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dd, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemset(da, 0x7f, n_out * 4));   // different fill in each: an unwritten score cannot match by chance
    ck(cudaMemset(db, 0x3e, n_out * 4));
    k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, da, nullptr);           // block_scores_kernel
    k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, db, nullptr, active);   // the thread kernel
#if defined(__HIPCC__)
    float* dref = nullptr;
    ck(cudaMalloc(&dref, n_out * 4));
    ck(cudaMemset(dref, 0x55, n_out * 4));
    const dim3 rgrid((unsigned) ((max_blocks + 7) / 8), (unsigned) c.queries);
    pre_pr_block_scores_kernel<<<rgrid, 256>>>(dp, dd, dq, dt, max_blocks, dref);
#endif
    ck(cudaDeviceSynchronize());
    std::vector<uint32_t> a(n_out), b(n_out);
    ck(cudaMemcpy(a.data(), da, n_out * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), db, n_out * 4, cudaMemcpyDeviceToHost));
#if defined(__HIPCC__)
    std::vector<uint32_t> ref(n_out);
    ck(cudaMemcpy(ref.data(), dref, n_out * 4, cudaMemcpyDeviceToHost));
    ck(cudaFree(dref));
#endif
    bool ok = true;
    int64_t checked = 0;
    for (int64_t i = 0; i < c.queries && ok; ++i) {
        const int64_t n_bid = steps[(size_t) (i * k::kStepCount + k::kStepNBid)];
        for (int64_t j = 0; j <= n_bid; ++j, ++checked) {
            const size_t at = (size_t) (i * max_blocks + j);
            if (a[at] != b[at]) {
                float fa, fb;
                std::memcpy(&fa, &a[at], 4); std::memcpy(&fb, &b[at], 4);
                std::fprintf(stderr, "FAIL ctx=%lld queries=%lld mode=%d query=%lld block=%lld: %.9g vs %.9g\n",
                             (long long) c.ctx, (long long) c.queries, c.mode, (long long) i, (long long) j, fa, fb);
                ok = false;
                break;
            }
#if defined(__HIPCC__)
            if (a[at] != ref[at]) {
                float fa, fr;
                std::memcpy(&fa, &a[at], 4); std::memcpy(&fr, &ref[at], 4);
                std::fprintf(stderr, "FAIL vs pre-PR ref ctx=%lld queries=%lld mode=%d query=%lld block=%lld: %.9g vs %.9g\n",
                             (long long) c.ctx, (long long) c.queries, c.mode, (long long) i, (long long) j, fa, fr);
                ok = false;
                break;
            }
#endif
        }
    }
#if defined(__HIPCC__)
    if (ok) std::printf("ok   ctx=%7lld queries=%5lld mode=%d: %lld scores identical to thread and pre-PR baseline\n",
                        (long long) c.ctx, (long long) c.queries, c.mode, (long long) checked);
#else
    if (ok) std::printf("ok   ctx=%7lld queries=%5lld mode=%d: %lld scores identical\n", (long long) c.ctx,
                        (long long) c.queries, c.mode, (long long) checked);
#endif
    if (ok && c.time) {
        cudaEvent_t e0, e1, e2;
        ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1)); ck(cudaEventCreate(&e2));
        constexpr int N = 5;
        ck(cudaEventRecord(e0));
        for (int r = 0; r < N; ++r) k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, da, nullptr);
        ck(cudaEventRecord(e1));
        for (int r = 0; r < N; ++r) k::qsa_block_scores(dp, dd, dq, dt, c.queries, max_blocks, s, db, nullptr, active);
        ck(cudaEventRecord(e2));
        ck(cudaEventSynchronize(e2));
        float t_old = 0.f, t_new = 0.f;
        ck(cudaEventElapsedTime(&t_old, e0, e1)); ck(cudaEventElapsedTime(&t_new, e1, e2));
        std::printf("     time per call: warp kernel %.2f ms, thread kernel %.2f ms (%.2fx)\n", t_old / N, t_new / N,
                    t_old / t_new);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    }
    ck(cudaFree(dp)); ck(cudaFree(dd)); ck(cudaFree(dq)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    return ok;
}

// Verify qsa_block_scores_rows and qsa_block_topk against qsa_block_scores (multi) and pre-PR ref for nq = 1..8,
// including nq = 5..8, differing n_bid values across queries, and partial tail blocks (n_kv % 4 != 0).
bool check_rows_1_to_8() {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    constexpr int MAXQ = 8;
    const int64_t ctxs[] = {103, 4096, 32771};
    // step_mode 0: consecutive positions with partial tails; 1: wide spread n_bid values and partial tails
    for (int64_t ctx : ctxs) {
        const int64_t max_blocks = ctx / s.idx_block + 256;
        std::mt19937 rng(44109 + (unsigned) ctx);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> pooled((size_t) (max_blocks * D)), dead((size_t) D), q((size_t) (MAXQ * H * D));
        for (auto& v : pooled) v = nd(rng);
        for (auto& v : dead) v = nd(rng);
        for (auto& v : q) v = nd(rng);
        float *dp = nullptr, *dd = nullptr, *dq = nullptr, *da = nullptr, *db = nullptr;
        int32_t *dt = nullptr, *ida = nullptr, *idb = nullptr;
        ck(cudaMalloc(&dp, pooled.size() * 4)); ck(cudaMalloc(&dd, dead.size() * 4)); ck(cudaMalloc(&dq, q.size() * 4));
        ck(cudaMalloc(&dt, MAXQ * k::kStepCount * 4));
        ck(cudaMalloc(&da, (size_t) (MAXQ * max_blocks) * 4));
        ck(cudaMalloc(&db, (size_t) (MAXQ * max_blocks) * 4));
        ck(cudaMalloc(&ida, (size_t) (MAXQ * cap) * 4));
        ck(cudaMalloc(&idb, (size_t) (MAXQ * cap) * 4));
        ck(cudaMemcpy(dp, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dd, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
#if defined(__HIPCC__)
        float* dref = nullptr;
        ck(cudaMalloc(&dref, (size_t) (MAXQ * max_blocks) * 4));
#endif
        for (int step_mode = 0; step_mode < 2; ++step_mode) {
            std::vector<int32_t> steps(MAXQ * k::kStepCount);
            for (int i = 0; i < MAXQ; ++i) {
                int32_t* st = steps.data() + i * k::kStepCount;
                const int64_t pos = step_mode == 0 ? (ctx + i) : std::max<int64_t>(17 + i * 3, ctx - (MAXQ - 1 - i) * 511 + (i % 4));
                const int64_t n_kv = pos + 1;
                st[k::kStepPos] = (int32_t) pos;
                st[k::kStepNKv] = (int32_t) n_kv;
                st[k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
                st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
            }
            ck(cudaMemcpy(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
            for (int nq = 1; nq <= MAXQ; ++nq) {
                const size_t n_sc = (size_t) (nq * max_blocks), n_id = (size_t) (nq * cap);
                ck(cudaMemset(da, 0x7f, n_sc * 4));
                ck(cudaMemset(db, 0x3e, n_sc * 4));
                ck(cudaMemset(ida, 0x7f, n_id * 4));
                ck(cudaMemset(idb, 0x3e, n_id * 4));
                k::qsa_block_scores(dp, dd, dq, dt, nq, max_blocks, s, da, nullptr);
                if (!k::qsa_block_scores_rows(dp, dd, dq, dt, nq, max_blocks, s, db, nullptr)) {
                    std::fprintf(stderr, "FAIL rows refused nq=%d\n", nq);
                    return false;
                }
                k::qsa_block_topk(da, dt, nq, max_blocks, cap, s, ida, nullptr);
                k::qsa_block_topk(db, dt, nq, max_blocks, cap, s, idb, nullptr);
#if defined(__HIPCC__)
                ck(cudaMemset(dref, 0x55, n_sc * 4));
                const dim3 rgrid((unsigned) ((max_blocks + 7) / 8), (unsigned) nq);
                pre_pr_block_scores_kernel<<<rgrid, 256>>>(dp, dd, dq, dt, max_blocks, dref);
#endif
                ck(cudaDeviceSynchronize());
                std::vector<uint32_t> a(n_sc), b(n_sc);
                std::vector<int32_t> ia(n_id), ib(n_id);
                ck(cudaMemcpy(a.data(), da, n_sc * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(b.data(), db, n_sc * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(ia.data(), ida, n_id * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(ib.data(), idb, n_id * 4, cudaMemcpyDeviceToHost));
#if defined(__HIPCC__)
                std::vector<uint32_t> ref(n_sc);
                ck(cudaMemcpy(ref.data(), dref, n_sc * 4, cudaMemcpyDeviceToHost));
#endif
                for (int i = 0; i < nq; ++i) {
                    const int64_t n_bid = steps[i * k::kStepCount + k::kStepNBid];
                    for (int64_t j = 0; j <= n_bid; ++j) {
                        const size_t at = (size_t) (i * max_blocks + j);
                        if (a[at] != b[at]) {
                            std::fprintf(stderr, "FAIL rows vs multi ctx=%lld step_mode=%d nq=%d q=%d b=%lld\n",
                                         (long long) ctx, step_mode, nq, i, (long long) j);
                            return false;
                        }
#if defined(__HIPCC__)
                        if (a[at] != ref[at]) {
                            std::fprintf(stderr, "FAIL multi vs pre-PR ref ctx=%lld step_mode=%d nq=%d q=%d b=%lld\n",
                                         (long long) ctx, step_mode, nq, i, (long long) j);
                            return false;
                        }
#endif
                    }
                }
                bool same_i = true;
                for (int i = 0; i < nq && same_i; ++i) {
                    const int64_t width = steps[i * k::kStepCount + k::kStepWidth];
                    same_i = std::memcmp(ia.data() + (size_t) i * cap, ib.data() + (size_t) i * cap,
                                         (size_t) width * sizeof(int32_t)) == 0;
                }
                if (!same_i) {
                    std::fprintf(stderr, "FAIL rows topk ids ctx=%lld step_mode=%d nq=%d\n", (long long) ctx, step_mode, nq);
                    return false;
                }
            }
        }
#if defined(__HIPCC__)
        ck(cudaFree(dref));
#endif
        ck(cudaFree(dp)); ck(cudaFree(dd)); ck(cudaFree(dq)); ck(cudaFree(dt));
        ck(cudaFree(da)); ck(cudaFree(db)); ck(cudaFree(ida)); ck(cudaFree(idb));
    }
    std::puts("ok   qsa_block_scores_rows & top-k nq=1..8 (consecutive + spread n_bid + partial tails): 48 cases identical");
    return true;
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    const Case cases[] = {
        {1111, 9, 0, false},      {1111, 64, 1, false},     {8192, 65, 0, false},     {8195, 200, 2, false},
        {32768, 256, 0, false},   {32771, 1000, 1, false},  {131072, 64, 2, false},   {131075, 333, 0, false},
        {262144, 129, 1, false},  {40000, 2048, 0, true},   {140000, 1024, 0, true},  {200000, 1024, 0, true},
    };
    for (const auto c : cases) if (!check(c)) return 1;
    if (!check_rows_1_to_8()) return 1;
    std::puts("PASS: block scores of the thread kernel bitwise identical to block_scores_kernel");
    return 0;
}
