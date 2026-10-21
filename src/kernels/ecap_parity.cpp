// ecap_parity - the verify window's expert-cap redirect against its contract.
//
// `verify_ecap` (STRATA_VERIFY_ECAP) rewrites the window's routing in place before resident_plan: token 0
// keeps all k picks, the speculative tail keeps the top (cap - k) entries by (weight desc, index asc), and
// every dropped entry is redirected round-robin onto a kept expert with weight 0.  Speculative decoding's
// exactness for the committed position and the plan's residency requirement both hang on the kernel keeping
// its promises, so this checks them on random data shaped like the router (256 experts, top-10, windows of
// 2..6 tokens):
//   * cap >= the entry count (or n <= k, or cap <= k): a no-op - ids and weights bitwise unchanged
//   * token 0's entries are never touched
//   * every kept entry keeps its id and weight; every dropped entry's weight is exactly 0 and its id is one
//     of the kept set's (so the plan never meets an expert the uncapped window would not have named)
//   * the same input gives the same output (the fold is deterministic)
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
}  // namespace

int main() {
    const int E = 256, K = 10;
    int bad = 0;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> id_dist(0, E - 1);
    std::uniform_real_distribution<float> w_dist(0.0f, 1.0f);
    for (int T : {2, 3, 4, 6}) {
        const int n = T * K;
        std::vector<int32_t> ids(n), ids0;
        std::vector<float> w(n), w0;
        // distinct ids per token, like the router's top-k produces
        for (int t = 0; t < T; ++t) {
            std::vector<int> pool(E);
            for (int i = 0; i < E; ++i) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            std::sort(pool.begin(), pool.begin() + K);
            for (int j = 0; j < K; ++j) ids[t * K + j] = pool[j];
        }
        for (auto& v : w) v = w_dist(rng);
        ids0 = ids; w0 = w;

        int32_t* d_ids; float* d_w;
        ck(cudaMalloc(&d_ids, n * 4), "ids"); ck(cudaMalloc(&d_w, n * 4), "w");
        auto up = [&] { ck(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice), "up ids");
                        ck(cudaMemcpy(d_w, w.data(), n * 4, cudaMemcpyHostToDevice), "up w"); };
        auto down = [&] { ck(cudaMemcpy(ids.data(), d_ids, n * 4, cudaMemcpyDeviceToHost), "down ids");
                          ck(cudaMemcpy(w.data(), d_w, n * 4, cudaMemcpyDeviceToHost), "down w"); };

        // a cap that keeps everything is a no-op, bitwise
        up();
        strata::kernels::verify_ecap(d_ids, d_w, n, K, n + 7, nullptr);
        ck(cudaDeviceSynchronize(), "sync noop");
        down();
        const bool noop = std::memcmp(ids.data(), ids0.data(), n * 4) == 0 &&
                          std::memcmp(w.data(), w0.data(), n * 4) == 0;
        std::printf("  cap>=n      T %d  %s\n", T, noop ? "no-op" : "*** CHANGED ***");
        if (!noop) ++bad;

        for (int cap : {K + 1, 16, 20, 26}) {
            if (cap >= n) continue;
            // the expected kept set: token 0's k, then the top (cap-k) tail entries by (w desc, index asc)
            std::vector<int> tail(n - K);
            for (int j = K; j < n; ++j) tail[j - K] = j;
            std::stable_sort(tail.begin(), tail.end(), [&](int a, int b) { return w0[a] > w0[b]; });
            std::vector<char> kept(n, 0);
            for (int j = 0; j < K; ++j) kept[j] = 1;
            for (int i = 0; i < cap - K && i < (int) tail.size(); ++i) kept[tail[i]] = 1;

            up();
            strata::kernels::verify_ecap(d_ids, d_w, n, K, cap, nullptr);
            ck(cudaDeviceSynchronize(), "sync cap");
            down();
            std::vector<int32_t> ids_a = ids; std::vector<float> w_a = w;
            up();                                                               // the original again
            strata::kernels::verify_ecap(d_ids, d_w, n, K, cap, nullptr);       // determinism: same input twice
            ck(cudaDeviceSynchronize(), "sync again");
            down();
            std::vector<int32_t> ids_b = ids; std::vector<float> w_b = w;
            const bool det = ids_a == ids_b && w_a == w_b;

            bool ok = det;
            std::vector<int32_t> kept_ids;
            for (int j = 0; j < n; ++j) if (kept[j]) kept_ids.push_back(ids0[j]);
            for (int j = 0; j < n && ok; ++j) {
                if (j < K) {                                     // token 0: untouched
                    ok = ids[j] == ids0[j] && w[j] == w0[j];
                } else if (kept[j]) {                            // kept: id and weight unchanged
                    ok = ids[j] == ids0[j] && w[j] == w0[j];
                } else {                                         // dropped: weight exactly 0, id in the kept set
                    ok = w[j] == 0.0f && std::find(kept_ids.begin(), kept_ids.end(), ids[j]) != kept_ids.end();
                }
            }
            int dropped = 0; for (int j = K; j < n; ++j) if (!kept[j]) ++dropped;
            std::printf("  cap %2d      T %d  dropped %2d  %s%s\n", cap, T, dropped, ok ? "ok" : "*** VIOLATION ***",
                        det ? "" : " (+ nondeterministic)");
            if (!ok) ++bad;
            ids = ids0; w = w0;
        }
        cudaFree(d_ids); cudaFree(d_w);
    }
    if (bad) { std::printf("ecap_parity FAILED\n"); return 1; }
    std::printf("ecap_parity OK\n");
    return 0;
}
