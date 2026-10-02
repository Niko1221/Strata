#include <cuda_runtime.h>
#include "strata/kernels/qsa_select.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;
void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}
template <typename T> T* upload(const std::vector<T>& source) {
    T* p = nullptr; check(cudaMalloc(&p, source.size() * sizeof(T)));
    check(cudaMemcpy(p, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
int main() {
    constexpr int blocks = 131072 / 4 + 2;
    auto s = k::qsa_real_shapes();
    std::mt19937 random(73);
    std::uniform_real_distribution<float> distribution(-2.f, 2.f);
    std::vector<float> keys(blocks * 128), dead(128), queries(257 * 512);
    for (auto& f : keys) f = distribution(random);
    for (auto& f : dead) f = distribution(random);
    for (auto& f : queries) f = distribution(random);
    keys[0] = std::numeric_limits<float>::quiet_NaN();
    float *dk = upload(keys), *dd = upload(dead), *dq = upload(queries), *a = nullptr, *b = nullptr;
    check(cudaMalloc(&a, 257ull * blocks * sizeof(float)));
    check(cudaMalloc(&b, 257ull * blocks * sizeof(float)));
    cudaStream_t stream = nullptr; check(cudaStreamCreate(&stream));
    for (int context : {1024, 32768, 131072}) for (int n : {1, 7, 8, 9, 17, 257}) {
        std::vector<int32_t> steps(n * k::kStepCount);
        for (int i = 0; i < n; ++i) {
            auto* st = steps.data() + i * k::kStepCount;
            const int cells = context - n + i + 1;
            st[k::kStepPos] = cells - 1; st[k::kStepNKv] = cells; st[k::kStepNBid] = cells / 4;
            st[k::kStepWidth] = (int32_t) k::qsa_selection_width(cells, s);
        }
        auto* ds = upload(steps);
        const int active = context / 4 + 1;
        auto ref = [&] { k::qsa_block_scores_ref(dk, dd, dq, ds, n, blocks, s, a, stream, active); };
        auto batch = [&] { k::qsa_block_scores(dk, dd, dq, ds, n, blocks, s, b, stream, active); };
        ref(); batch(); check(cudaStreamSynchronize(stream));
        cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); batch();
        check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&exec, graph, 0));
        for (int replay = 0; replay < 3; ++replay) check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        std::vector<float> expected(n * blocks), got(n * blocks);
        check(cudaMemcpy(expected.data(), a, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(got.data(), b, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            const size_t count = (size_t) steps[i * k::kStepCount + k::kStepNBid] + 1;
            if (std::memcmp(expected.data() + i * blocks, got.data() + i * blocks, count * sizeof(float))) {
                std::fprintf(stderr, "score mismatch context=%d queries=%d row=%d\n", context, n, i); return 2;
            }
        }
        cudaEvent_t start = nullptr, end = nullptr; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        auto timed = [&](auto run) {
            check(cudaEventRecord(start, stream));
            for (int j = 0; j < 10; ++j) run();
            check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
            float ms = 0; check(cudaEventElapsedTime(&ms, start, end)); return ms / 10;
        };
        const float reference_ms = timed(ref), batch_ms = timed(batch);
        std::printf("context=%d queries=%d reference_ms=%.5f batch_ms=%.5f bitwise=PASS\n",
                    context, n, reference_ms, batch_ms);
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
        check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); check(cudaFree(ds));
    }
    check(cudaStreamDestroy(stream));
    check(cudaFree(dk)); check(cudaFree(dd)); check(cudaFree(dq)); check(cudaFree(a)); check(cudaFree(b));
}
