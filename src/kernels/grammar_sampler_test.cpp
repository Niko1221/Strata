#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using strata::kernels::SamplerParams;
using strata::kernels::TokenMask;
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
static void check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }

int main() {
    try {
        constexpr int nv = 248320, rows = 3, history_len = 4, words = (nv + 31) / 32;
        const float inf = std::numeric_limits<float>::infinity();
        std::vector<float> logits(nv * rows, 20.0f), reference(logits.size());
        std::vector<int32_t> masks(rows * words), histories(rows * history_len, 1001);
        for (int r = 0; r < rows; ++r) {
            for (int id : {1001, 1002, 1003}) masks[r * words + id / 32] |= (int32_t) (1u << (id % 32));
            logits[r * nv + 1001] = 3;
            logits[r * nv + 1002] = 2;
            logits[r * nv + 1003] = 1;
        }
        for (int r = 0; r < rows; ++r)
            for (int id = 0; id < nv; ++id)
                reference[r * nv + id] = id >= 1001 && id <= 1003 ? logits[r * nv + id] : -inf;
        float *d_logits = nullptr, *d_reference = nullptr, *scratch = nullptr;
        int32_t *d_masks = nullptr, *d_history = nullptr, *out = nullptr, *expected = nullptr, *status = nullptr;
        check(cudaMalloc(&d_logits, logits.size() * sizeof(float)));
        check(cudaMalloc(&d_reference, reference.size() * sizeof(float)));
        check(cudaMalloc(&scratch, logits.size() * sizeof(float)));
        check(cudaMalloc(&d_masks, masks.size() * sizeof(int32_t)));
        check(cudaMalloc(&d_history, histories.size() * sizeof(int32_t)));
        check(cudaMalloc(&out, rows * sizeof(int32_t)));
        check(cudaMalloc(&expected, rows * sizeof(int32_t)));
        check(cudaMalloc(&status, rows * sizeof(int32_t)));
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        auto upload = [&] {
            check(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(d_masks, masks.data(), masks.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        };
        upload();
        check(cudaMemcpy(d_reference, reference.data(), reference.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_history, histories.data(), histories.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        TokenMask mask{d_masks, scratch, status};
        int cases = 0;
        for (bool capture : {false, true}) for (bool greedy : {false, true})
            for (bool penalty : {false, true}) for (int k : {1, 3, 20}) {
                SamplerParams p;
                p.greedy = greedy; p.temperature = greedy ? 0.0f : 0.8f;
                p.top_k = k; p.top_p = 0.8f; p.min_p = 0.1f; p.seed = 8429; p.counter = 25;
                p.penalty_last_n = penalty ? history_len : 0;
                p.penalty_repeat = 1.3f; p.penalty_freq = 0.6f; p.penalty_present = 0.2f;
                auto run = [&] {
                    strata::kernels::sample_tokens(d_logits, rows, nv, d_history, history_len, p, out, stream, &mask);
                    strata::kernels::sample_tokens(d_reference, rows, nv, d_history, history_len, p, expected, stream);
                };
                if (capture) {
                    cudaGraph_t graph; cudaGraphExec_t exec;
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); run();
                    check(cudaStreamEndCapture(stream, &graph));
                    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                    check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
                    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
                } else { run(); check(cudaStreamSynchronize(stream)); }
                std::vector<int32_t> actual(rows), wanted(rows), errors(rows);
                check(cudaMemcpy(actual.data(), out, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(wanted.data(), expected, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(errors.data(), status, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
                REQUIRE(actual == wanted);
                for (int r = 0; r < rows; ++r) {
                    REQUIRE(errors[r] == 0 && actual[r] >= 1001 && actual[r] <= 1003);
                    if (k == 1 || greedy) REQUIRE(actual[r] == (penalty ? 1002 : 1001));
                }
                ++cases;
            }
        // Empty grammar mask, hard -inf on every legal candidate, NaN/+inf and
        // overflow after penalties must fail. These kernels also run captured.
        for (bool capture : {false, true}) for (bool greedy : {false, true}) for (int kind = 0; kind < 5; ++kind) {
            std::fill(masks.begin(), masks.end(), 0);
            std::fill(logits.begin(), logits.end(), 50.0f);
            for (int r = 0; r < rows; ++r) {
                if (kind) masks[r * words + 1001 / 32] = (int32_t) (1u << (1001 % 32));
                logits[r * nv + 1001] = kind == 1 ? -inf : kind == 2 ? std::nanf("") : kind == 3 ? inf : -1e38f;
            }
            upload();
            SamplerParams p; p.top_k = 1; p.greedy = greedy;
            if (kind == 4) { p.penalty_last_n = history_len; p.penalty_repeat = 100; }
            auto run = [&] { strata::kernels::sample_tokens(d_logits, rows, nv, d_history, history_len, p, out, stream, &mask); };
            if (capture) {
                cudaGraph_t graph; cudaGraphExec_t exec;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); run();
                check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
                check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
            } else { run(); check(cudaStreamSynchronize(stream)); }
            std::vector<int32_t> actual(rows), errors(rows);
            check(cudaMemcpy(actual.data(), out, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(errors.data(), status, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int r = 0; r < rows; ++r) REQUIRE(actual[r] == -1 && errors[r] == (kind < 2 ? 1 : 2));
            ++cases;
        }
        // Reuse the same captured operation with changed device masks after
        // failures, including a high illegal NaN. No stale mask/status survives.
        SamplerParams p; p.greedy = true;
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        strata::kernels::sample_tokens(d_logits, rows, nv, nullptr, 0, p, out, stream, &mask);
        check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int id : {1001, 1002}) {
            std::fill(masks.begin(), masks.end(), 0);
            std::fill(logits.begin(), logits.end(), std::nanf(""));
            for (int r = 0; r < rows; ++r) {
                masks[r * words + id / 32] = (int32_t) (1u << (id % 32));
                logits[r * nv + id] = 0;
            }
            upload(); check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream));
            std::vector<int32_t> actual(rows);
            check(cudaMemcpy(actual.data(), out, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int selected : actual) REQUIRE(selected == id);
            ++cases;
        }
        check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
        for (void* p : {(void*) d_logits, (void*) d_reference, (void*) scratch, (void*) d_masks,
                         (void*) d_history, (void*) out, (void*) expected, (void*) status}) check(cudaFree(p));
        check(cudaStreamDestroy(stream));
        std::cout << "grammar sampler: " << cases << " captured/uncaptured cases passed; 248320 logits, 3 rows each\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "grammar sampler failed: " << error.what() << '\n'; return 1; }
}
