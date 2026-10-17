#include "strata/kernels/dflash2.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void checked(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { checked(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) { checked(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    std::vector<T> read(size_t n) { std::vector<T> v(n); checked(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
void near(float got, double want) {
    if (!std::isfinite(got) || std::abs(got - want) > 3e-5 * (1 + std::abs(want)))
        throw std::runtime_error("GPU/reference mismatch: " + std::to_string(got) + " vs " + std::to_string(want));
}
void convolve() {
    using namespace strata::kernels;
    const int hidden = 32;
    for (int width : {1, 2, 7, 8}) for (int taps : {1, 2, 5, 8}) for (int group : {1, 4, 16}) {
        const int rows = 2 * width, groups = hidden / group;
        std::vector<float> x(rows * hidden), dynamic(rows * 2 * taps * groups), base(2 * taps * hidden);
        for (size_t i = 0; i < x.size(); ++i) x[i] = ((int)(i % 31) - 15) * 0.13f;
        // The second block's radically different first row reveals cross-block leakage.
        for (int c = 0; c < hidden; ++c) x[width * hidden + c] = 10 + c;
        for (size_t i = 0; i < dynamic.size(); ++i) dynamic[i] = ((int)(i % 17) - 8) * 0.03f;
        for (size_t i = 0; i < base.size(); ++i) base[i] = ((int)(i % 13) - 6) * 0.02f;
        Device<float> dx(x), dd(dynamic), db(base), result(x.size());
        for (int side = 0; side < 2; ++side) {
            dflash2_convolve(dx.p, dd.p, db.p, result.p, rows, hidden, taps, group, side, width, nullptr);
            checked(cudaGetLastError());
            const auto got = result.read(x.size());
            for (int row = 0; row < rows; ++row) for (int c = 0; c < hidden; ++c) {
                double expected = 0;
                for (int offset = 0; offset < taps; ++offset) {
                    if (row % width < offset) continue;
                    const double weight = (double) base[(side * taps + offset) * hidden + c] +
                        dynamic[((row * 2 + side) * taps + offset) * groups + c / group];
                    expected += weight * x[(row - offset) * hidden + c];
                }
                near(got[row * hidden + c], expected);
            }
        }
    }
}
void selector() {
    using namespace strata::kernels;
    const int vocab = 11, rank = 3;
    int reranked = 0;
    for (int rows : {2, 3, 8}) for (int top_k : {1, 3, 11}) {
        std::vector<float> logits(rows * vocab), hidden(rows * rank), prev(vocab * rank), next(prev.size());
        for (size_t i = 0; i < logits.size(); ++i) logits[i] = ((int)(i % 7) - 3) * 0.05f + (i % vocab) * 0.00037f;
        std::fill(logits.begin(), logits.begin() + vocab, 0.1f); // top-k tie test
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = ((int)(i % 5) - 2) * 0.21f;
        for (size_t i = 0; i < prev.size(); ++i) { prev[i] = ((int)(i % 13) - 6) * 0.23f; next[i] = ((int)(i % 17) - 8) * 0.31f; }
        Device<float> dl(logits), dh(hidden), dp(prev), dn(next), du(rows * top_k);
        Device<int32_t> anchor(std::vector<int32_t>{7}), ids(rows * top_k), output(rows - 1);
        Device<float> confidence(rows - 1);
        dflash2_topk(dl.p, ids.p, du.p, rows, vocab, top_k, nullptr);
        checked(cudaGetLastError());
        auto got_ids = ids.read(rows * top_k);
        auto got_unary = du.read(rows * top_k);
        std::vector<int> order(vocab);
        for (int row = 0; row < rows; ++row) {
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[row * vocab + a] > logits[row * vocab + b]; });
            for (int k = 0; k < top_k; ++k) {
                if (got_ids[row * top_k + k] != order[k]) throw std::runtime_error("top-k IDs/tie order differ");
                near(got_unary[row * top_k + k], logits[row * vocab + order[k]]);
            }
        }
        for (bool probabilities : {false, true}) {
            dflash2_select(ids.p, du.p, dh.p, dp.p, dn.p, anchor.p, output.p,
                           probabilities ? confidence.p : nullptr, rows, top_k, rank, nullptr);
            checked(cudaGetLastError());
            auto got = output.read(rows - 1);
            auto prob = probabilities ? confidence.read(rows - 1) : std::vector<float>{};
            int predecessor = 7;
            for (int row = 1; row < rows; ++row) {
                std::vector<double> scores(top_k);
                for (int k = 0; k < top_k; ++k) {
                    int token = got_ids[row * top_k + k];
                    double interaction = 0;
                    for (int j = 0; j < rank; ++j) interaction += (double) prev[predecessor * rank + j] * hidden[row * rank + j] * next[token * rank + j];
                    scores[k] = logits[row * vocab + token] + interaction;
                }
                const int best = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
                predecessor = got_ids[row * top_k + best];
                if (got[row - 1] != predecessor) {
                    std::fprintf(stderr, "rows=%d topk=%d row=%d got=%d want=%d\n", rows, top_k, row, got[row - 1], predecessor);
                    for (int k = 0; k < top_k; ++k) std::fprintf(stderr, " candidate=%d score=%.12g\n", got_ids[row * top_k + k], scores[k]);
                    throw std::runtime_error("selector path differs");
                }
                reranked += best != 0;
                if (probabilities) {
                    double sum = 0;
                    for (double score : scores) sum += std::exp(score - scores[best]);
                    near(prob[row - 1], 1.0 / sum);
                }
            }
        }
        // Invalid logits must propagate failure, including every subsequent path position.
        checked(cudaMemset(dl.p, 0xff, logits.size() * sizeof(float))); // NaN
        dflash2_topk(dl.p, ids.p, du.p, rows, vocab, top_k, nullptr);
        dflash2_select(ids.p, du.p, dh.p, dp.p, dn.p, anchor.p, output.p, confidence.p, rows, top_k, rank, nullptr);
        for (auto id : output.read(rows - 1)) if (id != -1) throw std::runtime_error("nonfinite selector produced a token");
    }
    if (!reranked) throw std::runtime_error("fixture did not exercise selector reranking");
}
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try { convolve(); selector(); checked(cudaDeviceSynchronize()); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
    std::puts("PASS: DFlash2 convolution, block isolation, top-k, conditioned path, confidence and nonfinite refusal");
    return 0;
}
