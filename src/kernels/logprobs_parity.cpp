#include "strata/kernels/logprobs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using strata::kernels::LogprobItem;

namespace {

void cuda_ok(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(rc));
        std::exit(1);
    }
}

struct Case {
    int vocab;
    int rows;
    int top_k;
    const char* label;
};

float value_for(int row, int id, int vocab) {
    if (row == 1) return 1.0e20f;  // large common offset catches LSE cancellation
    if (row == 2) {
        if (id == 0 || id == vocab / 2 || id == vocab - 1) return 1.0e20f;
        if (id % 5 == 0) return -1.0e20f;
        return (float) ((id * 13) % 17) - 8.0f;
    }
    // Deterministic ties across and within 4096-logit partitions, plus a wide score range.
    if (id == 1 || id == 4095 || id == 4096) return 100.0f;
    return (float) ((id * 37 + row * 11) % 101) / 10.0f;
}

struct Expected {
    int id;
    float logprob;
};

std::vector<Expected> reference(const std::vector<float>& logits, int vocab, int row,
                                int chosen, int top_k) {
    const float* values = logits.data() + (size_t) row * vocab;
    const double mx = (double) *std::max_element(values, values + vocab);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) sum += std::exp((double) values[i] - mx);
    const double log_sum = std::log(sum);
    std::vector<int> order(vocab);
    for (int i = 0; i < vocab; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
        if (values[a] != values[b]) return values[a] > values[b];
        return a < b;
    });
    std::vector<Expected> result;
    result.reserve((size_t) top_k + 1);
    result.push_back({chosen, (float) (((double) values[chosen] - mx) - log_sum)});
    for (int i = 0; i < top_k; ++i)
        result.push_back({order[i], (float) (((double) values[order[i]] - mx) - log_sum)});
    return result;
}

bool run_case(const Case& c) {
    const size_t count = (size_t) c.rows * c.vocab;
    std::vector<float> h_logits(count);
    std::vector<int32_t> h_chosen(c.rows);
    for (int r = 0; r < c.rows; ++r) {
        for (int v = 0; v < c.vocab; ++v) h_logits[(size_t) r * c.vocab + v] = value_for(r, v, c.vocab);
        h_chosen[r] = r == 1 ? c.vocab - 1 : (r * 31 + 7) % c.vocab;
    }

    const size_t scratch_bytes = strata::kernels::logprobs_scratch_bytes(c.rows, c.vocab, c.top_k);
    if (!scratch_bytes) {
        std::fprintf(stderr, "%s: valid shape returned zero scratch size\n", c.label);
        return false;
    }
    float* d_logits = nullptr;
    int32_t* d_chosen = nullptr;
    void* d_scratch = nullptr;
    LogprobItem* d_out = nullptr;
    const size_t out_count = (size_t) c.rows * (c.top_k + 1);
    cuda_ok(cudaMalloc(&d_logits, count * sizeof(float)), "cudaMalloc logits");
    cuda_ok(cudaMalloc(&d_chosen, c.rows * sizeof(int32_t)), "cudaMalloc chosen");
    cuda_ok(cudaMalloc(&d_scratch, scratch_bytes), "cudaMalloc scratch");
    cuda_ok(cudaMalloc(&d_out, out_count * sizeof(LogprobItem)), "cudaMalloc output");
    cuda_ok(cudaMemcpy(d_logits, h_logits.data(), count * sizeof(float), cudaMemcpyHostToDevice), "copy logits");
    cuda_ok(cudaMemcpy(d_chosen, h_chosen.data(), c.rows * sizeof(int32_t), cudaMemcpyHostToDevice), "copy chosen");

    bool ok = strata::kernels::logprobs_extract(d_logits, c.rows, c.vocab, d_chosen, c.top_k,
                                                d_scratch, d_out, nullptr);
    if (!ok) {
        std::fprintf(stderr, "%s: extraction refused a valid input\n", c.label);
        ok = false;
    } else {
        std::vector<LogprobItem> got(out_count);
        cuda_ok(cudaMemcpy(got.data(), d_out, out_count * sizeof(LogprobItem), cudaMemcpyDeviceToHost), "copy output");
        for (int r = 0; r < c.rows && ok; ++r) {
            const auto want = reference(h_logits, c.vocab, r, h_chosen[r], c.top_k);
            for (int j = 0; j <= c.top_k; ++j) {
                const auto& actual = got[(size_t) r * (c.top_k + 1) + j];
                const auto& expected = want[(size_t) j];
                if (actual.id != expected.id || !std::isfinite(actual.logprob) ||
                    std::abs(actual.logprob - expected.logprob) > 3.0e-5f) {
                    std::fprintf(stderr, "%s row=%d item=%d got=(%d, %.9g) expected=(%d, %.9g)\n",
                                 c.label, r, j, actual.id, actual.logprob, expected.id, expected.logprob);
                    ok = false;
                    break;
                }
            }
        }
        if (ok && c.vocab > 100000) {
            const float expected = -std::log((float) c.vocab);
            for (int j = 0; j <= c.top_k; ++j) {
                const auto& item = got[(size_t) (c.top_k + 1) + j];
                if (std::abs(item.logprob - expected) > 3.0e-5f) {
                    std::fprintf(stderr, "%s all-equal analytic score: got %.9g expected %.9g\n",
                                 c.label, item.logprob, expected);
                    ok = false;
                    break;
                }
            }
        }
    }

    cuda_ok(cudaFree(d_out), "free output");
    cuda_ok(cudaFree(d_scratch), "free scratch");
    cuda_ok(cudaFree(d_chosen), "free chosen");
    cuda_ok(cudaFree(d_logits), "free logits");
    return ok;
}

bool invalid_shapes_refused() {
    return strata::kernels::logprobs_scratch_bytes(0, 1, 0) == 0 &&
           strata::kernels::logprobs_scratch_bytes(1, 0, 0) == 0 &&
           strata::kernels::logprobs_scratch_bytes(1, 262145, 0) == 0 &&
           strata::kernels::logprobs_scratch_bytes(1, 16, -1) == 0 &&
           strata::kernels::logprobs_scratch_bytes(1, 16, 21) == 0 &&
           strata::kernels::logprobs_scratch_bytes(1, 4, 5) == 0;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "logprobs parity requires a CUDA device\n");
        return 77;
    }
    const Case cases[] = {
        {1, 2, 0, "single-token vocabulary K=0"},
        {7, 2, 7, "K equals vocabulary"},
        {17, 3, 0, "K=0"},
        {4097, 2, 1, "partial-block K=1"},
        {8193, 4, 20, "multiple-block rows (accepted MTP rows)"},
        {248320, 4, 20, "full 61-block vocabulary"},
    };
    for (const auto& c : cases) {
        if (!run_case(c)) return 1;
    }
    if (!invalid_shapes_refused()) {
        std::fprintf(stderr, "invalid dimensions were accepted\n");
        return 1;
    }
    std::puts("logprobs parity passed");
    return 0;
}
