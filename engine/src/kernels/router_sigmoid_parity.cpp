// Compare sigmoid routing and expert weights against the CPU reference.
#include "strata/kernels/native_router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaError_t hipError_t
#else
#include <cuda_runtime.h>
#endif

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// llama.cpp's order, in double precision. `bias` may be empty.
void reference(const float* logits, const std::vector<float>& bias, int n_expert, int k, double scale,
               std::vector<int>& ids, std::vector<double>& w) {
    std::vector<double> probs((size_t) n_expert), sel((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        probs[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) logits[e]));
        sel[(size_t) e] = probs[(size_t) e] + (bias.empty() ? 0.0 : (double) bias[(size_t) e]);
    }
    std::vector<int> order((size_t) n_expert);
    std::iota(order.begin(), order.end(), 0);
    // Ties toward the lower index, which is the contract the other router kernels in this file document.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (sel[(size_t) a] != sel[(size_t) b]) return sel[(size_t) a] > sel[(size_t) b];
        return a < b;
    });
    ids.assign(order.begin(), order.begin() + k);
    double sum = 0.0;
    for (int r = 0; r < k; ++r) sum += probs[(size_t) ids[(size_t) r]];
    sum = std::max(sum, 6.103515625e-5);
    w.resize((size_t) k);
    for (int r = 0; r < k; ++r) w[(size_t) r] = probs[(size_t) ids[(size_t) r]] / sum * (scale != 0.0 ? scale : 1.0);
}

struct Dev {
    float* logits = nullptr;
    float* bias = nullptr;
    int32_t* ids = nullptr;
    float* w = nullptr;
    cudaStream_t stream{};
    ~Dev() {
        if (logits) cudaFree(logits);
        if (bias) cudaFree(bias);
        if (ids) cudaFree(ids);
        if (w) cudaFree(w);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Compare fused T1 publication to the reference-checked plain route. Use a
// non-identity slot map so confusing expert IDs and cache slots is observable.
void check_publication(Dev& d, int experts, float scale,
                       const std::vector<int32_t>& ids, const std::vector<float>& weights) {
    const int k = (int)ids.size();
    std::vector<int32_t> map(experts);
    for (int i = 0; i < experts; ++i) map[i] = experts - 1 - i;
    struct Mailbox {
        uint32_t request, fast, error, epoch[2];
        int32_t ids[16], slots[16], hit[16], miss[16];
    };
    Mailbox* device = nullptr;
    int32_t* slots = nullptr;
    cudaMalloc((void**)&device, sizeof(Mailbox));
    cudaMalloc((void**)&slots, experts * sizeof(int32_t));
    for (bool all_hit : {true, false}) {
        Mailbox m{};
        if (!all_hit) map[ids.front()] = -1;
        cudaMemcpy(device, &m, sizeof(m), cudaMemcpyHostToDevice);
        cudaMemcpy(slots, map.data(), experts * sizeof(int32_t), cudaMemcpyHostToDevice);
        strata::kernels::NativeRouterPublishPlan plan{&device->request, &device->fast, &device->error,
            device->ids, device->slots, slots, device->hit, device->miss, experts, experts, device->epoch};
        strata::kernels::native_router_sigmoid_publish(d.logits, d.bias, d.ids, d.w, scale, plan, d.stream);
        const bool completed = cudaStreamSynchronize(d.stream) == cudaSuccess;
        cudaMemcpy(&m, device, sizeof(m), cudaMemcpyDeviceToHost);
        std::vector<float> fused(k);
        cudaMemcpy(fused.data(), d.w, k * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = completed && fused == weights && m.request == 1 && m.fast == uint32_t(all_hit) &&
                  m.epoch[0] == m.request && m.epoch[1] == m.fast;
        for (int i = 0; i < k; ++i)
            ok &= m.ids[i] == ids[i] && (!all_hit || (m.slots[i] == map[ids[i]] &&
                                                       m.hit[i] == m.slots[i] && m.miss[i] == -1));
        check(ok, "fused router IDs, weights, slots and epoch, width=" + std::to_string(experts) +
                  (all_hit ? " all hits" : " with a miss"));
    }
    cudaFree(slots);
    cudaFree(device);
}

/// One case: n_tok rows of logits, an optional bias, a scale. Returns the max weight error seen.
double run_case(const std::string& name, const std::vector<float>& logits, const std::vector<float>& bias,
                int n_tok, int n_expert, int k, float scale, bool* ids_all_match = nullptr) {
    Dev d;
    if (cudaStreamCreate(&d.stream) != cudaSuccess) {
        std::printf("  %s: no stream\n", name.c_str());
        ++g_fail;
        return 0.0;
    }
    cudaMalloc((void**) &d.logits, logits.size() * sizeof(float));
    cudaMalloc((void**) &d.ids, (size_t) n_tok * k * sizeof(int32_t));
    cudaMalloc((void**) &d.w, (size_t) n_tok * k * sizeof(float));
    cudaMemcpy(d.logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (!bias.empty()) {
        cudaMalloc((void**) &d.bias, bias.size() * sizeof(float));
        cudaMemcpy(d.bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    strata::kernels::native_router_sigmoid(d.logits, d.bias, d.ids, d.w, n_expert, k, n_tok, scale, d.stream);
    if (cudaStreamSynchronize(d.stream) != cudaSuccess) {
        std::printf("  %s: the kernel did not complete\n", name.c_str());
        ++g_fail;
        return 0.0;
    }
    std::vector<int32_t> got_ids((size_t) n_tok * k);
    std::vector<float> got_w((size_t) n_tok * k);
    cudaMemcpy(got_ids.data(), d.ids, got_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_w.data(), d.w, got_w.size() * sizeof(float), cudaMemcpyDeviceToHost);

    if (n_tok == 1) check_publication(d, n_expert, scale, got_ids, got_w);

    double worst = 0.0;
    bool ids_ok = true;
    for (int t = 0; t < n_tok; ++t) {
        std::vector<int> r_ids;
        std::vector<double> r_w;
        reference(&logits[(size_t) t * n_expert], bias, n_expert, k, scale, r_ids, r_w);
        for (int r = 0; r < k; ++r) {
            if (got_ids[(size_t) t * k + (size_t) r] != r_ids[(size_t) r]) ids_ok = false;
            worst = std::max(worst, std::fabs((double) got_w[(size_t) t * k + (size_t) r] - r_w[(size_t) r]));
        }
    }
    if (ids_all_match) *ids_all_match = ids_ok;
    check(ids_ok, name + ": the selected experts match the reference");
    // float32 against a double reference: a few ulps of weights that sum to `scale`. Printed in scientific
    // notation - "0.000000" would hide whether the comparison is live at all.
    char worst_s[32];
    std::snprintf(worst_s, sizeof worst_s, "%.3g", worst);
    check(worst < 1e-6, name + ": weights within 1e-6 (worst " + worst_s + ")");
    // The normalisation is what makes the weights sum to the scale; checking it here means a kernel that
    // silently dropped the division would fail on more than a float comparison.
    double first_sum = 0.0;
    for (int r = 0; r < k; ++r) first_sum += (double) got_w[(size_t) r];
    const double want_sum = (scale != 0.f ? (double) scale : 1.0);
    char sum_s[64];
    std::snprintf(sum_s, sizeof sum_s, "%.6f vs %.6f", first_sum, want_sum);
    check(std::fabs(first_sum - want_sum) < 1e-4, name + ": the weights of token 0 sum to the scale (" + sum_s + ")");
    return worst;
}

}  // namespace

int main() {
    constexpr int NE = 512, K = 16;
    if (!strata::kernels::native_router_sigmoid_supported(NE, K)) {
        std::printf("512/16 is not supported by this build\n");
        return 1;
    }
    check(!strata::kernels::native_router_sigmoid_supported(512, 10),
          "512/10 is not claimed (it is the softmax path's shape, a different contract)");
    check(!strata::kernels::native_router_sigmoid_supported(15, 16), "top-k cannot exceed the expert count");

    std::mt19937 rng(20261106);
    std::normal_distribution<float> normal(0.f, 2.f);

    // Include a partial warp, a row wider than the register tile, and ties at the cutoff.
    for (int experts : {16, 31, 257, 384, 511, 513, 769}) {
        std::vector<float> logits(9 * experts), bias(experts);
        for (float& v : logits) v = normal(rng);
        for (float& v : bias) v = normal(rng) * 0.1f;
        run_case("width=" + std::to_string(experts), logits, bias, 9, experts, K, 3.998f);
        run_case("single width=" + std::to_string(experts),
                 std::vector<float>(logits.begin(), logits.begin() + experts), bias, 1, experts, K, 3.998f);
        std::fill(logits.begin(), logits.end(), 0.f);
        run_case("ties=" + std::to_string(experts), logits, {}, 9, experts, K, 1.f);
    }

    std::printf("no bias: sigmoid gating, scale 1\n");
    {
        const int n_tok = 7;
        std::vector<float> logits((size_t) n_tok * NE);
        for (float& v : logits) v = normal(rng);
        run_case("plain", logits, {}, n_tok, NE, K, 1.0f);
    }

    std::printf("hy4-flash's own scale, 3.998\n");
    {
        const int n_tok = 5;
        std::vector<float> logits((size_t) n_tok * NE);
        for (float& v : logits) v = normal(rng);
        run_case("scaled", logits, {}, n_tok, NE, K, 3.998f);
    }

    std::printf("a selection bias that REORDERS the picks - the weights must stay unbiased\n");
    {
        const int n_tok = 4;
        std::vector<float> logits((size_t) n_tok * NE);
        for (float& v : logits) v = normal(rng);
        // A bias big enough to move experts in and out of the top 16: sigmoid is in (0,1), so a bias of ~1
        // dominates the ordering wherever it is applied.
        std::vector<float> bias((size_t) NE, 0.f);
        for (int e = 0; e < NE; ++e) bias[(size_t) e] = (e % 7 == 0) ? 0.9f : (e % 11 == 0 ? -0.9f : 0.02f);

        bool ids_ok = false;
        run_case("biased", logits, bias, n_tok, NE, K, 3.998f, &ids_ok);

        // Verify that the negative control distinguishes selection bias from combine weights.
        std::vector<int> r_ids;
        std::vector<double> r_w;
        reference(logits.data(), bias, NE, K, 3.998, r_ids, r_w);
        // The same selection, but weights taken from the BIASED probabilities: the bug being guarded against.
        std::vector<double> buggy((size_t) K);
        double sum = 0.0;
        for (int r = 0; r < K; ++r) {
            const int e = r_ids[(size_t) r];
            sum += 1.0 / (1.0 + std::exp(-(double) logits[(size_t) e])) + (double) bias[(size_t) e];
        }
        sum = std::max(sum, 6.103515625e-5);
        double gap = 0.0;
        for (int r = 0; r < K; ++r) {
            const int e = r_ids[(size_t) r];
            buggy[(size_t) r] =
                (1.0 / (1.0 + std::exp(-(double) logits[(size_t) e])) + (double) bias[(size_t) e]) / sum * 3.998;
            gap = std::max(gap, std::fabs(buggy[(size_t) r] - r_w[(size_t) r]));
        }
        check(gap > 1e-3, "using the biased probability for the weight would differ here by " +
                              std::to_string(gap) + " - so the check above is not vacuous");
        check(ids_ok, "and the bias did reach the selection");
    }

    std::printf("a bias that only shifts everything equally must not change the picks\n");
    {
        const int n_tok = 3;
        std::vector<float> logits((size_t) n_tok * NE);
        for (float& v : logits) v = normal(rng);
        std::vector<float> flat((size_t) NE, 0.25f);
        std::vector<int> a, b;
        std::vector<double> wa, wb;
        reference(logits.data(), {}, NE, K, 1.0, a, wa);
        reference(logits.data(), flat, NE, K, 1.0, b, wb);
        check(a == b, "a constant bias leaves the ordering alone (reference self-check)");
        run_case("flat-bias", logits, flat, n_tok, NE, K, 1.0f);
    }

    std::printf("one token, and a block-boundary count (8 tokens per block)\n");
    {
        for (int n_tok : {1, 8, 9, 16, 17}) {
            std::vector<float> logits((size_t) n_tok * NE);
            for (float& v : logits) v = normal(rng);
            std::vector<float> bias((size_t) NE);
            for (int e = 0; e < NE; ++e) bias[(size_t) e] = normal(rng) * 0.1f;
            run_case("n_tok=" + std::to_string(n_tok), logits, bias, n_tok, NE, K, 3.998f);
        }
    }

    std::printf("saturated logits: sigmoid's tails, where many probabilities are equal\n");
    {
        const int n_tok = 2;
        std::vector<float> logits((size_t) n_tok * NE);
        // Mostly deep in the tails (sigmoid rounds to 0 or 1), with a handful in between: this is where ties are
        // common and the tie rule has to agree with the reference.
        for (size_t i = 0; i < logits.size(); ++i)
            logits[i] = (i % 5 == 0) ? 40.f : ((i % 5 == 1) ? -40.f : normal(rng) * 0.01f);
        run_case("saturated", logits, {}, n_tok, NE, K, 3.998f);
    }

    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
