// cann/tests/rope_parity.cpp - rotary embedding on the NPU against the formula llama.cpp applies.
//
// qwen35moe uses IMRoPE, but for text every position id is the same, so the three sections
// collapse to one theta base and the op is plain NeoX-paired RoPE over the first n_rot channels
// of each head: pairs (i, i + n_rot/2), theta_i = pos * freq_base^(-2i/n_rot).
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

namespace {

constexpr int64_t kHeadDim = 256;
constexpr int64_t kRot = 64;       // qwen35moe.rope.dimension_count
constexpr float kFreqBase = 1e7f;  // qwen35moe.rope.freq_base

std::vector<float> round_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    std::vector<float> out(v.size());
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void upload(const std::vector<float> & v, Buffer & b) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), h.size());
    b.copy_from_host(h.data(), h.size() * 2);
}

std::vector<float> to_host(Stream & stream, const Buffer & b, int64_t n) {
    std::vector<uint16_t> h((size_t) n);
    b.copy_to_host(h.data(), h.size() * 2);
    stream.synchronize();
    std::vector<float> out((size_t) n);
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::vector<int64_t> pos = { 0, 1, 2, 7, 129 };
    const int64_t T = (int64_t) pos.size();
    const int64_t half = kRot / 2;

    std::mt19937 rng(41);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> x((size_t) T * kHeadDim);
    for (auto & v : x) {
        v = dist(rng);
    }
    x = round_f16(x);

    // cos/sin tables, one row per token
    std::vector<float> cos_tab((size_t) T * half), sin_tab((size_t) T * half);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < half; ++i) {
            const double theta = (double) pos[(size_t) t] * std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
            cos_tab[(size_t) (t * half + i)] = (float) std::cos(theta);
            sin_tab[(size_t) (t * half + i)] = (float) std::sin(theta);
        }
    }

    // host reference
    std::vector<float> ref(x.size());
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < half; ++i) {
            const float c = cos_tab[(size_t) (t * half + i)];
            const float s = sin_tab[(size_t) (t * half + i)];
            const float x0 = x[(size_t) (t * kHeadDim + i)];
            const float x1 = x[(size_t) (t * kHeadDim + i + half)];
            ref[(size_t) (t * kHeadDim + i)] = x0 * c - x1 * s;
            ref[(size_t) (t * kHeadDim + i + half)] = x0 * s + x1 * c;
        }
        for (int64_t i = kRot; i < kHeadDim; ++i) {
            ref[(size_t) (t * kHeadDim + i)] = x[(size_t) (t * kHeadDim + i)];
        }
    }

    Device device(0);
    Stream stream(device);
    Buffer bx(x.size() * 2), bc(cos_tab.size() * 2), bs(sin_tab.size() * 2), bo(x.size() * 2);
    upload(x, bx);
    upload(cos_tab, bc);
    upload(sin_tab, bs);

    Tensor tx{ bx.data(), DType::F16, kHeadDim, T };
    Tensor tc{ bc.data(), DType::F16, half, T };
    Tensor ts{ bs.data(), DType::F16, half, T };
    Tensor to{ bo.data(), DType::F16, kHeadDim, T };
    rope(stream, tx, tc, ts, kRot, to);

    const std::vector<float> got = to_host(stream, bo, (int64_t) x.size());

    double max_diff = 0.0, num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        max_diff = std::max(max_diff, std::fabs(d));
        num += d * d;
        den += (double) ref[i] * ref[i];
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;

    printf("rope : head_dim %lld, n_rot %lld, freq_base %.0e, %lld tokens (pos", (long long) kHeadDim,
           (long long) kRot, (double) kFreqBase, (long long) T);
    for (auto p : pos) {
        printf(" %lld", (long long) p);
    }
    printf(")\n\n");
    printf("%-12s %12s %12s  %s\n", "check", "max|diff|", "rel L2", "result");
    printf("%-12s %12.6f %12.6f  %s\n", "rotated", max_diff, rel, (max_diff <= 0.01 && rel <= 0.01) ? "PASS" : "FAIL");

    // the tail must be untouched
    double tail = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = kRot; i < kHeadDim; ++i) {
            tail = std::max(tail, std::fabs((double) got[(size_t) (t * kHeadDim + i)] - x[(size_t) (t * kHeadDim + i)]));
        }
    }
    printf("%-12s %12.6f %12.6f  %s\n", "tail copy", tail, 0.0, tail <= 0.01 ? "PASS" : "FAIL");

    shutdown();
    if (max_diff > 0.01 || rel > 0.01 || tail > 0.01) {
        printf("\nrope differs from the reference\n");
        return 1;
    }
    printf("\nrope matches the reference on the NPU\n");
    return 0;
}
