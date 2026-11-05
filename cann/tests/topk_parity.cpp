// cann/tests/topk_parity.cpp - the device sampler against a host reference.
//
// Sampling needs two things from the logits: the k largest entries and the log-partition. Both are
// now computed on the device (aclnnTopk and aclnnLogSumExp), so only k values and one scalar come
// back. This checks them against a straightforward host implementation over the same numbers.
#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

namespace {

void upload_f32(const std::vector<float> & v, Buffer & b) {
    b.copy_from_host(v.data(), v.size() * 4);
}

std::vector<float> to_host_f32(Stream & stream, const Buffer & b, int64_t n) {
    std::vector<float> out((size_t) n);
    b.copy_to_host(out.data(), out.size() * 4);
    stream.synchronize();
    return out;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const int64_t vocab = 248320; // the model's real vocabulary size
    const int64_t k = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 4.0f);
    std::vector<float> logits((size_t) vocab);
    for (auto & v : logits) v = dist(rng);

    // host reference
    std::vector<int> order((size_t) vocab);
    for (int64_t i = 0; i < vocab; ++i) order[(size_t) i] = (int) i;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    double maxv = (double) logits[(size_t) order[0]];
    double sum = 0.0;
    for (int64_t i = 0; i < vocab; ++i) sum += std::exp((double) logits[(size_t) i] - maxv);
    const double log_z = maxv + std::log(sum);

    Device device(0);
    Stream stream(device);
    Buffer bx((size_t) vocab * 4), bv((size_t) k * 4), bi((size_t) k * 8), bz(4);
    upload_f32(logits, bx);

    Tensor tx{ bx.data(), DType::F32, vocab, 1 };
    Tensor tv{ bv.data(), DType::F32, k, 1 };
    Tensor ti{ bi.data(), DType::F32, k, 1 }; // only the pointer matters; the descriptor is int64
    Tensor tz{ bz.data(), DType::F32, 1, 1 };
    topk(stream, tx, k, 1, tv, ti);
    logsumexp(stream, tx, 1, tz);

    const std::vector<float> values = to_host_f32(stream, bv, k);
    const std::vector<float> lz = to_host_f32(stream, bz, 1);
    std::vector<int64_t> idx((size_t) k);
    bi.copy_to_host(idx.data(), idx.size() * 8);
    stream.synchronize();

    printf("device topk: vocab=%lld k=%lld\n\n", (long long) vocab, (long long) k);
    printf("%-10s %12s %12s  %s\n", "rank", "device", "host", "ids");
    bool ok = true;
    for (int64_t i = 0; i < k; ++i) {
        const float want = logits[(size_t) order[(size_t) i]];
        const bool v_ok = std::fabs(values[(size_t) i] - want) < 1e-3f;
        const bool i_ok = idx[(size_t) i] == order[(size_t) i];
        ok = ok && v_ok && i_ok;
        printf("%-10lld %12.6f %12.6f  %lld%s%s\n", (long long) i, (double) values[(size_t) i], (double) want,
               (long long) idx[(size_t) i], v_ok ? "" : " value", i_ok ? "" : " id");
    }
    const double dz = std::fabs((double) lz[0] - log_z);
    printf("\nlog-partition  device %.6f  host %.6f  diff %.2e  %s\n", (double) lz[0], log_z, dz,
           dz < 1e-3 ? "PASS" : "FAIL");
    ok = ok && dz < 1e-3;

    shutdown();
    if (!ok) {
        printf("\nthe device sampler does not match the host reference\n");
        return 1;
    }
    printf("\ndevice topk and logsumexp match the host\n");
    return 0;
}
