// cann/tests/parity.cpp - can the CANN device layer reproduce the CPU oracle?
//
// Each op runs on the 910B2C with random fp16 inputs and is compared against a float CPU
// reference. fp16 storage means the comparison is against the values after rounding the
// inputs to fp16, which is what the engine would actually feed the device.
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

namespace {

struct Case {
    std::string name;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double tolerance = 0.0;
    bool passed = false;
};

std::vector<float> random_uniform(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) {
        x = dist(rng);
    }
    return v;
}

// round-trip through fp16 so the oracle sees exactly what the device sees
std::vector<float> quantize_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    std::vector<float> out(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    f16_to_f32(h.data(), out.data(), v.size());
    return out;
}

Case compare(const std::string & name, const std::vector<float> & got, const std::vector<float> & ref,
             double tolerance) {
    Case c;
    c.name = name;
    c.tolerance = tolerance;
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double) got[i] - (double) ref[i]);
        c.max_abs = std::max(c.max_abs, d);
        sum += d;
    }
    c.mean_abs = sum / (double) got.size();
    c.passed = c.max_abs <= tolerance;
    return c;
}

// Upload `host` (fp16-rounded), run `op`, download into a float vector.
class Runner {
public:
    Runner(Stream & s, Device & d) : stream_(s), device_(d) {}

    Tensor upload(const std::vector<float> & host, int64_t ne0, int64_t ne1) {
        std::vector<float> q = quantize_f16(host);
        std::vector<uint16_t> h(q.size());
        f32_to_f16(q.data(), h.data(), q.size());
        buffers_.emplace_back(new Buffer(q.size() * 2));
        buffers_.back()->copy_from_host(h.data(), h.size() * 2);
        Tensor t;
        t.data = buffers_.back()->data();
        t.dtype = DType::F16;
        t.ne0 = ne0;
        t.ne1 = ne1;
        return t;
    }

    Tensor alloc(int64_t ne0, int64_t ne1) {
        buffers_.emplace_back(new Buffer((size_t) ne0 * ne1 * 2));
        Tensor t;
        t.data = buffers_.back()->data();
        t.dtype = DType::F16;
        t.ne0 = ne0;
        t.ne1 = ne1;
        return t;
    }

    std::vector<float> download(const Tensor & t) {
        std::vector<uint16_t> h(t.elements());
        aclrtMemcpy(h.data(), h.size() * 2, t.data, h.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST);
        stream_.synchronize();
        std::vector<float> out(t.elements());
        f16_to_f32(h.data(), out.data(), out.size());
        return out;
    }

private:
    Stream & stream_;
    Device & device_;
    std::vector<std::unique_ptr<Buffer>> buffers_;
};

} // namespace

int main() {
    init();
    const int count = device_count();
    if (count <= 0) {
        fprintf(stderr, "no CANN device\n");
        return 2;
    }
    DeviceInfo info = device_info(0);
    printf("device: %s, HBM %.1f GiB total / %.1f GiB free\n", info.soc.c_str(),
           (double) info.total_bytes / (1024.0 * 1024 * 1024), (double) info.free_bytes / (1024.0 * 1024 * 1024));

    Device device(0);
    Stream stream(device);
    Runner runner(stream, device);
    std::vector<Case> cases;

    // matmul 256x512 * 512x384
    {
        const int64_t M = 256, K = 512, N = 384;
        auto a = random_uniform((size_t) M * K, -0.5f, 0.5f, 1);
        auto b = random_uniform((size_t) K * N, -0.5f, 0.5f, 2);
        Tensor ta = runner.upload(a, K, M);
        Tensor tb = runner.upload(b, N, K);
        Tensor tc = runner.alloc(N, M);
        matmul(stream, ta, tb, tc);
        auto got = runner.download(tc);
        auto ref = matmul_ref(quantize_f16(a), quantize_f16(b), M, K, N);
        cases.push_back(compare("matmul 256x512x384", got, ref, 0.5));
    }

    // rms_norm 512 rows x 2048 cols
    {
        const int64_t R = 512, C = 2048;
        auto x = random_uniform((size_t) R * C, -1.0f, 1.0f, 3);
        auto g = random_uniform((size_t) C, 0.5f, 1.5f, 4);
        Tensor tx = runner.upload(x, C, R);
        Tensor tg = runner.upload(g, C, 1);
        Tensor ty = runner.alloc(C, R);
        rms_norm(stream, tx, tg, 1e-6, ty);
        auto got = runner.download(ty);
        auto ref = rms_norm_ref(quantize_f16(x), quantize_f16(g), R, C, 1e-6);
        cases.push_back(compare("rms_norm 512x2048", got, ref, 0.02));
    }

    // silu 4096 elements
    {
        auto x = random_uniform(4096, -8.0f, 8.0f, 5);
        Tensor tx = runner.upload(x, 4096, 1);
        Tensor ty = runner.alloc(4096, 1);
        silu(stream, tx, ty);
        auto got = runner.download(ty);
        auto ref = silu_ref(quantize_f16(x));
        cases.push_back(compare("silu 4096", got, ref, 0.02));
    }

    // add 4096 elements
    {
        auto a = random_uniform(4096, -2.0f, 2.0f, 6);
        auto b = random_uniform(4096, -2.0f, 2.0f, 7);
        Tensor ta = runner.upload(a, 4096, 1);
        Tensor tb = runner.upload(b, 4096, 1);
        Tensor tc = runner.alloc(4096, 1);
        add(stream, ta, tb, tc);
        auto got = runner.download(tc);
        auto ref = add_ref(quantize_f16(a), quantize_f16(b));
        cases.push_back(compare("add 4096", got, ref, 0.01));
    }

    // softmax 64 rows x 1024
    {
        const int64_t R = 64, C = 1024;
        auto x = random_uniform((size_t) R * C, -6.0f, 6.0f, 8);
        Tensor tx = runner.upload(x, C, R);
        Tensor ty = runner.alloc(C, R);
        softmax(stream, tx, ty);
        auto got = runner.download(ty);
        auto ref = softmax_ref(quantize_f16(x), R, C);
        cases.push_back(compare("softmax 64x1024", got, ref, 0.01));
    }

    // ---- shapes the GDN kernel relies on ----
    {
        const int64_t S = 8;
        auto x = random_uniform((size_t) S * S, -1.0f, 1.0f, 21);
        Tensor tx = runner.upload(x, S, S);
        Tensor ty = runner.alloc(S, S);
        muls(stream, tx, 0.5f, ty);
        auto got = runner.download(ty);
        std::vector<float> ref = quantize_f16(x);
        for (auto & v : ref) v *= 0.5f;
        cases.push_back(compare("muls 8x8 by 0.5", got, ref, 0.01));
    }
    {
        const int64_t S = 8;
        auto a = random_uniform((size_t) S * S, -1.0f, 1.0f, 22);
        auto b = random_uniform((size_t) S, -1.0f, 1.0f, 23);
        Tensor ta = runner.upload(a, S, S);
        Tensor tb = runner.upload(b, 1, S); // one column
        Tensor tc = runner.alloc(1, S);
        matmul(stream, ta, tb, tc);
        auto got = runner.download(tc);
        auto ref = matmul_ref(quantize_f16(a), quantize_f16(b), S, S, 1);
        cases.push_back(compare("matmul 8x8x1", got, ref, 0.01));
    }
    {
        const int64_t S = 8;
        auto a = random_uniform((size_t) S, -1.0f, 1.0f, 24);
        auto b = random_uniform((size_t) S, -1.0f, 1.0f, 25);
        Tensor ta = runner.upload(a, 1, S); // column
        Tensor tb = runner.upload(b, S, 1); // row
        Tensor tc = runner.alloc(S, S);
        mul(stream, ta, tb, tc);
        auto got = runner.download(tc);
        std::vector<float> ref((size_t) S * S);
        const auto aq = quantize_f16(a);
        const auto bq = quantize_f16(b);
        for (int64_t j = 0; j < S; ++j)
            for (int64_t i = 0; i < S; ++i) ref[(size_t) j * S + i] = aq[(size_t) j] * bq[(size_t) i];
        cases.push_back(compare("mul 8x1 * 1x8", got, ref, 0.01));
    }

    int failed = 0;
    printf("\n%-24s %12s %12s %10s  %s\n", "op", "max|diff|", "mean|diff|", "tolerance", "result");
    for (const Case & c : cases) {
        printf("%-24s %12.6f %12.6f %10.4f  %s\n", c.name.c_str(), c.max_abs, c.mean_abs, c.tolerance,
               c.passed ? "PASS" : "FAIL");
        failed += c.passed ? 0 : 1;
    }
    shutdown();
    return failed == 0 ? 0 : 1;
}
