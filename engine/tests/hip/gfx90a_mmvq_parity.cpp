// M1 synthetic portable MMVQ gate; not an inventory of a real IQ3_S pack.
// Uses production launches and GGML's independent host dequantizer. All dots
// use the SAME stored half-scale Q8_1 activations, not the original float x.
// Not covered: asymmetric Q4_K/Q5_K min terms, every other quant, full model
// shapes, alternate Wave64 layouts, graphs, or hardware performance.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(STRATA_HIP_GFX90A) || !defined(STRATA_HIP_WAVE64) || defined(STRATA_HIP_GFX906)
#error "Build with the gfx90a production target's wave64 definitions"
#endif

namespace K = strata::kernels;
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&p, bytes), "hipMalloc"); }
    ~Buffer() { if (p) hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const void* src, size_t bytes) { check(hipMemcpy(p, src, bytes, hipMemcpyHostToDevice), "upload"); }
    void get(void* dst, size_t bytes) const { check(hipMemcpy(dst, p, bytes, hipMemcpyDeviceToHost), "download"); }
    template<class T> T* as() const { return static_cast<T*>(p); }
};
struct Stream {
    hipStream_t s = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "create stream"); }
    ~Stream() { if (s) hipStreamDestroy(s); }
};

int device_check() {
    int count = 0;
    const hipError_t e = hipGetDeviceCount(&count);
    if (e == hipErrorNoDevice || (e == hipSuccess && count == 0)) {
        std::puts("SKIP gfx90a_mmvq_parity: no HIP device; executed=0");
        return 77;
    }
    check(e, "hipGetDeviceCount");
    int dev = 0;
    check(hipGetDevice(&dev), "hipGetDevice");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, dev), "hipGetDeviceProperties");
    const std::string raw = prop.gcnArchName;
    require(raw.substr(0, raw.find(':')) == "gfx90a" && prop.warpSize == 64,
            "requires exact gfx90a (feature suffix allowed) and physical wave64");
    std::printf("gfx90a_mmvq_parity device=%d arch=%s wave=%d\n", dev, raw.c_str(), prop.warpSize);
    return 0;
}

struct Format { const char* name; int type, block, bytes, scale; bool iq; };
constexpr Format formats[] = {
    {"IQ3_S", 21, 256, 110, 0, true}, // distinct from IQ3_XXS (type 18)
    {"IQ4_XS", 23, 256, 136, 0, false},
    {"IQ4_NL", 20, 32, 18, 0, true},
    {"Q2_0", 42, 64, 18, 0, false},
    {"Q6_K", 14, 256, 210, 208, false},
    {"Q8_0", 8, 32, 34, 0, false},
};

std::vector<uint8_t> weights(const Format& f, int rows, int cols, unsigned seed) {
    const size_t blocks = (size_t) rows * cols / f.block;
    std::vector<uint8_t> w(blocks * f.bytes);
    // Arbitrary codebook indices/signs are legal blocks; overwrite every fp16
    // scale with a finite normal power of two, never NaN/Inf random bit patterns.
    uint32_t state = seed;
    for (auto& b : w) { state = state * 1664525u + 1013904223u; b = (uint8_t) (state >> 24); }
    for (size_t b = 0; b < blocks; ++b) {
        const float d = std::ldexp((b & 1) ? -1.0f : 1.0f, -12 - (int) (b % 3));
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(w.data() + b * f.bytes + f.scale, &h, 2);
    }
    return w;
}

// Amax=127*d, d is a power of two, so production quantization must recover
// every code and scale exactly. This checks quantization as well as the dots.
std::vector<uint8_t> activations(int cols, int k, unsigned seed, std::vector<float>& x) {
    x.resize((size_t) cols * k);
    std::vector<uint8_t> q((size_t) cols * (k / 32) * 36, 0);
    for (int c = 0; c < cols; ++c) for (int b = 0; b < k / 32; ++b) {
        uint8_t* dst = q.data() + ((size_t) c * (k / 32) + b) * 36;
        const float d = c == 1 ? 0.0f : std::ldexp(1.0f, -8 + (b % 3));
        int sum = 0;
        for (int l = 0; l < 32; ++l) {
            int v = (int) ((l * 37 + b * 13 + c * 17 + seed) % 255) - 127;
            if (c == 1) v = 0;
            else if (c == 2) v = l == 0 ? 127 : 0;
            else if (c == 3) v = (l & 1) ? -127 : 127;
            else if (l == 0) v = 127;
            const int8_t code = (int8_t) v;
            std::memcpy(dst + 4 + l, &code, 1);
            x[(size_t) c * k + b * 32 + l] = d * v;
            sum += v;
        }
        const ggml_fp16_t dh = ggml_fp32_to_fp16(d), sh = ggml_fp32_to_fp16(d * sum);
        std::memcpy(dst, &dh, 2); std::memcpy(dst + 2, &sh, 2);
    }
    return q;
}

float activation(const uint8_t* q, int i) {
    ggml_fp16_t h;
    std::memcpy(&h, q + (i / 32) * 36, 2);
    int8_t v;
    std::memcpy(&v, q + (i / 32) * 36 + 4 + i % 32, 1);
    return ggml_fp16_to_fp32(h) * v;
}

void launch(const Format& f, const void* w, const void* q, float* y, int k, int n, int nc, hipStream_t s) {
    if (f.iq) K::iq_mmvq(f.type, w, q, y, k, n, nc, (void*) s);
    else K::native_mmvq(f.type, w, q, y, k, n, nc, (void*) s);
    check(hipGetLastError(), "MMVQ launch");
}

void run(const Format& f, int k, int n, unsigned seed, hipStream_t s, int& executed) {
    require(f.iq ? K::iq_supported(f.type) : K::native_mmvq_supported(f.type), "required format unavailable");
    const auto* tr = ggml_get_type_traits((ggml_type) f.type);
    require(tr && tr->to_float, "missing independent GGML host dequantizer");
    require(ggml_row_size((ggml_type) f.type, k) == (size_t) (k / f.block) * f.bytes, "weight ABI mismatch");
    const auto w = weights(f, n, k, seed);
    std::vector<float> wf((size_t) n * k), x;
    for (int r = 0; r < n; ++r)
        tr->to_float(w.data() + (size_t) r * (k / f.block) * f.bytes, wf.data() + (size_t) r * k, k);
    const auto q = activations(8, k, seed, x);
    Buffer dw(w.size()), dx(x.size() * 4), dq(q.size());
    constexpr size_t guard = 16;
    constexpr float fence = -12345.25f;
    const size_t out_count = (size_t) n * 8;
    Buffer dy((out_count + 2 * guard) * 4), dsingle(out_count * 4);
    dw.put(w.data(), w.size()); dx.put(x.data(), x.size() * 4);
    K::native_quantize_q8_1(dx.as<float>(), dq.p, k, 8, (void*) s);
    check(hipGetLastError(), "Q8_1 launch");
    check(hipStreamSynchronize(s), "quantize synchronize");
    std::vector<uint8_t> gotq(q.size());
    dq.get(gotq.data(), gotq.size());
    require(gotq == q, "production Q8_1 differs from exact CPU fixture");
    for (int nc = 1; nc <= 8; ++nc) {
        std::vector<float> y(out_count + 2 * guard, fence);
        std::fill(y.begin() + guard, y.begin() + guard + (size_t) nc * n, std::nanf(""));
        dy.put(y.data(), y.size() * 4);
        launch(f, dw.p, dq.p, dy.as<float>() + guard, k, n, nc, s);
        for (int c = 0; c < nc; ++c)
            launch(f, dw.p, dq.as<uint8_t>() + (size_t) c * (k / 32) * 36,
                   dsingle.as<float>() + (size_t) c * n, k, n, 1, s);
        check(hipStreamSynchronize(s), "MMVQ synchronize");
        dy.get(y.data(), y.size() * 4);
        std::vector<float> single((size_t) nc * n);
        dsingle.get(single.data(), single.size() * 4);
        require(std::memcmp(single.data(), y.data() + guard, single.size() * 4) == 0,
                "multi-column result differs from single-column calls");
        for (size_t i = 0; i < y.size(); ++i)
            if (i < guard || i >= guard + (size_t) nc * n) require(y[i] == fence, "output guard overwritten");
        double err2 = 0, ref2 = 0, max_abs = 0, max_rel = 0;
        const double u = std::ldexp(1.0, -24), gamma = (k * u) / (1.0 - k * u);
        for (int c = 0; c < nc; ++c) for (int r = 0; r < n; ++r) {
            double ref = 0, magnitude = 0;
            for (int i = 0; i < k; ++i) {
                const double term = (double) wf[(size_t) r * k + i] * activation(q.data() + (size_t) c * (k / 32) * 36, i);
                ref += term; magnitude += std::fabs(term);
            }
            const float got = y[guard + (size_t) c * n + r];
            const double error = std::fabs((double) got - ref);
            // Stored-scale dequant + integer dot are exact before FP32 scale
            // products/reduction. 8*gamma_K covers those rounding operations;
            // the small absolute term covers near-zero cancellation only.
            const double budget = 8 * gamma * magnitude + 2e-6;
            require(std::isfinite(got) && std::isfinite(ref) && error <= budget, "MMVQ exceeds stored-activation oracle budget");
            if (c == 1) require(got == 0.0f, "zero activation produced nonzero output");
            err2 += error * error; ref2 += ref * ref;
            max_abs = std::max(max_abs, error); max_rel = std::max(max_rel, error / std::max(std::fabs(ref), 1e-12));
        }
        require(ref2 > 0, "vacuous all-zero oracle");
        ++executed;
        std::printf("case=%d type=%s K=%d N=%d cols=%d abs=%.3e rel=%.3e l2=%.3e PASS\n",
                    executed, f.name, k, n, nc, max_abs, max_rel, std::sqrt(err2 / ref2));
    }
    std::vector<uint8_t> after(w.size()); dw.get(after.data(), after.size());
    require(after == w, "MMVQ modified weight input");
    dq.get(gotq.data(), gotq.size()); require(gotq == q, "MMVQ modified activation input");
    std::vector<float> afterx(x.size()); dx.get(afterx.data(), afterx.size() * 4);
    require(std::memcmp(afterx.data(), x.data(), x.size() * 4) == 0, "quantizer modified float input");
}

void rejection(hipStream_t s, int& executed) {
    Buffer w(4096), q(4096), y(64);
    const std::vector<uint8_t> sentinel(64, 0xa5);
    y.put(sentinel.data(), sentinel.size());
    int rejected = 0;
    for (int c = 0; c < 5; ++c) {
        try {
            K::native_mmvq(8, c == 1 ? nullptr : w.p, c == 2 ? nullptr : q.p,
                           c == 3 ? nullptr : y.as<float>(), c == 0 ? 31 : 32, 1, 1,
                           c == 4 ? nullptr : (void*) s);
        } catch (const std::invalid_argument&) { ++rejected; }
    }
    check(hipStreamSynchronize(s), "rejection synchronize");
    std::vector<uint8_t> after(64); y.get(after.data(), after.size());
    require(rejected == 5 && after == sentinel, "native invalid-argument call did not reject before writing output");
    executed += rejected;
}
} // namespace

int main() {
    int executed = 0;
    try {
        const int status = device_check(); if (status) return status;
        Stream stream;
        K::native_mmvq_set_multi_exact(true);
        for (const auto& f : formats) {
            for (int n : {1, 31, 32, 33, 63, 64, 65}) run(f, 256, n, 17u + (unsigned) n, stream.s, executed);
            run(f, 512, 33, 91, stream.s, executed);
        }
        rejection(stream.s, executed);
        require(executed == 389, "missing required MMVQ cases");
        std::printf("gfx90a_mmvq_parity PASS executed=%d expected=389 formats=6 MMQ=not-used\n", executed);
        std::puts("NOT_COVERED: real-pack inventory, other quants/min terms, full model shapes, alternate layouts, graphs, performance");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gfx90a_mmvq_parity FAIL executed=%d: %s\n", executed, e.what());
        return 1;
    }
}
