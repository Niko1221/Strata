// Check hyper-connection kernels against CPU arithmetic and the unfused CUDA path.
#include "strata/hy4flash_gguf/ihc.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-76s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

std::string sci(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.3g", v);
    return b;
}

struct Buf {
    void* p = nullptr;
    explicit Buf(size_t bytes) {
        if (bytes != 0 && cudaMalloc(&p, bytes) != cudaSuccess) p = nullptr;
    }
    ~Buf() {
        if (p) cudaFree(p);
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p(o.p) { o.p = nullptr; }   // so `upload` can return one
    template <class T>
    T* as() const { return static_cast<T*>(p); }
};

Buf upload(const void* host, size_t bytes) {
    Buf b(bytes);
    if (b.p != nullptr) cudaMemcpy(b.p, host, bytes, cudaMemcpyHostToDevice);
    return b;
}

double worst_diff(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    double w = 0.0;
    for (size_t i = 0; i < n && i < a.size() && i < b.size(); ++i)
        w = std::max(w, std::fabs((double) a[i] - (double) b[i]));
    return w;
}

/// reduce and post on the CPU, from the formula. Deliberately written from ihc.hpp's contract rather than
/// factored out of the .cu, so a mistake in the kernel does not reproduce itself here.
void cpu_reduce(const std::vector<float>& x, const std::vector<float>& w, int hidden, int hc, int nt,
                std::vector<float>& out) {
    out.assign((size_t) hidden * nt, 0.f);
    for (int t = 0; t < nt; ++t)
        for (int d = 0; d < hidden; ++d) {
            double acc = 0.0;
            for (int ih = 0; ih < hc; ++ih)
                acc += (double) x[(size_t) t * hidden * hc + (size_t) ih * hidden + (size_t) d] *
                       (double) w[(size_t) t * hc + (size_t) ih];
            out[(size_t) t * hidden + (size_t) d] = (float) acc;
        }
}

void cpu_post(const std::vector<float>& y, const std::vector<float>& res, const std::vector<float>& post,
              int hidden, int hc, int nt, std::vector<float>& out) {
    out.assign((size_t) hidden * hc * nt, 0.f);
    for (int t = 0; t < nt; ++t)
        for (int ih = 0; ih < hc; ++ih)
            for (int d = 0; d < hidden; ++d) {
                const size_t i = (size_t) t * hidden * hc + (size_t) ih * hidden + (size_t) d;
                out[i] = res[i] + y[(size_t) t * hidden + (size_t) d] * post[(size_t) t * hc + (size_t) ih];
            }
}

cudaStream_t g_stream{};

void synthetic_reduce_and_post() {
    constexpr int hidden = 320, hc = 4, nt = 3;
    std::mt19937 rng(20261106);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) hidden * hc * nt), w((size_t) hc * nt), y((size_t) hidden * nt);
    for (float& v : x) v = nd(rng);
    for (float& v : w) v = 0.5f + 0.1f * nd(rng);
    for (float& v : y) v = nd(rng);

    Buf dx = upload(x.data(), x.size() * 4), dw = upload(w.data(), w.size() * 4),
        dy = upload(y.data(), y.size() * 4);
    Buf dred((size_t) hidden * nt * 4), dout(x.size() * 4);
    strata::hy4_flash::hc_reduce(dx.as<float>(), dw.as<float>(), hidden, hc, nt, dred.as<float>(), g_stream);
    strata::hy4_flash::hc_post(dy.as<float>(), dx.as<float>(), dw.as<float>(), hidden, hc, nt, dout.as<float>(),
                             g_stream);
    cudaStreamSynchronize(g_stream);

    std::vector<float> got_red((size_t) hidden * nt), got_out(x.size()), want_red, want_out;
    cudaMemcpy(got_red.data(), dred.p, got_red.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_out.data(), dout.p, got_out.size() * 4, cudaMemcpyDeviceToHost);
    cpu_reduce(x, w, hidden, hc, nt, want_red);
    cpu_post(y, x, w, hidden, hc, nt, want_out);
    check(worst_diff(got_red, want_red, want_red.size()) < 1e-5,
          "hc_reduce matches the CPU formula (worst " + sci(worst_diff(got_red, want_red, want_red.size())) + ")");
    check(worst_diff(got_out, want_out, want_out.size()) < 1e-5,
          "hc_post matches the CPU formula (worst " + sci(worst_diff(got_out, want_out, want_out.size())) + ")");
}

void guards() {
    bool threw = false;
    try {
        strata::hy4_flash::hc_flat_rms(nullptr, nullptr, 12288, 1, 1e-5f, g_stream);
    } catch (const std::exception&) { threw = true; }
    check(threw, "hc_flat_rms refuses null buffers rather than launching on them");

    threw = false;
    try {   // n_out must be 2*hc with a post gate; hc alone is the output-reduce shape
        float dummy = 0.f;
        strata::hy4_flash::hc_gates(&dummy, &dummy, &dummy, 4, 1, 4, 1e-6f, 2.f, &dummy, &dummy, g_stream);
    } catch (const std::exception&) { threw = true; }
    check(threw, "hc_gates refuses n_out == hc when a post gate was asked for");
}

/// Compare fused flat RMS and quantization against the two independent launches.
void fused_pre_vs_split(int hc_width, int nt) {
    std::mt19937 rng(20261120u + (unsigned) hc_width + (unsigned) nt);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> x((size_t) hc_width * nt);
    for (float& v : x) v = nd(rng);
    const float eps = 1e-5f;
    const size_t qb = strata::kernels::native_q8_1_bytes_wide(hc_width, nt);
    const std::string what = "fused rms+quantise == split, width " + std::to_string(hc_width) + ", n_tok " +
                             std::to_string(nt);
    Buf dx = upload(x.data(), x.size() * 4), norm((size_t) hc_width * nt * 4), split_q(qb), fused_q(qb);
    if (dx.p == nullptr || norm.p == nullptr || split_q.p == nullptr || fused_q.p == nullptr) {
        check(false, what + " (device buffers)");
        return;
    }
    cudaMemset(split_q.p, 0x5a, qb);
    cudaMemset(fused_q.p, 0xa5, qb);
    strata::hy4_flash::hc_flat_rms(dx.as<float>(), norm.as<float>(), hc_width, nt, eps, g_stream);
    strata::kernels::native_quantize_q8_1_wide(norm.as<float>(), split_q.p, hc_width, nt, g_stream);
    strata::hy4_flash::hc_pre_rms_quant_fused(dx.as<float>(), fused_q.p, hc_width, nt, eps, g_stream);
    const bool ran = cudaStreamSynchronize(g_stream) == cudaSuccess;
    std::vector<uint8_t> a(qb), b(qb);
    cudaMemcpy(a.data(), split_q.p, qb, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), fused_q.p, qb, cudaMemcpyDeviceToHost);
    size_t bad = 0, first = qb;
    for (size_t i = 0; i < qb; ++i)
        if (a[i] != b[i]) {
            if (bad == 0) first = i;
            ++bad;
        }
    check(ran && bad == 0, what + (bad ? " (" + std::to_string(bad) + " of " + std::to_string(qb) +
                                             " bytes differ, first " + std::to_string(first) + ")"
                                       : ""));
}

/// The reference above IS hc_pre's unfused path: with the fusion off, hc_pre leaves the same Q8_1 in its
/// scratch (after the F32 norm) as hc_flat_rms + native_quantize_q8_1_wide. The Q4_K matrix is all zeros -
/// finite, and nothing here reads the mixes it produces.
void hc_pre_unfused_is_the_reference(int nt) {
    constexpr int hidden = 3072, hc = 4, hc_width = hidden * hc;
    if (strata::hy4_flash::hc_fuse_on()) {
        check(false, "hc_pre's reference needs STRATA_HY4_FLASH_HC_FUSE off (it is set in this environment)");
        return;
    }
    std::mt19937 rng(7u + (unsigned) nt);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> x((size_t) hc_width * nt);
    for (float& v : x) v = nd(rng);
    std::vector<uint8_t> fn(strata::kernels::native_mmvq_weight_bytes(12, hc_width, 2 * hc), 0);
    const std::vector<float> scale = {0.5f, -0.4f}, base((size_t) 2 * hc, 0.1f);
    const size_t qb = strata::kernels::native_q8_1_bytes_wide(hc_width, nt);
    Buf dx = upload(x.data(), x.size() * 4), dfn = upload(fn.data(), fn.size()),
        dsc = upload(scale.data(), scale.size() * 4), dba = upload(base.data(), base.size() * 4);
    Buf scratch(strata::hy4_flash::hc_scratch_bytes(hc_width, nt)), norm((size_t) hc_width * nt * 4), ref(qb);
    Buf dmix((size_t) 2 * hc * nt * 4), dpre((size_t) hc * nt * 4), dpost((size_t) hc * nt * 4),
        dred((size_t) hidden * nt * 4);
    strata::hy4_flash::hc_pre(dx.as<float>(), dfn.p, dsc.as<float>(), dba.as<float>(), hidden, hc, nt, 2 * hc,
                            1e-5f, 1e-6f, 2.0f, scratch.p, dmix.as<float>(), dpre.as<float>(), dpost.as<float>(),
                            dred.as<float>(), g_stream);
    strata::hy4_flash::hc_flat_rms(dx.as<float>(), norm.as<float>(), hc_width, nt, 1e-5f, g_stream);
    strata::kernels::native_quantize_q8_1_wide(norm.as<float>(), ref.p, hc_width, nt, g_stream);
    const bool ran = cudaStreamSynchronize(g_stream) == cudaSuccess;
    std::vector<uint8_t> a(qb), b(qb);
    cudaMemcpy(a.data(), static_cast<const char*>(scratch.p) + (size_t) hc_width * nt * 4, qb,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), ref.p, qb, cudaMemcpyDeviceToHost);
    check(ran && a == b, "hc_pre (fusion off) quantises as hc_flat_rms + quantise, n_tok " + std::to_string(nt));
}

/// The widths the launcher must refuse: not whole 1024-element slices (4608 would leave its last 512
/// elements unquantised), or outside the 1024-thread vector form (4096, 20480).
void fused_pre_refuses() {
    Buf dx((size_t) 20480 * 4), q(strata::kernels::native_q8_1_bytes_wide(20480, 1));
    for (int w : {4608, 4096, 20480}) {
        bool threw = false;
        try {
            strata::hy4_flash::hc_pre_rms_quant_fused(dx.as<float>(), q.p, w, 1, 1e-5f, g_stream);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "fused rms+quantise refuses width " + std::to_string(w));
    }
}

}  // namespace

int main() {
    if (cudaStreamCreate(&g_stream) != cudaSuccess) {
        std::printf("no CUDA stream: nothing to check\n");
        return 0;
    }
    std::printf("argument guards\n");
    guards();
    std::printf("hc_reduce / hc_post against the CPU formula\n");
    synthetic_reduce_and_post();
    std::printf("the fused HC pre-half against the split path\n");
    for (int nt : {1, 2, 3, 4}) hc_pre_unfused_is_the_reference(nt);
    for (int w : {5120, 9216, 12288, 16384})
        for (int nt : {1, 2, 3, 4}) fused_pre_vs_split(w, nt);
    fused_pre_refuses();

    cudaStreamDestroy(g_stream);
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
