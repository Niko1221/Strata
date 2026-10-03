// prefill_gdn_parity - the GDN prompt recurrence's fast path (gdn_rec_cols_pipe_fast_kernel, the four-accumulator
// chain split behind STRATA_GDN_REC_FAST=1) against the baseline pipelined kernel (STRATA_GDN_REC_FAST=0) on the
// same bounded deterministic inputs, for T = 1, 128, 4096.
//
// The split regroups the 32 products of k^T W and q^T W into four chains summed pairwise instead of one chain, so
// the two paths agree at FP32 level, not bitwise.  This bounds the divergence by scale (max|delta| / max|reference|)
// for the FP32 y and state and the FP16 y16, and requires every output to be finite.  It is not an implementation
// test: no timing, no element counts, only the recurrence's own outputs.
// Exit 77 without a CUDA device.
#include "strata/prefill/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S;   // 128, 16, 48, 10240 (kernels.cu)
constexpr float EPS = 1e-6f;                                        // prefill.cpp EPS

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// deterministic uniform RNG, bounded to the caller's range
struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    float next(float lo, float hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const float u = (float) ((s >> 11) * (1.0 / 9007199254740992.0));
        return lo + (hi - lo) * u;
    }
};

struct Out {
    std::vector<float> y, state;
    std::vector<uint16_t> y16;
};

std::string exceeds(const std::string& what, int64_t T, double got, double tol) {
    char b[192];
    std::snprintf(b, sizeof(b), "T=%lld %s: rel %.3e exceeds %.3e", (long long) T, what.c_str(), got, tol);
    return b;
}

std::string not_finite(const std::string& what, int64_t T) {
    char b[128];
    std::snprintf(b, sizeof(b), "T=%lld %s: not finite", (long long) T, what.c_str());
    return b;
}

void require_finite(const Out& o, const char* what, int64_t T) {
    for (float v : o.y) if (!std::isfinite(v)) throw std::runtime_error(not_finite(std::string(what) + " y", T));
    for (float v : o.state)
        if (!std::isfinite(v)) throw std::runtime_error(not_finite(std::string(what) + " state", T));
    for (uint16_t v : o.y16)
        if (!std::isfinite((double) __half2float(__ushort_as_half(v))))
            throw std::runtime_error(not_finite(std::string(what) + " y16", T));
}

// worst error relative to the reference's scale
double rel_f32(const std::vector<float>& a, const std::vector<float>& b) {
    double mx = 0, dmax = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > dmax) dmax = d;
        const double m = std::fabs((double) a[i]);
        if (m > mx) mx = m;
    }
    return mx > 0 ? dmax / mx : 0.0;
}

double rel_f16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    double mx = 0, dmax = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double fa = (double) __half2float(__ushort_as_half(a[i]));
        const double d = std::fabs(fa - (double) __half2float(__ushort_as_half(b[i])));
        if (d > dmax) dmax = d;
        const double m = std::fabs(fa);
        if (m > mx) mx = m;
    }
    return mx > 0 ? dmax / mx : 0.0;
}

void check_T(int64_t T, cudaStream_t stream) {
    const size_t ns = (size_t) S * HV * S;   // state: (S, HV, S)
    const size_t nh = (size_t) T * C;        // h:     (T, C)
    const size_t ng = (size_t) T * HV;       // gate/beta: (T, HV)
    const size_t nz = (size_t) T * HV * S;   // z / y / y16: (T, HV*S)

    std::vector<float> h(nh), gate(ng), beta(ng), z(nz), gamma(S), st_init(ns);
    Rng rng;
    for (float& v : h) v = rng.next(-1.0f, 1.0f);
    for (float& v : gate) v = rng.next(-2.0f, -0.5f);   // exp(g) < 1: contractive, the state stays bounded
    for (float& v : beta) v = rng.next(0.0f, 1.0f);
    for (float& v : z) v = rng.next(-1.0f, 1.0f);
    for (float& v : gamma) v = rng.next(-1.0f, 1.0f);
    for (float& v : st_init) v = rng.next(-0.5f, 0.5f);

    float *d_h, *d_gate, *d_beta, *d_z, *d_gamma, *d_state, *d_state0, *d_y;
    uint16_t* d_y16;
    const size_t bs = ns * 4, bh = nh * 4, bg = ng * 4, bz = nz * 4;
    ck(cudaMalloc(&d_h, bh), "malloc h");
    ck(cudaMalloc(&d_gate, bg), "malloc gate");
    ck(cudaMalloc(&d_beta, bg), "malloc beta");
    ck(cudaMalloc(&d_z, bz), "malloc z");
    ck(cudaMalloc(&d_gamma, S * 4), "malloc gamma");
    ck(cudaMalloc(&d_state, bs), "malloc state");
    ck(cudaMalloc(&d_state0, bs), "malloc state0");
    ck(cudaMalloc(&d_y, bz), "malloc y");
    ck(cudaMalloc(&d_y16, nz * 2), "malloc y16");
    ck(cudaMemcpy(d_h, h.data(), bh, cudaMemcpyHostToDevice), "cp h");
    ck(cudaMemcpy(d_gate, gate.data(), bg, cudaMemcpyHostToDevice), "cp gate");
    ck(cudaMemcpy(d_beta, beta.data(), bg, cudaMemcpyHostToDevice), "cp beta");
    ck(cudaMemcpy(d_z, z.data(), bz, cudaMemcpyHostToDevice), "cp z");
    ck(cudaMemcpy(d_gamma, gamma.data(), S * 4, cudaMemcpyHostToDevice), "cp gamma");
    ck(cudaMemcpy(d_state0, st_init.data(), bs, cudaMemcpyHostToDevice), "cp state0");

    const auto run = [&](const char* fast) {
        set_env("STRATA_GDN_REC_FAST", fast);
        ck(cudaMemcpyAsync(d_state, d_state0, bs, cudaMemcpyDeviceToDevice, stream), "reset");
        ck(cudaStreamSynchronize(stream), "reset sync");
        strata::prefill::gdn_recurrence(d_state, d_h, d_gate, d_beta, d_z, d_gamma, EPS, d_y, d_y16, T, stream);
        ck(cudaStreamSynchronize(stream), "run sync");
        Out o;
        o.y.resize(nz);
        o.y16.resize(nz);
        o.state.resize(ns);
        ck(cudaMemcpy(o.y.data(), d_y, bz, cudaMemcpyDeviceToHost), "cp y");
        ck(cudaMemcpy(o.y16.data(), d_y16, nz * 2, cudaMemcpyDeviceToHost), "cp y16");
        ck(cudaMemcpy(o.state.data(), d_state, bs, cudaMemcpyDeviceToHost), "cp state");
        return o;
    };

    const Out ref = run("0");
    const Out got = run("1");
    require_finite(ref, "reference", T);
    require_finite(got, "fast", T);

    const double dy = rel_f32(ref.y, got.y);
    const double ds = rel_f32(ref.state, got.state);
    const double d16 = rel_f16(ref.y16, got.y16);
    if (!(dy <= 1e-4)) throw std::runtime_error(exceeds("y", T, dy, 1e-4));
    if (!(ds <= 1e-5)) throw std::runtime_error(exceeds("state", T, ds, 1e-5));
    if (!(d16 <= 2e-3)) throw std::runtime_error(exceeds("y16", T, d16, 2e-3));

    cudaFree(d_h); cudaFree(d_gate); cudaFree(d_beta); cudaFree(d_z); cudaFree(d_gamma);
    cudaFree(d_state); cudaFree(d_state0); cudaFree(d_y); cudaFree(d_y16);
}

}  // namespace

int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        // the arms are the two pipelined kernels; do not let an unrelated A/B switch in the environment make the
        // comparison vacuous (pipe is latched on the first call)
        unset_env("STRATA_GDN_PIPELINE");
        unset_env("STRATA_GDN_REC_HEADS");
        cudaStream_t stream = nullptr;
        ck(cudaStreamCreate(&stream), "stream");
        const int64_t Ts[] = {1, 128, 4096};
        for (int64_t T : Ts) check_T(T, stream);
        ck(cudaStreamDestroy(stream), "destroy");
        std::printf("prefill GDN recurrence parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill GDN recurrence parity failed: %s\n", e.what());
        return 1;
    }
}
