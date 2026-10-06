#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Buffer {
    float* p = nullptr;
    size_t n;
    explicit Buffer(size_t count): n(count) { check(cudaMalloc((void**)&p, n * 4)); }
    ~Buffer() { cudaFree(p); }
    void fill(float scale, int seed) {
        std::vector<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = scale * (float)((int)((i * 17 + seed) % 101) - 50);
        check(cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice));
    }
    void copy(const Buffer& src) { check(cudaMemcpy(p, src.p, n * 4, cudaMemcpyDeviceToDevice)); }
    void equal(const Buffer& other, const char* label) const {
        std::vector<float> a(n), b(n);
        check(cudaMemcpy(a.data(), p, n * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), other.p, n * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(a.data(), b.data(), n * 4)) {
            std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1);
        }
    }
};

int main() {
    using namespace strata::kernels;
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    if (free < (8ull << 30) + (32ull << 20)) {
        std::fprintf(stderr, "SKIP: preserving 8 GiB reserve\n"); return 77;
    }
    constexpr int HK = 2, HV = 4, C = (2 * HK + HV) * 128, Z = HV * 128, ST = 128 * HV * 128;
    Buffer initial(2 * ST), history(2 * C * 3), state(2 * ST), hist(2 * C * 3),
           reference(2 * ST), refhist(2 * C * 3), weights(C * 4), gamma(128),
           qkv(8 * C), h(8 * C), gate(8 * HV), beta(8 * HV), z(8 * Z), y(8 * Z), refy(8 * Z), scratch(C);
    initial.fill(0.001f, 1); history.fill(0.005f, 3); weights.fill(0.002f, 7);
    gamma.fill(0.01f, 11); qkv.fill(0.003f, 13); gate.fill(0.001f, 17);
    beta.fill(0.001f, 19); z.fill(0.01f, 23);
    int32_t* keep = nullptr;
    check(cudaMalloc((void**)&keep, 4));
    auto commit_count = [&](int k) { check(cudaMemcpy(keep, &k, 4, cudaMemcpyHostToDevice)); };
    auto sequential = [&](int a, int b, int ka, int kb) {
        reference.copy(initial); refhist.copy(history);
        int offsets[] = {0, a}, counts[] = {ka, kb};
        commit_count(1);
        for (int slot = 0; slot < 2; ++slot) for (int j = 0; j < counts[slot]; ++j) {
            const int row = offsets[slot] + j;
            gdn_conv_l2_multi(refhist.p + slot * C * 3, qkv.p + row * C, weights.p, scratch.p, C, 2 * HK, 1e-6f, 1, nullptr);
            gdn_step_norm_multi(reference.p + slot * ST, scratch.p, C, gate.p + row * HV, beta.p + row * HV,
                                z.p + row * Z, gamma.p, 1e-6f, refy.p + row * Z, HK, HV, 1, keep, nullptr);
            gdn_conv_commit(refhist.p + slot * C * 3, qkv.p + row * C, C, keep, nullptr);
        }
        check(cudaDeviceSynchronize());
    };
    int cases = 0;
    for (int a = 1; a <= 4; ++a) for (int b = 1; b <= 4; ++b) {
        state.copy(initial); hist.copy(history);
        check(cudaMemset(y.p, 0, y.n * 4)); check(cudaMemset(refy.p, 0, refy.n * 4));
        int offsets[] = {0, a}, counts[] = {a, b};
        for (int slot = 0; slot < 2; ++slot) {
            const int row = offsets[slot], width = counts[slot];
            gdn_conv_l2_multi(hist.p + slot * C * 3, qkv.p + row * C, weights.p, h.p + row * C, C, 2 * HK, 1e-6f, width, nullptr);
            gdn_step_norm_multi(state.p + slot * ST, h.p + row * C, C, gate.p + row * HV, beta.p + row * HV,
                                z.p + row * Z, gamma.p, 1e-6f, y.p + row * Z, HK, HV, width, nullptr, nullptr);
        }
        check(cudaDeviceSynchronize());
        state.equal(initial, "verification mutated recurrent state"); hist.equal(history, "verification mutated convolution history");
        sequential(a, b, a, b);
        y.equal(refy, "segmented outputs differ from sequential outputs");
        for (int ka = 1; ka <= a; ++ka) for (int kb = 1; kb <= b; ++kb) {
            state.copy(initial); hist.copy(history);
            int accepted[] = {ka, kb};
            for (int slot = 0; slot < 2; ++slot) {
                const int row = offsets[slot];
                commit_count(accepted[slot]);
                gdn_conv_commit(hist.p + slot * C * 3, qkv.p + row * C, C, keep, nullptr);
                gdn_step_norm_multi(state.p + slot * ST, h.p + row * C, C, gate.p + row * HV, beta.p + row * HV,
                                    z.p + row * Z, gamma.p, 1e-6f, y.p + row * Z, HK, HV, counts[slot], keep, nullptr);
            }
            check(cudaDeviceSynchronize());
            sequential(a, b, ka, kb);
            state.equal(reference, "partial commit recurrent state"); hist.equal(refhist, "partial commit convolution history");
            ++cases;
        }
    }
    check(cudaFree(keep));
    std::printf("PASS: 16 two-slot layouts, %d partial commits, bitwise outputs and state\n", cases);
}
