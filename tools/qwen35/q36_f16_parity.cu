// Synthetic arithmetic checks, independent of the large model: F16 decode versus a double-precision oracle,
// including several columns, routed experts, nontrivial entry permutations and an empty group list.
#include "../../src/qwen35/q36_f16.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#define CK(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)

template<class T> T* device(const std::vector<T>& v) {
    T* p = nullptr;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template<class T> std::vector<T> read(const T* p, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

void compare(const char* name, const std::vector<float>& got, const std::vector<double>& want) {
    double err = 0., scale = 0.;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) { std::fprintf(stderr, "%s: nonfinite at %zu\n", name, i); std::exit(1); }
        err = std::max(err, std::abs(got[i] - want[i]));
        scale = std::max(scale, std::abs(want[i]));
    }
    std::printf("%s: %zu values, max error %.8g, scale %.8g\n", name, got.size(), err, scale);
    if (err > 2.e-5 * std::max(1., scale)) std::exit(1);
}

int main() {
    std::mt19937 rng(271828);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    for (int ni : {32, 2048, 4096}) {
        const int no = 257;
        std::vector<__half> w((size_t) ni * no);
        for (auto& v : w) v = __float2half(dist(rng));
        auto* dw = device(w);
        for (int cols : {1, 3, 8}) {
            std::vector<float> x((size_t) ni * cols), y((size_t) no * cols, 0.f);
            for (auto& v : x) v = dist(rng);
            auto* dx = device(x); auto* dy = device(y);
            q36f16::matvec<<<dim3((no + 7) / 8, cols), 256, 0, stream>>>(dw, dx, dy, ni, no);
            CK(cudaStreamSynchronize(stream));
            std::vector<double> ref(y.size(), 0.);
            for (int t = 0; t < cols; ++t)
                for (int r = 0; r < no; ++r)
                    for (int i = 0; i < ni; ++i)
                        ref[(size_t) t * no + r] += (double) __half2float(w[(size_t) r * ni + i]) * x[(size_t) t * ni + i];
            compare("matrix", read(dy, y.size()), ref);
            CK(cudaFree(dx)); CK(cudaFree(dy));
        }
        CK(cudaFree(dw));
    }

    const int n = 2048, ff = 512, ng = 3, nt = 4, entries = 8;
    const size_t blob = (size_t) 3 * n * ff;
    std::vector<__half> weights(blob * ng);
    for (auto& v : weights) v = __float2half(dist(rng));
    auto* dw = device(weights);
    std::vector<unsigned long long> ptr;
    for (int g = 0; g < ng; ++g) ptr.push_back(reinterpret_cast<unsigned long long>(dw + g * blob));
    const std::vector<int32_t> start{0, 3, 4, entries}, token{3, 0, 2, 1, 2, 1, 3, 0}, dest{7, 4, 1, 6, 3, 0, 5, 2};
    std::vector<float> x((size_t) n * nt), hidden((size_t) entries * ff), y((size_t) entries * n);
    for (auto& v : x) v = dist(rng);
    auto* dp = device(ptr); auto* ds = device(start); auto* dt = device(token); auto* dd = device(dest);
    auto* dg = device(std::vector<int32_t>{ng}); auto* dx = device(x);
    auto* dh = device(hidden); auto* dy = device(y);
    q36f16::gate_up<<<dim3((ff + 7) / 8, ng + 1), 256, 0, stream>>>(dp, ds, dg, dt, dx, dh, n, ff);
    q36f16::down<<<dim3((n + 7) / 8, ng + 1), 256, 0, stream>>>(dp, ds, dg, dd, dh, dy, n, ff);
    CK(cudaStreamSynchronize(stream));
    std::vector<double> hr(hidden.size()), ref(y.size());
    for (int g = 0; g < ng; ++g) {
        const auto* w = weights.data() + g * blob;
        for (int e = start[g]; e < start[g + 1]; ++e) {
            for (int r = 0; r < ff; ++r) {
                double a = 0., b = 0.;
                for (int i = 0; i < n; ++i) {
                    a += (double) __half2float(w[(size_t) r * n + i]) * x[(size_t) token[e] * n + i];
                    b += (double) __half2float(w[(size_t) (ff + r) * n + i]) * x[(size_t) token[e] * n + i];
                }
                hr[(size_t) e * ff + r] = a / (1. + std::exp(-a)) * b;
            }
            for (int r = 0; r < n; ++r)
                for (int i = 0; i < ff; ++i)
                    ref[(size_t) dest[e] * n + r] += (double) __half2float(w[(size_t) 2 * n * ff + (size_t) r * ff + i]) * hr[(size_t) e * ff + i];
        }
    }
    compare("grouped hidden", read(dh, hidden.size()), hr);
    compare("grouped output", read(dy, y.size()), ref);

    const int32_t zero = 0;
    CK(cudaMemcpy(dg, &zero, sizeof(zero), cudaMemcpyHostToDevice));
    const auto before = read(dy, y.size());
    q36f16::gate_up<<<dim3((ff + 7) / 8, ng + 1), 256, 0, stream>>>(dp, ds, dg, dt, dx, dh, n, ff);
    q36f16::down<<<dim3((n + 7) / 8, ng + 1), 256, 0, stream>>>(dp, ds, dg, dd, dh, dy, n, ff);
    CK(cudaStreamSynchronize(stream));
    if (before != read(dy, y.size())) { std::fprintf(stderr, "empty groups changed output\n"); return 1; }
    std::puts("empty groups: unchanged; PASS");
    CK(cudaFree(dw)); CK(cudaFree(dp)); CK(cudaFree(ds)); CK(cudaFree(dt)); CK(cudaFree(dd)); CK(cudaFree(dg));
    CK(cudaFree(dx)); CK(cudaFree(dh)); CK(cudaFree(dy)); CK(cudaStreamDestroy(stream));
    return 0;
}
