// Independent double-precision attention oracle: FP32/FP16 caches, GQA, causal
// tails, offset query rows, session pointer tables, and an actual 262144-cell KV.
#include "../../src/qwen35/q36_attn.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CK(x) do { const auto e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

struct Rows {
    const int32_t *cell = nullptr, *slot = nullptr, *delta = nullptr;
    float *const *kc = nullptr, *const *vc = nullptr;
};
template<class T> T* upload(const std::vector<T>& v) {
    T* p; CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template<class T> std::vector<T> read(const T* p, size_t n) {
    std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
float value(size_t i, unsigned seed) {
    unsigned v = (unsigned) i * 747796405u + seed;
    v = ((v >> ((v >> 28) + 4)) ^ v) * 277803737u; v = (v >> 22) ^ v;
    // Exact in both cache formats, so the oracle tests the storage/read path too.
    return ((int) (v & 255) - 128) * (1.f / 128.f);
}

template<class KV> void run(int ctx, int group, int tokens, bool sessions, cudaStream_t stream) {
    constexpr int D = 256, KVH = 2;
    const int H = KVH * group, chunks = (ctx + 255) / 256, slots = sessions ? 2 : 1;
    const int t0 = sessions ? 1 : 0, nt = tokens - t0;
    const size_t cache = (size_t) KVH * ctx * D;
    std::vector<std::vector<KV>> keys(slots), values(slots);
    std::vector<float*> kp, vp;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int s = 0; s < slots; ++s) {
        keys[s].resize(cache); values[s].resize(cache);
        for (size_t i = 0; i < cache; ++i) {
            const bool poison = sessions && s == 0 && (int) (i / D % ctx) > 128;
            keys[s][i] = (KV) (poison ? nan : value(i, 17 + s));
            values[s][i] = (KV) (poison ? nan : value(i, 31 + s));
        }
        kp.push_back(reinterpret_cast<float*>(upload(keys[s])));
        vp.push_back(reinterpret_cast<float*>(upload(values[s])));
    }
    std::vector<float> q((size_t) tokens * H * D);
    for (size_t i = 0; i < q.size(); ++i) q[i] = value(i, 43);
    float* dq = upload(q);
    const int32_t pos = ctx - tokens;
    int32_t* dp = upload(std::vector<int32_t>{pos});
    const std::vector<int32_t> cells{0, 128, 256}, slot{1, 0, 1};
    Rows rows;
    if (sessions) {
        rows.cell = upload(cells); rows.slot = upload(slot);
        rows.kc = upload(kp); rows.vc = upload(vp);
    }
    float *acc, *mx, *den;
    const size_t parts = (size_t) nt * H * chunks;
    CK(cudaMalloc(&acc, parts * D * sizeof(float)));
    CK(cudaMalloc(&mx, parts * sizeof(float))); CK(cudaMalloc(&den, parts * sizeof(float)));
    q36attn::part<<<dim3(chunks, nt * KVH), 256, 0, stream>>>(dq,
        reinterpret_cast<const KV*>(kp[0]), reinterpret_cast<const KV*>(vp[0]), dp,
        t0, H, KVH, ctx, 1.f / 16, chunks, acc, mx, den, rows);
    CK(cudaStreamSynchronize(stream));
    const auto a = read(acc, parts * D), m = read(mx, parts), l = read(den, parts);
    double error = 0;
    int compared = 0;
    for (int r = 0; r < nt; ++r) for (int h = 0; h < H; ++h) {
        // Four heads span both GQA groups in the large case; small cases cover all.
        if (ctx > 4096 && h != 0 && h != group - 1 && h != group && h != H - 1) continue;
        const int t = t0 + r, s = sessions ? slot[t] : 0;
        const int n = (sessions ? cells[t] : pos + t) + 1, nc = (n + 255) / 256;
        const size_t base = (size_t) (h / group) * ctx * D;
        std::vector<double> logits(n);
        double maximum = -INFINITY;
        for (int p = 0; p < n; ++p) {
            double dot = 0;
            for (int d = 0; d < D; ++d)
                dot += (double) q[((size_t) t * H + h) * D + d] * (float) keys[s][base + (size_t) p * D + d];
            logits[p] = dot / 16; maximum = std::max(maximum, logits[p]);
        }
        std::array<double, D> want{}, got{};
        double z = 0;
        for (int p = 0; p < n; ++p) {
            const double w = std::exp(logits[p] - maximum); z += w;
            for (int d = 0; d < D; ++d) want[d] += w * (float) values[s][base + (size_t) p * D + d];
        }
        const size_t off = (size_t) (r * H + h) * chunks;
        double global_max = -INFINITY, denom = 0;
        for (int c = 0; c < nc; ++c) global_max = std::max(global_max, (double) m[off + c]);
        for (int c = 0; c < nc; ++c) {
            const double w = std::exp((double) m[off + c] - global_max); denom += w * l[off + c];
            for (int d = 0; d < D; ++d) got[d] += w * a[(off + c) * D + d];
        }
        for (int d = 0; d < D; ++d) {
            const double v = got[d] / denom;
            if (!std::isfinite(v)) { std::fprintf(stderr, "nonfinite output\n"); std::exit(1); }
            error = std::max(error, std::abs(v - want[d] / z)); ++compared;
        }
    }
    std::printf("KV%zu ctx%d G%d rows%d offset%d sessions%d: %d values, max error %.9g\n",
                sizeof(KV) * 8, ctx, group, nt, t0, sessions, compared, error);
    if (error > 2.e-5) std::exit(1);
    for (float* p : kp) CK(cudaFree(p)); for (float* p : vp) CK(cudaFree(p));
    CK(cudaFree(dq)); CK(cudaFree(dp)); CK(cudaFree(acc)); CK(cudaFree(mx)); CK(cudaFree(den));
    if (sessions) {
        CK(cudaFree((void*) rows.cell)); CK(cudaFree((void*) rows.slot));
        CK(cudaFree((void*) rows.kc)); CK(cudaFree((void*) rows.vc));
    }
}
int main() {
    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    run<float>(5, 1, 1, false, stream); run<__half>(5, 1, 1, false, stream);
    run<float>(257, 8, 3, true, stream); run<__half>(257, 8, 3, true, stream);
    run<float>(4096, 8, 3, false, stream); run<__half>(4096, 8, 3, false, stream);
    run<float>(262144, 8, 1, false, stream); run<__half>(262144, 8, 1, false, stream);
    CK(cudaStreamDestroy(stream)); std::puts("PASS");
}
