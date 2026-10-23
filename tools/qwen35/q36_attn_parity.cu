// tools/qwen35/q36_attn_parity.cu - src/qwen35/q36_attn.cuh (#3641) against an independent double-precision oracle.
// Independent double-precision attention oracle for q36attn::part / part_q8 and merge: FP32/FP16/int8 caches, GQA, causal
// tails, offset query rows, session pointer tables, and an actual 262144-cell KV.
// The int8 cache (q36attn::q8_cell, --kv int8) is checked twice: against the oracle on the
// values its cells hold (the read path, as tight as FP32), and against the oracle on the
// FP32 values it was quantized from (the format's own error, a looser bound).
#include "../../src/qwen35/q36_attn.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
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
    // Exact in the FP32/FP16 cache formats, so the oracle tests the storage/read path too.
    return ((int) (v & 255) - 128) * (1.f / 128.f);
}
// A cache's storage from its FP32 values x (cells of 256); `seen` gets the values it holds.
template<class KV> std::vector<KV> store(const std::vector<float>& x, std::vector<float>& seen) {
    std::vector<KV> s(x.size());
    seen.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) { s[i] = (KV) x[i]; seen[i] = (float) s[i]; }
    return s;
}
// int8: q8_0's grid per 32 values as kv8_store writes it (d = amax / 127, q = round-to-nearest-even
// of x / d); a block holding a NaN gets a NaN scale (the poisoned cells).
template<> std::vector<q36attn::q8_cell> store(const std::vector<float>& x, std::vector<float>& seen) {
    std::vector<q36attn::q8_cell> s(x.size() / 256);
    seen.resize(x.size());
    for (size_t c = 0; c < s.size(); ++c)
        for (int b = 0; b < 8; ++b) {
            const float* v = x.data() + c * 256 + b * 32;
            float a = 0.f;
            bool nan = false;
            for (int j = 0; j < 32; ++j) { a = std::max(a, std::fabs(v[j])); nan = nan || std::isnan(v[j]); }
            const float d = nan ? std::numeric_limits<float>::quiet_NaN() : a / 127.f;
            s[c].d[b] = d;
            for (int j = 0; j < 32; ++j) {
                s[c].q[b * 32 + j] = (int8_t) (nan || !(d > 0.f) ? 0 : (int) std::nearbyint(v[j] / d));
                seen[c * 256 + b * 32 + j] = (float) s[c].q[b * 32 + j] * d;
            }
        }
    return s;
}
bool skip_x = false;   // below sm_80 the launcher never picks part_q8x: no comparison with it
bool small = false;    // `q36_attn_parity small`: without the 262144-cell cases (sanitizer runs)
bool q8_own = false;   // the int8 cache through part_q8 (what the engine runs), else part<q8_cell>
bool k8v4 = false;     // with q8_own: V as rotated q4_0 (q36attn::q4_cell, --kv k8v4), quantized by kv4_store on the GPU
template<class KV> const char* name() {
    return std::is_same<KV, float>::value ? "F32" : std::is_same<KV, __half>::value ? "F16"
         : k8v4 ? "K8V4 part_q8" : q8_own ? "Q8 part_q8" : "Q8";
}
// --kv k8v4's store: cell c of x (256 floats) into q4 cell c, as k_attn_prep does it. Grid (cells), 256 threads.
__global__ void store_v4(const float* x, q36attn::q4_cell* out) {
    __shared__ float sh[256];
    q36attn::kv4_store(reinterpret_cast<uint8_t*>(out + blockIdx.x), threadIdx.x, x[(size_t) blockIdx.x * 256 + threadIdx.x], sh);
}
// The orthonormal Walsh-Hadamard transform of 256 values (its own inverse).
void wht256(const double* in, double* out) {
    double v[256];
    for (int i = 0; i < 256; ++i) v[i] = in[i];
    for (int h = 1; h < 256; h *= 2)
        for (int i = 0; i < 256; ++i)
            if (!(i & h)) { const double a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
    for (int i = 0; i < 256; ++i) out[i] = v[i] / 16;
}
// The GPU-quantized k8v4 V of x: the device buffer, and in `seen` the (rotated) values it holds.
float* v4_cache(const std::vector<float>& x, std::vector<float>& seen) {
    const size_t cells = x.size() / 256;
    float* dx = upload(x);
    q36attn::q4_cell* d;
    CK(cudaMalloc(&d, cells * sizeof(q36attn::q4_cell)));
    store_v4<<<(unsigned) cells, 256>>>(dx, d);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(dx));
    const auto h = read(d, cells);
    seen.resize(x.size());
    for (size_t c = 0; c < cells; ++c)
        for (int i = 0; i < 256; ++i) {
            const int b = i / 32, j = i % 32;
            const uint8_t byte = h[c].qs[16 * b + (j & 15)];
            seen[c * 256 + i] = (float) ((j < 16 ? byte & 15 : byte >> 4) - 8) * __half2float(h[c].d[b]);
        }
    return reinterpret_cast<float*>(d);
}

template<class KV> void run(int ctx, int group, int tokens, bool sessions, cudaStream_t stream) {
    constexpr int D = 256, KVH = 2;
    constexpr bool q8 = std::is_same<KV, q36attn::q8_cell>::value;
    const int H = KVH * group, chunks = (ctx + 255) / 256, slots = sessions ? 2 : 1;
    const int t0 = sessions ? 1 : 0, nt = tokens - t0;
    const size_t cache = (size_t) KVH * ctx * D;
    // per slot: the FP32 values, and the values the cache holds
    std::vector<std::vector<float>> keys(slots), values(slots), kseen(slots), vseen(slots);
    std::vector<float*> kp, vp;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int s = 0; s < slots; ++s) {
        keys[s].resize(cache); values[s].resize(cache);
        for (size_t i = 0; i < cache; ++i) {
            const bool poison = sessions && s == 0 && (int) (i / D % ctx) > 128;
            keys[s][i] = poison ? nan : value(i, 17 + s);
            values[s][i] = poison ? nan : value(i, 31 + s);
        }
        kp.push_back(reinterpret_cast<float*>(upload(store<KV>(keys[s], kseen[s]))));
        if (q8 && k8v4) vp.push_back(v4_cache(values[s], vseen[s]));
        else vp.push_back(reinterpret_cast<float*>(upload(store<KV>(values[s], vseen[s]))));
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
    if constexpr (q8) {
        if (q8_own && k8v4) {
            q36attn::part_q8<<<dim3(chunks, nt * KVH), 256, 0, stream>>>(dq,
                reinterpret_cast<const KV*>(kp[0]), reinterpret_cast<const q36attn::q4_cell*>(vp[0]), dp,
                t0, H, KVH, ctx, 1.f / 16, chunks, acc, mx, den, rows);
        } else if (q8_own) {
            q36attn::part_q8<<<dim3(chunks, nt * KVH), 256, 0, stream>>>(dq,
                reinterpret_cast<const KV*>(kp[0]), reinterpret_cast<const KV*>(vp[0]), dp,
                t0, H, KVH, ctx, 1.f / 16, chunks, acc, mx, den, rows);
        }
    }
    if (!q8 || !q8_own)
        q36attn::part<<<dim3(chunks, nt * KVH), 256, 0, stream>>>(dq,
            reinterpret_cast<const KV*>(kp[0]), reinterpret_cast<const KV*>(vp[0]), dp,
            t0, H, KVH, ctx, 1.f / 16, chunks, acc, mx, den, rows);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    if constexpr (q8) {
        if (q8_own && !skip_x) {
            // part_q8x (the engine's kernel: cache tiles through shared memory, R window rows per block) against part_q8 above:
            // every partial it writes must have the same bits, for each cap on the rows per block; the last launch stays
            // in the buffers for the checks below (the default cap)
            const auto a0 = read(acc, parts * D), m0 = read(mx, parts), l0 = read(den, parts);
            const int caps[] = {1, 2, 4};
            for (int ci = 0; ci < 3; ++ci) {
                CK(cudaMemset(acc, 0xFF, parts * D * sizeof(float)));
                CK(cudaMemset(mx, 0xFF, parts * sizeof(float))); CK(cudaMemset(den, 0xFF, parts * sizeof(float)));
                if (k8v4)
                    q36attn::launch_part_q8x(dq, reinterpret_cast<const KV*>(kp[0]), reinterpret_cast<const q36attn::q4_cell*>(vp[0]),
                                             dp, t0, nt, H, KVH, ctx, 1.f / 16, chunks, chunks, acc, mx, den, rows, sessions, stream,
                                             caps[ci]);
                else
                    q36attn::launch_part_q8x(dq, reinterpret_cast<const KV*>(kp[0]), reinterpret_cast<const KV*>(vp[0]), dp, t0,
                                             nt, H, KVH, ctx, 1.f / 16, chunks, chunks, acc, mx, den, rows, sessions, stream,
                                             caps[ci]);
                CK(cudaGetLastError());
                CK(cudaStreamSynchronize(stream));
                const auto a1 = read(acc, parts * D), m1 = read(mx, parts), l1 = read(den, parts);
                size_t bad = 0, cmp = 0;
                for (int r = 0; r < nt; ++r) for (int h = 0; h < H; ++h) {
                    const int nn = (sessions ? cells[t0 + r] : pos + t0 + r) + 1, nc = (nn + 255) / 256;
                    for (int c = 0; c < nc; ++c) {
                        const size_t o = (size_t) (r * H + h) * chunks + c;
                        bad += std::memcmp(&a0[o * D], &a1[o * D], D * sizeof(float)) != 0;
                        bad += std::memcmp(&m0[o], &m1[o], sizeof(float)) != 0;
                        bad += std::memcmp(&l0[o], &l1[o], sizeof(float)) != 0;
                        ++cmp;
                    }
                }
                std::printf("  part_q8x (rows cap %d) vs part_q8: %zu partials, %zu differ\n", caps[ci], cmp, bad);
                if (bad) std::exit(1);
            }
        }
    }
    // q36attn::merge on the partials, gate 0 (a factor of 1/2): compared with the oracle below
    float *gate, *merged;
    CK(cudaMalloc(&gate, (size_t) tokens * H * D * sizeof(float))); CK(cudaMemset(gate, 0, (size_t) tokens * H * D * sizeof(float)));
    CK(cudaMalloc(&merged, (size_t) tokens * H * D * sizeof(float)));
    q36attn::merge<<<dim3(nt * H, D / 32), 256, 0, stream>>>(acc, mx, den, gate, dp, t0, H, chunks, merged, rows);
    CK(cudaStreamSynchronize(stream));
    const auto a = read(acc, parts * D), m = read(mx, parts), l = read(den, parts);
    const auto mo = read(merged, (size_t) tokens * H * D);
    // k8v4: the engine's decode tail, merge without the gate then unrotate_gate (gate 0: a factor of 1/2)
    std::vector<float> uo;
    if (q8 && k8v4) {
        q36attn::merge<<<dim3(nt * H, D / 32), 256, 0, stream>>>(acc, mx, den, (const float*) nullptr, dp, t0, H, chunks,
                                                               merged, rows);
        q36attn::unrotate_gate<<<nt * H, 32, 0, stream>>>(merged + (size_t) t0 * H * D, gate + (size_t) t0 * H * D);
        CK(cudaStreamSynchronize(stream));
        uo = read(merged, (size_t) tokens * H * D);
    }
    double uerror = 0;
    double merror = 0;
    double error = 0, ferror = 0;
    int compared = 0;
    // the attention of query row t, head h over the caches K / V (slot s) into out
    auto oracle = [&](const std::vector<float>& K, const std::vector<float>& V, int t, int h, int n,
                      std::array<double, D>& out) {
        const size_t base = (size_t) (h / group) * ctx * D;
        std::vector<double> logits(n);
        double maximum = -INFINITY;
        for (int p = 0; p < n; ++p) {
            double dot = 0;
            for (int d = 0; d < D; ++d) dot += (double) q[((size_t) t * H + h) * D + d] * K[base + (size_t) p * D + d];
            logits[p] = dot / 16; maximum = std::max(maximum, logits[p]);
        }
        out.fill(0);
        double z = 0;
        for (int p = 0; p < n; ++p) {
            const double w = std::exp(logits[p] - maximum); z += w;
            for (int d = 0; d < D; ++d) out[d] += w * V[base + (size_t) p * D + d];
        }
        for (int d = 0; d < D; ++d) out[d] /= z;
    };
    for (int r = 0; r < nt; ++r) for (int h = 0; h < H; ++h) {
        // Four heads span both GQA groups in the large case; small cases cover all.
        if (ctx > 4096 && h != 0 && h != group - 1 && h != group && h != H - 1) continue;
        const int t = t0 + r, s = sessions ? slot[t] : 0;
        const int n = (sessions ? cells[t] : pos + t) + 1, nc = (n + 255) / 256;
        std::array<double, D> want{}, fwant{}, got{};
        oracle(kseen[s], vseen[s], t, h, n, want);
        if (q8) oracle(keys[s], values[s], t, h, n, fwant);
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
            error = std::max(error, std::abs(v - want[d])); ++compared;
            merror = std::max(merror, std::abs(2.0 * mo[((size_t) t * H + h) * D + d] - want[d]));
            if (q8 && !k8v4) ferror = std::max(ferror, std::abs(v - fwant[d]));
        }
        if (q8 && k8v4) {   // rotated back: against the oracle's own rotation back, and against the FP32 values
            double wr[D], ww[D];
            for (int d = 0; d < D; ++d) ww[d] = want[d];
            wht256(ww, wr);
            for (int d = 0; d < D; ++d) {
                const double u = 2.0 * uo[((size_t) t * H + h) * D + d];
                uerror = std::max(uerror, std::abs(u - wr[d]));
                ferror = std::max(ferror, std::abs(u - fwant[d]));
            }
        }
    }
    std::printf("KV %s ctx%d G%d rows%d offset%d sessions%d: %d values, max error %.9g", name<KV>(), ctx, group, nt, t0,
                sessions, compared, error);
    if (q8) std::printf(", vs the FP32 values %.9g", ferror);
    std::printf(", merged %.9g", merror);
    if (q8 && k8v4) std::printf(", rotated back %.9g", uerror);
    std::printf("\n");
    if (error > 2.e-5 || merror > 2.e-5 || uerror > 2.e-5) std::exit(1);
    // k8v4 against FP32: the rotated values are ~N(0, 0.58), a q4_0 step is amax / 8 (~0.16 here)
    if (q8 && k8v4 && ferror > 0.1) std::exit(1);
    // int8 against FP32: values in [-1, 1), so a q8_0 step is <= 1/127 and its rounding <= 1/254 per value
    if (q8 && !k8v4 && ferror > 1.e-2) std::exit(1);
    for (float* p : kp) CK(cudaFree(p)); for (float* p : vp) CK(cudaFree(p));
    CK(cudaFree(gate)); CK(cudaFree(merged));
    CK(cudaFree(dq)); CK(cudaFree(dp)); CK(cudaFree(acc)); CK(cudaFree(mx)); CK(cudaFree(den));
    if (sessions) {
        CK(cudaFree((void*) rows.cell)); CK(cudaFree((void*) rows.slot));
        CK(cudaFree((void*) rows.kc)); CK(cudaFree((void*) rows.vc));
    }
}
int main(int argc, char** argv) {
    using q8 = q36attn::q8_cell;
    // ctest skips (SKIP_RETURN_CODE 77) without a device
    int ndev = 0, major = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::puts("SKIP: no GPU"); return 77; }
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    if (major < 8) { skip_x = true; std::printf("sm_%d0 (< sm_80): part_q8x is not compared (the launcher keeps part_q8)\n", major); }
    small = argc > 1 && std::strcmp(argv[1], "small") == 0;   // `small`: no 262144-cell cases (sanitizer runs)
    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    run<float>(5, 1, 1, false, stream); run<__half>(5, 1, 1, false, stream); run<q8>(5, 1, 1, false, stream);
    run<float>(257, 8, 3, true, stream); run<__half>(257, 8, 3, true, stream); run<q8>(257, 8, 3, true, stream);
    run<float>(4096, 8, 3, false, stream); run<__half>(4096, 8, 3, false, stream); run<q8>(4096, 8, 3, false, stream);
    // a partial last tile and chunk (n % 8 != 0, n % 256 != 0) with G = 8 and 8 query rows (a verify window)
    run<float>(1000, 8, 8, false, stream); run<q8>(1000, 8, 8, false, stream);
    if (!small) { run<float>(262144, 8, 1, false, stream); run<__half>(262144, 8, 1, false, stream); run<q8>(262144, 8, 1, false, stream); }
    q8_own = true;   // the engine's int8 kernel, the same cases, plus groups below 8
    run<q8>(5, 1, 1, false, stream); run<q8>(257, 8, 3, true, stream); run<q8>(4096, 8, 3, false, stream);
    run<q8>(1000, 8, 8, false, stream); run<q8>(4096, 4, 3, false, stream); run<q8>(257, 2, 3, true, stream);
    // windows whose rows cross a 256-cell chunk boundary (row 0 on the last cells of chunk 0, the others in chunk 1), and the
    // 4 + remainder splits of 5, 6 and 7 rows (pos = ctx - tokens)
    run<q8>(257, 8, 3, false, stream); run<q8>(258, 8, 4, false, stream); run<q8>(259, 8, 5, false, stream);
    run<q8>(260, 8, 6, false, stream); run<q8>(261, 8, 7, false, stream); run<q8>(264, 8, 8, false, stream);
    run<q8>(257, 8, 2, false, stream); run<q8>(513, 8, 3, false, stream); run<q8>(515, 8, 5, false, stream);
    if (!small) run<q8>(262144, 8, 1, false, stream);
    k8v4 = true;   // --kv k8v4: int8 K, rotated q4_0 V stored by the GPU's kv4_store; the decode tail rotates back
    run<q8>(5, 1, 1, false, stream); run<q8>(257, 8, 3, true, stream); run<q8>(4096, 8, 3, false, stream);
    run<q8>(1000, 8, 8, false, stream); run<q8>(4096, 4, 3, false, stream); run<q8>(257, 2, 3, true, stream);
    run<q8>(257, 8, 3, false, stream); run<q8>(259, 8, 5, false, stream); run<q8>(261, 8, 7, false, stream);
    run<q8>(264, 8, 8, false, stream); run<q8>(515, 8, 5, false, stream);
    if (!small) run<q8>(262144, 8, 1, false, stream);
    CK(cudaStreamDestroy(stream)); std::puts("PASS");
}
