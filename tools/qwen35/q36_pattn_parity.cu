// tools/qwen35/q36_pattn_parity.cu - src/qwen35/q36_pattn.cuh (the fused prompt attention from the int8 cache) against a
// double-precision oracle on sampled rows, next to the engine's current prompt path (FP16 copies of K / V, cuBLAS Q K^T,
// causal softmax to FP16, cuBLAS P V) on the same data, and the time of both for one attention layer. Synthetic cells,
// no model (the quick ctest case fills 2 x 20480 cells on the host; the full run 2 x 262144). Cases (T rows after p0 cached cells): small and large T, P up to 262144.
#include "../../src/qwen35/q36_pattn.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { const auto e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

using q36attn::q8_cell;
constexpr int HKV = 2, H = 16, HD = 256;

static uint64_t rs = 88172645463325252ull;
static float rnd() {   // uniform (-1, 1)
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (float) ((rs >> 11) * (1.0 / 9007199254740992.0)) * 2.f - 1.f;
}
static float gauss() { return (rnd() + rnd() + rnd()) * 1.0f; }   // sum of 3 uniforms: std 1

static void fill_cells(std::vector<q8_cell>& c, float amp) {
    std::vector<float> v(256);
    for (auto& cell : c) {
        for (auto& x : v) x = gauss() * amp;
        for (int b = 0; b < 8; ++b) {
            float a = 0.f;
            for (int j = 0; j < 32; ++j) a = std::max(a, std::fabs(v[b * 32 + j]));
            const float d = a / 127.f;
            cell.d[b] = d;
            for (int j = 0; j < 32; ++j) cell.q[b * 32 + j] = (int8_t) (d > 0.f ? (int) std::nearbyint(v[b * 32 + j] / d) : 0);
        }
    }
}

// ---- the engine's current path (q36_engine.cu: k_kv_f16, k_attn_softmax16, the cuBLAS slices)
__global__ void k_kv_f16(const q8_cell* kc, const q8_cell* vc, __half* k16, __half* v16, int ctx, int P) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD, kvh = i / per, rest = i % per;
    const size_t cell = (size_t) kvh * ctx + (size_t) (rest / HD);
    const int d = (int) (rest % HD);
    k16[i] = __float2half((float) kc[cell].q[d] * kc[cell].d[d >> 5]);
    v16[i] = __float2half((float) vc[cell].q[d] * vc[cell].d[d >> 5]);
}
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
__device__ float bmax(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = sh[0];
    for (int i = 1; i < (int) (blockDim.x >> 5); ++i) r = fmaxf(r, sh[i]);
    return r;
}
__device__ float bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) r += sh[i];
    return r;
}
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = bmax(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = bsum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}

struct Old {
    cublasHandle_t blas;
    size_t kAtt = (size_t) 64 << 20;
    float* att_s; __half *att_p16, *k16, *v16, *q16;
    void init(int ctx, int Tmax) {
        cublasCreate(&blas);
        CK(cudaMalloc(&att_s, kAtt * 4)); CK(cudaMalloc(&att_p16, kAtt * 2));
        CK(cudaMalloc(&k16, (size_t) HKV * ctx * HD * 2)); CK(cudaMalloc(&v16, (size_t) HKV * ctx * HD * 2));
        CK(cudaMalloc(&q16, (size_t) Tmax * H * HD * 2));
    }
    void run(const float* q, const q8_cell* kc, const q8_cell* vc, int ctx, int T, int p0, float* att, cudaStream_t st) {
        cublasSetStream(blas, st);
        const int P = p0 + T, ldp = (P + 7) / 8 * 8, G = H / HKV;
        const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAtt / ((size_t) H * ldp))));
        const float one = 1.f, zero = 0.f;
        k_kv_f16<<<(unsigned) (((int64_t) HKV * P * HD + 255) / 256), 256, 0, st>>>(kc, vc, k16, v16, ctx, P);
        k_to_f16<<<(unsigned) (((int64_t) T * H * HD + 255) / 256), 256, 0, st>>>(q, q16, (int64_t) T * H * HD);
        for (int t0 = 0; t0 < T; t0 += tq) {
            const int nt = std::min(tq, T - t0);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, k16 + (size_t) kvh * P * HD, CUDA_R_16F, HD,
                                           0, q16 + ((size_t) t0 * H + kvh * G) * HD, CUDA_R_16F, H * HD, HD, &zero,
                                           att_s + (size_t) kvh * G * nt * ldp, CUDA_R_32F, ldp, (long long) nt * ldp, G,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            }
            k_attn_softmax16<<<dim3(nt, H), 256, 0, st>>>(att_s, att_p16, ldp, nt, P, p0 + t0, 1.0f / sqrtf((float) HD));
            for (int kvh = 0; kvh < HKV; ++kvh) {
                cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, v16 + (size_t) kvh * P * HD, CUDA_R_16F, HD,
                                           0, att_p16 + (size_t) kvh * G * nt * ldp, CUDA_R_16F, ldp, (long long) nt * ldp, &zero,
                                           att + ((size_t) t0 * H + kvh * G) * HD, CUDA_R_32F, H * HD, HD, G,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            }
        }
    }
};

struct Err { double max_abs = 0, rms = 0, ref_max = 0; };
static Err compare(const std::vector<float>& got, const std::vector<std::vector<double>>& ref, const std::vector<std::pair<int, int>>& rows) {
    Err e;
    double ss = 0; size_t n = 0;
    for (size_t k = 0; k < rows.size(); ++k) {
        const size_t base = ((size_t) rows[k].first * H + rows[k].second) * HD;
        for (int d = 0; d < HD; ++d) {
            const double df = got[base + d] - ref[k][d];
            e.max_abs = std::max(e.max_abs, std::fabs(df));
            e.ref_max = std::max(e.ref_max, std::fabs(ref[k][d]));
            ss += df * df; ++n;
        }
    }
    e.rms = std::sqrt(ss / n);
    return e;
}

int main(int argc, char** argv) {
    int ndev = 0, major = 0, n_sm = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("q36_pattn_parity: no GPU, skipped\n"); return 77; }
    CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    if (major < 8) { std::printf("q36_pattn_parity: needs sm_80+, skipped\n"); return 77; }
    CK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0));
    const bool quick = argc > 1 && std::atoi(argv[1]) == 1;   // ctest: small cases only
    const int CTX = quick ? 20480 : 262144;
    struct Case { int T, p0; float qamp, kamp; const char* what; };
    std::vector<Case> cases = {
        {9, 0, 1.f, 1.f, "first chunk, 9 rows"},        {16, 1000, 1.f, 1.f, "16 rows"},
        {52, 19000, 1.f, 1.f, "52 rows, 19K ctx"},      {300, 19000, 1.f, 1.f, "300 rows"},
        {2048, 0, 1.f, 1.f, "first 2048 chunk"},        {2048, 14336, 1.f, 1.f, "2048 rows at 14K"},
        {300, 19000, 4.f, 2.f, "peaky softmax (scores x8)"}, {37, 4091, 1.f, 1.f, "odd T and p0"},
    };
    if (!quick) {
        cases.push_back({9, 262144 - 9, 1.f, 1.f, "9 rows at 262K"});
        cases.push_back({52, 262144 - 52, 1.f, 1.f, "52 rows at 262K"});
        cases.push_back({300, 262144 - 300, 1.f, 1.f, "300 rows at 262K"});
        cases.push_back({2048, 40960, 1.f, 1.f, "2048 rows at 40K"});
        cases.push_back({2048, 262144 - 2048, 1.f, 1.f, "2048 rows at 262K"});
    }
    int Tmax = 0;
    for (auto& c : cases) Tmax = std::max(Tmax, c.T);
    std::vector<q8_cell> hk((size_t) HKV * CTX), hv((size_t) HKV * CTX);
    fill_cells(hk, 1.f); fill_cells(hv, 1.f);
    q8_cell *dk, *dv;
    CK(cudaMalloc(&dk, hk.size() * sizeof(q8_cell))); CK(cudaMalloc(&dv, hv.size() * sizeof(q8_cell)));
    CK(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(q8_cell), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), hv.size() * sizeof(q8_cell), cudaMemcpyHostToDevice));
    float *dq, *dof, *dop, *pacc, *pm, *pl;
    CK(cudaMalloc(&dq, (size_t) Tmax * H * HD * 4)); CK(cudaMalloc(&dof, (size_t) Tmax * H * HD * 4)); CK(cudaMalloc(&dop, (size_t) Tmax * H * HD * 4));
    CK(cudaMalloc(&pacc, q36pattn::partial_acc_floats() * 4)); CK(cudaMalloc(&pm, q36pattn::kPartialRows * 4)); CK(cudaMalloc(&pl, q36pattn::kPartialRows * 4));
    Old old; old.init(CTX, Tmax);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int fails = 0;
    std::printf("%-30s %7s %8s | %-22s | %-22s | %9s %9s %6s\n", "case", "T", "p0", "fused err (max/rms)", "old err (max/rms)", "fused ms", "old ms", "x");
    for (const Case& c : cases) {
        if (c.p0 + c.T > CTX) continue;
        std::vector<float> hq((size_t) c.T * H * HD);
        for (auto& x : hq) x = gauss() * c.qamp;
        if (c.kamp != 1.f) { fill_cells(hk, c.kamp); CK(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(q8_cell), cudaMemcpyHostToDevice)); }
        CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
        // fused
        CK(q36pattn::launch(dq, dk, dv, CTX, H, HKV, c.T, c.p0, dof, pacc, pm, pl, n_sm, st));
        CK(cudaStreamSynchronize(st));
        old.run(dq, dk, dv, CTX, c.T, c.p0, dop, st);
        CK(cudaStreamSynchronize(st));
        std::vector<float> gf((size_t) c.T * H * HD), go(gf.size());
        CK(cudaMemcpy(gf.data(), dof, gf.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(go.data(), dop, go.size() * 4, cudaMemcpyDeviceToHost));
        // oracle on sampled rows (double, the cells' dequantised values, Q as given in FP32)
        std::vector<std::pair<int, int>> rows = {{0, 0}, {c.T - 1, H - 1}, {c.T / 2, 3}, {std::min(c.T - 1, 5), 9}, {c.T - 1, 0}};
        for (int k = 1; k < 8 && k < c.T; ++k) rows.push_back({c.T - 1 - k, (k * 5 + 1) % H});   // the last rows (the token-pair split at the end)
        std::vector<std::vector<double>> ref;
        for (auto& rw : rows) {
            const int t = rw.first, h = rw.second, kvh = h / (H / HKV), n = c.p0 + t + 1;
            std::vector<double> sc(n);
            double mx = -1e300;
            for (int p = 0; p < n; ++p) {
                const q8_cell& k = hk[(size_t) kvh * CTX + p];
                double s = 0;
                for (int d = 0; d < HD; ++d) s += (double) hq[((size_t) t * H + h) * HD + d] * ((double) k.q[d] * k.d[d >> 5]);
                sc[p] = s / 16.0; mx = std::max(mx, sc[p]);
            }
            double sum = 0;
            std::vector<double> o(HD, 0.0);
            for (int p = 0; p < n; ++p) {
                const double e = std::exp(sc[p] - mx);
                sum += e;
                const q8_cell& v = hv[(size_t) kvh * CTX + p];
                for (int d = 0; d < HD; ++d) o[d] += e * ((double) v.q[d] * v.d[d >> 5]);
            }
            for (auto& x : o) x /= sum;
            ref.push_back(o);
        }
        const Err ef = compare(gf, ref, rows), eo = compare(go, ref, rows);
        // time (3 warm-up, 5 timed)
        float tf = 0, to = 0;
        for (int r = 0; r < 8; ++r) {
            cudaEventRecord(e0, st);
            q36pattn::launch(dq, dk, dv, CTX, H, HKV, c.T, c.p0, dof, pacc, pm, pl, n_sm, st);
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (r >= 3) tf += ms / 5;
            cudaEventRecord(e0, st);
            old.run(dq, dk, dv, CTX, c.T, c.p0, dop, st);
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1); if (r >= 3) to += ms / 5;
        }
        // pass: the fused error is within 2x of the current path's (plus a floor), both against the oracle
        const bool ok = ef.max_abs <= 2.0 * eo.max_abs + 1e-3 * ef.ref_max && ef.rms <= 2.0 * eo.rms + 1e-4 * ef.ref_max;
        if (!ok) ++fails;
        std::printf("%-30s %7d %8d | %.2e / %.2e %s | %.2e / %.2e    | %9.3f %9.3f %5.2fx\n", c.what, c.T, c.p0, ef.max_abs, ef.rms,
                    ok ? "ok  " : "FAIL", eo.max_abs, eo.rms, tf, to, to / tf);
        if (c.kamp != 1.f) { fill_cells(hk, 1.f); CK(cudaMemcpy(dk, hk.data(), hk.size() * sizeof(q8_cell), cudaMemcpyHostToDevice)); }
    }
    std::printf(fails ? "q36_pattn_parity: %d FAILED\n" : "q36_pattn_parity: all ok\n", fails);
    return fails ? 1 : 0;
}
