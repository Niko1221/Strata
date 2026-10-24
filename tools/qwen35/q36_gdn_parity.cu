// tools/qwen35/q36_gdn_parity.cu - src/qwen35/q36_gdn.cuh (the chunked gated delta rule for prompt chunks) against a
// double-precision oracle (the token-by-token recurrence), next to the engine's current kernel (k_gdn_seq_cols, copied
// below) on the same data: raw outputs and the state left for the next decode step, and the time of both for one layer.
// Synthetic conv outputs (q / k L2-normed per 128-head), no model. Sage's geometry: 16 K heads, 32 V heads, 128 x 128.
#include "../../src/qwen35/q36_gdn.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { const auto e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

constexpr int S = 128, HK = 16, HV = 32, C = 2 * HK * S + HV * S;

// ---- the engine's current kernel (q36_engine.cu, constexpr kGdnColGroups .. k_gdn_seq_cols), copied unchanged: keep it in step
//      with the engine (a diff of the two bodies is the check)
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}

static uint64_t rs = 88172645463325252ull;
static double rnd() {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double) (rs >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}
static double gauss() { return rnd() + rnd() + rnd(); }

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("q36_gdn_parity: no GPU, skipped\n"); return 77; }
    const bool quick = argc > 1 && std::atoi(argv[1]) == 1;
    std::vector<int> Ts = {9, 16, 17, 32, 33, 64, 300, 2048};
    if (quick) Ts = {9, 33, 300};
    const int Tmax = Ts.back();
    float *dh, *dg, *db, *dst, *dst2, *dor, *dor2, *Lw, *Pw, *cw;
    CK(cudaMalloc(&dh, (size_t) Tmax * C * 4)); CK(cudaMalloc(&dg, (size_t) Tmax * HV * 4)); CK(cudaMalloc(&db, (size_t) Tmax * HV * 4));
    const size_t sn = (size_t) S * HV * S;
    CK(cudaMalloc(&dst, sn * 4)); CK(cudaMalloc(&dst2, sn * 4));
    CK(cudaMalloc(&dor, (size_t) Tmax * HV * S * 4)); CK(cudaMalloc(&dor2, (size_t) Tmax * HV * S * 4));
    CK(cudaMalloc(&Lw, q36gdn::mat_floats(Tmax, HV) * 4)); CK(cudaMalloc(&Pw, q36gdn::mat_floats(Tmax, HV) * 4));
    CK(cudaMalloc(&cw, q36gdn::cum_floats(Tmax, HV) * 4));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int fails = 0;
    std::printf("%-26s | %-23s | %-23s | %-23s | %8s %8s %6s\n", "case", "chunked o (max/rms)", "old o (max/rms)", "state chunked / old (max)", "new ms", "old ms", "x");
    // variant 0: usual gates; 1: strong decay; 2: correlated keys (k = normalised k0 + 0.05 noise per head), beta ~ 0.95, gate ~ -0.01:
    // the triangular system (I + L) is then at its largest
    std::vector<std::vector<double>> k0(HK, std::vector<double>(S));
    for (auto& v : k0) for (auto& x : v) x = gauss();
    for (int variant = 0; variant < 3; ++variant)
    for (int T : Ts) {
        // data: q / k L2-normed per head, v ~ N(0, 1), gate = -(0.01..2), beta = sigmoid(N(0,1)); a non-zero start state
        std::vector<float> h((size_t) T * C), gate((size_t) T * HV), beta((size_t) T * HV), st0(sn);
        for (int t = 0; t < T; ++t) {
            for (int blk = 0; blk < 2 * HK + HV; ++blk) {
                float* p = &h[(size_t) t * C + blk * S];
                double n = 0;
                for (int i = 0; i < S; ++i) { p[i] = (float) gauss(); n += (double) p[i] * p[i]; }
                if (variant == 2 && blk >= HK && blk < 2 * HK) {
                    n = 0;
                    for (int i = 0; i < S; ++i) { p[i] = (float) (k0[blk - HK][i] + 0.05 * p[i]); n += (double) p[i] * p[i]; }
                }
                if (blk < 2 * HK) for (int i = 0; i < S; ++i) p[i] = (float) (p[i] / std::sqrt(n));
            }
            for (int v = 0; v < HV; ++v) {
                // variant 1: strong decay (gates down to -6) and tiny ones; variant 0: the usual range
                gate[(size_t) t * HV + v] = (float) -(variant == 2 ? 0.01 : variant ? std::exp(3.0 * rnd() * 2.0) * 0.05 : 0.01 + 1.99 * std::fabs(rnd()));
                beta[(size_t) t * HV + v] = (float) (variant == 2 ? 0.95 : 1.0 / (1.0 + std::exp(-gauss())));
            }
        }
        for (auto& x : st0) x = (float) (0.1 * gauss());
        CK(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dst, st0.data(), sn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dst2, st0.data(), sn * 4, cudaMemcpyHostToDevice));
        k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(dst, dh, C, dg, db, dor, HK, HV, T);
        CK(q36gdn::launch(dst2, dh, C, dg, db, dor2, HK, HV, T, Lw, Pw, cw, st));
        CK(cudaStreamSynchronize(st));
        std::vector<float> oo((size_t) T * HV * S), on(oo.size()), so(sn), sc(sn);
        CK(cudaMemcpy(oo.data(), dor, oo.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(on.data(), dor2, on.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(so.data(), dst, sn * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sc.data(), dst2, sn * 4, cudaMemcpyDeviceToHost));
        // oracle on 4 heads, double
        double eo_new = 0, eo_old = 0, ss_new = 0, ss_old = 0, es_new = 0, es_old = 0, ref_max = 0, sref_max = 0;
        size_t cnt = 0, scnt = 0;
        for (int hv : {0, 1, 17, 31}) {
            const int qh = hv % HK;
            std::vector<double> W((size_t) S * S);
            for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) W[i * S + j] = st0[((size_t) i * HV + hv) * S + j];
            std::vector<double> kv(S), dl(S);
            for (int t = 0; t < T; ++t) {
                const float* ht = &h[(size_t) t * C];
                const double g = std::exp((double) gate[(size_t) t * HV + hv]), b = beta[(size_t) t * HV + hv];
                for (int j = 0; j < S; ++j) { double a = 0; for (int i = 0; i < S; ++i) a += W[i * S + j] * ht[(HK + qh) * S + i]; kv[j] = a; }
                for (int j = 0; j < S; ++j) dl[j] = (ht[2 * HK * S + hv * S + j] - g * kv[j]) * b;
                for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) W[i * S + j] = g * W[i * S + j] + ht[(HK + qh) * S + i] * dl[j];
                for (int j = 0; j < S; ++j) {
                    double o = 0; for (int i = 0; i < S; ++i) o += W[i * S + j] * ht[qh * S + i];
                    o /= std::sqrt((double) S);
                    const size_t ix = (size_t) t * HV * S + hv * S + j;
                    const double dn = on[ix] - o, dold = oo[ix] - o;
                    eo_new = std::max(eo_new, std::fabs(dn)); eo_old = std::max(eo_old, std::fabs(dold));
                    ss_new += dn * dn; ss_old += dold * dold; ref_max = std::max(ref_max, std::fabs(o)); ++cnt;
                }
            }
            for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) {
                const size_t ix = ((size_t) i * HV + hv) * S + j;
                es_new = std::max(es_new, std::fabs(sc[ix] - W[i * S + j])); es_old = std::max(es_old, std::fabs(so[ix] - W[i * S + j]));
                sref_max = std::max(sref_max, std::fabs(W[i * S + j])); ++scnt;
            }
        }
        float tn = 0, to = 0;
        for (int r = 0; r < 8; ++r) {
            float ms;
            cudaEventRecord(e0, st);
            q36gdn::launch(dst2, dh, C, dg, db, dor2, HK, HV, T, Lw, Pw, cw, st);
            cudaEventRecord(e1, st); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (r >= 3) tn += ms / 5;
            cudaEventRecord(e0, st);
            k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(dst, dh, C, dg, db, dor, HK, HV, T);
            cudaEventRecord(e1, st); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (r >= 3) to += ms / 5;
        }
        // pass: the chunked error is within 3x of the current kernel's (plus a floor), outputs and state
        const bool ok = eo_new <= 3.0 * eo_old + 1e-5 * ref_max && es_new <= 3.0 * es_old + 1e-5 * sref_max
                        && std::sqrt(ss_new / cnt) <= 3.0 * std::sqrt(ss_old / cnt) + 1e-6 * ref_max;
        if (!ok) ++fails;
        std::printf("T=%-5d %-18s | %.2e / %.2e %s | %.2e / %.2e     | %.2e / %.2e        | %8.3f %8.3f %5.1fx\n", T, variant == 2 ? "correlated keys" : variant ? "strong decay" : "usual gates",
                    eo_new, std::sqrt(ss_new / cnt), ok ? "ok  " : "FAIL", eo_old, std::sqrt(ss_old / cnt), es_new, es_old, tn, to, to / tn);
    }
    std::printf(fails ? "q36_gdn_parity: %d FAILED\n" : "q36_gdn_parity: all ok\n", fails);
    return fails ? 1 : 0;
}
