// The 27B's prompt path (Engine::prefill for FNX_MODEL=27b): a prompt's rows in chunks of up to FNX_PF_ROWS (default
// 4096), as llama.cpp's CUDA backend runs a batch on a V100: the dense matrices dequantized to f16 and multiplied with
// cuBLAS on the tensor cores (f16 x f16, fp32 accumulation), the f32 alpha/beta rows as f32 sgemm, the recurrent core
// through gdn_prompt. Attention over a chunk: each KV head's positions gathered from the pages into one block, then per
// query head S = q k^T (cuBLAS), the causal softmax (scaled, f16 probabilities), P v (cuBLAS), the gate; the queries in
// blocks small enough that S stays within FNX_PF_SCORE_MIB. The sums across the cards go round the ring on the copy
// engines (pf_ring_plan.hpp's steps, pf_common.cuh's pf_ring_run), f16 rows.
// What a chunk leaves is the state a run of windows over the same rows leaves: K/V, recurrent states and histories,
// and with the draft layer attached its K/V (the draft layer over the chunk's final rows, each paired with the token
// after it, up to its K/V).
#include "pf_common.cuh"
#include "engine.hpp"
#include "env.hpp"
#include "pf_ring_plan.hpp"
#include "s27_impl.hpp"

#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace fnx {

namespace {
// row t: R[t] += pend[t] (when pend), y = rms(R[t]) * w as f32 (yf, when not null) and f16 (yh); the decode path's add_norm
__global__ void k_pf_add_norm(float* R, const float* pend, const float* w, float* yf, __half* yh, float eps) {
    __shared__ float sh[32];
    float* r = R + (size_t) blockIdx.x * N;
    float q = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = r[i];
        if (pend) r[i] = x += pend[(size_t) blockIdx.x * N + i];
        q += x * x;
    }
    const float s = rsqrtf(s27_block_sum(q, sh) / (float) N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = r[i] * s * w[i];
        if (yf) yf[(size_t) blockIdx.x * N + i] = v;
        yh[(size_t) blockIdx.x * N + i] = __float2half(v);
    }
}
// [rms(e[t]) * we | rms(h[t]) * wh] as f16 rows of 2N: the draft layer's input (as k_norm_cat, then to f16)
__global__ void k_pf_norm_cat(const float* e, const float* h, const float* we, const float* wh, __half* out, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x[2] = {e + (size_t) t * N, h + (size_t) t * N};
    const float* w[2] = {we, wh};
    for (int s = 0; s < 2; ++s) {
        float q = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) q += x[s][i] * x[s][i];
        const float r = rsqrtf(s27_block_sum(q, sh) / (float) N + eps);
        for (int i = threadIdx.x; i < N; i += blockDim.x) out[(size_t) t * 2 * N + s * N + i] = __float2half(x[s][i] * r * w[s][i]);
    }
}
// h[t][j] = f16(silu(gate[t][j]) * up[t][j]) from the stacked [gate | up] rows of 2 ffs (swiglu_stacked's values)
__global__ void k_pf_swiglu(const float* __restrict__ gu, int ffs, int C, __half* __restrict__ h) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) C * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t t = e / ffs, j = e % ffs;
        const float g = gu[t * 2 * ffs + j], u = gu[t * 2 * ffs + ffs + j];
        h[e] = __float2half(g / (1.f + __expf(-g)) * u);
    }
}
// a KV head's K and V for positions [0, P) from the pages into contiguous f16 blocks [P][HD]
__global__ void k_pf_gather(const __half* __restrict__ pool, const int* __restrict__ pt, int slot, int kvh, int P, __half* __restrict__ k,
                            __half* __restrict__ v) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) P * HD / 8; e += (size_t) gridDim.x * blockDim.x) {
        const int p = (int) (e / (HD / 8)), d8 = (int) (e % (HD / 8));
        const size_t at = s27_k(pt, slot, p) + (size_t) kvh * HD + d8 * 8;
        reinterpret_cast<uint4*>(k)[e] = *reinterpret_cast<const uint4*>(pool + at);
        reinterpret_cast<uint4*>(v)[e] = *reinterpret_cast<const uint4*>(pool + at + (size_t) S27_PP * S27_ROWH);
    }
}
// rows [r0, r0 + n) of query head h (normed and rotated in qkv by k_s27_prep) as f16 [n][HD]
__global__ void k_pf_q16(const float* __restrict__ qkv, int h, int r0, int n, __half* __restrict__ q) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) n * HD; e += (size_t) gridDim.x * blockDim.x)
        q[e] = __float2half(qkv[(size_t) (r0 + e / HD) * S27_QKV + h * 2 * HD + e % HD]);
}
// row t of S ([n][P] f32, ld P): positions past the row's own (p0 + t) masked; softmax of S / sqrt(HD) as f16 into prob
__global__ void k_pf_softmax(const float* __restrict__ S, int P, int p0, __half* __restrict__ prob) {
    __shared__ float sh[32];
    const int t = blockIdx.x, len = p0 + t + 1;
    const float* s = S + (size_t) t * P;
    const float scale = rsqrtf((float) HD);
    float m = -INFINITY;
    for (int p = threadIdx.x; p < len; p += blockDim.x) m = fmaxf(m, s[p] * scale);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    m = -INFINITY;
    for (int w = 0; w < (int) (blockDim.x >> 5); ++w) m = fmaxf(m, sh[w]);
    __syncthreads();
    float z = 0.f;
    for (int p = threadIdx.x; p < len; p += blockDim.x) z += __expf(s[p] * scale - m);
    z = s27_block_sum(z, sh);
    const float iz = 1.f / z;
    __half* o = prob + (size_t) t * P;
    for (int p = threadIdx.x; p < P; p += blockDim.x) o[p] = __float2half(p < len ? __expf(s[p] * scale - m) * iz : 0.f);
}
// the same over f16 scores already scaled by 1 / sqrt(HD), in place: each thread's running max and sum over its
// share of the row in one read (online), the block's combined, then the probabilities written over the scores
__device__ __forceinline__ void pf_ml_merge(float& m, float& l, float m2, float l2) {
    const float mm = fmaxf(m, m2);
    l = (m == -INFINITY ? 0.f : l * __expf(m - mm)) + (m2 == -INFINITY ? 0.f : l2 * __expf(m2 - mm));
    m = mm;
}
__global__ void k_pf_softmax16(__half* __restrict__ S, int P, int p0) {
    __shared__ float sm[32], sl[32];
    const int t = blockIdx.x, len = p0 + t + 1;
    __half* s = S + (size_t) t * P;
    float m = -INFINITY, l = 0.f;
    for (int p = threadIdx.x; p < len; p += blockDim.x) {
        const float x = __half2float(s[p]);
        if (x > m) {
            l = l * __expf(m - x) + 1.f;
            m = x;
        } else {
            l += __expf(x - m);
        }
    }
    for (int o = 16; o; o >>= 1) pf_ml_merge(m, l, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, l, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m, sl[threadIdx.x >> 5] = l;
    __syncthreads();
    m = -INFINITY, l = 0.f;
    for (int w = 0; w < (int) (blockDim.x >> 5); ++w) pf_ml_merge(m, l, sm[w], sl[w]);
    const float il = 1.f / l;
    for (int p = threadIdx.x; p < P; p += blockDim.x) s[p] = __float2half(p < len ? __expf(__half2float(s[p]) - m) * il : 0.f);
}
// att[r0 + t][h] = O[t] * sigmoid(gate) as f16, rows of S27_NQ * HD (wo's input)
__global__ void k_pf_gate(const float* __restrict__ O, const float* __restrict__ qkv, int h, int r0, int n, __half* __restrict__ att) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) n * HD; e += (size_t) gridDim.x * blockDim.x) {
        const size_t t = r0 + e / HD, d = e % HD;
        const float g = qkv[t * S27_QKV + h * 2 * HD + HD + d];
        att[t * S27_NQ * HD + h * HD + d] = __float2half(O[e] / (1.f + __expf(-g)));
    }
}
constexpr unsigned PF_SEQ_RESET = 1u << 31;
constexpr int PF_K = 2;   // chunks a pass: one's sums cross the links while the other computes
}  // namespace

struct Engine::Pf {
    int dev = 0;
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    __half* wh = nullptr;          // a matrix as f16
    size_t wh_halves = 0;
    // each chunk of a pass: its residual, a half's sum, its (position, tag)s; the chunk running points at its own
    float *Rk[PF_K] = {}, *bok[PF_K] = {};
    int* ctlk[PF_K] = {};
    float *R = nullptr, *bo = nullptr, *xf = nullptr;   // the running chunk's residual and sum; the normed rows (f32)
    __half* xh = nullptr;          // f16 inputs, up to rows of 2 N
    float* big = nullptr;          // qkvz, qkv, [gate | up] or the draft layer's input: the largest
    float *ab = nullptr, *gdn_xc = nullptr, *gdn_sq = nullptr, *gdn_y = nullptr, *gdn_ch = nullptr;
    __half *gu16 = nullptr;        // the FFN's swiglu as f16
    // attention: a KV head's positions gathered, a query block, its scores, probabilities and output
    __half *kc = nullptr, *vc = nullptr, *q16 = nullptr, *prob = nullptr;
    float *score = nullptr, *o = nullptr;
    size_t score_floats = 0;
    int* ctl = nullptr;            // [1 + C / MAXT][2]: the chunk's (position, tag), then each group's
    // the sums run on rst: ev_go[j] chunk j's half ready for its sum, ev_done[j] its sum finished (the chunk's next half
    // waits for that alone, while the pass's other chunk computes)
    cudaStream_t rst = nullptr;
    cudaEvent_t ev_go[PF_K] = {}, ev_done[PF_K] = {};
    // the ring: a card's share of rows to send, a receive slot per reduce hop and set, the gathered rows per set,
    // flags [2 sets][PF_RING_STEPS] set by the previous card; n*: the next card's
    __half *send = nullptr, *in[NCARD - 1][2] = {}, *ag[2] = {};
    __half *nin[NCARD - 1][2] = {}, *nag[2] = {};
    unsigned *flags = nullptr, *nflags = nullptr;
    int next_dev = 0;
    std::vector<PfRingOp> plan;
    unsigned seq = 0;
};

namespace {
// out[t * ldo + r] = W[r] . X[t] for t < C: W f16 [rows][cols], X f16 [C][cols]
void pf_gemm(Engine::Pf& p, const __half* W, int rows, int cols, const __half* X, int C, float* out, int ldo) {
    const float one = 1.0f, zero = 0.0f;
    ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, out, CUDA_R_32F, ldo,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "gemm");
}
// out = W . X for a Q8_0 matrix (the split layout, dequantized to f16 first)
void pf_dense(Engine::Pf& p, const DW& w, const __half* X, int C, float* out) {
    if (!w.s.q) throw std::runtime_error("prefill: a dense matrix without the split Q8_0 layout");
    const size_t n = (size_t) w.n_out * w.n_in;
    if (n > p.wh_halves) throw std::runtime_error("prefill: a matrix larger than the f16 staging");
    k_q8s_f16<<<grid_for(n), 256, 0, p.st>>>(w.s.q, w.s.d, n, p.wh);
    pf_gemm(p, p.wh, w.n_out, w.n_in, X, C, out, w.n_out);
}
// x (C rows of N on card g) summed over the cards, one way round the ring by r.plan (pf_common.cuh's pf_ring_run);
// this ring takes its sequence number from itself and runs on its copy-engine stream r.rst
void pf_ring(Engine::Pf& r, int g, float* x, int C, unsigned* err) {
    const unsigned seq = ++r.seq;
    pf_ring_run(r, r.dev, g, x, C, seq, err, r.rst);
    ckc(cudaGetLastError(), "ring");
}
}  // namespace

void Engine::prefill_alloc() {
    const size_t C = (size_t) pf_c_;
    // FNX_PF_SCORE_MIB: the attention scores' room a card (default 256: query blocks of 512 rows at 128k positions)
    const long long score_mib = env_int("FNX_PF_SCORE_MIB", 256);
    if (score_mib < 16) throw std::runtime_error("FNX_PF_SCORE_MIB: want 16 MiB or more, got " + std::to_string(score_mib));
    if (((size_t) score_mib << 18) < (size_t) MAXT * (max_ctx_ + S27_PP))
        throw std::runtime_error("FNX_PF_SCORE_MIB: " + std::to_string(score_mib) + " MiB holds fewer than " + std::to_string(MAXT) +
                                 " queries' scores over the context of " + std::to_string(max_ctx_) + " positions");
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[(size_t) c];
        const CardW& w = m_.cards[c];
        ckc(cudaSetDevice(b.dev), "set");
        size_t free0 = 0, tot0 = 0;
        ckc(cudaMemGetInfo(&free0, &tot0), "mem info");
        pf_card_bytes = 0;   // palloc's out-of-memory message says what this card's prompt path holds
        Pf* p = new Pf();
        pf_.push_back(p);
        p->dev = b.dev;
        p->st = b.st;
        own([p] { delete p; });
        ckb(cublasCreate(&p->blas), "create");
        {
            const int d = current_card();
            cublasHandle_t h = p->blas;
            own([d, h] { cudaSetDevice(d); cublasDestroy(h); });
        }
        ckb(cublasSetStream(p->blas, b.st), "stream");
        ckb(cublasSetMathMode(p->blas, CUBLAS_TENSOR_OP_MATH), "math mode");
        size_t most = 0;
        auto see = [&](const DW& d) { most = std::max(most, (size_t) d.n_out * d.n_in); };
        for (const LayerW& L : w.L)
            for (const DW* d : {&L.qkvz, &L.ssm_out, &L.wq, &L.wo, &L.ffn_gu, &L.ffn_down}) see(*d);
        for (const DW* d : {&w.mtp.wq, &w.mtp_eh}) see(*d);
        p->wh_halves = most;
        p->wh = palloc<__half>(most);
        const int nv = w.nv(), ch = w.conv_ch(), ffs = w.fd1 - w.fd0;
        for (int k = 0; k < PF_K; ++k) {
            p->Rk[k] = palloc<float>(C * N);
            p->bok[k] = palloc<float>(C * N);
            p->ctlk[k] = palloc<int>((1 + C / MAXT) * 2);
            ckc(cudaEventCreateWithFlags(&p->ev_go[k], cudaEventDisableTiming), "event");
            own_event(p->ev_go[k]);
            ckc(cudaEventCreateWithFlags(&p->ev_done[k], cudaEventDisableTiming), "event");
            own_event(p->ev_done[k]);
        }
        ckc(cudaStreamCreateWithFlags(&p->rst, cudaStreamNonBlocking), "ring stream");
        own_stream(p->rst);
        p->xf = palloc<float>(C * N);
        p->xh = palloc<__half>(C * 2 * N);
        p->big = palloc<float>(C * std::max({(size_t) ch + (size_t) nv * HS, (size_t) S27_QKV, (size_t) 2 * ffs}));
        p->ab = palloc<float>(C * 2 * nv);
        p->gdn_xc = palloc<float>(C * ch);
        p->gdn_sq = palloc<float>(C * nv * 4);
        p->gdn_y = palloc<float>(C * nv * HS);
        p->gdn_ch = palloc<float>(gdn_prompt_work_floats((int) C, nv));
        p->gu16 = palloc<__half>(C * ffs);
        p->kc = palloc<__half>((size_t) (max_ctx_ + S27_PP) * HD);
        p->vc = palloc<__half>((size_t) (max_ctx_ + S27_PP) * HD);
        p->score_floats = (size_t) score_mib << 18;
        p->score = palloc<float>(p->score_floats);
        p->prob = palloc<__half>(p->score_floats);
        p->q16 = palloc<__half>(C * HD);
        p->o = palloc<float>(C * HD);
        const size_t share = (C + NCARD - 1) / NCARD * N;
        p->send = palloc<__half>(share);
        for (int k = 0; k < 2; ++k) {
            for (int h = 0; h < NCARD - 1; ++h) p->in[h][k] = palloc<__half>(share);
            p->ag[k] = palloc<__half>(C * N);
        }
        p->flags = palloc<unsigned>(2 * PF_RING_STEPS);
        ckc(cudaMemset(p->flags, 0, 2 * PF_RING_STEPS * sizeof(unsigned)), "ring flags");
        if (env_flag("FNX_PF_TIMING"))
            std::fprintf(stderr, "prefill buffers on card %d: %zu MiB (%zu MiB free before)\n", c, pf_card_bytes >> 20, free0 >> 20);
    }
    for (int c = 0; c < NCARD; ++c) {
        Pf& p = *pf_[(size_t) c];
        const Pf& n = *pf_[(size_t) (c + 1) % NCARD];
        for (int k = 0; k < 2; ++k) {
            for (int h = 0; h < NCARD - 1; ++h) p.nin[h][k] = n.in[h][k];
            p.nag[k] = n.ag[k];
        }
        p.nflags = n.flags;
        p.next_dev = n.dev;
        p.plan = pf_ring_plan(NCARD, c);
    }
    for (int c = 0; c < NCARD; ++c) {
        ckc(cudaSetDevice(cb_[(size_t) c]->dev), "set");
        ckc(cudaDeviceSynchronize(), "prompt path buffers");
    }
}

void Engine::prefill(const std::vector<int>& tokens, int after) {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("prefill: a window is not committed");
    if (mtp_ && after < 0) throw std::runtime_error("prefill: the draft layer needs the token after the last row");
    const size_t nt = tokens.size();
    if (pos_ + (int) nt > max_ctx_) throw Refused("prefill: context full");
    // FNX_PF_WINDOWS=1 (a check): the prompt through decode windows of MAXT rows, the draft layer over each, instead
    if (env_flag("FNX_PF_WINDOWS")) {
        for (size_t i = 0; i < nt; i += MAXT) {
            const size_t e = std::min(nt, i + MAXT);
            const int T = (int) (e - i), p0 = pos_;
            verify(std::vector<int>(tokens.begin() + (long) i, tokens.begin() + (long) e));
            commit(T);
            if (mtp_) {
                std::vector<int> next(tokens.begin() + (long) i + 1, tokens.begin() + (long) e);
                next.push_back(e < nt ? tokens[e] : after);
                draft(T, true, 0, next, p0);
            }
            if (on_prefill_chunk && (e % (size_t) pf_c_ == 0 || e == nt)) on_prefill_chunk((int) e);
        }
        return;
    }
    if (pf_[0]->seq >= PF_SEQ_RESET) {   // every card idle (each chunk ends synchronized): the flags zeroed, numbers from 0
        for (Pf* p : pf_) {
            ckc(cudaSetDevice(p->dev), "set");
            ckc(cudaMemset(p->flags, 0, 2 * PF_RING_STEPS * sizeof(unsigned)), "ring flags");
            p->seq = 0;
        }
    }
    if (!pf_emb_[0]) {
        ckc(cudaHostAlloc(&pf_emb_[0], (size_t) (PF_K * pf_c_ + 1) * N * sizeof(float), cudaHostAllocPortable), "pinned rows");
        own_pinned(pf_emb_[0]);
        for (int k = 0; k < PF_K; ++k) {
            ckc(cudaHostAlloc(&pf_ctl_[k], (size_t) (1 + pf_c_ / MAXT) * 2 * sizeof(int), cudaHostAllocPortable), "pinned ctl");
            own_pinned(pf_ctl_[k]);
        }
    }
    static const bool timing = env_flag("FNX_PF_TIMING");
    // FNX_PF_SKIP=ring, attn or ring,attn with FNX_PF_SKIP_CONFIRM=1: leave the sums or the attention out. The rows come
    // out wrong and stay in the K/V and the recurrent states: a timing breakdown only, said on stderr when it is on
    static bool skip_ring = false, skip_attn = false;
    static const bool skip_read = [] {
        const char* v = std::getenv("FNX_PF_SKIP");
        if (!v) return true;
        const std::string s = v;
        for (size_t a = 0; a <= s.size();) {
            const size_t e = std::min(s.find(',', a), s.size());
            const std::string part = s.substr(a, e - a);
            if (part == "ring") skip_ring = true;
            else if (part == "attn") skip_attn = true;
            else throw std::runtime_error("FNX_PF_SKIP: ring, attn or ring,attn, not '" + s + "'");
            a = e + 1;
        }
        if (!env_flag("FNX_PF_SKIP_CONFIRM"))
            throw std::runtime_error("FNX_PF_SKIP=" + s + " makes every prompt read wrong: set FNX_PF_SKIP_CONFIRM=1 beside it to time with it");
        std::fprintf(stderr, "fnx: FNX_PF_SKIP=%s: the prompt path leaves out%s%s, its rows are wrong (timing only)\n", v,
                     skip_ring ? " the ring sums" : "", skip_attn ? " the attention" : "");
        return true;
    }();
    (void) skip_read;
    // FNX_PF_SCORES: the attention's scores as f16 (the default: scaled first, half the bytes through memory) or f32. On
    // ref/long64 f16 read 4-7% faster (32k 2,187 against 2,096 tok/s) and as close to llama.cpp: KL 2.65e-4 against 2.89e-4
    static const bool scores16 = [] {
        const char* v = std::getenv("FNX_PF_SCORES");
        if (v && std::strcmp(v, "f16") && std::strcmp(v, "f32")) throw std::runtime_error(std::string("FNX_PF_SCORES: f16 or f32, not '") + v + "'");
        return !v || !std::strcmp(v, "f16");
    }();
    const float eps = m_.eps;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ckc(cudaSetDevice(cb_[(size_t) c]->dev), "set");
            f(c, *cb_[(size_t) c], m_.cards[c], *pf_[(size_t) c]);
        }
    };
    // a position's token: the history's before the read, the prompt's after (h0 taken once: each pass adds its rows to
    // the history)
    const int h0 = (int) history_.size();
    const auto tok_at = [&](int q) -> int { return q < h0 ? history_[(size_t) q] : tokens[(size_t) (q - h0)]; };
    for (size_t i = 0; i < nt; i += (size_t) PF_K * pf_c_) {
        const auto t_pass = std::chrono::steady_clock::now();
        const int rows = (int) std::min(nt - i, (size_t) PF_K * pf_c_), nk = (rows + pf_c_ - 1) / pf_c_;
        int Cs[PF_K] = {}, ngs[PF_K] = {}, p0s[PF_K] = {};
        for (int j = 0; j < nk; ++j) {
            Cs[j] = std::min(pf_c_, rows - j * pf_c_);
            ngs[j] = (Cs[j] + MAXT - 1) / MAXT;
            p0s[j] = pos_ + j * pf_c_;
        }
        ensure_pages(pos_ + rows);
        // the pass's rows, then (for the draft layer) the token after its last row, embedded on the host
        for (int t = 0; t < rows; ++t) embed_token(tok_at(pos_ + t), pos_ + t, pf_emb_[0] + (size_t) t * N);
        if (mtp_) embed_token(i + rows < nt ? tokens[i + rows] : after, pos_ + rows, pf_emb_[0] + (size_t) rows * N);
        for (int j = 0; j < nk; ++j) {   // a tag per group of MAXT rows, so the recurrent history's parity steps per group
            int* ctl = pf_ctl_[j];
            const unsigned tag0 = take_tags(ngs[j]);
            ctl[0] = p0s[j];
            ctl[1] = (int) tag0;
            for (int g = 0; g < ngs[j]; ++g) {
                ctl[2 + 2 * g] = p0s[j] + g * MAXT;
                ctl[3 + 2 * g] = (int) (tag0 + (unsigned) g);
            }
        }
        each([&](int, CardBuf& b, const CardW&, Pf& p) {
            for (int j = 0; j < nk; ++j) {
                ckc(cudaMemcpyAsync(p.Rk[j], pf_emb_[0] + (size_t) j * pf_c_ * N, (size_t) Cs[j] * N * 4, cudaMemcpyHostToDevice, b.st), "rows");
                ckc(cudaMemcpyAsync(p.ctlk[j], pf_ctl_[j], (size_t) (1 + ngs[j]) * 2 * 4, cudaMemcpyHostToDevice, b.st), "ctl");
            }
        });
        int C = 0, ng = 0, pos0 = 0, cur = 0;
        auto use = [&](int j) {
            C = Cs[j], ng = ngs[j], pos0 = p0s[j], cur = j;
            for (Pf* p : pf_) p->R = p->Rk[j], p->bo = p->bok[j], p->ctl = p->ctlk[j];
        };
        // chunk cur's half sum: on the ring stream once its rows are ready; the chunk's next half waits for it
        auto sum = [&](int c, CardBuf& b, Pf& p) {
            if (skip_ring) return;
            ckc(cudaEventRecord(p.ev_go[cur], b.st), "ring go");
            ckc(cudaStreamWaitEvent(p.rst, p.ev_go[cur], 0), "ring go");
            pf_ring(p, c, p.bo, C, b.err);
            ckc(cudaEventRecord(p.ev_done[cur], p.rst), "ring done");
        };
        auto summed = [&](CardBuf& b, Pf& p) {
            if (!skip_ring) ckc(cudaStreamWaitEvent(b.st, p.ev_done[cur], 0), "ring wait");
        };
        // the attention over the running chunk on card c (slot's K/V stored by k_s27_prep already): per KV head the
        // positions [0, pos0 + C) gathered, then per query head of it the queries in blocks
        auto attention = [&](CardBuf& b, const CardW& w, Pf& p, int slot) {
            const int P = pos0 + C;
            const int qb = (int) std::min<size_t>((size_t) C, std::max<size_t>(MAXT, p.score_floats / (size_t) P / MAXT * MAXT));
            const float one = 1.0f, zero = 0.0f, scale = 1.0f / std::sqrt((float) HD);
            for (int j = 0; j < S27_NKV; ++j) {
                k_pf_gather<<<grid_for((size_t) P * HD / 8), 256, 0, b.st>>>(b.pool, b.pt, slot, j, P, p.kc, p.vc);
                for (int h = 0; h < S27_NQ; ++h) {
                    if ((w.q0 + h) / (NQH / NKVH) - w.kv0 != j) continue;
                    for (int r0 = 0; r0 < C; r0 += qb) {
                        const int n = std::min(qb, C - r0), Pb = pos0 + r0 + n;
                        k_pf_q16<<<grid_for((size_t) n * HD), 256, 0, b.st>>>(p.big, h, r0, n, p.q16);
                        if (scores16) {
                            ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, Pb, n, HD, &scale, p.kc, CUDA_R_16F, HD, p.q16, CUDA_R_16F, HD,
                                             &zero, p.prob, CUDA_R_16F, Pb, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                                "scores");
                            k_pf_softmax16<<<n, 256, 0, b.st>>>(p.prob, Pb, pos0 + r0);
                        } else {
                            ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, Pb, n, HD, &one, p.kc, CUDA_R_16F, HD, p.q16, CUDA_R_16F, HD, &zero,
                                             p.score, CUDA_R_32F, Pb, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                                "scores");
                            k_pf_softmax<<<n, 256, 0, b.st>>>(p.score, Pb, pos0 + r0, p.prob);
                        }
                        ckb(cublasGemmEx(p.blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, n, Pb, &one, p.vc, CUDA_R_16F, HD, p.prob, CUDA_R_16F, Pb, &zero,
                                         p.o, CUDA_R_32F, HD, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                            "values");
                        k_pf_gate<<<grid_for((size_t) n * HD), 256, 0, b.st>>>(p.o, p.big, h, r0, n, p.xh);
                    }
                }
            }
        };
        for (int l = 0; l < NL; ++l) {
            for (int j = 0; j < nk; ++j) {
                use(j);
                each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    if (l > 0) summed(b, p);
                    k_pf_add_norm<<<C, 1024, 0, b.st>>>(p.R, l > 0 ? p.bo : nullptr, L.attn_norm, p.xf, p.xh, eps);
                    if (!is_attn(l)) {
                        const int nv = w.nv();
                        pf_dense(p, L.qkvz, p.xh, C, p.big);
                        const float one = 1.0f, zero = 0.0f;
                        ckb(cublasSgemm(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * nv, C, N, &one, L.ab, N, p.xf, N, &zero, p.ab, 2 * nv), "alpha beta");
                        gdn_prompt(p.big, p.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], p.ctl, ng, w.nk, nv, w.vmap, C, eps,
                                   p.gdn_xc, p.gdn_sq, p.gdn_y, p.gdn_ch, b.st);
                        k_f32_f16<<<grid_for((size_t) C * nv * HS), 256, 0, b.st>>>(p.gdn_y, (size_t) C * nv * HS, p.xh);
                        pf_dense(p, L.ssm_out, p.xh, C, p.bo);
                    } else {
                        pf_dense(p, L.wq, p.xh, C, p.big);
                        k_s27_prep<<<dim3(S27_NQ + 2 * S27_NKV, C), HD, 0, b.st>>>(p.big, S27_QKV, L.q_norm, L.k_norm, b.pool, s27_slot(l), p.ctl,
                                                                                    b.pt, LaneRows{}, nullptr, eps, rope_of(b.mrope));
                        if (!skip_attn) attention(b, w, p, s27_slot(l));
                        pf_dense(p, L.wo, p.xh, C, p.bo);
                    }
                    ckc(cudaGetLastError(), "mixer");
                    sum(c, b, p);
                });
            }
            for (int j = 0; j < nk; ++j) {
                use(j);
                each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    const int ffs = w.fd1 - w.fd0;
                    summed(b, p);
                    k_pf_add_norm<<<C, 1024, 0, b.st>>>(p.R, p.bo, L.ffn_norm, nullptr, p.xh, eps);
                    pf_dense(p, L.ffn_gu, p.xh, C, p.big);
                    k_pf_swiglu<<<grid_for((size_t) C * ffs), 256, 0, b.st>>>(p.big, ffs, C, p.gu16);
                    pf_dense(p, L.ffn_down, p.gu16, C, p.bo);
                    ckc(cudaGetLastError(), "ffn");
                    sum(c, b, p);
                });
            }
        }
        for (int j = 0; j < nk; ++j) {
            use(j);
            each([&](int, CardBuf& b, const CardW& w, Pf& p) {
                summed(b, p);
                if (!mtp_) return;
                // the draft layer over the chunk, for its K/V: row t's h is its final row after the output norm, its e the
                // token after it
                k_pf_add_norm<<<C, 1024, 0, b.st>>>(p.R, p.bo, w.out_norm, p.xf, p.xh, eps);
                ckc(cudaMemcpyAsync(p.bo, pf_emb_[0] + ((size_t) j * pf_c_ + 1) * N, (size_t) C * N * 4, cudaMemcpyHostToDevice, b.st),
                    "draft rows");
                k_pf_norm_cat<<<C, 1024, 0, b.st>>>(p.bo, p.xf, w.mtp_enorm, w.mtp_hnorm, p.xh, eps);
                pf_dense(p, w.mtp_eh, p.xh, C, p.R);
                k_pf_add_norm<<<C, 1024, 0, b.st>>>(p.R, nullptr, w.mtp.attn_norm, nullptr, p.xh, eps);
                pf_dense(p, w.mtp.wq, p.xh, C, p.big);
                k_s27_prep<<<dim3(S27_NQ + 2 * S27_NKV, C), HD, 0, b.st>>>(p.big, S27_QKV, w.mtp.q_norm, w.mtp.k_norm, b.pool, S27_SLOTS - 1, p.ctl,
                                                                            b.pt, LaneRows{}, nullptr, eps, rope_of(b.mrope));
                ckc(cudaGetLastError(), "draft layer (prompt)");
            });
        }
        // bounded: a ring step waits on its flag without a timeout of its own
        unsigned err = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(240);
        each([&](int, CardBuf& b, const CardW&, Pf& p) {
            for (cudaStream_t q : {b.st, p.rst})
                for (;;) {
                    const cudaError_t e = cudaStreamQuery(q);
                    if (e == cudaSuccess) break;
                    if (e != cudaErrorNotReady) ckc(e, "prefill sync");
                    if (std::chrono::steady_clock::now() > deadline)
                        throw std::runtime_error("prefill: a pass did not finish in 240 s (a ring step's flag never came?)");
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            unsigned e = 0;
            ckc(cudaMemcpy(&e, b.err, 4, cudaMemcpyDeviceToHost), "err");
            err |= e;
        });
        if (err) throw std::runtime_error("prefill: error bits " + std::to_string(err) + (err & 2u ? " (a sum past fp16's range or NaN)" : ""));
        history_.insert(history_.end(), tokens.begin() + (long) i, tokens.begin() + (long) i + rows);
        pos_ += rows;
        if (timing)
            std::fprintf(stderr, "prefill pass at %d: %d rows in %.1f ms\n", pos_ - rows, rows,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pass).count());
        if (on_prefill_chunk) on_prefill_chunk((int) (i + rows));
    }
}

}  // namespace fnx
