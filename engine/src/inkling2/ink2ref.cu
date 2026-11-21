// src/inkling2/ink2ref.cu - NEW-MODELS-PLAN.md phase T1 for Inkling-2-Mini (thinkingmachines/Inkling-2-Mini,
// model_type inkling2_loop): the reference path, exactness first.  One GPU, every weight resident in VRAM in its
// checkpoint form (BF16 dense, NVFP4 experts on layers 0-19, MXFP8 experts on layers 20-23: ~18 GiB), one token at a
// time (the prompt too), fp32 activations.  The GPU only runs the matrix-vector products (plain kernels of this file);
// everything else is host fp32, written to mirror the transformers reference (PR #51050,
// models/inkling2_loop/modeling_inkling2_loop.py) line for line, so a mismatch points at one operation.
// No expert cache, no batching, no MTP, no vision.  Checked against that reference by tools/inkling2/ (per layer
// application outputs, logits, greedy tokens).
//
// CBA: host-side attention and convolutions, one cudaMemcpy pair per GEMV - a reference, ~0.1-0.3 s per token; the
// engine path (T2) moves all of it onto the GPU.
//
// The forward pass (per token):
//   h = rms(embed[tok]) * embed_norm;  B = []                       (B: "block states", snapshots of h)
//   layers 0-5, then the loop layers 6-17 (pass 0), then passes 1 and 2 of 6-17 restarted from the pre-loop (h, B) and
//   fed "messages" from the previous pass, then layers 18-23;  a layer whose index is a multiple of 6 appends its
//   INPUT to B.  Each pass of a loop layer has its own KV cache and convolution states (48 slots in all).
//   h = attnres(final_query, B, h); logits = softcap30(unembed (rms(h) * norm))
// One layer L (full attention when L % 6 == 5, else sliding window 1024):
//   a = attnres(attention_input[L], B, h)  (absent for L 0; three queries q/k/v on full layers)
//   [loop passes 1-2: a = message_gate(a, message)]   x = rms(a) * norm   (q/k/v norms on full layers)
//   att = attention(xq, xk, xv); att = shortconv(att)
//   r = (L % 6 == 0) ? att : h + att                (on a block boundary the residual restarts; B gets h)
//   m = attnres(mlp_input[L], B, r) [loop passes 1-2: message_gate(m, message)]
//   out = r + shortconv(MoE(rms(m) * mlp_norm))
//
// Usage: strata-ink2ref --model DIR (--tokens 1,2,3 | --tokens-file F) [--n 16] [--dump-layers FILE] [--dump-logits FILE]
//                       [--pos-offset N]
//   --dump-layers writes every position's 48 layer-application outputs (f32, [pos][48][2304]); --dump-logits every
//   position's logits (f32, [pos][200058]); prints the greedy continuation ("gen: id id ...").
#include "../image/vae/safetensors_lite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

using vec = std::vector<float>;
using strata::image::StFile;
using strata::image::StTensor;

// config.json (text_config) of Inkling-2-Mini.  CBA: fixed geometry, read from config.json if a second size ships.
constexpr int H = 2304, NL = 24, NQ = 24, NKV = 8, HD = 128, NREP = NQ / NKV, DREL = 16, EXTENT = 1024, WINDOW = 1024;
constexpr int NE = 128, TOPK = 8, FF = 1152, KC = 4, VOCAB = 200058;
constexpr int LOOP_BEGIN = 6, LOOP_END = 18, NLOOP = LOOP_END - LOOP_BEGIN, PASSES = 3, BLOCK = 6;
constexpr int NCALLS = NL + (PASSES - 1) * NLOOP;   // 48 layer applications
constexpr int MXFP8_FIRST = 20;                      // hf_quant_config.json: experts of layers 20-23 are MXFP8
constexpr float EPS = 1e-6f, QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f, LOG_ALPHA = 0.1f, LOG_FLOOR = 8192.0f;
constexpr float GATE_CAP = 4.0f, UP_CAP = 25.0f, ROUTE_SCALE = 8.0f, LOGIT_CAP = 30.0f;

bool full_layer(int L) { return L % BLOCK == BLOCK - 1; }   // local_layer_ids omit 5, 11, 17, 23

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// GPU: the matrix-vector products, one warp per output row, fp32 accumulation.

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }

__device__ __forceinline__ float e4m3f(uint8_t v) {   // float8_e4m3fn: bias 7, no infinities, S.1111.111 = NaN
    const int e = (v >> 3) & 15, m = v & 7;
    float r = e == 0 ? (float) m * 0.001953125f /* m/8 * 2^-6 */ : ldexpf(1.0f + (float) m * 0.125f, e - 7);
    if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
    return (v & 0x80) ? -r : r;
}

__constant__ float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float warp_sum(float a) {
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    return a;
}

__global__ void gemv_bf16(const uint16_t* W, const float* x, float* y, int N, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc += bf16f(w[k]) * x[k];
    acc = warp_sum(acc);
    if (!lane) y[row] = acc;
}

// NVFP4 (ModelOpt layout of this checkpoint): byte b of a row holds columns 2b (low nibble) and 2b+1 (high nibble);
// one e4m3 scale per 16 columns; one fp32 factor per expert (amax / outer_scale_denominator), applied to the sum.
__global__ void gemv_nvfp4(const uint8_t* W, const uint8_t* S, float factor, const float* x, float* y, int N, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint8_t* w = W + (size_t) row * (K / 2);
    const uint8_t* s = S + (size_t) row * (K / 16);
    float acc = 0.0f;
    for (int b = lane; b < K / 2; b += 32) {
        const uint8_t v = w[b];
        const float sc = e4m3f(s[b / 8]);
        acc += (kE2M1[v & 15] * sc) * x[2 * b] + (kE2M1[v >> 4] * sc) * x[2 * b + 1];
    }
    acc = warp_sum(acc);
    if (!lane) y[row] = acc * factor;
}

// MXFP8: e4m3 values, one ue8m0 exponent per 32 columns (2^(s - 127)).
__global__ void gemv_mxfp8(const uint8_t* W, const uint8_t* S, const float* x, float* y, int N, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint8_t* w = W + (size_t) row * K;
    const uint8_t* s = S + (size_t) row * (K / 32);
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc += (e4m3f(w[k]) * exp2f((float) s[k / 32] - 127.0f)) * x[k];
    acc = warp_sum(acc);
    if (!lane) y[row] = acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// Checkpoint access

struct Ckpt {
    std::string path;
    StFile st;
    const StTensor& t(const std::string& name) const {
        auto it = st.tensors.find(name);
        if (it == st.tensors.end()) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
        return it->second;
    }
    bool has(const std::string& name) const { return st.tensors.count(name) != 0; }
    std::vector<unsigned char> bytes(const std::string& name) const {
        std::vector<unsigned char> b;
        const std::string e = strata::image::st_read_tensor(path, st, t(name), b);
        if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); std::exit(1); }
        return b;
    }
    vec f32(const std::string& name) const {   // BF16 or F32 -> fp32 (exact)
        const StTensor& d = t(name);
        auto b = bytes(name);
        vec out;
        if (d.dtype == "F32") {
            out.resize(b.size() / 4);
            std::memcpy(out.data(), b.data(), b.size());
        } else if (d.dtype == "BF16") {
            out.resize(b.size() / 2);
            for (size_t i = 0; i < out.size(); ++i) {
                uint32_t u = (uint32_t) (b[2 * i] | (b[2 * i + 1] << 8)) << 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else { std::fprintf(stderr, "%s: dtype %s not handled\n", name.c_str(), d.dtype.c_str()); std::exit(1); }
        return out;
    }
};

const Ckpt* g_ck = nullptr;
size_t g_vram = 0;

void* upload(const std::vector<unsigned char>& b) {
    void* d = nullptr;
    CK(cudaMalloc(&d, b.size()));
    CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice));
    g_vram += b.size();
    return d;
}

struct DMat {   // a BF16 matrix [n, k] in VRAM
    const uint16_t* d = nullptr;
    int n = 0, k = 0;
};

DMat dmat(const std::string& name) {
    const StTensor& t = g_ck->t(name);
    if (t.dtype != "BF16") { std::fprintf(stderr, "%s: expected BF16, got %s\n", name.c_str(), t.dtype.c_str()); std::exit(1); }
    DMat m;
    m.n = (int) t.shape[0];
    m.k = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) m.k *= (int) t.shape[i];
    m.d = (const uint16_t*) upload(g_ck->bytes(name));
    return m;
}

float* g_dx = nullptr;
float* g_dy = nullptr;


vec mv(const DMat& W, const vec& x, int rows = -1) {   // W x, the first `rows` rows (all by default)
    const int n = rows < 0 ? W.n : rows;
    if ((int) x.size() != W.k) { std::fprintf(stderr, "mv: %zu != %d\n", x.size(), W.k); std::exit(1); }
    CK(cudaMemcpy(g_dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    gemv_bf16<<<(n + 7) / 8, 256>>>(W.d, g_dx, g_dy, n, W.k);
    vec y((size_t) n);
    CK(cudaMemcpy(y.data(), g_dy, y.size() * 4, cudaMemcpyDeviceToHost));
    return y;
}

struct Experts {
    bool nvfp4 = true;
    const uint8_t *w13 = nullptr, *s13 = nullptr, *w2 = nullptr, *s2 = nullptr;
    vec f13, f2;   // NVFP4: amax / denominator per expert
};

// gate_up rows of expert e: [2*FF rows (gate/up interleaved: row 2j gate j, 2j+1 up j), H columns]; down: [H, FF]
vec expert_mv(const Experts& X, int e, bool up_proj, const vec& x) {
    const int N = up_proj ? 2 * FF : H, K = up_proj ? H : FF;
    CK(cudaMemcpy(g_dx, x.data(), (size_t) K * 4, cudaMemcpyHostToDevice));
    if (X.nvfp4) {
        const uint8_t* w = (up_proj ? X.w13 : X.w2) + (size_t) e * N * (K / 2);
        const uint8_t* s = (up_proj ? X.s13 : X.s2) + (size_t) e * N * (K / 16);
        gemv_nvfp4<<<(N + 7) / 8, 256>>>(w, s, (up_proj ? X.f13 : X.f2)[e], g_dx, g_dy, N, K);
    } else {
        const uint8_t* w = (up_proj ? X.w13 : X.w2) + (size_t) e * N * K;
        const uint8_t* s = (up_proj ? X.s13 : X.s2) + (size_t) e * N * (K / 32);
        gemv_mxfp8<<<(N + 7) / 8, 256>>>(w, s, g_dx, g_dy, N, K);
    }
    vec y((size_t) N);
    CK(cudaMemcpy(y.data(), g_dy, y.size() * 4, cudaMemcpyDeviceToHost));
    return y;
}

// ---------------------------------------------------------------------------------------------------------------------
// Host fp32 operations (each one named after its transformers module)

float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

vec rms(const vec& x, const vec* w) {   // Inkling2LoopRMSNorm (w) / Inkling2LoopUnweightedRMSNorm (nullptr)
    double ss = 0.0;   // CBA: double for the mean; torch reduces fp32 pairwise, the difference is ~1e-7 relative
    for (float v : x) ss += (double) v * v;
    const float r = 1.0f / std::sqrt((float) (ss / x.size()) + EPS);
    vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = w ? (*w)[i] * (x[i] * r) : x[i] * r;
    return y;
}

float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

int g_pos_offset = 0;   // --pos-offset: shifts the positions Inkling2LoopLogScale sees (only), to test it past 8191

float log_scale(int pos) {   // Inkling2LoopLogScale
    pos += g_pos_offset;
    return 1.0f + LOG_ALPHA * std::log(std::max((float) (pos + 1) / LOG_FLOOR, 1.0f));
}

void l2_clip(float* v, int n, float thr) {   // Inkling2LoopL2Clip
    const float norm = std::sqrt(dot(v, v, n));
    const float s = std::min(thr / (norm + 1e-6f), 1.0f);
    for (int i = 0; i < n; ++i) v[i] *= s;
}

// Inkling2LoopBlockAttentionResidual / MessageAttentionResidual: softmax over the normalized stack [B..., h] of
// (q . s_j) / sqrt(H), returns sum_j p_j s_j (the NORMALIZED states), one output per query.
std::vector<vec> attnres(const float* queries, int nq, const std::vector<vec>& B, const vec& h) {
    std::vector<vec> s;
    for (const vec& b : B) s.push_back(rms(b, nullptr));
    s.push_back(rms(h, nullptr));
    std::vector<vec> out;
    const float scale = 1.0f / std::sqrt((float) H);
    for (int q = 0; q < nq; ++q) {
        vec lg(s.size());
        for (size_t j = 0; j < s.size(); ++j) lg[j] = dot(queries + (size_t) q * H, s[j].data(), H) * scale;
        const float mx = *std::max_element(lg.begin(), lg.end());
        float z = 0.0f;
        for (float& v : lg) { v = std::exp(v - mx); z += v; }
        vec o(H, 0.0f);
        for (size_t j = 0; j < s.size(); ++j)
            for (int i = 0; i < H; ++i) o[i] += (lg[j] / z) * s[j][i];
        out.push_back(o);
    }
    return out;
}

struct MsgGate {   // Inkling2LoopMessageGate (checkpoint: controllers.{attn,k,v,mlp}.{down,forget,gain}_weight)
    DMat down, forget, gain;
};

vec message_gate(const MsgGate& g, const vec& x, const vec& m) {   // (1 - tanh(F d)) * x + tanh(G d) * m
    vec cat = rms(x, nullptr);
    vec mn = rms(m, nullptr);
    cat.insert(cat.end(), mn.begin(), mn.end());
    const vec d = mv(g.down, cat);
    const vec f = mv(g.forget, d), gn = mv(g.gain, d);
    vec y(H);
    for (int i = 0; i < H; ++i) y[i] = (1.0f - std::tanh(f[i])) * x[i] + std::tanh(gn[i]) * m[i];
    return y;
}

// Inkling2LoopShortConvolution: depthwise causal conv over the sequence (taps w[c][0..3], w[c][3] on the current
// token), out = (conv + x) / sqrt(5).  `hist` holds this slot's previous inputs, oldest first.
vec short_conv(const vec& x, std::vector<vec>& hist, const vec& w) {
    const int C = (int) x.size();
    vec y(C);
    for (int c = 0; c < C; ++c) {
        float acc = w[(size_t) c * KC + KC - 1] * x[c];
        for (int j = 1; j < KC; ++j) {
            const int back = (int) hist.size() - j;
            if (back >= 0) acc += w[(size_t) c * KC + KC - 1 - j] * hist[back][c];
        }
        y[c] = (acc + x[c]) / std::sqrt((float) (KC + 1));
    }
    hist.push_back(x);
    return y;
}

// Inkling2LoopDynamicShortConvolution (full layers' keys and values): taps = static_w[lag][c] + (Wp xin)[head*4+lag],
// out = (sum_lag taps * x[t - lag] + x) / sqrt(9).
vec dyn_conv(const vec& x, const vec& xin, std::vector<vec>& hist, const vec& static_w, const DMat& wproj) {
    const vec dw = mv(wproj, xin);   // [NKV * KC]
    const int C = (int) x.size(), per_head = C / NKV;
    vec y(C);
    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int lag = 0; lag < KC; ++lag) {
            const float tap = static_w[(size_t) lag * C + c] + dw[(c / per_head) * KC + lag];
            const float* past = lag == 0 ? x.data() : ((int) hist.size() - lag >= 0 ? hist[hist.size() - lag].data() : nullptr);
            if (past) acc += tap * past[c];
        }
        y[c] = (acc + x[c]) / std::sqrt((float) (2 * KC + 1));
    }
    hist.push_back(x);
    return y;
}

// ---------------------------------------------------------------------------------------------------------------------
// The model

struct Layer {
    bool full = false, loop = false;
    DMat wq, wk, wv, wo, gate_attn, wr, sink, router;
    DMat k_wproj, v_wproj;                 // full layers
    vec q_scale, rel_scale, rel_proj;      // [24*128], [24*16], [16*1024]
    vec k_conv, v_conv, k_static, v_static, attn_conv, mlp_conv;
    vec attn_norm, q_norm, k_norm, v_norm, mlp_norm;
    vec attn_query, mlp_query;             // attention_input (1 or 3 queries; empty on layer 0), mlp_input
    vec router_bias;
    Experts ex;
    MsgGate gates[4];                      // q, k, v, mlp (loop layers)
};

struct Slot {                              // one layer application's state across positions
    std::vector<vec> K, V;                 // [pos][NKV*HD] after the clip / norm
    std::vector<vec> khist, vhist, ahist, mhist;
};

struct Model {
    vec embed_norm, norm, final_query, msg_queries;   // msg_queries: [NLOOP][4 (q,k,v,mlp)][H]
    std::vector<uint16_t> embed;                     // host BF16 [201024 * H]
    DMat unembed;
    Layer L[NL];
    Slot slots[NCALLS];
};

Model* M = nullptr;

void load(const Ckpt& ck) {
    M = new Model();
    const std::string p = "model.llm.";
    M->embed_norm = ck.f32(p + "embed_norm.weight");
    M->norm = ck.f32(p + "norm.weight");
    M->final_query = ck.f32(p + "attn_res.output_query");
    {
        auto b = ck.bytes(p + "embed.weight");
        M->embed.resize(b.size() / 2);
        std::memcpy(M->embed.data(), b.data(), b.size());
    }
    M->unembed = dmat(p + "unembed.weight");
    {   // Inkling2LoopMergeMessageQueries: queries rows (2i, 2i+1) = (q, mlp), kv_queries rows (2i, 2i+1) = (k, v)
        const vec q = ck.f32(p + "middle_cycle_cross_pass_queries"), kv = ck.f32(p + "middle_cycle_cross_pass_kv_queries");
        M->msg_queries.resize((size_t) NLOOP * 4 * H);
        for (int i = 0; i < NLOOP; ++i) {
            const float* src[4] = {&q[(size_t) (2 * i) * H], &kv[(size_t) (2 * i) * H], &kv[(size_t) (2 * i + 1) * H],
                                   &q[(size_t) (2 * i + 1) * H]};
            for (int j = 0; j < 4; ++j) std::memcpy(&M->msg_queries[((size_t) i * 4 + j) * H], src[j], H * 4);
        }
    }
    for (int l = 0; l < NL; ++l) {
        Layer& Y = M->L[l];
        const std::string q = p + "layers." + std::to_string(l) + ".";
        Y.full = full_layer(l);
        Y.loop = l >= LOOP_BEGIN && l < LOOP_END;
        Y.wq = dmat(q + "attn.wq_du.weight");
        Y.wk = dmat(q + "attn.wk_dv.weight");
        Y.wv = dmat(q + "attn.wv_dv.weight");
        Y.wo = dmat(q + "attn.wo_ud.weight");
        Y.gate_attn = dmat(q + "attn.gate_attn.weight");
        Y.wr = dmat(q + "attn.wr_du.weight");
        Y.sink = dmat(q + "attn.attn_sink_proj.weight");
        Y.router = dmat(q + "mlp.gate.weight");
        Y.router_bias = ck.f32(q + "mlp.gate.bias");
        Y.q_scale = ck.f32(q + "attn.q_l2_clip.scale");
        Y.rel_scale = ck.f32(q + "attn.rel_l2_clip.scale");
        Y.rel_proj = ck.f32(q + "attn.rel_logits_proj.proj");
        Y.attn_conv = ck.f32(q + "attn_sconv.weight");
        Y.mlp_conv = ck.f32(q + "mlp_sconv.weight");
        Y.mlp_norm = ck.f32(q + "mlp_norm.weight");
        if (Y.full) {
            Y.k_static = ck.f32(q + "attn.k_sconv.static_w");
            Y.v_static = ck.f32(q + "attn.v_sconv.static_w");
            Y.k_wproj = dmat(q + "attn.k_sconv.weight_proj");
            Y.v_wproj = dmat(q + "attn.v_sconv.weight_proj");
            Y.q_norm = ck.f32(q + "q_input_norm.weight");
            Y.k_norm = ck.f32(q + "k_input_norm.weight");
            Y.v_norm = ck.f32(q + "v_input_norm.weight");
        } else {
            Y.k_conv = ck.f32(q + "attn.k_sconv.weight");
            Y.v_conv = ck.f32(q + "attn.v_sconv.weight");
            Y.attn_norm = ck.f32(q + "attn_norm.weight");
        }
        if (l > 0) Y.attn_query = ck.f32(p + "attn_res.attn_queries." + std::to_string(l - 1));   // offset 1
        Y.mlp_query = ck.f32(p + "attn_res.mlp_queries." + std::to_string(l));
        if ((int) Y.attn_query.size() != (l == 0 ? 0 : (Y.full ? 3 : 1)) * H) {
            std::fprintf(stderr, "layer %d: attention_input query size %zu\n", l, Y.attn_query.size());
            std::exit(1);
        }
        if (Y.loop) {
            const char* names[4] = {"attn", "k", "v", "mlp"};
            for (int g = 0; g < 4; ++g) {
                const std::string c = q + "controllers." + names[g] + ".";
                Y.gates[g] = {dmat(c + "down_weight"), dmat(c + "forget_weight"), dmat(c + "gain_weight")};
            }
        }
        Experts& X = Y.ex;
        X.nvfp4 = l < MXFP8_FIRST;
        X.w13 = (const uint8_t*) upload(ck.bytes(q + "mlp.experts.w13_weight"));
        X.s13 = (const uint8_t*) upload(ck.bytes(q + "mlp.experts.w13_weight_scale"));
        X.w2 = (const uint8_t*) upload(ck.bytes(q + "mlp.experts.w2_weight"));
        X.s2 = (const uint8_t*) upload(ck.bytes(q + "mlp.experts.w2_weight_scale"));
        if (X.nvfp4) {
            const vec a13 = ck.f32(q + "mlp.experts.w13_weight_amax"), d13 = ck.f32(q + "mlp.experts.w13_weight_outer_scale_denominator");
            const vec a2 = ck.f32(q + "mlp.experts.w2_weight_amax"), d2 = ck.f32(q + "mlp.experts.w2_weight_outer_scale_denominator");
            X.f13.resize(NE);
            X.f2.resize(NE);
            for (int e = 0; e < NE; ++e) { X.f13[e] = a13[e] / d13[0]; X.f2[e] = a2[e] / d2[0]; }
        } else if (ck.has(q + "mlp.experts.w13_weight_amax")) {
            std::fprintf(stderr, "layer %d: MXFP8 expected but the experts carry an amax\n", l);
            std::exit(1);
        }
        std::fprintf(stderr, "\rloaded layer %d/%d, %.1f GiB in VRAM", l + 1, NL, g_vram / 1073741824.0);
    }
    std::fprintf(stderr, "\n");
}

// Inkling2LoopSlidingAttention / Inkling2LoopFullAttention for one token at `pos`
vec attention(const Layer& Y, Slot& S, int pos, const vec& xq, const vec& xk, const vec& xv) {
    const float tau = log_scale(pos);
    vec q = mv(Y.wq, xq);
    for (int h = 0; h < NQ; ++h) {   // (Log)ScaledL2Clip
        float* v = &q[(size_t) h * HD];
        for (int d = 0; d < HD; ++d) v[d] *= Y.q_scale[(size_t) h * HD + d];
        l2_clip(v, HD, QK_CLIP);
        if (Y.full) for (int d = 0; d < HD; ++d) v[d] *= tau;
    }
    vec k = mv(Y.wk, xk), v = mv(Y.wv, xv);
    if (Y.full) {
        k = dyn_conv(k, xk, S.khist, Y.k_static, Y.k_wproj);
        v = dyn_conv(v, xv, S.vhist, Y.v_static, Y.v_wproj);
    } else {
        k = short_conv(k, S.khist, Y.k_conv);
        v = short_conv(v, S.vhist, Y.v_conv);
    }
    for (int g = 0; g < NKV; ++g) {
        l2_clip(&k[(size_t) g * HD], HD, QK_CLIP);   // keys: L2Clip, no scale
        vec vh(v.begin() + (size_t) g * HD, v.begin() + (size_t) (g + 1) * HD);
        vh = rms(vh, nullptr);                        // values: UnweightedRMSNorm per head
        std::copy(vh.begin(), vh.end(), v.begin() + (size_t) g * HD);
    }
    S.K.push_back(k);
    S.V.push_back(v);

    vec rs = mv(Y.wr, xq);   // relative position bias states [24][16]
    for (int h = 0; h < NQ; ++h) {
        float* r = &rs[(size_t) h * DREL];
        for (int d = 0; d < DREL; ++d) r[d] *= Y.rel_scale[(size_t) h * DREL + d];
        l2_clip(r, DREL, REL_CLIP);
        if (Y.full) for (int d = 0; d < DREL; ++d) r[d] *= tau;
    }
    const vec sink = mv(Y.sink, xq);
    const vec gate = mv(Y.gate_attn, xq);

    const int T = (int) S.K.size();   // == pos + 1
    const float scale = 1.0f / std::sqrt((float) HD);
    vec att((size_t) NQ * HD, 0.0f);
    for (int h = 0; h < NQ; ++h) {
        const int g = h / NREP;
        vec lg((size_t) T, -INFINITY);
        float mx = sink[h];
        for (int j = 0; j < T; ++j) {
            const int dist = (T - 1) - j;
            if (!Y.full && dist >= WINDOW) continue;
            float bias = 0.0f;
            if (dist < EXTENT) {
                for (int r = 0; r < DREL; ++r) bias += rs[(size_t) h * DREL + r] * Y.rel_proj[(size_t) r * EXTENT + dist];
            }
            lg[j] = dot(&q[(size_t) h * HD], &S.K[j][(size_t) g * HD], HD) * scale + bias;
            mx = std::max(mx, lg[j]);
        }
        float z = std::exp(sink[h] - mx);   // the sink: one more logit with a zero value
        for (int j = 0; j < T; ++j) { lg[j] = std::exp(lg[j] - mx); z += lg[j]; }
        float* o = &att[(size_t) h * HD];
        for (int j = 0; j < T; ++j)
            for (int d = 0; d < HD; ++d) o[d] += (lg[j] / z) * S.V[j][(size_t) g * HD + d];
    }
    for (size_t i = 0; i < att.size(); ++i) att[i] *= sigmoidf(gate[i]);   // GatedOutputProjection
    return mv(Y.wo, att);
}

vec moe(const Layer& Y, const vec& x) {   // Inkling2LoopSparseMoeBlock (router in fp32, sigmoid scores)
    const vec lg = mv(Y.router, x);
    vec s(NE), sel(NE);
    for (int e = 0; e < NE; ++e) { s[e] = sigmoidf(lg[e]); sel[e] = s[e] + Y.router_bias[e]; }
    std::vector<int> idx(NE);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), [&](int a, int b) { return sel[a] > sel[b]; });
    float sum = 0.0f;
    for (int k = 0; k < TOPK; ++k) sum += s[idx[k]];
    vec out(H, 0.0f);
    for (int k = 0; k < TOPK; ++k) {
        const int e = idx[k];
        const float w = s[e] / (sum + 1e-20f) * ROUTE_SCALE;
        const vec gu = expert_mv(Y.ex, e, true, x);
        vec act(FF);
        for (int j = 0; j < FF; ++j) {   // SoftcappedSwiGLU
            const float g = gu[2 * j], u = gu[2 * j + 1];
            act[j] = (std::tanh(g / GATE_CAP) * GATE_CAP) * sigmoidf(g) * (std::tanh(u / UP_CAP) * UP_CAP);
        }
        const vec y = expert_mv(Y.ex, e, false, act);
        for (int i = 0; i < H; ++i) out[i] += y[i] * w;
    }
    return out;
}

// One decoder layer application; `msgs` = the 4 messages (q, k, v, mlp) on loop passes 1-2, else nullptr.
vec layer_forward(int l, Slot& S, int pos, const vec& h, const std::vector<vec>& B, const std::vector<vec>* msgs) {
    const Layer& Y = M->L[l];
    vec xq, xk, xv;
    if (Y.full) {
        vec qi = h, ki = h, vi = h;
        if (l > 0) {
            auto r = attnres(Y.attn_query.data(), 3, B, h);
            qi = r[0]; ki = r[1]; vi = r[2];
        }
        if (msgs) {
            qi = message_gate(Y.gates[0], qi, (*msgs)[0]);
            ki = message_gate(Y.gates[1], ki, (*msgs)[1]);
            vi = message_gate(Y.gates[2], vi, (*msgs)[2]);
        }
        xq = rms(qi, &Y.q_norm); xk = rms(ki, &Y.k_norm); xv = rms(vi, &Y.v_norm);
    } else {
        const vec a = l > 0 ? attnres(Y.attn_query.data(), 1, B, h)[0] : h;
        if (msgs) {
            xq = rms(message_gate(Y.gates[0], a, (*msgs)[0]), &Y.attn_norm);
            xk = rms(message_gate(Y.gates[1], a, (*msgs)[1]), &Y.attn_norm);
            xv = rms(message_gate(Y.gates[2], a, (*msgs)[2]), &Y.attn_norm);
        } else {
            xq = xk = xv = rms(a, &Y.attn_norm);
        }
    }
    vec att = attention(Y, S, pos, xq, xk, xv);
    att = short_conv(att, S.ahist, Y.attn_conv);
    std::vector<vec> B2 = B;
    vec r(H);
    if (l % BLOCK == 0) { r = att; B2.push_back(h); }
    else for (int i = 0; i < H; ++i) r[i] = h[i] + att[i];
    vec m = attnres(Y.mlp_query.data(), 1, B2, r)[0];
    if (msgs) m = message_gate(Y.gates[3], m, (*msgs)[3]);
    vec y = moe(Y, rms(m, &Y.mlp_norm));
    y = short_conv(y, S.mhist, Y.mlp_conv);
    for (int i = 0; i < H; ++i) y[i] += r[i];
    return y;
}

vec forward_token(int tok, int pos, std::vector<vec>* calls) {
    vec e(H);
    for (int i = 0; i < H; ++i) {
        uint32_t u = (uint32_t) M->embed[(size_t) tok * H + i] << 16;
        std::memcpy(&e[i], &u, 4);
    }
    vec h = rms(e, &M->embed_norm);
    std::vector<vec> B;
    auto run = [&](int l, int slot, const std::vector<vec>* msgs) {
        vec out = layer_forward(l, M->slots[slot], pos, h, B, msgs);
        if (l % BLOCK == 0) B.push_back(h);
        h = out;
        if (calls) calls->push_back(h);
    };
    for (int l = 0; l < LOOP_BEGIN; ++l) run(l, l, nullptr);
    const vec pre_h = h;
    const std::vector<vec> pre_B = B;
    for (int l = LOOP_BEGIN; l < LOOP_END; ++l) run(l, l, nullptr);
    for (int p = 1; p < PASSES; ++p) {
        const std::vector<vec> prevB(B.begin() + pre_B.size() + 1, B.end());   // +1: skip the first loop block
        const vec prev_h = h;
        h = pre_h;
        B = pre_B;
        for (int i = 0; i < NLOOP; ++i) {
            const std::vector<vec> msgs = attnres(&M->msg_queries[(size_t) i * 4 * H], 4, prevB, prev_h);
            run(LOOP_BEGIN + i, NL + (p - 1) * NLOOP + i, &msgs);
        }
    }
    for (int l = LOOP_END; l < NL; ++l) run(l, l, nullptr);
    h = attnres(M->final_query.data(), 1, B, h)[0];
    vec logits = mv(M->unembed, rms(h, &M->norm), VOCAB);
    for (float& v : logits) v = std::tanh(v / LOGIT_CAP) * LOGIT_CAP;
    return logits;
}

std::vector<int> parse_ids(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        v.push_back((int) std::strtol(p, &end, 10));
        if (end == p) break;
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir, dump_layers, dump_logits;
    std::vector<int> ids;
    int n_gen = 16;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--model") dir = argv[i + 1];
        else if (a == "--tokens") ids = parse_ids(argv[i + 1]);
        else if (a == "--n") n_gen = std::atoi(argv[i + 1]);
        else if (a == "--pos-offset") g_pos_offset = std::atoi(argv[i + 1]);
        else if (a == "--tokens-file") {
            FILE* f = std::fopen(argv[i + 1], "r");
            if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[i + 1]); return 2; }
            std::string text;
            for (int c; (c = std::fgetc(f)) != EOF;) text += (char) c;
            std::fclose(f);
            ids = parse_ids(text.c_str());
        }
        else if (a == "--dump-layers") dump_layers = argv[i + 1];
        else if (a == "--dump-logits") dump_logits = argv[i + 1];
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (dir.empty() || ids.empty()) {
        std::fprintf(stderr, "usage: strata-ink2ref --model DIR --tokens 1,2,3 [--n 16] [--dump-layers F] [--dump-logits F]\n");
        return 2;
    }
    Ckpt ck;
    ck.path = dir + "/model-00001-of-00001.safetensors";
    const std::string e = strata::image::st_read_header(ck.path, ck.st);
    if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    g_ck = &ck;
    CK(cudaMalloc(&g_dx, (size_t) 2 * H * 4));
    CK(cudaMalloc(&g_dy, (size_t) 201024 * 4));
    load(ck);

    FILE* fl = dump_layers.empty() ? nullptr : std::fopen(dump_layers.c_str(), "wb");
    FILE* fg = dump_logits.empty() ? nullptr : std::fopen(dump_logits.c_str(), "wb");
    vec logits;
    for (size_t t = 0; t < ids.size(); ++t) {
        std::vector<vec> calls;
        logits = forward_token(ids[t], (int) t, &calls);
        if ((int) calls.size() != NCALLS) { std::fprintf(stderr, "%zu layer calls\n", calls.size()); return 1; }
        if (fl) for (const vec& c : calls) std::fwrite(c.data(), 4, H, fl);
        if (fg) std::fwrite(logits.data(), 4, logits.size(), fg);
        std::fprintf(stderr, "\rprompt %zu/%zu", t + 1, ids.size());
    }
    std::fprintf(stderr, "\n");
    if (fl) std::fclose(fl);
    if (fg) std::fclose(fg);
    std::printf("top:");
    {
        std::vector<int> o(logits.size());
        std::iota(o.begin(), o.end(), 0);
        std::partial_sort(o.begin(), o.begin() + 5, o.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        for (int k = 0; k < 5; ++k) std::printf(" %d:%.4f", o[k], logits[o[k]]);
    }
    std::printf("\ngen:");
    int pos = (int) ids.size();
    for (int k = 0; k < n_gen; ++k) {
        const int next = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::printf(" %d", next);
        std::fflush(stdout);
        logits = forward_token(next, pos++, nullptr);
    }
    std::printf("\n");
    return 0;
}
