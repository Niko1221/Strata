// src/flint/flint_engine.cu - strata-flint: native C++/CUDA engine for Flint 40B Preview.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/flint/engine.hpp"
#include "strata/flint/sampling.hpp"
#include "flint_kernels.cuh"

#if defined(STRATA_FLINT_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace K = strata::kernels;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-flint: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CK_BLAS(x) \
    do { \
        cublasStatus_t s_ = (x); \
        if (s_ != CUBLAS_STATUS_SUCCESS) { \
            std::fprintf(stderr, "strata-flint: %s failed: cublasStatus %d (%s:%d)\n", #x, (int) s_, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;
constexpr size_t kMmqTail = 256;

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline float f16_to_f32(uint16_t h) {
    return strata::kernels::f32_from_f16(h);
}

float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) {
        std::memcpy(h.data(), src, h.size() * 4);
    } else if (t.type == 30) {
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    } else if (t.type == 1) {
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    } else {
        std::fprintf(stderr, "strata-flint: tensor %s has unexpected type %u\n", t.name.c_str(), t.type);
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

struct Mat {
    const uint8_t* d = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool is_moe = true;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;

    // Attention
    Mat wq, wk, wv, wo;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    const float* sinks = nullptr;

    // Dense FFN (Layers 0, 1)
    Mat dense_gate, dense_up, dense_down;

    // MoE FFN (Layers 2..48)
    const float* router = nullptr; // [n_expert][N]
    Mat sh_gate, sh_up, sh_down;
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;
    size_t stride = 0;
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    int resident_slots = 0;
    std::vector<int32_t> slot_of;
    int32_t* slot_of_d = nullptr;
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    int top_k = 6;
    float eps = 1e-6f;
    float rope_base = 10000.0f;
    int n_rot = 64;
    int sliding_window = 513;
    bool hybrid = false;
};

void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {

    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;

    m.g.n_layers = 49;
    m.g.n_embd = 2048;
    m.g.n_head = 32;
    m.g.n_head_kv = 4;
    m.g.head_dim = 128;
    m.g.n_rot = 64;
    m.g.n_expert = 256;
    m.g.n_expert_used = 6;
    m.g.n_ff = 512;
    m.g.n_ff_shexp = 1024;
    m.top_k = 6;
    m.eps = 1e-6f;
    m.rope_base = 10000.0f;
    m.sliding_window = 513;
    m.g.n_vocab = 150144;

    const std::string p = "flint.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.sliding_window")) m.sliding_window = (int) v->u;

    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-flint: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };

    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };

    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-flint: %s is %s, native MMVQ does not support it\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        w.d = d;
        return w;
    };

    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };

    m.head = mat("output.weight");
    m.out_norm = f32("output_norm.weight");

    size_t se = 0;
    const strata::TensorInfo* te = find("token_embd.weight", se);
    m.embd_type = (int) te->type;
    int el = 0, by = 0;
    strata::block_geometry(te->type, el, by);
    m.embd_row = (size_t) m.g.n_embd / el * by;
    const uint64_t nb = strata::tensor_payload_bytes(*te);
    uint8_t* d_emb = nullptr;
    CK(cudaMalloc(&d_emb, nb));
    CK(cudaMemcpy(d_emb, gm.shard(se).tensor_data(*te), nb, cudaMemcpyHostToDevice));
    m.embd = d_emb;
    std::fprintf(stderr, "strata-flint: token_embd uploaded to VRAM: %.2f MB\n", (double) nb / (1 << 20));

    m.layers.resize((size_t) m.g.n_layers);
    for (int64_t l = 0; l < m.g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.ffn_norm = f32(L(l, "ffn_norm.weight"));

        // Attention
        y.wq = mat(L(l, "attn_q.weight"));
        y.wk = mat(L(l, "attn_k.weight"));
        y.wv = mat(L(l, "attn_v.weight"));
        y.wo = mat(L(l, "attn_output.weight"));
        y.q_norm = f32(L(l, "attn_q_norm.weight"));
        y.k_norm = f32(L(l, "attn_k_norm.weight"));
        y.sinks = f32(L(l, "attn_sinks.weight"));

        if (l < 2) {
            y.is_moe = false;
            y.dense_gate = mat(L(l, "ffn_gate.weight"));
            y.dense_up = mat(L(l, "ffn_up.weight"));
            y.dense_down = mat(L(l, "ffn_down.weight"));
        } else {
            y.is_moe = true;
            y.router = f32(L(l, "ffn_gate_inp.weight"));
            y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
            y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
            y.sh_down = mat(L(l, "ffn_down_shexp.weight"));

            size_t sg = 0, su = 0, sd = 0;
            const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
            const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
            const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);

            y.L = K::native_expert_layout((int) tg->type, (int) td->type, m.g.n_embd, m.g.n_ff);
            const size_t al = 16;
            y.stride = (y.L.bytes + al - 1) / al * al;
            if (l == 2) {
                std::fprintf(stderr, "strata-flint: layer 2: gate_type=%d (%s), down_type=%d (%s), L.bytes=%zu, stride=%zu\n",
                             (int) tg->type, tg->type_name(), (int) td->type, td->type_name(), y.L.bytes, y.stride);
            }
            y.src_g = gm.shard(sg).tensor_data(*tg);
            y.src_u = gm.shard(su).tensor_data(*tu);
            y.src_d = gm.shard(sd).tensor_data(*td);
            y.src_gsz = (size_t) m.g.n_ff * y.L.gu_row;
            y.src_dsz = (size_t) m.g.n_embd * y.L.d_row;
        }
    }

    std::fprintf(stderr, "strata-flint: 49 layers loaded in %.2f s\n", ms_since(t0) / 1000.0);
    return true;
}

struct Engine {
    const Model* m = nullptr;
    float margin = 1.0f;
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;

    float *x = nullptr, *h = nullptr, *y = nullptr;
    float *q = nullptr, *kx = nullptr, *vx = nullptr, *att = nullptr;
    float *pacc = nullptr, *pm = nullptr, *pl = nullptr;
    float *logits_r = nullptr, *eparts = nullptr, *moe_out = nullptr;
    float *dense_act = nullptr, *sh_g = nullptr, *sh_u = nullptr, *sh_act = nullptr, *sh_out = nullptr;
    float *logits = nullptr;

    void* q8 = nullptr;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr;
    int32_t *ent_dst = nullptr, *ent_tok = nullptr, *tok = nullptr, *pos0 = nullptr, *pick = nullptr;
    int32_t* hpin = nullptr;
    unsigned long long* grp_ptr = nullptr;
    void* moe_scratch = nullptr;
    float *ewts = nullptr;
    int n_chunk = 0;
    std::vector<flintk::BlockQ80*> kc, vc; // Hardware-quantized Q8_0 KV Cache per layer

    cudaGraphExec_t g1 = nullptr;
    bool use_graph = true;

    void allocate(const Model& model, int max_ctx, int max_chunk) {
        m = &model;
        ctx = max_ctx;
        tmax = max_chunk;
        CK(cudaStreamCreate(&st));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-flint: cublasCreate failed\n");
            std::exit(1);
        }
        CK_BLAS(cublasSetStream(blas, st));

        const int N = (int) m->g.n_embd;
        const int H = (int) m->g.n_head, HKV = (int) m->g.n_head_kv, HD = (int) m->g.head_dim;
        const int Kp = m->top_k;
        const int ne = (int) m->g.n_expert;
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;

        x = dalloc((size_t) tmax * N);
        h = dalloc((size_t) tmax * N);
        y = dalloc((size_t) tmax * N);

        q = dalloc((size_t) tmax * H * HD);
        kx = dalloc((size_t) tmax * HKV * HD);
        vx = dalloc((size_t) tmax * HKV * HD);
        att = dalloc((size_t) tmax * H * HD);

        pacc = dalloc((size_t) tmax * H * n_chunk * HD);
        pm = dalloc((size_t) tmax * H * n_chunk);
        pl = dalloc((size_t) tmax * H * n_chunk);

        logits_r = dalloc((size_t) tmax * ne);
        ewts = dalloc((size_t) tmax * Kp);
        eparts = dalloc((size_t) tmax * Kp * N);
        moe_out = dalloc((size_t) tmax * N);

        // Dense FFN buffers (width 6144)
        dense_act = dalloc((size_t) tmax * 6144);

        // Shared expert buffers (width 1024)
        sh_g = dalloc((size_t) tmax * 1024);
        sh_u = dalloc((size_t) tmax * 1024);
        sh_act = dalloc((size_t) tmax * 1024);
        sh_out = dalloc((size_t) tmax * N);

        logits = dalloc((size_t) tmax * m->g.n_vocab);

        CK(cudaMalloc(&q8, (size_t) tmax * (8192 >> 5) * sizeof(flintk::QB)));
        CK(cudaMallocHost(&hpin, 4096 * sizeof(int32_t)));

        tok = dalloc_t<int32_t>(tmax);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(tmax);

        ids = dalloc_t<int32_t>(tmax * Kp);
        grp_start = dalloc_t<int32_t>(ne + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(tmax * Kp);
        ent_tok = dalloc_t<int32_t>(tmax * Kp);
        grp_ptr = dalloc_t<unsigned long long>(ne + 1);
        const size_t scratch_sz = K::native_expert_scratch_bytes((size_t) tmax * Kp, m->g.n_ff);
        CK(cudaMalloc(&moe_scratch, scratch_sz));


        size_t fr_b = 0, tot_b = 0;
        CK(cudaMemGetInfo(&fr_b, &tot_b));
        std::fprintf(stderr, "strata-flint: allocate: ctx=%d, tmax=%d, VRAM free before KV=%.2f GiB / %.2f GiB\n",
                     ctx, tmax, (double) fr_b / (1 << 30), (double) tot_b / (1 << 30));

        // KV cache allocation: hardware-accelerated Q8_0 (4 blocks of 34B per head per token)
        kc.resize((size_t) m->g.n_layers);
        vc.resize((size_t) m->g.n_layers);
        const size_t kv_blocks = (size_t) HKV * ctx * 4;
        const size_t kv_bytes_layer = kv_blocks * sizeof(flintk::BlockQ80) * 2;
        for (int64_t l = 0; l < m->g.n_layers; ++l) {
            kc[(size_t) l] = dalloc_t<flintk::BlockQ80>(kv_blocks);
            vc[(size_t) l] = dalloc_t<flintk::BlockQ80>(kv_blocks);
        }
        std::fprintf(stderr, "strata-flint: KV Cache: Q8_0, 49 layers, ctx=%d, footprint=%.2f GiB (%.1f MB/layer)\n",
                     ctx, (double) (kv_bytes_layer * m->g.n_layers) / (1 << 30),
                     (double) kv_bytes_layer / (1 << 20));
    }


    void prep(const float* xin, int n, int T) {
        flintk::QB* dst = reinterpret_cast<flintk::QB*>(q8);
        constexpr int MAX_NCOLS = 8;
        for (int j = 0; j < T; j += MAX_NCOLS) {
            const int nj = std::min(MAX_NCOLS, T - j);
            K::native_quantize_q8_1(xin + (size_t) j * n, dst + (size_t) j * (n / 32), n, nj, st);
        }
    }

    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= 8) {
            flintk::k_gemv_f32<<<dim3((n_out + 7) / 8, T), 256, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        const float alpha = 1.0f, beta = 0.0f;
        CK_BLAS(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &alpha, W, n_in, xin, n_in, &beta, out, n_out));
    }

    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T);
        K::native_mmvq(w.type, w.d, q8, out, w.n_in, w.n_out, T, st);
    }

    void layer(const Layer& L, size_t li, int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        const int Kp = m->top_k;

        // Attention Q, K, V projections
        mm(L.wq, h, q, T);
        mm(L.wk, h, kx, T);
        mm(L.wv, h, vx, T);

        // RoPE, QK-RMSNorm, and write to KV Cache
        flintk::k_flint_attn_prep<<<dim3(H + HKV, T), 256, 0, st>>>(
            q, kx, vx, L.q_norm, L.k_norm, q, kc[li], vc[li],
            H, HKV, HD, m->n_rot, pos0, ctx, m->rope_base, m->eps);

        // Split-K FlashAttention
        const int nc = std::min<int>(n_chunk, (int) (p0 + T + kChunkCells - 1) / kChunkCells);
        flintk::k_flint_attn_part<<<dim3(nc, T * H), kChunkCells, 0, st>>>(
            q, kc[li], vc[li], pos0, 0, H, HKV, ctx,
            1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, L.sinks, m->sliding_window);

        flintk::k_flint_attn_merge<<<T * H, HD, 0, st>>>(pacc, pm, pl, pos0, 0, H, n_chunk, att);

        // Attention Out
        mm(L.wo, att, y, T);

        // Add residual and compute ffn_norm
        flintk::k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.ffn_norm, h, N, m->eps);

        if (!L.is_moe) {
            // Dense FFN: gate, up, swiglu, down
            mm(L.dense_gate, h, sh_g, T);
            mm(L.dense_up, h, sh_u, T);
            flintk::k_swiglu<<<(T * 6144 + 255) / 256, 256, 0, st>>>(sh_g, sh_u, dense_act, T * 6144);
            mm(L.dense_down, dense_act, y, T);
            flintk::k_add_rms<<<T, kThreads, 0, st>>>(x, y, (li + 1 < (size_t) g.n_layers) ? m->layers[li + 1].attn_norm : m->out_norm, h, N, m->eps);
        } else {
            // MoE Router: compute router logits via GEMV (FP32)
            f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);

            // Top-6 router selection into ids and ewts
            flintk::route_topk<256, 6><<<(unsigned) ((T + 7) / 8), dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T);

            // Residency-biased routing: prioritize resident experts within margin (zero PCIe latency)
            if (margin > 0.0f) {
                flintk::route_resident<256, 6><<<(unsigned) T, 32, 0, st>>>(logits_r, ids, ewts, L.slot_of_d, T, margin);
            }

            const int n_ent = T * Kp;
            // Grouping: entries directly mapped to resident expert slots in VRAM
            flintk::k_group_hy<<<1, 256, 0, st>>>(
                ids, n_ent, Kp, (int) g.n_expert, L.slot_of_d, L.blobs, L.stride,
                grp_ptr, grp_start, n_groups, ent_dst, ent_tok);

            // Quantize h into q8 for routed expert execution
            prep(h, N, T);

            // Compute routed experts on GPU via grouped execution
            K::native_expert_grouped(L.L, (const unsigned long long*) grp_ptr, grp_start, n_groups,
                                     ent_dst, ent_tok, (int64_t) std::min<int>(n_ent, 256), (int64_t) n_ent,
                                     q8, moe_scratch, eparts, st);

            // Shared expert: gate, up, swiglu, down
            mm(L.sh_gate, h, sh_g, T);
            mm(L.sh_up, h, sh_u, T);
            flintk::k_swiglu<<<(T * 1024 + 255) / 256, 256, 0, st>>>(sh_g, sh_u, sh_act, T * 1024);
            mm(L.sh_down, sh_act, sh_out, T);

            // MoE combine: sum weighted experts + shared expert, add residual & compute next layer's norm
            flintk::k_flint_moe_out<<<T, 1024, 0, st>>>(
                x, eparts, nullptr, ewts, Kp, sh_out,
                (li + 1 < (size_t) g.n_layers) ? m->layers[li + 1].attn_norm : m->out_norm,
                h, N, m->eps);
        }
    }

    void forward(int T, int64_t p0, int heads) {
        const int N = (int) m->g.n_embd;
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        flintk::k_add_rms<<<T, kThreads, 0, st>>>(x, x, m->layers[0].attn_norm, h, N, m->eps);
        for (size_t l = 0; l < m->layers.size(); ++l) {
            layer(m->layers[l], l, T, p0);
        }
        if (heads > 0) {
            mm(m->head, h + (size_t) (T - heads) * N, logits, heads);
        }
    }
};

bool place_experts(Model& m, Engine& E, double max_gib, std::string& err) {
    const auto t0 = Clock::now();
    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));

    // Reserve 1.2 GiB headroom for activations, OS, and scratch
    const size_t headroom = (size_t) (1.2 * (1 << 30));
    size_t budget = free_b > headroom ? free_b - headroom : 0;
    if (max_gib > 0) {
        size_t cap = (size_t) (max_gib * (1 << 30));
        if (cap < budget) budget = cap;
    }

    const int64_t ne = m.g.n_expert;
    const size_t per_layer_budget = budget / 47;
    std::vector<uint8_t> blob;
    size_t total_vram_used = 0;
    int total_res_slots = 0;

    for (size_t l = 0; l < m.layers.size(); ++l) {
        Layer& L = m.layers[l];
        if (!L.is_moe) continue;

        int res_slots = std::max<int>(1, std::min<int>((int) ne, (int) (per_layer_budget / L.stride)));
        L.resident_slots = res_slots;

        size_t lyr_sz = L.stride * (size_t) res_slots + kMmqTail;
        CK(cudaMalloc(&L.blobs, lyr_sz));
        CK(cudaMemset(L.blobs, 0, lyr_sz));

        blob.assign(L.L.bytes, 0);
        for (int64_t e = 0; e < res_slots; ++e) {
            expert_blob(L, e, blob.data());
            CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
        }

        // Initialize slot_of lookup table
        L.slot_of.assign((size_t) ne, -1);
        for (int e = 0; e < res_slots; ++e) L.slot_of[e] = e;
        CK(cudaMalloc(&L.slot_of_d, (size_t) ne * sizeof(int32_t)));
        CK(cudaMemcpy(L.slot_of_d, L.slot_of.data(), (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice));

        total_vram_used += lyr_sz;
        total_res_slots += res_slots;

        if (l % 10 == 0 || l == 48) {
            std::fprintf(stderr, "strata-flint: packed layer %zu / 48 (%d / %lld experts in VRAM)\n",
                         l, res_slots, (long long) ne);
        }
    }

    std::fprintf(stderr, "strata-flint: High-Speed VRAM Layout: ~%d / %lld experts per layer resident (%.2f GiB VRAM, %.1f%% resident, loaded in %.2f s)\n",
                 total_res_slots / 47, (long long) ne, (double) total_vram_used / (1 << 30),
                 (double) total_res_slots / (47 * ne) * 100.0, ms_since(t0) / 1000.0);
    return true;
}

// ---------------------------------------------------------------- Server Protocol (--serve) & Sampling

struct Req {
    long long max_new = 128;
    std::vector<int32_t> ids;
    strata::flint::Sampling sampling;
};

bool parse_gen(const std::string& line, Req& r) {
    const char* p = line.c_str() + 4; // skip "GEN "
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            const char* idp = tok.c_str();
            while (*idp) {
                char* next_p = nullptr;
                long x = std::strtol(idp, &next_p, 10);
                if (next_p == idp) break;
                r.ids.push_back((int32_t) x);
                idp = next_p;
                while (*idp == ',' || *idp == ' ') ++idp;
            }
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.sampling.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.sampling.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.sampling.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.sampling.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.sampling.seed = std::strtoull(v.c_str(), nullptr, 10);
    }
    return !r.ids.empty();
}

int sample_logits(std::vector<float>& lg, const strata::flint::Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    strata::flint::apply_penalties(lg, s, hist);
    if (s.temperature <= 0.0f) {
        return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    }
    const float inv_t = 1.0f / s.temperature;
    for (float& v : lg) v *= inv_t;
    float max_v = *std::max_element(lg.begin(), lg.end());
    float sum_exp = 0.0f;
    for (float& v : lg) {
        v = std::exp(v - max_v);
        sum_exp += v;
    }
    const float inv_sum = 1.0f / sum_exp;
    for (float& v : lg) v *= inv_sum;

    std::vector<std::pair<float, int>> cands;
    cands.reserve(lg.size());
    for (size_t i = 0; i < lg.size(); ++i) {
        if (lg[i] > 1e-6f) cands.push_back({lg[i], (int) i});
    }
    std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    if (s.top_k > 0 && (int) cands.size() > s.top_k) {
        cands.resize((size_t) s.top_k);
    }
    if (s.top_p > 0.0f && s.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < cands.size(); ++i) {
            cum += cands[i].first;
            cutoff = i + 1;
            if (cum >= s.top_p) break;
        }
        cands.resize(cutoff);
    }

    float norm = 0.0f;
    for (const auto& c : cands) norm += c.first;
    std::uniform_real_distribution<float> dist(0.0f, norm);
    float r = dist(rng);
    float cur = 0.0f;
    for (const auto& c : cands) {
        cur += c.first;
        if (cur >= r) return c.second;
    }
    return cands.empty() ? 0 : cands[0].second;
}

void serve(Engine& E, const Model& m) {
    std::printf("READY %d stop\n", E.ctx);
    std::fflush(stdout);
    std::string line;
    std::mt19937_64 rng(1337);
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) == 0) {
            Req r;
            if (!parse_gen(line, r)) {
                std::printf("DONE 0 0 0.0 0.0 stop 0 0 0 0 0\n");
                std::fflush(stdout);
                continue;
            }
            if (r.sampling.seed != 0) rng.seed(r.sampling.seed);

            const auto t_start = Clock::now();
            const int prompt_len = (int) r.ids.size();

            // Chunked prefill
            int offset = 0;
            const int chunk_size = E.tmax;
            bool ok = true;
            try {
                while (offset < prompt_len) {
                    int cur = std::min(chunk_size, prompt_len - offset);
                    bool is_last = (offset + cur == prompt_len);
                    CK(cudaMemcpy(E.tok, r.ids.data() + offset, cur * sizeof(int32_t), cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(E.pos0, &E.pos, sizeof(int32_t), cudaMemcpyHostToDevice));
                    E.forward(cur, E.pos, is_last ? 1 : 0);
                    CK(cudaStreamSynchronize(E.st));
                    E.pos += cur;
                    offset += cur;
                    if (offset < prompt_len) {
                        double ms = ms_since(t_start);
                        std::printf("PP %d %d %.1f %.1f 0\n", offset, prompt_len, ms,
                                    ms > 0 ? (double) offset / (ms / 1000.0) : 0.0);
                        std::fflush(stdout);
                    }
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata-flint serve: prefill error: %s\n", e.what());
                ok = false;
            }

            if (!ok) {
                std::printf("DONE 0 %d 0.0 0.0 cancel 0 0 0 0 0\n", prompt_len);
                std::fflush(stdout);
                continue;
            }

            const double prefill_ms = ms_since(t_start);

            // Sample first token
            std::vector<float> lg((size_t) m.g.n_vocab);
            CK(cudaMemcpy(lg.data(), E.logits, m.g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<int32_t> hist = r.ids;
            int next_tok = sample_logits(lg, r.sampling, hist, rng);
            hist.push_back(next_tok);
            std::printf("T %d\n", next_tok);
            std::fflush(stdout);

            const auto t_decode = Clock::now();
            int generated = 1;
            const char* finish = "length";
            while (generated < r.max_new && E.pos < E.ctx) {
                CK(cudaMemcpy(E.tok, &next_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(E.pos0, &E.pos, sizeof(int32_t), cudaMemcpyHostToDevice));
                E.forward(1, E.pos, 1);
                CK(cudaStreamSynchronize(E.st));
                E.pos++;

                CK(cudaMemcpy(lg.data(), E.logits, m.g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
                next_tok = sample_logits(lg, r.sampling, hist, rng);
                hist.push_back(next_tok);
                std::printf("T %d\n", next_tok);
                std::fflush(stdout);
                generated++;
                if (next_tok == 0 || next_tok == 150143) { finish = "stop"; break; }
            }
            const double decode_ms = ms_since(t_decode);
            std::printf("DONE %d %d %.1f %.1f %s 0 0 0 0 0\n",
                        generated, prompt_len, prefill_ms, decode_ms, finish);
            std::fflush(stdout);
        }
    }
}

} // namespace

namespace strata::flint {

int run(int argc, char** argv) {
    std::string model_path;
    int max_context = 8192;
    int chunk = 1024;
    double expert_vram = -1.0;
    bool serve_mode = false;
    std::vector<int32_t> test_tokens;
    int max_new = 32;
    float margin = 1.0f;
    strata::flint::Sampling sampling;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" || a == "--native") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (a == "--max-context" || a == "--kv-resident") {
            if (i + 1 < argc) max_context = std::atoi(argv[++i]);
        } else if (a == "--chunk" || a == "--prefill" || a == "--batch-size" || a == "--ubatch-size") {
            if (i + 1 < argc) chunk = std::atoi(argv[++i]);
        } else if (a == "--expert-vram") {
            if (i + 1 < argc) expert_vram = std::atof(argv[++i]);
        } else if (a == "--serve") {
            serve_mode = true;
        } else if (a == "--tokens") {
            if (i + 1 < argc) {
                std::stringstream ss(argv[++i]);
                int tok_id;
                while (ss >> tok_id) {
                    test_tokens.push_back(tok_id);
                    if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
                }
            }
        } else if (a == "--max-new" || a == "-n") {
            if (i + 1 < argc) max_new = std::atoi(argv[++i]);
        } else if (a == "--margin") {
            if (i + 1 < argc) margin = (float) std::atof(argv[++i]);
        } else if (a == "--temp" || a == "--temperature") {
            if (i + 1 < argc) sampling.temperature = (float) std::atof(argv[++i]);
        } else if (a == "--top-p") {
            if (i + 1 < argc) sampling.top_p = (float) std::atof(argv[++i]);
        } else if (a == "--top-k") {
            if (i + 1 < argc) sampling.top_k = std::atoi(argv[++i]);
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "usage: strata-flint --model [model.gguf] [--serve] [--max-context N] [--chunk N]\n");
        return 1;
    }

    Model model;
    std::string err;
    if (!load(model_path, model, err)) {
        std::fprintf(stderr, "strata-flint: load error: %s\n", err.c_str());
        return 1;
    }

    Engine engine;
    engine.margin = margin;
    engine.allocate(model, max_context, chunk);

    if (!place_experts(model, engine, expert_vram, err)) {
        std::fprintf(stderr, "strata-flint: place_experts error: %s\n", err.c_str());
        return 1;
    }

    if (serve_mode) {
        serve(engine, model);
        return 0;
    }

    if (test_tokens.empty()) {
        test_tokens = {1, 150, 2048, 512, 100};
    }

    std::fprintf(stderr, "strata-flint: running prompt of %zu tokens (chunk=%d)...\n", test_tokens.size(), chunk);
    const auto t0 = Clock::now();
    int prompt_len = (int) test_tokens.size();
    int offset = 0;
    while (offset < prompt_len) {
        int cur = std::min(chunk, prompt_len - offset);
        bool is_last = (offset + cur == prompt_len);
        CK(cudaMemcpy(engine.tok, test_tokens.data() + offset, cur * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(engine.pos0, &engine.pos, sizeof(int32_t), cudaMemcpyHostToDevice));
        engine.forward(cur, engine.pos, is_last ? 1 : 0);
        CK(cudaStreamSynchronize(engine.st));
        engine.pos += cur;
        offset += cur;
    }
    const double pp_ms = ms_since(t0);
    const double pp_tps = (double) prompt_len / (pp_ms / 1000.0);
    std::fprintf(stderr, "strata-flint: prefill %d tokens in %.2f ms (%.1f tok/s)\n",
                 prompt_len, pp_ms, pp_tps);

    std::vector<float> lg((size_t) model.g.n_vocab);
    CK(cudaMemcpy(lg.data(), engine.logits, model.g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
    std::mt19937_64 rng(1337);
    std::vector<int32_t> hist = test_tokens;
    int next_tok = sample_logits(lg, sampling, hist, rng);
    hist.push_back(next_tok);

    std::fprintf(stderr, "strata-flint: decoding %d tokens...\n", max_new);
    const auto t_dec = Clock::now();
    for (int gen = 0; gen < max_new; ++gen) {
        CK(cudaMemcpy(engine.tok, &next_tok, sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(engine.pos0, &engine.pos, sizeof(int32_t), cudaMemcpyHostToDevice));
        engine.forward(1, engine.pos, 1);
        CK(cudaStreamSynchronize(engine.st));
        engine.pos++;
        CK(cudaMemcpy(lg.data(), engine.logits, model.g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost));
        next_tok = sample_logits(lg, sampling, hist, rng);
        hist.push_back(next_tok);
        std::printf("%d ", next_tok);
        std::fflush(stdout);
        if (next_tok == 0 || next_tok == 150143) break;
    }
    const double dec_ms = ms_since(t_dec);
    const double dec_tps = (double) max_new / (dec_ms / 1000.0);
    std::fprintf(stderr, "\nstrata-flint: decode %d tokens in %.2f ms (%.1f tok/s)\n",
                 max_new, dec_ms, dec_tps);

    return 0;
}

}  // namespace strata::flint
