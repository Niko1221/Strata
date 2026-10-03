// qwen36_model.cpp - see qwen36_model.hpp.  Reads GGUF with Strata's own reader, runs on Strata's native_* kernels where
// they apply (GEMV on GGUF quantisations, the GDN recurrence and its preprocessing) and on qwen36_kernels.cu elsewhere.
#include "qwen36_model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "moe_group.hpp"
#include "qwen36_kernels.cuh"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace K = strata::kernels;

namespace q36 {
namespace {

#define CK(call)                                                                                             \
    do {                                                                                                     \
        cudaError_t e_ = (call);                                                                             \
        if (e_ != cudaSuccess)                                                                               \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " #call);   \
    } while (0)

inline bool is_float_type(int t) { return t == 0 || t == 1 || t == 30; }
inline size_t float_bytes(int t) { return t == 0 ? 4 : 2; }

// Any GGUF tensor type -> F32 on the host (small tensors and the embedding row).
void dequant_buf(uint32_t type, const uint8_t* src, int64_t n, float* out) {
    switch (type) {
    case 0: strata::dequantize_f32(src, out, (int)n); return;
    case 1: strata::dequantize_f16(src, out, (int)n); return;
    case 30: strata::dequantize_bf16(src, out, (int)n); return;
    default: break;
    }
    int be = 0, bb = 0;
    if (!strata::block_geometry(type, be, bb) || n % be)
        throw std::runtime_error(std::string("cannot dequantise tensor type ") + strata::ggml_type_name(type));
    using Fn = void (*)(const uint8_t*, float*);
    Fn fn = nullptr;
    switch (type) {
    case 2: fn = strata::dequantize_q4_0; break;
    case 6: fn = strata::dequantize_q5_0; break;
    case 8: fn = strata::dequantize_q8_0; break;
    case 11: fn = strata::dequantize_q3_K; break;
    case 12: fn = strata::dequantize_q4_K; break;
    case 13: fn = strata::dequantize_q5_K; break;
    case 14: fn = strata::dequantize_q6_K; break;
    case 20: fn = strata::dequantize_iq4_nl; break;
    case 23: fn = strata::dequantize_iq4_xs; break;
    case 42: fn = strata::dequantize_q2_0; break;
    default: throw std::runtime_error(std::string("no dequantiser for ") + strata::ggml_type_name(type));
    }
    for (int64_t b = 0; b < n / be; ++b) fn(src + b * bb, out + b * be);
}

}  // namespace

struct Model::Impl {
    strata::GgufModel gg;
    const Config& c;
    size_t& vram;
    cudaStream_t st = nullptr;
    std::vector<void*> allocs;
    std::vector<LayerWeights> L;
    Mat output;
    float* out_norm = nullptr;
    const strata::TensorInfo* emb = nullptr;
    const uint8_t* emb_data = nullptr;
    size_t emb_row_bytes = 0;
    // activations
    float *x, *xn, *tmp, *a, *b, *raw, *cs, *alpha, *beta, *gate, *o, *on, *k, *v, *qr, *gt, *ao;
    float *rlog, *wts, *g, *u, *h, *down, *sg, *logits;
    int* ids;
    void *q8x, *q8h;
    size_t q8h_slot = 0;
    int ffm = 0;
    float* h_emb = nullptr;
    int* h_ids = nullptr;
    // state
    std::vector<float*> gdn_state, conv_hist;
    std::vector<__half*> kc, vc;
    std::vector<int> layer_slot;  // index among GDN layers or attention layers
    int max_ctx;

    // ---- block-prefill buffers (token-major [B][width])
    struct BatchBufs {
        float *x, *xn, *tmp, *a, *b, *cs, *alpha, *beta, *gate, *o, *on, *k, *v, *qr, *gt, *ao;
        float *rlog, *wts, *sg, *xg, *G, *U, *H, *Dall, *Dsh, *Gs, *Us, *Hs;
        int *ids, *perm, *pos;
        void *q8b, *q8g, *q8h, *q8s;
        float* h_emb = nullptr;
        int *h_ids = nullptr, *h_perm = nullptr, *h_pos = nullptr;
        std::vector<int> cnt, off, cur;
    } bb;

    Impl(const std::string& path, const Config& cfg, size_t& vr, int mc) : gg(strata::GgufModel::open(path)), c(cfg), vram(vr), max_ctx(mc) {}

    void* dalloc(size_t bytes) {
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("out of GPU memory allocating ") + std::to_string(bytes >> 20) +
                                     " MiB (" + cudaGetErrorString(e) + "); use a smaller quantisation or --max-context");
        allocs.push_back(p);
        vram += bytes;
        return p;
    }
    float* falloc(size_t n) { return (float*)dalloc(n * sizeof(float)); }

    const uint8_t* host_ptr(const strata::TensorInfo& t, const std::string& name) {
        size_t sh = 0;
        gg.find(name, &sh);
        if (!gg.in_bounds(t, sh)) throw std::runtime_error("tensor lies outside its file (truncated download?): " + name);
        return gg.shard(sh).tensor_data(t);
    }

    Mat mat(const std::string& name) {
        const strata::TensorInfo& t = need(gg, name);
        if (t.shape.size() < 2 || t.shape.size() > 3) throw std::runtime_error("unexpected tensor rank: " + name);
        Mat m;
        m.type = (int)t.type;
        m.n_in = (int)t.shape[0];
        m.n_out = (int)t.shape[1];
        m.n_expert = t.shape.size() == 3 ? (int)t.shape[2] : 1;
        if (is_float_type(m.type)) {
            m.expert_stride = (size_t)m.n_in * m.n_out * float_bytes(m.type);
        } else {
            if (!K::native_mmvq_supported(m.type))
                throw std::runtime_error(std::string("tensor ") + name + " has type " + t.type_name() +
                                         ", which the native GEMV does not support (use Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/...)");
            int be = 0, bb = 0;
            strata::block_geometry(t.type, be, bb);
            if (m.n_in % be) throw std::runtime_error("row length not a multiple of the quant block: " + name);
            m.expert_stride = K::native_mmvq_weight_bytes(m.type, m.n_in, m.n_out);
        }
        const uint64_t bytes = strata::tensor_payload_bytes(t);
        if (bytes != m.expert_stride * (uint64_t)m.n_expert)
            throw std::runtime_error("tensor size does not match its shape/type: " + name);
        void* d = dalloc(bytes);
        CK(cudaMemcpy(d, host_ptr(t, name), bytes, cudaMemcpyHostToDevice));
        m.d = d;
        return m;
    }

    // First existing name of `names` as an F32 device vector with exactly `n` elements.
    float* vec(std::initializer_list<std::string> names, int64_t n) {
        for (const std::string& name : names) {
            const strata::TensorInfo* t = gg.find(name);
            if (!t) continue;
            if ((int64_t)t->elements() != n)
                throw std::runtime_error(name + ": expected " + std::to_string(n) + " elements, found " + std::to_string(t->elements()));
            std::vector<float> h((size_t)n);
            dequant_buf(t->type, host_ptr(*t, name), n, h.data());
            float* d = falloc((size_t)n);
            CK(cudaMemcpy(d, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
            return d;
        }
        throw std::runtime_error("tensor not found in the GGUF: " + *names.begin());
    }

    void load() {
        const strata::TensorInfo& e = need(gg, "token_embd.weight");
        emb = &e;
        emb_data = host_ptr(e, "token_embd.weight");
        int be = 0, bb = 0;
        if (!strata::block_geometry(e.type, be, bb) || c.n_embd % be) throw std::runtime_error("unsupported token_embd type");
        emb_row_bytes = (size_t)(c.n_embd / be) * bb;

        const int n = c.n_embd;
        L.resize(c.n_layer);
        layer_slot.resize(c.n_layer);
        int gi = 0, ai = 0;
        for (int il = 0; il < c.n_layer; ++il) {
            LayerWeights& w = L[il];
            w.gdn = c.is_gdn[il];
            layer_slot[il] = w.gdn ? gi++ : ai++;
            w.attn_norm = vec({blk(il, "attn_norm.weight")}, n);
            w.post_norm = vec({blk(il, "post_attention_norm.weight")}, n);
            if (w.gdn) {
                w.qkv = mat(blk(il, "attn_qkv.weight"));
                w.z = mat(blk(il, "attn_gate.weight"));
                w.alpha = mat(blk(il, "ssm_alpha.weight"));
                w.beta = mat(blk(il, "ssm_beta.weight"));
                w.ssm_out = mat(blk(il, "ssm_out.weight"));
                w.conv = vec({blk(il, "ssm_conv1d.weight")}, (int64_t)c.ssm_conv * c.ssm_qkv);
                w.ssm_a = vec({blk(il, "ssm_a"), blk(il, "ssm_a.weight")}, c.ssm_hv);
                w.dt = vec({blk(il, "ssm_dt.bias"), blk(il, "ssm_dt"), blk(il, "ssm_dt.weight")}, c.ssm_hv);
                w.ssm_norm = vec({blk(il, "ssm_norm.weight")}, c.ssm_S);
            } else {
                w.q = mat(blk(il, "attn_q.weight"));
                w.k = mat(blk(il, "attn_k.weight"));
                w.v = mat(blk(il, "attn_v.weight"));
                w.o = mat(blk(il, "attn_output.weight"));
                w.qn = vec({blk(il, "attn_q_norm.weight")}, c.head_dim);
                w.kn = vec({blk(il, "attn_k_norm.weight")}, c.head_dim);
            }
            w.gate_inp = mat(blk(il, "ffn_gate_inp.weight"));
            w.gate_exps = mat(blk(il, "ffn_gate_exps.weight"));
            w.up_exps = mat(blk(il, "ffn_up_exps.weight"));
            w.down_exps = mat(blk(il, "ffn_down_exps.weight"));
            w.sh_gate = mat(blk(il, "ffn_gate_shexp.weight"));
            w.sh_up = mat(blk(il, "ffn_up_shexp.weight"));
            w.sh_down = mat(blk(il, "ffn_down_shexp.weight"));
            w.sh_gate_inp = vec({blk(il, "ffn_gate_inp_shexp.weight")}, n);
            if (il % 8 == 7 || il == c.n_layer - 1)
                std::fprintf(stderr, "[qwen36] loaded layer %d/%d  (%.1f GiB on GPU)\n", il + 1, c.n_layer,
                             (double)vram / (1024.0 * 1024.0 * 1024.0));
        }
        out_norm = vec({"output_norm.weight"}, n);
        output = mat(gg.find("output.weight") ? "output.weight" : "token_embd.weight");
        if (output.n_in != n || output.n_out != c.n_vocab) throw std::runtime_error("output.weight shape");
    }

    void alloc_runtime() {
        const int n = c.n_embd;
        const int qkvw = std::max(c.ssm_qkv, 2 * c.n_head * c.head_dim);
        const int bw = std::max(c.ssm_inner, c.n_head * c.head_dim);
        x = falloc(n); xn = falloc(n); tmp = falloc(n);
        a = falloc(qkvw); b = falloc(bw); raw = falloc(std::max(c.ssm_qkv, 1)); cs = falloc(std::max(c.ssm_qkv, 1));
        alpha = falloc(std::max(c.ssm_hv, 1)); beta = falloc(std::max(c.ssm_hv, 1)); gate = falloc(std::max(c.ssm_hv, 1));
        o = falloc(std::max(c.ssm_inner, 1)); on = falloc(std::max(c.ssm_inner, 1));
        k = falloc(std::max(c.n_head_kv * c.head_dim, 1)); v = falloc(std::max(c.n_head_kv * c.head_dim, 1));
        qr = falloc(std::max(c.n_head * c.head_dim, 1)); gt = falloc(std::max(c.n_head * c.head_dim, 1));
        ao = falloc(std::max(c.n_head * c.head_dim, 1));
        ffm = std::max(c.n_ff_exp, c.n_ff_shexp);
        const int slots = c.n_used + 1;
        rlog = falloc(c.n_expert); wts = falloc(c.n_used); sg = falloc(1);
        g = falloc((size_t)slots * ffm); u = falloc((size_t)slots * ffm); h = falloc((size_t)slots * ffm);
        down = falloc((size_t)slots * n);
        ids = (int*)dalloc(sizeof(int) * c.n_used);
        logits = falloc(c.n_vocab);
        const int q8n = std::max({n, c.ssm_inner, c.n_head * c.head_dim});
        q8x = dalloc(K::native_q8_1_bytes(q8n, 1));
        q8h_slot = K::native_q8_1_bytes(ffm, 1);
        q8h = dalloc(q8h_slot * slots);
        CK(cudaMallocHost((void**)&h_emb, (size_t)n * 4));
        CK(cudaMallocHost((void**)&h_ids, sizeof(int) * c.n_used));
        for (int il = 0; il < c.n_layer; ++il) {
            if (L[il].gdn) {
                gdn_state.push_back(falloc((size_t)c.ssm_S * c.ssm_hv * c.ssm_S));
                conv_hist.push_back(falloc((size_t)c.ssm_qkv * (c.ssm_conv - 1)));
            } else {
                const size_t kv = (size_t)c.n_head_kv * max_ctx * c.head_dim * sizeof(__half);
                kc.push_back((__half*)dalloc(kv));
                vc.push_back((__half*)dalloc(kv));
            }
        }
        alloc_batch();
    }

    void alloc_batch() {
        constexpr int MB = Model::kMaxBatch;
        const int n = c.n_embd, ku = c.n_used, np = MB * ku;
        const int qkvw = std::max(c.ssm_qkv, 2 * c.n_head * c.head_dim);
        const int bw = std::max(c.ssm_inner, c.n_head * c.head_dim);
        bb.x = falloc((size_t)MB * n); bb.xn = falloc((size_t)MB * n); bb.tmp = falloc((size_t)MB * n);
        bb.a = falloc((size_t)MB * qkvw); bb.b = falloc((size_t)MB * bw);
        bb.cs = falloc((size_t)MB * std::max(c.ssm_qkv, 1));
        bb.alpha = falloc((size_t)MB * std::max(c.ssm_hv, 1)); bb.beta = falloc((size_t)MB * std::max(c.ssm_hv, 1));
        bb.gate = falloc((size_t)MB * std::max(c.ssm_hv, 1));
        bb.o = falloc((size_t)MB * std::max(c.ssm_inner, 1)); bb.on = falloc((size_t)MB * std::max(c.ssm_inner, 1));
        const int kvw = std::max(c.n_head_kv * c.head_dim, 1), qw = std::max(c.n_head * c.head_dim, 1);
        bb.k = falloc((size_t)MB * kvw); bb.v = falloc((size_t)MB * kvw);
        bb.qr = falloc((size_t)MB * qw); bb.gt = falloc((size_t)MB * qw); bb.ao = falloc((size_t)MB * qw);
        bb.rlog = falloc((size_t)MB * c.n_expert); bb.wts = falloc(np); bb.sg = falloc(MB);
        bb.ids = (int*)dalloc(sizeof(int) * np); bb.perm = (int*)dalloc(sizeof(int) * np); bb.pos = (int*)dalloc(sizeof(int) * np);
        bb.xg = falloc((size_t)np * n);
        bb.G = falloc((size_t)np * c.n_ff_exp); bb.U = falloc((size_t)np * c.n_ff_exp); bb.H = falloc((size_t)np * c.n_ff_exp);
        bb.Dall = falloc((size_t)np * n); bb.Dsh = falloc((size_t)MB * n);
        bb.Gs = falloc((size_t)MB * c.n_ff_shexp); bb.Us = falloc((size_t)MB * c.n_ff_shexp);
        bb.Hs = falloc((size_t)MB * c.n_ff_shexp);
        const int widest = std::max({n, c.ssm_inner, c.n_head * c.head_dim});
        bb.q8b = dalloc(K::native_q8_1_bytes(widest * MB, 1));
        bb.q8g = dalloc(K::native_q8_1_bytes(n * np, 1));
        bb.q8h = dalloc(K::native_q8_1_bytes(c.n_ff_exp * np, 1));
        bb.q8s = dalloc(K::native_q8_1_bytes(c.n_ff_shexp * MB, 1));
        CK(cudaMallocHost((void**)&bb.h_emb, (size_t)MB * n * 4));
        CK(cudaMallocHost((void**)&bb.h_ids, sizeof(int) * np));
        CK(cudaMallocHost((void**)&bb.h_perm, sizeof(int) * np));
        CK(cudaMallocHost((void**)&bb.h_pos, sizeof(int) * np));
        bb.cnt.assign(c.n_expert, 0); bb.off.assign(c.n_expert + 1, 0); bb.cur.assign(c.n_expert, 0);
    }

    // ---- multi-column helpers
    void quantc(const float* X, int n_in, int ncols, void* q8) {   // ncols columns == one vector of n_in*ncols (blocks of 32)
        if (ncols > 0) K::native_quantize_q8_1(X, q8, n_in * ncols, 1, st);
    }
    // Y[col*n_out + row] for `ncols` columns; X columns are n_in apart, q8 columns native_q8_1_bytes(n_in) apart.
    void mvc(const Mat& m, int e, const float* X, const void* q8, float* Y, int ncols) {
        const char* w = static_cast<const char*>(m.d) + (size_t)e * m.expert_stride;
        if (is_float_type(m.type)) { q36::gemv_float_cols(m.type, w, X, Y, m.n_in, m.n_out, ncols, st); return; }
        const size_t col_bytes = K::native_q8_1_bytes(m.n_in, 1);
        for (int c0 = 0; c0 < ncols; c0 += 8) {
            const int nc = std::min(8, ncols - c0);
            K::native_mmvq(m.type, w, static_cast<const char*>(q8) + (size_t)c0 * col_bytes, Y + (size_t)c0 * m.n_out, m.n_in,
                           m.n_out, nc, st);
        }
    }

    void gdn_layer_b(int il, int B) {
        const LayerWeights& w = L[il];
        const int gi = layer_slot[il], hk = c.ssm_hk, hv = c.ssm_hv, S = c.ssm_S;
        quantc(bb.xn, c.n_embd, B, bb.q8b);
        mvc(w.qkv, 0, bb.xn, bb.q8b, bb.a, B);
        mvc(w.z, 0, bb.xn, bb.q8b, bb.b, B);
        mvc(w.alpha, 0, bb.xn, bb.q8b, bb.alpha, B);
        mvc(w.beta, 0, bb.xn, bb.q8b, bb.beta, B);
        q36::gdn_conv_silu_b(conv_hist[gi], bb.a, w.conv, bb.cs, c.ssm_qkv, B, st);
        q36::gdn_l2norm_qk_b(bb.cs, c.ssm_qkv, 2 * hk, S, B, 1e-6f, st);
        q36::gdn_gate_beta_b(bb.alpha, w.dt, w.ssm_a, bb.gate, bb.beta, hv, B, st);
        q36::gdn_step_b(gdn_state[gi], bb.cs, bb.gate, bb.beta, bb.o, hk, hv, c.ssm_qkv, B, st);
        q36::gdn_out_norm_silu(bb.on, bb.o, bb.b, w.ssm_norm, B * hv, S, c.eps, st);
        quantc(bb.on, c.ssm_inner, B, bb.q8b);
        mvc(w.ssm_out, 0, bb.on, bb.q8b, bb.tmp, B);
    }

    void attn_layer_b(int il, int pos0, int B) {
        const LayerWeights& w = L[il];
        const int ai = layer_slot[il], nh = c.n_head, nkv = c.n_head_kv, hd = c.head_dim;
        quantc(bb.xn, c.n_embd, B, bb.q8b);
        mvc(w.q, 0, bb.xn, bb.q8b, bb.a, B);
        mvc(w.k, 0, bb.xn, bb.q8b, bb.k, B);
        mvc(w.v, 0, bb.xn, bb.q8b, bb.v, B);
        q36::attn_prep_q_b(bb.qr, bb.gt, bb.a, w.qn, nh, hd, c.rot_dim, c.rope_theta, pos0, B, c.eps, st);
        q36::attn_prep_kv_b(kc[ai], vc[ai], bb.k, bb.v, w.kn, nkv, hd, c.rot_dim, c.rope_theta, pos0, B, max_ctx, c.eps, st);
        q36::attn_decode_b(bb.ao, bb.qr, bb.gt, kc[ai], vc[ai], nh, nkv, hd, pos0, B, max_ctx, st);
        quantc(bb.ao, nh * hd, B, bb.q8b);
        mvc(w.o, 0, bb.ao, bb.q8b, bb.tmp, B);
    }

    // x += MoE(xn) for B tokens.  Pairs (token, slot) are sorted by expert so each active expert runs once (in chunks of 8
    // columns, the native GEMV limit) on all the tokens routed to it.
    void moe_layer_b(int il, int B) {
        const LayerWeights& w = L[il];
        const int n = c.n_embd, ku = c.n_used, fe = c.n_ff_exp, fs = c.n_ff_shexp, ne = c.n_expert, np = B * ku;
        quantc(bb.xn, n, B, bb.q8b);
        mvc(w.gate_inp, 0, bb.xn, bb.q8b, bb.rlog, B);
        q36::router_topk_rows(bb.rlog, B, ne, ku, bb.ids, bb.wts, st);
        CK(cudaMemcpyAsync(bb.h_ids, bb.ids, sizeof(int) * np, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));  // the host needs the ids to group tokens by expert
        q36::group_pairs(bb.h_ids, B, ku, ne, bb.cnt, bb.off, bb.cur, bb.h_perm, bb.h_pos);
        CK(cudaMemcpyAsync(bb.perm, bb.h_perm, sizeof(int) * np, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(bb.pos, bb.h_pos, sizeof(int) * np, cudaMemcpyHostToDevice, st));
        q36::gather_rows(bb.xg, bb.xn, bb.perm, np, n, st);
        quantc(bb.xg, n, np, bb.q8g);
        const size_t q8n = K::native_q8_1_bytes(n, 1), q8f = K::native_q8_1_bytes(fe, 1);
        for (int e = 0; e < ne; ++e) {
            const int o = bb.off[e], m = bb.cnt[e];
            if (!m) continue;
            const void* qg = static_cast<const char*>(bb.q8g) + (size_t)o * q8n;
            mvc(w.gate_exps, e, bb.xg + (size_t)o * n, qg, bb.G + (size_t)o * fe, m);
            mvc(w.up_exps, e, bb.xg + (size_t)o * n, qg, bb.U + (size_t)o * fe, m);
        }
        q36::silu_mul(bb.H, bb.G, bb.U, np * fe, st);
        quantc(bb.H, fe, np, bb.q8h);
        for (int e = 0; e < ne; ++e) {
            const int o = bb.off[e], m = bb.cnt[e];
            if (!m) continue;
            mvc(w.down_exps, e, bb.H + (size_t)o * fe, static_cast<const char*>(bb.q8h) + (size_t)o * q8f,
                bb.Dall + (size_t)o * n, m);
        }
        mvc(w.sh_gate, 0, bb.xn, bb.q8b, bb.Gs, B);
        mvc(w.sh_up, 0, bb.xn, bb.q8b, bb.Us, B);
        q36::silu_mul(bb.Hs, bb.Gs, bb.Us, B * fs, st);
        quantc(bb.Hs, fs, B, bb.q8s);
        mvc(w.sh_down, 0, bb.Hs, bb.q8s, bb.Dsh, B);
        q36::dot_rows(bb.sg, w.sh_gate_inp, bb.xn, B, n, st);
        q36::moe_combine_b(bb.x, bb.Dall, bb.wts, bb.pos, bb.Dsh, bb.sg, B, ku, n, st);
    }

    // ------------------------------------------------------------------ helpers
    void quant(const float* in, int n_in, void* q8) { K::native_quantize_q8_1(in, q8, n_in, 1, st); }
    void mv(const Mat& m, int e, const float* in, const void* q8, float* y) {
        const char* w = static_cast<const char*>(m.d) + (size_t)e * m.expert_stride;
        if (is_float_type(m.type)) q36::gemv_float(m.type, w, in, y, m.n_in, m.n_out, st);
        else K::native_mmvq(m.type, w, q8, y, m.n_in, m.n_out, 1, st);
    }

    void gdn_layer(int il) {
        const LayerWeights& w = L[il];
        const int gi = layer_slot[il], hk = c.ssm_hk, hv = c.ssm_hv, S = c.ssm_S;
        quant(xn, c.n_embd, q8x);
        mv(w.qkv, 0, xn, q8x, a);
        mv(w.z, 0, xn, q8x, b);
        mv(w.alpha, 0, xn, q8x, alpha);
        mv(w.beta, 0, xn, q8x, beta);
        K::native_gdn_conv_silu(conv_hist[gi], a, w.conv, raw, cs, c.ssm_qkv, c.ssm_conv, st);
        K::native_gdn_l2_norm(cs, hk, S, 1e-6f, st);                 // q
        K::native_gdn_l2_norm(cs + (size_t)hk * S, hk, S, 1e-6f, st);  // k
        K::native_gdn_gate(alpha, w.dt, w.ssm_a, gate, hv, st);
        K::native_gdn_beta_gate(beta, hv, st);
        strata::kernels::GdnShapes sh;
        sh.S = S; sh.h_k = hk; sh.h_v = hv;
        K::native_gdn_step(gdn_state[gi], cs, cs + (size_t)hk * S, cs + (size_t)2 * hk * S, gate, beta, o, sh, st);
        q36::gdn_out_norm_silu(on, o, b, w.ssm_norm, hv, S, c.eps, st);
        quant(on, c.ssm_inner, q8x);
        mv(w.ssm_out, 0, on, q8x, tmp);
    }

    void attn_layer(int il, int pos) {
        const LayerWeights& w = L[il];
        const int ai = layer_slot[il], nh = c.n_head, nkv = c.n_head_kv, hd = c.head_dim;
        quant(xn, c.n_embd, q8x);
        mv(w.q, 0, xn, q8x, a);
        mv(w.k, 0, xn, q8x, k);
        mv(w.v, 0, xn, q8x, v);
        q36::attn_prep_q(qr, gt, a, w.qn, nh, hd, c.rot_dim, c.rope_theta, pos, c.eps, st);
        q36::attn_prep_kv(kc[ai], vc[ai], k, v, w.kn, nkv, hd, c.rot_dim, c.rope_theta, pos, max_ctx, c.eps, st);
        q36::attn_decode(ao, qr, gt, kc[ai], vc[ai], nh, nkv, hd, pos + 1, max_ctx, st);
        quant(ao, nh * hd, q8x);
        mv(w.o, 0, ao, q8x, tmp);
    }

    // x += MoE(xn)
    void moe_layer(int il) {
        const LayerWeights& w = L[il];
        const int n = c.n_embd, ku = c.n_used, fe = c.n_ff_exp, fs = c.n_ff_shexp;
        quant(xn, n, q8x);
        mv(w.gate_inp, 0, xn, q8x, rlog);
        q36::router_topk(rlog, c.n_expert, ku, ids, wts, st);
        CK(cudaMemcpyAsync(h_ids, ids, sizeof(int) * ku, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));  // the host needs the 8 ids to address the experts
        for (int j = 0; j < ku; ++j) {
            const int e = h_ids[j];
            if (e < 0 || e >= c.n_expert) throw std::runtime_error("router returned an invalid expert id");
            mv(w.gate_exps, e, xn, q8x, g + (size_t)j * ffm);
            mv(w.up_exps, e, xn, q8x, u + (size_t)j * ffm);
            q36::silu_mul(h + (size_t)j * ffm, g + (size_t)j * ffm, u + (size_t)j * ffm, fe, st);
            quant(h + (size_t)j * ffm, fe, static_cast<char*>(q8h) + (size_t)j * q8h_slot);
            mv(w.down_exps, e, h + (size_t)j * ffm, static_cast<char*>(q8h) + (size_t)j * q8h_slot, down + (size_t)j * n);
        }
        mv(w.sh_gate, 0, xn, q8x, g + (size_t)ku * ffm);
        mv(w.sh_up, 0, xn, q8x, u + (size_t)ku * ffm);
        q36::silu_mul(h + (size_t)ku * ffm, g + (size_t)ku * ffm, u + (size_t)ku * ffm, fs, st);
        quant(h + (size_t)ku * ffm, fs, static_cast<char*>(q8h) + (size_t)ku * q8h_slot);
        mv(w.sh_down, 0, h + (size_t)ku * ffm, static_cast<char*>(q8h) + (size_t)ku * q8h_slot, down + (size_t)ku * n);
        q36::dot_f32(sg, w.sh_gate_inp, xn, n, st);
        q36::moe_combine(x, down, wts, down + (size_t)ku * n, sg, ku, n, st);
    }
};

Model::Model(const std::string& path, int max_ctx) : max_ctx_(max_ctx) {
    if (max_ctx < 1) throw std::runtime_error("max_ctx must be >= 1");
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (!ndev) throw std::runtime_error("no CUDA device");
    p_.reset(new Impl(path, cfg_, vram_, max_ctx));
    cfg_ = load_config(p_->gg);
    std::fprintf(stderr, "[qwen36] %s\n", cfg_.describe().c_str());
    CK(cudaStreamCreate(&p_->st));
    p_->load();
    p_->alloc_runtime();
    h_logits_.assign(cfg_.n_vocab, 0.f);
    reset();
    std::fprintf(stderr, "[qwen36] ready: %.2f GiB on GPU, context %d\n", (double)vram_ / (1024.0 * 1024.0 * 1024.0), max_ctx_);
}

Model::~Model() {
    if (!p_) return;
    if (p_->st) cudaStreamSynchronize(p_->st);
    for (void* q : p_->allocs) cudaFree(q);
    if (p_->h_emb) cudaFreeHost(p_->h_emb);
    if (p_->h_ids) cudaFreeHost(p_->h_ids);
    if (p_->bb.h_emb) cudaFreeHost(p_->bb.h_emb);
    if (p_->bb.h_ids) cudaFreeHost(p_->bb.h_ids);
    if (p_->bb.h_perm) cudaFreeHost(p_->bb.h_perm);
    if (p_->bb.h_pos) cudaFreeHost(p_->bb.h_pos);
    if (p_->st) cudaStreamDestroy(p_->st);
}

void Model::reset() {
    for (float* s : p_->gdn_state) CK(cudaMemsetAsync(s, 0, (size_t)cfg_.ssm_S * cfg_.ssm_hv * cfg_.ssm_S * 4, p_->st));
    for (float* s : p_->conv_hist) CK(cudaMemsetAsync(s, 0, (size_t)cfg_.ssm_qkv * (cfg_.ssm_conv - 1) * 4, p_->st));
    CK(cudaStreamSynchronize(p_->st));
    pos_ = 0;
}

void Model::forward(int token, bool want_logits) {
    Impl& m = *p_;
    const Config& c = cfg_;
    if (pos_ >= max_ctx_) throw std::runtime_error("context is full");
    if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id out of range");
    dequant_buf(m.emb->type, m.emb_data + (size_t)token * m.emb_row_bytes, c.n_embd, m.h_emb);
    CK(cudaMemcpyAsync(m.x, m.h_emb, (size_t)c.n_embd * 4, cudaMemcpyHostToDevice, m.st));
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights& w = m.L[il];
        q36::rmsnorm(m.xn, m.x, w.attn_norm, 1, c.n_embd, c.eps, m.st);
        if (w.gdn) m.gdn_layer(il);
        else m.attn_layer(il, pos_);
        q36::add_inplace(m.x, m.tmp, c.n_embd, m.st);
        q36::rmsnorm(m.xn, m.x, w.post_norm, 1, c.n_embd, c.eps, m.st);
        m.moe_layer(il);
    }
    if (want_logits) {
        q36::rmsnorm(m.xn, m.x, m.out_norm, 1, c.n_embd, c.eps, m.st);
        m.quant(m.xn, c.n_embd, m.q8x);
        m.mv(m.output, 0, m.xn, m.q8x, m.logits);
        CK(cudaMemcpyAsync(h_logits_.data(), m.logits, (size_t)c.n_vocab * 4, cudaMemcpyDeviceToHost, m.st));
    }
    CK(cudaStreamSynchronize(m.st));
    CK(cudaGetLastError());
    ++pos_;
}

void Model::forward_batch(const int* tokens, int n, bool want_logits) {
    Impl& m = *p_;
    const Config& c = cfg_;
    if (n < 1 || n > kMaxBatch) throw std::runtime_error("forward_batch: batch size out of range");
    if (pos_ + n > max_ctx_) throw std::runtime_error("context is full");
    for (int i = 0; i < n; ++i) {
        if (tokens[i] < 0 || tokens[i] >= c.n_vocab) throw std::runtime_error("token id out of range");
        dequant_buf(m.emb->type, m.emb_data + (size_t)tokens[i] * m.emb_row_bytes, c.n_embd,
                    m.bb.h_emb + (size_t)i * c.n_embd);
    }
    CK(cudaMemcpyAsync(m.bb.x, m.bb.h_emb, (size_t)n * c.n_embd * 4, cudaMemcpyHostToDevice, m.st));
    const int pos0 = pos_;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights& w = m.L[il];
        q36::rmsnorm(m.bb.xn, m.bb.x, w.attn_norm, n, c.n_embd, c.eps, m.st);
        if (w.gdn) m.gdn_layer_b(il, n);
        else m.attn_layer_b(il, pos0, n);
        q36::add_inplace(m.bb.x, m.bb.tmp, n * c.n_embd, m.st);
        q36::rmsnorm(m.bb.xn, m.bb.x, w.post_norm, n, c.n_embd, c.eps, m.st);
        m.moe_layer_b(il, n);
    }
    if (want_logits) {
        q36::rmsnorm(m.xn, m.bb.x + (size_t)(n - 1) * c.n_embd, m.out_norm, 1, c.n_embd, c.eps, m.st);
        m.quant(m.xn, c.n_embd, m.q8x);
        m.mv(m.output, 0, m.xn, m.q8x, m.logits);
        CK(cudaMemcpyAsync(h_logits_.data(), m.logits, (size_t)c.n_vocab * 4, cudaMemcpyDeviceToHost, m.st));
    }
    CK(cudaStreamSynchronize(m.st));
    CK(cudaGetLastError());
    pos_ += n;
}

}  // namespace q36
