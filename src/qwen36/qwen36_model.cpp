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
#include "strata/kernels/iq_kernels.hpp"
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
    // ---- grouped experts: one launch per layer, grouping on the device, no host synchronisation (Q36_GROUPED=0 turns it off)
    bool grouped_enabled = true;
    struct Grouped {
        unsigned long long* ptr = nullptr;
        int *start = nullptr, *ngroups = nullptr, *tok = nullptr, *dst = nullptr, *err = nullptr;
        void* scratch = nullptr;
        int* h_err = nullptr;     // pinned copy of *err
    } gx;
    int n_grouped_layers = 0;
    std::vector<std::string> type_notes;   // "layer 3: gate/up IQ4_NL, down Q6_K -> per-expert path" for the load summary
    // state
    std::vector<float*> gdn_state, conv_hist;
    std::vector<void*> kc, vc;      // per attention layer (+ the MTP layer): K and V caches, layout chosen by kv8
    int kv8 = 0;                    // 0: F16 cells, 1: int8 codes + one F16 scale per 32 values
    bool kv_unified = false;        // one allocation for every layer's K and V
    char* kv_pool = nullptr;
    size_t kv_off = 0, kv_cache_bytes = 0;
    std::vector<int> layer_slot;  // index among GDN layers or attention layers
    int max_ctx;

    // ---- MTP head (blk.<n_layer>.nextn.*) and the checkpoints speculative decoding rolls back to
    bool want_mtp = false, mtp_on = false, spec_ckpt = false, have_carry = false;
    int spec_max = 3;      // most draft tokens per round (verify batch = spec_max + 1 rows)
    int variant = 1;       // bit0: MTP input hidden is post final norm; bit1: concat order [hidden|emb]; bit2: nextn norms get +1
    std::vector<float*> ck_state, ck_hist;
    Mat eh_proj, mtp_head;
    float *enorm[2] = {nullptr, nullptr}, *hnorm[2] = {nullptr, nullptr}, *shnorm[2] = {nullptr, nullptr};
    const strata::TensorInfo* memb = nullptr;      // optional nextn.embed_tokens
    const uint8_t* memb_data = nullptr;
    size_t memb_row_bytes = 0;
    float *hcap = nullptr, *carry = nullptr, *me = nullptr, *mh = nullptr, *mhs = nullptr, *mcat = nullptr;
    float *mtp_out = nullptr, *mtp_hn = nullptr, *vlog = nullptr;
    std::vector<int>* probe_preds = nullptr;       // --selftest-mtp: MTP argmax per position
    std::vector<float> mlog;
    std::vector<float>* h_vlog = nullptr;          // host copy target for vlog (owned by Model)

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
    float* vec(std::initializer_list<std::string> names, int64_t n, float add = 0.f) {
        for (const std::string& name : names) {
            const strata::TensorInfo* t = gg.find(name);
            if (!t) continue;
            if ((int64_t)t->elements() != n)
                throw std::runtime_error(name + ": expected " + std::to_string(n) + " elements, found " + std::to_string(t->elements()));
            std::vector<float> h((size_t)n);
            dequant_buf(t->type, host_ptr(*t, name), n, h.data());
            if (add != 0.f) for (float& f : h) f += add;
            float* d = falloc((size_t)n);
            CK(cudaMemcpy(d, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
            return d;
        }
        throw std::runtime_error("tensor not found in the GGUF: " + *names.begin());
    }

    // Repacks layer il's gate/up/down expert tensors into one blob per expert ([gate rows | up rows | down rows], the layout
    // native_expert_grouped reads).  One cudaMemcpy2D per tensor.  Returns false (and loads nothing) when the layer's types or
    // sizes do not fit the grouped kernel: the caller then loads the three tensors for the per-expert path.
    bool load_grouped(int il, LayerWeights& w) {
        if (!grouped_enabled) return false;
        const strata::TensorInfo& tg = need(gg, blk(il, "ffn_gate_exps.weight"));
        const strata::TensorInfo& tu = need(gg, blk(il, "ffn_up_exps.weight"));
        const strata::TensorInfo& td = need(gg, blk(il, "ffn_down_exps.weight"));
        const int gt = (int)tg.type, ut = (int)tu.type, dt = (int)td.type;
        const int n = c.n_embd, ff = c.n_ff_exp, ne = c.n_expert;
        auto note = [&](const char* why) {
            type_notes.push_back("layer " + std::to_string(il) + ": gate " + tg.type_name() + ", up " + tu.type_name() +
                                 ", down " + td.type_name() + " -> per-expert path (" + why + ")");
            return false;
        };
        if (gt != ut) return note("gate and up differ");
        if (!K::native_expert_supported(gt, dt, n, ff)) return note("type pair not in the grouped kernel");
        if (td.shape.size() != 3 || (int)td.shape[0] != ff || (int)td.shape[1] != n || (int)td.shape[2] != ne)
            return note("down shape");
        const K::NativeExpertLayout xl = K::native_expert_layout(gt, dt, n, ff);
        const size_t gbytes = (size_t)ff * xl.gu_row, dbytes = (size_t)n * xl.d_row;
        if (strata::tensor_payload_bytes(tg) != gbytes * ne || strata::tensor_payload_bytes(tu) != gbytes * ne ||
            strata::tensor_payload_bytes(td) != dbytes * ne)
            return note("tensor size differs from the row layout");
        const size_t stride = (xl.bytes + 255) & ~(size_t)255;
        char* base = static_cast<char*>(dalloc(stride * (size_t)ne));
        CK(cudaMemcpy2D(base, stride, host_ptr(tg, blk(il, "ffn_gate_exps.weight")), gbytes, gbytes, ne, cudaMemcpyHostToDevice));
        CK(cudaMemcpy2D(base + xl.up_off, stride, host_ptr(tu, blk(il, "ffn_up_exps.weight")), gbytes, gbytes, ne,
                        cudaMemcpyHostToDevice));
        CK(cudaMemcpy2D(base + xl.down_off, stride, host_ptr(td, blk(il, "ffn_down_exps.weight")), dbytes, dbytes, ne,
                        cudaMemcpyHostToDevice));
        w.grouped = true;
        w.blobs = base;
        w.blob_stride = stride;
        w.gu_type = gt;
        w.d_type = dt;
        return true;
    }

    // Everything one decoder layer needs (attention or GDN part, router, experts, shared expert).
    void load_layer(int il, LayerWeights& w) {
        const int n = c.n_embd;
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
        if (load_grouped(il, w)) ++n_grouped_layers;
        else {
            w.gate_exps = mat(blk(il, "ffn_gate_exps.weight"));
            w.up_exps = mat(blk(il, "ffn_up_exps.weight"));
            w.down_exps = mat(blk(il, "ffn_down_exps.weight"));
        }
        w.sh_gate = mat(blk(il, "ffn_gate_shexp.weight"));
        w.sh_up = mat(blk(il, "ffn_up_shexp.weight"));
        w.sh_down = mat(blk(il, "ffn_down_shexp.weight"));
        w.sh_gate_inp = vec({blk(il, "ffn_gate_inp_shexp.weight")}, n);
    }

    void load() {
        if (const char* g = std::getenv("Q36_GROUPED")) grouped_enabled = !(g[0] == '0');
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
            load_layer(il, w);
            if (il % 8 == 7 || il == c.n_layer - 1)
                std::fprintf(stderr, "[qwen36] loaded layer %d/%d  (%.1f GiB on GPU)\n", il + 1, c.n_layer,
                             (double)vram / (1024.0 * 1024.0 * 1024.0));
        }
        out_norm = vec({"output_norm.weight"}, n);
        output = mat(gg.find("output.weight") ? "output.weight" : "token_embd.weight");
        if (output.n_in != n || output.n_out != c.n_vocab) throw std::runtime_error("output.weight shape");
        load_mtp();
        std::fprintf(stderr, "[qwen36] grouped experts: %d of %d layers%s\n", n_grouped_layers, c.n_layer,
                     grouped_enabled ? "" : " (disabled by Q36_GROUPED=0)");
        for (size_t i = 0; i < type_notes.size() && i < 6; ++i) std::fprintf(stderr, "[qwen36]   %s\n", type_notes[i].c_str());
        if (type_notes.size() > 6) std::fprintf(stderr, "[qwen36]   ... and %zu more layers on the per-expert path\n", type_notes.size() - 6);
    }


    // The MTP ("nextn") head: one extra attention+MoE layer blk.<n_layer> with its own KV cache, fed concat(norm(embed(next
    // token)), norm(hidden)) through eh_proj.  Optional: any missing or odd tensor just leaves MTP off.
    void load_mtp() {
        if (!want_mtp) return;
        const int il = c.n_layer, n = c.n_embd;
        if (!gg.find(blk(il, "nextn.eh_proj.weight"))) {
            std::fprintf(stderr, "[qwen36] --mtp: this GGUF has no blk.%d.nextn.* tensors (many quantisations drop the MTP head); "
                                 "MTP is off\\n", il);
            return;
        }
        try {
            L.resize(il + 1);
            layer_slot.resize(il + 1);
            LayerWeights& w = L[il];
            w.gdn = false;
            layer_slot[il] = c.n_attn;          // one KV slot after the real attention layers
            load_layer(il, w);
            eh_proj = mat(blk(il, "nextn.eh_proj.weight"));
            if (eh_proj.n_in != 2 * n || eh_proj.n_out != n) throw std::runtime_error("nextn.eh_proj.weight shape");
            for (int v = 0; v < 2; ++v) {      // [1] = the same norms with +1 (zero-centred checkpoints converted without it)
                enorm[v] = vec({blk(il, "nextn.enorm.weight")}, n, v ? 1.f : 0.f);
                hnorm[v] = vec({blk(il, "nextn.hnorm.weight")}, n, v ? 1.f : 0.f);
                shnorm[v] = vec({blk(il, "nextn.shared_head_norm.weight")}, n, v ? 1.f : 0.f);
            }
            if (gg.find(blk(il, "nextn.shared_head_head.weight"))) mtp_head = mat(blk(il, "nextn.shared_head_head.weight"));
            else mtp_head = output;
            if (mtp_head.n_in != n || mtp_head.n_out != c.n_vocab) throw std::runtime_error("MTP head shape");
            if (const strata::TensorInfo* t = gg.find(blk(il, "nextn.embed_tokens.weight"))) {
                int be = 0, bb2 = 0;
                if (!strata::block_geometry(t->type, be, bb2) || n % be || t->shape.size() != 2 || (int)t->shape[1] != c.n_vocab)
                    throw std::runtime_error("nextn.embed_tokens.weight layout");
                memb = t;
                memb_data = host_ptr(*t, blk(il, "nextn.embed_tokens.weight"));
                memb_row_bytes = (size_t)(n / be) * bb2;
            }
            mtp_on = true;
            std::fprintf(stderr, "[qwen36] MTP head loaded (blk.%d), variant %d, up to %d drafts per round\\n", il, variant, spec_max);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[qwen36] --mtp: cannot load the MTP head (%s); MTP is off\\n", e.what());
            L.resize(il);
            layer_slot.resize(il);
            mtp_on = false;
        }
    }

    void alloc_mtp() {
        if (!mtp_on) return;
        constexpr int MB = Model::kMaxBatch;
        const int n = c.n_embd;
        hcap = falloc((size_t)MB * n); carry = falloc(n);
        me = falloc((size_t)MB * n); mh = falloc((size_t)MB * n); mhs = falloc((size_t)MB * n);
        mcat = falloc((size_t)MB * 2 * n);
        mtp_out = falloc(n); mtp_hn = falloc(n);
        vlog = falloc((size_t)std::max(spec_max + 1, 8) * c.n_vocab);
        const size_t st_f = (size_t)c.ssm_S * c.ssm_hv * c.ssm_S, hi_f = (size_t)c.ssm_qkv * (c.ssm_conv - 1);
        for (int gi = 0; gi < c.n_gdn; ++gi) {
            ck_state.push_back(falloc((size_t)(spec_max + 1) * st_f));
            ck_hist.push_back(falloc((size_t)(spec_max + 1) * hi_f));
        }
        kc.push_back(kv_take());      // the MTP layer's own K/V
        vc.push_back(kv_take());
    }

    // Scratch for the grouped path, sized for the biggest block (kMaxBatch tokens x n_used pairs).
    void alloc_grouped() {
        if (!n_grouped_layers) return;
        const int np = Model::kMaxBatch * c.n_used, cap = std::min(c.n_expert, np);
        gx.ptr = (unsigned long long*)dalloc(sizeof(unsigned long long) * (size_t)cap);
        gx.start = (int*)dalloc(sizeof(int) * (size_t)(cap + 1));
        gx.ngroups = (int*)dalloc(sizeof(int));
        gx.tok = (int*)dalloc(sizeof(int) * (size_t)np);
        gx.dst = (int*)dalloc(sizeof(int) * (size_t)np);
        gx.err = (int*)dalloc(sizeof(int));
        gx.scratch = dalloc(K::native_expert_scratch_bytes(np, c.n_ff_exp));
        CK(cudaMemset(gx.err, 0, sizeof(int)));
        CK(cudaMallocHost((void**)&gx.h_err, sizeof(int)));
        *gx.h_err = 0;
    }

    // Queued after the last kernel of a forward pass; checked once the stream has been synchronised.
    void queue_err_check() {
        if (gx.err) CK(cudaMemcpyAsync(gx.h_err, gx.err, sizeof(int), cudaMemcpyDeviceToHost, st));
    }
    void check_err() {
        if (gx.err && *gx.h_err) {
            *gx.h_err = 0;
            CK(cudaMemset(gx.err, 0, sizeof(int)));
            throw std::runtime_error("the router returned an invalid expert id (NaN logits?)");
        }
    }

    // Experts of B tokens through native_expert_grouped.  ids_d: router output (device, sanitised in place), D: [B*n_used rows]
    // of n_embd floats, one per sorted (token, slot) pair; pos_d[b*k+j] = the row of pair (b, j).  No host synchronisation.
    void grouped_experts(const LayerWeights& w, int B, const float* xin, int* ids_d, float* D, int* pos_d) {
        const int n = c.n_embd, ku = c.n_used, ne = c.n_expert, np = B * ku;
        q36::moe_group_dev(ids_d, B, ku, ne, (unsigned long long)(uintptr_t)w.blobs, (unsigned long long)w.blob_stride, gx.ptr,
                           gx.start, gx.ngroups, gx.tok, gx.dst, pos_d, gx.err, st);
        K::quantize_q8_1_rows(xin, B, n, bb.q8g, st);
        const K::NativeExpertLayout xl = K::native_expert_layout(w.gu_type, w.d_type, n, c.n_ff_exp);
        K::native_expert_grouped(xl, gx.ptr, gx.start, gx.ngroups, gx.dst, gx.tok, std::min(ne, np), np, bb.q8g, gx.scratch, D, st);
    }

    // Bytes of ONE K (or V) cache of one attention layer, rounded to 256.
    size_t kv_bytes_one() const {
        const size_t cells = (size_t)c.n_head_kv * max_ctx;
        const size_t b = kv8 ? cells * c.head_dim + cells * (c.head_dim / 32) * 2 : cells * c.head_dim * 2;
        return (b + 255) & ~(size_t)255;
    }
    void* kv_take() {
        if (!kv_unified) return dalloc(kv_cache_bytes);
        void* p = kv_pool + kv_off;
        kv_off += kv_cache_bytes;
        return p;
    }
    void alloc_kv_pool() {
        kv_cache_bytes = c.n_attn ? kv_bytes_one() : 0;
        const int layers = c.n_attn + (mtp_on ? 1 : 0);
        if (kv_unified && layers) kv_pool = static_cast<char*>(dalloc(kv_cache_bytes * 2 * (size_t)layers));
        std::fprintf(stderr, "[qwen36] KV cache: %s, %.2f GiB for %d layers x context %d%s\n", kv8 ? "int8 (+F16 scale per 32)" : "F16",
                     (double)(kv_cache_bytes * 2 * (size_t)layers) / (1024.0 * 1024.0 * 1024.0), layers, max_ctx,
                     kv_unified ? ", one unified buffer" : "");
    }

    void alloc_runtime() {
        alloc_kv_pool();
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
        const int q8n = std::max({2 * n, c.ssm_inner, c.n_head * c.head_dim});
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
                kc.push_back(kv_take());
                vc.push_back(kv_take());
            }
        }
        alloc_batch();
        alloc_grouped();
        alloc_mtp();
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
        const int widest = std::max({2 * n, c.ssm_inner, c.n_head * c.head_dim});
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
        q36::gdn_conv_silu_b(conv_hist[gi], bb.a, w.conv, bb.cs, c.ssm_qkv, B, spec_ckpt ? ck_hist[gi] : nullptr, st);
        q36::gdn_l2norm_qk_b(bb.cs, c.ssm_qkv, 2 * hk, S, B, 1e-6f, st);
        q36::gdn_gate_beta_b(bb.alpha, w.dt, w.ssm_a, bb.gate, bb.beta, hv, B, st);
        q36::gdn_step_b(gdn_state[gi], bb.cs, bb.gate, bb.beta, bb.o, hk, hv, c.ssm_qkv, B, spec_ckpt ? ck_state[gi] : nullptr, st);
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
        q36::attn_prep_kv_b(kc[ai], vc[ai], bb.k, bb.v, w.kn, nkv, hd, c.rot_dim, c.rope_theta, pos0, B, max_ctx, c.eps, kv8, st);
        q36::attn_decode_b(bb.ao, bb.qr, bb.gt, kc[ai], vc[ai], nh, nkv, hd, pos0, B, max_ctx, kv8, st);
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
        if (w.grouped) {
            // one grouped launch for the whole layer: grouping, gate/up, SwiGLU, requantisation and down on the device
            grouped_experts(w, B, bb.xn, bb.ids, bb.Dall, bb.pos);
        } else {
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
        q36::attn_prep_kv(kc[ai], vc[ai], k, v, w.kn, nkv, hd, c.rot_dim, c.rope_theta, pos, max_ctx, c.eps, kv8, st);
        q36::attn_decode(ao, qr, gt, kc[ai], vc[ai], nh, nkv, hd, pos + 1, max_ctx, kv8, st);
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
        if (w.grouped) {
            // decode without a host round trip: the 8 routed experts are one grouped launch; rows come out in sorted order and
            // moe_combine_b reads them back through pos
            grouped_experts(w, 1, xn, ids, down, bb.pos);
            mv(w.sh_gate, 0, xn, q8x, g + (size_t)ku * ffm);
            mv(w.sh_up, 0, xn, q8x, u + (size_t)ku * ffm);
            q36::silu_mul(h + (size_t)ku * ffm, g + (size_t)ku * ffm, u + (size_t)ku * ffm, fs, st);
            quant(h + (size_t)ku * ffm, fs, static_cast<char*>(q8h) + (size_t)ku * q8h_slot);
            mv(w.sh_down, 0, h + (size_t)ku * ffm, static_cast<char*>(q8h) + (size_t)ku * q8h_slot, down + (size_t)ku * n);
            q36::dot_f32(sg, w.sh_gate_inp, xn, n, st);
            q36::moe_combine_b(x, down, wts, bb.pos, down + (size_t)ku * n, sg, 1, ku, n, st);
            return;
        }
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

    // ------------------------------------------------------------------ MTP compute
    void embed_host(int token, float* out) {
        if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id out of range");
        const strata::TensorInfo* e = memb ? memb : emb;
        const uint8_t* d = memb ? memb_data : emb_data;
        const size_t rb = memb ? memb_row_bytes : emb_row_bytes;
        dequant_buf(e->type, d + (size_t)token * rb, c.n_embd, out);
    }

    // The MTP layer over `rows` (token, main hidden) pairs at MTP positions pos0 .. pos0+rows-1.  hsrc: [rows][n] pre-final-norm
    // hidden of the main model.  Leaves the layer output in bb.x.  Writes K/V of the MTP layer for those positions.
    void mtp_rows(const float* hsrc, const int* toks, int rows, int pos0) {
        const int n = c.n_embd, il = c.n_layer;
        const bool post = variant & 1, swap = (variant >> 1) & 1;
        const int p1 = (variant >> 2) & 1;
        for (int i = 0; i < rows; ++i) embed_host(toks[i], bb.h_emb + (size_t)i * n);
        CK(cudaMemcpyAsync(me, bb.h_emb, (size_t)rows * n * 4, cudaMemcpyHostToDevice, st));
        const float* hh = hsrc;
        if (post) { q36::rmsnorm(mh, hsrc, out_norm, rows, n, c.eps, st); hh = mh; }
        if (swap) q36::cat_norm2(mcat, hh, hnorm[p1], me, enorm[p1], rows, n, c.eps, st);
        else q36::cat_norm2(mcat, me, enorm[p1], hh, hnorm[p1], rows, n, c.eps, st);
        quantc(mcat, 2 * n, rows, bb.q8b);
        mvc(eh_proj, 0, mcat, bb.q8b, bb.x, rows);
        const LayerWeights& w = L[il];
        q36::rmsnorm(bb.xn, bb.x, w.attn_norm, rows, n, c.eps, st);
        attn_layer_b(il, pos0, rows);
        q36::add_inplace(bb.x, bb.tmp, rows * n, st);
        q36::rmsnorm(bb.xn, bb.x, w.post_norm, rows, n, c.eps, st);
        moe_layer_b(il, rows);
    }

    // Argmax of the shared head over the `rows` rows of bb.x (probe only), 8 at a time.
    void mtp_head_argmax(int rows, std::vector<int>& out) {
        const int n = c.n_embd, V = c.n_vocab, p1 = (variant >> 2) & 1;
        q36::rmsnorm(bb.xn, bb.x, shnorm[p1], rows, n, c.eps, st);
        quantc(bb.xn, n, rows, bb.q8b);
        const size_t colb = K::native_q8_1_bytes(n, 1);
        for (int r0 = 0; r0 < rows; r0 += 8) {
            const int nc = std::min(8, rows - r0);
            mvc(mtp_head, 0, bb.xn + (size_t)r0 * n, static_cast<const char*>(bb.q8b) + (size_t)r0 * colb, vlog, nc);
            mlog.resize((size_t)nc * V);
            CK(cudaMemcpyAsync(mlog.data(), vlog, (size_t)nc * V * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            for (int r = 0; r < nc; ++r) {
                const float* l = mlog.data() + (size_t)r * V;
                out.push_back((int)(std::max_element(l, l + V) - l));
            }
        }
    }

    // After a main pass of n tokens at positions pos0..: hcap holds their hidden rows.  Builds the MTP rows (hidden of position
    // j, token j+1) for j = pos0-1 .. pos0+n-2 (the first one from `carry`, the hidden of the previous pass), keeps the last
    // hidden as the new carry and synchronises.
    void mtp_track(const int* toks, int n, int pos0) {
        if (!mtp_on) return;
        const int d = c.n_embd;
        int rows = 0, first = 1, mpos0 = pos0;
        if (pos0 > 0 && have_carry) { CK(cudaMemcpyAsync(mhs, carry, (size_t)d * 4, cudaMemcpyDeviceToDevice, st)); rows = 1; first = 0; mpos0 = pos0 - 1; }
        if (n > 1) CK(cudaMemcpyAsync(mhs + (size_t)rows * d, hcap, (size_t)(n - 1) * d * 4, cudaMemcpyDeviceToDevice, st));
        const int total = rows + (n - 1);
        if (total > 0) {
            mtp_rows(mhs, toks + first, total, mpos0);
            if (probe_preds) {
                std::vector<int> pr;
                mtp_head_argmax(total, pr);
                if ((int)probe_preds->size() < mpos0 + total) probe_preds->resize(mpos0 + total, -1);
                for (int i = 0; i < total; ++i) (*probe_preds)[mpos0 + i] = pr[i];
            }
        }
        CK(cudaMemcpyAsync(carry, hcap + (size_t)(n - 1) * d, (size_t)d * 4, cudaMemcpyDeviceToDevice, st));
        have_carry = true;
        queue_err_check();
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        check_err();
    }

    // One MTP step for a single token at MTP position `pos` (decode path).  hin: [n] hidden, either the main model's
    // (main_hidden) or the previous step's chain hidden.  Returns the greedy draft; the chain hidden is mtp_hn / mtp_out.
    int mtp_step(const float* hin, bool main_hidden, int token, int pos) {
        const int n = c.n_embd, il = c.n_layer, V = c.n_vocab;
        const bool post = variant & 1, swap = (variant >> 1) & 1;
        const int p1 = (variant >> 2) & 1;
        embed_host(token, h_emb);
        CK(cudaMemcpyAsync(me, h_emb, (size_t)n * 4, cudaMemcpyHostToDevice, st));
        const float* hh = hin;
        if (main_hidden && post) { q36::rmsnorm(mh, hin, out_norm, 1, n, c.eps, st); hh = mh; }
        if (swap) q36::cat_norm2(mcat, hh, hnorm[p1], me, enorm[p1], 1, n, c.eps, st);
        else q36::cat_norm2(mcat, me, enorm[p1], hh, hnorm[p1], 1, n, c.eps, st);
        quant(mcat, 2 * n, q8x);
        mv(eh_proj, 0, mcat, q8x, x);
        const LayerWeights& w = L[il];
        q36::rmsnorm(xn, x, w.attn_norm, 1, n, c.eps, st);
        attn_layer(il, pos);
        q36::add_inplace(x, tmp, n, st);
        q36::rmsnorm(xn, x, w.post_norm, 1, n, c.eps, st);
        moe_layer(il);
        CK(cudaMemcpyAsync(mtp_out, x, (size_t)n * 4, cudaMemcpyDeviceToDevice, st));
        q36::rmsnorm(xn, x, shnorm[p1], 1, n, c.eps, st);
        CK(cudaMemcpyAsync(mtp_hn, xn, (size_t)n * 4, cudaMemcpyDeviceToDevice, st));
        quant(xn, n, q8x);
        mv(mtp_head, 0, xn, q8x, logits);
        mlog.resize(V);
        CK(cudaMemcpyAsync(mlog.data(), logits, (size_t)V * 4, cudaMemcpyDeviceToHost, st));
        queue_err_check();
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        check_err();
        return (int)(std::max_element(mlog.begin(), mlog.end()) - mlog.begin());
    }
};

Model::Model(const std::string& path, int max_ctx, bool want_mtp, int mtp_k, int mtp_variant, int kv_mode, bool kv_unified)
    : max_ctx_(max_ctx) {
    if (max_ctx < 1) throw std::runtime_error("max_ctx must be >= 1");
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (!ndev) throw std::runtime_error("no CUDA device");
    p_.reset(new Impl(path, cfg_, vram_, max_ctx));
    cfg_ = load_config(p_->gg);
    std::fprintf(stderr, "[qwen36] %s\n", cfg_.describe().c_str());
    p_->want_mtp = want_mtp;
    p_->spec_max = std::max(1, std::min(mtp_k, 7));     // the native GEMV does 8 columns: verify batch <= 8 rows
    p_->variant = mtp_variant & 7;
    if (kv_mode < 0 || kv_mode > 1) throw std::runtime_error("kv mode must be fp16 (0) or int8 (1)");
    p_->kv8 = kv_mode;
    p_->kv_unified = kv_unified;
    CK(cudaStreamCreate(&p_->st));
    p_->load();
    p_->alloc_runtime();
    h_logits_.assign(cfg_.n_vocab, 0.f);
    if (p_->mtp_on) h_vlogits_.assign((size_t)std::max(p_->spec_max + 1, 8) * cfg_.n_vocab, 0.f);
    reset();
    std::fprintf(stderr, "[qwen36] ready: %.2f GiB on GPU, context %d\n", (double)vram_ / (1024.0 * 1024.0 * 1024.0), max_ctx_);
}

Model::~Model() {
    if (!p_) return;
    if (p_->st) cudaStreamSynchronize(p_->st);
    for (void* q : p_->allocs) cudaFree(q);
    if (p_->h_emb) cudaFreeHost(p_->h_emb);
    if (p_->h_ids) cudaFreeHost(p_->h_ids);
    if (p_->gx.h_err) cudaFreeHost(p_->gx.h_err);
    if (p_->bb.h_emb) cudaFreeHost(p_->bb.h_emb);
    if (p_->bb.h_ids) cudaFreeHost(p_->bb.h_ids);
    if (p_->bb.h_perm) cudaFreeHost(p_->bb.h_perm);
    if (p_->bb.h_pos) cudaFreeHost(p_->bb.h_pos);
    for (Snap& s : snaps_) if (s.host) cudaFreeHost(s.host);
    if (p_->st) cudaStreamDestroy(p_->st);
}

// ---------------------------------------------------------------------------------------------------- snapshots
void Model::set_snapshot_limit(int n) {
    snap_max_ = std::max(0, n);
    while ((int)snaps_.size() > snap_max_) {
        if (snaps_.back().host) cudaFreeHost(snaps_.back().host);
        snaps_.pop_back();
    }
}

void Model::snapshot(const int* tokens, int n) {
    Impl& m = *p_;
    if (snap_max_ <= 0 || n < 1 || n != pos_) return;
    for (Snap& s : snaps_)
        if (s.pos == n && std::equal(s.toks.begin(), s.toks.end(), tokens)) { s.stamp = ++stamp_; return; }
    const Config& c = cfg_;
    const size_t st_f = (size_t)c.ssm_S * c.ssm_hv * c.ssm_S, hi_f = (size_t)c.ssm_qkv * (c.ssm_conv - 1);
    const size_t per = m.gdn_state.size() * (st_f + hi_f) + (size_t)c.n_embd;
    int idx = -1;
    if ((int)snaps_.size() < snap_max_) {
        float* host = nullptr;
        if (cudaMallocHost((void**)&host, per * 4) != cudaSuccess) {
            cudaGetLastError();
            std::fprintf(stderr, "[qwen36] snapshots: cannot pin %.0f MiB of host RAM; snapshots disabled\n", per * 4 / 1048576.0);
            snap_max_ = 0;
            return;
        }
        snaps_.emplace_back();
        idx = (int)snaps_.size() - 1;
        snaps_[idx].host = host;
    } else {
        idx = 0;
        for (int i = 1; i < (int)snaps_.size(); ++i) if (snaps_[i].stamp < snaps_[idx].stamp) idx = i;
    }
    Snap& s = snaps_[idx];
    float* h = s.host;
    for (size_t gi = 0; gi < m.gdn_state.size(); ++gi) {
        CK(cudaMemcpyAsync(h, m.gdn_state[gi], st_f * 4, cudaMemcpyDeviceToHost, m.st)); h += st_f;
        CK(cudaMemcpyAsync(h, m.conv_hist[gi], hi_f * 4, cudaMemcpyDeviceToHost, m.st)); h += hi_f;
    }
    s.carry = m.mtp_on && m.have_carry;
    if (s.carry) CK(cudaMemcpyAsync(h, m.carry, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, m.st));
    CK(cudaStreamSynchronize(m.st));
    s.pos = n;
    s.toks.assign(tokens, tokens + n);
    s.stamp = ++stamp_;
}

int Model::restore_best(const std::vector<int>& prompt) {
    Impl& m = *p_;
    int best = -1;
    for (int i = 0; i < (int)snaps_.size(); ++i) {
        const Snap& s = snaps_[i];
        if (s.pos < (int)prompt.size() && s.pos <= max_ctx_ && (best < 0 || s.pos > snaps_[best].pos) &&
            std::equal(s.toks.begin(), s.toks.end(), prompt.begin()))
            best = i;
    }
    if (best < 0) return -1;
    Snap& s = snaps_[best];
    const Config& c = cfg_;
    const size_t st_f = (size_t)c.ssm_S * c.ssm_hv * c.ssm_S, hi_f = (size_t)c.ssm_qkv * (c.ssm_conv - 1);
    const float* h = s.host;
    for (size_t gi = 0; gi < m.gdn_state.size(); ++gi) {
        CK(cudaMemcpyAsync(m.gdn_state[gi], h, st_f * 4, cudaMemcpyHostToDevice, m.st)); h += st_f;
        CK(cudaMemcpyAsync(m.conv_hist[gi], h, hi_f * 4, cudaMemcpyHostToDevice, m.st)); h += hi_f;
    }
    m.have_carry = s.carry;
    if (s.carry) CK(cudaMemcpyAsync(m.carry, h, (size_t)c.n_embd * 4, cudaMemcpyHostToDevice, m.st));
    CK(cudaStreamSynchronize(m.st));
    pos_ = s.pos;
    v_n_ = 0;
    s.stamp = ++stamp_;
    return s.pos;
}

void Model::invalidate_snapshots(const std::vector<int>& prompt, int start) {
    for (size_t i = 0; i < snaps_.size();) {
        const Snap& s = snaps_[i];
        // K/V at positions < start is not touched.  Beyond it the request writes the prompt and then generated tokens: a snapshot
        // survives only if it lies inside the prompt and is a prefix of it (the same tokens rewrite the same K/V).
        const bool keep = s.pos <= start ||
                          (s.pos <= (int)prompt.size() && std::equal(s.toks.begin(), s.toks.end(), prompt.begin()));
        if (keep) { ++i; continue; }
        if (s.host) cudaFreeHost(s.host);
        snaps_.erase(snaps_.begin() + i);
    }
}

bool Model::mtp_enabled() const { return p_->mtp_on; }
int Model::spec_max() const { return p_->spec_max; }
int Model::mtp_variant() const { return p_->variant; }
void Model::set_mtp_variant(int v) { p_->variant = v & 7; }

void Model::reset() {
    for (float* s : p_->gdn_state) CK(cudaMemsetAsync(s, 0, (size_t)cfg_.ssm_S * cfg_.ssm_hv * cfg_.ssm_S * 4, p_->st));
    for (float* s : p_->conv_hist) CK(cudaMemsetAsync(s, 0, (size_t)cfg_.ssm_qkv * (cfg_.ssm_conv - 1) * 4, p_->st));
    CK(cudaStreamSynchronize(p_->st));
    pos_ = 0;
    p_->have_carry = false;
    v_n_ = 0;
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
    if (m.mtp_on) CK(cudaMemcpyAsync(m.hcap, m.x, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, m.st));
    if (want_logits) {
        q36::rmsnorm(m.xn, m.x, m.out_norm, 1, c.n_embd, c.eps, m.st);
        m.quant(m.xn, c.n_embd, m.q8x);
        m.mv(m.output, 0, m.xn, m.q8x, m.logits);
        CK(cudaMemcpyAsync(h_logits_.data(), m.logits, (size_t)c.n_vocab * 4, cudaMemcpyDeviceToHost, m.st));
    }
    m.queue_err_check();
    CK(cudaStreamSynchronize(m.st));
    CK(cudaGetLastError());
    m.check_err();
    const int pos0 = pos_;
    ++pos_;
    m.mtp_track(&token, 1, pos0);
}

// mode: 0 = no logits, 1 = logits of the last row (h_logits_), 2 = logits of every row (h_vlogits_, speculative verification)
void Model::batch_pass(const int* tokens, int n, int mode) {
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
    if (m.mtp_on) CK(cudaMemcpyAsync(m.hcap, m.bb.x, (size_t)n * c.n_embd * 4, cudaMemcpyDeviceToDevice, m.st));
    if (mode == 1) {
        q36::rmsnorm(m.xn, m.bb.x + (size_t)(n - 1) * c.n_embd, m.out_norm, 1, c.n_embd, c.eps, m.st);
        m.quant(m.xn, c.n_embd, m.q8x);
        m.mv(m.output, 0, m.xn, m.q8x, m.logits);
        CK(cudaMemcpyAsync(h_logits_.data(), m.logits, (size_t)c.n_vocab * 4, cudaMemcpyDeviceToHost, m.st));
    } else if (mode == 2) {
        q36::rmsnorm(m.bb.xn, m.bb.x, m.out_norm, n, c.n_embd, c.eps, m.st);
        m.quantc(m.bb.xn, c.n_embd, n, m.bb.q8b);
        m.mvc(m.output, 0, m.bb.xn, m.bb.q8b, m.vlog, n);
        CK(cudaMemcpyAsync(h_vlogits_.data(), m.vlog, (size_t)n * c.n_vocab * 4, cudaMemcpyDeviceToHost, m.st));
    }
    m.queue_err_check();
    CK(cudaStreamSynchronize(m.st));
    CK(cudaGetLastError());
    m.check_err();
    pos_ += n;
}

void Model::forward_batch(const int* tokens, int n, bool want_logits) {
    const int pos0 = pos_;
    batch_pass(tokens, n, want_logits ? 1 : 0);
    p_->mtp_track(tokens, n, pos0);
}

// ---------------------------------------------------------------------------------------------------- speculative decoding
void Model::mtp_draft(int next, int k, std::vector<int>& out) {
    Impl& m = *p_;
    out.clear();
    k = std::min({k, m.spec_max, max_ctx_ - pos_ - 1});
    if (!m.mtp_on || !m.have_carry || pos_ < 1 || k < 1) return;
    int tok = next;
    const float* h = m.carry;
    bool main_h = true;
    for (int i = 0; i < k; ++i) {
        const int d = m.mtp_step(h, main_h, tok, pos_ - 1 + i);
        out.push_back(d);
        tok = d;
        h = (m.variant & 1) ? m.mtp_hn : m.mtp_out;
        main_h = false;
    }
}

void Model::forward_verify(const int* tokens, int n) {
    Impl& m = *p_;
    if (!m.mtp_on) throw std::runtime_error("forward_verify needs the MTP head (--mtp)");
    if (n < 1 || n > m.spec_max + 1) throw std::runtime_error("forward_verify: too many rows");
    v_pos0_ = pos_;
    v_n_ = n;
    v_tokens_.assign(tokens, tokens + n);
    if (n == 1) {      // plain decode step (it tracks the MTP rows itself)
        v_single_ = true;
        forward(tokens[0], true);
        std::copy(h_logits_.begin(), h_logits_.end(), h_vlogits_.begin());
        return;
    }
    v_single_ = false;
    m.spec_ckpt = true;
    try { batch_pass(tokens, n, 2); } catch (...) { m.spec_ckpt = false; throw; }
    m.spec_ckpt = false;
}

// Keep the first `keep` rows of the last verification: restore the GDN state / conv history to the state after row keep-1, move
// the position back (K/V rows beyond are overwritten later) and rebuild the MTP rows for the kept tokens.
void Model::commit(int keep) {
    Impl& m = *p_;
    if (keep < 1 || keep > v_n_) throw std::runtime_error("commit: bad row count");
    if (v_single_) return;
    const Config& c = cfg_;
    if (keep < v_n_) {
        const size_t st_f = (size_t)c.ssm_S * c.ssm_hv * c.ssm_S, hi_f = (size_t)c.ssm_qkv * (c.ssm_conv - 1);
        for (size_t gi = 0; gi < m.gdn_state.size(); ++gi) {
            CK(cudaMemcpyAsync(m.gdn_state[gi], m.ck_state[gi] + (size_t)(keep - 1) * st_f, st_f * 4, cudaMemcpyDeviceToDevice, m.st));
            CK(cudaMemcpyAsync(m.conv_hist[gi], m.ck_hist[gi] + (size_t)(keep - 1) * hi_f, hi_f * 4, cudaMemcpyDeviceToDevice, m.st));
        }
        pos_ = v_pos0_ + keep;
    }
    m.mtp_track(v_tokens_.data(), keep, v_pos0_);     // synchronises
    v_n_ = 0;
}

// Fraction of positions in [from, seq.size()-2) where the MTP head's argmax equals the real next-next token, for the current
// variant: run the sequence through the model (block prefill), collecting the head's prediction at every position.
double Model::mtp_probe(const std::vector<int>& seq, int from, int* counted) {
    Impl& m = *p_;
    if (!m.mtp_on) throw std::runtime_error("mtp_probe needs the MTP head (--mtp)");
    std::vector<int> preds;
    m.probe_preds = &preds;
    reset();
    try {
        const int N = (int)seq.size();
        for (int i = 0; i < N;) {
            const int nb = std::min(N - i, (int)kMaxBatch);
            if (nb == 1) forward(seq[i], false); else forward_batch(&seq[i], nb, false);
            i += nb;
        }
    } catch (...) { m.probe_preds = nullptr; throw; }
    m.probe_preds = nullptr;
    int ok = 0, tot = 0;
    for (int j = std::max(from, 0); j + 2 < (int)seq.size() && j < (int)preds.size(); ++j) {
        ++tot;
        ok += preds[j] == seq[j + 2];
    }
    if (counted) *counted = tot;
    return tot ? (double)ok / tot : 0.0;
}

}  // namespace q36
