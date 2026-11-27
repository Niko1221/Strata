#include "strata/hy4flash_gguf/dfly.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace strata::hy4_flash {

namespace {

constexpr int kF32 = 0, kQ8_0 = 8;
constexpr int kThreads = 256;
constexpr int kRows = DflyDrafter::kMaxK + 1;   // the widest block this build drafts

unsigned blocks_for(long long n) {
    long long b = (n + kThreads - 1) / kThreads;
    return (unsigned) std::max(1LL, std::min(b, 65535LL));
}

/// x += d * g, the fusion's RAW gate: a learned vector, no sigmoid. Multiply then add, as two roundings, which
/// is what the reference's ggml_mul + ggml_add are.
__global__ void gate_add_k(float* x, const float* d, const float* g, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = __fadd_rn(x[i], __fmul_rn(d[i], g[i]));
}

__global__ void iota_k(int32_t* p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = i;
}

/// Row j of the drafter's key list: the target's selection for the source row (`n_ctx` cells, -1 padded, all
/// <= a-1), then the block rows a .. a+j, then -1. The kernel enforces causality on its own as well.
__global__ void sel_rows_k(const int32_t* ctx, int n_ctx, int a, int w, int32_t* rows) {
    const int j = blockIdx.x, width = n_ctx + w;
    for (int t = threadIdx.x; t < width; t += blockDim.x) {
        const int b = t - n_ctx;
        rows[(long long) j * width + t] = t < n_ctx ? ctx[t] : (b <= j ? a + b : -1);
    }
}

/// Read the chosen ID on device and gather its embedding for the next slot.
__global__ void row_by_id_k(const uint8_t* src, std::size_t row_bytes, const int32_t* id, int n_rows, uint8_t* dst) {
    const int r = *id;
    const uint8_t* row = src + (std::size_t) (r >= 0 && r < n_rows ? r : 0) * row_bytes;
    for (std::size_t i = blockIdx.x * (std::size_t) blockDim.x + threadIdx.x; i < row_bytes;
         i += (std::size_t) gridDim.x * blockDim.x)
        dst[i] = row[i];
}

/// c = sigmoid(w . h + b), one block.
__global__ void confidence_k(const float* h, const float* w, const float* b, float* out, int n) {
    __shared__ float part[kThreads];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += h[i] * w[i];
    part[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if ((int) threadIdx.x < k) part[threadIdx.x] += part[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = 1.f / (1.f + expf(-(part[0] + b[0])));
}

const char* kLayerNames[] = {
    "attn_norm", "attn_q_a", "attn_q_a_norm", "attn_q_b", "attn_kv_a_mqa", "attn_kv_a_norm", "attn_k_b",
    "attn_v_b", "attn_sinks", "attn_gate", "attn_output", "post_attention_norm", "ffn_norm", "ffn_gate",
    "ffn_up", "ffn_down",
};
enum LayerT {
    L_ATTN_NORM, L_Q_A, L_Q_A_NORM, L_Q_B, L_KV_A, L_KV_NORM, L_K_B, L_V_B, L_SINKS, L_GATE, L_WO, L_POST_NORM,
    L_FFN_NORM, L_FFN_GATE, L_FFN_UP, L_FFN_DOWN, L_COUNT
};
enum GlobalT {
    G_OUT_NORM, G_F_ENORM, G_F_HNORM, G_F_EH, G_F_GATE, G_C_HNORM, G_C_ENORM, G_C_GATE, G_C_UP, G_C_DOWN, G_CONF_W,
    G_CONF_B, G_COUNT
};
const char* kGlobalNames[] = {
    "output_norm.weight", "dfly_fusion_enorm.weight", "dfly_fusion_hnorm.weight", "dfly_fusion_eh.weight",
    "dfly_fusion_gate.weight", "dfly_correction_hnorm.weight", "dfly_correction_enorm.weight",
    "dfly_correction_gate.weight", "dfly_correction_up.weight", "dfly_correction_down.weight",
    "dfly_confidence.weight", "dfly_confidence.bias",
};

struct Expect {
    std::string name;
    int type;
    std::vector<uint64_t> shape;
};

std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
}

}  // namespace

const char* dfly_result_name(DflyResult r) {
    switch (r) {
    case DflyResult::ok: return "ok";
    case DflyResult::safe_fallback: return "safe_fallback";
    case DflyResult::abort_request: return "abort_request";
    case DflyResult::restart_worker: return "restart_worker";
    }
    return "?";
}

bool dfly_context_from_capture(const DflyCapture& cap, int row, const std::array<int, 5>& layers,
                               DflyContext& out, std::string& err) {
    out = DflyContext{};
    if (!cap.valid) { err = "dfly context: the capture is not valid"; return false; }
    if (row < 0 || row >= cap.n_rows) {
        err = "dfly context: row " + std::to_string(row) + " outside the capture's " + std::to_string(cap.n_rows);
        return false;
    }
    out.source_pos = cap.pos_begin + row;
    out.hidden = (const float*) cap.hidden + (std::size_t) row * cap.n_embd;
    out.engaged = cap.engaged;
    out.n_sel = cap.engaged ? cap.n_sel : 0;
    if (cap.engaged) {
        for (int i = 0; i < 5; ++i) {
            int at = -1;
            for (std::size_t s = 0; s < cap.sel_blocks.size(); ++s)
                if (cap.sel_blocks[s] == layers[(std::size_t) i]) at = (int) s;
            if (at < 0 || at >= (int) cap.sel.size() || cap.sel[(std::size_t) at] == nullptr) {
                err = "dfly context: the capture holds no selection for target block " +
                      std::to_string(layers[(std::size_t) i]);
                out = DflyContext{};
                return false;
            }
            out.sel[(std::size_t) i] = (const int32_t*) cap.sel[(std::size_t) at] + (std::size_t) row * cap.n_sel;
        }
    }
    out.owner = &cap;
    out.generation = cap.generation;
    return true;
}

struct DflyDrafter::Impl {
    // ---- the drafter's hyper-parameters, from its file
    int n_embd = 0, n_ff = 0, n_head = 0, q_lora = 0, kv_lora = 0, n_rot = 0, qk = 0, nope = 0, v_head = 0;
    int n_corr = 0, top_k = 0, mask_id = -1;
    float eps = 0.f, rope_base = 0.f;
    std::array<int, 5> file_layers{};
    // ---- what the target geometry says, kept for bind
    int vocab = 0, t_kpool = 0;
    // ---- the weights, one device arena
    void* arena = nullptr;
    std::size_t arena_bytes = 0;
    struct Wt { const void* p = nullptr; int type = 0; };
    Wt layer[5][L_COUNT];
    Wt glob[G_COUNT];
    bool is_loaded = false;

    // ---- the binding
    const DenseResidency* res = nullptr;
    KvCache* cache = nullptr;
    DflyTargetWeights tw;
    std::array<int, 5> layers{};
    std::array<void*, 5> kv_ptr{}, pe_ptr{};
    int max_tokens = 0, max_sel = 0, max_drafts = 0;
    cudaStream_t stream = nullptr;
    void* scratch = nullptr;
    std::size_t scratch_bytes = 0;
    bool is_bound = false;
    Pending pending;
    std::array<int32_t, kRows> host_pos{};
    std::vector<int32_t> host_pq;                 // MlaPerQuery's two arrays for this round, staged
    std::array<int32_t, DflyDrafter::kMaxK> host_ids{};
    std::array<uint32_t, DflyDrafter::kMaxK * kernels::kNonfiniteBlocks> host_flags{};
    bool pq_ok = false;                           // the attention takes every query in one call a stage
    std::size_t emb_row_bytes = 0;                // one row of the target's token_embd

    // scratch views
    int32_t *d_pos = nullptr, *d_kv_pos = nullptr, *d_sel = nullptr;
    int32_t *d_pq = nullptr, *d_ids = nullptr;    // [W key counts | W * n_head head map]; the k picks
    uint32_t* d_flags = nullptr;                  // logits_nonfinite_rows, a row a slot
    void* arg_sc = nullptr;                       // argmax_rows' partials, zero between launches
    uint8_t* emb_row = nullptr;
    float *emask = nullptr, *e0 = nullptr, *eprev = nullptr;
    float *x = nullptr, *u = nullptr, *attn_out = nullptr, *t = nullptr, *ffn_out = nullptr, *hs = nullptr;
    void *u_q8 = nullptr, *t_q8 = nullptr, *f_q8 = nullptr, *ff_q8 = nullptr, *c_q8 = nullptr, *cff_q8 = nullptr,
         *h_q8 = nullptr;
    float *f_in = nullptr, *f_delta = nullptr;
    void *proj_sc = nullptr, *abs_sc = nullptr, *att_sc = nullptr, *out_sc = nullptr;
    float *absorbed = nullptr, *kqv = nullptr, *gate = nullptr, *up = nullptr;
    float *c_in = nullptr, *cg = nullptr, *cu = nullptr, *c_delta = nullptr, *h_corr = nullptr, *logits = nullptr,
          *conf = nullptr;

    ~Impl() { release_binding(); if (arena) cudaFree(arena); }

    void release_binding() {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); stream = nullptr; }
        if (scratch) { cudaFree(scratch); scratch = nullptr; }
        scratch_bytes = 0;
        res = nullptr; cache = nullptr; tw = DflyTargetWeights{};
        is_bound = false;
        pending = Pending{};
    }

    DflyResult run(int32_t anchor, int a, const DflyContext& ctx, int k, int32_t* draft, float* confidence,
                   std::string& err);
};

DflyDrafter::DflyDrafter() : p_(std::make_unique<Impl>()) {}
DflyDrafter::~DflyDrafter() = default;

bool DflyDrafter::loaded() const { return p_->is_loaded; }
bool DflyDrafter::bound() const { return p_->is_bound; }
const std::array<int, 5>& DflyDrafter::target_layers() const { return p_->is_bound ? p_->layers : p_->file_layers; }
int DflyDrafter::max_drafts() const { return p_->max_drafts; }
int DflyDrafter::mask_id() const { return p_->mask_id; }
int DflyDrafter::vocab() const { return p_->vocab; }
std::size_t DflyDrafter::weight_bytes() const { return p_->arena_bytes; }
std::size_t DflyDrafter::scratch_bytes() const { return p_->scratch_bytes; }
const DflyDrafter::Pending& DflyDrafter::pending() const { return p_->pending; }
void DflyDrafter::drop_pending() { p_->pending = Pending{}; }
bool DflyDrafter::consume_pending(int a, int verify_w, uint64_t generation, std::string& err) {
    const Pending& p = p_->pending;
    if (!p.active) { err = "no proposal is pending"; return false; }
    if (p.a != a || p.generation != generation) {
        err = "the pending proposal is at " + std::to_string(p.a) + " (generation " + std::to_string(p.generation) +
              "), not at " + std::to_string(a) + " (generation " + std::to_string(generation) + ")";
        return false;
    }
    if (verify_w < 1 || verify_w > p.w) {
        err = "a verify of " + std::to_string(verify_w) + " rows is not a prefix of the " + std::to_string(p.w) +
              " the proposal wrote";
        return false;
    }
    p_->pending = Pending{};
    return true;
}

bool DflyDrafter::load(const std::string& path, const Geometry& g, std::string& err) {
    Impl& m = *p_;
    if (m.is_loaded) { err = "dfly load: already loaded"; return false; }
    const std::string P = "dfly load: " + path + ": ";
    try {
        const strata::GgufFile f(path);
        const auto key = [&](const std::string& k) { return f.get(k); };
        const std::string A = "hy_v4_next_dfly.";
        std::vector<std::string> bad;
        const auto u = [&](const std::string& k, MetaType want = MetaType::U32) -> uint64_t {
            const strata::MetaValue* v = key(k);
            if (v == nullptr) { bad.push_back("missing key " + k); return 0; }
            if (v->type != want) { bad.push_back("key " + k + " has the wrong type"); return 0; }
            return v->u;
        };
        const auto fl = [&](const std::string& k) -> float {
            const strata::MetaValue* v = key(k);
            if (v == nullptr || v->type != MetaType::F32) { bad.push_back("missing or non-f32 key " + k); return 0.f; }
            return (float) v->f;
        };
        const auto str = [&](const std::string& k) -> std::string {
            const strata::MetaValue* v = key(k);
            if (v == nullptr || v->type != MetaType::STRING) { bad.push_back("missing or non-string key " + k); return ""; }
            return v->s;
        };
        const auto arr = [&](const std::string& k) -> std::vector<uint64_t> {
            const strata::MetaValue* v = key(k);
            std::vector<uint64_t> r;
            if (v == nullptr || v->type != MetaType::ARRAY || v->elem != MetaType::U32 || v->count != v->items.size()) {
                bad.push_back("missing or non-u32-array key " + k);
                return r;
            }
            for (const auto& it : v->items) r.push_back(it.u);
            return r;
        };
        {
            const strata::MetaValue* arch = key("general.architecture");
            if (arch == nullptr || arch->s != "hy_v4_next_dfly") {
                err = P + "general.architecture is not hy_v4_next_dfly";
                return false;
            }
        }
        const uint64_t schema = u(A + "schema_version"), block_size = u(A + "block_size"),
                       n_layer = u(A + "block_count"), n_embd = u(A + "embedding_length"),
                       n_ff = u(A + "feed_forward_length"), n_head = u(A + "attention.head_count"),
                       n_head_kv = u(A + "attention.head_count_kv"), q_lora = u(A + "attention.q_lora_rank"),
                       kv_lora = u(A + "attention.kv_lora_rank"), key_len = u(A + "attention.key_length"),
                       val_len = u(A + "attention.value_length"), k_mla = u(A + "attention.key_length_mla"),
                       v_mla = u(A + "attention.value_length_mla"), n_rot = u(A + "rope.dimension_count"),
                       top_k = u(A + "target_index_top_k"), n_corr = u(A + "hidden_correction_length"),
                       mask = u("tokenizer.ggml.mask_token_id");
        const bool causal = u(A + "attention.causal", MetaType::BOOL) != 0;
        const bool has_conf = u(A + "has_confidence_head", MetaType::BOOL) != 0;
        const float rope_base = fl(A + "rope.freq_base"), eps = fl(A + "attention.layer_norm_rms_epsilon");
        const std::string fusion = str(A + "mtp_fusion"), rope_layout = str(A + "rope_layout");
        const std::vector<uint64_t> kv_layers = arr(A + "target_kv_layers"),
                                    ix_layers = arr(A + "target_index_layers");
        const strata::MetaValue* toks = key("tokenizer.ggml.tokens");
        if (!bad.empty()) { err = P + bad.front(); return false; }

        // ---- schema v1 semantics, then this build's limits, then agreement with the target
        const auto need = [&](bool ok, const std::string& what) { if (!ok) bad.push_back(what); };
        need(schema == 1, "schema_version " + std::to_string(schema) + ", expected 1");
        need(block_size >= 2, "block_size " + std::to_string(block_size) + " must be >= 2");
        if (block_size >= 2) m.max_drafts = (int) std::min<uint64_t>(block_size - 1, kMaxK);
        need(causal, "attention.causal is false; only the causal schema is supported");
        need(has_conf, "has_confidence_head is false");
        need(fusion == "slot0", "mtp_fusion '" + fusion + "', expected 'slot0'");
        need(rope_layout == "norm_adjacent", "rope_layout '" + rope_layout + "', expected 'norm_adjacent'");
        need(n_layer == 5, "block_count " + std::to_string(n_layer) + ", expected 5");
        need(n_head_kv == 1, "head_count_kv must be 1");
        need(key_len == kv_lora + n_rot, "key_length must be kv_lora_rank + rope.dimension_count");
        need(val_len == kv_lora, "value_length must be kv_lora_rank");
        need(k_mla > n_rot, "key_length_mla must exceed rope.dimension_count");
        need(std::isfinite(eps) && eps > 0.f, "layer_norm_rms_epsilon must be finite and > 0");
        need(std::isfinite(rope_base) && rope_base > 0.f, "rope.freq_base must be finite and > 0");
        need(kv_layers.size() == 5 && kv_layers == ix_layers, "target_kv_layers must be 5 entries equal to target_index_layers");
        const std::vector<uint64_t> prod = {25, 29, 33, 37, 41};
        need(kv_layers == prod, "target_kv_layers differ from [25,29,33,37,41], the only pairing this build supports");
        // the target
        const int n_trunk = (int) g.layers.size() - g.n_mtp_blocks;
        need((int) n_embd == g.hidden_dim, "embedding_length differs from the target's");
        need((int) kv_lora == g.kv_lora_rank, "kv_lora_rank differs from the target's");
        need((int) n_rot == g.rope_dim, "rope.dimension_count differs from the target's");
        need((int) top_k == g.idx_top_k, "target_index_top_k differs from the target's indexer top_k");
        need(g.idx_kpool == 4 && g.idx_select_tail, "the target's indexer must have kpool 4 and the forced tail");
        need(rope_base == g.rope_theta_sparse, "rope.freq_base differs from the target's sparse layers");
        need(mask < (uint64_t) g.vocab_size, "mask_token_id outside the target vocab");
        need(toks != nullptr && toks->type == MetaType::ARRAY && toks->count == (uint64_t) g.vocab_size,
             "tokenizer.ggml.tokens count differs from the target vocab");
        for (uint64_t l : kv_layers) {
            if ((int) l >= n_trunk) { need(false, "target layer " + std::to_string(l) + " is not a trunk block"); continue; }
            const LayerSpec& ls = g.layers[(std::size_t) l];
            need(ls.attention == Attention::Sparse && ls.has_indexer,
                 "target layer " + std::to_string(l) + " is not a full-attention layer with an indexer");
            need(ls.n_heads == (int) n_head, "target layer " + std::to_string(l) + " has another head count");
        }
        // unknown keys of this architecture
        for (const auto& kv : f.metadata()) {
            if (kv.first.rfind(A, 0) != 0) continue;
            static const char* known[] = {
                "schema_version", "block_size", "attention.causal", "block_count", "context_length",
                "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv",
                "attention.q_lora_rank", "attention.kv_lora_rank", "attention.key_length", "attention.value_length",
                "attention.key_length_mla", "attention.value_length_mla", "rope.dimension_count", "rope.freq_base",
                "attention.layer_norm_rms_epsilon", "target_kv_layers", "target_index_layers", "target_index_top_k",
                "hidden_correction_length", "has_confidence_head", "mtp_fusion", "rope_layout",
                "source_mrope_sections"};
            bool ok = false;
            for (const char* k : known) ok = ok || kv.first == A + k;
            need(ok, "unknown key " + kv.first);
        }
        if (!bad.empty()) { err = P + bad.front() + (bad.size() > 1 ? " (+" + std::to_string(bad.size() - 1) + " more)" : ""); return false; }

        m.n_embd = (int) n_embd; m.n_ff = (int) n_ff; m.n_head = (int) n_head; m.q_lora = (int) q_lora;
        m.kv_lora = (int) kv_lora; m.n_rot = (int) n_rot; m.qk = (int) k_mla; m.nope = (int) (k_mla - n_rot);
        m.v_head = (int) v_mla; m.n_corr = (int) n_corr; m.top_k = (int) top_k; m.mask_id = (int) mask;
        m.eps = eps; m.rope_base = rope_base; m.vocab = g.vocab_size; m.t_kpool = g.idx_kpool;
        for (int i = 0; i < 5; ++i) m.file_layers[(std::size_t) i] = (int) kv_layers[(std::size_t) i];

        // ---- the 92 tensors: names, types, shapes, and nothing else
        const uint64_t E = n_embd, H = n_head;
        std::vector<Expect> want;
        for (int i = 0; i < 5; ++i) {
            const std::string b = "blk." + std::to_string(i) + ".";
            const std::vector<Expect> l = {
                {b + "attn_norm.weight", kF32, {E}},
                {b + "attn_q_a.weight", kQ8_0, {E, q_lora}},
                {b + "attn_q_a_norm.weight", kF32, {q_lora}},
                {b + "attn_q_b.weight", kQ8_0, {q_lora, H * k_mla}},
                {b + "attn_kv_a_mqa.weight", kQ8_0, {E, kv_lora + n_rot}},
                {b + "attn_kv_a_norm.weight", kF32, {kv_lora}},
                {b + "attn_k_b.weight", kQ8_0, {k_mla - n_rot, kv_lora, H}},
                {b + "attn_v_b.weight", kQ8_0, {kv_lora, v_mla, H}},
                {b + "attn_sinks.weight", kF32, {H}},
                {b + "attn_gate.weight", kQ8_0, {E, H * v_mla}},
                {b + "attn_output.weight", kQ8_0, {H * v_mla, E}},
                {b + "post_attention_norm.weight", kF32, {E}},
                {b + "ffn_norm.weight", kF32, {E}},
                {b + "ffn_gate.weight", kQ8_0, {E, n_ff}},
                {b + "ffn_up.weight", kQ8_0, {E, n_ff}},
                {b + "ffn_down.weight", kQ8_0, {n_ff, E}},
            };
            want.insert(want.end(), l.begin(), l.end());
        }
        const std::vector<Expect> gl = {
            {kGlobalNames[G_OUT_NORM], kF32, {E}},
            {kGlobalNames[G_F_ENORM], kF32, {E}},
            {kGlobalNames[G_F_HNORM], kF32, {E}},
            {kGlobalNames[G_F_EH], kQ8_0, {2 * E, E}},
            {kGlobalNames[G_F_GATE], kF32, {E}},
            {kGlobalNames[G_C_HNORM], kF32, {E}},
            {kGlobalNames[G_C_ENORM], kF32, {E}},
            {kGlobalNames[G_C_GATE], kQ8_0, {2 * E, n_corr}},
            {kGlobalNames[G_C_UP], kQ8_0, {2 * E, n_corr}},
            {kGlobalNames[G_C_DOWN], kQ8_0, {n_corr, E}},
            {kGlobalNames[G_CONF_W], kF32, {E}},
            {kGlobalNames[G_CONF_B], kF32, {1}},
        };
        want.insert(want.end(), gl.begin(), gl.end());
        if (f.tensors().size() != want.size()) {
            err = P + std::to_string(f.tensors().size()) + " tensors, expected " + std::to_string(want.size());
            return false;
        }
        std::map<std::string, std::size_t> offset;
        std::size_t total = 0;
        for (const Expect& x : want) {
            const strata::TensorInfo* t = f.find(x.name);
            if (t == nullptr) { err = P + "missing tensor " + x.name; return false; }
            std::vector<uint64_t> shape = t->shape;
            if (x.name == kGlobalNames[G_CONF_W] && shape.size() == 2 && shape[1] == 1) shape.pop_back();
            if ((int) t->type != x.type) {
                err = P + x.name + " has type " + t->type_name() + ", expected " + (x.type == kF32 ? "F32" : "Q8_0");
                return false;
            }
            if (shape != x.shape) {
                err = P + x.name + " has shape " + shape_str(t->shape) + ", expected " + shape_str(x.shape);
                return false;
            }
            const uint64_t bytes = tensor_payload_bytes(*t);
            if (bytes == 0 || t->offset + bytes + f.data_start() > f.file_size()) {
                err = P + x.name + " lies outside the file";
                return false;
            }
            offset[x.name] = total;
            total += ((std::size_t) bytes + 255) & ~(std::size_t) 255;
        }
        if (cudaMalloc(&m.arena, total) != cudaSuccess) {
            cudaGetLastError();
            m.arena = nullptr;
            err = P + "cannot allocate " + std::to_string(total >> 20) + " MiB for the weights";
            return false;
        }
        m.arena_bytes = total;
        for (const Expect& x : want) {
            const strata::TensorInfo* t = f.find(x.name);
            char* dst = (char*) m.arena + offset[x.name];
            if (cudaMemcpy(dst, f.tensor_data(*t), (std::size_t) tensor_payload_bytes(*t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                cudaGetLastError();
                err = P + "upload of " + x.name + " failed";
                return false;
            }
        }
        for (int i = 0; i < 5; ++i)
            for (int l = 0; l < L_COUNT; ++l) {
                const std::string n = "blk." + std::to_string(i) + "." + kLayerNames[l] + ".weight";
                m.layer[i][l] = {(char*) m.arena + offset[n], (int) f.find(n)->type};
            }
        for (int gi = 0; gi < G_COUNT; ++gi)
            m.glob[gi] = {(char*) m.arena + offset[kGlobalNames[gi]], (int) f.find(kGlobalNames[gi])->type};
    } catch (const std::exception& e) {
        err = P + e.what();
        return false;
    }
    m.is_loaded = true;
    return true;
}

bool DflyDrafter::bind(const DenseResidency& res, KvCache& cache, std::string& err) {
    return bind(res, cache, err, BindOptions{});
}

bool DflyDrafter::bind(const DenseResidency& res, KvCache& cache, std::string& err, const BindOptions& o) {
    Impl& m = *p_;
    if (!m.is_loaded) { err = "dfly bind: load first"; return false; }
    m.release_binding();
    DflyTargetWeights tw;
    if (!dfly_target_weights(res, tw, err)) return false;
    const std::vector<int64_t> eo = {m.n_embd, m.vocab};
    if (tw.token_embd_shape != eo || tw.output_shape != eo) {
        err = "dfly bind: the target's token_embd/output are not [n_embd, vocab]";
        return false;
    }
    if (tw.output_type != 30 && !kernels::native_mmvq_supported(tw.output_type)) {
        err = "dfly bind: the target's output type " + std::to_string(tw.output_type) + " has no matvec";
        return false;
    }
    if (kv_format(cache) == kKvF32 || cache.kv_lora != m.kv_lora || cache.n_rot != m.n_rot) {
        err = "dfly bind: the KV cache must be F16 or Q8_0 with the drafter's latent and rope widths";
        return false;
    }
    std::array<int, 5> layers = m.file_layers;
    if (o.reverse_layers) std::reverse(layers.begin(), layers.end());
    for (int i = 0; i < 5; ++i) {
        const int b = layers[(std::size_t) i];
        if (b < 0 || b >= cache.n_blocks || cache.kv_cmpr[(std::size_t) b] == nullptr ||
            cache.k_pe[(std::size_t) b] == nullptr) {
            err = "dfly bind: the KV cache has no block " + std::to_string(b);
            return false;
        }
        m.kv_ptr[(std::size_t) i] = cache.kv_cmpr[(std::size_t) b];
        m.pe_ptr[(std::size_t) i] = cache.k_pe[(std::size_t) b];
    }
    m.layers = layers;
    m.max_tokens = cache.max_tokens;
    m.pq_ok = true;
    for (int i = 0; i < 5; ++i)
        m.pq_ok = m.pq_ok && mla_per_query_supported(m.kv_lora, m.n_rot, m.n_head, cache.fp16,
                                                     m.layer[i][L_K_B].type, m.layer[i][L_V_B].type);
    {
        int elems = 0, bytes = 0;
        if (!strata::block_geometry((uint32_t) tw.token_embd_type, elems, bytes) || elems <= 0 ||
            m.n_embd % elems != 0) {
            err = "dfly bind: no row size for the target's token_embd type " + std::to_string(tw.token_embd_type);
            return false;
        }
        m.emb_row_bytes = (std::size_t) (m.n_embd / elems) * bytes;
    }
    // The context is at most the target's per-query cell count (top_k plus the tail), and the block adds w.
    m.max_sel = m.top_k + m.t_kpool + kRows;

    // ---- scratch, one allocation
    const int n = m.n_embd, W = kRows;
    std::vector<std::pair<void**, std::size_t>> plan;
    const auto f4 = [](long long c) { return (std::size_t) c * 4; };
    plan = {
        {(void**) &m.d_pos, f4(W)},
        {(void**) &m.d_pq, f4(W + (long long) W * m.n_head)}, {(void**) &m.d_ids, f4(DflyDrafter::kMaxK)},
        {(void**) &m.d_flags, f4((long long) DflyDrafter::kMaxK * kernels::kNonfiniteBlocks)},
        {&m.arg_sc, (std::size_t) kernels::argmax_rows_scratch_bytes(1)},
        {(void**) &m.emb_row, m.emb_row_bytes},
        {(void**) &m.d_kv_pos, f4(m.max_tokens)},
        {(void**) &m.d_sel, f4((long long) W * m.max_sel)},
        {(void**) &m.emask, f4(n)}, {(void**) &m.e0, f4(n)}, {(void**) &m.eprev, f4(n)},
        {(void**) &m.x, f4((long long) W * n)}, {(void**) &m.u, f4((long long) W * n)},
        {(void**) &m.attn_out, f4((long long) W * n)}, {(void**) &m.t, f4((long long) W * n)},
        {(void**) &m.ffn_out, f4((long long) W * n)}, {(void**) &m.hs, f4((long long) W * n)},
        {&m.u_q8, kernels::native_q8_1_bytes_wide(n, W)}, {&m.t_q8, kernels::native_q8_1_bytes_wide(n, W)},
        {(void**) &m.f_in, f4(2 * n)}, {&m.f_q8, kernels::native_q8_1_bytes_wide(2 * n, 1)},
        {(void**) &m.f_delta, f4(n)},
        {&m.proj_sc, mla_project_scratch_bytes(n, m.q_lora, m.kv_lora, m.qk, m.n_rot, m.n_head, W)},
        {&m.abs_sc, mla_absorb_scratch_bytes(m.nope, W, m.n_head)},
        {(void**) &m.absorbed, f4((long long) W * m.kv_lora * m.n_head)},
        {&m.att_sc, std::max(mla_attention_scratch_bytes(m.kv_lora, m.v_head, m.n_head, 1, m.max_tokens, m.max_sel),
                             mla_attention_scratch_bytes(m.kv_lora, m.v_head, m.n_head, W, m.max_tokens, m.max_sel,
                                                         /*per_query=*/true))},
        {(void**) &m.kqv, f4((long long) W * m.v_head * m.n_head)},
        {&m.out_sc, mla_output_scratch_bytes(n, m.v_head * m.n_head, W)},
        {(void**) &m.gate, f4((long long) W * m.n_ff)}, {(void**) &m.up, f4((long long) W * m.n_ff)},
        {&m.ff_q8, kernels::native_q8_1_bytes_wide(m.n_ff, W)},
        {(void**) &m.c_in, f4(2 * n)}, {&m.c_q8, kernels::native_q8_1_bytes_wide(2 * n, 1)},
        {(void**) &m.cg, f4(m.n_corr)}, {(void**) &m.cu, f4(m.n_corr)},
        {&m.cff_q8, kernels::native_q8_1_bytes_wide(m.n_corr, 1)},
        {(void**) &m.c_delta, f4(n)}, {(void**) &m.h_corr, f4(n)}, {&m.h_q8, kernels::native_q8_1_bytes_wide(n, 1)},
        {(void**) &m.logits, f4(m.vocab)}, {(void**) &m.conf, f4(DflyDrafter::kMaxK)},
    };
    std::size_t total = 0;
    for (auto& e : plan) total += (e.second + 255) & ~(std::size_t) 255;
    if (cudaMalloc(&m.scratch, total) != cudaSuccess) {
        cudaGetLastError();
        m.scratch = nullptr;
        err = "dfly bind: cannot allocate " + std::to_string(total >> 20) + " MiB of scratch";
        return false;
    }
    m.scratch_bytes = total;
    std::size_t at = 0;
    for (auto& e : plan) {
        *e.first = (char*) m.scratch + at;
        at += (e.second + 255) & ~(std::size_t) 255;
    }
    if (cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        m.stream = nullptr;
        m.release_binding();
        err = "dfly bind: cannot create a stream";
        return false;
    }
    try {
        iota_k<<<blocks_for(m.max_tokens), kThreads, 0, m.stream>>>(m.d_kv_pos, m.max_tokens);
        if (cudaMemsetAsync(m.arg_sc, 0, (std::size_t) kernels::argmax_rows_scratch_bytes(1), m.stream) != cudaSuccess)
            throw std::runtime_error("the argmax scratch did not clear");
        kernels::dequant_f32(tw.token_embd_type, tw.token_embd, m.mask_id, 1, n, m.emask, m.stream);
    } catch (const std::exception& e) {
        cudaStreamSynchronize(m.stream);
        m.release_binding();
        err = std::string("dfly bind: ") + e.what();
        return false;
    }
    if (cudaStreamSynchronize(m.stream) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        m.release_binding();
        err = "dfly bind: the setup kernels failed";
        return false;
    }
    m.res = &res;
    m.cache = &cache;
    m.tw = tw;
    m.host_pq.assign((std::size_t) W + (std::size_t) W * m.n_head, 0);
    m.is_bound = true;
    return true;
}

void DflyDrafter::unbind() { p_->release_binding(); }

DflyResult DflyDrafter::propose(int32_t anchor, int a, const DflyContext& ctx, int k, int32_t* draft,
                                float* confidence, std::string& err) {
    Impl& m = *p_;
    err.clear();
    const auto refuse = [&](const std::string& why) { err = "dfly propose: " + why; return DflyResult::abort_request; };
    if (!m.is_bound) return refuse("not bound");
    if (k < 0 || k > m.max_drafts) return refuse("k " + std::to_string(k) + " outside 0.." + std::to_string(m.max_drafts));
    if (k == 0) return DflyResult::ok;
    if (draft == nullptr || confidence == nullptr) return refuse("null output buffers");
    if (m.pending.active) return refuse("a previous proposal is still pending");
    if (anchor < 0 || anchor >= m.vocab) return refuse("anchor " + std::to_string(anchor) + " outside the vocab");
    const int w = k + 1;
    if (a != m.cache->len) return refuse("a = " + std::to_string(a) + " but cache.len = " + std::to_string(m.cache->len));
    if (a < 1 || a + w > m.max_tokens) return refuse("the block [a, a+w) does not fit the KV reservation");
    if (ctx.owner == nullptr || !ctx.owner->valid || ctx.owner->generation != ctx.generation)
        return refuse("the context's capture is no longer valid");
    if (ctx.source_pos != a - 1)
        return refuse("context from position " + std::to_string(ctx.source_pos) + ", expected " + std::to_string(a - 1));
    if (ctx.hidden == nullptr) return refuse("the context has no hidden row");
    if (ctx.engaged) {
        if (ctx.n_sel <= 0 || ctx.n_sel + w > m.max_sel) return refuse("the context's selection width is out of range");
        for (const int32_t* s : ctx.sel) if (s == nullptr) return refuse("the context lacks a layer's selection");
    }
    // The borrowed pointers are the ones bound; a residency or cache rebuilt since is refused, not followed.
    {
        DflyTargetWeights now;
        std::string e2;
        if (!dfly_target_weights(*m.res, now, e2) || now.token_embd != m.tw.token_embd || now.output != m.tw.output)
            return refuse("the target's E/O moved since bind");
        for (int i = 0; i < 5; ++i) {
            const int b = m.layers[(std::size_t) i];
            if (m.cache->kv_cmpr[(std::size_t) b] != m.kv_ptr[(std::size_t) i] ||
                m.cache->k_pe[(std::size_t) b] != m.pe_ptr[(std::size_t) i] || m.cache->max_tokens != m.max_tokens)
                return refuse("the KV cache moved since bind");
        }
    }
    try {
        const DflyResult r = m.run(anchor, a, ctx, k, draft, confidence, err);
        if (r == DflyResult::ok) m.pending = Pending{true, a, w, ctx.generation};
        return r;
    } catch (const std::exception& e) {
        err = std::string("dfly propose: ") + e.what();
        if (cudaStreamSynchronize(m.stream) != cudaSuccess) {
            cudaGetLastError();
            return DflyResult::restart_worker;
        }
        return DflyResult::abort_request;
    }
}

DflyResult DflyDrafter::Impl::run(int32_t anchor, int a, const DflyContext& ctx, int k, int32_t* draft,
                                  float* confidence, std::string& err) {
    const int n = n_embd, w = k + 1;
    cudaStream_t s = stream;
    std::string e;
    const auto check = [&](cudaError_t c, const char* what) {
        if (c != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(c));
    };
    // rms + Q8_1, out of place
    const auto norm_q8 = [&](const float* in, const float* wt, float* out, void* q8, int rows, int cols) {
        if (!kernels::native_rms_norm_quantize_q8_1(in, wt, out, q8, rows, cols, eps, s)) {
            kernels::rms_norm_block(in, wt, out, rows, cols, eps, s);
            kernels::native_quantize_q8_1_wide(out, q8, cols, rows, s);
        }
    };
    const auto F = [&](int gi) { return (const float*) glob[gi].p; };

    // A non-sticky error left by an earlier, unrelated call must not be read as this round's failure below.
    (void) cudaGetLastError();
    for (int j = 0; j < w; ++j) host_pos[(std::size_t) j] = a + j;
    check(cudaMemcpyAsync(d_pos, host_pos.data(), (std::size_t) w * 4, cudaMemcpyHostToDevice, s), "positions");
    // Each query's key count (its own position + 1) and the group -> head map, for the one-call attention.
    MlaPerQuery pq{};
    if (pq_ok) {
        for (int j = 0; j < w; ++j) host_pq[(std::size_t) j] = a + j + 1;
        for (int x = 0; x < n_head * w; ++x) host_pq[(std::size_t) (w + x)] = x / w;
        check(cudaMemcpyAsync(d_pq, host_pq.data(), (std::size_t) (w + n_head * w) * 4, cudaMemcpyHostToDevice, s),
              "per-query counts");
        pq.n_kv = d_pq;
        pq.head_of = d_pq + w;
    }

    // ---- x = E([anchor, MASK x k]); fusion into row 0 only
    kernels::dequant_f32(tw.token_embd_type, tw.token_embd, anchor, 1, n, e0, s);
    check(cudaMemcpyAsync(x, e0, (std::size_t) n * 4, cudaMemcpyDeviceToDevice, s), "embedding");
    for (int j = 1; j < w; ++j)
        check(cudaMemcpyAsync(x + (std::size_t) j * n, emask, (std::size_t) n * 4, cudaMemcpyDeviceToDevice, s),
              "mask embedding");
    kernels::rms_norm_block(e0, F(G_F_ENORM), f_in, 1, n, eps, s);            // E first
    kernels::rms_norm_block(ctx.hidden, F(G_F_HNORM), f_in + n, 1, n, eps, s); // then H
    kernels::native_quantize_q8_1_wide(f_in, f_q8, 2 * n, 1, s);
    kernels::native_mmvq(glob[G_F_EH].type, glob[G_F_EH].p, f_q8, f_delta, 2 * n, n, 1, s);
    gate_add_k<<<blocks_for(n), kThreads, 0, s>>>(x, f_delta, F(G_F_GATE), n);

    // ---- five layers
    const float kq_scale = 1.0f / std::sqrt((float) qk);
    for (int i = 0; i < 5; ++i) {
        const Wt* L = layer[i];
        const int blk = layers[(std::size_t) i];
        norm_q8(x, (const float*) L[L_ATTN_NORM].p, u, u_q8, w, n);
        const MatIn u_in = mat_q8_only(u_q8, n);
        MlaProjected pj{};
        mla_project(u, L[L_Q_A].p, L[L_Q_A].type, (const float*) L[L_Q_A_NORM].p, L[L_Q_B].p, L[L_Q_B].type,
                    L[L_KV_A].p, L[L_KV_A].type, (const float*) L[L_KV_NORM].p, n, q_lora, kv_lora, qk, n_rot,
                    n_head, w, d_pos, eps, rope_base, proj_sc, pj, u_in, s);
        // The block's latents go to the target's rows [a, a+w) of this layer: past cache.len, so dead tail
        // that the same-window verify overwrites before anything reads it.
        if (!kv_cache_append(*cache, blk, a, pj.kv_cmpr, pj.k_pe, w, e, s)) throw std::runtime_error(e);
        if (ctx.engaged)
            sel_rows_k<<<w, kThreads, 0, s>>>(ctx.sel[(std::size_t) i], ctx.n_sel, a, w, d_sel);
        const int width = ctx.engaged ? ctx.n_sel + w : 0;
        // Every query as a decode step at its position would run it: in one call a stage (MlaPerQuery), the
        // same arithmetic as the call-a-query loop below, which is kept for a geometry that has no such call.
        if (pq_ok) {
            mla_absorb(pj.q, L[L_K_B].p, L[L_K_B].type, qk, nope, kv_lora, n_head, w, absorbed, abs_sc, s, &pq);
            mla_attention(absorbed, pj.q_pe, cache->kv_cmpr[(std::size_t) blk], cache->k_pe[(std::size_t) blk],
                          kv_format(*cache), (const float*) L[L_SINKS].p, L[L_V_B].p, L[L_V_B].type, kv_lora, n_rot, v_head,
                          n_head, w, a + w, d_pos, d_kv_pos, /*causal_window=*/0, kq_scale, kqv, att_sc, s,
                          max_tokens, nullptr, ctx.engaged ? d_sel : nullptr, width, max_sel, &pq);
        } else
        for (int j = 0; j < w; ++j) {
            float* abs_j = absorbed + (std::size_t) j * kv_lora * n_head;
            mla_absorb(pj.q + (std::size_t) j * qk * n_head, L[L_K_B].p, L[L_K_B].type, qk, nope, kv_lora, n_head,
                       1, abs_j, abs_sc, s);
            mla_attention(abs_j, pj.q_pe + (std::size_t) j * n_rot * n_head, cache->kv_cmpr[(std::size_t) blk],
                          cache->k_pe[(std::size_t) blk], kv_format(*cache), (const float*) L[L_SINKS].p, L[L_V_B].p,
                          L[L_V_B].type, kv_lora, n_rot, v_head, n_head, 1, a + j + 1, d_pos + j, d_kv_pos,
                          /*causal_window=*/0, kq_scale, kqv + (std::size_t) j * v_head * n_head, att_sc, s,
                          max_tokens, nullptr, ctx.engaged ? d_sel + (std::size_t) j * width : nullptr, width,
                          max_sel);
        }
        // gate(u) on the head-wide value, wo, then the DRAFTER's post_attention_norm
        mla_output(kqv, u, L[L_GATE].p, L[L_GATE].type, L[L_WO].p, L[L_WO].type, (const float*) L[L_POST_NORM].p, n,
                   v_head * n_head, w, eps, attn_out, out_sc, s, u_in);
        kernels::add_inplace(x, attn_out, (int64_t) w * n, s);
        // the MLP: no post-FFN norm, so not dense_ffn
        norm_q8(x, (const float*) L[L_FFN_NORM].p, t, t_q8, w, n);
        if (L[L_FFN_GATE].type != L[L_FFN_UP].type ||
            !kernels::native_mmvq_pair(L[L_FFN_GATE].type, L[L_FFN_GATE].p, L[L_FFN_UP].p, t_q8, gate, up, n, n_ff,
                                       w, s)) {
            kernels::native_mmvq(L[L_FFN_GATE].type, L[L_FFN_GATE].p, t_q8, gate, n, n_ff, w, s);
            kernels::native_mmvq(L[L_FFN_UP].type, L[L_FFN_UP].p, t_q8, up, n, n_ff, w, s);
        }
        kernels::native_swiglu_quantize_q8_1(gate, up, ff_q8, n_ff, w, s);
        kernels::native_mmvq(L[L_FFN_DOWN].type, L[L_FFN_DOWN].p, ff_q8, ffn_out, n_ff, n, w, s);
        kernels::add_inplace(x, ffn_out, (int64_t) w * n, s);
    }
    kernels::rms_norm_block(x, F(G_OUT_NORM), hs, w, n, eps, s);

    // Keep correction-head picks on device. Break argmax ties by lowest ID.
    for (int j = 1; j <= k; ++j) {
        const float* hj = hs + (std::size_t) j * n;
        const float* prev = j == 1 ? e0 : eprev;
        kernels::rms_norm_block(hj, F(G_C_HNORM), c_in, 1, n, eps, s);       // H first
        kernels::rms_norm_block(prev, F(G_C_ENORM), c_in + n, 1, n, eps, s); // then E
        kernels::native_quantize_q8_1_wide(c_in, c_q8, 2 * n, 1, s);
        kernels::native_mmvq(glob[G_C_GATE].type, glob[G_C_GATE].p, c_q8, cg, 2 * n, n_corr, 1, s);
        kernels::native_mmvq(glob[G_C_UP].type, glob[G_C_UP].p, c_q8, cu, 2 * n, n_corr, 1, s);
        kernels::native_swiglu_quantize_q8_1(cg, cu, cff_q8, n_corr, 1, s);
        kernels::native_mmvq(glob[G_C_DOWN].type, glob[G_C_DOWN].p, cff_q8, c_delta, n_corr, n, 1, s);
        check(cudaMemcpyAsync(h_corr, hj, (std::size_t) n * 4, cudaMemcpyDeviceToDevice, s), "h copy");
        kernels::add_inplace(h_corr, c_delta, n, s);
        if (tw.output_type == 30) {
            // The shared BF16 head consumes the original F32 activation, as in the target.
            kernels::bf16_gemv_fp32_mmvf_cols(h_corr, static_cast<const uint16_t*>(tw.output), logits,
                                             n, vocab, 1, s);
        } else {
            kernels::native_quantize_q8_1_wide(h_corr, h_q8, n, 1, s);
            kernels::native_mmvq(tw.output_type, tw.output, h_q8, logits, n, vocab, 1, s);
        }
        kernels::argmax_rows(logits, 1, vocab, arg_sc, d_ids + (j - 1), s);
        kernels::logits_nonfinite_rows(logits, 1, vocab, d_flags + (std::size_t) (j - 1) * kernels::kNonfiniteBlocks, s);
        // The confidence head consumes the corrected hidden state.
        confidence_k<<<1, kThreads, 0, s>>>(h_corr, F(G_CONF_W), F(G_CONF_B), conf + (j - 1), n);
        if (j < k) {
            row_by_id_k<<<blocks_for((int) emb_row_bytes), kThreads, 0, s>>>(
                (const uint8_t*) tw.token_embd, emb_row_bytes, d_ids + (j - 1), vocab, emb_row);
            kernels::dequant_f32(tw.token_embd_type, emb_row, 0, 1, n, eprev, s);
        }
    }
    check(cudaMemcpyAsync(host_ids.data(), d_ids, (std::size_t) k * 4, cudaMemcpyDeviceToHost, s), "draft copy");
    check(cudaMemcpyAsync(host_flags.data(), d_flags, (std::size_t) k * kernels::kNonfiniteBlocks * 4,
                          cudaMemcpyDeviceToHost, s), "flag copy");
    check(cudaMemcpyAsync(confidence, conf, (std::size_t) k * 4, cudaMemcpyDeviceToHost, s), "confidence copy");
    if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        err = "dfly propose: the device failed at the end of the round";
        return DflyResult::restart_worker;
    }
    for (int j = 1; j <= k; ++j) {
        bool bad = host_ids[(std::size_t) (j - 1)] < 0 || host_ids[(std::size_t) (j - 1)] >= vocab;
        for (int b = 0; b < kernels::kNonfiniteBlocks; ++b)
            bad = bad || host_flags[(std::size_t) (j - 1) * kernels::kNonfiniteBlocks + b] != 0;
        if (bad) {
            err = "dfly propose: slot " + std::to_string(j) + " has non-finite logits";
            return DflyResult::safe_fallback;   // drained above; nothing in flight, nothing pending
        }
        draft[j - 1] = host_ids[(std::size_t) (j - 1)];
    }
    return DflyResult::ok;
}

}  // namespace strata::hy4_flash
