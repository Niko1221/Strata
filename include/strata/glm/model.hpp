// include/strata/glm/model.hpp - GLM-5.3 (glm_moe_dsa): the dense weights, the KV cache and the forward pass.
//
// THE FORWARD PASS, per layer, on S rows at positions pos0 .. pos0 + S - 1 (colibri.c is the reference this was
// checked against token for token, see docs/GLM53.md):
//
//   h  = rmsnorm(x, input_layernorm)
//   MLA (absorbed):
//     qr = rmsnorm(q_a(h), q_a_layernorm);  q = q_b(qr)          -> per head 192 nope + 64 rope
//     c  = kv_a(h)                                                -> 512 latent + 64 rope
//     L  = rmsnorm(c[:512], kv_a_layernorm);  R = rope(c[512:])   -> the cache row (shared by all heads)
//     per head: qabs = W_k^T q_nope (the key half of kv_b, absorbed);  score_t = (qabs.L_t + rope(q_rope).R_t) / 16
//               ctx = W_v (sum_t softmax(score)_t L_t)            (the value half of kv_b)
//     x += o(ctx)
//   h  = rmsnorm(x, post_attention_layernorm)
//   layers 0..2: x += down(silu(gate(h)) * up(h))
//   layers 3..77: sigmoid router + e_score_correction_bias picks 8 of 256 (the bias only picks); the weights are the
//                 sigmoids, renormalised and scaled by 2.5; x += sum_k w_k expert_k(h) + shared_expert(h)
//   final: logits = lm_head(rmsnorm(x, norm))
//
// WHAT IS NOT HERE, because this container carries no weights for it (docs/GLM53.md): the DSA indexer (attention
// is dense over the whole context, which is exactly what DSA computes up to index_topk = 2048 tokens), and the
// NextN/MTP block.
#pragma once

#include "strata/glm/container.hpp"
#include "strata/glm/experts.hpp"
#include "strata/glm/io.hpp"
#include "strata/glm/kernels.hpp"
#include "strata/glm/pool.hpp"
#include "strata/glm/kv.hpp"
#include "strata/glm/watchdog.hpp"
#ifdef STRATA_GLM_CUDA
#include "strata/glm/cuda.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

struct ModelOptions {
    int threads = 0;              ///< compute threads (0: physical cores)
    int io_threads = 8;
    int max_context = 8192;
    uint64_t expert_ram = 0;      ///< bytes for the routed-expert cache (0: what is left of `ram_budget`)
    uint64_t ram_budget = 0;      ///< the engine's RAM budget (0: 85% of the available memory at start)
    KvFormat kv = KvFormat::F32;
    bool prefetch = false;
    bool gpu = false;
    int device = 0, promote_after = 2;
    uint64_t vram_budget = 0, vram_reserve = 700ull << 20;
    bool gpu_prefill = true;
    double watchdog_seconds = 1800;
    bool verbose = true;
};

struct LayerW {
    const float *in_ln = nullptr, *post_ln = nullptr, *q_a_ln = nullptr, *kv_a_ln = nullptr;
    Q4 q_a, q_b, kv_a, kv_b, o;
    bool sparse = false;
    Q4 gate, up, down;                      // dense layers
    const float* router = nullptr;          // sparse: n_experts x hidden, f32
    const float* router_bias = nullptr;     // sparse: e_score_correction_bias
    Q4 sh_gate, sh_up, sh_down;             // sparse: the shared expert
};

struct ForwardStats {
    double attn_ms = 0, moe_ms = 0, dense_ms = 0, head_ms = 0, expert_wait_ms = 0;
};

class GlmModel {
public:
    bool load(const std::string& dir, const ModelOptions& opt, std::string& err);

    const GlmConfig& config() const { return c_; }
    int max_context() const { return max_ctx_; }
    int cached_tokens() const { return n_past_; }
    /// Forget KV rows from `n` on (a conversation that diverged at n keeps its first n tokens).
    void truncate(int n) { if (n < n_past_) n_past_ = std::max(0, n); }

    /// Run `S` tokens at positions cached_tokens() .. + S - 1, appending them to the cache.  `logits` gets the
    /// last row's logits (vocab floats) when non-null; `all_logits` gets every row's (S x vocab) when non-null.
    /// `on_layer(n)` runs after each of the n_layers layers (a long prompt's progress: one forward of 1,536 tokens
    /// takes minutes, and the server ends an engine that says nothing for too long).
    bool forward(const int* ids, int S, float* logits, float* all_logits, std::string& err,
                 const std::atomic<bool>* cancel = nullptr, int attention_chunk = 1024,
                 const std::function<void(int)>* on_layer = nullptr);
    /// All layers see the entire prompt, so each routed expert is read at most once per layer.
    bool forward_prefill(const int* ids, int S, float* logits, std::string& err,
                         const std::atomic<bool>* cancel = nullptr, int attention_chunk = 1024,
                         const std::function<void(int)>* on_layer = nullptr) {
        return forward(ids, S, logits, nullptr, err, cancel, attention_chunk, on_layer);
    }

    ExpertCache& experts() { return *experts_; }
    Pool& pool() { return *pool_; }
    ForwardStats& stats() { return st_; }
    uint64_t dense_bytes() const { return dense_bytes_; }
    uint64_t kv_bytes() const { return kv_.bytes(); }
    const char* kv_type() const { return kv_name(kv_.format()); }
    uint64_t io_bytes() const { return io_->bytes_read(); }
    int gpu_slots() const;
    uint64_t gpu_hits() const;

private:
    const uint8_t* load_raw(const std::string& name, uint64_t expect_bytes, const char* dtype, std::string& err);
    bool load_q4(const std::string& name, int O, int I, Q4& out, std::string& err);
    const float* load_f32(const std::string& name, uint64_t n, std::string& err);

    void attention(int layer, const float* h, int S, int pos0, float* out);
    void dense_mlp(const LayerW& L, const float* h, int S, float* out);
    void gemm(const Q4& w, const float* x, int S, float* out);
    void prefetch(int layer, const float* x);
    bool moe(int layer, const float* h, int S, float* out, std::string& err, const std::atomic<bool>* cancel);

    GlmConfig c_;
    Container ct_;
    std::unique_ptr<Pool> pool_;
    std::unique_ptr<IoPool> io_;
    std::unique_ptr<ExpertCache> experts_;
    std::unique_ptr<Watchdog> watchdog_;
    struct Pending;
    std::shared_ptr<Pending> pending_;
#ifdef STRATA_GLM_CUDA
    std::unique_ptr<CudaBackend> gpu_;
#endif
    bool prefetch_ = false, gpu_prefill_ = false;
    std::vector<void*> blocks_;              // aligned reads backing the dense weights
    uint64_t dense_bytes_ = 0;
    Q8R embed_, lm_head_;
    const float* final_norm_ = nullptr;
    std::vector<LayerW> layers_;
    Rope rope_;
    int max_ctx_ = 0;
    int n_past_ = 0;
    KvCache kv_;                            // per layer: max_ctx x (latent | roped key)
    ForwardStats st_;

public:
    ~GlmModel();
};

}  // namespace strata::glm
