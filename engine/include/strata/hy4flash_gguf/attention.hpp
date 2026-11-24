#pragma once
#include "strata/hy4flash_gguf/vision.hpp"

#include "strata/hy4flash_gguf/mat.hpp"
#include <cstddef>
#include <cstdint>

/// Gated MLA uses 192 unrotated and 64 rotary channels per query head.
namespace strata::hy4_flash {

/// Apply unscaled GGML normal RoPE to adjacent pairs. Channels beyond n_rot pass through.
/// x is [head_dim, n_head, n_tok] and out may alias x. src_head_dim and src_off select a strided input slice.
void rope_norm(const float* x, float* out, int head_dim, int n_rot, int n_head, int n_tok,
               const int32_t* positions, float freq_base, void* stream, int src_head_dim = 0,
               int src_off = 0, ImageRope image = {});

/// Rotate the last n_rot channels of each head_dim row. Supports in-place output.
void rope_norm_head_tail(const float* x, float* out, int head_dim, int n_rot, int n_head, int n_tok,
                         const int32_t* positions, float freq_base, void* stream, ImageRope image = {});

/// Gather a contiguous slice from each strided head row.
void gather_slice(const float* src, float* out, int src_head_dim, int off, int n, int n_head, int n_tok,
                  void* stream);

/// Project normalized hidden rows into the query and compressed KV branches.
std::size_t mla_project_scratch_bytes(int n_embd, int q_lora, int kv_lora, int qk_head, int n_rot,
                                      int n_head, int n_tok);

struct MlaProjected {
    float* q = nullptr;         ///< [qk_head * n_head, nt]; each head's nope part at offset 0, rope at `nope`
    float* q_pe = nullptr;      ///< [n_rot, n_head, nt], roped
    float* kv_cmpr = nullptr;   ///< [kv_lora, nt], normed
    float* k_pe = nullptr;      ///< [n_rot, 1, nt], roped
    float* q_resid = nullptr;   ///< [q_lora, nt]; kept because the indexer reads it too
    /// Reuse wq_b's quantized query residual for the indexer's query projection.
    const void* q_resid_q8 = nullptr;
};

/// Share the staged activation across projections that consume the same hidden state.
void mla_project(const float* cur, const void* wq_a, int wq_a_type, const float* q_a_norm, const void* wq_b,
                 int wq_b_type, const void* wkv_a_mqa, int wkv_a_mqa_type, const float* kv_a_norm,
                 int n_embd, int q_lora, int kv_lora, int qk_head, int n_rot, int n_head, int n_tok,
                 const int32_t* positions, float rms_eps, float rope_base, void* scratch, MlaProjected& out,
                 const MatIn& cur_in,
                 void* stream, ImageRope image = {});

/// Absorb the per-head key projection into the query before scoring compressed keys.
struct MlaPerQuery {
    const int32_t* n_kv = nullptr;     ///< [n_tok]: query t sees cache slots [0, n_kv[t])
    const int32_t* head_of = nullptr;  ///< [n_head * n_tok]: group h * n_tok + t -> h
};

/// Whether `MlaPerQuery` can run for this geometry: the tensor-core attention and the one-column grouped
/// wk_b/wv_b products. A caller that gets false keeps calling once a query.
bool mla_per_query_supported(int kv_lora, int n_rot, int n_head, bool kv_fp16, int wk_b_type, int wv_b_type);

/// Fill n_kv[t] = d_len[0] + t + 1 and head_of[g] = g / n_tok on device for graph replay.
void mla_per_query_meta(const int32_t* d_len, int32_t* n_kv, int32_t* head_of, int n_tok, int n_head,
                        void* stream);

void mla_absorb(const float* q, const void* wk_b, int wk_b_type, int qk_head, int nope, int kv_lora,
                int n_head, int n_tok, float* out, void* scratch, void* stream,
                const MlaPerQuery* per_query = nullptr);

/// Scratch for one mla_absorb call: the Q8_1 staging of one head's nope rows.
std::size_t mla_absorb_scratch_bytes(int nope, int n_tok, int n_head);

/// Score visible keys with the per-head sink, reduce the latent values and apply the value projection.
std::size_t mla_attention_scratch_bytes(int kv_lora, int v_head, int n_head, int n_tok, int n_kv,
                                       int max_sel = 0, bool per_query = false);

/// `kv_cmpr` and `k_pe` are the cache's rows in `kv_format` (a KvFormat, kv_cache.hpp): F32, F16, or a Q8_0
/// latent with an F16 rope key. A `bool` still means F32 / F16 (KvCache::fp16).
void mla_attention(const float* q_absorbed, const float* q_pe, const void* kv_cmpr, const void* k_pe,
                   int kv_format,
                   const float* sinks, const void* wv_b, int wv_b_type, int kv_lora, int n_rot, int v_head,
                   int n_head, int n_tok, int n_kv, const int32_t* q_pos, const int32_t* kv_pos,
                   int causal_window, float kq_scale, float* out, void* scratch, void* stream,
                   int kv_stride = 0, const int32_t* d_n_kv = nullptr,
                   /// The compressed indexer's selection, `[n_sel, n_tok]` absolute key positions, or null
                   /// for dense attention over the whole causal prefix. A negative entry is a padding slot
                   /// and is skipped, which is how a row shorter than `n_sel` is expressed.
                   const int32_t* sel = nullptr, int n_sel = 0, int max_sel = 0,
                   /// See `MlaPerQuery`. Then `n_kv` is only the largest of the per-query counts and
                   /// `d_n_kv` is unused; the scratch must come from the `per_query` sizing.
                   const MlaPerQuery* per_query = nullptr);

/// Apply the attention gate, output projection and post-attention normalization.
std::size_t mla_output_scratch_bytes(int n_embd, int attn_width, int n_tok);

void mla_output(float* attn_kqv, const float* cur, const void* w_gate, int w_gate_type, const void* wo,
                int wo_type, const float* post_norm, int n_embd, int attn_width, int n_tok, float rms_eps,
                float* out, void* scratch, void* stream, const MatIn& cur_in = {},
                void* gate_join = nullptr);

/// Write the attention gate into the gate region of mla_output scratch.
void mla_gate(const void* w_gate, int w_gate_type, const MatIn& cur_in, int n_embd, int attn_width,
              int n_tok, void* scratch, void* stream);

/// Block 0 uses plain SwiGLU followed by post-FFN normalization. Only routed experts use the clamp.
std::size_t dense_ffn_scratch_bytes(int n_embd, int n_ff, int n_tok);

/// Dequantize the weight to FP32 for projection without Q8_1 activation rounding.
bool f32_matvec_dequant(const void* w, int type, const float* x, float* y, int n_in, int n_out, int n_tok,
                        void* stream);

/// `x_q8` is the caller's Q8_1 image of `x`, when it already has one - the same one the router and the
/// shared expert read, so the block quantises its hidden state once rather than once per consumer.
void dense_ffn(const float* x, const void* w_gate, int w_gate_type, const void* w_up, int w_up_type,
               const void* w_down, int w_down_type, const float* post_norm, int n_embd, int n_ff, int n_tok,
               float rms_eps, float* out, void* scratch, void* stream, const MatIn& x_in = {});

/// Apply latent MoE routing, combine weighted expert outputs and add the shared expert.
std::size_t latent_moe_scratch_bytes(int n_embd, int expert_dim, int expert_ff, int n_expert, int k,
                                     int n_tok, int gate_type, int up_type, int down_type, int mmq_slots = 0, int mmq_rows = 0,
                                     bool exact_window = false);

/// Cumulative cost of the per-block routed-id readback, which drains the pipeline every routing block.
void latent_moe_route_sync_cost(double& secs, double& backlog, long& count);

/// Expert matrix addresses are indexed from each plane's base in units of its byte stride.
struct ExpertGroup {
    const void* gate = nullptr;
    std::size_t gate_stride = 0;
    const void* up = nullptr;
    std::size_t up_stride = 0;
    const void* down = nullptr;
    std::size_t down_stride = 0;
    const int32_t* index = nullptr;   ///< device, k entries; nullptr means group j is at offset j
};

struct alignas(64) ExpertAsyncMailbox {
    volatile uint32_t request = 0, plan_ready = 0, miss_ready = 0, done = 0, error = 0;
    int32_t ids[16] = {};
    int32_t hit[16] = {};
    int32_t miss[16] = {};
    // GPU-authored epoch: all selected experts were in the stable device residency snapshot.
    // Published before request; CPU validates the hit snapshot and updates CLOCK without rewriting it.
    volatile uint32_t fast_epoch = 0;
    // CPU lane: input_ready is GPU-published; cpu_ready is published by the CPU
    // only after its unweighted result rows have reached device memory.
    volatile uint32_t input_ready = 0, cpu_ready = 0, cpu_mask = 0;
};
struct ExpertAsyncGroup {
    ExpertGroup planes;
    ExpertAsyncMailbox* mailbox = nullptr; // CUDA mapped device alias
    int32_t* hit_index = nullptr;          // stable device[16]
    int32_t* miss_index = nullptr;         // stable device[16]
    void* shared_stream = nullptr;       // preallocated nonblocking CUDA stream
    void* shared_fork_event = nullptr;   // preallocated disable-timing events
    void* shared_join_event = nullptr;
    void* shared_q8 = nullptr;           // exclusive shared-branch quantization scratch
    // Optional exact all-hit plan. Row of the global block/expert -> pool-local slot table.
    // Host refreshes the entire table only while the GPU is stopped on a real miss, before B.
    const int32_t* device_slot_of = nullptr;
    int n_expert = 0;
    int permanent_slots = 0;
    bool single_phase=false; // opt-in wait-all route; safe for actual misses
    float* cpu_input = nullptr;       // mapped host input, device alias (latent width)
    float* cpu_parts = nullptr; // stable writable device [16, latent width]; may be final mixed parts
    uint32_t* device_epoch = nullptr; // optional device-owned [request, fast_epoch]; host copies remain
    // Mapped timestamps bracket the GPU branch. Read them only after the layer signals completion.
    unsigned long long* split_stats = nullptr;
};

struct LatentMoeWeights {
    const struct WindowExperts* window = nullptr;
    /// CPU execution context for host-resident MTP experts.
    const struct DraftExperts* draft_cpu = nullptr;
    bool verify_exact = false;
    const ExpertAsyncGroup* async_group = nullptr;
    const float* gate_inp = nullptr;   ///< F32 [n_embd, n_expert] - the router, unquantised in this model
    const float* exp_probs_b = nullptr;///< F32 [n_expert] or nullptr
    const void* routed_down = nullptr; int routed_down_type = 0;
    const void* gate_exps = nullptr;   int gate_exps_type = 0;   ///< [expert_ff, expert_dim, n_expert]
    const void* up_exps = nullptr;     int up_exps_type = 0;
    const void* down_exps = nullptr;   int down_exps_type = 0;
    const void* routed_up = nullptr;   int routed_up_type = 0;
    const void* gate_shexp = nullptr;  int gate_shexp_type = 0;
    const void* up_shexp = nullptr;    int up_shexp_type = 0;
    const void* down_shexp = nullptr;  int down_shexp_type = 0;
    const float* post_norm = nullptr;  ///< F32 [n_embd] - post_ffw_norm
    /// What the grouped expert matvecs are told the call's token count is. 0 (default): `n_tok`, as always.
    /// 1: the one-token partition, whatever `n_tok` is - what an exact multi-row verify asks for so its rows
    /// take the template a decode step takes. Opt-in; the default path is unchanged.
    int context_n_tok = 0;
    /// Sort verify rows by expert to reuse weight loads while preserving single-token arithmetic.
    bool exact_window = false;
    /// Use token-major groups without sorting or sharing weights across verify rows.
    bool exact_wide = false;

    /// Fetch selected weights by role (gate, up, down). Each pointer remains valid until the next fetch for
    /// that role. A null pointer aborts the call.
    bool (*fetch_experts)(const int32_t* ids, int k, ExpertGroup* out, void* stream, void* user) = nullptr;
    void* fetch_user = nullptr;

    /// Stage up to batch_max expert IDs from host memory into the prompt ring.
    /// The returned device map contains the ring slot for each staged expert and -1 elsewhere.
    /// The callback may synchronize the supplied stream before returning.
    bool (*stage_experts)(int block_id, const int32_t* ids, int n, ExpertGroup* out,
                          const int32_t** slot_of_dev, void* stream, void* user) = nullptr;
    // Optional bounded lookahead. plan owns a copy of the miss list until all batches are consumed;
    // release records the last device consumer, including admission, before a bank can be overwritten.
    void (*stage_plan)(int block_id, const int32_t* ids, int n, void* stream, void* user) = nullptr;
    void (*stage_release)(void* stream, void* user) = nullptr;
    void* stage_user = nullptr;
    int stage_batch_max = 0;      ///< the ring's width; 0 disables the batched prompt path
    int stage_scratch_rows = 0; ///< optional group workspace row capacity
    int stage_scratch_slots = 0; ///< maximum physical-slot extent reserved by the caller
    int stage_n_slot = 0;         ///< slots the ring exposes, i.e. the MMQ launch's expert count
    int stage_block = -1;         ///< which block's experts to stage
    /// Device map from expert ID to pool-local slot, or -1 for a miss that requires staging.
    const int32_t* stage_slot_of = nullptr;
    const ExpertGroup* stage_resident = nullptr;
    int stage_resident_slots = 0;

    /// A fully resident layer keeps a stable expert-to-slot mapping across calls.
    const ExpertGroup* resident = nullptr;
    const int32_t* resident_slot_of = nullptr;   ///< device, `n_expert` entries, expert -> slot
    /// For full residency, block_base locates the layer's first pool-local slot.
    int resident_base = -1;
};

/// Optional scratch views for comparing intermediate expert results.
struct LatentMoeTrace {
    const float* routed_in = nullptr;   ///< [expert_dim, n_tok]
    const float* parts = nullptr;       ///< [expert_dim, k, n_tok] - BEFORE the router weights
    const float* latent = nullptr;      ///< [expert_dim, n_tok] - the weighted sum
    const float* shexp = nullptr;       ///< [n_embd, n_tok] - the shared expert alone
    const int32_t* ids = nullptr;       ///< [k, n_tok]
    const float* weights = nullptr;     ///< [k, n_tok]
    const float* logits = nullptr;      ///< [n_expert, n_tok] - the router's input scores
    const float* first_gate = nullptr;  ///< [expert_ff] - gate_e . routed_in for (token 0, slot 0)
    const float* first_up = nullptr;    ///< [expert_ff] - the same expert's up projection
};

void latent_moe(const float* cur, const LatentMoeWeights& w, int n_embd, int expert_dim, int expert_ff,
                int n_expert, int k, int n_tok, float routed_scale, float swiglu_clamp, float rms_eps,
                float* out, void* scratch, void* stream, LatentMoeTrace* trace = nullptr,
                const MatIn& cur_in = {});

/// Look up each token embedding and initialize the hyper-connection residual streams.
std::size_t embed_tokens_scratch_bytes(int n_embd, int n_tok);

void embed_tokens(const void* token_embd, int type, const int32_t* tokens, int n_embd, int hc, int n_tok,
                  float* out, void* scratch, void* stream);

/// Reduce the residual streams, normalize and apply the output head.
std::size_t output_head_scratch_bytes(int n_embd, int hc, int vocab, int n_tok);

void output_head(const float* x, const void* hc_fn, int hc_fn_type, const float* hc_scale,
                 const float* hc_base, const float* output_norm, const void* output, int output_type,
                 int n_embd, int hc, int vocab, int n_tok, float rms_eps, float hc_eps, float* logits,
                 void* scratch, void* stream, MatCtx* mc = nullptr, float* hidden_out = nullptr,
                 int logits_rows = -1,
                 /// Optional: copy the LAST `export_rows` rows of the hidden (after `output_norm`, before the LM
                 /// head's quantisation) to this device pointer, `[export_rows, n_embd]`. nullptr: nothing.
                 float* export_hidden = nullptr, int export_rows = 0);

}  // namespace strata::hy4_flash
