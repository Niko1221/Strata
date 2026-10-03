// qwen36_model.hpp - Qwen3.6-35B-A3B (arch qwen35moe) decoder on one GPU, one token at a time (Phase 1).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "qwen36_config.hpp"

namespace q36 {

struct Mat {
    int type = -1;                 // ggml type id
    const void* d = nullptr;       // device pointer
    int n_in = 0, n_out = 0;
    int n_expert = 1;
    size_t expert_stride = 0;      // bytes between consecutive experts (3-D tensors)
};

struct LayerWeights {
    bool gdn = false;
    float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, z, alpha, beta, ssm_out;
    float *conv = nullptr, *ssm_a = nullptr, *dt = nullptr, *ssm_norm = nullptr;
    // attention
    Mat q, k, v, o;
    float *qn = nullptr, *kn = nullptr;
    // MoE
    Mat gate_inp, gate_exps, up_exps, down_exps, sh_gate, sh_up, sh_down;
    float* sh_gate_inp = nullptr;  // [n_embd] vector
    // Grouped experts (strata::kernels::native_expert_grouped): when `grouped`, gate/up/down of every expert are one blob
    // [gate rows | up rows | down rows] at blobs + e * blob_stride, and gate_exps/up_exps/down_exps above are NOT loaded.
    bool grouped = false;
    const void* blobs = nullptr;
    size_t blob_stride = 0;
    int gu_type = -1, d_type = -1;
};

class Model {
  public:
    // want_mtp: also load the MTP head (blk.<n_layer>.nextn.*) if the GGUF has it; mtp_k = most draft tokens per round (<= 7);
    // kv_mode: 0 = F16 KV cache, 1 = int8 (+ F16 scale per 32 values, ~47% smaller); kv_unified: one allocation for all K/V.
    // mtp_variant: input conventions of the head (see Model::mtp_probe / --selftest-mtp).
    Model(const std::string& gguf_path, int max_ctx, bool want_mtp = false, int mtp_k = 3, int mtp_variant = 1,
          int kv_mode = 0, bool kv_unified = false);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Config& cfg() const { return cfg_; }
    int max_ctx() const { return max_ctx_; }
    int pos() const { return pos_; }
    void reset();                                   // forget everything (zero GDN state / conv history, pos = 0)
    // Feed one token at position pos(); then pos()++.  With want_logits the logits are computed and copied to the host.
    void forward(int token, bool want_logits);
    // Block prefill: feed n tokens (1 <= n <= kMaxBatch) at positions pos() .. pos()+n-1 in one pass.  Equivalent to n calls
    // of forward() (checked by --selftest-batch); with want_logits only the LAST token's logits are produced.
    static constexpr int kMaxBatch = 128;
    void forward_batch(const int* tokens, int n, bool want_logits);
    const std::vector<float>& logits() const { return h_logits_; }
    size_t vram_bytes() const { return vram_; }

    // ---- speculative decoding with the MTP head.  Loop (see main.cpp): `next` is a sampled token not yet fed to the model;
    // mtp_draft(next, k) guesses the k tokens after it; forward_verify({next, d1..dk}) runs them in ONE pass and gives the logits
    // after every row; the caller samples from row i and accepts d(i+1) when it is the sample; commit(keep) then keeps the
    // first `keep` rows (state after row keep-1) and drops the rest.  Draft quality never changes the output distribution.
    bool mtp_enabled() const;
    int spec_max() const;
    int mtp_variant() const;
    void set_mtp_variant(int v);
    void mtp_draft(int next, int k, std::vector<int>& out);
    void forward_verify(const int* tokens, int n);
    const float* verify_logits(int row) const { return h_vlogits_.data() + (size_t)row * cfg_.n_vocab; }
    void commit(int keep);
    // ---- prompt snapshots.  The GDN state cannot be rewound, so reusing a cached prefix after the conversation diverged (a chat
    // template that rewrites the previous turn, an agent that edits its history) needs saved states.  A snapshot is the GDN state
    // + conv history (+ MTP carry) after the first n tokens, kept in pinned host RAM (~62 MB each); the K/V of those positions is
    // still in the GPU caches as long as nothing else was written there (invalidate_snapshots takes care of that).
    void set_snapshot_limit(int n);                          // 0 disables (default 12)
    void snapshot(const int* tokens, int n);                 // requires pos() == n
    int restore_best(const std::vector<int>& prompt);        // longest saved prefix of `prompt` shorter than it: its length, or -1
    // Call once per request, with the position it starts from, BEFORE processing: drops the snapshots whose K/V it will overwrite.
    void invalidate_snapshots(const std::vector<int>& prompt, int start);
    int snapshot_count() const { return (int)snaps_.size(); }

    // MTP accuracy at positions >= from of `seq` under the current variant (fills *counted with the number of positions).
    double mtp_probe(const std::vector<int>& seq, int from, int* counted);

  private:
    struct Snap {
        int pos = 0;
        std::vector<int> toks;
        float* host = nullptr;
        bool carry = false;
        unsigned long long stamp = 0;
    };
    std::vector<Snap> snaps_;
    unsigned long long stamp_ = 0;
    int snap_max_ = 12;
    void batch_pass(const int* tokens, int n, int mode);
    struct Impl;
    Config cfg_;
    int max_ctx_ = 0, pos_ = 0;
    size_t vram_ = 0;
    std::vector<float> h_logits_, h_vlogits_;
    std::vector<int> v_tokens_;
    int v_pos0_ = 0, v_n_ = 0;
    bool v_single_ = false;
    std::unique_ptr<Impl> p_;
};

}  // namespace q36
