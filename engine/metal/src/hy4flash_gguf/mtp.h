// engine/metal/src/hy4flash_gguf/mtp.h - the MTP draft block (blk.42) and greedy self-speculative decoding.
//
// WHAT THE BLOCK IS (llama.cpp hy-v4-flash.cpp graph_mtp; src/hy4flash_gguf/mtp.cu):
//
//     row q  = (x_q, h_{q-1})                        h = the trunk's result_norm, h_{-1} = 0 (llama.cpp's pending_h)
//     cur    = eh_proj . [rms(embd(x_q)) * enorm ; rms(h_{q-1}) * hnorm]
//     cur   += post_attention_norm(MLA(rms(cur) * attn_norm))      a sparse block with its own indexer, at position q
//     cur   += post_ffw_norm(latent_moe(rms(cur) * ffn_norm))       BF16 experts, top-16, + the shared expert
//     s_q    = rms(cur) * shared_head_norm                          fed to the NEXT draft step as its h
//     draft  = argmax(output . s_q)                                 the trunk's own head weight, shared
//
// A ROW'S KV DEPENDS ON (x_q, h_{q-1}) ONLY - eh_proj and attn_norm are before anything that mixes rows - so the rows
// whose drafts nobody asks for (the prompt, and the positions a verify accepted) are run CACHE-ONLY: their MLA latent
// and indexer rows are written and nothing else (the CUDA engine's `cache_only`).  Only the last row of a draft step
// runs the attention, the MoE and the head.  That is why the prompt's catch-up costs one 6144 x 3072 GEMV a position
// and no expert reads.
//
// STATE AND ROLLBACK.  `len()` rows are committed: row q < len() was computed from the TRUE token and the
// trunk's h.  A draft chain writes rows past that from the MTP's OWN s, and `truncate(n)` forgets them - rows are
// rewritten by the next run and the pooled-key count is clamped, the same rule as the trunk's (forward.h, reset()).
// The `h` history `hidden(q)` is indexed by MTP row: hidden(q) = h_{q-1}.
//
// THE EXPERTS are 512 x 14.16 MB of BF16 = 7.25 GB, which do not fit beside the trunk's arena.  They are read on
// demand into a fixed LRU of `HY4_FLASH_MTP_EXPERT_GB` (default 2 GB) - a separate pool, so the trunk's arena and its
// profile are not touched - with F_NOCACHE preads on persistent threads.  A draft step needs 16 of them.
//
// QUANTIZED BY DEFAULT (for the draft only): the block's matrices and a copy of the output head run as Q8_0
// (ggml's reference quantizer), the experts from models/derived/<model>/mtp_experts_q8_0.bin, built on the first
// run (3.85 GB; bound to the shard's size and mtime).  HY4_FLASH_MTP_QUANT=bf16 / HY4_FLASH_MTP_HEAD=bf16 turn it off.  It
// changes the drafts at most, never the output: the verify is the trunk's own.
#pragma once

#include "forward.h"
#include "model.h"
#include "mtl_rt.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// Greedy acceptance of a draft chain against the verify window's argmax, exactly.  `drafts[0..k)` were
/// proposed for positions L+1..L+k; `verify[0..k]` is the trunk's argmax at positions L..L+k.  Returns how many
/// drafts are accepted: the longest prefix with drafts[j] == verify[j].  The tokens the round emits are then
/// drafts[0..n) followed by verify[n] - which equals verify[0..n] exactly when the prefix matched.
inline int accept_prefix(const int32_t* drafts, const int32_t* verify, int k) {
    int n = 0;
    while (n < k && drafts[n] == verify[n]) ++n;
    return n;
}

/// The MTP block's expert pool: `n` slots, least recently used out.  Host bookkeeping only (the bytes are the
/// caller's), so it is unit-testable without a GPU.
class ExpertLru {
public:
    explicit ExpertLru(int slots = 0, int n_experts = 512) { reset(slots, n_experts); }
    void reset(int slots, int n_experts);
    int slots() const { return n_slots_; }
    /// Places every expert of `want` (distinct ids) in a slot, the ones already resident first, and returns the
    /// slot of each in `slot`.  `load` lists the (slot, expert) pairs whose bytes the caller must read.  Fails when
    /// `want` holds more experts than there are slots.
    bool place(const std::vector<int32_t>& want, std::vector<int>& slot, std::vector<std::pair<int, int32_t>>& load);
    /// An EMPTY pool's starting contents: ranked[i] (most used first) into slot i, for as many as there are slots, and
    /// the (slot, expert) pairs to read in `load`.  The stamps put rank 0 last in line for eviction.  Neither hits nor
    /// misses count it.  Returns false (and changes nothing) when the pool is not empty or an id is out of range.
    bool preload(const std::vector<int32_t>& ranked, std::vector<std::pair<int, int32_t>>& load);
    uint64_t hits = 0, misses = 0;

private:
    int n_slots_ = 0;
    uint64_t clock_ = 0;
    std::vector<int> slot_of_;       // expert -> slot or -1
    std::vector<int32_t> expert_;    // slot -> expert or -1
    std::vector<uint64_t> used_;     // slot -> last use
};

struct MtpStats {
    uint64_t rounds = 0, offered = 0, accepted = 0;   // verify windows, drafts proposed, drafts accepted
    uint64_t full_rows = 0, cache_rows = 0;           // draft steps; rows run cache-only (catch-up)
    uint64_t expert_hits = 0, expert_misses = 0;
    double expert_bytes = 0;
    double draft_ms = 0, verify_ms = 0, catchup_ms = 0, expert_read_ms = 0, commit_ms = 0;
    // the expert pool by half of the generation (speculative_greedy sets the half from the tokens made so far):
    // what separates the pool's cold start from its steady state
    uint64_t half_hits[2] = {0, 0}, half_misses[2] = {0, 0};
    int half = 0;
    std::vector<uint64_t> accept_hist;                // [n accepted] -> rounds
};

class Mtp {
public:
    Mtp() = default;
    ~Mtp();
    Mtp(const Mtp&) = delete;
    Mtp& operator=(const Mtp&) = delete;

    /// `lib` is the trunk's compiled hy4-flash library (Forward::library).  `chunk` = rows per cache-only command buffer
    /// (the prompt's catch-up).  The MTP's dense tensors must be resident (open(..., with_mtp = true)).
    std::string open(Model& m, mtl::Context& cx, int lib, int chunk = 1024);
    /// Forgets the rows and the pooled keys; hidden(0) = 0.
    void reset();
    /// hidden(q) = h_{q-1}: `n` rows of the trunk's result_norm for positions q-1 .. q+n-2.
    void set_hidden(int q, int n, const float* h);
    float* hidden(int q);
    /// Rows [len(), q_end) cache-only.  `tokens` is the sequence from position 0.
    std::string catch_up(const int32_t* tokens, int q_end);
    /// Rows [len(), q] with row q full: its draft into `*draft`, its state s into state().  len() = q + 1 after.
    std::string step(const int32_t* tokens, int q, int32_t* draft);
    const float* state() const;     ///< [n_embd] s of the last full row
    int len() const { return len_; }
    void truncate(int n);
    MtpStats& stats() { return st_; }
    uint64_t expert_pool_bytes() const;

    struct Impl;

private:
    Impl* im_ = nullptr;
    int len_ = 0;
    MtpStats st_;
};

/// What `speculative_greedy` is told to do wrong ON PURPOSE (HY4_FLASH_MTP_PROBE), so the lossless check has cases it
/// must react to (a check never seen to fail is not evidence):
///   draft_random  every draft is replaced by a pseudo-random id: the output must NOT change, the acceptance ~0
///   accept_all    the verify is skipped and every draft accepted: the output MUST change
enum class MtpProbe { none, draft_random, accept_all };
MtpProbe mtp_probe_from_env();

/// Greedy decoding with k drafts a round.  On entry the trunk has run positions [0, seq.size()) and
/// `fw.last_token()` is the next token; the MTP has caught up to row seq.size() - 1 with hidden() set to match.
/// Emits `gen` tokens (fewer at `context`) through `emit`, appending them to `seq`.  k = 0 is plain decoding.
std::string speculative_greedy(Forward& fw, Mtp& mtp, std::vector<int32_t>& seq, int gen, int k, int context,
                               MtpProbe probe, const std::function<void(int32_t)>& emit);

}  // namespace strata::hy4_flash
