// include/strata/core/prompt_cache.hpp - the --serve prompt cache.
//
// A request continues from the longest prefix of its prompt that the engine still holds, instead of processing
// the whole conversation again.  Three things hold prefixes:
//
//   * the LIVE sequence: the tokens whose cells the session state holds right now;
//   * CHECKPOINTS of the live sequence (session_ckpt_*): the fixed-size state at one position.  The K/V cells
//     are indexed by position, so restoring a checkpoint at `pos` and overwriting the cells from `pos` on
//     continues the sequence from there;
//   * STASHED sequences: a sequence the live one replaced, kept in host memory with its K/V cells and its
//     checkpoints, so a client that alternates between conversations (a chat and its title request, an agent
//     and its subagents) does not start over on every switch.
//
// Checkpoint buffers are a shared pool of `max_ckpts`; the least recently used one goes when it is full.  The
// stash's K/V copies share a byte budget in the same way.
#pragma once

#include "strata/core/session.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class PromptCache {
public:
    /// `kv`: every state whose cells a stashed sequence keeps (the QSA layers and the draft layer).
    PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream, int max_ckpts,
                uint64_t stash_bytes);
    ~PromptCache();
    PromptCache(const PromptCache&) = delete;
    PromptCache& operator=(const PromptCache&) = delete;

    bool enabled() const { return max_ckpts_ > 0; }
    uint64_t ckpt_bytes() const { return ckpt_bytes_; }

    /// Sets the session up for a prompt of `ids`: continues from the longest prefix (at most ids.size() - 1
    /// cells, the last token always goes through a verify window) that the live sequence, a checkpoint or a
    /// stashed sequence holds, or zeroes the session.  With `cacheable` false (an image request, whose cells the
    /// ids do not describe) it zeroes the session and the live sequence is not kept.  Returns the cells reused.
    int64_t begin(const std::vector<int64_t>& ids, bool cacheable);
    /// The cells [pos, pos + n) now hold `tokens`; pos <= live length, and anything after pos is gone.
    void set(int64_t pos, const int64_t* tokens, int64_t n);
    void set(int64_t pos, const int32_t* tokens, int64_t n);
    /// Checkpoints the session, which is at `pos` == the live length.
    void checkpoint(int64_t pos);
    /// Forgets the live sequence (after an image request).
    void forget();

    /// One line for the log: checkpoints and stashed sequences.
    std::string summary() const;

private:
    struct Ckpt { int64_t pos; void* host; uint64_t stamp; };
    struct Seq {
        std::vector<int64_t> tokens;
        std::vector<Ckpt> ckpts;
        std::unique_ptr<uint8_t[]> kv;   // cells [0, kv_cells) of every K/V state (stashed sequences only)
        int64_t kv_cells = 0;
        uint64_t kv_bytes = 0;
        uint64_t stamp = 0;
    };

    /// One device region of the K/V cells, stored back to back in a stashed sequence's buffer.
    struct Part { void* dev; uint64_t bytes; };

    /// Cells of `seq` a prompt can continue from: the whole sequence when it is a prefix of the prompt (live
    /// only: the state is there), else its best checkpoint within the common prefix (its position; -1: none).
    int64_t reusable(const Seq& seq, const std::vector<int64_t>& ids, bool live) const;
    static Ckpt* find(Seq& seq, int64_t pos);
    void* take_buffer();
    void release(Seq& seq, int64_t keep_upto);   // frees the checkpoints after `keep_upto`
    /// Moves a copy of the live sequence to the stash: its K/V cells, a checkpoint of its end, and its
    /// checkpoints after `keep_upto` (the ones before stay with the live sequence, which continues from there).
    void stash_live(int64_t keep_upto);
    std::vector<Part> kv_parts(int64_t cells) const;
    bool kv_copy(const std::vector<Part>& parts, uint8_t* host, bool to_host);
    bool restore(Seq& seq, int64_t pos);   // K/V cells (stashed only) and the checkpoint at `pos`

    const ModelGeometry& g_;
    SessionState& ss_;
    std::vector<QsaState*> kv_;
    void* stream_;
    int max_ckpts_;
    uint64_t stash_budget_;
    uint64_t ckpt_bytes_;
    int allocated_ = 0;
    uint64_t clock_ = 0;
    Seq live_;
    std::vector<Seq> stash_;
    std::vector<void*> free_;
    std::vector<std::pair<void*, bool>> owned_;   // every buffer, and whether it is pinned
};

}  // namespace strata::core
