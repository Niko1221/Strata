// Strata. Where a batched sequence's KV lives (M2 of docs/DSV41_BATCHING.md).
//
// The single-sequence engine keeps one ring per layer sized by --max-context. A batch keeps one block table per
// sequence instead: the KV arena is cut into blocks of block_tokens, a sequence owns a list of block indices, and
// position pos of sequence seq lives in table(seq)[pos / block_tokens] at offset pos % block_tokens.
// Nothing here touches CUDA: it is the index arithmetic a varlen attention kernel will use, kept testable.
//
// When the arena is full the least recently used sequence is evicted (its blocks return to the free list) - the
// same order the conversation slots and the expert cache use.
#pragma once

#include <cstdint>
#include <vector>

namespace strata {
namespace ds4 {

struct BatchKv {
    int32_t block_tokens = 256;
    int32_t blocks = 0;                 // blocks in the arena
    int32_t max_seqs = 0;               // sequences the batch admits
};

/// A sequence's block table inside the arena.
class KvSeq {
public:
    KvSeq() = default;
    KvSeq(int32_t seq, int32_t block_tokens) : seq_(seq), block_tokens_(block_tokens) {}

    int32_t seq() const { return seq_; }
    int32_t tokens() const { return tokens_; }
    size_t blocks_held() const { return table_.size(); }

    /// The block a token at pos goes to, or -1 when the table does not cover it yet.
    int32_t block_for(int64_t pos) const {
        const size_t b = static_cast<size_t>(pos / block_tokens_);
        return b < table_.size() ? table_[b] : -1;
    }

    /// Ensure a block exists for pos; returns it, or -1 when the arena has none left.
    int32_t grow(int64_t pos, std::vector<int32_t>& free_blocks) {
        const size_t want = static_cast<size_t>(pos / block_tokens_) + 1;
        while (table_.size() < want) {
            if (free_blocks.empty()) return -1;
            table_.push_back(free_blocks.back());
            free_blocks.pop_back();
        }
        return table_[want - 1];
    }

    /// Position pos -> the flat token slot in the arena (-1: not covered).
    int64_t slot(int64_t pos) const {
        const int32_t b = block_for(pos);
        if (b < 0) return -1;
        return static_cast<int64_t>(b) * block_tokens_ + (pos % block_tokens_);
    }

    const std::vector<int32_t>& blocks() const { return table_; }

    void advance(int32_t n) { tokens_ += n; }

    /// Return the blocks to the free list.
    void release(std::vector<int32_t>& free_blocks) {
        for (int32_t b : table_) free_blocks.push_back(b);
        table_.clear();
        tokens_ = 0;
    }

private:
    int32_t seq_ = 0;
    int32_t block_tokens_ = 256;
    int32_t tokens_ = 0;
    std::vector<int32_t> table_;
};

/// The batch's sequences with an LRU order (front is the oldest).
class KvBatch {
public:
    explicit KvBatch(BatchKv cfg) : cfg_(cfg) {
        free_.reserve(static_cast<size_t>(cfg.blocks));
        for (int32_t b = cfg.blocks - 1; b >= 0; --b) free_.push_back(b);
    }

    const BatchKv& cfg() const { return cfg_; }
    size_t active() const { return seqs_.size(); }
    int32_t free_blocks() const { return static_cast<int32_t>(free_.size()); }

    const KvSeq* find(int32_t seq) const {
        for (const auto& s : seqs_)
            if (s.seq() == seq) return &s;
        return nullptr;
    }

    /// Add a sequence (its table grows on demand); false when the batch is full.
    bool open(int32_t seq) {
        if (find(seq)) return true;
        if (static_cast<int32_t>(seqs_.size()) >= cfg_.max_seqs) return false;
        seqs_.emplace_back(seq, cfg_.block_tokens);
        return true;
    }

    /// Make room for pos of seq, evicting the oldest other sequence when the arena is full.
    /// Returns the token slot, and the evicted sequence id in *evicted (-1 when none was evicted).
    int64_t touch(int32_t seq, int64_t pos, int32_t* evicted) {
        *evicted = -1;
        KvSeq* s = nullptr;
        for (auto& x : seqs_)
            if (x.seq() == seq) { s = &x; break; }
        if (s == nullptr) return -1;
        int32_t b = s->grow(pos, free_);
        while (b < 0) {
            size_t victim = seqs_.size();
            for (size_t i = 0; i < seqs_.size(); ++i)
                if (seqs_[i].seq() != seq) { victim = i; break; }
            if (victim == seqs_.size()) return -1;
            *evicted = seqs_[victim].seq();
            seqs_[victim].release(free_);
            seqs_.erase(seqs_.begin() + static_cast<long>(victim));
            for (auto& x : seqs_)
                if (x.seq() == seq) { b = x.grow(pos, free_); break; }
            if (find(seq) == nullptr) return -1;
        }
        return static_cast<int64_t>(b) * cfg_.block_tokens + (pos % cfg_.block_tokens);
    }

    /// Move seq to the back of the LRU order.
    void touch_order(int32_t seq) {
        for (size_t i = 0; i < seqs_.size(); ++i)
            if (seqs_[i].seq() == seq) {
                KvSeq s = seqs_[i];
                seqs_.erase(seqs_.begin() + static_cast<long>(i));
                seqs_.push_back(s);
                return;
            }
    }

    void advance(int32_t seq, int32_t n) {
        for (auto& s : seqs_)
            if (s.seq() == seq) { s.advance(n); return; }
    }

    void close(int32_t seq) {
        for (size_t i = 0; i < seqs_.size(); ++i)
            if (seqs_[i].seq() == seq) {
                seqs_[i].release(free_);
                seqs_.erase(seqs_.begin() + static_cast<long>(i));
                return;
            }
    }

    const std::vector<KvSeq>& sequences() const { return seqs_; }

private:
    BatchKv cfg_;
    std::vector<int32_t> free_;
    std::vector<KvSeq> seqs_;        // LRU order: front is the oldest
};

}  // namespace ds4
}  // namespace strata
