// include/strata/spec/suffix_drafter.hpp - plan v0.3 P6: the suffix-lookup drafter (no weights, no GPU).
//
// Proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix: when the
// output quotes its input (edits, refactors, repeated code) the continuation is usually exact, and a verify pass
// accepts many tokens at once. It needs no model and costs microseconds.
//
// Index: an open-addressing table maps every trigram of the history (prompt + accepted output) to its most recent
// end position, and each position links to its trigram's previous one, so appends are O(1) (the table is rebuilt,
// larger if the history needs it, when half full). A proposal checks the trigram's WAYS most recent earlier
// occurrences, extends each match backwards up to `max_match`, and takes the longest (most recent on ties).
// Matches shorter than `min_match` propose nothing.
//
// `assign` replaces the history by another text that may share a prefix with it (a conversation's next request):
// the positions past the common prefix are undone newest first, which leaves the index as appending the prefix
// alone would, and the rest is appended. Proposals therefore depend on the history alone.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

class SuffixDrafter {
public:
    static constexpr int WAYS = 16;   // earlier occurrences a proposal checks (in code a trigram recurs often)

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19);

    void reset();
    /// Add tokens to the history (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }
    /// Make the history `tokens` (see above).
    void assign(const int32_t* tokens, size_t n);

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);

    /// Length of the match behind the last proposal (0 if none).
    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }

private:
    struct Slot {
        uint64_t key = 0;             // trigram hash (0 = empty)
        int32_t last = -1;            // the trigram's most recent end position (-1: none left after `assign`)
    };
    Slot* find_slot(uint64_t key, bool insert);
    uint64_t key_at(size_t end) const;
    void link(size_t end);            // index hist_[end], the newest token
    void rebuild();

    int min_match_, max_match_;
    std::vector<int32_t> hist_;
    std::vector<int32_t> prev_;       // per end position: its trigram's previous end position (-1: none)
    std::vector<Slot> table_;
    size_t mask_ = 0;
    size_t used_ = 0;                 // occupied slots, those of undone positions included
    int last_match_ = 0;
};

}  // namespace strata::spec
