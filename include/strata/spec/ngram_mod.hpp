// include/strata/spec/ngram_mod.hpp - the ngram-mod drafter's table: hash(n tokens) -> the token that followed.
//
// A CPU-only companion to the suffix drafter (suffix_drafter.hpp) with the opposite trade. Prompt lookup searches the
// history at draft time for the longest repeat of the context; ngram-mod has already compressed every observation
// into one hash cell while the tokens were committed, so drafting is O(n_match) multiplies with no search at all -
// and a cell holds ONE continuation, so two different continuations of the same n-gram overwrite each other and the
// last one wins. That lossy map is the point (llama.cpp's common_ngram_mod, PR #19164): bounded memory, O(1) adds,
// deterministic collisions, and drafting that costs nothing next to a verify window. Everything it proposes goes
// through the verify window like any other draft, so it can only change the speed, never the output.
//
// The hash is llama.cpp's, bit for bit: an LCG over the key's token ids,
//     hash = hash * 6364136223846793005ULL + token   (for each of the n_match key tokens)
// then `hash % entries`. Collisions overwrite. EMPTY (-1) marks a never-written cell; Strata token ids are
// non-negative, so a real token never reads as empty. The default table is 4M entries of int32 (16 MiB) whatever the
// context length - the suffix drafter's ~20 bytes per token grows with the conversation, this does not.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

class NgramMod {
public:
    using entry_t = int32_t;
    static constexpr entry_t EMPTY = -1;                 ///< a cell never written (token ids are >= 0)
    static constexpr size_t kDefaultEntries = 4u * 1024 * 1024;   ///< 16 MiB, llama.cpp's table size

    /// `n`: the key's length in tokens (n_match; 1 hashes single tokens). `entries`: fixed cell count.
    explicit NgramMod(int n, size_t entries = kDefaultEntries);

    /// The cell an n-gram key lands in (the LCG above, mod the cell count). Public for tests.
    size_t index(const entry_t* tokens) const;
    /// Store `tokens[n]` as the continuation of the key `tokens[0..n)`. Overwrites on collisions.
    void add(const entry_t* tokens);
    /// The stored continuation of `tokens[0..n)`, or EMPTY.
    entry_t get(const entry_t* tokens) const;

    /// Empty every cell (occupancy protection, low-acceptance protection, --ngram-mod off in tests).
    void reset();

    int n() const { return n_; }
    size_t used() const { return used_; }                ///< cells holding a token
    size_t entries() const { return entries_.size(); }
    size_t bytes() const { return entries_.size() * sizeof(entry_t); }
    double occupancy() const { return (double) used_ / (double) entries_.size(); }

private:
    int n_;                                              ///< key length (n_match)
    size_t used_ = 0;
    std::vector<entry_t> entries_;
};

}  // namespace strata::spec
