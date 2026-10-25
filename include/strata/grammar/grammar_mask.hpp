// include/strata/grammar/grammar_mask.hpp - the per-step bitmask for a grammar's stack state.
//
// One uint32 word per 32 tokens; bit = 1 BANS the token (the sampler sets -inf at the logit_bias
// site). This is llama.cpp's llguidance mask convention (common/llguidance.cpp:62-65), and the
// accept semantics match apply_impl: a token survives when at least one stack accepts its whole
// piece; EOG rides only at the terminal state (any stack empty); Strata additionally bans every
// CONTROL token (spec §5.1, recorded divergence), and padding bits stay banned.
//
// The vendored core has no mask cache; this is the design's per-state cache (spec §5.1). The key is
// the WHOLE stack state: (rule, offset) of every element of every live stack (a top-only key collides
// across states that share their top and hands one of them the other's mask).
#pragma once

#include "strata/grammar/grammar.hpp"

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace strata {
namespace grammar {

inline int grammar_mask_words(int64_t n_vocab) {
    return (int)((n_vocab + 31) / 32);
}

// fills words[0 .. grammar_mask_words(vocab.pieces.size())) for the given stacks
void grammar_build_mask(const GrammarVocab& vocab, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, uint32_t* words);

class GrammarMaskCache {
public:
    GrammarMaskCache(const GrammarVocab& vocab, const grammar_rules& rules, size_t cap = 4096);

    // the mask for this stack state; shared_ptr-sized rows, LRU-bounded (cap entries)
    const std::vector<uint32_t>& mask_for(const std::vector<grammar_stack>& stacks);

    // The stacks' elements are POINTERS INTO THE GRAMMAR'S OWN rules vector, and `grammar_clone` copies that
    // vector, so a clone's stacks point into the CLONE, not the proto. The key is (rule, offset) of every
    // stack's top element, located by range-scan over `rules`: pass the rules that OWN the stacks
    // (grammar_get_rules of the grammar the stacks came from). With the proto's rules every clone pointer
    // misses the range scan, all elements key to (NONE, NONE), and every state collapses to one key.
    const std::vector<uint32_t>& mask_for(const std::vector<grammar_stack>& stacks, const grammar_rules& rules);

    size_t size() const { return index_.size(); }

private:
    using key_t = std::vector<std::pair<uint32_t, uint32_t>>; // per stack: a boundary, then (rule, offset) of every element

    key_t make_key(const std::vector<grammar_stack>& stacks, const grammar_rules& rules) const;

    const GrammarVocab& vocab_;
    const grammar_rules& rules_;
    int words_;
    size_t cap_;
    std::list<std::pair<key_t, std::shared_ptr<std::vector<uint32_t>>>> lru_;
    std::map<key_t, std::list<std::pair<key_t, std::shared_ptr<std::vector<uint32_t>>>>::iterator> index_;
};

} // namespace grammar
} // namespace strata
