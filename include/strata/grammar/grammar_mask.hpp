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
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata {
namespace grammar {

inline int grammar_mask_words(int64_t n_vocab) {
    return (int)((n_vocab + 31) / 32);
}

// The vocabulary prepared once for mask building: each token's piece decoded to code points, and the
// candidate list the reject fold walks.  Building this per mask meant decoding every one of the
// model's ~150k pieces, and allocating a vector for each, every time a grammar state the cache had not
// seen came up - and a tool-call JSON region produces a new state per token.  XGrammar's matcher
// pre-decodes its vocabulary for the same reason (cpp/grammar_matcher.cc, prebuilt trie); llama.cpp's
// apply_impl decodes on each call, which is where its per-step cost comes from.
class MaskVocab {
public:
    explicit MaskVocab(const GrammarVocab& vocab);

    // Not copyable: cands_[i].code_points points into storage_[i], so a copy would carry pointers into
    // the source and dangle once it died.  Moving is safe (a moved vector<vector<>> keeps its inner
    // buffers), and every instance lives in a shared_ptr, so nothing needs to copy one.
    MaskVocab(const MaskVocab&) = delete;
    MaskVocab& operator=(const MaskVocab&) = delete;
    MaskVocab(MaskVocab&&) = default;
    MaskVocab& operator=(MaskVocab&&) = default;

    // tokens the grammar may allow: every piece decoded and no control/EOG/empty/NUL token
    const std::vector<grammar_candidate>& candidates() const { return cands_; }

    // words in a mask row for this vocabulary: grammar_mask_words(pieces.size())
    int words() const { return words_; }

    // every bit banned, as grammar_build_mask starts (the caller clears what a stack accepts)
    void ban_all(uint32_t* words) const;

    // the EOG bits, cleared when a stack is at the end of the grammar
    const std::vector<int32_t>& eog() const { return eog_; }

    // The candidates a grammar halfway through a multi-byte character can still allow, as
    // (candidate index, raw piece).  A piece that starts with anything but a UTF-8 continuation
    // byte (0x80..0xBF) begins a new character, which decode_utf8 rejects with n_remain = -1 and
    // every stack then refuses, so the rest of the vocabulary cannot ride in that state.  Keeping
    // them apart means such a mask decodes a few hundred pieces rather than all ~150k.
    const std::vector<std::pair<size_t, std::string>>& continuation() const { return continuation_; }

private:
    int words_ = 0;
    std::vector<std::vector<uint32_t>> storage_;   // per token: code points, 0-terminated
    std::vector<grammar_candidate>     cands_;
    std::vector<int32_t>               eog_;
    std::vector<std::pair<size_t, std::string>> continuation_; // (candidate index, piece) that may ride mid-character
};

// fills words[0 .. grammar_mask_words(vocab.pieces.size())) for the given stacks
void grammar_build_mask(const GrammarVocab& vocab, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, uint32_t* words);

// the same, reusing a prepared vocabulary (no per-call decode or allocation)
void grammar_build_mask(const MaskVocab& prepped, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, uint32_t* words);

// The two above assume a character boundary.  A grammar that is halfway through a multi-byte
// character is a different state: only the pieces that finish it may ride, so each candidate is
// decoded from `partial` (llama.cpp's apply_impl: decode_utf8(piece, grammar.partial_utf8)).
// The prepared code points cannot serve here - they were decoded from a boundary.
void grammar_build_mask(const MaskVocab& prepped, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, partial_utf8 partial, uint32_t* words);

class GrammarMaskCache {
public:
    GrammarMaskCache(const GrammarVocab& vocab, const grammar_rules& rules, size_t cap = 4096);

    // `pre` (optional) is a vocabulary prepared by the caller and shared across grammars; without it
    // this cache prepares its own once, which is still once per cache rather than once per miss.
    GrammarMaskCache(const GrammarVocab& vocab, const grammar_rules& rules, size_t cap,
                     std::shared_ptr<const MaskVocab> pre);

    // the mask for this stack state; shared_ptr-sized rows, LRU-bounded (cap entries)
    const std::vector<uint32_t>& mask_for(const std::vector<grammar_stack>& stacks);

    // The stacks' elements are POINTERS INTO THE GRAMMAR'S OWN rules vector. Grammars that share one
    // immutable rules vector (grammar_clone now does) resolve those pointers directly; the constructor
    // builds the pointer -> (rule, offset) index once, so a lookup no longer walks every rule and
    // element. `rules` still names the owner, and a pointer the index does not know (a foreign
    // grammar's rules) falls back to the old range scan.
    const std::vector<uint32_t>& mask_for(const std::vector<grammar_stack>& stacks, const grammar_rules& rules);

    // the same, for a grammar halfway through a multi-byte character.  `partial` is a part of the
    // state, not a caller's hint: the row for (stacks, mid-character) differs from the row for the
    // same stacks on a boundary, so it is part of the cache key.
    const std::vector<uint32_t>& mask_for(const std::vector<grammar_stack>& stacks, const grammar_rules& rules,
                                          partial_utf8 partial);

    size_t size() const { return index_.size(); }

private:
    using key_t = std::vector<std::pair<uint32_t, uint32_t>>; // the partial state, then per stack: a boundary
                                                              // and (rule, offset) of every element

    key_t make_key(const std::vector<grammar_stack>& stacks, const grammar_rules& rules,
                   partial_utf8 partial) const;

    // the (rule, offset) of one stack element, by hash for an indexed pointer and by a scan of
    // `owner` for a foreign one
    std::pair<uint32_t, uint32_t> locate_element(const grammar_element* e, const grammar_rules& owner) const;

    void index_rules();

    const grammar_rules& rules_;
    int words_;
    size_t cap_;
    std::shared_ptr<const MaskVocab> prepped_;
    std::unordered_map<const grammar_element*, std::pair<uint32_t, uint32_t>> where_;
    std::list<std::pair<key_t, std::shared_ptr<std::vector<uint32_t>>>> lru_;
    std::map<key_t, std::list<std::pair<key_t, std::shared_ptr<std::vector<uint32_t>>>>::iterator> index_;
};

// Process-wide count of the rows a GrammarMaskCache answered from its LRU (a hit, one key compare)
// and the rows it had to build (a miss, one reject fold over the whole vocabulary).  A miss is the
// expensive one, so this pair is what tells a caller whether the per-window row cost is fold time or
// key time.  Counters only: nothing about a mask depends on them.  They are free functions rather
// than members on purpose: two size_t members change sizeof(GrammarMaskCache), and that layout change
// alone makes grammar_test's Debug /RTC check report a stack corruption around `cache`.
void grammar_mask_cache_counters(size_t* hits, size_t* misses);

} // namespace grammar
} // namespace strata
