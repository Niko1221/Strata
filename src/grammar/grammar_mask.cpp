// src/grammar/grammar_mask.cpp - the bitmask builder and its per-state cache.
#include "strata/grammar/grammar_mask.hpp"

#include <algorithm>

namespace strata {
namespace grammar {

// The mask for one stack state, over a vocabulary prepared once.  Same semantics as llama.cpp's
// apply_impl (a token survives when at least one stack accepts its whole piece; EOG only at a
// terminal state; Strata additionally bans every control token).
static void grammar_build_mask_impl(const MaskVocab& prepped, const grammar_rules& rules,
                                    const std::vector<grammar_stack>& stacks, uint32_t* words) {
    // every bit starts banned; a token clears its bit when a stack accepts the whole piece
    prepped.ban_all(words);

    // apply_impl's allow_eog: a stack is at the end of the grammar (empty)
    const bool allow_eog = std::any_of(stacks.begin(), stacks.end(),
                                       [](const grammar_stack& stack) { return stack.empty(); });
    if (allow_eog) {
        for (const int32_t id : prepped.eog()) {
            words[id >> 5] &= ~(1u << (id & 31));
        }
    }

    const grammar_candidates& cands = prepped.candidates();
    // Order matters, and it is the original's: every candidate bit is cleared BEFORE the early
    // return, so an empty stack list leaves all candidates allowed (the reject fold is the only
    // thing that narrows them, and it is skipped).  Moving this below the return looked harmless
    // and banned the whole vocabulary instead - the same shape of mistake as a mask that is
    // "conservative" and then stalls generation.
    for (const grammar_candidate& c : cands) {
        words[c.id >> 5] &= ~(1u << (c.id & 31)); // allowed until proven rejected
    }
    if (stacks.empty() || cands.empty()) {
        return;
    }
    const grammar_candidates rejected = grammar_reject_candidates(rules, stacks, cands);
    for (const grammar_candidate& c : rejected) {
        words[c.id >> 5] |= (1u << (c.id & 31));
    }
}

MaskVocab::MaskVocab(const GrammarVocab& vocab) : words_(grammar_mask_words((int64_t)vocab.pieces.size())) {
    const int32_t n_vocab = (int32_t)vocab.pieces.size();
    storage_.resize((size_t)n_vocab);
    cands_.reserve((size_t)n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        if (vocab.is_eog(id)) {
            eog_.push_back(id);
            continue;
        }
        if ((size_t)id < vocab.control.size() && vocab.control[(size_t)id]) {
            continue; // Strata bans every control token (spec §5.1)
        }
        const std::string piece = vocab.token_to_piece(id);
        if (piece.empty() || piece.find('\0') != std::string::npos) {
            // apply_impl bans an empty piece and one whose first byte is NUL; any embedded NUL joins
            // it here: the accept walk stops at a NUL (the code_points terminator), so such a token
            // would "match" every state as an empty piece and leak its byte into the output - the
            // byte-0x00 token exactly (its piece IS a NUL once the GPT2 byte map is fully decoded).
            continue;
        }
        auto decoded = decode_utf8(piece, partial_utf8{0, 0});
        storage_[(size_t)id] = std::move(decoded.first);
        grammar_candidate c;
        c.index = cands_.size();
        c.code_points = storage_[(size_t)id].data();
        c.partial_utf8 = decoded.second;
        c.id = id;
        cands_.push_back(c);
    }
}

void MaskVocab::ban_all(uint32_t* words) const {
    std::fill_n(words, (size_t)words_, 0xFFFFFFFFu);
}

void grammar_build_mask(const MaskVocab& prepped, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, uint32_t* words) {
    grammar_build_mask_impl(prepped, rules, stacks, words);
}

void grammar_build_mask(const GrammarVocab& vocab, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, uint32_t* words) {
    const MaskVocab prepped(vocab);
    grammar_build_mask_impl(prepped, rules, stacks, words);
}

GrammarMaskCache::GrammarMaskCache(const GrammarVocab& vocab, const grammar_rules& rules, size_t cap)
    : GrammarMaskCache(vocab, rules, cap, nullptr) {}

GrammarMaskCache::GrammarMaskCache(const GrammarVocab& vocab, const grammar_rules& rules, size_t cap,
                                   std::shared_ptr<const MaskVocab> pre)
    // words_ is declared before prepped_, so `pre` is still intact when the row width is taken from it.
    // The width must come from the prepared vocabulary: grammar_build_mask bans and clears bits by the
    // ids in it, so a row sized from a different (larger) vocab would be written past its end.
    : rules_(rules), words_(pre ? pre->words() : grammar_mask_words((int64_t)vocab.pieces.size())), cap_(cap),
      prepped_(pre ? std::move(pre) : std::make_shared<const MaskVocab>(vocab)) {
    index_rules();
}

void GrammarMaskCache::index_rules() {
    // Built once: the (rule, offset) of every element.  Lookups then cost one hash per stack element
    // instead of a walk over the whole rules vector, which ran even on a cache hit.
    size_t count = 0;
    for (const auto& rule : rules_) count += rule.size();
    where_.reserve(count);
    for (size_t r = 0; r < rules_.size(); ++r)
        for (size_t o = 0; o < rules_[r].size(); ++o)
            where_.emplace(rules_[r].data() + o, std::make_pair((uint32_t) r, (uint32_t) o));
}

std::pair<uint32_t, uint32_t> GrammarMaskCache::locate_element(const grammar_element* e,
                                                               const grammar_rules& owner) const {
    // one hash probe for a pointer this cache indexed (every clone of the grammar it was built for)
    const auto it = where_.find(e);
    if (it != where_.end()) {
        return it->second;
    }
    // a foreign pointer: scan the owner's own rules, and key it (NONE, NONE) if even that misses, so
    // an unknown element never aliases a state the cache does know.
    for (size_t r = 0; r < owner.size(); ++r) {
        for (size_t o = 0; o < owner[r].size(); ++o) {
            if (owner[r].data() + o == e) {
                return std::make_pair((uint32_t) r, (uint32_t) o);
            }
        }
    }
    return std::make_pair(0xFFFFFFFFu, 0xFFFFFFFFu);
}

GrammarMaskCache::key_t GrammarMaskCache::make_key(const std::vector<grammar_stack>& stacks,
                                                   const grammar_rules& rules) const {
    // the WHOLE stack is the state.  A top-only key collides: a string element inside the steps array
    // and one inside an object property share their top, and one of them then sampled through the
    // other's mask (measured on the rig: the walk then rejected what the mask had allowed -
    // "Unexpected empty grammar stack" - or accepted a turn the state never had, which surfaced as a
    // finished reply that failed its own schema).
    key_t key;
    key.reserve(stacks.size() * 8 + 1);
    key.emplace_back(0xFFFFFFFFu, (uint32_t)stacks.size()); // (rules identity is the cache's own)
    for (const grammar_stack& stack : stacks) {
        key.emplace_back(0xFFFFFFFEu, (uint32_t)stack.size()); // the stack boundary; depth 0 = terminal
        for (const grammar_element* e : stack) {
            const std::pair<uint32_t, uint32_t> pos = locate_element(e, rules);
            key.emplace_back(pos.first, pos.second);
        }
    }
    return key;
}

const std::vector<uint32_t>& GrammarMaskCache::mask_for(const std::vector<grammar_stack>& stacks) {
    return mask_for(stacks, rules_);
}

const std::vector<uint32_t>& GrammarMaskCache::mask_for(const std::vector<grammar_stack>& stacks,
                                                         const grammar_rules& rules) {
    const key_t key = make_key(stacks, rules);
    auto it = index_.find(key);
    if (it != index_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second); // hit: mark most-recent
        return *it->second->second;
    }
    auto row = std::make_shared<std::vector<uint32_t>>((size_t)words_);
    grammar_build_mask(*prepped_, rules, stacks, row->data());
    lru_.emplace_front(key, row);
    index_[key] = lru_.begin();
    while (lru_.size() > cap_) {
        index_.erase(lru_.back().first);
        lru_.pop_back();
    }
    return *row;
}

} // namespace grammar
} // namespace strata
