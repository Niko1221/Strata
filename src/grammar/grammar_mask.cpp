// src/grammar/grammar_mask.cpp - the bitmask builder and its per-state cache.
#include "strata/grammar/grammar_mask.hpp"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace strata {
namespace grammar {

void grammar_build_mask(const GrammarVocab& vocab, const grammar_rules& rules,
                        const std::vector<grammar_stack>& stacks, uint32_t* words) {
    const int32_t n_vocab = (int32_t)vocab.pieces.size();
    const int words_n = grammar_mask_words(n_vocab);

    // every bit starts banned; a token clears its bit when its stack accepts the whole piece
    for (int w = 0; w < words_n; ++w) {
        words[w] = 0xFFFFFFFFu;
    }

    // apply_impl's allow_eog: a stack is at the end of the grammar (empty)
    bool allow_eog = false;
    for (const grammar_stack& stack : stacks) {
        if (stack.empty()) {
            allow_eog = true;
        }
    }

    std::vector<std::vector<uint32_t>> storage((size_t)n_vocab);
    grammar_candidates cands;
    cands.reserve((size_t)n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        if (vocab.is_eog(id)) {
            if (allow_eog) {
                words[id >> 5] &= ~(1u << (id & 31));
            }
            continue; // CONTROL + non-terminal EOG stay banned
        }
        if (id >= 0 && (size_t)id < vocab.control.size() && vocab.control[(size_t)id]) {
            continue; // Strata bans every control token (spec §5.1)
        }
        const std::string piece = vocab.token_to_piece(id);
        if (piece.empty()) {
            continue; // apply_impl: an empty piece is rejected
        }
        auto decoded = decode_utf8(piece, partial_utf8{0, 0});
        storage[(size_t)id] = std::move(decoded.first);
        grammar_candidate c;
        c.index = cands.size();
        c.code_points = storage[(size_t)id].data();
        c.partial_utf8 = decoded.second;
        c.id = id;
        cands.push_back(c);
        words[id >> 5] &= ~(1u << (id & 31)); // allowed until proven rejected
    }

    if (stacks.empty() || cands.empty()) {
        return;
    }
    const grammar_candidates rejected = grammar_reject_candidates(rules, stacks, cands);
    for (const grammar_candidate& c : rejected) {
        words[c.id >> 5] |= (1u << (c.id & 31));
    }
}

GrammarMaskCache::GrammarMaskCache(const GrammarVocab& vocab, const grammar_rules& rules, size_t cap)
    : vocab_(vocab), rules_(rules), words_(grammar_mask_words((int64_t)vocab.pieces.size())), cap_(cap) {}

GrammarMaskCache::key_t GrammarMaskCache::make_key(const std::vector<grammar_stack>& stacks,
                                                   const grammar_rules& rules) const {
    // the WHOLE stack is the state.  A top-only key collides: a string element inside the steps array
    // and one inside an object property share their top, and one of them then sampled through the
    // other's mask (measured on the rig: the walk then rejected what the mask had allowed -
    // "Unexpected empty grammar stack" - or accepted a turn the state never had, which surfaced as a
    // finished reply that failed its own schema).
    std::unordered_map<const grammar_element*, std::pair<uint32_t, uint32_t>> where;
    for (size_t r = 0; r < rules.size(); ++r) {
        const grammar_rule& rule_vec = rules[r];
        for (size_t o = 0; o < rule_vec.size(); ++o)
            where.emplace(rule_vec.data() + o, std::make_pair((uint32_t) r, (uint32_t) o));
    }
    key_t key;
    key.reserve(stacks.size() * 8 + 1);
    key.emplace_back(0xFFFFFFFFu, (uint32_t)stacks.size()); // (rules identity is the cache's own)
    for (const grammar_stack& stack : stacks) {
        key.emplace_back(0xFFFFFFFEu, (uint32_t)stack.size()); // the stack boundary; depth 0 = terminal
        for (const grammar_element* e : stack) {
            const auto it = where.find(e);
            key.emplace_back(it != where.end() ? it->second.first : 0xFFFFFFFFu,
                             it != where.end() ? it->second.second : 0xFFFFFFFFu);
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
    grammar_build_mask(vocab_, rules, stacks, row->data());
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
