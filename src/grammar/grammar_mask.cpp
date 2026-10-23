// src/grammar/grammar_mask.cpp - the bitmask builder and its per-state cache.
#include "strata/grammar/grammar_mask.hpp"

#include <cstring>
#include <stdexcept>

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

GrammarMaskCache::key_t GrammarMaskCache::make_key(const std::vector<grammar_stack>& stacks) const {
    key_t key;
    key.reserve(stacks.size() + 1);
    key.emplace_back(0xFFFFFFFFu, (uint32_t)stacks.size()); // (rules identity is the cache's own)
    for (const grammar_stack& stack : stacks) {
        if (stack.empty()) {
            key.emplace_back(0, 0xFFFFFFFFu); // the terminal marker
            continue;
        }
        const grammar_element* top = stack.back();
        uint32_t rule = 0xFFFFFFFFu, offset = 0xFFFFFFFFu;
        for (size_t r = 0; r < rules_.size(); ++r) {
            const grammar_rule& rule_vec = rules_[r];
            if (!rule_vec.empty() && top >= rule_vec.data() && top < rule_vec.data() + rule_vec.size()) {
                rule = (uint32_t)r;
                offset = (uint32_t)(top - rule_vec.data());
                break;
            }
        }
        key.emplace_back(rule, offset);
    }
    return key;
}

const std::vector<uint32_t>& GrammarMaskCache::mask_for(const std::vector<grammar_stack>& stacks) {
    const key_t key = make_key(stacks);
    auto it = index_.find(key);
    if (it != index_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second); // hit: mark most-recent
        return *it->second->second;
    }
    auto row = std::make_shared<std::vector<uint32_t>>((size_t)words_);
    grammar_build_mask(vocab_, rules_, stacks, row->data());
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
