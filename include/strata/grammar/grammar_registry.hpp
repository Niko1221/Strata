// The engine's registered grammars, keyed by the content hash the server computes (sha256 hex,
// first 16 characters; the engine treats it as an opaque cache key).  Parsed once per key, cloned
// per slot.  LRU-bounded at `cap` entries; an entry a slot pins with a shared_ptr is never evicted.
#pragma once

#include "strata/grammar/grammar.hpp"
#include "strata/grammar/grammar_mask.hpp"
#include "strata/grammar/grammar_vocab.hpp"

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata {
namespace grammar {

struct GrammarEntry {
    std::shared_ptr<grammar> proto;            // the parsed grammar, cloned when a slot admits
    std::shared_ptr<const GrammarVocab> vocab; // the shared model vocabulary
    int mask_words = 0;                        // (n_vocab + 31) / 32
    std::shared_ptr<GrammarMaskCache> cache;   // stacks -> mask rows, shared by the slots of this grammar
};

// One slot's live grammar: a fresh clone of the entry's proto (its stacks start at root) plus the
// staged mask for the next pick.  The entry keeps the vocab and the shared cache alive.
struct SlotGrammar {
    std::shared_ptr<const GrammarEntry> entry;
    std::shared_ptr<grammar> g;         // the clone; grammar_free deletes it
    std::vector<uint32_t> mask;         // the mask for g->stacks, staged for the engine

    explicit SlotGrammar(std::shared_ptr<const GrammarEntry> e);

    // consume the token the engine emitted and restage `mask` for the next pick.  False on
    // rejection (no stack survives the token: the engine fed a token the mask had banned).
    bool advance(int32_t token, std::string& err);
};

class GrammarRegistry {
public:
    explicit GrammarRegistry(std::shared_ptr<const GrammarVocab> vocab, size_t cap = 64)
        : vocab_(std::move(vocab)), cap_(cap) {}

    // parse `text` and store it under `sha16`.  Re-registering a known key is a no-op that
    // returns true (the server re-sends the text with every registration; the key is the check).
    // On failure returns false and fills `err` with a one-line message.
    bool register_grammar(const std::string& sha16, const std::string& text, std::string& err);

    // the entry for `sha16`, or null; a hit moves the entry to the LRU head
    std::shared_ptr<GrammarEntry> find(const std::string& sha16);

    size_t size() const { return index_.size(); }

private:
    struct Row {
        std::string sha;
        std::shared_ptr<GrammarEntry> entry;
    };

    void evict_to_cap();

    std::shared_ptr<const GrammarVocab> vocab_;
    size_t cap_;
    std::list<Row> lru_; // front = most recently used
    std::map<std::string, std::list<Row>::iterator> index_;
};

} // namespace grammar
} // namespace strata
