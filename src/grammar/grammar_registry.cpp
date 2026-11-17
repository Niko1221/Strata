#include "strata/grammar/grammar_registry.hpp"

#include <exception>

namespace strata {
namespace grammar {

bool GrammarRegistry::register_grammar(const std::string& sha16, const std::string& text, std::string& err) {
    err.clear();
    if (!vocab_) {
        // grammar=0 on the INFO line keeps the server from sending a GRAMMAR line, but the engine's
        // stdin is a wire: a hand-written one must not take the process down.
        err = "no grammar vocabulary";
        return false;
    }
    auto it = index_.find(sha16);
    if (it != index_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second); // known key: no reparse
        return true;
    }
    grammar* parsed = nullptr;
    try {
        parsed = grammar_init(vocab_.get(), text.c_str(), "root");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (!parsed) {
        err = "grammar parse failed";
        return false;
    }
    auto entry = std::make_shared<GrammarEntry>();
    entry->proto = std::shared_ptr<grammar>(parsed, [](grammar* g) { grammar_free(g); });
    entry->vocab = vocab_;
    entry->mask_words = (int) ((vocab_->pieces.size() + 31) / 32);
    // one mask cache per grammar: every slot that clones this proto shares the stack->mask rows, and
    // the vocabulary they are built over is the registry's one prepared copy
    entry->cache = std::make_shared<GrammarMaskCache>(*vocab_, grammar_get_rules(entry->proto.get()),
                                                     4096, prepped_);
    lru_.push_front(Row{sha16, entry});
    index_[sha16] = lru_.begin();
    evict_to_cap();
    return true;
}

std::shared_ptr<GrammarEntry> GrammarRegistry::find(const std::string& sha16) {
    auto it = index_.find(sha16);
    if (it == index_.end())
        return nullptr;
    lru_.splice(lru_.begin(), lru_, it->second);
    return it->second->entry;
}

SlotGrammar::SlotGrammar(std::shared_ptr<const GrammarEntry> e)
    : entry(std::move(e)), g(grammar_clone(*entry->proto), [](grammar* p) { grammar_free(p); }) {
    // The stacks point into THIS clone's rules vector (grammar_clone copies it), so the cache keys must be
    // resolved against the clone's rules, not the proto's - the (rule, offset) range scan misses every clone
    // pointer otherwise and every state collapses to one key. The layout is identical, so the masks match.
    mask = entry->cache->mask_for(grammar_stacks(g.get()), grammar_get_rules(g.get()), g->partial_utf8); // the root state
}

bool SlotGrammar::advance(int32_t token, std::string& err) {
    err.clear();
    try {
        grammar_accept(*g, token, entry->vocab->token_to_piece(token));
    } catch (const std::exception& ex) {
        err = ex.what(); // "Unexpected empty grammar stack after accepting piece: ..."
        return false;
    }
    // g->partial_utf8 is part of the state: a byte-fallback vocabulary splits a character across
    // tokens, and halfway through one only the finishing pieces may ride.
    mask = entry->cache->mask_for(grammar_stacks(g.get()), grammar_get_rules(g.get()), g->partial_utf8);
    return true;
}

void GrammarRegistry::evict_to_cap() {
    // drop from the LRU tail; an entry a slot pins (use_count > 1) stays, so the cap can be
    // exceeded for as long as the slots hold on
    while (index_.size() > cap_) {
        auto victim = lru_.end();
        for (auto r = lru_.rbegin(); r != lru_.rend(); ++r) {
            if (r->entry.use_count() == 1) {
                victim = std::next(r).base();
                break;
            }
        }
        if (victim == lru_.end())
            break;
        index_.erase(victim->sha);
        lru_.erase(victim);
    }
}

} // namespace grammar
} // namespace strata
