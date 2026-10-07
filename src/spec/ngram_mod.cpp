// src/spec/ngram_mod.cpp - see include/strata/spec/ngram_mod.hpp.
#include "strata/spec/ngram_mod.hpp"

#include <algorithm>

namespace strata::spec {

NgramMod::NgramMod(int n, size_t entries) : n_(std::max(1, n)) {
    entries_.assign(std::max<size_t>(entries, 1), EMPTY);
}

size_t NgramMod::index(const entry_t* tokens) const {
    uint64_t h = 0;
    for (int i = 0; i < n_; ++i) h = h * 6364136223846793005ull + (uint64_t) (int64_t) tokens[i];
    return (size_t) (h % entries_.size());
}

void NgramMod::add(const entry_t* tokens) {
    const size_t i = index(tokens);
    if (entries_[i] == EMPTY) ++used_;
    entries_[i] = tokens[n_];
}

NgramMod::entry_t NgramMod::get(const entry_t* tokens) const {
    return entries_[index(tokens)];
}

void NgramMod::reset() {
    std::fill(entries_.begin(), entries_.end(), EMPTY);
    used_ = 0;
}

}  // namespace strata::spec
