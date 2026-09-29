// src/spec/suffix_drafter.cpp - see include/strata/spec/suffix_drafter.hpp.
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>

namespace strata::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
    size_t cap = 16;
    while (cap < capacity_tokens * 2) cap <<= 1;          // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
    prev_.reserve(capacity_tokens);
}

void SuffixDrafter::reset() {
    hist_.clear();
    prev_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    used_ = 0;
    last_match_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) {
    for (size_t i = key & mask_;; i = (i + 1) & mask_) {    // never full: rebuilt at half load
        Slot& s = table_[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            ++used_;
            return &s;
        }
    }
}

void SuffixDrafter::link(size_t end) {
    int32_t before = -1;
    if (end >= 2) {
        Slot* s = find_slot(key_at(end), true);
        before = s->last;
        s->last = (int32_t) end;
    }
    prev_.push_back(before);
}

// The indexed positions again, without the slots of undone ones, in a table that holds them at a quarter load.
void SuffixDrafter::rebuild() {
    const size_t n = prev_.size();
    size_t cap = table_.size();
    while (cap < 4 * n) cap <<= 1;
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    used_ = 0;
    prev_.clear();
    for (size_t e = 0; e < n; ++e) link(e);
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (2 * (used_ + 1) > table_.size()) rebuild();
        hist_.push_back(tokens[i]);
        link(hist_.size() - 1);
    }
}

void SuffixDrafter::assign(const int32_t* tokens, size_t n) {
    size_t keep = 0;
    const size_t common = std::min(n, hist_.size());
    while (keep < common && hist_[keep] == tokens[keep]) ++keep;
    for (size_t e = hist_.size(); e-- > std::max<size_t>(keep, 2);)   // newest first: each trigram's last goes back
        if (Slot* s = find_slot(key_at(e), false)) s->last = prev_[e];
    hist_.resize(keep);
    prev_.resize(keep);
    last_match_ = 0;
    append(tokens + keep, n - keep);
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    size_t best_end = 0;
    int best_len = 0;
    int32_t p = prev_[cur];                                 // the current trigram's earlier occurrences, newest first
    for (int w = 0; w < WAYS && p >= 0; ++w, p = prev_[(size_t) p]) {
        int len = 0;
        while (len < max_match_ && len <= p && hist_[(size_t) (p - len)] == hist_[cur - len]) ++len;
        if (len > best_len) { best_len = len; best_end = (size_t) p; }   // most recent first, so ties keep the newer
        if (best_len == max_match_) break;
    }
    if (best_len < min_match_) return 0;
    last_match_ = best_len;
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

}  // namespace strata::spec
