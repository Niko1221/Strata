#pragma once

#include "strata/core/progress.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>
#include <unordered_map>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace strata::core {

// Read once; restart to change mode.
inline bool conversation_dedup_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_PARKED_DEDUP");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}

struct ConversationStorageStats {
    std::atomic<size_t> payload{0}, logical{0}, peak{0}, objects{0}, hits{0}, compared{0}, cow{0};
    void add(size_t n) {
        const size_t now = payload.fetch_add(n) + n;
        size_t old = peak.load();
        while (old < now && !peak.compare_exchange_weak(old, now)) {}
    }
    // Start a new high-water window at the current ledger (tests, diagnostics).
    size_t reset_peak() { const size_t now = payload.load(); peak.store(now); return now; }
    // Dedup mode admits the storage a capture adds: ledger growth, not image size.
    bool added_within(size_t start, size_t admitted) const { return peak.load() - start <= admitted; }
};
inline ConversationStorageStats& conversation_storage_stats() {
    static ConversationStorageStats s;
    return s;
}

// 1 MiB blocks fall below glibc's dynamic mmap threshold, so freed blocks stay
// in the heap. Trim before re-reading MemAvailable after an eviction.
inline void conversation_release_free_heap() {
#if defined(__GLIBC__)
    if (conversation_dedup_enabled()) malloc_trim(0);
#endif
}

// Share only frozen payloads; mutable access and mutable copies detach.
// Complete outstanding writes before freeze().
class ConversationBytes {
    struct Payload;
    struct Pool {
        std::mutex mutex;
        // Weak entries never keep payloads alive; each payload erases its own.
        std::unordered_multimap<uint64_t, std::weak_ptr<const Payload>> index;
    };
    // Never destroyed: payloads may outlive other statics at exit.
    static Pool& pool() { static Pool& p = *new Pool; return p; }
    struct Payload {
        std::vector<uint8_t> value;
        uint64_t key = 0;
        bool indexed = false;
        explicit Payload(std::vector<uint8_t>&& v) : value(std::move(v)) {}
        ~Payload() {
            conversation_storage_stats().payload.fetch_sub(value.capacity());
            if (!indexed) return;
            // Every expired entry belongs to a payload being destroyed, so
            // erasing one per destructor keeps the index exact.
            auto& p = pool(); std::lock_guard<std::mutex> lock(p.mutex);
            auto [first, last] = p.index.equal_range(key);
            for (auto i = first; i != last; ++i)
                if (i->second.expired()) { p.index.erase(i); return; }
        }
    };
    std::vector<uint8_t> writable_;
    std::shared_ptr<const Payload> frozen_;
    size_t logical_charge_ = 0;
    void sync_logical() {
        if (!conversation_dedup_enabled()) return;
        const size_t n = capacity();
        if (n >= logical_charge_) conversation_storage_stats().logical.fetch_add(n - logical_charge_);
        else conversation_storage_stats().logical.fetch_sub(logical_charge_ - n);
        logical_charge_ = n;
    }
    void charge(size_t old) {
        if (!conversation_dedup_enabled()) return;
        const size_t n = writable_.capacity();
        if (n == old) return;
        conversation_storage_stats().add(n); // Count old and new allocations at peak.
        conversation_storage_stats().payload.fetch_sub(old);
        sync_logical();
    }
    // Copy the first `keep` bytes of a frozen payload into `room` private bytes.
    void detach(size_t keep, size_t room) {
        std::vector<uint8_t> copy; copy.reserve(room);
        copy.insert(copy.end(), frozen_->value.begin(), frozen_->value.begin() + keep);
        conversation_storage_stats().add(copy.capacity());
        writable_ = std::move(copy);
        frozen_.reset();
        ++conversation_storage_stats().cow;
        sync_logical();
    }
    void detach() { if (frozen_) detach(frozen_->value.size(), frozen_->value.size()); }
    const std::vector<uint8_t>& value() const { return frozen_ ? frozen_->value : writable_; }
    static uint64_t hash(const uint8_t* p, size_t n) {
#if defined(STRATA_DEDUP_TEST_HASH_COLLISION)
        return 0;
#endif
        uint64_t h = 0x9e3779b97f4a7c15ULL ^ n;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint64_t w; std::memcpy(&w, p + i, 8);
            h ^= w * 0xbf58476d1ce4e5b9ULL;
            h = (h << 27 | h >> 37) * 0x94d049bb133111ebULL;
            if ((i & ((1 << 20) - 1)) == 0) progress_beat();
        }
        for (; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
        return h;
    }
public:
    using value_type = uint8_t;
    // Ledger allowances per live wrapper and per index node (cache metadata_bytes()).
    static constexpr size_t wrapper_allowance = 128, index_node_allowance = 128;
    ConversationBytes() { if (conversation_dedup_enabled()) ++conversation_storage_stats().objects; }
    ConversationBytes(std::initializer_list<uint8_t> v) : ConversationBytes() { assign(v.begin(), v.end()); }
    ConversationBytes(const ConversationBytes& b) : ConversationBytes() {
        if (b.frozen_) { frozen_ = b.frozen_; sync_logical(); }
        else { writable_ = b.writable_; charge(0); }
    }
    ConversationBytes(ConversationBytes&& b) noexcept : ConversationBytes() {
        writable_ = std::move(b.writable_); frozen_ = std::move(b.frozen_);
        logical_charge_ = std::exchange(b.logical_charge_, 0);
    }
    ~ConversationBytes() {
        if (conversation_dedup_enabled()) {
            conversation_storage_stats().payload.fetch_sub(writable_.capacity());
            conversation_storage_stats().logical.fetch_sub(logical_charge_);
            --conversation_storage_stats().objects;
        }
    }
    ConversationBytes& operator=(const ConversationBytes& b) {
        if (this != &b) { ConversationBytes copy(b); swap(copy); }
        return *this;
    }
    ConversationBytes& operator=(ConversationBytes&& b) noexcept {
        if (this != &b) { ConversationBytes move(std::move(b)); swap(move); }
        return *this;
    }
    ConversationBytes& operator=(std::initializer_list<uint8_t> values) {
        assign(values.begin(), values.end()); return *this;
    }
    ConversationBytes& operator=(std::vector<uint8_t>&& b) {
        ConversationBytes replacement;
        replacement.writable_ = std::move(b); replacement.charge(0); swap(replacement); return *this;
    }
    void swap(ConversationBytes& b) noexcept { writable_.swap(b.writable_); frozen_.swap(b.frozen_); std::swap(logical_charge_, b.logical_charge_); }
    size_t size() const { return value().size(); }
    size_t capacity() const { return value().capacity(); }
    bool empty() const { return size() == 0; }
    bool shared() const { return bool(frozen_); }
    // Frozen payload identity and owner count, for eviction planning.
    const void* payload_id() const { return frozen_.get(); }
    long payload_owners() const { return frozen_.use_count(); }
    bool indexed() const { return frozen_ && frozen_->indexed; }
    const uint8_t* data() const { return value().data(); }
    uint8_t* data() { detach(); return writable_.data(); }
    auto begin() const { return value().begin(); }
    auto end() const { return value().end(); }
    auto begin() { detach(); return writable_.begin(); }
    auto end() { detach(); return writable_.end(); }
    const uint8_t& operator[](size_t n) const { return value()[n]; }
    uint8_t& operator[](size_t n) { detach(); return writable_[n]; }
    void reserve(size_t n) {
        if (n <= capacity()) return;
        if (frozen_) { detach(frozen_->value.size(), n); return; }
        const size_t old = writable_.capacity();
        writable_.reserve(n); charge(old);
    }
    void resize(size_t n, uint8_t v = 0) {
        if (n == size()) return;
        // A frozen source copies only the bytes that survive, at the final size.
        if (frozen_) detach(std::min(n, frozen_->value.size()), n);
        const size_t old = writable_.capacity();
        writable_.resize(n, v); charge(old);
    }
    void clear() { resize(0); }
    void pop_back() { resize(size() - 1); }
    void push_back(uint8_t v) { const size_t n = size(); resize(n + 1); writable_[n] = v; }
    void assign(size_t n, uint8_t v) {
        detach(); const size_t old = writable_.capacity(); writable_.assign(n, v); charge(old);
    }
    template<class It> void assign(It a, It b) {
        detach(); const size_t old = writable_.capacity(); writable_.assign(a, b); charge(old);
    }
    bool operator==(const ConversationBytes& b) const { return value() == b.value(); }
    bool operator==(const std::vector<uint8_t>& b) const { return value() == b; }
    // Intern only after comparing all bytes.
    void freeze() {
        if (!conversation_dedup_enabled() || frozen_ || empty()) return;
        const uint64_t key = hash(writable_.data(), writable_.size());
        // Released after the lock: a last reference would re-enter the pool.
        std::vector<std::shared_ptr<const Payload>> rejected;
        auto& p = pool(); std::lock_guard<std::mutex> lock(p.mutex);
        auto [first, last] = p.index.equal_range(key);
        for (auto i = first; i != last; ++i) {
            auto candidate = i->second.lock();
            if (!candidate) continue;
            if (candidate->value.size() != writable_.size()) { rejected.push_back(std::move(candidate)); continue; }
            conversation_storage_stats().compared += writable_.size();
            bool equal = true;
            for (size_t at = 0; at < writable_.size(); at += 1 << 20) {
                const size_t n = std::min<size_t>(1 << 20, writable_.size() - at);
                if (std::memcmp(writable_.data() + at, candidate->value.data() + at, n)) { equal = false; break; }
                progress_beat();
            }
            if (!equal) { rejected.push_back(std::move(candidate)); continue; }
            const size_t old = writable_.capacity();
            std::vector<uint8_t>().swap(writable_);
            conversation_storage_stats().payload.fetch_sub(old);
            frozen_ = std::move(candidate); ++conversation_storage_stats().hits; sync_logical();
            return;
        }
        // Allocate before moving so failure leaves the buffer intact.
        auto created = std::make_shared<Payload>(std::move(writable_));
        created->key = key;
        try { p.index.emplace(key, created); created->indexed = true; }
        catch (const std::bad_alloc&) {} // shareable later only by its own copies
        frozen_ = std::move(created); sync_logical();
    }
    static size_t index_entries() {
        auto& p = pool(); std::lock_guard<std::mutex> lock(p.mutex);
        return p.index.size();
    }
    static size_t index_bytes() {
        auto& p = pool(); std::lock_guard<std::mutex> lock(p.mutex);
        // Allow for index nodes, retained buckets and allocator overhead.
        return p.index.bucket_count() * sizeof(void*) + p.index.size() * index_node_allowance;
    }
};

} // namespace strata::core
