// Optional, allocation-scoped Linux residency. No environment parsing and no GPU operations here.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>
#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace strata::platform {

enum class MemoryCategory { weights, context };
enum class MemoryOwner { generic, core, ds4 };
inline const char* memory_owner_name(MemoryOwner o) {
    return o == MemoryOwner::core ? "strata" : o == MemoryOwner::ds4 ? "ds4" : "memory";
}
inline const char* memory_category_name(MemoryCategory c) {
    return c == MemoryCategory::weights ? "weights" : "context";
}
inline size_t memory_page_size() {
#if defined(__linux__)
    static const size_t size = [] { const long n = ::sysconf(_SC_PAGESIZE); return n > 0 ? size_t(n) : size_t(4096); }();
    return size;
#else
    return 4096;
#endif
}

// A single process-wide registry, shared by all owners. Linux mlock does not stack: an owner's
// munlock must not unlock a page still referenced by another owner (including another category).
// Entries are intervals, not one heap object per page of a multi-GiB expert arena.
class MemoryLock {
    struct Entry { uintptr_t lo, hi; MemoryCategory category; MemoryOwner owner; };
    struct Counters {
        uint64_t failures[2] = {}, bytes[2] = {}, operations[2] = {};
        uint64_t reasons[2][256] = {};
    };
    struct Registry {
        std::mutex mutex;
        std::list<Entry> entries;
        uint64_t bytes[2] = {}; // union of successful page ranges, maintained at allocation/release time
        Counters owners[3];
    };
    static Registry& registry() {
        // Outlive static storage owners, whose destructors may release locks at program exit.
        alignas(Registry) static unsigned char storage[sizeof(Registry)];
        static Registry* r = ::new (static_cast<void*>(storage)) Registry;
        return *r;
    }
    Entry* entry_ = nullptr;

    // Visit portions not covered by any existing entry. No allocations, even during destruction.
    template<class Fn> static bool gaps(Registry& r, uintptr_t lo, uintptr_t hi, Fn fn,
                                       const MemoryCategory* category = nullptr, const MemoryOwner* owner = nullptr) {
        while (lo < hi) {
            uintptr_t covered = lo, next = hi;
            for (const auto& e : r.entries) {
                if (category && e.category != *category) continue;
                if (owner && e.owner != *owner) continue;
                if (e.lo <= lo && e.hi > covered) covered = e.hi;
                if (e.lo > lo) next = std::min(next, e.lo);
            }
            if (covered > lo) { lo = std::min(hi, covered); continue; }
            if (!fn(lo, next)) return false;
            lo = next;
        }
        return true;
    }
    static void failure(Registry& r, MemoryOwner owner, MemoryCategory c, size_t bytes, int error) {
        const size_t k = c == MemoryCategory::weights ? 0 : 1;
        auto& count = r.owners[size_t(owner)];
        ++count.failures[k];
        const size_t code = error > 0 && error < 256 ? size_t(error) : 0;
        if (count.reasons[k][code]++) return;
        std::fprintf(stderr, "%s mlock: category=%s bytes=%zu failed errno=%d (%s); "
                             "optional lock not acquired (pageable unless otherwise pinned); further failures with this category/errno suppressed\n",
                      memory_owner_name(owner), memory_category_name(c), bytes, error, std::strerror(error));
    }
public:
    MemoryLock() = default;
    MemoryLock(MemoryCategory c, const void* p, size_t bytes, bool enabled = true,
               MemoryOwner owner = MemoryOwner::generic) { acquire(c, p, bytes, enabled, owner); }
    MemoryLock(const MemoryLock&) = delete;
    MemoryLock& operator=(const MemoryLock&) = delete;
    MemoryLock(MemoryLock&& other) noexcept : entry_(std::exchange(other.entry_, nullptr)) {}
    MemoryLock& operator=(MemoryLock&& other) noexcept {
        if (this != &other) { reset(); entry_ = std::exchange(other.entry_, nullptr); }
        return *this;
    }
    ~MemoryLock() { reset(); }
    bool locked() const { return entry_ != nullptr; }
    static void allocation_failed(MemoryCategory c, size_t bytes, MemoryOwner owner = MemoryOwner::generic) {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        failure(r, owner, c, bytes, ENOMEM);
    }
    void acquire(MemoryCategory c, const void* p, size_t bytes, bool enabled = true,
                 MemoryOwner owner = MemoryOwner::generic) {
        reset();
        if (!enabled || bytes == 0) return;
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        const uintptr_t addr = reinterpret_cast<uintptr_t>(p), page = memory_page_size();
        if (!p || bytes > UINTPTR_MAX - addr || addr + bytes > UINTPTR_MAX - (page - 1)) {
            failure(r, owner, c, bytes, EINVAL); return;
        }
        const uintptr_t lo = addr / page * page, hi = (addr + bytes + page - 1) / page * page;
#if defined(__linux__)
        // Allocate bookkeeping before changing kernel state, so bad_alloc cannot strand a lock.
        std::list<Entry> pending;
        try { pending.push_back({lo, hi, c, owner}); }
        catch (const std::bad_alloc&) { failure(r, owner, c, bytes, ENOMEM); return; }
        uintptr_t touched = lo;
        int error = 0;
        const bool ok = gaps(r, lo, hi, [&](uintptr_t a, uintptr_t b) {
            touched = b;
            if (::mlock(reinterpret_cast<void*>(a), b - a) == 0) return true;
            error = errno; return false;
        });
        if (!ok) {
            // Include the failed syscall's span: Linux may have locked part before an error.
            gaps(r, lo, touched, [](uintptr_t a, uintptr_t b) { ::munlock(reinterpret_cast<void*>(a), b - a); return true; });
            failure(r, owner, c, hi - lo, error); return;
        }
        gaps(r, lo, hi, [&](uintptr_t a, uintptr_t b) {
            r.bytes[c == MemoryCategory::weights ? 0 : 1] += b - a; return true;
        }, &c);
        gaps(r, lo, hi, [&](uintptr_t a, uintptr_t b) {
            r.owners[size_t(owner)].bytes[size_t(c)] += b - a; return true;
        }, &c, &owner);
        ++r.owners[size_t(owner)].operations[size_t(c)];
        entry_ = &pending.front();
        r.entries.splice(r.entries.end(), pending);
#else
        (void) lo; (void) hi;
        failure(r, owner, c, bytes, ENOSYS);
#endif
    }
    // Must run BEFORE the owner frees/unmaps storage; a failed acquisition owns nothing.
    void reset() noexcept {
        if (!entry_) return;
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        const uintptr_t lo = entry_->lo, hi = entry_->hi;
        const MemoryCategory category = entry_->category;
        const MemoryOwner owner = entry_->owner;
        for (auto it = r.entries.begin(); it != r.entries.end(); ++it)
            if (&*it == entry_) { r.entries.erase(it); break; }
        entry_ = nullptr;
        gaps(r, lo, hi, [&](uintptr_t a, uintptr_t b) {
            r.bytes[category == MemoryCategory::weights ? 0 : 1] -= b - a; return true;
        }, &category);
        gaps(r, lo, hi, [&](uintptr_t a, uintptr_t b) {
            r.owners[size_t(owner)].bytes[size_t(category)] -= b - a; return true;
        }, &category, &owner);
#if defined(__linux__)
        gaps(r, lo, hi, [](uintptr_t a, uintptr_t b) { ::munlock(reinterpret_cast<void*>(a), b - a); return true; });
#else
        (void) lo; (void) hi;
#endif
    }
    static uint64_t locked_bytes(MemoryCategory c) {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        return r.bytes[c == MemoryCategory::weights ? 0 : 1];
    }
    struct Statistics { uint64_t bytes, operations, failures; };
    static Statistics statistics(MemoryOwner owner, MemoryCategory c) {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        const auto& n = r.owners[size_t(owner)];
        return {n.bytes[size_t(c)], n.operations[size_t(c)], n.failures[size_t(c)]};
    }
    static uint64_t failure_reason(MemoryOwner owner, MemoryCategory c, int error) {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        return r.owners[size_t(owner)].reasons[size_t(c)][error > 0 && error < 256 ? error : 0];
    }
    static void report(MemoryOwner owner = MemoryOwner::generic) {
        auto& r = registry();
        std::lock_guard<std::mutex> guard(r.mutex);
        const auto& n = r.owners[size_t(owner)];
        std::fprintf(stderr, "%s mlock: tracked locked virtual page bytes weights=%llu context=%llu; "
                              "successful operations weights=%llu context=%llu; failed attempts weights=%llu context=%llu\n",
                     memory_owner_name(owner), (unsigned long long) n.bytes[0], (unsigned long long) n.bytes[1],
                     (unsigned long long) n.operations[0], (unsigned long long) n.operations[1],
                     (unsigned long long) n.failures[0], (unsigned long long) n.failures[1]);
    }
};

// Immutable engine attribution; the base registry still protects overlaps process-wide.
template<MemoryOwner Owner> class OwnedMemoryLock : public MemoryLock {
public:
    OwnedMemoryLock() = default;
    OwnedMemoryLock(MemoryCategory c, const void* p, size_t n, bool enabled = true)
        : MemoryLock(c, p, n, enabled, Owner) {}
    void acquire(MemoryCategory c, const void* p, size_t n, bool enabled = true) {
        MemoryLock::acquire(c, p, n, enabled, Owner);
    }
    static void allocation_failed(MemoryCategory c, size_t n) { MemoryLock::allocation_failed(c, n, Owner); }
    static uint64_t locked_bytes(MemoryCategory c) { return MemoryLock::statistics(Owner, c).bytes; }
    static void report() { MemoryLock::report(Owner); }
};

inline void memory_lock_startup(bool weights, bool context, const char* scope,
                                const char* weight_setting, const char* context_setting) {
    std::fprintf(stderr, "%s mlock: %s=%d %s=%d\n", scope, weight_setting, int(weights), context_setting, int(context));
#if defined(__linux__)
    rlimit limit{};
    if (::getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
        const auto print = [](const char* name, rlim_t n) {
            if (n == RLIM_INFINITY) std::fprintf(stderr, " %s=unlimited", name);
            else std::fprintf(stderr, " %s=%llu bytes", name, (unsigned long long) n);
        };
        std::fprintf(stderr, "%s mlock: RLIMIT_MEMLOCK", scope);
        print("soft", limit.rlim_cur); print("hard", limit.rlim_max);
        std::fprintf(stderr, "; Linux mlock available (success depends on limits, permissions and RAM)\n");
    } else std::fprintf(stderr, "%s mlock: getrlimit failed errno=%d (%s); Linux mlock available\n", scope, errno, std::strerror(errno));
#else
    std::fprintf(stderr, "%s mlock: unavailable (Linux only); RLIMIT_MEMLOCK unavailable\n", scope);
#endif
}

// Selected persistent vectors get page-isolated backing when enabled. This avoids locking allocator
// neighbours (e.g. weights and context sharing a malloc page). Growth/copy/move/free use vector's normal
// ownership, with locks acquired only on allocation, never on element access. Policy is immutable.
template<class T, class Policy> class LockingAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;
    template<class U> struct rebind { using other = LockingAllocator<U, Policy>; };
    LockingAllocator() = default;
    template<class U> LockingAllocator(const LockingAllocator<U, Policy>&) noexcept {}
    T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
#if defined(__linux__)
        if (Policy::enabled()) {
            const size_t page = memory_page_size(), bytes = n * sizeof(T);
            static_assert(alignof(T) <= 4096, "over-aligned locking allocation unsupported");
            if (bytes > SIZE_MAX - 2 * page) throw std::bad_array_new_length();
            const size_t total = (bytes + page - 1) / page * page + page;
            void* base = ::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED) throw std::bad_alloc();
            auto* p = static_cast<unsigned char*>(base) + page;
            try { ::new (base) MemoryLock(Policy::category, p, bytes, true, Policy::owner); }
            catch (...) { ::munmap(base, total); throw; }
            return reinterpret_cast<T*>(p);
        }
#endif
        return std::allocator<T>{}.allocate(n);
    }
    void deallocate(T* p, size_t n) noexcept {
#if defined(__linux__)
        if (Policy::enabled()) {
            const size_t page = memory_page_size();
            auto* base = reinterpret_cast<unsigned char*>(p) - page;
            reinterpret_cast<MemoryLock*>(base)->~MemoryLock();
            ::munmap(base, (n * sizeof(T) + page - 1) / page * page + page);
            return;
        }
#endif
        std::allocator<T>{}.deallocate(p, n);
    }
    template<class U> bool operator==(const LockingAllocator<U, Policy>&) const noexcept { return true; }
    template<class U> bool operator!=(const LockingAllocator<U, Policy>&) const noexcept { return false; }
};

// Uninitialized byte storage, for caches whose producer fills every row before publishing it.
// Unlike vector(size), this preserves the original new T[n] path's lack of zero-filling.
template<class T, class Policy> class LockingBuffer {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
    LockingAllocator<T, Policy> allocator_;
    T* data_ = nullptr;
    size_t size_ = 0;
public:
    explicit LockingBuffer(size_t n) : data_(allocator_.allocate(n)), size_(n) {}
    LockingBuffer(const LockingBuffer&) = delete;
    LockingBuffer& operator=(const LockingBuffer&) = delete;
    LockingBuffer(LockingBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    ~LockingBuffer() { if (data_) allocator_.deallocate(data_, size_); }
    T* data() { return data_; }
};

} // namespace strata::platform
