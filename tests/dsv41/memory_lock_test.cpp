// Linux syscall tests, using the same allocator / arena lifecycle as DS4. No GPU or model download.
#include "ds4_mlock.hpp"
#if defined(__linux__)
static thread_local bool demand_lock_probe = false;
static thread_local size_t demand_lock_calls = 0;
static thread_local bool fail_cache_mapping = false;
extern "C" void* __real_mmap(void*, size_t, int, int, int, off_t);
extern "C" void* __wrap_mmap(void* p, size_t n, int prot, int flags, int fd, off_t off) {
    if (fail_cache_mapping) { errno = ENOMEM; return MAP_FAILED; }
    return __real_mmap(p, n, prot, flags, fd, off);
}
extern "C" int __real_mlock(const void*, size_t);
extern "C" int __wrap_mlock(const void* p, size_t n) {
    if (demand_lock_probe) ++demand_lock_calls;
    return __real_mlock(p, n);
}
#endif
#include "ds4_os.hpp"
#include "ds4_store.hpp"
#include "ds4_engram_io.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace strata;
using platform::MemoryCategory;
using platform::MemoryLock;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)

#if defined(__linux__)
static size_t vm_locked() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmLck:", 0) == 0) return std::stoull(line.substr(6)) * 1024;
    }
    throw std::runtime_error("VmLck missing");
}
static size_t smaps_locked(const void* address) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    bool selected = false;
    while (std::getline(f, line)) {
        unsigned long long lo, hi;
        if (std::sscanf(line.c_str(), "%llx-%llx", &lo, &hi) == 2)
            selected = uintptr_t(address) >= lo && uintptr_t(address) < hi;
        if (selected && line.rfind("Locked:", 0) == 0) return std::stoull(line.substr(7)) * 1024;
    }
    throw std::runtime_error("smaps mapping missing");
}
static size_t rollup_locked() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("Locked:", 0) == 0) return std::stoull(line.substr(7)) * 1024;
    throw std::runtime_error("smaps_rollup Locked missing");
}
static void overlaps() {
    const size_t page = platform::memory_page_size(), baseline = vm_locked();
    void* p = ds4::os::anon_alloc(4 * page, -1, 1);
    CHECK(p);
    {
        MemoryLock a(MemoryCategory::weights, static_cast<char*>(p) + 3, 2 * page - 3);
        CHECK(a.locked());
        {
            MemoryLock duplicate(MemoryCategory::weights, p, page);
            CHECK(duplicate.locked());
            CHECK(MemoryLock::locked_bytes(MemoryCategory::weights) == 2 * page);
        }
        CHECK(vm_locked() == baseline + 2 * page);
        MemoryLock b(MemoryCategory::context, static_cast<char*>(p) + page, 2 * page);
        CHECK(b.locked());
        CHECK(vm_locked() == baseline + 3 * page);
        a.reset();
        CHECK(vm_locked() == baseline + 2 * page);
        CHECK(smaps_locked(static_cast<char*>(p) + page) >= 2 * page);
        MemoryLock moved(std::move(b));
        CHECK(!b.locked() && moved.locked());
    }
    CHECK(vm_locked() == baseline);
    ds4::os::anon_free(p, 4 * page);
}
static void mapped_weights() {
    const size_t page = platform::memory_page_size(), baseline = vm_locked();
    // Anonymous temporary file: exercise file-backed CPU mappings without model files.
    FILE* file = std::tmpfile();
    CHECK(file);
    CHECK(::ftruncate(::fileno(file), off_t(2 * page)) == 0);
    void* p = ds4::os::map_ro(::fileno(file), 2 * page);
    CHECK(p);
    {
        MemoryLock lock(MemoryCategory::weights, p, 2 * page, ds4::WeightLockPolicy::enabled());
        CHECK(lock.locked() == ds4::WeightLockPolicy::enabled());
        CHECK(vm_locked() == baseline + (lock.locked() ? 2 * page : 0));
    }
    CHECK(vm_locked() == baseline);
    ds4::os::unmap(p, 2 * page);
    std::fclose(file);
}
static void categories() {
    const size_t page = platform::memory_page_size(), baseline = vm_locked();
    const size_t rollup_baseline = rollup_locked();
    const bool weights = ds4::WeightLockPolicy::enabled(), context = ds4::ContextLockPolicy::enabled();
    {
        ds4::WeightVector<uint8_t> model(page, 7);
        ds4::ContextVector<uint8_t> first(page, 3), second(page, 5);
        CHECK(MemoryLock::locked_bytes(MemoryCategory::weights) == (weights ? page : 0));
        CHECK(MemoryLock::locked_bytes(MemoryCategory::context) == (context ? 2 * page : 0));
        CHECK(smaps_locked(model.data()) == (weights ? page : 0));
        CHECK(smaps_locked(first.data()) == (context ? 2 * page : 0) ||
              smaps_locked(first.data()) == (context ? page : 0)); // adjacent locked VMAs can merge
        CHECK(vm_locked() == baseline + (weights ? page : 0) + (context ? 2 * page : 0));
        CHECK(rollup_locked() == rollup_baseline + (weights ? page : 0) + (context ? 2 * page : 0));
        std::printf("proc evidence: DS4_MLOCK_WEIGHTS=%d DS4_MLOCK_CONTEXT=%d VmLck=%zu smaps_rollup.Locked=%zu bytes\n",
                    int(weights), int(context), vm_locked(), rollup_locked());
        const auto saved = first; // checkpoint copy
        first.resize(4 * page, 9); // growth must lock the new allocation before old storage dies
        CHECK(first[0] == 3 && first[3 * page] == 9 && second[0] == 5);
        CHECK(MemoryLock::locked_bytes(MemoryCategory::context) == (context ? 6 * page : 0));
        first = saved; // restore with retained capacity
        CHECK(first.size() == page && first[0] == 3);
        ds4::ContextVector<uint8_t> restored(std::move(second));
        CHECK(restored[0] == 5);
        MemoryLock::report();
    }
    CHECK(vm_locked() == baseline);
    CHECK(MemoryLock::locked_bytes(MemoryCategory::weights) == 0);
    CHECK(MemoryLock::locked_bytes(MemoryCategory::context) == 0);
}
static void experts() {
    const size_t baseline = vm_locked();
    FILE* file = std::tmpfile();
    CHECK(file && ::ftruncate(::fileno(file), 16384) == 0);
    std::vector<uint8_t> pattern(16384);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = uint8_t(i % 251);
    CHECK(pwrite(fileno(file), pattern.data(), pattern.size(), 0) == (ssize_t) pattern.size());
    {
        ds4::ExpertStore::Cfg cfg;
        cfg.budget_bytes = 128 * 1024;
        cfg.floor_bytes = 0;
        ds4::ExpertStore::LayerGeo geo;
        geo.n_exp = 1;
        geo.blob = 4096;
        for (int r = 0; r < 3; ++r) {
            geo.file[r] = 0; geo.off[r] = r * 32; geo.bytes[r] = 16; geo.tr[r] = r * 16;
        }
        ds4::ExpertStore store;
        CHECK(store.init(cfg, {geo}, {ds4::os::kBadFd}, {::fileno(file)}));
        CHECK(store.slots_allocated() > 0);
        CHECK((MemoryLock::locked_bytes(MemoryCategory::weights) > 0) == ds4::WeightLockPolicy::enabled());
        CHECK(store.shrink_chunks(100) > 0);
        CHECK(MemoryLock::locked_bytes(MemoryCategory::weights) == 0);
        CHECK(vm_locked() == baseline);
        const int expert = 0;
        std::vector<char> ok;
        demand_lock_probe = true;
        const bool first = store.ensure(0, &expert, 1, ok);
        demand_lock_probe = false;
        CHECK(demand_lock_calls == 0);
        if (ds4::WeightLockPolicy::enabled()) {
            CHECK(!first); // no ready backing: immediate existing file/mapped fallback
            for (int i = 0; i < 1000 && store.slots_allocated() == 0; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CHECK(store.slots_allocated() > 0);
            demand_lock_probe = true;
            CHECK(store.ensure(0, &expert, 1, ok));
            demand_lock_probe = false;
            CHECK(demand_lock_calls == 0);
        } else CHECK(first); // original synchronous growth, no new locking
        CHECK(ok.size() == 1 && ok[0]);
        const int slot = store.find(0, expert); CHECK(slot >= 0);
        for (int r = 0; r < 3; ++r)
            CHECK(std::memcmp(store.mat(slot, r), pattern.data() + r * 32, 16) == 0);
        CHECK((MemoryLock::locked_bytes(MemoryCategory::weights) > 0) == ds4::WeightLockPolicy::enabled());
        store.release(0, &expert, 1, ok);
        CHECK(store.shrink_chunks(100) > 0);
        CHECK(vm_locked() == baseline);
    }
    {
        ds4::EgRowCache cache;
        cache.init(8192, 16);
        demand_lock_probe = true;
        CHECK(cache.put(42));
        CHECK(cache.find(42));
        for (uint64_t k = 43; k < 9000; ++k) {
            auto* p = cache.put(k); CHECK(p); p[0] = uint8_t(k);
            CHECK(cache.find(k)[0] == uint8_t(k));
        }
        demand_lock_probe = false;
        CHECK(demand_lock_calls == 0); // first miss, block boundary, capacity exhaustion/eviction
        CHECK((MemoryLock::locked_bytes(MemoryCategory::weights) > 0) == ds4::WeightLockPolicy::enabled());
    }
    CHECK(vm_locked() == baseline);
    std::fclose(file);
    if (ds4::WeightLockPolicy::enabled()) {
        fail_cache_mapping = true;
        ds4::EgRowCache failed;
        failed.init(8192, 16);
        fail_cache_mapping = false;
        CHECK(failed.capacity() == 0 && failed.put(1) == nullptr);
        CHECK(vm_locked() == baseline);
    }
}
static void partial_failure_and_threads() {
    const size_t baseline = vm_locked(), page = platform::memory_page_size();
    void* p = ds4::os::anon_alloc(3 * page, -1, 1);
    CHECK(p);
    {
        MemoryLock owned(MemoryCategory::weights, static_cast<char*>(p) + 2 * page, page);
        CHECK(owned.locked());
        CHECK(::mprotect(static_cast<char*>(p) + page, page, PROT_NONE) == 0);
        MemoryLock failed(MemoryCategory::context, p, 3 * page);
        CHECK(!failed.locked());
        CHECK(vm_locked() == baseline + page); // partial rollback must not unlock 'owned'
        CHECK(::mprotect(static_cast<char*>(p) + page, page, PROT_READ | PROT_WRITE) == 0);
    }
    ds4::os::anon_free(p, 3 * page);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([page] {
        for (int n = 0; n < 20; ++n) {
            ds4::ContextVector<uint8_t> context(page, 17);
            ds4::WeightVector<uint8_t> weights(page, 19);
        }
    });
    for (auto& t : threads) t.join();
    CHECK(vm_locked() == baseline);
}
static void preparation_lifecycle() {
    if (!ds4::WeightLockPolicy::enabled()) return;
    const size_t baseline = vm_locked();
    std::atomic<uint64_t> available{0};
    ds4::ExpertStore::Cfg cfg;
    cfg.budget_bytes = 128 * 1024; cfg.floor_bytes = 4096;
    cfg.ram_avail = [&] { return available.load(); };
    ds4::ExpertStore::LayerGeo geo; geo.blob = 4096; geo.n_exp = 1;
    for (int r = 0; r < 3; ++r) { geo.file[r] = 0; geo.bytes[r] = 16; geo.tr[r] = r * 16; }
    // No read is issued while no ready slot exists, so bad file handles are safe.
    {
        ds4::ExpertStore store;
        CHECK(store.init(cfg, {geo}, {ds4::os::kBadFd}, {ds4::os::kBadFd}));
        CHECK(store.slots_allocated() == 0); // startup stopped at RAM floor
        std::vector<char> ok; std::vector<int> slots; const int expert = 0;
        demand_lock_probe = true; store.begin(0, &expert, 1, ok, slots); demand_lock_probe = false;
        CHECK(slots[0] == -1 && demand_lock_calls == 0);
        // Destruction joins even a pending floor-limited preparation request.
    }
    CHECK(vm_locked() == baseline);
    {
        ds4::ExpertStore store;
        CHECK(store.init(cfg, {geo}, {ds4::os::kBadFd}, {ds4::os::kBadFd}));
        available = ~uint64_t{0};
        std::vector<char> ok; std::vector<int> slots; const int expert = 0;
        demand_lock_probe = true; store.begin(0, &expert, 1, ok, slots); demand_lock_probe = false;
        CHECK(slots[0] == -1 && demand_lock_calls == 0);
        for (int i = 0; i < 1000 && store.slots_allocated() == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(store.slots_allocated() > 0); // background preparation resumes within original budget
        CHECK(store.slots_allocated() <= store.slots_max());
    }
    CHECK(vm_locked() == baseline);
    // Simulated registration failure after commit/lock: unwind and decommit,
    // without publishing a slot or stranding ownership. No GPU registration is performed.
    available = 0;
    std::atomic<bool> attempted{false};
    cfg.map_chunk = [&](void*, size_t) -> void* { attempted = true; throw std::bad_alloc(); };
    {
        ds4::ExpertStore store;
        CHECK(store.init(cfg, {geo}, {ds4::os::kBadFd}, {ds4::os::kBadFd}));
        available = ~uint64_t{0};
        std::vector<char> ok; std::vector<int> slots; const int expert = 0;
        store.begin(0, &expert, 1, ok, slots);
        for (int i = 0; i < 1000 && !attempted.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(attempted.load());
    }
    CHECK(vm_locked() == baseline);
}
static void failure() {
    const size_t baseline = vm_locked(), page = platform::memory_page_size();
    rlimit old{};
    CHECK(::getrlimit(RLIMIT_MEMLOCK, &old) == 0);
    rlimit limit = old;
    limit.rlim_cur = 0;
    CHECK(::setrlimit(RLIMIT_MEMLOCK, &limit) == 0); // test only, never production code
    {
        ds4::ContextVector<uint8_t> ctx(page, 19);
        ds4::WeightVector<uint8_t> weights(page, 23);
        if (ds4::WeightLockPolicy::enabled()) {
            ds4::EgRowCache cache; cache.init(4097, 16);
            demand_lock_probe = true;
            auto* row = cache.put(17); CHECK(row); row[0] = 29;
            CHECK(cache.find(17)[0] == 29);
            demand_lock_probe = false; CHECK(demand_lock_calls == 0);
        }
        CHECK(ctx[0] == 19 && weights[0] == 23);
        CHECK(MemoryLock::locked_bytes(MemoryCategory::weights) == 0);
        CHECK(MemoryLock::locked_bytes(MemoryCategory::context) == 0);
        void* p = ds4::os::anon_alloc(page, -1, 1);
        CHECK(p);
        MemoryLock a(MemoryCategory::weights, p, page), b(MemoryCategory::weights, p, page);
        CHECK(!a.locked() && !b.locked());
        ds4::os::anon_free(p, page);
    }
    CHECK(::setrlimit(RLIMIT_MEMLOCK, &old) == 0);
    CHECK(vm_locked() == baseline);
    MemoryLock bad(MemoryCategory::context, reinterpret_cast<void*>(UINTPTR_MAX - 1), 64);
    CHECK(!bad.locked());
}
#endif
int main() {
    try {
        ds4::mlock_startup();
        const auto requested = [](const char* name) {
            const char* value = std::getenv(name);
            return value && std::strcmp(value, "1") == 0;
        };
        CHECK(ds4::WeightLockPolicy::enabled() == requested("DS4_MLOCK_WEIGHTS"));
        CHECK(ds4::ContextLockPolicy::enabled() == requested("DS4_MLOCK_CONTEXT"));
        CHECK(!ds4::mapped_experts_need_lock(true, {0, -1}, {-1, 0}));
        CHECK(!ds4::mapped_experts_need_lock(true, {-1, 0}, {0, -1})); // same copied layer after a GPU swap
        CHECK(ds4::mapped_experts_need_lock(true, {0, -1}, {-1, -1}));
        CHECK(ds4::mapped_experts_need_lock(true, {0, -1, -1}, {-1, 0, -1}));
        CHECK(ds4::mapped_experts_need_lock(false, {0, -1}, {-1, 0})); // evictable RAM tier
        CHECK(ds4::mapped_experts_need_lock(false, {0, -1}, {}));
        CHECK(!ds4::mapped_experts_need_lock(false, {0, 1}, {})); // GPU-only: no CPU fallback
#if defined(__linux__)
        categories();
        overlaps();
        mapped_weights();
        experts();
        partial_failure_and_threads();
        preparation_lifecycle();
        failure();
#endif
        std::puts("memory locking tests passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "memory locking tests failed: %s\n", e.what()); return 1;
    }
}
