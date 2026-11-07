#include "strata/core/memory_policy.hpp"
#include "ds4_mlock.hpp"
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>
using namespace strata;
using platform::MemoryCategory;
using platform::MemoryOwner;
using platform::MemoryLock;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
static bool locked(const void* p) {
    std::ifstream f("/proc/self/smaps"); std::string line; bool selected = false;
    while (std::getline(f, line)) {
        unsigned long lo, hi;
        if (std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2) selected = uintptr_t(p) >= lo && uintptr_t(p) < hi;
        if (selected && line.starts_with("VmFlags:")) return line.find(" lo ") != std::string::npos || line.ends_with(" lo");
    }
    throw std::runtime_error("mapping unavailable");
}
static void overlap(bool reverse) {
    const size_t page = platform::memory_page_size();
    void* p = mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(p != MAP_FAILED);
    const auto w = MemoryCategory::weights, c = MemoryCategory::context;
    const auto before = MemoryLock::statistics(MemoryOwner::core, w);
    core::EngineMemoryLock a(w, p, 2 * page), duplicate(w, p, page), empty(w, nullptr, 0);
    ds4::EngineMemoryLock b(c, static_cast<char*>(p) + page, 2 * page);
    CHECK(a.locked() && b.locked() && !empty.locked());
    CHECK(MemoryLock::statistics(MemoryOwner::core, w).bytes == 2 * page);
    CHECK(MemoryLock::statistics(MemoryOwner::ds4, c).bytes == 2 * page);
    CHECK(MemoryLock::statistics(MemoryOwner::core, w).operations == before.operations + 2);
    duplicate.reset();
    auto moved = std::move(a); CHECK(!a.locked() && moved.locked());
    if (reverse) {
        b.reset(); CHECK(locked(p) && locked(static_cast<char*>(p) + page));
        CHECK(!locked(static_cast<char*>(p) + 2 * page)); moved.reset();
    } else {
        moved.reset(); CHECK(!locked(p)); CHECK(locked(static_cast<char*>(p) + page)); b.reset();
    }
    CHECK(!locked(static_cast<char*>(p) + page));
    // Partial syscall failure must roll back only new pages, including across owners.
    core::EngineMemoryLock survivor(w, static_cast<char*>(p) + 2 * page, page);
    CHECK(mprotect(static_cast<char*>(p) + page, page, PROT_NONE) == 0);
    const auto failures = MemoryLock::statistics(MemoryOwner::ds4, c).failures;
    ds4::EngineMemoryLock failed(c, p, 3 * page);
    CHECK(!failed.locked() && locked(static_cast<char*>(p) + 2 * page) && !locked(p));
    CHECK(MemoryLock::statistics(MemoryOwner::ds4, c).failures == failures + 1);
    survivor.reset(); CHECK(mprotect(static_cast<char*>(p) + page, page, PROT_READ | PROT_WRITE) == 0);
    CHECK(munmap(p, 3 * page) == 0);
}
static void warnings() {
    FILE* f = std::tmpfile(); CHECK(f);
    const int saved = dup(STDERR_FILENO); CHECK(saved >= 0); CHECK(dup2(fileno(f), STDERR_FILENO) >= 0);
    for (int i = 0; i < 2; ++i) {
        core::EngineMemoryLock a(MemoryCategory::weights, nullptr, 1);
        ds4::EngineMemoryLock b(MemoryCategory::weights, nullptr, 1);
    }
    std::fflush(stderr); CHECK(dup2(saved, STDERR_FILENO) >= 0); close(saved);
    std::rewind(f); std::string text; char buf[1024];
    while (std::fgets(buf, sizeof buf, f)) text += buf;
    std::fclose(f);
    CHECK(text.find("strata mlock:") != std::string::npos && text.find("ds4 mlock:") != std::string::npos);
    size_t messages = 0, pos = 0;
    while ((pos = text.find("failed errno=", pos)) != std::string::npos) { ++messages; ++pos; }
    CHECK(messages == 2);
    for (auto owner : {MemoryOwner::core, MemoryOwner::ds4})
        CHECK(MemoryLock::failure_reason(owner, MemoryCategory::weights, EINVAL) == 2);
}
static void low_limit() {
    rlimit old{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &old) == 0);
    rlimit zero = old; zero.rlim_cur = 0; CHECK(setrlimit(RLIMIT_MEMLOCK, &zero) == 0);
    const auto cw = MemoryLock::statistics(MemoryOwner::core, MemoryCategory::weights).failures;
    const auto dw = MemoryLock::statistics(MemoryOwner::ds4, MemoryCategory::weights).failures;
    {
        core::WeightVector<char> a(4096, 17); ds4::WeightVector<char> b(4096, 19);
        core::ContextVector<char> c(4096, 23); ds4::ContextVector<char> d(4096, 29);
        CHECK(a[0] == 17 && b[0] == 19 && c[0] == 23 && d[0] == 29);
        CHECK(!locked(a.data()) && !locked(b.data()) && !locked(c.data()) && !locked(d.data()));
        CHECK(MemoryLock::statistics(MemoryOwner::core, MemoryCategory::weights).failures == cw + core::WeightLockPolicy::enabled());
        CHECK(MemoryLock::statistics(MemoryOwner::ds4, MemoryCategory::weights).failures == dw + ds4::WeightLockPolicy::enabled());
    }
    rlimit after{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &after) == 0 && after.rlim_cur == 0 && after.rlim_max == old.rlim_max);
    CHECK(setrlimit(RLIMIT_MEMLOCK, &old) == 0);
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::strcmp(argv[1], "ds4-first") == 0) { ds4::mlock_startup(); core::memory_lock_startup(); }
        else { core::memory_lock_startup(); ds4::mlock_startup(); }
        const size_t page = platform::memory_page_size();
        {
            core::WeightVector<char> cw(page); core::ContextVector<char> cc(page);
            ds4::WeightVector<char> dw(page); ds4::ContextVector<char> dc(page);
            CHECK(core::EngineMemoryLock::locked_bytes(MemoryCategory::weights) == (core::WeightLockPolicy::enabled() ? page : 0));
            CHECK(core::EngineMemoryLock::locked_bytes(MemoryCategory::context) == (core::ContextLockPolicy::enabled() ? page : 0));
            CHECK(ds4::EngineMemoryLock::locked_bytes(MemoryCategory::weights) == (ds4::WeightLockPolicy::enabled() ? page : 0));
            CHECK(ds4::EngineMemoryLock::locked_bytes(MemoryCategory::context) == (ds4::ContextLockPolicy::enabled() ? page : 0));
        }
        overlap(false); overlap(true); warnings(); low_limit();
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) threads.emplace_back([page] {
            for (int j = 0; j < 20; ++j) {
                core::WeightVector<char> a(page); ds4::ContextVector<char> b(page);
            }
        });
        for (auto& t : threads) t.join();
        for (auto owner : {MemoryOwner::core, MemoryOwner::ds4}) for (auto c : {MemoryCategory::weights, MemoryCategory::context})
            CHECK(MemoryLock::statistics(owner, c).bytes == 0);
        std::puts("independent owners and global overlap tests passed (real Linux syscalls)");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
