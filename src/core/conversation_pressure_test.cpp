// CPU only: RAM pressure is injected through fake readings; one real check that an evicted image's RAM leaves the
// process (Windows private bytes / Linux RSS after malloc_trim).
#include "strata/core/conversation_pressure.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t MiB = 1ull << 20;

// a parked conversation of `kv_mib` MiB of K/V; ids distinct per `tag`, so put() supersedes nothing
SavedConversation image(int32_t tag, size_t kv_mib, bool pinned = false) {
    SavedConversation s;
    s.live.ids = {tag, tag + 1, tag + 2};
    s.kv.resize(1);
    s.kv[0].k.resize(kv_mib * MiB, (uint8_t) tag);
    if (pinned) {
        ConversationCheckpoint c;
        c.ids = {tag, tag + 1};
        c.pinned = true;
        s.checkpoints.push_back(std::move(c));
    }
    return s;
}

// is image(tag) still parked (a prompt that goes on past its tokens finds it)
bool parked(const ConversationCache& cache, int32_t tag) {
    const std::vector<int32_t> ids = {tag, tag + 1, tag + 2, -1};
    return cache.best(ids, {}, true).tokens == 3;
}

// The host: `free_now` is what a reading shows; freed bytes land in `pending` and show up at settle() (the allocator's
// freed heap).
struct FakeHost {
    uint64_t free_now = 0, pending = 0;
    size_t samples = 0, settles = 0;
    bool unknown = false;
    std::optional<uint64_t> sample() { ++samples; if (unknown) return std::nullopt; return free_now; }
    void settle() { ++settles; free_now += pending; pending = 0; }
};

std::optional<uint64_t> process_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pm{};
    pm.cb = sizeof pm;
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &pm, sizeof pm)) return std::nullopt;
    return (uint64_t) pm.PrivateUsage;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t size = 0, resident = 0;
    if (!(statm >> size >> resident)) return std::nullopt;
    return resident * (uint64_t) sysconf(_SC_PAGESIZE);
#else
    return std::nullopt;
#endif
}
} // namespace

int main() {
    const uint64_t floor = 100 * MiB, hyst = 16 * MiB;
    // a cache of `n` entries of `each` MiB, oldest first (tags 10, 20, ...), wired to a fake host
    auto fill = [](ConversationCache& cache, int n, size_t each) {
        for (int i = 0; i < n; ++i) check(cache.put(image(10 * (i + 1), each)), "fixture parked");
    };
    auto run = [&](ConversationCache& cache, FakeHost& host, size_t bound = SIZE_MAX) {
        return conversation_pressure_release(
            floor, hyst, [&] { return host.sample(); }, [&] { host.settle(); },
            [&]() -> std::optional<PressureUnit> {
                auto u = conversation_cache_release_one(cache);
                if (u) host.pending += u->bytes;
                return u;
            },
            bound);
    };
    {   // above the floor: nothing happens, one reading, no settle
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 4, 8);
        FakeHost host{floor};
        const auto r = run(cache, host);
        check(!r.below && r.released() == 0 && cache.size() == 4, "at the floor exactly: nothing released");
        check(host.samples == 1 && host.settles == 0, "the cheap path reads once and settles nothing");
        host.free_now = floor + 1;
        check(run(cache, host).released() == 0, "above the floor: nothing released");
    }
    {   // inside the hysteresis band (above the floor, below floor + hysteresis): not triggered
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 4, 8);
        FakeHost host{floor + hyst / 2};
        check(run(cache, host).released() == 0 && cache.size() == 4, "the band above the floor does not trigger");
    }
    {   // below the floor: the oldest go until floor + hysteresis is free
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 6, 8);   // 6 x ~8 MiB
        FakeHost host{floor - 4 * MiB};
        const auto r = run(cache, host);
        // need 20 MiB back: three entries of ~8 MiB
        check(r.below && r.dialogs == 3 && r.others == 0, "three parked conversations released");
        check(r.after && *r.after >= floor + hyst, "released until floor + hysteresis");
        check(r.before && *r.before == floor - 4 * MiB, "the first reading is reported");
        check(cache.size() == 3 && cache.evictions() == 3, "three left");
        for (int32_t t = 10; t <= 60; t += 10) check(parked(cache, t) == (t >= 40), "the oldest went, the newest stay");
        check(r.bytes >= 3 * 8 * MiB && r.bytes < 3 * 9 * MiB, "the released bytes are the entries' bytes");
        check(!r.exhausted, "not exhausted");
        // the next check sees enough RAM: nothing more goes
        check(run(cache, host).released() == 0 && cache.size() == 3, "back above: nothing more released");
    }
    {   // the freed heap alone brings RAM back: nothing is evicted
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 3, 8);
        FakeHost host{floor - MiB};
        host.pending = 2 * hyst;
        const auto r = run(cache, host);
        check(r.below && r.released() == 0 && cache.size() == 3 && host.settles == 1, "settling first can be enough");
    }
    {   // the retained K/V goes before any parked conversation
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 3, 8);
        std::vector<ConversationKv> kv(1);
        kv[0].k.resize(30 * MiB, 1);
        cache.retain(std::move(kv), 100);
        check(cache.retained_bytes() >= 30 * MiB, "fixture retained");
        FakeHost host{floor - 4 * MiB};
        const auto r = run(cache, host);
        check(r.others == 1 && r.dialogs == 0 && cache.size() == 3 && cache.retained_bytes() == 0,
              "the retained K/V alone was enough");
    }
    {   // pinned shared prefixes stay; with only them left the event is exhausted
        ConversationCache cache(1ull << 30, 8);
        check(cache.put(image(10, 8, true)), "pinned fixture");
        check(cache.put(image(20, 8)), "fixture");
        check(cache.put(image(30, 8, true)), "pinned fixture");
        FakeHost host{floor - 50 * MiB};
        const auto r = run(cache, host);
        check(r.dialogs == 1 && r.exhausted && cache.size() == 2, "only the unpinned one went");
        check(parked(cache, 10) && !parked(cache, 20) && parked(cache, 30), "the pinned ones stay");
    }
    {   // an unknown reading releases nothing; a reading lost midway stops the release
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 4, 8);
        FakeHost host{0};
        host.unknown = true;
        check(run(cache, host).released() == 0 && cache.size() == 4 && host.settles == 0, "unknown: nothing released");
        host.unknown = false;
        size_t n = 0;
        const auto r = conversation_pressure_release(
            floor, hyst, [&]() -> std::optional<uint64_t> { return ++n > 2 ? std::nullopt : host.sample(); },
            [&] { host.settle(); }, [&] { return conversation_cache_release_one(cache); }, SIZE_MAX);
        check(r.below && r.released() == 1 && cache.size() == 3, "a reading lost midway stops after that step");
    }
    {   // the bound
        ConversationCache cache(1ull << 30, 8);
        fill(cache, 5, 8);
        FakeHost host{0};
        check(run(cache, host, 2).released() == 2 && cache.size() == 3, "at most `bound` units");
    }
    {   // the RAM really leaves the process: 256 MiB parked, released, given back
        ConversationCache cache(1ull << 31, 8);
        // glibc: a first round of 16 MiB chunks raises the mmap threshold, so the second one lives in the heap and
        // only malloc_trim gives it back (the case the engine meets after its first evictions)
        for (int round = 0; round < 2; ++round) {
            check(cache.put(image(10 + round, 256)), "large fixture parked");
            const auto held = process_bytes();
            const auto u = conversation_cache_release_one(cache);
            conversation_return_freed_heap();
            const auto after = process_bytes();
            check(u && u->dialog && u->bytes >= 256 * MiB, "the large entry released");
            if (held && after)
                check(*held > *after && *held - *after >= 240 * MiB, "its RAM left the process");
            if (held && after)
                std::printf("round %d: process %llu -> %llu MiB\n", round, (unsigned long long) (*held >> 20),
                            (unsigned long long) (*after >> 20));
        }
    }
    std::printf("conversation_pressure_test: %d checks passed\n", checks);
}
