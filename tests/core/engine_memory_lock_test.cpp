#include "strata/core/pinned.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/ngram/ple_reader.hpp"
#include "src/core/conversation_copy.hpp"
#include "strata/core/checkpoint_allocation.hpp"
#include <cuda_runtime.h> // test-only injection control
#include <fstream>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
using namespace strata;
// Host copy-helper test: no GPU checkpoint DMA is launched by this fixture.
namespace strata::core { void conversation_checkpoint_join(const void*) {} }
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)

static size_t vm_locked() {
    std::ifstream f("/proc/self/status"); std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmLck:", 0) == 0) return std::stoull(line.substr(6)) * 1024;
    throw std::runtime_error("VmLck unavailable");
}
static size_t rollup_locked() {
    std::ifstream f("/proc/self/smaps_rollup"); std::string line;
    while (std::getline(f, line))
        if (line.rfind("Locked:", 0) == 0) return std::stoull(line.substr(7)) * 1024;
    throw std::runtime_error("smaps_rollup Locked unavailable");
}
static bool mapping_locked(const void* p) {
    std::ifstream f("/proc/self/smaps"); std::string line;
    bool selected = false;
    while (std::getline(f, line)) {
        unsigned long lo = 0, hi = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx", &lo, &hi) == 2)
            selected = uintptr_t(p) >= lo && uintptr_t(p) < hi;
        if (selected && line.rfind("VmFlags:", 0) == 0)
            return line.find(" lo ") != std::string::npos || line.ends_with(" lo");
    }
    throw std::runtime_error("mapping VmFlags unavailable");
}
template<class T> static void put(std::ofstream& f, T n) { f.write(reinterpret_cast<const char*>(&n), sizeof(n)); }
static size_t make_table(const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    put<uint32_t>(f, 0x46554747); put<uint32_t>(f, 3); put<uint64_t>(f, 1); put<uint64_t>(f, 0);
    const std::string name = "per_layer_token_embd.weight";
    put<uint64_t>(f, name.size()); f.write(name.data(), name.size());
    put<uint32_t>(f, 2); put<uint64_t>(f, 160); put<uint64_t>(f, 256);
    put<uint32_t>(f, 20); put<uint64_t>(f, 0); // real GGUF IQ4_NL tensor
    while (size_t(f.tellp()) % 32) put<uint8_t>(f, 0);
    const size_t offset = size_t(f.tellp());
    std::vector<char> data(256 * 90, 0); f.write(data.data(), data.size());
    CHECK(f.good()); return offset;
}
static void arenas() {
    const size_t page = platform::memory_page_size(), baseline = vm_locked();
    const bool enabled = core::WeightLockPolicy::enabled();
    setenv("STRATA_NO_LARGEPAGES", "1", 1);
    for (bool registered : {false, true}) {
        strata_test_register_success = registered;
        {
            core::PinnedArena weights(2 * page);
            CHECK(weights.valid());
            CHECK(weights.weight_lock.locked() == enabled);
            CHECK(vm_locked() == baseline + (enabled ? 2 * page : 0));
            CHECK(rollup_locked() == vm_locked());
            CHECK(mapping_locked(weights.base) == enabled);
            std::memset(weights.data(), 7, 2 * page);
        }
        CHECK(vm_locked() == baseline);
        {
            const std::vector<uint64_t> bounds{0, page, 2 * page};
            core::PinnedArena weights(2 * page, bounds, core::PinnedArena::Deferred{});
            CHECK(weights.valid()); CHECK(vm_locked() == baseline);
            std::atomic<int> ready{0}; weights.register_slices(ready);
            CHECK(weights.weight_lock.locked() == enabled);
            CHECK(vm_locked() == baseline + (enabled ? 2 * page : 0));
        }
        CHECK(vm_locked() == baseline);
        {
            core::PinnedArena weights(2 * page, core::PinnedArena::ByNeed{}, {{0, page}});
            CHECK(weights.valid()); CHECK(weights.weight_lock.locked() == enabled);
        }
        CHECK(vm_locked() == baseline);
    }
    // Same real MAP_SHARED allocation path used for shared expert arenas, including its header offset.
    char path[] = "/tmp/opencode/strata-shared-mlock-XXXXXX";
    const int fd = mkstemp(path); CHECK(fd >= 0); close(fd);
    {
        core::PinnedArena weights(2 * page, std::vector<uint64_t>{}, 0, path, 0x1234);
        CHECK(weights.valid() && weights.mapping_base != weights.base);
        CHECK(weights.weight_lock.locked() == enabled);
        CHECK(vm_locked() == baseline + (enabled ? 2 * page : 0)); // header not selected
    }
    unlink(path); CHECK(vm_locked() == baseline);
}
static void contexts() {
    const size_t page = platform::memory_page_size(), baseline = vm_locked();
    const bool enabled = core::ContextLockPolicy::enabled();
    {
        core::ConversationBuffer first, second;
        const size_t peak = first.allocation_peak(page);
        first.resize(page, 3); second.resize(page, 5);
        CHECK(first.bytes() <= peak);
        core::ConversationCheckpoint checkpoint; checkpoint.gdn.assign(page, 7);
        CHECK(platform::MemoryLock::locked_bytes(platform::MemoryCategory::context) == (enabled ? 3 * page : 0));
        CHECK(vm_locked() == baseline + (enabled ? 3 * page : 0));
        first.visit(0, 1, [&](const uint8_t* p, size_t, size_t) {
            CHECK(mapping_locked(p) == enabled); return true;
        });
        CHECK(rollup_locked() == vm_locked());
        std::printf("main proc fixture: weights=%d context=%d VmLck=%zu smaps_rollup.Locked=%zu bytes\n",
                    int(core::WeightLockPolicy::enabled()), int(enabled), vm_locked(), rollup_locked());
        auto saved = first; auto saved_checkpoint = checkpoint;
        const size_t grow_peak = first.allocation_peak(2 * page);
        first.resize(2 * page, 9);
        CHECK(first.bytes() <= grow_peak);
        CHECK(platform::MemoryLock::locked_bytes(platform::MemoryCategory::context) > 0 || !enabled);
        first = saved; checkpoint = saved_checkpoint;
        CHECK(first == saved && checkpoint.gdn == saved_checkpoint.gdn);
        core::ConversationBuffer restored(std::move(second));
        CHECK(restored.size() == page && second.empty());
        restored.resize(0); first.resize(0);
    }
    {
        // Actual checkpoint-pool copy helper: reuse larger backing, then grow it.
        core::ConversationCheckpoint pooled, source;
        pooled.gdn.reserve(3 * page); pooled.ple.reserve(page);
        source.gdn.assign(page + 1, 23); source.ple.assign(17, 29);
        source.stage_parts.resize(1); source.stage_parts[0].dead.assign(7, 31);
        auto* backing = pooled.gdn.data();
        const auto bound = core::checkpoint_copy_bound(pooled, source);
        core::conversation_detail::copy_checkpoint_into(pooled, source);
        CHECK(pooled.bytes() <= bound.resident);
        CHECK(pooled.gdn.data() == backing && pooled.gdn == source.gdn);
        CHECK(pooled.bytes() >= core::context_allocation_bytes(3 * page) +
              core::context_allocation_bytes(page) + core::context_allocation_bytes(7));
        source.gdn.resize(4 * page, 37);
        const size_t held = pooled.bytes();
        const auto growth = core::checkpoint_copy_bound(pooled, source);
        core::conversation_detail::copy_checkpoint_into(pooled, source);
        CHECK(pooled.bytes() <= growth.resident);
        CHECK(growth.additional >= core::context_allocation_bytes(4 * page));
        CHECK(core::memory_bytes_sum({held, growth.additional}) >= pooled.bytes());
        CHECK(pooled.gdn == source.gdn && pooled.stage_parts[0].dead == source.stage_parts[0].dead);
        CHECK((platform::MemoryLock::locked_bytes(platform::MemoryCategory::context) > 0) == enabled);
    }
    {
        // Reproduced review mismatch: a logical 4 KiB checkpoint receives 64 KiB
        // pooled storage. Include zero-length retained payloads and nested directories.
        core::ConversationCheckpoint source, pooled;
        source.gdn.resize(page, 7); source.stage_parts.resize(3);
        source.stage_parts[1].ple.resize(1, 9);
        pooled.gdn.reserve(16 * page); pooled.ple.reserve(2 * page);
        pooled.stage_parts.reserve(7);
        const auto bound = core::checkpoint_copy_bound(pooled, source);
        CHECK(bound.resident >= core::context_allocation_bytes(16 * page) + core::context_allocation_bytes(2 * page));
        core::conversation_detail::copy_checkpoint_into(pooled, source);
        CHECK(pooled.bytes() == bound.resident);
        CHECK(pooled.ple.empty() && pooled.ple.capacity() >= 2 * page);
        std::vector<core::ConversationCheckpoint> copies{source}, storage, moved;
        storage.push_back(std::move(pooled)); storage.emplace_back();
        storage.back().gdn.reserve(32 * page);
        moved.emplace_back(); moved.back().gdn.reserve(8 * page);
        const core::CheckpointShape live{0, 0, page, 0, 0, 0, 0};
        const auto plan = core::checkpoint_capture_bound(live, copies, storage, moved);
        core::SavedConversation image;
        image.live = std::move(storage.back()); storage.pop_back(); image.live.gdn.resize(page);
        image.checkpoints = core::conversation_detail::copy_checkpoints_beating(copies, &storage, moved.size());
        image.checkpoints.push_back(std::move(moved.back()));
        CHECK(image.bytes() == plan.resident);
        // No new payload for the live, pooled GDN/PLE or moved allocation.
        CHECK(plan.additional < core::context_allocation_bytes(8 * page));
        const auto checkpoint_bytes = image.bytes();
        core::ConversationCache exact(checkpoint_bytes, 1), short_budget(checkpoint_bytes - 1, 1);
        CHECK(exact.make_room(plan.resident, 0)); CHECK(!short_budget.make_room(plan.resident, 0));
        core::SavedConversation staged;
        staged.stage_images.reserve(3);
        for (size_t i = 0; i < 3; ++i) staged.stage_images.push_back(image);
        CHECK(staged.bytes() <= core::memory_bytes_sum({3 * sizeof(core::SavedConversation),
              plan.resident, plan.resident, plan.resident}));
        std::vector<core::ConversationKv> retained(1);
        retained[0].k.resize(17, 3);
        const size_t target = 2 * core::ConversationBuffer::segment_bytes + 3;
        const size_t items = core::park_capture_item_bound(retained, target, retained.size());
        const size_t peak = retained[0].k.allocation_peak(target);
        retained[0].k.resize(target, 7);
        CHECK(retained[0].k.bytes() <= peak);
        size_t actual_items = 0;
        retained[0].k.visit(0, target, [&](const uint8_t*, size_t, size_t) { ++actual_items; return true; });
        CHECK(items >= actual_items);
        CHECK(core::checkpoint_array_bound(SIZE_MAX, 0, 2, false).resident == SIZE_MAX);
    }
    CHECK(vm_locked() == baseline);
}
static void legacy_arenas() {
    const size_t baseline = vm_locked(), page = platform::memory_page_size();
    strata_test_register_success = false;
    for (const char* legacy : {"0", "1"}) {
        setenv("STRATA_ARENA_LOCK", legacy, 1);
        setenv("STRATA_ARENA_PIN_GIB", "0", 1);
        const bool expected = core::WeightLockPolicy::enabled() || *legacy == '1';
        {
            core::PinnedArena weights(2 * page);
            CHECK(weights.valid());
            CHECK(mapping_locked(weights.base) == expected);
            CHECK(vm_locked() == baseline + (expected ? 2 * page : 0));
        }
        CHECK(vm_locked() == baseline);
        {
            core::PinnedArena weights(2 * page, std::vector<uint64_t>{0, page, 2 * page}, core::PinnedArena::Deferred{});
            std::atomic<int> ready{0}; weights.register_slices(ready);
            CHECK(mapping_locked(weights.base) == expected);
            CHECK(vm_locked() == baseline + (expected ? 2 * page : 0));
        }
        CHECK(vm_locked() == baseline);
        {
            core::PinnedArena weights(2 * page, core::PinnedArena::ByNeed{}, {{0, page}});
            CHECK(mapping_locked(weights.base) == expected);
            CHECK(vm_locked() == baseline + (expected ? 2 * page : 0));
        }
        CHECK(vm_locked() == baseline);
        setenv("STRATA_ARENA_PIN_GIB", "none", 1);
        {
            core::PinnedArena weights(2 * page);
            CHECK(mapping_locked(weights.base) == expected);
            CHECK(vm_locked() == baseline + (expected ? 2 * page : 0));
        }
        CHECK(vm_locked() == baseline);
    }
    setenv("STRATA_ARENA_PIN_GIB", "0", 1);
    setenv("STRATA_ARENA_LOCK", "0", 1);
}
static void ple(const std::string& path, size_t offset) {
    const size_t baseline = vm_locked();
    const bool enabled = core::WeightLockPolicy::enabled();
    rlimit before{}, after{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &before) == 0);
    {
        kernels::PleTable table; kernels::PleIoOptions io; io.mode = kernels::PleIo::Mmap;
        std::string error; CHECK(table.open(path, error, io));
        CHECK(table.locked() == enabled);
        CHECK((vm_locked() > baseline) == enabled);
        if (enabled) CHECK(!table.unlock()); // automatic give-back must not undo explicit weight residency
        table.close(); CHECK(vm_locked() == baseline);
        io.lock = true; CHECK(table.open(path, error, io));
        CHECK(table.locked()); // io.lock is honored for BOTH zero and unset
        CHECK(table.unlock() == !enabled);
        CHECK(table.locked() == enabled);
        table.close(); CHECK(vm_locked() == baseline);
    }
    {
        ngram::PleReader reader; std::string error;
        CHECK(reader.open(path, offset, 256, 2, 32, error, false, 90));
        CHECK((platform::MemoryLock::locked_bytes(platform::MemoryCategory::weights) > 0) == enabled);
        uint32_t row = 0; uint8_t bytes[90]{};
        const auto ticket = reader.issue(&row, 1, bytes);
        CHECK(reader.collect(ticket, error));
        reader.close(); CHECK(vm_locked() == baseline);
    }
    CHECK(getrlimit(RLIMIT_MEMLOCK, &after) == 0);
    CHECK(before.rlim_cur == after.rlim_cur && before.rlim_max == after.rlim_max);
}
static void ple_limit_compatibility(const std::string& path, size_t offset) {
    rlimit parent{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &parent) == 0);
    const size_t page = platform::memory_page_size();
    for (bool low_hard : {false, true}) for (int mode = 0; mode < 3; ++mode) {
        const pid_t pid = fork(); CHECK(pid >= 0);
        if (pid == 0) {
            try {
                // Only this short-lived child lowers limits; the shell and parent are untouched.
                rlimit limit = parent; limit.rlim_cur = 0;
                if (low_hard) limit.rlim_max = std::min<rlim_t>(limit.rlim_max, page);
                CHECK(setrlimit(RLIMIT_MEMLOCK, &limit) == 0);
                kernels::PleIoOptions io;
                io.mode = mode == 2 ? kernels::PleIo::Direct : kernels::PleIo::Mmap;
                io.lock = mode != 0; io.cache_rows = 0;
                kernels::PleTable table; std::string error;
                CHECK(table.open(path, error, io));
                const bool legacy_raise = !core::WeightLockPolicy::enabled() && mode == 1;
                const rlim_t span = rlim_t(offset % page + 256 * 90);
                const rlim_t expected = legacy_raise ? std::min<rlim_t>(limit.rlim_max, span + (rlim_t(1) << 30)) : 0;
                rlimit after{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &after) == 0);
                CHECK(after.rlim_cur == expected && after.rlim_max == limit.rlim_max);
                const rlim_t rounded = (span + page - 1) / page * page;
                CHECK(table.locked() == (legacy_raise && expected >= rounded));
                if (table.locked()) CHECK(table.unlock()); // legacy give-back still works
                table.close(); CHECK(vm_locked() == 0);
                // Enabled low-limit paths must report failure, not successful residency.
                if (core::WeightLockPolicy::enabled() && mode != 2)
                    CHECK(platform::MemoryLock::statistics(platform::MemoryOwner::core,
                        platform::MemoryCategory::weights).failures > 0);
                _exit(0);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "PLE limit child: %s\n", e.what()); _exit(1);
            }
        }
        int status = 0; CHECK(waitpid(pid, &status, 0) == pid);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        rlimit after{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &after) == 0);
        CHECK(after.rlim_cur == parent.rlim_cur && after.rlim_max == parent.rlim_max);
    }
}
static void failure() {
    rlimit old{}; CHECK(getrlimit(RLIMIT_MEMLOCK, &old) == 0);
    rlimit zero = old; zero.rlim_cur = 0; CHECK(setrlimit(RLIMIT_MEMLOCK, &zero) == 0);
    strata_test_register_success = false;
    {
        core::PinnedArena weights(4096); CHECK(weights.valid() && !weights.weight_lock.locked());
        core::ConversationBuffer context; context.resize(4096, 19);
        CHECK(platform::MemoryLock::locked_bytes(platform::MemoryCategory::weights) == 0);
        CHECK(platform::MemoryLock::locked_bytes(platform::MemoryCategory::context) == 0);
        uint8_t value = 0; CHECK(context.read(&value, 0, 1) && value == 19);
    }
    CHECK(setrlimit(RLIMIT_MEMLOCK, &old) == 0);
}
int main() {
    char path[] = "/tmp/opencode/strata-ple-mlock-XXXXXX";
    try {
        core::memory_lock_startup();
        const auto requested = [](const char* name) {
            const char* value = std::getenv(name);
            return value && std::strcmp(value, "1") == 0;
        };
        CHECK(core::WeightLockPolicy::enabled() == requested("STRATA_MLOCK_WEIGHTS"));
        CHECK(core::ContextLockPolicy::enabled() == requested("STRATA_MLOCK_CONTEXT"));
        CHECK(core::legacy_weight_lock_allowed() == !requested("STRATA_MLOCK_WEIGHTS"));
        // Ignore pre-existing legacy lock defaults for the unset-policy test; the new flags are tested separately.
        setenv("STRATA_ARENA_LOCK", "0", 1);
        arenas(); legacy_arenas(); contexts();
        int fd = mkstemp(path); CHECK(fd >= 0); close(fd);
        const size_t offset = make_table(path); ple(path, offset); ple_limit_compatibility(path, offset); unlink(path);
        failure(); platform::MemoryLock::report();
        CHECK(vm_locked() == 0);
        std::puts("main-engine allocation locking tests passed (GPU registration mocked, Linux mlock real)");
        return 0;
    } catch (const std::exception& e) {
        unlink(path); std::fprintf(stderr, "engine mlock test failed: %s\n", e.what()); return 1;
    }
}
