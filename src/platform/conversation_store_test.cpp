#include "strata/platform/conversation_store.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#ifdef CONVERSATION_STORE_TEST_FAULTS
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
namespace {
int fail_sync = 0;
bool fail_write = false, fail_rename = false, crash_write = false, crash_rename = false;
std::filesystem::path observed_directory;
uint64_t peak_bytes = 0;
size_t peak_entries = 0;
}
extern "C" int __real_fsync(int);
extern "C" size_t __real_fwrite(const void*, size_t, size_t, FILE*);
extern "C" int __real_rename(const char*, const char*);
extern "C" int __wrap_fsync(int fd) {
    if (fail_sync && --fail_sync == 0) { errno = EIO; return -1; }
    return __real_fsync(fd);
}
extern "C" size_t __wrap_fwrite(const void* data, size_t width, size_t count, FILE* file) {
    if (fail_write) { errno = ENOSPC; return 0; }
    const auto result = __real_fwrite(data, width, count, file);
    if (!observed_directory.empty()) {
        std::fflush(file);
        uint64_t bytes = 0; size_t entries = 0;
        for (const auto& entry : std::filesystem::directory_iterator(observed_directory)) {
            if (entry.path().extension() == ".snap" || entry.path().extension() == ".tmp") {
                bytes += entry.file_size(); ++entries;
            }
        }
        peak_bytes = std::max(peak_bytes, bytes); peak_entries = std::max(peak_entries, entries);
    }
    if (crash_write) { std::fflush(file); _exit(73); }
    return result;
}
extern "C" int __wrap_rename(const char* from, const char* to) {
    if (fail_rename) { errno = EIO; return -1; }
    const int result = __real_rename(from, to);
    if (crash_rename) _exit(result == 0 ? 74 : 75);
    return result;
}
#endif

using namespace strata::core;
using namespace strata::platform;
namespace fs = std::filesystem;
namespace {
int checks = 0;
int heartbeats = 0;
void heartbeat() noexcept { ++heartbeats; }
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct TempDirectory {
    fs::path path;
    TempDirectory() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned i = 0; i < 100; ++i) {
            auto candidate = fs::temp_directory_path() / ("strata-store-test-" + std::to_string(tick) + "-" + std::to_string(i));
            if (fs::create_directory(candidate)) { path = std::move(candidate); return; }
        }
        check(false, "create temporary directory");
    }
    ~TempDirectory() { std::error_code ec; fs::remove_all(path, ec); }
};
SavedConversation fixture(int32_t branch, size_t length = 3) {
    SavedConversation s;
    for (size_t i = 0; i < length; ++i) s.live.ids.push_back(branch + int32_t(i));
    s.live.gdn = {0, 255, uint8_t(branch)};
    s.live.ple = {1, 2, 3}; s.live.tails = {4, 5}; s.live.dead = {6}; s.live.block_pos = {7};
    s.checkpoints = {s.live}; s.checkpoints[0].ids.resize(1);
    ConversationKv kv;
    kv.format = 3; kv.cells = 3; kv.heads = 1; kv.head_dim = 64;
    kv.k.resize(192, 42); kv.v.resize(96, 17); kv.k_scale = {0, 127};
    s.kv.push_back(std::move(kv));
    return s;
}
std::string encoded(const SavedConversation& image) {
    std::ostringstream stream;
    std::string error;
    check(conversation_file_write(stream, image, {}, error), "encode comparison image");
    return stream.str();
}
ConversationStore::Candidate best(ConversationStore& store, int32_t branch, size_t length = 4,
                                  const std::vector<fs::path>& excluded = {}) {
    std::vector<int64_t> prompt;
    for (size_t i = 0; i < length; ++i) prompt.push_back(branch + int32_t(i));
    ConversationStore::Candidate candidate;
    std::string error;
    check(store.best(prompt, {}, true, 1 << 20, excluded, candidate, error), "find disk prefix");
    return candidate;
}
size_t count_files(const fs::path& root, const char* extension) {
    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(root / "strata-conversations-v2"))
        if (entry.path().extension() == extension) ++count;
    return count;
}
uint64_t disk_bytes(const fs::path& root) {
    uint64_t size = 0;
    for (const auto& entry : fs::directory_iterator(root / "strata-conversations-v2"))
        if (entry.path().extension() == ".snap" || entry.path().extension() == ".tmp") size += entry.file_size();
    return size;
}
} // namespace

int main() {
    TempDirectory temp;
    ConversationStore store;
    std::string error;
    check(store.open(temp.path / "disabled", {}, 0, 4, error, heartbeat) && !store.is_open(), "zero bytes disables store");
    check(heartbeats == 0, "disabled store reports no I/O progress");
    check(!fs::exists(temp.path / "disabled"), "disabled store performs no filesystem creation");
    check(store.open(temp.path / "disabled", {}, 1000, 0, error) && !store.is_open(), "zero slots disables store");
    check(!fs::exists(temp.path / "disabled"), "zero slots performs no filesystem creation");
    check(!store.put(fixture(1), error), "closed store declines writes");
    check(best(store, 1).match.tokens == 0, "closed store has no candidates");

    const auto root = temp.path / "cache";
    const auto a = fixture(10), b = fixture(20), c = fixture(30);
    const uint64_t size = encoded(a).size();
    check(store.open(root, {}, size * 3, 3, error, heartbeat) && store.is_open(), "open bounded cache");
    ConversationStore competing;
    check(!competing.open(root, {}, size * 3, 3, error) && !competing.is_open(), "second writer is refused");
    heartbeats = 0;
    check(store.put(a, error), "spill first image");
    check(heartbeats > 0, "store forwards write progress");
    check(count_files(root, ".snap") == 1 && count_files(root, ".tmp") == 0, "publication leaves only complete file");
    heartbeats = 0;
    const auto selected = best(store, 10);
    check(heartbeats > 0, "store forwards prefix-scan progress");
    check(selected.match.tokens == 3 && selected.match.live, "live prefix selected");
    check(best(store, 10, 2).match.tokens == 1 && !best(store, 10, 2).match.live, "early checkpoint selected");
    check(best(store, 90).match.tokens == 0, "unrelated prompt misses");
    SavedConversation image;
    heartbeats = 0;
    check(store.read(selected, 1 << 20, 1 << 21, 0, image, error) && encoded(image) == encoded(a), "read preserves entire shared image");
    check(heartbeats > 0, "store forwards read progress");
    check(!store.read(selected, 1 << 20, std::nullopt, 0, image, error), "unknown RAM declines disk staging");
    check(!store.read(selected, selected.match.staging_bytes - 1, 1 << 21, 0, image, error), "staging budget enforced again on read");
    check(!store.read(selected, 1 << 20, selected.match.staging_bytes + 9, 10, image, error), "physical floor applies to staged read");
    check(encoded(image) == encoded(a), "failed reads leave caller image unchanged");
    check(store.put(b, error) && store.put(c, error), "fill entry quota");
    const auto time = fs::file_time_type::clock::now();
    fs::last_write_time(selected.path, time - std::chrono::hours(3));
    fs::last_write_time(best(store, 20).path, time - std::chrono::hours(2));
    fs::last_write_time(best(store, 30).path, time - std::chrono::hours(1));
    check(store.touch(selected, error), "successful promotion updates LRU time");
    check(store.put(fixture(40), error), "evict to admit fourth image");
    check(best(store, 20).match.tokens == 0 && best(store, 10).match.tokens == 3, "least recently used snapshot evicted");
    check(disk_bytes(root) <= size * 3 && count_files(root, ".snap") == 3, "retention bounded by bytes and slots");
    auto huge = fixture(50); huge.live.gdn.resize(size * 4);
    check(!store.put(huge, error), "oversized spill rejected");
    check(best(store, 10).match.tokens == 3 && count_files(root, ".snap") == 3, "oversize rejection does not evict existing entries");
    store.close();
    check(competing.open(root, {}, size * 3, 3, error), "lock released on normal close");
    check(best(competing, 10).match.tokens == 3, "snapshot discovered after reopen");
    competing.close();
    check(store.open(root, {}, size * 2, 8, error), "reopen with tighter byte quota");
    check(disk_bytes(root) <= size * 2 && count_files(root, ".snap") == 2, "startup enforces changed byte quota");
    store.close();
    check(store.open(root, {}, size * 8, 1, error), "reopen with tighter entry quota");
    check(count_files(root, ".snap") == 1, "startup enforces changed entry quota");

    const auto other_root = temp.path / "foreign";
    ConversationIdentity foreign{}; foreign.digest[0] = 1;
    ConversationStore other;
    check(other.open(other_root, foreign, size * 3, 3, error) && other.put(a, error), "write foreign identity");
    other.close();
    check(other.open(other_root, {}, size * 3, 3, error), "open directory with different model identity");
    check(best(other, 10).match.tokens == 0, "foreign model has no candidate");
    check(other.put(b, error), "new model can use same bounded directory");
    check(count_files(other_root, ".snap") == 2 && disk_bytes(other_root) == size * 2, "quota includes foreign entries");
    const auto valid = best(other, 20);
    auto impostor = valid; impostor.path = selected.path;
    check(!other.read(impostor, 1 << 20, 1 << 21, 0, image, error), "candidate outside owned directory refused");
    check(!other.touch(impostor, error), "cannot change other directory's retention time");
    {
        std::fstream file(valid.path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekg(-1, std::ios::end); char bad = 0; file.read(&bad, 1); bad ^= 1;
        file.seekp(-1, std::ios::end); file.write(&bad, 1);
    }
    check(!other.read(valid, 1 << 20, 1 << 21, 0, image, error), "corrupt footer refused before promotion");
    check(other.put(fixture(20, 2), error), "write shorter valid candidate");
    check(best(other, 20, 4, {valid.path}).match.tokens == 2, "failed candidate exclusion exposes next valid prefix");

    const auto interrupted = other_root / "strata-conversations-v2" / (std::string(32, 'a') + ".tmp");
    const auto unrelated = other_root / "strata-conversations-v2" / "user-notes.tmp";
    { std::ofstream f(interrupted); f << "incomplete snapshot"; }
    { std::ofstream f(unrelated); f << "leave this file alone"; }
    other.close();
    check(other.open(other_root, {}, size * 3, 3, error), "recover interrupted write directory");
    check(!fs::exists(interrupted) && fs::exists(unrelated), "cleanup removes only managed temporary files");

    const auto eviction_root = temp.path / "eviction";
    ConversationStore eviction_store;
    check(eviction_store.open(eviction_root, {}, size * 4, 4, error), "prepare RAM eviction tier");
    struct SpillState {
        ConversationStore* store;
        const ConversationStore::Candidate* protected_entry = nullptr;
        size_t calls = 0, failed = 0;
    } state{&eviction_store};
    const auto spill = [](void* user, const SavedConversation& snapshot) noexcept {
        auto& state = *static_cast<SpillState*>(user);
        ++state.calls;
        try {
            std::string error;
            if (!state.store->put(snapshot, error, state.protected_entry)) ++state.failed;
        } catch (...) { ++state.failed; }
    };
    {
        ConversationCache ram(1 << 20, 1, spill, &state);
        auto first = a, second = b;
        check(ram.put(std::move(first)), "park initial RAM image");
        check(state.calls == 0 && best(eviction_store, 10).match.tokens == 0, "parking alone performs no disk write");
        check(ram.put(std::move(second)), "RAM slot eviction triggers spill");
        check(state.calls == 1 && state.failed == 0 && best(eviction_store, 10).match.tokens == 3,
              "only evicted image reaches disk");
        check(best(eviction_store, 20).match.tokens == 0, "retained RAM image stays off disk");
        eviction_store.close();
        auto third = c;
        check(ram.put(std::move(third)), "failed disk spill does not stop RAM insertion");
        check(state.failed == 1 && ram.size() == 1 && ram.evictions() == 2,
              "failed spill still drops evicted image and preserves RAM bounds");
    }
    check(state.calls == 2, "RAM destruction does not persist a continuing or active turn");

    const auto protected_root = temp.path / "protected";
    ConversationStore protected_store;
    check(protected_store.open(protected_root, {}, size, 1, error) && protected_store.put(a, error), "prepare single-entry protected cache");
    const auto protected_hit = best(protected_store, 10);
    check(!protected_store.put(b, error, &protected_hit), "spill declined when only quota victim is protected");
    check(protected_store.read(protected_hit, 1 << 20, 1 << 21, 0, image, error), "selected image survives admission-driven spill");
    check(best(protected_store, 20).match.tokens == 0 && disk_bytes(protected_root) == size, "protected decline preserves disk quota");
    SpillState admission{&protected_store, &protected_hit};
    ConversationCache admission_ram(protected_hit.match.staging_bytes, 2, spill, &admission);
    auto parked_b = b;
    check(admission_ram.put(std::move(parked_b)), "retain RAM branch before disk promotion");
    check(admission_ram.make_staging_room(protected_hit.match.staging_bytes), "reserve full disk staging by evicting RAM branch");
    check(admission.calls == 1 && admission.failed == 1 && admission_ram.bytes() == 0,
          "protected disk hit makes conflicting spill fail without breaking RAM admission");
    check(protected_store.read(protected_hit, protected_hit.match.staging_bytes, 1 << 21, 0, image, error),
          "protected image loads after RAM admission eviction");
    check(admission_ram.bytes() + image.bytes() <= protected_hit.match.staging_bytes, "RAM plus staged image respects shared budget");
    admission.protected_entry = nullptr;
    check(protected_store.put(b, error), "entry becomes evictable after protection ends");
    check(best(protected_store, 10).match.tokens == 0 && best(protected_store, 20).match.tokens == 3, "normal eviction resumes");

#ifdef CONVERSATION_STORE_TEST_FAULTS
    const auto quota_root = temp.path / "quota";
    ConversationStore quota;
    check(quota.open(quota_root, {}, size * 3, 3, error), "prepare quota observation");
    observed_directory = quota_root / "strata-conversations-v2";
    for (int32_t branch : {10, 20, 30, 40, 50}) check(quota.put(fixture(branch), error), "write while observing temporary disk usage");
    observed_directory.clear();
    check(peak_bytes == size * 3 && peak_entries == 3, "quota holds during writes including temporary file, not only afterward");
    quota.close();

    const auto fault_root = temp.path / "faults";
    ConversationStore faults;
    check(faults.open(fault_root, {}, size * 4, 4, error) && faults.put(a, error), "prepare failure fixture");
    for (int fault = 0; fault < 4; ++fault) {
        fail_write = fault == 0; fail_sync = fault == 1 ? 1 : fault == 3 ? 2 : 0; fail_rename = fault == 2;
        const bool ok = faults.put(b, error);
        fail_write = false; fail_sync = 0; fail_rename = false;
        check(!ok, "write/file-sync/rename/directory-sync failure reported");
        check(count_files(fault_root, ".snap") == 1 && count_files(fault_root, ".tmp") == 0, "failed publication cleaned up");
        check(best(faults, 10).match.tokens == 3 && best(faults, 20).match.tokens == 0, "failed publication preserves earlier complete image");
    }
    faults.close();
    for (bool after_rename : {false, true}) {
        const pid_t child = fork();
        check(child >= 0, "fork crash fixture");
        if (child == 0) {
            ConversationStore crashed;
            if (!crashed.open(fault_root, {}, size * 4, 4, error)) _exit(70);
            crash_write = !after_rename; crash_rename = after_rename;
            crashed.put(b, error);
            _exit(71);
        }
        int status = 0;
        check(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == (after_rename ? 74 : 73),
              "child terminated at chosen publication boundary");
        check(faults.open(fault_root, {}, size * 4, 4, error), "kernel releases lock after crash");
        check(count_files(fault_root, ".tmp") == 0, "restart removes interrupted temporary write");
        check(best(faults, 10).match.tokens == 3, "earlier snapshot survives writer crash");
        check(best(faults, 20).match.tokens == (after_rename ? 3 : 0), "restart sees only atomically published snapshot");
        if (after_rename)
            check(faults.read(best(faults, 20), 1 << 20, 1 << 21, 0, image, error) && encoded(image) == encoded(b),
                  "published bytes survive process restart with integrity intact");
        faults.close();
    }
#endif
    std::printf("conversation_store_test: %d checks passed\n", checks);
}
