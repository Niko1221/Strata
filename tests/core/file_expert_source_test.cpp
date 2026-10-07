#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
#endif

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

#if defined(_WIN32)
void check_mapped_release(strata::core::FileExpertSource& source, int64_t layer, int64_t expert, uint64_t bytes) {
    const uint8_t* p = source.blob(layer, expert);
    require(p != nullptr, "release fixture has no mapped blob");
    const std::vector<uint8_t> saved(p, p + (size_t) bytes);
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const uintptr_t page = info.dwPageSize;
    const uintptr_t first = ((uintptr_t) p + page - 1) / page * page;
    const uintptr_t end = ((uintptr_t) p + bytes) / page * page;
    require(end > first, "release fixture has no full interior page");
    auto resident = [&](uintptr_t address) {
        PSAPI_WORKING_SET_EX_INFORMATION state{};
        state.VirtualAddress = (void*) address;
        require(QueryWorkingSetEx(GetCurrentProcess(), &state, sizeof state) != 0,
                "QueryWorkingSetEx failed for the release fixture");
        return state.VirtualAttributes.Valid != 0;
    };
    const char* setting = std::getenv("STRATA_FILE_RELEASE");
    const bool enabled = setting != nullptr && std::strcmp(setting, "1") == 0;
    for (int round = 0; round < 3; ++round) {
        require(std::memcmp(p, saved.data(), saved.size()) == 0, "release changed the mapped expert bytes");
        require(resident(first), "reading the expert did not bring its interior page into the working set");
        const uint64_t released = source.release(layer, expert);
        require(released == (enabled ? end - first : 0), "release did not honor full pages or its opt-in switch");
        // Off: the page stays.  On: Windows may keep a trimmed page on its standby list and map it back at once, so a
        // page that is still valid is reported, not failed (the byte count above is the contract).
        if (!enabled) require(resident(first), "release trimmed pages with the switch off");
        else if (resident(first)) std::fprintf(stderr, "note: the trimmed page is already back in the working set\n");
        if ((uintptr_t) p < first)
            require(resident((uintptr_t) p), "release trimmed the expert's shared first page");
        if (end < (uintptr_t) p + bytes)
            require(resident(end), "release trimmed the expert's shared last page");
    }
    require(std::memcmp(p, saved.data(), saved.size()) == 0, "the expert changed after repeated release/read cycles");
    require(source.release(-1, expert) == 0 && source.release(layer, -1) == 0 &&
                source.release(std::numeric_limits<int64_t>::max(), expert) == 0 &&
                source.release(layer, std::numeric_limits<int64_t>::max()) == 0,
            "release accepted an invalid layer or expert");
}
#endif

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");
#if defined(_WIN32)
    check_mapped_release(source, 0, 0, BLOB);
    check_mapped_release(source, 0, 1, BLOB);
#endif

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
    require(source.release(0, 0) == 0, "a closed file source released pages");
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");
#if defined(_WIN32)
    check_mapped_release(source, 0, 1, first_fmt.bytes);
    check_mapped_release(source, 1, 1, second_fmt.bytes);
#endif

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

void test_complement_plan() {
    using namespace strata::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
}

void test_resident_lend_region() {
    using namespace strata::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_resident_exchange() {
    using namespace strata::core;
    using namespace strata::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

// #1250: when the page-locked complement is registered in steps, and what a meminfo reading says
void test_pin_pacing() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    // the reporter's reading: 54 GiB available but 16 GiB really free for a 26 GiB complement
    require(pin_depends_on_reclaim(16 * GiB, 26 * GiB, 4 * GiB), "a complement bigger than the free pages was not flagged");
    require(!pin_depends_on_reclaim(40 * GiB, 26 * GiB, 4 * GiB), "a complement the free pages cover was flagged");
    require(pin_depends_on_reclaim(30 * GiB, 26 * GiB, 4 * GiB) == false, "exactly free = bytes + reserve is enough");
    require(!pin_depends_on_reclaim(0, 0, 4 * GiB), "no complement, nothing to pin");
    require(pin_step_fits(6 * GiB, 1 * GiB, 4 * GiB) && !pin_step_fits(4 * GiB, 1 * GiB, 4 * GiB),
            "a step must leave the reserve");
    const std::string info =
        "MemTotal:       65011484 kB\nMemFree:         1284096 kB\nMemAvailable:   33463296 kB\nCached:         54507520 kB\n";
    require(meminfo_bytes(info, "MemFree") == 1284096ull * 1024, "MemFree was misread");
    require(meminfo_bytes(info, "MemAvailable") == 33463296ull * 1024, "MemAvailable was misread");
    require(meminfo_bytes(info, "Mem") == 0 && meminfo_bytes(info, "Shmem") == 0, "an absent key was not 0");
}

void test_resident_memory_budget() {
    using strata::core::detail::clamp_resident_budget;
    constexpr uint64_t GiB = 1ull << 30, margin = 256ull << 20, headroom = 4 * GiB;
    constexpr uint64_t unlimited = std::numeric_limits<uint64_t>::max();
    // #730: RAM can hold the requested cache, but Windows cannot commit it.
    require(clamp_resident_budget(66 * GiB, 69 * GiB, 47 * GiB, headroom) == 43 * GiB - margin,
            "a RAM budget exceeded available commit capacity");
    require(clamp_resident_budget(66 * GiB, 47 * GiB, 69 * GiB, headroom) == 43 * GiB - margin,
            "a RAM budget exceeded available physical memory");
    require(clamp_resident_budget(40 * GiB, 69 * GiB, 47 * GiB, headroom) == 40 * GiB,
            "a smaller explicit budget was changed");
    require(clamp_resident_budget(unlimited, 69 * GiB, 47 * GiB, headroom) == 43 * GiB - margin,
            "the what-fits budget ignored commit capacity");
    require(clamp_resident_budget(66 * GiB, 47 * GiB, unlimited, headroom) == 43 * GiB - margin,
            "a platform without a commit reading lost its RAM limit");
    require(clamp_resident_budget(43 * GiB, 69 * GiB, 47 * GiB, headroom) == 43 * GiB,
            "a budget that exactly fits received the clamping margin");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, 0, headroom) == 0,
            "exhausted commit capacity was treated as unlimited");
    require(clamp_resident_budget(66 * GiB, 0, 69 * GiB, headroom) == 0,
            "exhausted RAM produced a positive budget");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, headroom - 1, headroom) == 0 &&
                clamp_resident_budget(66 * GiB, 69 * GiB, headroom, headroom) == 0,
            "subtracting headroom underflowed");
    require(clamp_resident_budget(66 * GiB, 69 * GiB, headroom + margin - 1, headroom) == 0 &&
                clamp_resident_budget(66 * GiB, 69 * GiB, headroom + margin, headroom) == 0,
            "subtracting the clamping margin underflowed");
}

void test_cgroup_memory_budget() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    CgroupMemoryStat clean_cache{40 * GiB, 32 * GiB, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB, "clean inactive file cache was not credited against the cgroup cap");

    CgroupMemoryStat dirty_cache{40 * GiB, 32 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, dirty_cache, bytes), "valid dirty-cache counters were rejected");
    require(bytes == 42 * GiB, "dirty/writeback pages were incorrectly counted as reclaimable");

    CgroupMemoryStat oversized_inactive{10 * GiB, 20 * GiB, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_inactive, bytes),
            "valid oversized inactive-file counter was rejected");
    require(bytes == 12 * GiB, "inactive-file accounting exceeded charged current usage");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes),
            "large memory.stat counters were rejected");
    require(bytes == 0, "dirty/writeback subtraction overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes),
            "valid over-limit memory counters were rejected");
    require(bytes == 0, "over-limit cgroup usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// #286: the unbuffered file tier (Windows: FILE_FLAG_NO_BUFFERING; Linux: O_DIRECT + io_submit) hands out the same
// bytes as the mapping - one blob at a time, a batch of adjacent experts (merged into one request), and copy_blob - and
// never falls back to the mapping.  The pack lives in the working directory, not the temp directory: a tmpfs /tmp has
// no O_DIRECT, and there the test says so and skips.
void test_unbuffered_reads() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    std::vector<uint8_t> bytes((size_t) total);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint8_t& b : bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "could not map the synthetic pack: " + err);
    set_env("STRATA_UNBUFFERED_LOAD", "1");
    std::string why;
    const bool unbuffered = source.set_unbuffered(0, why);
    set_env("STRATA_UNBUFFERED_LOAD", nullptr);
#if defined(_WIN32) || defined(__linux__)
    if (!unbuffered) {
        std::cout << "file_expert_source_test: unbuffered reads skipped (" << why << ")\n";
        source.close();
        return;
    }
#else
    require(!unbuffered, "unbuffered reads on a platform without them");
    source.close();
    return;
#endif
    require(source.unbuffered(), "set_unbuffered succeeded but the source is not unbuffered");
    const int64_t batch[experts] = {0, 1, 2};
    source.prefetch(1, batch, experts);   // layer 1 in one batch: three adjacent blobs
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < experts; ++e) {
            const uint8_t* b = source.blob(l, e);
            require(b != nullptr, "an unbuffered blob lookup failed");
            require(std::memcmp(b, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB),
                                (size_t) BLOB) == 0,
                    "an unbuffered blob differs from the file");
        }
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(0, 2, copy.data()) &&
                std::memcmp(copy.data(), bytes.data() + (size_t) (2 * BLOB), (size_t) BLOB) == 0,
            "an unbuffered copy_blob differs from the file");
    require(source.direct_fallbacks() == 0, "an unbuffered read fell back to the mapping");
    source.close();
}

// #633: the host RAM probe with fake /proc and cgroup trees: v2 (a limit, "max", a missing or malformed limit), v1
// (a limit, unlimited), and no cgroup line at all.  Elsewhere than Linux it reads the machine's RAM.
void test_host_memory() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    HostMemory m;
#if defined(__linux__)
    TempDirectory t;
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((t.path / rel).parent_path());
        std::ofstream(t.path / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (t.path / "meminfo").string(), cg = (t.path / "cgroup").string(),
                      root = (t.path / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        m.commit = 0;
        const bool ok = host_available_memory(m, mi, cg, root);
        require(m.commit == ~0ull, "Linux memory probe retained a commit limit");
        return ok;
    };
    // no cgroup line at all: MemAvailable alone (before #633: "cannot determine")
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 36 * GiB && m.cgroup_limit == 48 * GiB, "v2 limit");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n"), "v2 malformed limit accepted");
    fs::remove(t.path / "fs/box/memory.max");
    require(!probe("0::/box\n"), "v2 group without memory.max accepted");
    // v1: the memory controller's group (a hybrid line list), a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v1 unlimited");
    // a v1 group not visible here (another namespace): MemAvailable alone
    require(probe("4:memory:/elsewhere\n") && m.available == 100 * GiB, "v1 group not mounted");
#else
    require(host_available_memory(m) && m.available > 0 && m.cgroup_limit == ~0ull, "this PC's RAM");
#if defined(_WIN32)
    require(m.commit != ~0ull, "Windows memory probe did not report commit capacity");
#else
    require(m.commit == ~0ull, "a platform without commit accounting imposed a limit");
#endif
    (void) GiB;
#endif
}

// STRATA_IO_PREFETCH: the experts.bin file tier hands out the same bytes with the Linux I/O path on (whole-blob
// preads, the read-ahead workers, the mapping for cached blobs), however the reads and the predictions interleave.
void test_io_prefetch() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 3;
    constexpr int64_t experts = 8;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir(fs::current_path());
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
    std::vector<uint8_t> bytes((size_t) total);
    uint64_t x = 0xD1B54A32D192ED03ull;
    for (uint8_t& b : bytes) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        b = (uint8_t) (x >> 56);
    }
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
        out.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        require((bool) out, "could not write the synthetic experts.bin");
    }
    FileExpertSource source;
    require(source.open(dir.path.string(), layers, experts, err), "could not map the synthetic pack: " + err);
    source.set_io_prefetch(true, 3);
#if defined(__linux__)
    require(source.io_prefetch() && source.warms(), "io prefetch did not turn on");
    // predictions for the next layers, some of them wrong or repeated, while the layers are read
    for (int round = 0; round < 20; ++round)
        for (int64_t l = 0; l < layers; ++l) {
            const int64_t pred[5] = {(round + l) % experts, (round * 3 + 1) % experts, (round + 2 * l + 5) % experts,
                                     (round + l) % experts, 7};
            source.warm((l + 1) % layers, pred, 5);
            source.begin_layer(l, nullptr, 0);
            std::vector<int64_t> want = {(round + l) % experts, (round * 5 + 2) % experts, (l * 3 + round) % experts};
            source.prefetch(l, want.data(), (int64_t) want.size());
            for (int64_t e : want) {
                const uint8_t* b = source.blob(l, e);
                require(b != nullptr, "an io-prefetch blob lookup failed");
                require(std::memcmp(b, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB), (size_t) BLOB) == 0,
                        "an io-prefetch blob differs from the file");
                const uint8_t* s2 = source.blob_stable(l, e);
                require(s2 != nullptr && std::memcmp(s2, bytes.data() + (size_t) (l * (int64_t) layer_bytes + e * (int64_t) BLOB), (size_t) BLOB) == 0,
                        "a blob_stable pointer differs from the file");
            }
        }
    std::vector<uint8_t> copy((size_t) BLOB);
    require(source.copy_blob(1, 6, copy.data()) &&
                std::memcmp(copy.data(), bytes.data() + (size_t) (layer_bytes + 6 * BLOB), (size_t) BLOB) == 0,
            "an io-prefetch copy_blob differs from the file");
    const FileExpertSource::IoCounters c = source.io_counters();
    require(c.cached_bytes + c.uncached_bytes > 0, "no residency counted");
    std::cout << "io prefetch: " << c.pf_read_blobs << " read ahead, " << c.pf_used << " used, " << c.pf_resident_skips
              << " skipped, pread " << c.pread_bytes << " B: bytes PASS\n";
#else
    require(!source.io_prefetch(), "io prefetch on a platform without it");
#endif
    source.close();
}

void test_rotating_source(bool rotate, bool pin) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
#if defined(_WIN32)
    _putenv_s("STRATA_EXCHANGE_ROTATE", rotate ? "1" : "0");
#else
    setenv("STRATA_EXCHANGE_ROTATE", rotate ? "1" : "0", 1);
#endif
    TempDirectory dir;
    std::string err;
    require(expert_layout_load(dir.path.string(), 1, 5, err), err);
    const size_t bytes = (size_t)BLOB;
    std::vector<std::vector<uint8_t>> truth(5, std::vector<uint8_t>(bytes));
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        for (size_t e = 0; e < truth.size(); ++e) {
            for (size_t i = 0; i < bytes; ++i) truth[e][i] = (uint8_t)((i * 131u + e * 37u) ^ (i >> 8));
            out.write((const char*)truth[e].data(), (std::streamsize)bytes);
        }
        require((bool)out, "fixture write failed");
    }
    FileExpertSource src;
    ExpertCache cache;
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(cache.open(2, 1, 5, BLOB, err), err);
    for (int e : {2, 3}) {
        const int slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, truth[e].data(), err), err);
    }
    const std::vector<std::pair<int32_t, int32_t>> rank{{0,0},{0,1},{0,2},{0,3},{0,4}};
    // Keep expert 4 on the file tier to check that the fallback survives rotation.
    require(src.pin_cache_complement(cache, err, pin, {}, -1, 0, 2 * bytes, &rank), err);
    require(src.reserve_exchanges(2, err), err);
    require(src.exchange_rotation() == (rotate && pin), "rotation activation/fallback wrong");
    require(src.reserve_exchanges(1, err), "smaller capacity rejected");
    if (rotate && pin) require(!src.reserve_exchanges(3, err), "live arena growth accepted");
    int incoming[2] = {0, 1}, outgoing[2] = {2, 3};
    std::vector<uint8_t> actual(bytes);
    double commit_ms = 0;
    for (int round = 0; round < 32; ++round) {
        const uint8_t* prior[2]{};
        uint8_t* eviction[2]{};
        for (int q = 0; q < 2; ++q) {
            prior[q] = src.blob(0, incoming[q]); eviction[q] = src.exchange_buffer(q);
            require(prior[q] && !std::memcmp(prior[q], truth[incoming[q]].data(), bytes), "wrong incoming bytes");
            require(cudaMemcpy(eviction[q], cache.device_slot(q), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                    "D2H eviction failed");
            require(src.stage_exchange(0, incoming[q], outgoing[q], q), "stage rejected");
            require(!std::memcmp(src.blob(0, outgoing[q]), truth[outgoing[q]].data(), bytes), "staged override wrong");
            require(cache.fill_slot_blocking(q, prior[q], err), err); // H2D completed before commit
        }
        require(!src.stage_exchange(0, incoming[0], outgoing[0], 0), "duplicate stage accepted");
        require(!src.stage_exchange(0, incoming[0], 5, 0), "invalid expert accepted");
        const auto t0 = std::chrono::steady_clock::now();
        require(src.commit_exchanges() == 2, "commit count wrong");
        commit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        require(src.commit_exchanges() == 0, "empty commit not empty");
        for (int q = 0; q < 2; ++q) {
            require(!src.has_resident(0, incoming[q]) && src.has_resident(0, outgoing[q]), "residency wrong");
            const uint8_t* held = src.blob(0, outgoing[q]);
            require(!std::memcmp(held, truth[outgoing[q]].data(), bytes), "evicted bytes corrupted");
            require(held == ((rotate && pin) ? eviction[q] : prior[q]), "wrong storage selected");
            if (rotate && pin) require(src.exchange_buffer(q) == prior[q], "old input not recycled");
            require(src.copy_blob(0, outgoing[q], actual.data()) && actual == truth[outgoing[q]], "copy_blob mismatch");
            require(!src.transient(0, outgoing[q]), "resident became transient");
            if (pin) {
                require(src.pinned(0, outgoing[q]) && src.device_alias(0, outgoing[q]), "mapping lost");
                require(cudaMemcpy(actual.data(), src.device_alias(0, outgoing[q]), bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                        "device alias read failed");
                require(actual == truth[outgoing[q]], "CUDA alias points at stale expert");
            }
            require(cache.verify_slot(q, truth[incoming[q]].data(), err), err);
            std::swap(incoming[q], outgoing[q]);
        }
        require(!src.has_resident(0, 4) && src.copy_blob(0, 4, actual.data()) && actual == truth[4], "file fallback changed");
    }
    require(src.exchanges() == 64, "exchange accounting wrong");
    require(src.rotated_exchanges() == ((rotate && pin) ? 64u : 0u), "rotation accounting wrong");
    require(src.avoided_exchange_copy_bytes() == ((rotate && pin) ? 64 * bytes : 0u), "avoided bytes wrong");
    std::cout << "rotation integration: rotate=" << rotate << " pinned=" << pin
              << " exchanges=64 commit_ms=" << commit_ms << " exact_bytes=PASS\n";
    src.close();
    require(!src.exchange_rotation() && !src.exchange_buffer(0), "close retained storage");
    require(src.open(dir.path.string(), 1, 5, err), err);
    require(!src.has_resident(0, 0) && !src.exchange_rotation(), "reopen retained residency");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        test_complement_plan();
        test_resident_lend_region();
        test_resident_exchange();
        test_resident_memory_budget();
        test_pin_pacing();
        test_cgroup_memory_budget();
        test_host_memory();
        test_canonical_layout();
        test_unbuffered_reads();
        test_io_prefetch();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
#endif
        if (argc == 2 && std::string(argv[1]) == "--rotation-gpu") {
            test_rotating_source(false, true);
            test_rotating_source(true, true);
            test_rotating_source(true, false);
        }
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
