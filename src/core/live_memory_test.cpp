#include "strata/core/expert_cache.hpp"
#include "strata/core/live_memory.hpp"
#include "strata/core/live_prefill.hpp"
#if !defined(STRATA_LIVE_DEVICE_TEST_ONLY)
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#endif

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int checks = 0;
void require(bool ok, const std::string& label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
void inject(const char* value) {
#if defined(_WIN32)
    _putenv_s("STRATA_TEST_LIVE_MAP_FAIL_AFTER", value);
#else
    if (*value) setenv("STRATA_TEST_LIVE_MAP_FAIL_AFTER", value, 1);
    else unsetenv("STRATA_TEST_LIVE_MAP_FAIL_AFTER");
#endif
}
void inject_ram(bool enabled) {
#if defined(_WIN32)
    _putenv_s("STRATA_TEST_LIVE_RAM_FAIL_AFTER_ALLOC", enabled ? "1" : "");
#else
    if (enabled) setenv("STRATA_TEST_LIVE_RAM_FAIL_AFTER_ALLOC", "1", 1);
    else unsetenv("STRATA_TEST_LIVE_RAM_FAIL_AFTER_ALLOC");
#endif
}
void protocol() {
    using namespace strata::core;
    LiveMemoryRequest request;
    require(parse_live_memory_request("MEMORY 7 0 900", request) && request.id == 7 &&
            request.resident_mib == 0 && request.vram_reserve_mib == 900, "zero RAM target is valid");
    for (const char* line : {"MEMORY", "MEMORY 0 1 2", "MEMORY -1 2 3", "MEMORY 1 -1 3",
                             "MEMORY 1 2 +3", "MEMORY 1 2 3 extra", "MEMORY 1 1048577 3",
                             "MEMORY 18446744073709551616 1 2", "MEMORY 1 2 3.5"})
        require(!parse_live_memory_request(line, request), std::string("reject malformed: ") + line);
}
void pressure_direction() {
    using namespace strata::core;
    constexpr uint64_t MiB = 1ull << 20, quantum = 32 * MiB;
    const LiveMemoryRequest shrink{1, 32000, 1536};
    const bool shrink_growth = live_memory_gpu_growth_allowed(shrink, 35000 * MiB, 1536);
    require(!shrink_growth, "unchanged-reserve RAM shrink prohibits GPU growth at admission");
    require(live_memory_gpu_budget(1578 * MiB, 2752 * MiB, 1536 * MiB, quantum, shrink_growth) == 2752 * MiB,
            "regression: RAM release must not trigger the refused extra GPU mapping");
    require(live_memory_gpu_budget(4096 * MiB, 2752 * MiB, 1536 * MiB, quantum, shrink_growth) == 2752 * MiB,
            "latched pressure direction survives free-memory rebound after partial RAM release");
    require(live_memory_gpu_budget(1000 * MiB, 2752 * MiB, 1536 * MiB, quantum, shrink_growth) < 2752 * MiB,
            "pressure direction still permits GPU eviction to satisfy reserve");
    require(live_memory_gpu_budget(1000 * MiB, 0, 1536 * MiB, quantum, shrink_growth) == 0,
            "GPU floor remains zero while RAM release continues");
    require(!live_memory_gpu_growth_allowed({2, 40000, 2048}, 35000 * MiB, 1536),
            "raising reserve prohibits GPU growth even when RAM target grows");
    require(!live_memory_gpu_growth_allowed({3, 32000, 1024}, 35000 * MiB, 1536),
            "RAM pressure release takes priority over a lower reserve");
    require(live_memory_gpu_growth_allowed({4, 40000, 1024}, 35000 * MiB, 1536),
            "capacity recovery may grow the GPU tier");
    require(live_memory_gpu_growth_allowed({5, 35000, 1024}, 35000 * MiB + MiB / 2, 1536),
            "unchanged reported RAM tolerates its unreported fraction during GPU recovery");
    require(live_memory_gpu_budget(1578 * MiB, 2752 * MiB, 1536 * MiB, quantum, true) == 2752 * MiB,
            "legitimate growth retains one additional granule of headroom");
    require(live_memory_gpu_budget(1632 * MiB, 2752 * MiB, 1536 * MiB, quantum, true) == 2816 * MiB,
            "growth admits only the space beyond reserve and granule margin");
}
void loan_geometry() {
    using namespace strata::core;
    const uint64_t offsets[] = {0, 5, 8, 18, 22, 41, 47};
    require(live_prefill_floor(offsets, 6, 20, 2) == 5, "sized loan floor retains decode prefix and whole prompt buffers");
    require(live_prefill_first(offsets, 4, 20, 2) == -1, "loan rejects a shrunken cache below its minimum floor");
    require(live_prefill_first(offsets, 5, 20, 2) == 3, "post-shrink loan uses the current end, not the old tail");
    require(live_prefill_first(offsets, 6, 20, 2) == 4, "regrowth rebinds the loan to its new tail");
    require(live_prefill_first(offsets, 6, 39, 2) == 2, "exactly fitting loan retains the complete decode prefix");
    require(live_prefill_floor(offsets, 6, 40, 2) == -1, "unreachable prompt floor is refused");
    for (int64_t slots = 3; slots <= 6; ++slots)
        for (uint64_t bytes = 1; bytes <= offsets[slots] - offsets[2]; ++bytes) {
            const int64_t first = live_prefill_first(offsets, slots, bytes, 2);
            require(first >= 2 && first < slots && offsets[slots] - offsets[first] >= bytes &&
                    (first + 1 == slots || offsets[slots] - offsets[first + 1] < bytes),
                    "every variable-sized loan is sufficient and starts at the latest valid slot");
        }
}
void device_arena() {
    using strata::core::ExpertCache;
    std::string err;
    std::vector<int64_t> sizes(64, 4ll << 20);
    sizes[0] = (2ll << 20) + 256; // unequal, aligned offsets must remain identical through every resize
    ExpertCache cache;
    require(cache.open_live(sizes, 24, 2, 32, err), "open VMM: " + err);
    auto* base = cache.device_slot(0);
    const auto* offsets = cache.slot_offsets();
    const uint64_t initial = cache.committed_bytes();
    std::vector<uint8_t> blob((size_t) sizes[0], 0x57);
    require(cache.fill_slot_blocking(0, blob.data(), err, sizes[0]), "fill live slot: " + err);
    require(cache.verify_slot(0, blob.data(), err, sizes[0]), "initial live bytes: " + err);
    uint8_t* readback = nullptr;
    require(cudaHostAlloc((void**) &readback, 256, cudaHostAllocDefault) == cudaSuccess, "graph readback allocation");
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "graph stream");
    require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess, "capture begin");
    require(cudaMemcpyAsync(readback, base, 256, cudaMemcpyDeviceToHost, stream) == cudaSuccess, "capture stable base");
    require(cudaStreamEndCapture(stream, &graph) == cudaSuccess &&
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess, "capture instantiate");
    size_t free_before = 0, total = 0, free_after = 0;
    require(cudaDeviceSynchronize() == cudaSuccess && cudaMemGetInfo(&free_before, &total) == cudaSuccess,
            "measure before shrink");
    require(cache.resize_live(2, err), "shrink: " + err);
    require(cache.committed_bytes() < initial && cache.slots() == 2 && cache.device_slot(2) == nullptr,
            "shrink unmaps physical blocks and hides inactive slots");
    require(cudaMemGetInfo(&free_after, &total) == cudaSuccess, "measure after shrink");
    std::printf("VMM release: committed %llu -> %llu, free %llu -> %llu bytes\n",
                (unsigned long long) initial, (unsigned long long) cache.committed_bytes(),
                (unsigned long long) free_before, (unsigned long long) free_after);
    require(free_after + (8ull << 20) >= free_before + initial - cache.committed_bytes(),
            "driver reports physical release (8 MiB telemetry tolerance)");
    const uint64_t shrunk = cache.committed_bytes();
    inject("1");
    const bool grown = cache.resize_live(48, err);
    inject("");
    require(!grown && cache.slots() == 2 && cache.committed_bytes() == shrunk,
            "allocation failure after one new map rolls back only the delta");
    require(cache.verify_slot(0, blob.data(), err, sizes[0]), "failed growth retains original bytes");
    require(cache.resize_live(48, err), "regrow: " + err);
    require(cache.device_slot(0) == base && cache.slot_offsets() == offsets &&
            cache.slot_offsets()[1] == (uint64_t) sizes[0], "base and complete offset table stay stable");
    require(cudaGraphLaunch(exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess,
            "pre-resize captured graph replays after regrowth");
    require(std::memcmp(readback, blob.data(), 256) == 0, "captured graph reads preserved bytes");
    // A prompt borrows only the active tail; after refill its raw views must be rebound before a shrink can
    // retire that tail. Exercise the same sized-offset planner against real VMM mappings, not fake pointers.
    const uint64_t loan_bytes = 12ull << 20;
    const int64_t original_first = strata::core::live_prefill_first(offsets, cache.slots(), loan_bytes, 2);
    auto* original_view = cache.device_slot(original_first);
    require(cudaMemset(original_view, 0x23, (size_t) loan_bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
            "borrowed view writes the mapped tail");
    const int64_t new_slots = 12;
    const int64_t rebound_first = strata::core::live_prefill_first(offsets, new_slots, loan_bytes, 2);
    auto* rebound_view = cache.device_slot(rebound_first);
    require(rebound_view != original_view && rebound_first >= 2 && cache.resize_live(new_slots, err),
            "raw view rebind precedes retirement of the old tail");
    require(cudaMemset(rebound_view, 0x6a, (size_t) loan_bytes) == cudaSuccess &&
            cudaMemcpy(readback, rebound_view, 256, cudaMemcpyDeviceToHost) == cudaSuccess && readback[0] == 0x6a,
            "rebound prompt view survives physical shrink");
    require(cache.verify_slot(0, blob.data(), err, sizes[0]), "borrowed buffers preserve the retained decode prefix");
    require(cache.resize_live(48, err), "loan cache regrows");
    require(!cache.resize_live(65, err) && cache.slots() == 48, "invalid growth preserves active layout");
    require(cache.resize_live(0, err) && cache.committed_bytes() == 0 && cache.valid(),
            "zero cache releases every mapping but retains the virtual base");
    require(cache.resize_live(1, err) && cache.device_slot(0) == base, "regrow from zero uses the same address");
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream); cudaFreeHost(readback);
}
#if !defined(STRATA_LIVE_DEVICE_TEST_ONLY)
void ram_blocks(bool pin, bool mixed = false) {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    namespace fs = std::filesystem;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path dir = fs::current_path() / ("live-memory-fixture-" + std::to_string(stamp));
    fs::create_directory(dir);
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{dir};
    std::string err;
    require(expert_layout_load(dir.string(), 2, 4, err), "load canonical fixture: " + err);
    {
        std::ofstream file(dir / "experts.bin", std::ios::binary);
        for (int i = 0; i < 8; ++i) {
            std::vector<char> data((size_t) BLOB, (char) (i + 1));
            file.write(data.data(), (std::streamsize) data.size());
        }
        require((bool) file, "write fixture");
    }
    FileExpertSource source;
    require(source.open(dir.string(), 2, 4, err), "open fixture: " + err);
    const uint64_t block = (uint64_t) BLOB * 2;
    require(source.enable_live_resident(pin, err, mixed ? block : UINT64_MAX), "enable live RAM: " + err);
    std::vector<int32_t> res(8, -1);
    res[0] = 0;
    const std::vector<std::pair<int32_t, int32_t>> rank{{0, 1}, {0, 2}, {0, 3}, {1, 0}};
    bool done = false;
    require(source.resize_live_resident(block * 2, block, 0, res, rank, done, err) && !done,
            "first bounded RAM growth: " + err);
    const auto* first = source.blob(0, 1);
    require(first && first[0] == 2 && source.resident_bytes() == block && !source.has_resident(0, 0),
            "RAM stores requested experts and skips authoritative GPU residents");
    require(source.resize_live_resident(block * 2, block, 0, res, rank, done, err) && done,
            "second independent RAM block: " + err);
    require(source.blob(0, 1) == first && source.blob(1, 0)[0] == 5, "growth preserves old pointers and bytes");
    const uint64_t pinned = pin ? (mixed ? block : block * 2) : 0;
    require(source.pinned_bytes() == pinned && source.pinned(0, 1) == pin &&
            source.pinned(1, 0) == (pin && !mixed), "mixed backing reports actual pinned bytes and expert ownership");
    if (mixed)
        require(source.device_alias(1, 0) == nullptr && source.has_resident(1, 0),
                "pageable fallback retains resident bytes without advertising a mapped alias");
    require(source.resize_live_resident(0, 32ull << 20, 0, res, rank, done, err) && !done &&
            source.resident_bytes() == block && source.blob(0, 1) == first,
            "even small RAM blocks shrink by only one block per safe point");
    require(source.pinned_bytes() == (pin ? block : 0), "tail release adjusts the actual block's pin counter");
    require(source.resize_live_resident(block * 2, block, 0, res, rank, done, err) && done,
            "regrow after bounded shrink");
    require(source.pinned_bytes() == pinned, "regrowth respects the aggregate pin cap");
    require(!source.resize_live_resident(block * 3, block, UINT64_MAX, res, rank, done, err) &&
            source.resident_bytes() == block * 2 && source.blob(0, 1) == first,
            "headroom admission refusal preserves committed RAM");
    inject_ram(true);
    const bool growth = source.resize_live_resident(block * 3, block, 0, res, rank, done, err);
    inject_ram(false);
    require(!growth && source.resident_bytes() == block * 2 && source.blob(0, 1) == first,
            "exception after allocation releases uncommitted RAM and preserves old ownership");
    require(source.reserve_exchanges(1, err), "reserve adaptive exchange: " + err);
    std::memset(source.exchange_buffer(0), 1, (size_t) BLOB);
    require(source.stage_exchange(0, 1, 0, 0) && source.commit_exchanges() == 1, "adaptive owner exchange");
    res[0] = -1; res[1] = 0;
    require(source.blob(0, 0) == first && first[0] == 1 && !source.has_resident(0, 1),
            "exchange changes the actual owner of a RAM block slot");
    if (pin) {
        uint8_t got = 0;
        require(source.pinned(0, 0) && source.device_alias(0, 0) &&
                cudaMemcpy(&got, source.device_alias(0, 0), 1, cudaMemcpyDeviceToHost) == cudaSuccess && got == 1,
                "live pinned alias reads exchanged bytes");
    }
    require(source.resize_live_resident(block, block, 0, res, rank, done, err) && done &&
            source.resident_bytes() == block && source.blob(0, 0) == first && !source.has_resident(1, 0),
            "shrink frees tail block and preserves exchanged prefix ownership");
    require(source.blob(1, 0)[0] == 5, "freed RAM expert falls back to identical file bytes");
    require(source.resize_live_resident(0, block, 0, res, rank, done, err) && done &&
            source.resident_bytes() == 0 && !source.has_resident(0, 0) && source.blob(0, 0)[0] == 1,
            "zero RAM target releases all blocks and keeps file fallback");
    require(source.pinned_bytes() == 0 && !source.pcie_layer(0), "zero RAM releases all mapped-host ownership");
    require(source.resize_live_resident(block, block, 0, res, rank, done, err) && done &&
            !source.has_resident(0, 1), "regrowth uses current GPU residency rather than startup slot_of");
    const std::vector<std::pair<int32_t, int32_t>> hot{{0, 1}, {0, 0}};
    require(source.resize_live_resident(block * 2, block, 0, res, hot, done, err, true) && done &&
            source.has_resident(0, 1) && source.resident_bytes() == block * 2,
            "borrow coverage duplicates a current GPU expert within the same bounded RAM target");
    const uint8_t* duplicate = source.blob(0, 1);
    require(!source.stage_exchange(0, 0, 1, 0) && source.blob(0, 1) == duplicate && duplicate[0] == 2,
            "already-backed adaptive victim cannot overwrite or transfer a duplicate RAM owner");
    if (mixed) {
        require(source.resize_live_resident(block * 2, block, 0, res, rank, done, err) && done,
                "close fixture with both pinned and pageable blocks alive");
        source.close();
        require(source.resident_bytes() == 0 && source.pinned_bytes() == 0, "mixed block close releases every owner");
    }
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        protocol();
        pressure_direction();
        loan_geometry();
        if (argc <= 1 || std::strcmp(argv[1], "--protocol-only") != 0) {
            device_arena();
#if !defined(STRATA_LIVE_DEVICE_TEST_ONLY)
            ram_blocks(false);
            ram_blocks(true);
            ram_blocks(true, true);
#endif
        }
        std::printf("live memory: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        inject("");
        inject_ram(false);
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
