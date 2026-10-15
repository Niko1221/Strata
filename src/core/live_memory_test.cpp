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
            request.resident_mib == 0 && request.vram_reserve_mib == 900 && !request.hold_gpu,
            "legacy zero RAM target is valid without GPU hold");
    require(parse_live_memory_request("MEMORY 8 32768 320 hold", request) && request.id == 8 &&
            request.resident_mib == 32768 && request.vram_reserve_mib == 320 && request.hold_gpu,
            "optional hold is explicit and preserves numerical targets");
    require(parse_live_memory_request("MEMORY 9 0 900  ", request) && !request.hold_gpu,
            "a later legacy command clears the previous parsed hold flag");
    for (const char* line : {"MEMORY", "MEMORY 0 1 2", "MEMORY -1 2 3", "MEMORY 1 -1 3",
                             "MEMORY 1 2 +3", "MEMORY 1 2 3 extra", "MEMORY 1 1048577 3",
                             "MEMORY 18446744073709551616 1 2", "MEMORY 1 2 3.5",
                             "MEMORY 1 2 3 HOLD", "MEMORY 1 2 3 hold extra", "MEMORY 1 2 3 hold hold",
                             "MEMORY 1 2 3 hold=1", "MEMORY 1 -2 3 hold"})
        require(!parse_live_memory_request(line, request), std::string("reject malformed: ") + line);
    require(request.id == 9 && !request.hold_gpu, "malformed requests cannot change an accepted control");
}
void prompt_pause() {
    using strata::core::live_prefill_pause;
    for (int fail = 0; fail <= 6; ++fail) {
        bool active = true;
        bool cancel = fail == 1;
        bool returned = false, resized = false, rebound = false;
        std::string order, err;
        const bool ok = live_prefill_pause(active,
            [&](std::string&) { order += 'D'; return fail != 3; },
            [&](std::string&) { order += 'R'; returned = fail != 4; return returned; },
            [&](std::string&) {
                require(!active && returned, "resize is permitted only after readers drain and loan returns");
                order += 'S'; resized = true; cancel = fail == 2; return fail != 5;
            },
            [&] { return cancel; },
            [&](std::string&) {
                require(!active && resized, "rebind/lend follows the completed resize transaction");
                order += 'L'; rebound = fail != 6; return rebound;
            }, err);
        require(ok == (fail == 0), "pause success follows all required steps");
        if (fail == 0) require(order == "DRSL" && active && rebound, "resume lends once after a safe resize");
        if (fail == 1) require(order == "DR" && !active && err == "cancelled", "cancellation returns loan without resizing or relending");
        if (fail == 2) require(order == "DRS" && !active && err == "cancelled", "cancellation arriving during resize never relends");
        if (fail == 3) require(order == "D" && active, "drain failure cannot release an in-flight loan");
        if (fail == 4) require(order == "DR" && active, "loan-return failure cannot authorize resize");
        if (fail == 5) require(order == "DRS" && !active, "failed control service leaves no borrowed views active");
        if (fail == 6) require(order == "DRSL" && !active, "failed rebind cannot advertise an active prompt loan");
    }
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
    const LiveMemoryRequest recovering{8, 40000, 512};
    require(live_memory_supersedes({9, 31000, 512}, recovering, 32000 * MiB), "new RAM pressure cancels unfinished recovery");
    require(live_memory_supersedes({9, 32000, 1024}, recovering, 32000 * MiB), "higher reserve and non-growing RAM may preempt recovery");
    require(!live_memory_supersedes({9, 35000, 1024}, recovering, 32000 * MiB), "retarget cannot conceal additional RAM growth");
    require(!live_memory_supersedes({9, 31000, 256}, recovering, 32000 * MiB), "RAM pressure cannot supersede while reducing GPU reserve");
    require(!live_memory_supersedes({8, 31000, 1024}, recovering, 32000 * MiB), "reused acknowledgement ID cannot supersede");
    require(!live_memory_supersedes({9, 40000, 512}, recovering, 45000 * MiB), "identical target remains busy rather than resetting progress");
    const LiveMemoryRequest held{10, 32000, 320, true};
    require(!live_memory_gpu_growth_allowed(held, 32000 * MiB, 1536),
            "lease GPU hold forbids growth even with unchanged RAM and lower reserve");
    for (uint64_t free_mib : {0ull, 250ull, 320ull, 2048ull, 8192ull}) {
        const bool growth = live_memory_gpu_growth_allowed(held, 32000 * MiB, 1536);
        const auto budget = live_memory_gpu_budget(free_mib * MiB, 2048 * MiB, 320 * MiB, quantum, growth);
        require(budget <= 2048 * MiB, "hold remains a ceiling when external memory is freed between control and execution");
        if (free_mib < 320) require(budget < 2048 * MiB, "hold still permits pressure relief");
    }
    const bool held_growth = live_memory_gpu_growth_allowed(held, 32000 * MiB, 1536);
    const uint64_t after_shrink = live_memory_gpu_budget(0, 2048 * MiB, 320 * MiB, quantum, held_growth);
    require(live_memory_gpu_budget(8192 * MiB, after_shrink, 320 * MiB, quantum, held_growth) == after_shrink,
            "same held command cannot regrow a cache it already shrank");
    require(live_memory_supersedes({11, 32000, 512, true}, recovering, 32000 * MiB),
            "held pressure request may supersede old growth");
    require(live_memory_supersedes({11, 32000, 320, true}, {10, 32000, 320}, 32000 * MiB),
            "adding hold alone safely restricts an otherwise identical control");
    require(!live_memory_supersedes({11, 31000, 1024}, held, 32000 * MiB),
            "superseding a held control may not remove its GPU hold");
    require(!live_memory_supersedes({11, 32000, 320, true}, held, 32000 * MiB),
            "unchanged hold does not reset progress");
    require(!live_memory_supersedes({11, 33000, 512, true}, recovering, 32000 * MiB),
            "GPU hold is not permission to supersede with RAM growth");
    require(live_memory_gpu_growth_allowed({12, 32000, 320}, 32000 * MiB, 320),
            "separate ordinary recovery after lease release preserves legacy growth behavior");
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
void loan_donors() {
    using strata::core::live_prefill_donor;
    const int32_t slots[] = {0, -1, 5, 6, -1, 1};
    const float heat[] = {9, 2, 0, 20, 1, 10};
    bool backed[] = {true, true, true, false, true, true};
    auto held = [&](int32_t e) { return backed[e]; };
    require(live_prefill_donor(slots, heat, 6, 5, held) == 0,
            "loan donor prefers a GPU-backed duplicate over colder RAM-only experts");
    backed[0] = false;
    require(live_prefill_donor(slots, heat, 6, 5, held) == 5, "next donor stays outside the active loan");
    backed[5] = false;
    require(live_prefill_donor(slots, heat, 6, 5, held) == 4, "RAM-only fallback chooses the coldest measured donor");
    backed[4] = false; backed[3] = true;
    require(live_prefill_donor(slots, heat, 6, 5, held) == 1,
            "newly retained loan occupants cannot be evicted again in the same pass");
    require(live_prefill_donor(slots, nullptr, 6, 5, held) == -1,
            "without routing measurements, retain RAM-only experts and use file fallback");
    backed[1] = false;
    require(live_prefill_donor(slots, heat, 6, 5, held) == -1, "no eligible same-layer donor preserves file fallback");
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
    require(cache.slots() == 24 && cache.bytes() == (int64_t) offsets[24] && cache.mapped_bytes() == initial &&
            cache.full_slots() == 64 && cache.capacity() == 64 && cache.full_bytes() == (int64_t) offsets[64],
            "live open reports its active prefix and keeps full capacity separate");
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
    require(cache.bytes() == (int64_t) offsets[2] && cache.full_slots() == 64 &&
            cache.full_bytes() == (int64_t) offsets[64] && cache.slots_for_bytes(offsets[48]) == 48,
            "shrink reports active bytes while capacity and growth geometry remain intact");
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
    // The same pause coordinator as a real prompt: an in-flight reader references the old tail.
    // Its copy must finish before that mapping retires, while the replacement view lives in the
    // retained prefix. This checks real VMM lifetime ordering rather than just a mocked sequence.
    require(cudaMemcpyAsync(readback, original_view, 256, cudaMemcpyDeviceToHost, stream) == cudaSuccess,
            "queue old prompt-view reader before the resize boundary");
    bool prompt_active = true;
    require(strata::core::live_prefill_pause(prompt_active,
        [&](std::string&) { return cudaStreamSynchronize(stream) == cudaSuccess; },
        [&](std::string&) { return readback[0] == 0x23; }, // all residual consumers have completed
        [&](std::string& why) {
            return rebound_view != original_view && rebound_first >= 2 && cache.resize_live(new_slots, why);
        }, [] { return false; },
        [&](std::string&) { return cudaMemset(rebound_view, 0x6a, (size_t) loan_bytes) == cudaSuccess; }, err) && prompt_active,
        "drained prompt-view rebind precedes retirement of the old VMM tail: " + err);
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
    cache.close();
    require(!cache.valid() && !cache.live() && cache.slots() == 0 && cache.full_slots() == 0 &&
            cache.mapped_bytes() == 0, "close clears live and full cache accounting");
    cache.set_segment_bytes(8ll << 20);
    require(cache.open_sized(std::vector<int64_t>(32, 1ll << 20), 2, 32, err) &&
            !cache.live() && cache.segmented() && cache.slots() == 32 && cache.full_slots() == 32,
            "the cache can reopen in upstream segmented mode after releasing live mappings");
    require(cache.shrink(9ll << 20, err) && cache.slots() == 16 && cache.full_slots() == 32 &&
            cache.bytes() == (16ll << 20), "segmented prefix accounting survives live-cache reuse");
    require(cache.open_live(sizes, 2, 2, 32, err) && cache.live() && !cache.segmented() &&
            cache.slots() == 2 && cache.full_slots() == 64,
            "reopening live releases prior segmented mappings and restores full live geometry");
}
// Real mappings and stream lifetime with a policy-sized prefix. This fixture tests
// loan ownership/refill bytes, not model kernels or first-use MMQ allocations.
void pressure_prefix_arena() {
    using namespace strata::core;
    constexpr uint64_t MiB = 1ull << 20, loan = 16 * MiB;
    std::string err;
    ExpertCache cache;
    require(cache.open_live(std::vector<int64_t>(160, MiB), 160, 2, 80, err), "pressure prefix VMM open: " + err);
    const auto* offsets = cache.slot_offsets();
    void* const stable_base = cache.device_slot(0);
    std::vector<uint8_t> original(MiB);
    auto refill = [&](int64_t begin, int64_t end, std::string& why) {
        for (int64_t slot = begin; slot < end; ++slot) {
            std::fill(original.begin(), original.end(), (uint8_t) (slot + 1));
            if (!cache.fill_slot_blocking((int32_t) slot, original.data(), why, MiB)) return false;
        }
        return true;
    };
    require(refill(0, 160, err), "fill immutable per-slot fixture bytes");
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    uint8_t* output = nullptr;
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaHostAlloc((void**) &output, 256, cudaHostAllocDefault) == cudaSuccess, "pressure fixture stream/readback");
    require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess &&
            cudaMemcpyAsync(output, stable_base, 256, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamEndCapture(stream, &graph) == cudaSuccess &&
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess, "capture fixed-address resident read");
    for (const int64_t minimum : {0ll, 32ll}) {
        const uint64_t before = cache.committed_bytes();
        const int64_t normal = live_prefill_floor(offsets, cache.capacity(), loan);
        const int64_t floor = live_prefill_floor(offsets, cache.capacity(), loan, minimum);
        require(normal == 144 && floor == minimum + 16, "physical fixture exact minimum loan floor");
        require(live_prefill_control_floor(offsets, cache.slots(), normal, floor, cache.live_block_bytes(), true) == floor,
                "fresh pressure chooses a physically useful lower floor");
        const int64_t old_first = live_prefill_first(offsets, cache.slots(), loan);
        void* old_view = cache.device_slot(old_first);
        const int64_t keep = live_prefill_retained(offsets, floor, loan, minimum);
        const int64_t first = live_prefill_first(offsets, floor, loan, keep);
        void* const new_view = cache.device_slot(first); // arithmetic plan before unmapping
        require(keep == minimum && first == minimum, "minimum-sized cache borrows exactly the planned tail");
        require(cudaMemsetAsync(old_view, 0x5a, loan, stream) == cudaSuccess &&
                cudaMemcpyAsync(output, old_view, 256, cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "queue a real consumer of the old loan before retirement");
        bool active = true;
        std::vector<int32_t> residency(160);
        for (int i = 0; i < 160; ++i) residency[i] = i;
        require(live_prefill_pause(active,
            [&](std::string& why) {
                if (cudaStreamSynchronize(stream) == cudaSuccess && output[0] == 0x5a) return true;
                why = "old-view stream consumer did not observe its ordered write";
                return false;
            },
            [&](std::string& why) { return refill(old_first, cache.slots(), why); },
            [&](std::string& why) {
                for (auto& slot : residency) if (slot >= floor) slot = -1;
                return cache.resize_live(floor, why);
            }, [] { return false; },
            [&](std::string&) {
                for (auto& slot : residency) if (slot >= first) slot = -1;
                return cudaMemset(new_view, 0xa7, loan) == cudaSuccess;
            }, err) && active, "old reads and refill finish before the pressure loan retires mappings: " + err);
        require(cache.committed_bytes() == live_prefill_mapped_bytes(offsets[floor], cache.live_block_bytes()) &&
                cache.committed_bytes() < before && cache.device_slot(0) == stable_base,
                "real physical commitment matches rounded planner and preserves virtual base");
        for (int32_t slot : residency) require(slot < first, "no active residency points into borrowed or retired slots");
        if (minimum == 0)
            require(live_prefill_donor(residency.data(), nullptr, 160, 0, [](int32_t) { return true; }) == -1,
                    "keep0 cannot invent a GPU duplicate donor; immutable file fallback remains available");
        require(cudaDeviceSynchronize() == cudaSuccess && refill(first, floor, err), "refill every new borrower before decode");
        for (int64_t slot = first; slot < floor; ++slot) {
            std::fill(original.begin(), original.end(), (uint8_t) (slot + 1));
            require(cache.verify_slot((int32_t) slot, original.data(), err, MiB), "refilled bytes match immutable source exactly");
        }
        require(cudaGraphLaunch(exec, stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess && output[0] == 1,
                "captured graph reads correct expert bytes after zero-prefix loan/refill");
        require(cache.resize_live(160, err) && refill(floor, 160, err), "recovery regrows and fills expert tail");
        require(live_prefill_retained(offsets, cache.slots(), loan, minimum) == 128,
                "regrowth restores normal retention without enlarging minimum workspace");
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream); cudaFreeHost(output);
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
    require(source.resident_blob(0, 1) == first && source.resident_blob(0, 0) == nullptr &&
            source.resident_blob(-1, 1) == nullptr && source.resident_blob(2, 0) == nullptr,
            "the public resident accessor uses live block ownership and rejects absent or invalid experts");
    require(source.resize_live_resident(block * 2, block, 0, res, rank, done, err) && done,
            "second independent RAM block: " + err);
    require(source.blob(0, 1) == first && source.blob(1, 0)[0] == 5, "growth preserves old pointers and bytes");
    const uint64_t pinned = pin ? (mixed ? block : block * 2) : 0;
    require(source.pinned_bytes() == pinned && source.pinned(0, 1) == pin &&
            source.pinned(1, 0) == (pin && !mixed), "mixed backing reports actual pinned bytes and expert ownership");
    if (mixed)
        require(source.device_alias(1, 0) == nullptr && source.has_resident(1, 0),
                "pageable fallback retains resident bytes without advertising a mapped alias");
    // Non-block-aligned pressure target: 5,529,600 -> target 4,194,304 -> actual 2,764,800.
    // The 1,429,504-byte gap fits another expert (1,382,400), so the old next tick tried to grow
    // and failed its headroom guard. Exact block targets and headroom=0 do not expose this.
    const uint64_t rounded_target = 4ull << 20;
    require(block < rounded_target && rounded_target < block * 2 && rounded_target - block >= BLOB,
            "rounding fixture leaves room for the old unintended allocation");
    bool ram_done = false;
    int ram_steps = 0;
    auto pressure_step = [&] {
        return live_memory_ram_step(ram_done, [&](bool& reached) {
            ++ram_steps;
            return source.resize_live_resident(rounded_target, block, UINT64_MAX, res, rank, reached, err);
        });
    };
    require(pressure_step() && ram_done && source.resident_bytes() == block && source.blob(0, 1) == first,
            "nonaligned shrink completes below target even with unavailable positive headroom");
    for (int gpu_step = 0; gpu_step < 3; ++gpu_step)
        require(pressure_step() && ram_done && ram_steps == 1 && source.resident_bytes() == block,
                "later GPU steps retain RAM completion without refilling the rounded gap");
    bool new_growth_done = false;
    require(!live_memory_ram_step(new_growth_done, [&](bool& reached) {
                return source.resize_live_resident(rounded_target, block, UINT64_MAX, res, rank, reached, err);
            }) && !new_growth_done && source.resident_bytes() == block &&
            err == "live RAM: physical or commit headroom would be exceeded",
            "separate genuine growth remains an error and preserves the rounded-down layout");
    require(source.resize_live_resident(block * 2, block, 0, res, rank, done, err) && done,
            "restore fixture after pressure rounding regression");
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
    require(source.resize_live_resident(block, block, 0, res, rank, done, err) && done,
            "prepare a missing borrower and two resident donor slots");
    res[1] = 5; res[2] = 0;
    const float heat[] = {0, 9, 8, 0};
    const int32_t donor = live_prefill_donor(res.data(), heat, 4, 5,
        [&](int32_t e) { return source.has_resident(0, e); });
    const uint8_t* donor_ptr = source.blob(0, donor);
    const uint8_t* donor_device = source.device_alias(0, donor);
    const uint64_t pinned_before = source.pinned_bytes();
    require(donor == 2 && donor_ptr[0] == 3 && !source.has_resident(0, 1),
            "coverage selects the GPU-backed same-layer RAM slot rather than a cold CPU expert");
    std::memset(source.exchange_buffer(0), 2, (size_t) BLOB); // prepared borrower bytes, as after the drained D2H copy
    require(source.stage_exchange(0, donor, 1, 0) && source.blob(0, 1)[0] == 2 && donor_ptr[0] == 3,
            "staged coverage exposes prepared bytes without overwriting donor ownership before commit");
    require(source.commit_exchanges() == 1 && source.blob(0, 1) == donor_ptr &&
            source.device_alias(0, 1) == donor_device && source.pinned_bytes() == pinned_before &&
            source.resident_bytes() == block && !source.has_resident(0, donor),
            "coverage publication transfers block and alias ownership without growing either RAM counter");
    require(source.blob(0, donor)[0] == 3 && source.blob(0, 1)[0] == 2 && res[1] == 5 && res[2] == 0,
            "evicted donor retains immutable-file bytes and GPU residency remains authoritative");
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
        prompt_pause();
        pressure_direction();
        loan_geometry();
        loan_donors();
        if (argc <= 1 || std::strcmp(argv[1], "--protocol-only") != 0) {
            device_arena();
            pressure_prefix_arena();
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
