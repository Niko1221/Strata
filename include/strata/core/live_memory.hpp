#pragma once

#include <charconv>
#include <cstdint>
#include <sstream>
#include <string>

namespace strata::core {

struct LiveMemoryRequest {
    uint64_t id = 0, resident_mib = 0, vram_reserve_mib = 0;
};

// Completion belongs to the whole request. In particular, a shrink may finish below its
// target after releasing a whole block; later GPU steps must not refill that rounded gap.
template<class Resize>
bool live_memory_ram_step(bool& done, Resize resize) {
    if (done) return true;
    bool reached = false;
    if (!resize(reached)) return false;
    done = reached;
    return true;
}

// Capture once at admission. Releasing RAM can increase reported VRAM free space, but a pressure request
// must not turn that new space into a GPU allocation on its next tick.
inline bool live_memory_gpu_growth_allowed(const LiveMemoryRequest& request, uint64_t resident_bytes,
                                           uint64_t previous_reserve_mib) {
    // The protocol reports whole MiB: its unchanged value must not become a pressure signal because of
    // the unreported fractional MiB in the last expert block.
    return request.resident_mib >= (resident_bytes >> 20) && request.vram_reserve_mib <= previous_reserve_mib;
}

inline uint64_t live_memory_gpu_budget(uint64_t free_bytes, uint64_t committed, uint64_t reserve,
                                       uint64_t quantum, bool allow_growth) {
    const uint64_t available = free_bytes + committed;
    const uint64_t target = available > reserve ? (available - reserve) / quantum * quantum : 0;
    if (target <= committed) return target;
    if (!allow_growth) return committed;
    // Leave an additional mapping granule only for growth: WDDM can charge a touched mapping differently
    // from its pre-allocation free-memory estimate. Do not force extra eviction to create this margin.
    const uint64_t spare = free_bytes > reserve + quantum ? free_bytes - reserve - quantum : 0;
    return committed + spare / quantum * quantum;
}

// Exact grammar; unsigned extraction alone would accept a negative value by wrapping it.
inline bool parse_live_memory_request(const std::string& line, LiveMemoryRequest& out) {
    std::istringstream in(line);
    std::string verb, id, ram, vram, extra;
    if (!(in >> verb >> id >> ram >> vram) || verb != "MEMORY" || (in >> extra)) return false;
    auto number = [](const std::string& s, uint64_t& value) {
        if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
        const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
        return result.ec == std::errc{} && result.ptr == s.data() + s.size();
    };
    LiveMemoryRequest next;
    if (!number(id, next.id) || !number(ram, next.resident_mib) || !number(vram, next.vram_reserve_mib) ||
        next.id == 0 || next.resident_mib > (1ull << 20) || next.vram_reserve_mib > (1ull << 20)) return false;
    out = next;
    return true;
}

} // namespace strata::core
