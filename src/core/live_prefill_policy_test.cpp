#include "strata/core/live_prefill.hpp"
#include "strata/core/live_memory.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
int checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}

void parsing() {
    using namespace strata::core;
    int64_t value = 77;
    for (const char* bad : {"", "-1", "+1", "129", "1.0", " 0", "0 ", "0x0", "no", "999999999999999999999"})
        require(!live_prefill_parse_min_retained(bad, value) && value == 77, "bad argument cannot publish a value");
    for (int n : {0, 16, 32, 128})
        require(live_prefill_parse_min_retained(std::to_string(n), value) && value == n, "valid strict argument");
    require(!live_prefill_kv_grow_requested(false, nullptr), "omitted environment uses fixed CLI");
    require(live_prefill_kv_grow_requested(true, nullptr), "CLI elastic KV is rejected");
    require(live_prefill_kv_grow_requested(false, "1"), "environment elastic KV overrides CLI");
    require(!live_prefill_kv_grow_requested(true, "0"), "environment fixed KV overrides CLI exactly as the session does");
    require(live_prefill_kv_grow_requested(true, ""), "empty environment uses CLI");
    require(live_prefill_kv_grow_requested(false, "yes"), "session treats any nonzero leading character as enabled");
}

void geometry() {
    using namespace strata::core;
    std::vector<uint64_t> off(194);
    for (size_t i = 1; i < off.size(); ++i) off[i] = off[i - 1] + 1 + (i * 37) % 23;
    for (int minimum : {0, 16, 32, 128}) {
        for (int slots = 1; slots < (int) off.size(); ++slots) {
            for (uint64_t bytes : {1ull, 17ull, 128ull, 256ull, 511ull, 900ull, 4096ull}) {
                int64_t oracle = -1;
                // Independent linear oracle; it does not use either loan helper's binary search.
                for (int k = minimum; k <= 128 && k < slots; ++k)
                    if (off[slots] - off[k] >= bytes) oracle = k;
                const auto keep = live_prefill_retained(off.data(), slots, bytes, minimum);
                require(keep == oracle, "largest retained prefix agrees with brute force sized-offset oracle");
                if (keep >= 0) {
                    const auto first = live_prefill_first(off.data(), slots, bytes, keep);
                    require(first >= keep && off[slots] - off[first] >= bytes, "whole prompt scratch fits");
                }
                if (minimum == 128)
                    require(keep == (live_prefill_first(off.data(), slots, bytes) >= 0 ? 128 : -1),
                            "omitted/default128 preserves the previous plan");
            }
        }
    }
    require(live_prefill_retained(off.data(), 193, 1, -1) == -1 &&
            live_prefill_retained(off.data(), 193, 1, 129) == -1, "invalid minimum rejected before offset access");
    require(live_prefill_first(off.data(), 1, 1, -1) == -1 &&
            live_prefill_floor(off.data(), 193, 1, -1) == -1, "negative retention cannot index before offsets");
}

void physical_pressure() {
    using namespace strata::core;
    constexpr uint64_t MiB = 1ull << 20, quantum = 32 * MiB, loan = 16 * MiB;
    std::vector<uint64_t> off(161);
    for (size_t i = 0; i < off.size(); ++i) off[i] = i * MiB;
    const auto normal = live_prefill_floor(off.data(), 160, loan);
    const auto pressure = live_prefill_floor(off.data(), 160, loan, 0);
    require(normal == 144 && pressure == 16, "separate exact startup and pressure floors");
    require(live_prefill_mapped_bytes(off[normal], quantum) == 160 * MiB &&
            live_prefill_mapped_bytes(off[pressure], quantum) == 32 * MiB, "whole physical block geometry");
    require(live_prefill_mapped_bytes(off[159], quantum) == live_prefill_mapped_bytes(off[144], quantum),
            "logical slot reduction within one block releases no physical bytes");
    require(live_prefill_control_floor(off.data(), 160, normal, pressure, quantum, false) == normal,
            "healthy RAM-only control preserves normal retained prefix");
    require(live_prefill_control_floor(off.data(), 160, normal, pressure, quantum, true) == pressure,
            "fresh GPU deficit can lower the floor");
    require(live_prefill_control_floor(off.data(), 160, 144, 140, quantum, true) == 144,
            "no lowered floor without useful physical release");
    const int64_t current = 32;
    const int64_t floor = live_prefill_control_floor(off.data(), current, normal, pressure, quantum, false);
    const auto budget = live_memory_gpu_budget(4096 * MiB, 32 * MiB, 576 * MiB, quantum, false);
    require(floor == current && budget == 32 * MiB, "healthy hold below normal floor does not regrow");
    require(live_prefill_retained(off.data(), current, loan, 0) == 16, "reduced geometry retains every slot that fits");
    require(live_prefill_retained(off.data(), 144, loan, 0) == 128, "legitimate recovery restores normal retention");
    require(live_prefill_mapped_bytes(UINT64_MAX, quantum) == UINT64_MAX &&
            live_prefill_mapped_bytes(3, 0) == UINT64_MAX, "invalid rounded sizes cannot wrap into a small commitment");
}

void transactional_rebind() {
    using namespace strata::core;
    for (int failure = 0; failure < 5; ++failure) {
        int raw_view = 19, apply_calls = 0, restore_calls = 0;
        std::string err;
        const auto result = live_prefill_rebind([&](std::string& e) {
            ++apply_calls;
            raw_view = 7; // equivalent to a partially mutated Prefill carve
            if (failure == 3) throw std::runtime_error("partial carve exception");
            e = "chosen layout failed";
            return failure == 0;
        }, [&](std::string& e) {
            ++restore_calls;
            if (failure == 4) throw std::runtime_error("rollback exception");
            if (failure == 2) { e = "rollback failed"; return false; }
            raw_view = 19;
            return true;
        }, err);
        require(apply_calls == 1, "chosen layout is committed once, never probed or retried");
        if (!failure) require(result == LivePrefillRebind::applied && restore_calls == 0 && raw_view == 7,
                              "successful relayout publishes chosen views without rollback");
        else if (failure == 1 || failure == 3)
            require(result == LivePrefillRebind::restored && restore_calls == 1 && raw_view == 19,
                    "partial mutation failure restores old views before control returns");
        else require(result == LivePrefillRebind::invalid && restore_calls == 1,
                     "double failure is fatal, never a successful control or generation");
    }
}
}

int main() {
    try {
        parsing(); geometry(); physical_pressure(); transactional_rebind();
        std::printf("live prefill policy: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
