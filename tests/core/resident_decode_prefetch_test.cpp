#include "strata/core/resident_decode_prefetch.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

int main() {
    using namespace strata::core;
    size_t checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("resident decode invariant failed"); };
    ResidentDecodePrefetch mode;
    mode.serving = true;
    std::vector<int32_t> gpu{0, -1, 1, -1};
    std::vector<bool> ram{false, true, false, true}, staged(4, false);
    auto eligible = [&](ResidentDecodePrefetch m) {
        return resident_decode_needs_no_prediction(m, gpu, 4, [&](size_t i) { return ram[i]; });
    };
    auto covered = [&] {
        for (size_t i = 0; i < gpu.size(); ++i)
            if (gpu[i] < 0 && !ram[i] && !staged[i]) return false;
        return true;
    };
    check(eligible(mode));
    // Startup can reserve GPU slots permanently after the RAM complement is built.
    // The policy must see the final table, not cache the earlier complete cover.
    gpu[0] = -1;
    check(!eligible(mode));
    ram[0] = true;
    check(eligible(mode));
    gpu[0] = 0; ram[0] = false;
    auto m = mode; m.serving = false; check(!eligible(m));
    m = mode; m.batched = true; check(!eligible(m));
    m = mode; m.elastic_kv = true; check(!eligible(m));
    m = mode; m.independent_consumer = true; check(!eligible(m));
    m = mode; m.explicit_lookahead = true; check(!eligible(m));
    ram[1] = false; check(!eligible(mode)); ram[1] = true;
    check(!resident_decode_needs_no_prediction(mode, gpu, 5, [&](size_t i) { return ram.at(i); }));
    check(!resident_decode_needs_no_prediction(mode, {}, 0, [](size_t) { return true; }));

    // Model the actual exchange ordering: D2H save -> fence GPU victim -> new GPU
    // expert -> RAM commit. A staged copy covers the victim during both middle phases.
    // This checks the lifecycle invariant, not CUDA event scheduling or memory visibility.
    ram[2] = true; // A duplicate RAM copy exercises swaps that need no exchange.
    for (int round = 0; round < 20000; ++round) {
        size_t out = (size_t) (std::find_if(gpu.begin(), gpu.end(), [](int r) { return r >= 0; }) - gpu.begin());
        if (round & 1) for (size_t i = out; i < gpu.size(); ++i) if (gpu[i] >= 0) out = i;
        const size_t in = (size_t) (std::find(gpu.begin(), gpu.end(), -1) - gpu.begin());
        const int slot = gpu[out];
        check(covered() && ram[in]);
        const bool exchange = !ram[out];
        if (exchange) staged[out] = true;
        gpu[out] = -1; check(covered());
        gpu[in] = slot; check(covered());
        if (exchange) { ram[out] = true; ram[in] = false; staged[out] = false; }
        check(covered() && eligible(mode));
        if (round % 13 == 0) {
            // Serial prompt can temporarily need files, but must return the same loan
            // before any decode. Concurrent prompt/decode therefore cannot use this proof.
            const size_t loan = in;
            gpu[loan] = -1;
            if (!ram[loan]) check(!covered());
            gpu[loan] = slot; check(covered());
        }
    }
    const auto it = std::find_if(gpu.begin(), gpu.end(), [](int r) { return r >= 0; });
    const size_t victim = (size_t) (it - gpu.begin());
    ram[victim] = false;gpu[victim] = -1;
    check(!covered()); // Elastic KV can permanently break the invariant, hence exclusion.
    std::printf("PASS: %zu checks; 20000 modeled exchange cycles; consumer/override and lifetime exclusions\n", checks);
}
