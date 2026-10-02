// tests/core/remote_budget_test.cpp - the helper tier's pair selection, without a GPU or a model.
//
// With `--layer-split` every card the run already uses gives its leftover VRAM to a helper tier filled with the
// globally hottest pairs no stage cache can hold.  `select_remote_pairs` is the part of that which is pure
// arithmetic - walk the ranked profile, skip what CUDA0's cache (`primary`) or an earlier tier (`claimed`)
// already holds, and stop at a slot count or at a byte budget - so it is the part that can be checked here.
//
//   1. A REFERENCE WALK.  `expect()` restates the rule the simplest possible way, one pass and one accumulator,
//      and 400 random inputs are compared against the real function pair for pair, size for size, byte for byte.
//      This is what catches a `continue`/`break` swapped in the budget path, or an off-by-one in the aligned
//      total, which are the two mistakes this walk can make and neither of which a single hand-picked case sees.
//   2. THE NAMED PROPERTIES: never over budget; rank order preserved; `claimed` respected; and `slots` mode
//      keeping the pre-budget behaviour exactly (the first N unclaimed pairs, in rank order).
//   3. THE REASON A BUDGET SKIPS RATHER THAN STOPS.  A native pack's blobs differ per layer (IQ3_S: 2,176,000
//      bytes at layer 0, 1,510,400 at layer 1), so a hot blob too big for what is left must not hide the colder
//      pairs behind it that still fit.  Loaded from the real layout in tests/data/native_experts; in the
//      canonical layout every blob is the same size and the two behaviours coincide, so the case is only
//      meaningful - and only asserted - once a native layout has loaded.
#include "strata/core/expert_cache.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using P = std::pair<int32_t, int32_t>;
using strata::kernels::cpu::expert_layout;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-86s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct Result {
    std::vector<P> selected;
    std::vector<int64_t> sizes;
    uint64_t needed = 0;
};

/// The documented rule, written the obvious way.  Deliberately NOT sharing code with the implementation: the
/// point of a reference walk is that the two disagree when one of them is wrong.
Result expect(const std::vector<P>& ranked, const std::vector<uint8_t>& claimed, int64_t layers, int64_t experts,
              int64_t slots, int64_t byte_budget) {
    const auto& lay = expert_layout();
    Result r;
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const P& p : ranked) {
        if (p.first < 0 || p.first >= layers || p.second < 0 || p.second >= experts) continue;
        const size_t i = (size_t) p.first * (size_t) experts + (size_t) p.second;
        if (claimed[i] || picked[i]) continue;
        const int64_t blob = (int64_t) lay.blob_bytes(p.first);
        const int64_t cost = lay.native ? (blob + 255) / 256 * 256 : blob;
        if (byte_budget > 0 && (int64_t) r.needed + cost > byte_budget) continue;
        picked[i] = 1;
        r.selected.push_back(p);
        if (lay.native) r.sizes.push_back(blob);
        r.needed += (uint64_t) cost;
        if (slots > 0 && (int64_t) r.selected.size() >= slots) break;
    }
    return r;
}

bool same(const Result& a, const Result& b) {
    return a.selected == b.selected && a.sizes == b.sizes && a.needed == b.needed;
}

/// The real IQ3_S layout, so the walk sees blobs that differ per layer.  False if the data is not there.
bool load_native(const fs::path& data) {
    const fs::path tmp = fs::temp_directory_path() / "strata_remote_budget_native";
    std::error_code ec;
    fs::create_directories(tmp, ec);
    std::ifstream in(data / "iq3_s.txt", std::ios::binary);
    if (!in) return false;
    std::ofstream out(tmp / "native_experts.txt", std::ios::binary);
    out << in.rdbuf();
    out.close();
    std::string err;
    return strata::kernels::cpu::expert_layout_load(tmp.string(), 48, 512, err);
}
} // namespace

int main(int argc, char** argv) {
    std::printf("remote_budget_test: the helper tier's pair selection\n");
    const fs::path data = argc > 1 ? fs::path(argv[1]) : fs::path("tests/data/native_experts");
    const bool native = load_native(data);   // fills the process-wide layout the walk reads
    (void) native;
    const auto& lay = expert_layout();
    std::printf("  layout: %s%s\n", lay.native ? "native (IQ3_S)" : "canonical",
                lay.native ? "" : " - blob sizes are uniform, so the skip case is not asserted");

    // Never opened, so `slot_of` answers kNotResident for everything: the primary cache contributes nothing and
    // the `claimed` bitmap is the whole of "already held by someone else".
    strata::core::ExpertCache primary;

    const int64_t L = 6, E = 16;
    const size_t N = (size_t) L * E;
    const int64_t unit = (int64_t) lay.blob_bytes(0);
    const int64_t aligned = lay.native ? (unit + 255) / 256 * 256 : unit;

    // ---- 1. the differential walk
    std::mt19937 rng(20261001);
    int mismatch = 0;
    int nonempty = 0;
    for (int trial = 0; trial < 400; ++trial) {
        std::vector<P> ranked;
        ranked.reserve(N);
        for (int64_t l = 0; l < L; ++l)
            for (int64_t e = 0; e < E; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        std::shuffle(ranked.begin(), ranked.end(), rng);
        std::vector<uint8_t> claimed(N, 0);
        for (size_t i = 0; i < N; ++i)
            if (rng() % 5 == 0) claimed[i] = 1;
        const int64_t slots = (int64_t) (rng() % 12);
        // a quarter of the trials are slot-bounded, the rest budget-bounded (0 = unlimited)
        const int64_t budget = (rng() % 4 == 0) ? 0 : (int64_t) (rng() % 20) * aligned;

        Result got;
        strata::core::select_remote_pairs(ranked, primary, claimed, L, E, slots, budget, got.selected, got.sizes,
                                          got.needed);
        const Result want = expect(ranked, claimed, L, E, slots, budget);
        if (!same(got, want)) {
            if (mismatch < 3)
                std::printf("  trial %d MISMATCH: slots=%lld budget=%lld got %zu pairs/%llu bytes, "
                            "want %zu pairs/%llu bytes\n", trial, (long long) slots, (long long) budget,
                            got.selected.size(), (unsigned long long) got.needed, want.selected.size(),
                            (unsigned long long) want.needed);
            ++mismatch;
        }
        if (!got.selected.empty()) ++nonempty;
    }
    check(mismatch == 0, "400 random walks match the reference exactly (" + std::to_string(nonempty) +
                             " of them non-empty)");

    // ---- 2. the named properties, on one fixed input
    std::vector<P> ranked;
    for (int64_t l = 0; l < L; ++l)
        for (int64_t e = 0; e < E; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
    std::vector<uint8_t> claimed(N, 0);
    // the first three RANKS, so "the first N unclaimed" is a plain offset into `ranked`
    claimed[(size_t) (0 * E + 0)] = 1;
    claimed[(size_t) (0 * E + 1)] = 1;
    claimed[(size_t) (0 * E + 2)] = 1;

    std::vector<P> sel;
    std::vector<int64_t> sizes;
    uint64_t needed = 0;
    strata::core::select_remote_pairs(ranked, primary, claimed, L, E, 4, 0, sel, sizes, needed);
    bool first_four = sel.size() == 4;
    for (size_t i = 0; i < sel.size() && first_four; ++i)
        if (sel[i] != ranked[i + 3]) first_four = false;
    check(first_four, "slots mode takes the first N unclaimed pairs in rank order");
    bool none_claimed = true;
    for (const P& p : sel) none_claimed = none_claimed && claimed[(size_t) p.first * E + (size_t) p.second] == 0;
    check(none_claimed, "no pair a stage's cache or an earlier tier holds is selected");

    bool within = true;
    for (int64_t b = 1; b <= 24; ++b) {
        sel.clear(); sizes.clear(); needed = 0;
        strata::core::select_remote_pairs(ranked, primary, claimed, L, E, 0, b * aligned, sel, sizes, needed);
        within = within && needed <= (uint64_t) (b * aligned);
    }
    check(within, "no budget, from 1 to 24 blobs, is ever exceeded");

    sel.clear(); sizes.clear(); needed = 0;
    strata::core::select_remote_pairs(ranked, primary, claimed, L, E, 0, (int64_t) N * aligned * 4, sel, sizes,
                                      needed);
    check(sel.size() == N - 3, "a budget larger than the table takes every unclaimed pair");
    check(sizes.empty() || sizes.size() == sel.size(), "one size per selected pair, when the layout is native");

    // ---- 3. the skip, only where blobs really differ
    if (lay.native) {
        int64_t big = 0, small = 0;
        for (int64_t l = 1; l < 48; ++l) {
            if (lay.blob_bytes(l) > lay.blob_bytes(big)) big = l;
            if (lay.blob_bytes(l) < lay.blob_bytes(small)) small = l;
        }
        const int64_t small_cost = (int64_t) ((lay.blob_bytes(small) + 255) / 256 * 256);
        std::vector<P> two{P{(int32_t) big, 0}, P{(int32_t) small, 0}};
        std::vector<uint8_t> none((size_t) 48 * 512, 0);
        sel.clear(); sizes.clear(); needed = 0;
        strata::core::select_remote_pairs(two, primary, none, 48, 512, 0, small_cost, sel, sizes, needed);
        check(sel.size() == 1 && sel[0].first == small,
              "a budget that fits only the smaller blob skips the hotter larger one and still fills");
    } else {
        check(true, "the skip case needs a native layout (uniform blobs cannot express it)");
    }

    std::printf("%s\n", g_fail == 0 ? "remote_budget_test: all checks pass" : "remote_budget_test: FAILURES");
    return g_fail == 0 ? 0 : 1;
}
