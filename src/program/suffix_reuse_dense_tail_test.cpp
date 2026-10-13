// suffix_reuse_dense_tail_test — FW-1: position-dependent checkpoint cadence.
//
// The cadence is a PURE function so it is testable without a GPU:
//   dense_tail_next_check(done, pp_total, every, dense_tail) -> the `done`
//   threshold at/after which the NEXT periodic checkpoint fires.
//
// Contract (plan M1, ruling 4 default-off):
//   - dense_tail == 0            -> EXACTLY today's cadence: next = done + every
//     (byte-identical behavior when the flag is off; pinned against
//     generate.cpp:8206 `pp_next_check = done + o.prompt_cache_every`).
//   - dense_tail > 0             -> in the last `dense_tail` tokens of the
//     prompt, cadence is `every` (dense: anchors near the tail where CLM edits
//     land); before that zone, cadence is max(every, 8192) (sparse: cheap
//     cold read, one deep rescue anchor).
//   - The zone is measured against pp_total as KNOWN when the decision is
//     made.  The prompt pass discovers pp_total incrementally (streams), so
//     the function must behave sanely when pp_total grows between calls: it
//     only ever returns a threshold; the caller checks `done >= threshold`.
//
// RED state: dense_tail_next_check does not exist yet (compile FAIL).

#include "strata/program/suffix_reuse.hpp"

#include <cstdio>

namespace suffix = strata::program::suffix_reuse;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

int main() {
    std::printf("=== suffix_reuse_dense_tail_test ===\n");

    // --- DTT1: off (0) is byte-identical to today's cadence ---
    {
        // today: after a checkpoint at `done`, next fires at done + every
        check(suffix::dense_tail_next_check(/*done=*/1000, /*pp_total=*/5670,
                                            /*every=*/1024, /*dense_tail=*/0)
                  == 1000 + 1024,
              "DTT1: dense_tail=0 -> next = done + every (today's cadence)");
        // multiple steps walk the same ladder as the live loop
        int64_t next = 0;
        int steps = 0;
        while (next < 5670 && steps < 100) {
            const int64_t t = suffix::dense_tail_next_check(next, 5670, 1024, 0);
            if (t <= next) break;
            next = t;
            ++steps;
        }
        check(next == 6 * 1024, "DTT1: ladder from 0 reaches 6144 (6*1024) like the live loop");
    }

    // --- DTT2: in the dense zone, cadence is `every` ---
    {
        // pp_total 5670, dense_tail 2048 -> zone starts at 3622.
        // A checkpoint at done=4096 is inside the zone: next fires 1024 later.
        check(suffix::dense_tail_next_check(4096, 5670, 1024, 2048) == 4096 + 1024,
              "DTT2: inside the dense zone -> next = done + every");
    }

    // --- DTT3: before the zone, cadence is max(every, 8192) ---
    {
        // long prompt (20000): sparse step from 1024 lands at 9216, well
        // before the zone at 17952 -- the pure sparse-cadence case.
        check(suffix::dense_tail_next_check(1024, 20000, 1024, 2048) == 1024 + 8192,
              "DTT3: before the zone, every=1024 -> sparse step max(every,8192)=8192");
        // every LARGER than 8192 is respected (user asked for sparser head)
        check(suffix::dense_tail_next_check(0, 20000, 16384, 2048) == 16384,
              "DTT3: before the zone, every=16384 -> sparse step 16384 (user's every wins when bigger)");
    }

    // --- DTT4: boundary at exactly pp_total - dense_tail ---
    {
        const int64_t zone_start = 5670 - 2048;   // 3622
        // done exactly at the boundary: dense cadence applies from here
        check(suffix::dense_tail_next_check(zone_start, 5670, 1024, 2048) == zone_start + 1024,
              "DTT4: done == zone start -> dense cadence");
        // done one token before the boundary: the sparse next would overshoot
        // the zone (3621+8192), so it clamps to the zone start -- an anchor
        // lands exactly where the dense zone begins (same rule as DTT5).
        check(suffix::dense_tail_next_check(zone_start - 1, 5670, 1024, 2048) == zone_start,
              "DTT4: done == zone start - 1 -> clamps to zone start (never skip the zone)");
    }

    // --- DTT5: a sparse next that lands INSIDE the zone must not be skipped past ---
    {
        // done=0, zone starts 3622: sparse next = 8192, but the prompt ends at
        // 5670 -- the dense zone [3622,5670) would be skipped entirely if the
        // caller only woke at 8192.  The function must clamp sparse nexts into
        // the zone start when they would overshoot the zone.
        check(suffix::dense_tail_next_check(0, 5670, 1024, 2048) == 5670 - 2048,
              "DTT5: sparse next overshooting the zone clamps to zone start (no skipped anchors)");
    }

    // --- DTT6: growing pp_total (streams) ---
    {
        // decided at done=1024 with pp_total=2000 (not yet in any zone):
        // sparse next 9216.  Later pp_total grows to 5670; the caller re-asks
        // with the SAME done: the answer may differ (zone clamp), but must
        // never be in the past and never skip the zone.
        const int64_t a = suffix::dense_tail_next_check(1024, 2000, 1024, 2048);
        const int64_t b = suffix::dense_tail_next_check(1024, 5670, 1024, 2048);
        check(a > 1024 && b > 1024, "DTT6: next is always in the future");
        check(b <= 5670 - 2048 + 1024, "DTT6: re-ask after growth never skips the dense zone");
        // pp_total smaller than dense_tail: everything is dense (tiny prompts)
        check(suffix::dense_tail_next_check(0, 1000, 1024, 2048) == 1024,
              "DTT6: pp_total < dense_tail -> whole prompt dense");
    }

    // --- DTT7: validator integration ---
    {
        using suffix::validate_suffix_reuse_options;
        // dense_tail without prompt_cache_every refuses
        check(!validate_suffix_reuse_options(10, 6, "int8", 0, false, /*every=*/0, /*dense_tail=*/2048).empty(),
              "DTT7: --prompt-cache-dense-tail without --prompt-cache-every refuses");
        // dense_tail with everything sane passes (int8 accepted)
        check(validate_suffix_reuse_options(10, 6, "int8", 0, false, 1024, 2048).empty(),
              "DTT7: dense_tail with every>0 and supported kv passes");
    }

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
