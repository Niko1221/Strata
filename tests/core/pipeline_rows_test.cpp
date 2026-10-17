// include/strata/program/pipeline_rows.hpp: a pipelined group's window lists its active slots only, in order, with
// their tokens and positions; idle slots are untouched; the packed rows stay as they were when a slot changes later.
// Header-only, CPU.  Checks do not depend on assert (they hold in Release builds too).
#include "strata/program/pipeline_rows.hpp"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what, int count, int mask, int base) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (slots %d, active mask %d, base %d)\n", what, count, mask, base);
        ++failures;
    }
}
struct Slot {
    bool active;
    int32_t x;
    int64_t p;
    bool cached;
};
}  // namespace

int main() {
    for (int count : {2, 4, 8})
        for (int mask = 0; mask < (1 << count); ++mask) {
            std::array<Slot, 8> slots{};
            for (int b = 0; b < count; ++b) slots[(size_t) b] = {(mask & (1 << b)) != 0, 100 + b, 10000 + b, true};
            const auto before = slots;
            for (int gs : {1, 2, 4})
                for (int base = 0; base + gs <= count; base += gs) {
                    int rows[8]{};
                    int32_t tokens[8]{};
                    int64_t pos[8]{};
                    const int n = strata::program::pipeline_rows::pack(slots, base, gs, rows, tokens, pos);
                    int k = 0;
                    for (int b = base; b < base + gs; ++b)
                        if (slots[(size_t) b].active) {
                            check(k < n && rows[k] == b && tokens[k] == slots[(size_t) b].x &&
                                      pos[k] == slots[(size_t) b].p, "active slot packed in order", count, mask, base);
                            ++k;
                        }
                    check(k == n, "only the active slots", count, mask, base);
                    bool untouched = true;
                    for (int b = 0; b < count; ++b) untouched = untouched && slots[(size_t) b].cached == before[(size_t) b].cached;
                    check(untouched, "idle slots keep their cache flag", count, mask, base);
                    if (n > 0) {        // a slot that changes after the window was packed does not move its rows
                        const int first = rows[0];
                        slots[(size_t) first].active = !slots[(size_t) first].active;
                        check(rows[0] == first, "packed rows are the window's own copy", count, mask, base);
                        slots = before;
                    }
                }
        }
    if (failures == 0) std::printf("pipeline_rows_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
