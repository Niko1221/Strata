#include "strata/qwen35/gfx803_batch_limits.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
void check(int chunk, int rows, int expected_chunk, int expected_rows) {
    strata::qwen35::clamp_gfx803_batch(chunk, rows);
    if (chunk != expected_chunk || rows != expected_rows) {
        std::fprintf(stderr, "gfx803 batch clamp: got chunk=%d rows=%d, expected %d,%d\n",
                     chunk, rows, expected_chunk, expected_rows);
        std::exit(1);
    }
}
}

int main() {
    check(128, 8, 8, 8);
    check(32, 64, 32, 32);
    check(0, 128, 0, 32);  // auto chunk is selected before this clamp
    check(1, 0, 1, 1);
    check(24, -4, 1, 1);
    check(16, 4, 4, 4);
    std::puts("q36_gfx803_batch_limits_test OK");
}
