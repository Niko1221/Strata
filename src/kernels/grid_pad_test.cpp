// src/kernels/grid_pad_test.cpp - grid_pad's rule (no GPU needed): which launches get empty blocks and how many.
#include "strata/kernels/grid_pad.hpp"

#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::kernels::grid_pad_rule;
    constexpr unsigned S = 2048;   // R9700 / 9070 XT: 32 WGPs x 64 wave slots
    // the measured stalls (4 waves a block): each padded to 16 x k waves past the k-th multiple
    CHECK(grid_pad_rule(512, 4, S) == 516);      // 2048 waves (512-row projections, 1 column): +16 waves
    CHECK(grid_pad_rule(2560, 4, S) == 2580);    // 10240 waves (ssm_out, 1 column): +80
    CHECK(grid_pad_rule(3072, 4, S) == 3096);    // 12288 waves (attn_gate, 2-4 columns): +96
    CHECK(grid_pad_rule(5120, 4, S) == 5160);    // 20480 waves (qkv, 2 columns): +160
    CHECK(grid_pad_rule(125, 16, S) == 129);     // ROCm #6689's probe: 125-128 blocks of 16 waves slow, 129 fast
    CHECK(grid_pad_rule(128, 16, S) == 129);
    CHECK(grid_pad_rule(2047, 1, S) == 2064);    // one-wave blocks just below / at / past 2048
    CHECK(grid_pad_rule(2049, 1, S) == 2064);
    CHECK(grid_pad_rule(6144, 1, S) == 6192);    // ROCm #6689: 6144 needed +32..64
    CHECK(grid_pad_rule(49152, 1, S) == 49408);  // the padding stops growing at 256 waves
    // launches that were not slow stay as they are
    CHECK(grid_pad_rule(124, 16, S) == 124);
    CHECK(grid_pad_rule(129, 16, S) == 129);
    CHECK(grid_pad_rule(2032, 1, S) == 2032);    // 16 waves below: outside the window
    CHECK(grid_pad_rule(1280, 4, S) == 1280);    // 5120 waves = 2.5 multiples
    CHECK(grid_pad_rule(320, 4, S) == 320);
    CHECK(grid_pad_rule(1, 1, S) == 1);
    CHECK(grid_pad_rule(1000, 4, 0) == 1000);    // no slots (padding off)
    CHECK(grid_pad_rule(0, 4, S) == 0);
    // idempotent: a padded grid is not padded again
    for (unsigned b : {512u, 2560u, 3072u, 5120u, 128u}) CHECK(grid_pad_rule(grid_pad_rule(b, 4, S), 4, S) == grid_pad_rule(b, 4, S));
    if (fails) return 1;
    std::printf("grid_pad_test: ok\n");
    return 0;
}
