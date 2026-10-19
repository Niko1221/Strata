// metal/tests/rope_parity.cpp - Metal host-side RoPE table parity test.
// Verifies float64 reference trig and RopeScaling (none, linear, YaRN) against
// the Metal engine's inline build_rope_table implementation in engine/rope_table.h.

#include "engine/rope_table.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: rope_parity [--selftest]\n"); return 2; }
    }

    const int n_rot = 64;          // dimension_count
    const double theta = 1.0e7;    // freq_base
    const int max_pos = 32;
    const int half = n_rot / 2;
    int bad = 0;

    // ---- 1. Unscaled table against float64 reference
    std::vector<float> hcos((size_t) max_pos * half), hsin((size_t) max_pos * half);
    strata::metal_engine::build_rope_table(n_rot, theta, max_pos, hcos.data(), hsin.data());
    long long table_bad = 0;
    double table_worst = 0.0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            const float rc = (float) std::cos(ang), rs = (float) std::sin(ang);
            if (std::memcmp(&rc, &hcos[(size_t) p * half + i], 4) != 0) ++table_bad;
            if (std::memcmp(&rs, &hsin[(size_t) p * half + i], 4) != 0) ++table_bad;
            table_worst = std::max(table_worst, std::fabs((double) rc - (double) hcos[(size_t) p * half + i]));
        }
    }
    std::printf("  unscaled rope table vs float64 reference   %s (%lld of %d entries differ, worst %.1e)\n",
                table_bad ? "*** WRONG ***" : "bit-exact", table_bad, max_pos * half * 2, table_worst);
    bad += (int) table_bad;

    // ---- 2. Scaled table (None vs unscaled overload)
    std::vector<float> nc((size_t) max_pos * half), ns((size_t) max_pos * half);
    strata::kernels::RopeScaling none;
    none.freq_base = theta;
    strata::metal_engine::build_rope_table(n_rot, none, max_pos, nc.data(), ns.data());
    const long long none_bad = (long long) (std::memcmp(nc.data(), hcos.data(), nc.size() * 4) != 0) +
                               (long long) (std::memcmp(ns.data(), hsin.data(), ns.size() * 4) != 0);
    std::printf("  scaled table (type None) vs unscaled       %s\n", none_bad ? "*** WRONG ***" : "bit-identical");
    bad += (int) none_bad;

    // ---- 3. Linear scaling (factor 4)
    strata::kernels::RopeScaling lin;
    lin.type = strata::kernels::RopeScalingType::Linear;
    lin.freq_base = theta;
    lin.factor = 4.0;
    std::vector<float> lc((size_t) max_pos * half), ls((size_t) max_pos * half);
    strata::metal_engine::build_rope_table(n_rot, lin, max_pos, lc.data(), ls.data());
    long long lin_bad = 0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv / 4.0;
            const float rc = (float) std::cos(ang), rs = (float) std::sin(ang);
            if (std::memcmp(&rc, &lc[(size_t) p * half + i], 4) != 0) ++lin_bad;
            if (std::memcmp(&rs, &ls[(size_t) p * half + i], 4) != 0) ++lin_bad;
        }
    }
    std::printf("  scaled table (Linear factor 4)            %s (%lld of %d entries differ)\n",
                lin_bad ? "*** WRONG ***" : "bit-exact", lin_bad, max_pos * half * 2);
    bad += (int) lin_bad;

    // ---- 4. YaRN scaling (factor 4)
    strata::kernels::RopeScaling yarn;
    yarn.type = strata::kernels::RopeScalingType::YaRN;
    yarn.freq_base = theta;
    yarn.factor = 4.0;
    yarn.ext_factor = 1.0;
    std::vector<float> yc((size_t) max_pos * half), ys((size_t) max_pos * half);
    strata::metal_engine::build_rope_table(n_rot, yarn, max_pos, yc.data(), ys.data());
    const double fs = yarn.freq_scale();
    const double ms = yarn.mscale();
    double cd[2];
    yarn.corr_dims(n_rot, cd);
    long long yarn_bad = 0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;
            const double interp = fs * extrap;
            const double ramp = (double) strata::kernels::rope_yarn_ramp((float) cd[0], (float) cd[1], i) * yarn.ext_factor;
            const double ang = interp * (1.0 - ramp) + extrap * ramp;
            const float rc = (float) (std::cos(ang) * ms), rs = (float) (std::sin(ang) * ms);
            if (std::memcmp(&rc, &yc[(size_t) p * half + i], 4) != 0) ++yarn_bad;
            if (std::memcmp(&rs, &ys[(size_t) p * half + i], 4) != 0) ++yarn_bad;
        }
    }
    std::printf("  scaled table (YaRN factor 4)              %s (%lld of %d entries differ)\n",
                yarn_bad ? "*** WRONG ***" : "bit-exact", yarn_bad, max_pos * half * 2);
    bad += (int) yarn_bad;

    std::printf("\nrope: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("rope_parity OK\n");
    return 0;
}
