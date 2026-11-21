// S2 probe (debug only, STRATA_KV_S2_STATS=<path>): the K channel axis before and after the Hadamard.
//
// The probe is outside the default path: with the env unset there is no D2H, no allocation and no
// change to the append.  It records the first four moments per channel (mean, variance, skew,
// excess kurtosis) and max |x|, split into the 64 RoPE-rotated and the 192 unrotated dimensions,
// per QSA layer.  The flush overwrites the file with the cumulative moments.
#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

namespace strata::core::s2 {

struct Moments {
    double n = 0;
    double s1[256] = {};
    double s2[256] = {};
    double s3[256] = {};
    double s4[256] = {};
    double mx[256] = {};
};

struct Layer {
    Moments pre, post;
    long long tokens = 0;
};

inline void accumulate(Moments& m, const float* host, long long rows, int nchan) {
    for (long long r = 0; r < rows; ++r) {
        const float* row = host + (size_t) r * (size_t) nchan;
        for (int c = 0; c < nchan; ++c) {
            const double x = (double) row[c];
            const double x2 = x * x;
            m.s1[c] += x;
            m.s2[c] += x2;
            m.s3[c] += x2 * x;
            m.s4[c] += x2 * x2;
            const double ax = std::fabs(x);
            if (ax > m.mx[c]) m.mx[c] = ax;
        }
    }
    m.n += (double) rows;
}

inline void flush(const char* path, const std::vector<Layer>& acc) {
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) return;
    std::fprintf(f, "layer  channel  group  n  mean_pre  var_pre  skew_pre  exkurt_pre  maxabs_pre  "
                    "mean_post  var_post  skew_post  exkurt_post  maxabs_post\n");
    for (size_t l = 0; l < acc.size(); ++l) {
        if (acc[l].tokens == 0 || acc[l].pre.n <= 0.0) continue;
        for (int c = 0; c < 256; ++c) {
            const char* grp = c < 64 ? "rot" : "unrot";
            double out[2][5] = {};
            const Moments* ms[2] = {&acc[l].pre, &acc[l].post};
            for (int k = 0; k < 2; ++k) {
                const Moments& m = *ms[k];
                const double n = m.n;
                const double mean = m.s1[c] / n;
                double var = m.s2[c] / n - mean * mean;
                if (var < 0.0) var = 0.0;
                const double sd = std::sqrt(var);
                double skew = 0.0, exkurt = 0.0;
                if (sd > 0.0) {
                    skew = (m.s3[c] / n - 3.0 * mean * (m.s2[c] / n) + 2.0 * mean * mean * mean) / (sd * sd * sd);
                }
                if (var > 0.0) {
                    exkurt = (m.s4[c] / n - 4.0 * mean * (m.s3[c] / n) + 6.0 * mean * mean * (m.s2[c] / n) -
                              3.0 * mean * mean * mean * mean) / (var * var) - 3.0;
                }
                out[k][0] = mean; out[k][1] = var; out[k][2] = skew; out[k][3] = exkurt; out[k][4] = m.mx[c];
            }
            std::fprintf(f, "%zu  %d  %s  %lld  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g  %.6g\n",
                         l, c, grp, acc[l].tokens, out[0][0], out[0][1], out[0][2], out[0][3], out[0][4],
                         out[1][0], out[1][1], out[1][2], out[1][3], out[1][4]);
        }
    }
    std::fclose(f);
}

}  // namespace strata::core::s2
