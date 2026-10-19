#pragma once

#include "strata/kernels/rope_scaling.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

namespace strata::metal_engine {

namespace detail {

// The table's math, written once and shared by every layout, so the decode window and the parity test
// cannot drift apart.  Writer(p, i, cos, sin) receives the float32 pair for that entry.
//
// `inv` depends only on the pair index, so it is computed once per table rather than once per entry:
// at a 524,288-token context the window builds 524,368 positions x 32 pairs, and recomputing it per
// entry would be 16.8 M std::pow calls for a value that is one of 32.  The angle itself still depends
// on the position, so the trig stays in the loop.
template <class Writer>
inline void rope_table(int n_rot, const strata::kernels::RopeScaling& sc, int max_pos, Writer&& w) {
    const int half = n_rot / 2;
    std::vector<double> inv((size_t) half);
    for (int i = 0; i < half; ++i) inv[(size_t) i] = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);

    if (sc.type == strata::kernels::RopeScalingType::None) {
        for (int p = 0; p < max_pos; ++p)
            for (int i = 0; i < half; ++i) {
                const double ang = (double) p * inv[(size_t) i];
                w(p, i, (float) std::cos(ang), (float) std::sin(ang));
            }
        return;
    }

    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;
    for (int p = 0; p < max_pos; ++p)
        for (int i = 0; i < half; ++i) {
            const double extrap = (double) p * inv[(size_t) i];
            const double interp = fs * extrap;
            double ang = interp;
            if (correct) {
                const double ramp = (double) strata::kernels::rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            w(p, i, (float) (std::cos(ang) * ms), (float) (std::sin(ang) * ms));
        }
}

}  // namespace detail

// The trained angles, exactly: the same table the CUDA/HIP/SYCL host builders produce.
inline void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    strata::kernels::RopeScaling sc;
    sc.freq_base = theta;   // type None: no scaling, no magnitude correction
    const int half = n_rot / 2;
    detail::rope_table(n_rot, sc, max_pos, [&](int p, int i, float c, float s) {
        cos_tab[(size_t) p * half + i] = c;
        sin_tab[(size_t) p * half + i] = s;
    });
}

// The same table with a resolved scaling (none, linear or yarn) folded into the cos/sin contents.
inline void build_rope_table(int n_rot, const strata::kernels::RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    detail::rope_table(n_rot, sc, max_pos, [&](int p, int i, float c, float s) {
        cos_tab[(size_t) p * half + i] = c;
        sin_tab[(size_t) p * half + i] = s;
    });
}

// The decode window's layout: float2 [pos][half], (cos, sin) interleaved, which is what the MSL
// kernels read out of `cs_`.  Built in place: the window has the destination in hand, and staging it
// planar first would cost 134 MB of transient host memory at a 524,288-token context for nothing.
inline void build_rope_table_interleaved(int n_rot, const strata::kernels::RopeScaling& sc, int max_pos, float* dst) {
    const int half = n_rot / 2;
    detail::rope_table(n_rot, sc, max_pos, [&](int p, int i, float c, float s) {
        dst[((size_t) p * half + i) * 2] = c;
        dst[((size_t) p * half + i) * 2 + 1] = s;
    });
}

}  // namespace strata::metal_engine
