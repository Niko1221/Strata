// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/dsv41/vendor/moe_win.cu - the window's expert grouping kernel.
// Extracted from Project Maya's src/kernels/cuda/glm_fast.cu (MIT, see third_party/project-maya/NOTICE.md).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "strata/dsv41/moe_win.hpp"
#include <cstdio>
#include <cstdlib>

namespace strata::dsv41k::glmf {
namespace {
constexpr int kWinEnt = kMaxWinTokens * 8;

// the entries grouped by expert slot (in order of first appearance), each group's members in entry order (t * 8 + i)
// (the tables in shared memory: as locals they were a 2.6 KB stack a thread, which the driver reserves for every thread
// the device can hold - ~420 MB on a V100 - and a launch failed "out of memory" once the expert pool had the rest)
__dpct_inline__ void moe_win_group_kernel(const MoeWin w) {
    auto &gp = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        unsigned long long[kWinEnt]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &cnt =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kWinEnt]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &gof =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kWinEnt]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &fill =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[kWinEnt + 1]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) !=
        0) return;
    int ng = 0;
#pragma unroll
    for (int t = 0; t < w.T; ++t)
        for (int i = 0; i < 8; ++i) {
            const int e = t * 8 + i;
            gof[e] = -1;
            const unsigned long long p = i < w.k ? w.plan_ptr[e] : 0ull;
            if (p == 0ull) continue;
            int g = 0;
            while (g < ng && gp[g] != p) ++g;
            if (g == ng) {
                gp[ng] = p;
                cnt[ng++] = 0;
            }
            ++cnt[g];
            gof[e] = g;
        }
    fill[0] = 0;
#pragma unroll
    for (int g = 0; g < ng; ++g) {
        w.grp_ptr[g] = gp[g];
        w.grp_start[g] = fill[g];
        fill[g + 1] = fill[g] + cnt[g];
    }
    w.grp_start[ng] = fill[ng];
#pragma unroll
    for (int e = 0; e < w.T * 8; ++e)
        if (gof[e] >= 0) w.grp_ent[fill[gof[e]]++] = e;
    *w.n_groups = ng;
}
}  // namespace

void moe_win_group(const MoeWin &w, dpct::queue_ptr s) {
    s->parallel_for(
        sycl::nd_range<3>(sycl::range(1, 1, 32), sycl::range(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            moe_win_group_kernel(w);
        });
    /*
    DPCT1010:340: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1000:339: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009:341: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        std::fprintf(stderr, "dsv41: moe_win_group: %s\n",
                     dpct::get_error_string_dummy(e));
        /*
        DPCT1001:338: The statement could not be removed.
        */
        std::abort();
    }
}
}  // namespace strata::dsv41k::glmf
