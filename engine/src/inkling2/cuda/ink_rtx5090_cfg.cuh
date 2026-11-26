// engine/src/inkling2/cuda/ink_rtx5090_cfg.cuh - Per-device OFF-by-default RTX 5090 (sm_120) feature switches
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// All RTX 5090 (sm_120) speed optimizations are strictly OFF by default and enabled only when
// STRATA_INK_RTX5090=1 (or an explicit granular STRATA_INK_SM120_*=1 sub-switch) is set in the
// environment, AND only on devices whose compute capability is sm_120 (major == 12 && minor == 0),
// cached per device via strata::core::PerDeviceFlag.
#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include "strata/core/ink_caps.hpp"

namespace strata::ink::rtx5090 {

inline int env_tri_state(const char* name) {
    const char* e = std::getenv(name);
    if (!e || !*e) return -1;
    return (e[0] != '0') ? 1 : 0;
}

inline bool env_master_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_INK_RTX5090");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    return v;
}

inline bool is_sm120_device(int dev) {
    static strata::core::PerDeviceFlag flag;
    return flag.get(dev, [dev] {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        return p.major == 12 && p.minor == 0;
    });
}

inline bool current_device_is_sm120() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return is_sm120_device(dev);
}

inline bool enabled_for_device(int dev) {
    return env_master_on() && is_sm120_device(dev);
}

inline bool enabled() {
    return env_master_on() && current_device_is_sm120();
}

inline bool sub_flag(int tri) {
    const bool on = (tri >= 0) ? (tri != 0) : env_master_on();
    return on && current_device_is_sm120();
}

// Granular sub-switches (all OFF when STRATA_INK_RTX5090 is unset/0; follow master when set unless overridden)
inline bool exact_mode_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_INK_RTX5090_EXACT");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    return v && current_device_is_sm120();
}

inline bool pf_attn_on() {
    if (exact_mode_on()) return false;
    static const int t = env_tri_state("STRATA_INK_SM120_PF_ATTN");
    return sub_flag(t);
}

inline bool moe_opt_on() {
    if (exact_mode_on()) return false;
    static const int t = env_tri_state("STRATA_INK_SM120_MOE_OPT");
    return sub_flag(t);
}

inline bool dense_opt_on() {
    if (exact_mode_on()) return false;
    static const int t = env_tri_state("STRATA_INK_SM120_DENSE_OPT");
    return sub_flag(t);
}

inline bool dspark_opt_on() {
    static const int t = env_tri_state("STRATA_INK_SM120_DSPARK_OPT");
    return sub_flag(t);
}

inline bool c64_spec_on() {
    static const int t = env_tri_state("STRATA_INK_SM120_C64_SPEC");
    return sub_flag(t);
}

inline bool fp8_trunk_on() {
    if (exact_mode_on()) return false;
    static const int t = env_tri_state("STRATA_INK_SM120_FP8_TRUNK");
    return sub_flag(t);
}

}  // namespace strata::ink::rtx5090
