// engine/src/inkling2/cuda/ink_dflash2_path.cuh - where the opt-in DFlash 2 drafter's weights are (STRATA_INK_DRAFTER=dflash2); no CUDA here
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::ink {

inline std::string& df2_model_path() {
    static std::string p;
    return p;
}

// The default drafter when STRATA_INK_DRAFTER is not set: DFlash 2 (owner decision 2026-11-26, after the measurements in phaseA-tests/rel0223/dflash2/RESULTS.md), only up to 16
// parallel slots and only when its sidecar file is found (else DSpark, with one line saying so). STRATA_INK_DRAFTER=dspark always selects DSpark, =dflash2 always asks for DFlash 2.
inline constexpr bool kDflash2Default = true;

// The number of parallel slots the engine was started with (set by the backend before anything loads): the default drafter is DFlash 2 only up to 16 of them (its
// per-slot context rings do not scale to 64 slots; DSpark's multi-user path does).
inline int& df2_parallel() {
    static int n = 1;
    return n;
}

inline bool df2_requested() {
    const char* e = std::getenv("STRATA_INK_DRAFTER");
    if (!e || !*e) return kDflash2Default && df2_parallel() <= 16;
    std::string s(e);
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s == "dflash2";
}

// STRATA_INK_DFLASH2 (a file, or a folder holding model.safetensors / dflash2.safetensors), else a sidecar in the model's folder. "" = none found.
inline std::string df2_resolve_path(const std::string& model_path) {
    auto exists = [](const std::string& p) { FILE* f = std::fopen(p.c_str(), "rb"); if (f) { std::fclose(f); return true; } return false; };
    if (const char* e = std::getenv("STRATA_INK_DFLASH2")) {
        std::string s(e);
        if (exists(s)) return s;
        if (exists(s + "/model.safetensors")) return s + "/model.safetensors";
        if (exists(s + "/dflash2.safetensors")) return s + "/dflash2.safetensors";
        return "";
    }
    std::string dir = model_path;
    const size_t sl = dir.find_last_of("/\\");
    dir = (sl == std::string::npos) ? std::string(".") : dir.substr(0, sl);
    const char* names[] = {"/dflash2.safetensors", "/Inkling-2-Mini-DFlash2.safetensors", "/Inkling-2-Mini-DFlash2/model.safetensors",
                           "/Inkling-2-Mini-DFlash2/dflash2.safetensors"};
    for (const char* n : names)
        if (exists(dir + n)) return dir + n;
    return "";
}

// The DSpark weights are not loaded into the card when DFlash 2 will be the drafter and its file is there (saves ~330 MiB on the 16 GB card).
inline bool df2_replaces_dspark(const std::string& model_path) { return df2_requested() && !df2_resolve_path(model_path).empty(); }

}  // namespace strata::ink
