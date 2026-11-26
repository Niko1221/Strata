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


// ---- the format of the block weights (attention q/o, feed-forward gate/up/down); the rest (fc, K/V and convolution projections, selector, norms, codebooks) stays bf16 / f32 ----
// kinds: 0 bf16 (reference), 1 Q8 row-scaled, 2 4-bit groups of 32, 3 SF4 (E2M1 + UE4M3 per 16 + fp32 per tensor).
struct Df2Variant {
    int attn = 1, ffn = 1;
};

inline Df2Variant df2_variant(bool small_vram) {
    Df2Variant v;
    v.attn = v.ffn = small_vram ? 2 : 1;   // what shipped first: 4-bit groups on cards below 24 GB, Q8 above
    if (const char* e = std::getenv("STRATA_INK_DFLASH2_W4")) v.attn = v.ffn = (std::string(e) == "1") ? 2 : 1;
    if (const char* e = std::getenv("STRATA_INK_DFLASH2_WQ")) {
        const std::string s(e);
        if (s == "bf16") v.attn = v.ffn = 0;
        else if (s == "q8") v.attn = v.ffn = 1;
        else if (s == "q4") v.attn = v.ffn = 2;
        else if (s == "sf4") v.attn = v.ffn = 3;
        else if (s == "mix") { v.attn = 1; v.ffn = 3; }
        else if (s == "mixq4") { v.attn = 1; v.ffn = 2; }
    }
    return v;
}

inline double df2_bytes_per_param(int kind) {
    return kind == 0 ? 2.0 : (kind == 1 ? 1.0 : 0.5625);   // 4-bit kinds: 0.5 + one 8/16-bit scale per 16 or 32
}

// MiB of the block weights of the 5 layers (attention 18.9 M parameters, feed-forward 47.8 M per layer)
inline double df2_big_mib(Df2Variant v) {
    return (5.0 * 18.874368e6 * df2_bytes_per_param(v.attn) + 5.0 * 47.775744e6 * df2_bytes_per_param(v.ffn)) / 1048576.0;
}

// what the drafter holds on a small card: the bf16 rest (fc, K/V, convolutions, selector, norms: 112 MiB), the block weights, rings (42 MiB a slot, up to 8), scratch and a margin
inline size_t df2_total_mib(Df2Variant v, int slots) {
    return (size_t)(112.0 + df2_big_mib(v) + 42.0 * (double)(slots < 8 ? slots : 8) + 120.0);
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
