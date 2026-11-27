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

// The drafter when STRATA_INK_DRAFTER is not set: DFlash 2, the only drafter of Inkling 2 Mini (owner decision 2026-11-26), for every slot count. Without its sidecar file
// (dflash2.safetensors beside the model, or STRATA_INK_DFLASH2) the run has no drafter, and the engine says so; it does not fall back to DSpark. The DSpark code is
// dormant: only STRATA_INK_DRAFTER=dspark selects it (measurements, and a model file that carries it).
inline constexpr bool kDflash2Default = true;

// The number of parallel slots the engine was started with (set by the backend before anything loads).
inline int& df2_parallel() {
    static int n = 1;
    return n;
}

inline std::string df2_drafter_env() {
    const char* e = std::getenv("STRATA_INK_DRAFTER");
    std::string s = e ? e : "";
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

// NV-OLDER: set by the backend when the primary card is below sm_80 (V100, RTX 20, GTX 10, P100). DFlash 2 then runs its FP32 form: the BF16 matrices are held as FP32
// (the checkpoint's values, exactly) and multiplied with cuBLAS FP32, the block weights are never BF16, the head screen on sm_60 is a dp4a kernel instead of the cuBLAS INT8 GEMM,
// and the skinny GEMVs use 4 activation rows per block where 8 would not fit 48 KB of shared memory. The drafter still only proposes: the target verifies.
inline bool& df2_f32_path() {
    static bool b = false;
    return b;
}

// STRATA_INK_DRAFTER=dspark: the dormant DSpark drafter is asked for by name
inline bool df2_dspark_selected() { return df2_drafter_env() == "dspark"; }

// The residency-first rule (strata/core/ink_draft_rule.hpp): set once by the backend before anything loads when DFlash 2 as the DEFAULT drafter would move the model from full residency to a
// hybrid load (or cut the expert tier noticeably). It never applies when the drafter was asked for by name (STRATA_INK_DRAFTER=dflash2, `--draft on`, STRATA_INK_DFLASH2_FORCE=1).
inline bool& df2_residency_veto() {
    static bool v = false;
    return v;
}

inline bool df2_forced_by_env() {
    const char* e = std::getenv("STRATA_INK_DFLASH2_FORCE");
    return e && e[0] == '1';
}

// Set by the backend before anything loads: the card has less than 24 GiB.
inline bool& df2_small_card() {
    static bool b = false;
    return b;
}

inline bool df2_requested() {
    const std::string s = df2_drafter_env();
    if (s == "dflash2") return true;
    // 2..15 slots on a card below 24 GiB: no drafter (no multi-user chain there, and the drafter's memory only costs experts: RTX 5060 Ti -14 / -16 percent at 4 / 8 slots, fast)
    if (s.empty()) return kDflash2Default && !df2_residency_veto() && !(df2_small_card() && df2_parallel() >= 2 && df2_parallel() < 16);
    return false;
}


// ---- the format of the block weights (attention q/o, feed-forward gate/up/down); the rest (fc, K/V and convolution projections, selector, norms, codebooks) stays bf16 / f32 ----
// kinds: 0 bf16 (reference), 1 Q8 row-scaled, 2 4-bit groups of 32, 3 SF4 (E2M1 + UE4M3 per 16 + fp32 per tensor).
struct Df2Variant {
    int attn = 1, ffn = 1;
};

inline std::string df2_resolve_path(const std::string& model_path);
inline bool df2_parse_variant(const std::string& s, Df2Variant& v) {
    if (s == "bf16") v.attn = v.ffn = 0;
    else if (s == "q8") v.attn = v.ffn = 1;
    else if (s == "q4") v.attn = v.ffn = 2;
    else if (s == "sf4") v.attn = v.ffn = 3;
    else if (s == "mix") { v.attn = 1; v.ffn = 3; }
    else if (s == "mixq4") { v.attn = 1; v.ffn = 2; }
    else return false;
    return true;
}

// the "variant" of a pre-quantized sidecar (strata-dflash2-q1): found in the safetensors header's __metadata__; "" for the original bf16 checkpoint
inline std::string df2_file_variant(const std::string& file) {
    if (file.empty()) return "";
    FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) return "";
    unsigned char len8[8];
    std::string out;
    if (std::fread(len8, 1, 8, f) == 8) {
        uint64_t n = 0;
        for (int k = 7; k >= 0; --k) n = (n << 8) | len8[k];
        if (n > 0 && n < (64u << 20)) {
            std::string text(n, '\0');
            if (std::fread(&text[0], 1, n, f) == n) {
                const size_t fm = text.find("\"format\":\"strata-dflash2-q1\"");
                const size_t vp = text.find("\"variant\":\"");
                if (fm != std::string::npos && vp != std::string::npos) {
                    const size_t b = vp + 11, e = text.find('"', b);
                    if (e != std::string::npos) out = text.substr(b, e - b);
                }
            }
        }
    }
    std::fclose(f);
    return out;
}

inline Df2Variant df2_variant(bool small_vram) {
    Df2Variant v;
    // The format of the block weights when the file is the original BF16 checkpoint: BF16 on cards of 24 GB and more (the fastest at 4 and 8 slots in fast mode, measured), the mix
    // (Q8 attention, SF4 feed-forward: the same bits as the pre-quantized sidecar) below 24 GB, where it is what fits and keeps acceptance.
    if (small_vram) { v.attn = 1; v.ffn = 3; } else { v.attn = v.ffn = 0; }
    if (const char* e = std::getenv("STRATA_INK_DFLASH2_W4")) v.attn = v.ffn = (std::string(e) == "1") ? 2 : 1;
    if (const char* e = std::getenv("STRATA_INK_DFLASH2_WQ")) df2_parse_variant(std::string(e), v);
    // a pre-quantized sidecar decides its own format
    // (on a card of 24 GiB and more whose row is bf16 the file's format is not taken: the loader expands the sidecar to bf16, which is faster at 16 and more slots)
    const std::string fv = df2_file_variant(df2_resolve_path(df2_model_path()));
    // (a card below sm_80 takes the file's format as before: there is no BF16 form there to expand to)
    if (!fv.empty() && (small_vram || df2_f32_path() || std::getenv("STRATA_INK_DFLASH2_WQ"))) df2_parse_variant(fv, v);
    // NV-OLDER: no BF16 block weights on a card below sm_80 (the BF16 GEMM of kind 0 is not there): Q8 instead; the other kinds are dp4a GEMVs and run everywhere
    if (df2_f32_path() && v.attn == 0) v.attn = 1;
    if (df2_f32_path() && v.ffn == 0) v.ffn = 1;
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
    return (size_t)((df2_f32_path() ? 224.0 : 112.0) + df2_big_mib(v) + 42.0 * (double)(slots < 8 ? slots : 8) + 120.0);
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
