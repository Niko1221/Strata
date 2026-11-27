// engine/src/inkling2/ink_vision_policy.h - which picture reader Inkling-2-Mini uses (GPU encoder or the CPU reference), one rule for every backend
// MIT License - Copyright (c) 2026 Strata contributors
//
// The rule is the one Qwen3.8-Flash-Next uses (docs/DETAILS.md "Images (vision)", terminal/strata_control/vision_reader.py default_mode):
//   * NVIDIA and AMD with under 24 GB of memory: the CPU reader, so the experts keep all of the card.
//   * 24 GB or more (23.5 GiB reported: a "24 GB" card reports 23.5 to 24 GiB): the card's reader.  On NVIDIA with one GPU the engine puts the reader on
//     the card with the first picture and takes it off again after vision_idle_s seconds (default 120) without one (Qwen's elastic reader).
//   * A Mac: the GPU (Metal), whatever its memory.
//   * Intel Arc: the CPU reader unless asked (Qwen: INTEL_GPU_READER_BY_DEFAULT = False), at any memory size.
//   * Unified-memory machines (Strix Halo, DGX Spark): the same number rule on the memory the card reports, which there is the shared pool, so they read on the GPU.
//   * Asked for: `--vision-reader gpu|cpu` (and STRATA_INK_VISION_READER) wins, with a one-line note on a small card.  The product's `--vision gpu|cpu|no`
//     (Terminal, `strata launch`, the app) becomes this flag, or leaves the Vision file out for `no`.
// The Control side (strata_control/vision_reader.py) holds the same constants; terminal/tests keep the two the same.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ink2 {

// Same number as strata_control/vision_reader.py GPU_MIN_VRAM_GB.
constexpr double kVisionGpuMinGiB = 23.5;
// Qwen's idle time before the elastic reader gives its VRAM back.
constexpr int kVisionIdleSecondsDefault = 120;
// Same as strata_control/vision_reader.py INTEL_GPU_READER_BY_DEFAULT.
constexpr bool kVisionIntelGpuByDefault = false;

enum class VisionAsk { Auto, Gpu, Cpu };

inline VisionAsk vision_ask_from(const std::string& s) {
    if (s == "gpu") return VisionAsk::Gpu;
    if (s == "cpu") return VisionAsk::Cpu;
    return VisionAsk::Auto;
}

struct VisionPlan {
    bool gpu = false;        // read pictures on the card (when the backend has an encoder; otherwise the CPU reader runs anyway)
    bool forced = false;     // the person asked (--vision-reader gpu|cpu)
    std::string reason;      // one plain sentence for the log
};

// backend: backend_name() ("cuda", "cuda-ls", "cuda-ep2", "hip", "hip-ep2", "metal", "sycl", "cpu" ...).  mem_gib: what the backend reports for its main card
// (0 = unknown).  The environment variable only replaces an "auto" ask.
inline VisionPlan plan_vision_reader(VisionAsk ask, const std::string& backend, double mem_gib) {
    VisionPlan p;
    const auto starts = [&](const char* k) { return backend.rfind(k, 0) == 0; };
    const bool is_cuda = starts("cuda"), is_hip = starts("hip"), is_metal = starts("metal"), is_sycl = starts("sycl");
    char mem[64];
    std::snprintf(mem, sizeof(mem), "%.1f GiB", mem_gib);
    if (!(is_cuda || is_hip || is_metal || is_sycl)) {
        p.gpu = false;
        p.reason = "this engine runs on the CPU, so pictures are read on the CPU";
        return p;
    }
    if (ask == VisionAsk::Cpu) {
        p.gpu = false;
        p.forced = true;
        p.reason = "pictures are read on the CPU, as you asked (--vision cpu); the card keeps all its memory for the model";
        return p;
    }
    if (ask == VisionAsk::Gpu) {
        p.gpu = true;
        p.forced = true;
        if (mem_gib > 0.0 && mem_gib < kVisionGpuMinGiB && !is_metal) {
            p.reason = std::string("pictures are read on the card, as you asked (--vision gpu); it keeps about 1.3 GB of the card's ") + mem +
                       " that the experts would use (--vision cpu reads them on the CPU instead): kept as you chose";
        } else {
            p.reason = "pictures are read on the card, as you asked (--vision gpu)";
        }
        return p;
    }
    if (is_metal) {
        p.gpu = true;
        p.reason = "pictures are read on the Mac's GPU (its memory is the system's)";
    } else if (is_sycl && !kVisionIntelGpuByDefault) {
        p.gpu = false;
        p.reason = "pictures are read on the CPU on an Intel Arc card (--vision gpu uses the card)";
    } else if (mem_gib > 0.0 && mem_gib < kVisionGpuMinGiB) {
        p.gpu = false;
        p.reason = std::string("pictures are read on the CPU: this card has ") + mem + " (under 24 GB), so the experts keep all of it (--vision gpu uses the card anyway)";
    } else {
        p.gpu = true;
        p.reason = mem_gib > 0.0 ? std::string("pictures are read on the card (") + mem + ", 24 GB or more)" : std::string("pictures are read on the card");
    }
    return p;
}

// The part of the room a GPU reader that is not elastic or elastic keeps in the plan.  The CUDA backend adds it to the memory it holds back from the experts.
inline double vision_idle_seconds(int opt) { return opt >= 0 ? (double)opt : (double)kVisionIdleSecondsDefault; }

}  // namespace ink2
