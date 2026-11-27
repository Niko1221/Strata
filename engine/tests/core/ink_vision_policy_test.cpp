// engine/tests/core/ink_vision_policy_test.cpp - the picture-reader rule of Inkling-2-Mini (ink_vision_policy.h) on every backend and card size
// MIT License - Copyright (c) 2026 Strata contributors
#include "../../src/inkling2/ink_vision_policy.h"

#include <cstdio>
#include <string>

using namespace ink2;

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_fail;
    }
}

int main() {
    // NVIDIA and AMD: CPU reader under 23.5 GiB, the card's from there; asked-for wins
    for (const char* be : {"cuda", "cuda-ep2", "cuda-ls", "hip", "hip-ep2"}) {
        for (double gib : {7.9, 11.6, 15.7, 19.8, 23.4}) {
            const VisionPlan p = plan_vision_reader(VisionAsk::Auto, be, gib);
            check(!p.gpu && !p.forced, std::string(be) + " " + std::to_string(gib) + " GiB: CPU reader by default");
        }
        for (double gib : {23.5, 23.7, 24.0, 31.4, 47.5, 95.0}) {
            const VisionPlan p = plan_vision_reader(VisionAsk::Auto, be, gib);
            check(p.gpu && !p.forced, std::string(be) + " " + std::to_string(gib) + " GiB: card's reader by default");
        }
        check(plan_vision_reader(VisionAsk::Auto, be, 0.0).gpu, std::string(be) + ": unknown memory takes the card's reader");
        VisionPlan g = plan_vision_reader(VisionAsk::Gpu, be, 11.6);
        check(g.gpu && g.forced && g.reason.find("kept as you chose") != std::string::npos, std::string(be) + ": --vision gpu on 12 GB is kept with a note");
        g = plan_vision_reader(VisionAsk::Gpu, be, 32.0);
        check(g.gpu && g.forced && g.reason.find("kept as you chose") == std::string::npos, std::string(be) + ": --vision gpu on 32 GB needs no note");
        const VisionPlan c = plan_vision_reader(VisionAsk::Cpu, be, 32.0);
        check(!c.gpu && c.forced, std::string(be) + ": --vision cpu on 32 GB is kept");
    }
    // Mac: always the GPU, whatever the memory; cpu asked is kept
    for (double gib : {8.0, 16.0, 24.0, 128.0}) {
        check(plan_vision_reader(VisionAsk::Auto, "metal", gib).gpu, "metal " + std::to_string(gib) + " GiB: GPU");
    }
    check(!plan_vision_reader(VisionAsk::Cpu, "metal", 16.0).gpu, "metal: --vision cpu kept");
    {
        const VisionPlan p = plan_vision_reader(VisionAsk::Gpu, "metal", 8.0);
        check(p.gpu && p.reason.find("kept as you chose") == std::string::npos, "metal: gpu asked on 8 GB gives no small-card note");
    }
    // Intel Arc: CPU by default at any size (Qwen's INTEL_GPU_READER_BY_DEFAULT = False), gpu when asked
    for (double gib : {8.0, 16.0, 32.0}) {
        check(!plan_vision_reader(VisionAsk::Auto, "sycl", gib).gpu, "sycl " + std::to_string(gib) + " GiB: CPU by default");
        check(plan_vision_reader(VisionAsk::Gpu, "sycl", gib).gpu, "sycl " + std::to_string(gib) + " GiB: --vision gpu kept");
    }
    // the CPU backend and unknown names read on the CPU, even when gpu is asked
    check(!plan_vision_reader(VisionAsk::Auto, "cpu", 64.0).gpu, "cpu backend: CPU");
    check(!plan_vision_reader(VisionAsk::Gpu, "cpu", 64.0).gpu, "cpu backend: gpu asked cannot be met");
    // the numbers Control holds (terminal/strata_control/vision_reader.py): the parity test in terminal/ reads them from this header
    check(kVisionGpuMinGiB == 23.5, "kVisionGpuMinGiB");
    check(kVisionIdleSecondsDefault == 120, "kVisionIdleSecondsDefault");
    check(!kVisionIntelGpuByDefault, "kVisionIntelGpuByDefault");
    check(vision_idle_seconds(-1) == 120.0 && vision_idle_seconds(0) == 0.0 && vision_idle_seconds(30) == 30.0, "vision_idle_seconds");
    check(vision_ask_from("gpu") == VisionAsk::Gpu && vision_ask_from("cpu") == VisionAsk::Cpu && vision_ask_from("auto") == VisionAsk::Auto &&
              vision_ask_from("") == VisionAsk::Auto,
          "vision_ask_from");
    if (g_fail == 0) std::printf("ink_vision_policy_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
