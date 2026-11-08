// metal/src/engine/ds4_vision_metal.cpp - DeepSeek-V4.1's vision tower on Metal: the host side.
//
// Runs src/dsv41/ds4_vision.cu's encode() as dispatches of msl/ds4_vision.metal on the dsv41 engine's one mtl::Context (the
// queue every other dsv41 kernel orders with).  The weights are read in place (the tower's pinned host block is an MTLBuffer of
// the Metal runtime, so the GPU takes its device address): no slot copies.
//
// Credit: the structure of a DeepSeek-V4 vision tower on Metal (which operations are their own kernels, the BF16 rounding
// points) follows antirez/ds4 (MIT), commit 890898cf, metal/deepseek4_vision.metal and ds4_metal.m
// (ds4_gpu_deepseek4_vision_encode).  Copyright (c) 2026 The ds4.c authors, (c) 2023-2026 The ggml authors, (c) 2023 DeepSeek.
// See third_party/antirez-ds4/LICENSE and NOTICE.md.
#include "ds4_vision_metal.h"

#include "mtl_rt.h"

#include <strata/gpu/runtime.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::dsv41::metal_host {
mtl::Context& finish_cx();
}

namespace strata::ds4::vismetal {

// the capability string: compiled in only here, where the Metal tower is built
static const char kMarker[] = "STRATA_DS4_METAL_VISION_V1";
const char* marker() { return kMarker; }

namespace {

struct PatchArgs { uint64_t rgb, out; int32_t width, gw, p; };
struct AddArgs { uint64_t x, y, bias; int32_t ldy, dim; };
struct NormArgs { uint64_t x, w, out; int32_t dim; float eps; };
struct RopeArgs { uint64_t y, bias, inv_freq, qkv; int32_t r0, gw, heads, hd; };
struct SoftmaxArgs { uint64_t s, p; int32_t n; float scale; };
struct SiluArgs { uint64_t y, mid; int32_t inter; };
struct UnfoldArgs { uint64_t xn, un; int32_t gh, gw, m, dim, lw; };
struct BiasActArgs { uint64_t y, bias, out; int32_t n, gelu; };
struct MppArgs { uint64_t X, W, Y; int32_t M, N, K, ldx, ldw, ldy; };
struct GemmArgs { uint64_t X, W, Y; uint32_t rows, in, out, ldx, ldw, ldy; float alpha; uint32_t flags; };

struct Runner {
    mtl::Context& cx;
    int lib = -1;
    std::string err;
    bool ok = true;
    explicit Runner(mtl::Context& c) : cx(c) {}

    mtl::Pipeline pipe(const char* name) {
        try {
            return cx.pipeline(lib, name);
        } catch (...) {
            err = std::string("vision: no pipeline ") + name;
            ok = false;
            return mtl::Pipeline();
        }
    }
    template <class A>
    void go(const mtl::Pipeline& p, const A& args, uint32_t gx, uint32_t gy, uint32_t tg) {
        if (!ok || gx == 0 || gy == 0) return;
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::V(args)};
        d.grid[0] = gx;
        d.grid[1] = gy;
        d.grid[2] = 1;
        d.tg[0] = tg;
        d.tg[1] = d.tg[2] = 1;
        mtl::require_fit(p, tg, p.name.c_str());
        cx.run({d});
    }
};

}  // namespace

namespace {

// Whether the tensor-op GEMM (ds4_vision_mpp.metal) can be trusted on this Mac: Metal 4 on the GPU and the OS, the library compiles, and
// it computes what the plain kernel computes on a small problem with odd sizes and strides.  Decided once; the answer is logged.
// DS4_VISION_NO_MPP=1 asks for the plain kernel; DS4_VISION_FORCE_MPP_FAIL=1 (tests) makes the check fail.
struct MppChoice {
    bool use = false;
    int lib = -1;
};

float bf_to_f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f_to_bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);   // (truncated: the test only needs BF16 values)
}

bool mpp_selftest(mtl::Context& cx, int plain_lib, int mpp_lib, std::string& why) {
    const int M = 70, N = 90, K = 100, LDX = 104, LDW = 104, LDY = 96, NV = 64, LDV = 136, LDO = 72;
    void *dX = nullptr, *dW = nullptr, *dV = nullptr, *dY0 = nullptr, *dY1 = nullptr, *dO0 = nullptr, *dO1 = nullptr;
    const size_t bX = (size_t) M * LDX * 2, bW = (size_t) N * LDW * 2, bV = (size_t) K * LDV * 2, bY = (size_t) M * LDY * 4, bO = (size_t) M * LDO * 2;
    bool ok = strata::gpu::malloc_device(&dX, bX) == strata::gpu::kSuccess && strata::gpu::malloc_device(&dW, bW) == strata::gpu::kSuccess &&
              strata::gpu::malloc_device(&dV, bV) == strata::gpu::kSuccess && strata::gpu::malloc_device(&dY0, bY) == strata::gpu::kSuccess &&
              strata::gpu::malloc_device(&dY1, bY) == strata::gpu::kSuccess && strata::gpu::malloc_device(&dO0, bO) == strata::gpu::kSuccess &&
              strata::gpu::malloc_device(&dO1, bO) == strata::gpu::kSuccess;
    auto release = [&]() {
        for (void* q : {dX, dW, dV, dY0, dY1, dO0, dO1})
            if (q) strata::gpu::free_device(q);
    };
    if (!ok) {
        release();
        why = "no memory for the self-test";
        return false;
    }
    uint32_t seed = 12345u;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float) ((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;   // [-1, 1)
    };
    auto fill = [&](void* d, size_t bytes) {
        uint16_t* h = (uint16_t*) strata::gpu::host_view(d);
        std::memset(h, 0, bytes);
        for (size_t i = 0; i < bytes / 2; ++i) h[i] = f_to_bf(rnd());
        return h != nullptr;
    };
    ok = fill(dX, bX) && fill(dW, bW) && fill(dV, bV);
    std::memset(strata::gpu::host_view(dY0), 0, bY);
    std::memset(strata::gpu::host_view(dY1), 0, bY);
    std::memset(strata::gpu::host_view(dO0), 0, bO);
    std::memset(strata::gpu::host_view(dO1), 0, bO);
    if (!ok) {
        release();
        why = "no host view for the self-test";
        return false;
    }
    auto addr = [](void* d) { return (uint64_t) (uintptr_t) d; };
    try {
        const mtl::Pipeline pPlain = cx.pipeline(plain_lib, "vis_gemm"), pK = cx.pipeline(mpp_lib, "vis_mpp_k_f32_64_64"),
                            pNbf = cx.pipeline(mpp_lib, "vis_mpp_n_bf_64_64");
        Runner R(cx);
        GemmArgs g0{addr(dX), addr(dW), addr(dY0), (uint32_t) M, (uint32_t) K, (uint32_t) N, (uint32_t) LDX, (uint32_t) LDW, (uint32_t) LDY, 1.0f, 0u};
        R.go(pPlain, g0, (uint32_t) ((N + 63) / 64), (uint32_t) ((M + 63) / 64), 128);
        MppArgs m0{addr(dX), addr(dW), addr(dY1), M, N, K, LDX, LDW, LDY};
        R.go(pK, m0, (uint32_t) ((N + 63) / 64), (uint32_t) ((M + 63) / 64), 128);
        GemmArgs g1{addr(dX), addr(dV), addr(dO0), (uint32_t) M, (uint32_t) K, (uint32_t) NV, (uint32_t) LDX, (uint32_t) LDV, (uint32_t) LDO, 1.0f, 3u};
        R.go(pPlain, g1, (uint32_t) ((NV + 63) / 64), (uint32_t) ((M + 63) / 64), 128);
        MppArgs m1{addr(dX), addr(dV), addr(dO1), M, NV, K, LDX, LDV, LDO};
        R.go(pNbf, m1, (uint32_t) ((NV + 63) / 64), (uint32_t) ((M + 63) / 64), 128);
        cx.flush_runs();
    } catch (...) {
        release();
        why = "the self-test could not be dispatched";
        return false;
    }
    const float* y0 = (const float*) strata::gpu::host_view(dY0);
    const float* y1 = (const float*) strata::gpu::host_view(dY1);
    const uint16_t* o0 = (const uint16_t*) strata::gpu::host_view(dO0);
    const uint16_t* o1 = (const uint16_t*) strata::gpu::host_view(dO1);
    double worst_f = 0.0, worst_b = 0.0;
    bool good = true;
    for (int r = 0; r < M && good; ++r) {
        for (int c = 0; c < N; ++c) {
            const float a = y0[(size_t) r * LDY + c], b = y1[(size_t) r * LDY + c];
            const double d = std::fabs((double) a - (double) b);
            worst_f = std::max(worst_f, d);
            if (!(d <= 1e-3 + 1e-4 * std::fabs((double) a))) good = false;   // (also catches NaN)
        }
        for (int c = 0; c < NV; ++c) {
            const float a = bf_to_f(o0[(size_t) r * LDO + c]), b = bf_to_f(o1[(size_t) r * LDO + c]);
            const double d = std::fabs((double) a - (double) b);
            worst_b = std::max(worst_b, d);
            if (!(d <= 1e-3 + 0.02 * std::fabs((double) a))) good = false;
        }
    }
    release();
    if (!good) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "the self-test differs from the plain kernel (max abs %.3g FP32, %.3g BF16)", worst_f, worst_b);
        why = buf;
        return false;
    }
    return true;
}

MppChoice choose_mpp(mtl::Context& cx, int plain_lib) {
    MppChoice c;
    std::string why;
    if (std::getenv("DS4_VISION_NO_MPP") != nullptr || std::getenv("DS4_VISION_REF_GEMM") != nullptr) {
        why = "asked for by DS4_VISION_NO_MPP / DS4_VISION_REF_GEMM";
    } else if (!cx.supports_metal4()) {
        why = "this GPU or macOS has no Metal 4";
    } else if (std::getenv("DS4_VISION_FORCE_MPP_FAIL") != nullptr) {
        why = "DS4_VISION_FORCE_MPP_FAIL (test)";
    } else {
        std::string msg;
        const int lib = cx.compile_try({"../src/engine/msl/ds4_vision_mpp.metal"}, "", false, &msg);
        if (lib < 0) {
            why = "the tensor-op kernels do not compile here (" + msg.substr(0, 120) + ")";
        } else if (mpp_selftest(cx, plain_lib, lib, why)) {
            c.use = true;
            c.lib = lib;
        }
    }
    if (c.use)
        std::fprintf(stderr, "ds4: vision tower GEMM: Metal 4 tensor ops (self-test against the plain kernel ok)\n");
    else
        std::fprintf(stderr, "ds4: vision tower GEMM: plain kernel (%s)\n", why.c_str());
    return c;
}

}  // namespace

bool encode(const Args& a, std::string& err) {
    mtl::Context& cx = strata::dsv41::metal_host::finish_cx();
    static int s_lib = -2;
    if (s_lib == -2) s_lib = cx.compile({"../src/engine/msl/ds4_vision.metal"}, "", false);
    if (s_lib < 0) {
        err = "vision: compiling ds4_vision.metal failed";
        return false;
    }
    // the tensor-op GEMM (ds4_vision_mpp.metal) when this Mac can run it and it checks out; else the plain kernel (logged once)
    static const MppChoice mc = choose_mpp(cx, s_lib);
    const bool use_mpp = mc.use;
    const int s_lib_mpp = mc.lib;
    Runner R(cx);
    R.lib = s_lib;
    const mtl::Pipeline pPatch = R.pipe("vis_patchify"), pAdd = R.pipe("vis_add_rows"), pNorm = R.pipe("vis_rmsnorm"),
                        pRope = R.pipe("vis_qkv_rope"), pSm = R.pipe("vis_softmax"), pSilu = R.pipe("vis_silu_mul"),
                        pUnf = R.pipe("vis_unfold"), pBias = R.pipe("vis_bias_act"), pGemm0 = R.pipe("vis_gemm"),
                        pGemmRef = R.pipe("vis_gemm_ref");
    static const char* const ref_env = std::getenv("DS4_VISION_REF_GEMM");   // (tests) 1: the sequential-FMA GEMM, 2: summed in chunks of 64
    static const bool ref_gemm = ref_env != nullptr;
    static const uint32_t ref_flags = ref_env != nullptr && ref_env[0] == '2' ? 4u : 0u;
    const mtl::Pipeline& pGemm = ref_gemm ? pGemmRef : pGemm0;
    mtl::Pipeline pMppK128, pMppK64, pMppNbf;
    if (use_mpp) {
        try {
            pMppK128 = cx.pipeline(s_lib_mpp, "vis_mpp_k_f32_128_64");
            pMppK64 = cx.pipeline(s_lib_mpp, "vis_mpp_k_f32_64_64");
            pMppNbf = cx.pipeline(s_lib_mpp, "vis_mpp_n_bf_64_64");
        } catch (...) {
            err = "vision: no tensor-op pipeline";
            return false;
        }
    }
    if (!R.ok) {
        err = R.err;
        return false;
    }
    const int D = a.dim, I = a.inter, P = a.patch, M = a.merge, H3 = 3 * D, hd = a.hd, N = a.n, L = a.l;
    const uint32_t smT = (uint32_t) std::min(512, (int) pSm.max_threads / 32 * 32);
    if ((int) smT * 20 < N) {
        err = "vision: this GPU's softmax group (" + std::to_string(smT) + " threads) is too small for " + std::to_string(N) + " patches";
        return false;
    }
    auto at = [](uint64_t base, size_t elems, size_t esz) { return base + (uint64_t) (elems * esz); };
    auto gemm = [&](uint64_t X, uint32_t ldx, uint64_t W, uint32_t ldw, bool wn, uint64_t Y, uint32_t ldy, bool ybf, int rows,
                    int in, int out, float alpha) {
        if (use_mpp && alpha == 1.0f && (wn ? ybf : !ybf)) {   // k-major W to FP32, or n-major W (V) to BF16
            MppArgs m{X, W, Y, rows, out, in, (int32_t) ldx, (int32_t) ldw, (int32_t) ldy};
            if (wn) {
                R.go(pMppNbf, m, (uint32_t) ((out + 63) / 64), (uint32_t) ((rows + 63) / 64), 128);
            } else if (rows >= 128) {
                R.go(pMppK128, m, (uint32_t) ((out + 63) / 64), (uint32_t) ((rows + 127) / 128), 128);
            } else {
                R.go(pMppK64, m, (uint32_t) ((out + 63) / 64), (uint32_t) ((rows + 63) / 64), 128);
            }
            return;
        }
        GemmArgs g{X, W, Y, (uint32_t) rows, (uint32_t) in, (uint32_t) out, ldx, ldw, ldy, alpha, (wn ? 1u : 0u) | (ybf ? 2u : 0u) | (ref_gemm ? ref_flags : 0u)};
        if (ref_gemm) R.go(pGemm, g, (uint32_t) ((out + 31) / 32), (uint32_t) rows, 32);
        else R.go(pGemm, g, (uint32_t) ((out + 63) / 64), (uint32_t) ((rows + 63) / 64), 128);
    };
    auto add_rows = [&](uint64_t x, uint64_t y, int ldy, uint64_t bias, int rows) {
        AddArgs g{x, y, bias, ldy, D};
        R.go(pAdd, g, (uint32_t) rows, 1, 256);
    };
    auto rmsnorm = [&](uint64_t x, uint64_t w, uint64_t out) {
        NormArgs g{x, w, out, D, a.eps};
        R.go(pNorm, g, (uint32_t) N, 1, 256);
    };

    // the patches
    {
        PatchArgs g{a.rgb, a.patches, a.width, a.gw, P};
        R.go(pPatch, g, (uint32_t) N, 1, 196);
    }
    gemm(a.patches, 3 * P * P, a.pe_w, 3 * P * P, false, a.x, D, false, N, 3 * P * P, D, 1.0f);
    add_rows(a.x, 0, 0, a.pe_b, N);

    for (int l = 0; l < a.layers && R.ok; ++l) {
        const uint64_t wqkv = at(a.weights, (size_t) l * a.block_n, 2), wo = at(wqkv, (size_t) H3 * D, 2),
                       w1 = at(wo, (size_t) D * D, 2), w2 = at(w1, (size_t) 2 * I * D, 2);
        rmsnorm(a.x, a.n1[l], a.xn);
        for (int r0 = 0; r0 < N; r0 += a.rb) {
            const int nr = std::min(a.rb, N - r0);
            gemm(at(a.xn, (size_t) r0 * D, 2), D, wqkv, D, false, a.y, H3, false, nr, D, H3, 1.0f);
            RopeArgs g{a.y, a.bqkv[l], a.inv_freq, a.qkv, r0, a.gw, a.heads, hd};
            R.go(pRope, g, (uint32_t) nr, 1, 256);
        }
        for (int q0 = 0; q0 < N; q0 += a.qb) {
            const int nq = std::min(a.qb, N - q0);
            for (int hh = 0; hh < a.heads; ++hh) {
                // S [query][key] = scale * Q K^T
                gemm(at(a.qkv, (size_t) q0 * H3 + (size_t) hh * hd, 2), H3, at(a.qkv, (size_t) D + (size_t) hh * hd, 2), H3, false,
                     a.s, N, false, nq, hd, N, use_mpp ? 1.0f : a.scale);
                SoftmaxArgs sg{a.s, a.p, N, use_mpp ? a.scale : 1.0f};
                R.go(pSm, sg, (uint32_t) nq, 1, smT);
                // O[:, head] = P V
                gemm(a.p, N, at(a.qkv, (size_t) 2 * D + (size_t) hh * hd, 2), H3, true, at(a.o, (size_t) hh * hd, 2), D, true, nq, N, hd,
                     1.0f);
            }
            gemm(a.o, D, wo, D, false, a.y, D, false, nq, D, D, 1.0f);
            add_rows(at(a.x, (size_t) q0 * D, 4), a.y, D, a.bo[l], nq);
        }
        rmsnorm(a.x, a.n2[l], a.xn);
        for (int r0 = 0; r0 < N; r0 += a.rb) {
            const int nr = std::min(a.rb, N - r0);
            gemm(at(a.xn, (size_t) r0 * D, 2), D, w1, D, false, a.y, 2 * I, false, nr, D, 2 * I, 1.0f);
            SiluArgs sg{a.y, a.mid, I};
            R.go(pSilu, sg, (uint32_t) ((I + 255) / 256), (uint32_t) nr, 256);
            gemm(a.mid, I, w2, I, false, a.y, D, false, nr, I, D, 1.0f);
            add_rows(at(a.x, (size_t) r0 * D, 4), a.y, D, 0, nr);
        }
        cx.split_runs();   // each block is its own command buffer: no long-running one for the system to cut short
    }
    rmsnorm(a.x, a.norm, a.xn);

    // the aligner
    const int in1 = D * M * M;
    {
        UnfoldArgs g{a.xn, a.un, a.gh, a.gw, M, D, a.nw};
        R.go(pUnf, g, (uint32_t) L, 1, 256);
    }
    gemm(a.un, in1, at(a.weights, a.al1_off, 2), in1, false, a.y, a.out, false, L, in1, a.out, 1.0f);
    {
        BiasActArgs g{a.y, a.al1_b, a.a16, a.out, 1};
        R.go(pBias, g, (uint32_t) ((a.out + 255) / 256), (uint32_t) L, 256);
    }
    gemm(a.a16, a.out, at(a.weights, a.al2_off, 2), a.out, false, a.y, a.out, false, L, a.out, a.out, 1.0f);
    {
        BiasActArgs g{a.y, a.al2_b, a.outb, a.out, 0};
        R.go(pBias, g, (uint32_t) ((a.out + 255) / 256), (uint32_t) L, 256);
    }
    cx.flush_runs();
    if (!R.ok) {
        err = R.err;
        return false;
    }
    return true;
}

}  // namespace strata::ds4::vismetal
