// metal/include/metal_compat/ds4_vision_metal.h - the Metal side of src/dsv41/ds4_vision.cu's encode(): one call that runs the
// whole vision tower on the Metal runtime (metal/src/engine/ds4_vision_metal.cpp, kernels in msl/ds4_vision.metal).
//
// The layout of a DeepSeek-V4 vision tower on Metal follows antirez/ds4 (MIT, commit 890898cf); see third_party/antirez-ds4.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ds4::vismetal {

// every pointer is a device address (what strata::gpu's cudaMalloc / cudaHostGetDevicePointer hand out)
struct Args {
    int width = 0, height = 0, gh = 0, gw = 0, n = 0, l = 0, nw = 0;   // image, patch grid, patches, aligner cells, aligner width
    int dim = 0, inter = 0, patch = 0, merge = 0, heads = 0, hd = 0, layers = 0, out = 0, rb = 0, qb = 0;
    float eps = 0.0f, scale = 0.0f;
    uint64_t rgb = 0, patches = 0, x = 0, xn = 0, qkv = 0, y = 0, o = 0, mid = 0, s = 0, p = 0, un = 0, a16 = 0, outb = 0, inv_freq = 0;
    uint64_t weights = 0;                    // the streamed weights (BF16): every block's [wqkv, wo, w1, w2], then aligner w1, w2
    size_t block_n = 0, al1_off = 0, al2_off = 0;
    uint64_t pe_w = 0, pe_b = 0, norm = 0, al1_b = 0, al2_b = 0;
    std::vector<uint64_t> n1, n2, bqkv, bo;
};

/// Runs patchify .. aligner into a.outb ([a.l][a.out] BF16) and returns once the GPU is done.
bool encode(const Args& a, std::string& err);

/// The capability string the picture tower's startup line carries (the Control looks for it in the binary).
const char* marker();

}  // namespace strata::ds4::vismetal
