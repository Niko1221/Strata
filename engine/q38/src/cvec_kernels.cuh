#pragma once
// Control vector on the residual stream (Strata control-vector feature ported
// to the dense Q38 runner). Off by default; when a vector is loaded, each
// steered layer applies, after its FFN write:
//   project: h <- h - s_l (h . v_l) v_l,  v_l = d_l/s_l, s_l = |d_l|
//   add:     h <- h + d_l
// Only tensors whose width equals the model hidden size are accepted.
#include "strata/qwen38/admission.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen38 {
namespace cvec_detail {
__global__ void cvec_apply_kernel(float* __restrict__ x, int T, int hidden,
                                  const float* __restrict__ dir, const float* __restrict__ scale,
                                  int layer, int mode, int on) {
    if (!on) return;
    const int tok = blockIdx.x;
    if (tok >= T) return;
    float* row = x + static_cast<size_t>(tok) * hidden;
    const float* d = dir + static_cast<size_t>(layer) * hidden;
    const float s = scale[layer];
    if (s == 0.f) return;
    __shared__ float dot_part[8], dot_all, ss;
    const int i = threadIdx.x;
    float acc = 0.f;
    for (int k = i; k < hidden; k += 256) acc = fmaf(row[k], d[k], acc);
    for (int stride = 16; stride; stride >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, stride);
    if ((i & 31) == 0) dot_part[i >> 5] = acc;
    __syncthreads();
    if (i == 0) {
        float v = 0.f;
        for (int w = 0; w < 8; ++w) v += dot_part[w];
        dot_all = v;
    }
    __syncthreads();
    if (mode == 0) {
        // project: d_l holds the unit vector v_l, scale holds |d_l|.
        const float a = s * dot_all;
        for (int k = i; k < hidden; k += 256) row[k] = fmaf(-a, d[k], row[k]);
    } else {
        for (int k = i; k < hidden; k += 256) row[k] += d[k];
    }
}
} // namespace cvec_detail

struct CvecRuntime {
    float* d_dir = nullptr;    // [layers * hidden]
    float* d_scale = nullptr;  // [layers]
    int* d_on = nullptr;
    std::vector<float> h_dir, h_scale;
    int layers = 0, first = 1, last = -1, mode = 0; // 0 project, 1 add
    bool loaded = false;
    ~CvecRuntime() { if (d_dir) cudaFree(d_dir); if (d_scale) cudaFree(d_scale); if (d_on) cudaFree(d_on); }
    bool covers(int l) const { return loaded && l >= first && l <= last && l < layers && h_scale[l] != 0.f; }
    void apply(float* x, int T, int hidden, int layer, cudaStream_t s) const {
        if (!loaded || !covers(layer)) return;
        cvec_detail::cvec_apply_kernel<<<T, 256, 0, s>>>(x, T, hidden, d_dir, d_scale, layer, mode, 1);
    }
};
/// Load a control-vector GGUF. Returns false with `err` on any mismatch; the
/// runtime stays empty unless every direction tensor matches `hidden`.
inline bool cvec_load(CvecRuntime& out, const std::string& path, float scale_value,
                      int first, int last, int mode, int hidden, std::string& err) {
    try {
        strata::GgufFile f(path, {}, true);
        out.layers = 0;
        // Control-vector layers are 1-based (direction.1 .. direction.N), so
        // allocate index `last` inclusive.
        const size_t slots = size_t(max(0, last)) + 1;
        out.h_dir.assign(slots * size_t(hidden), 0.f);
        out.h_scale.assign(slots, 0.f);
        size_t found = 0;
        for (const auto& t : f.tensors()) {
            const auto p = t.name.rfind("direction.", 0);
            if (p != 0 || t.shape.size() != 1 || t.shape[0] != uint64_t(hidden) || t.type != 0)
                continue;
            const int layer = std::atoi(t.name.c_str() + 10);
            if (layer <= 0 || layer > last) continue;   // only steered layers are loaded
            if (layer > out.layers) out.layers = layer;
            const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
            for (int k = 0; k < hidden; ++k) out.h_dir[size_t(layer) * size_t(hidden) + k] += scale_value * src[k];
            float norm2 = 0.f;
            for (int k = 0; k < hidden; ++k) norm2 += out.h_dir[size_t(layer) * size_t(hidden) + k] *
                                                    out.h_dir[size_t(layer) * size_t(hidden) + k];
            const float s = sqrtf(norm2);
            if (mode == 0 && s > 0.f) {
                for (int k = 0; k < hidden; ++k) out.h_dir[size_t(layer) * size_t(hidden) + k] /= s;
            }
            out.h_scale[layer] = (mode == 0) ? s : 1.f;
            ++found;
        }
        if (!found) { err = "no direction.<layer> tensor with matching hidden size"; return false; }
        if (cudaMalloc(&out.d_dir, out.h_dir.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&out.d_scale, out.h_scale.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&out.d_on, sizeof(int)) != cudaSuccess) {
            err = "CUDA allocation for control vector failed"; return false;
        }
        cudaMemcpy(out.d_dir, out.h_dir.data(), out.h_dir.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(out.d_scale, out.h_scale.data(), out.h_scale.size() * sizeof(float), cudaMemcpyHostToDevice);
        const int on = 1;
        cudaMemcpy(out.d_on, &on, sizeof(int), cudaMemcpyHostToDevice);
        out.first = first; out.last = last; out.mode = mode; out.loaded = true;
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

} // namespace strata::qwen38
