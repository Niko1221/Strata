// metal/src/engine/msl/cvec.metal - the control vector on the residual stream (include/strata/kernels/cvec.hpp; the
// CUDA engine's src/kernels/cuda/cvec.cu) for a window or prompt chunk of T rows:
//
//   (write)   h <- h + bo * 2 sigmoid(inj / hc)     the layer's pending FFN write, gr_norm's arithmetic (gr.metal)
//   project:  h <- h - s_l (h . v_l) v_l            add:  h <- h + v_l
//
// on every one of the hc streams of every row.  `on` is the per-request switch: with it 0 (or s_l 0) only the write
// is applied, bitwise gr_norm's own, so the next read (apply = 0) sees the values the fused read would have made.

#include <metal_stdlib>
using namespace metal;

struct CvecArgs {
    uint n_embd;   // 2560
    uint hc;       // 4
    uint layer;
    uint write;    // 1: apply the pending write first
    uint mode;     // 0 = project, 1 = add
};

constant constexpr int CV_THREADS = 256, CV_MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

// grid (hc, T), 256 threads
kernel void cvec_apply(device float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                       device const float* inj [[buffer(2)]], device const float* dir [[buffer(3)]],
                       device const float* s_l [[buffer(4)]], device const int* on [[buffer(5)]],
                       constant CvecArgs& a [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device float* r = R + (t * a.hc + c) * n;
    const float s = s_l[a.layer];
    const bool steer = on[0] != 0 && s != 0.0f;   // uniform over the threadgroup
    if (!steer && !a.write) return;
    device const float* v = dir + a.layer * n;
    const float g = a.write ? 2.0f / (1.0f + exp(-inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    float x[CV_MAXK];
    float dot = 0.0f;
    for (int k = 0; k < CV_MAXK; ++k) {
        const uint d = tid + k * CV_THREADS;
        if (d < n) {
            float xv = r[d];
            if (a.write) xv = fma(bo[t * n + d], g, xv);
            x[k] = xv;
            if (steer && a.mode == 0) dot = fma(xv, v[d], dot);
        }
    }
    if (steer && a.mode == 0) {
        dot = simd_sum(dot);
        if (lane == 0) part[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (int i = 0; i < CV_THREADS / 32; ++i) tot += part[i];
        dot = tot * s;   // s (h . v)
    }
    for (int k = 0; k < CV_MAXK; ++k) {
        const uint d = tid + k * CV_THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = a.mode == 0 ? fma(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}
