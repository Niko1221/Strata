// metal/src/engine/msl/cvec.metal - the control vector on the residual stream (include/strata/kernels/cvec.hpp; the
// CUDA engine's src/kernels/cuda/cvec.cu) for a window of T rows.  After a steered layer l, every hyper-connection
// stream h of every row becomes
//
//   project:  h <- h - s_l (h . v_l) v_l        v_l the unit direction, s_l its scale
//   add:      h <- h + d_l
//
// With `write`, the layer's pending FFN write R += bo * 2 sigmoid(inj / hc) is applied first, with gr_norm's own
// arithmetic (gr.metal): the window folds that write into the next layer's read, and a steered layer's next read
// then runs without it.  The window encodes this kernel only while the vector is on (cvec_enabled()), so a request
// with it off runs the stock dispatches.
//
// Shapes: per row t, R[t] (hc * n_embd), bo[t] (n_embd), inj[t] (hc); v is layer l's row of the table (n_embd).

#include <metal_stdlib>
using namespace metal;

struct CvecArgs {
    uint n_embd;   // 2560
    uint hc;       // 4
    uint mode;     // 0 project, 1 add
    uint write;    // 1: the pending FFN write first
    float s;       // project: s_l (add: unused)
};

constant constexpr int CV_MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

// one threadgroup per (stream c, row t), 256 threads
kernel void cvec_apply(device float* R [[buffer(0)]], device const float* bo [[buffer(1)]],
                       device const float* inj [[buffer(2)]], device const float* v [[buffer(3)]],
                       constant CvecArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint c = tg.x, t = tg.y, n = a.n_embd;
    device float* r = R + (t * a.hc + c) * n;
    const float g = a.write ? 2.0f / (1.0f + exp(-inj[t * a.hc + c] / float(a.hc))) : 0.0f;
    const bool proj = a.mode == 0;
    float x[CV_MAXK];
    float dot = 0.0f;
    for (int k = 0; k < CV_MAXK; ++k) {
        const uint d = tid + 256u * (uint) k;
        if (d < n) {
            float xv = r[d];
            if (a.write) xv = fma(bo[t * n + d], g, xv);
            x[k] = xv;
            if (proj) dot = fma(xv, v[d], dot);
        }
    }
    if (proj) {
        dot = simd_sum(dot);
        if (lane == 0) part[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += part[i];
        dot = tot * a.s;   // s (h . v)
    }
    for (int k = 0; k < CV_MAXK; ++k) {
        const uint d = tid + 256u * (uint) k;
        if (d < n) r[d] = proj ? fma(-dot, v[d], x[k]) : x[k] + v[d];
    }
}
