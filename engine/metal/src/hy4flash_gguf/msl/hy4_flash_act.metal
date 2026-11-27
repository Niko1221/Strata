// engine/metal/src/hy4flash_gguf/msl/hy4_flash_act.metal - the routed experts' clamped SwiGLU (semantics.md §7).
//
//   g' = min(g, limit)                  (one-sided: only the upper bound)
//   u' = clamp(u, -limit, limit)        (two-sided)
//   y  = g' / (1 + exp(-g')) * u'
//
// limit is swiglu_clamp_exp, read per layer from the GGUF (10 in this checkpoint).  This is the ROUTED experts' form
// only: the dense FFN of block 0 and the shared expert use plain SiLU (hy4_block.py dense_ffn), which is this kernel
// with a limit of +inf - so the same entry point serves both, and the clamp is skipped when `limit` is not finite.
// Elementwise, so T invariance is immediate; the test still covers it, because a stride bug would break it.

struct Hy4FlashGluArgs {
    uint n;          // elements in a row
    uint g_stride;   // floats between rows of g
    uint u_stride;
    uint y_stride;
    float limit;
};

// grid (ceil(n / HY4_FLASH_TG) * HY4_FLASH_TG, n_tok)
kernel void hy4_flash_swiglu_clamp(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                             device float* y [[buffer(2)]], constant Hy4FlashGluArgs& a [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    const float gv = min(g[(ulong)gid.y * a.g_stride + gid.x], a.limit);
    const float uv = clamp(u[(ulong)gid.y * a.u_stride + gid.x], -a.limit, a.limit);
    y[(ulong)gid.y * a.y_stride + gid.x] = hy4_flash_silu(gv) * uv;
}
