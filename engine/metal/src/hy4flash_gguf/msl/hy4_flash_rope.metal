// engine/metal/src/hy4flash_gguf/msl/hy4_flash_rope.metal - RoPE NORM (adjacent pairs) on the tail n_rot dims of a row.
//
// This model is LLAMA_ROPE_TYPE_NORM, not NeoX (semantics.md §9: hc_mult = 4 selects it), so the rotated pair is
// (x[2k], x[2k+1]):
//
//   theta_k        = pos * theta_scale^k          built by f32 recurrence, theta_scale = powf(base, -2 / n_rot)
//   out[2k]        = x[2k] * cos(theta_k) - x[2k+1] * sin(theta_k)
//   out[2k+1]      = x[2k] * sin(theta_k) + x[2k+1] * cos(theta_k)
//
// The angles must come from the SAME f32 recurrence ggml-cpu uses (rope_yarn with ext_factor 0, freq_scale 1,
// attn_factor 1, so mscale = 1 and theta_base is multiplied by theta_scale once per pair); powf(theta_scale, k) is a
// different number after a few dozen steps.  Each thread therefore repeats the recurrence from the position instead
// of carrying it between threads: at most 31 multiplies here (n_rot = 64), and the value is then independent of how
// the pairs are spread over threads, over heads and over tokens.
//
// Three users, three shapes (all of them one dispatch of this kernel):
//   q_pe       n_head heads of head_dim = 576, off = 512 (k_mla - n_rot), base = rope.freq_base / _swa
//   k_pe       one row of 64, off = 0
//   indexer    n_head heads of head_dim = 128, off = 64, base = the indexer's own
// `pos` is per token, so a prefill block and a decode window pass the same numbers.

struct Hy4FlashRopeArgs {
    uint n_rot;        // rotated dims at the end of the row (even)
    uint n_head;       // rows per token
    uint head_dim;     // floats in a row
    uint off;          // first rotated element of a row (head_dim - n_rot for q, 0 for k_pe)
    uint x_stride;     // floats between tokens of x (n_head * head_dim when packed)
    uint y_stride;
    float theta_scale; // powf(base, -2 / n_rot), computed on the host exactly as ggml does
    // M-RoPE only (hy4_flash_rope_mrope): `pos` is [3][pos_stride] instead of [n_tok], and mlen1 / mlen2 are
    // 3 * mrope_section[1] and 3 * mrope_section[2].  Zero on the text path, which never reads them.
    uint pos_stride;
    uint mlen1, mlen2;
};

// grid (n_rot / 2, n_head, n_tok), one thread per rotated pair.  x and y may be the same buffer: a thread reads and
// writes only its own pair.  The dims before `off` are NOT copied - the caller either rotates in place or has
// already placed the un-rotated part.
template <bool MROPE>
kernel void hy4_flash_rope_t(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       device const int* pos [[buffer(2)]], constant Hy4FlashRopeArgs& a [[buffer(3)]],
                       uint3 gid [[thread_position_in_grid]]) {
    const uint k = gid.x;
    if (2 * k >= a.n_rot || gid.y >= a.n_head) return;
    // The text path indexes pos[token] exactly as before; the image path picks the plane for this pair.
    float th = float(MROPE ? pos[hy4_flash_mrope_axis(k, a.mlen1, a.mlen2) * a.pos_stride + gid.z] : pos[gid.z]);
    for (uint j = 0; j < k; ++j) th *= a.theta_scale;
    const float c = precise::cos(th), s = precise::sin(th);
    const ulong o = (ulong)gid.y * a.head_dim + a.off + 2 * k;
    device const float* xr = x + (ulong)gid.z * a.x_stride + o;
    device float* yr = y + (ulong)gid.z * a.y_stride + o;
    const float x0 = xr[0], x1 = xr[1];
    yr[0] = x0 * c - x1 * s;
    yr[1] = x0 * s + x1 * c;
}

template [[host_name("hy4_flash_rope_norm")]] kernel decltype(hy4_flash_rope_t<false>) hy4_flash_rope_t<false>;
template [[host_name("hy4_flash_rope_mrope")]] kernel decltype(hy4_flash_rope_t<true>) hy4_flash_rope_t<true>;
