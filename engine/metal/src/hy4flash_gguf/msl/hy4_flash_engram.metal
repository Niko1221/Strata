// engine/metal/src/hy4flash_gguf/msl/hy4_flash_engram.metal - the Engram injection (semantics.md §4).
//
// The hash, the row ids and the reads are the host's (engram.h: the rows arrive while the GPU is busy).  What is
// here is everything that touches the numbers, in the reference's order (hy4_block.py engram):
//
//   r      = rms_norm(row, 1.1920929e-07) * engram_lookup_norm[col] * valid[t][col]     per (token, column)
//   kv     = engram_kv . r + engram_kv_b                                                [hc + 1][n_embd]
//   dot    = sum_d rms_norm(x)[d] * rms_norm(key)[d] * engram_gate[i][d] / sqrt(n_embd) per (token, stream)
//   gate   = sigmoid(sqrt(max(|dot|, 1e-6)) * sign(dot))             sign(0) = +1 (step(-dot) * -2 + 1)
//   delta  = rms_norm(value) * engram_pre_gamma * (gate + engram_hc_scale[i]) * any_valid
//   out    = x + delta
//
// THE EPSILON IS NOT THE MODEL'S.  Every rms_norm here uses 1.1920929e-07 (FLT_EPSILON), not
// attention.layer_norm_rms_epsilon (1e-5): the Engram path in the reference calls ggml_rms_norm with its own
// value, and using the block's would move the result by ~1e-5 relative.
//
// F8_E4M3 (ggml type 43) IS NEW CODE.  A row is 96 values of ONE BYTE, no scale: bias 7, the OCP FN variant, so
// S.1111.111 is NaN and there are no infinities.  The table is the only F8_E4M3 tensor in the model and its 59 GiB
// never become resident - the host preads the rows it needs (engram.h) into the buffer this reads.
//
// T INVARIANCE: one thread group per output row (a (token, column) row of 96, or a (token, stream) pair), with a
// compile-time thread count, so no reduction sees the window.

struct Hy4FlashEngramRowArgs {
    uint dim;        // engram_head_dim (96)
    uint columns;    // engram_heads (16)
    uint row_bytes;  // bytes of one table row in the file (96 for F8_E4M3)
    uint out_stride; // floats between tokens of the output (columns * dim)
    float eps;
};

// E4M3, the OCP FN variant (tmp/tests/hy4_flash/pyref/dequant.py _fp8_e4m3_table, which is bitwise ggml's to_float).
static inline float hy4_flash_f8_e4m3(uchar b) {
    const float s = (b & 0x80) ? -1.0f : 1.0f;
    const uint e = (b >> 3) & 0xF;
    const uint m = b & 7;
    if (e == 0xF && m == 7) return NAN;
    if (e == 0) return s * (float(m) / 8.0f) * 0.015625f;            // 2^-6
    return s * (1.0f + float(m) / 8.0f) * exp2(float(int(e) - 7));
}

// grid (columns * n_tok), threads HY4_FLASH_TG.  `rows` is token major: row (t, c) at (t * columns + c) * row_bytes.
kernel void hy4_flash_engram_rows(device const uchar* rows [[buffer(0)]], device const float* wnorm [[buffer(1)]],
                            device const float* valid [[buffer(2)]], device float* y [[buffer(3)]],
                            constant Hy4FlashEngramRowArgs& a [[buffer(4)]],
                            uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HY4_FLASH_TG_SG];
    const uint r = tgp.x;                      // t * columns + c
    const uint c = r % a.columns, t = r / a.columns;
    device const uchar* src = rows + (ulong)r * a.row_bytes;
    float ss = 0.0f;
    for (uint d = tid; d < a.dim; d += HY4_FLASH_TG) {
        const float v = hy4_flash_f8_e4m3(src[d]);
        ss += v * v;
    }
    const float scale = precise::rsqrt(hy4_flash_tg_sum(ss, part, sg, lane) / float(a.dim) + a.eps);
    const float keep = valid[r];
    device float* o = y + (ulong)t * a.out_stride + (ulong)c * a.dim;
    device const float* w = wnorm + (ulong)c * a.dim;
    for (uint d = tid; d < a.dim; d += HY4_FLASH_TG)
        o[d] = hy4_flash_f8_e4m3(src[d]) * scale * w[d] * keep;
}

struct Hy4FlashEngramArgs {
    uint n_embd;
    uint hc;          // hyper-connection streams (4)
    uint x_stride;    // floats between tokens of x (hc * n_embd)
    uint kv_stride;   // floats between tokens of kv ((hc + 1) * n_embd)
    float sqrt_n;     // sqrt(n_embd) as f32.  The reference DIVIDES the sum by it (hy4_block.py engram:
                      // `.sum(-1) / math.sqrt(self.n_embd)`), so this kernel divides too - multiplying by a
                      // reciprocal is a different rounding, and getting the direction wrong is a factor of
                      // n_embd = 3072 in the gate's argument, which saturates the sigmoid to 0 or 1
    float eps;
};

// dot and the gate, one thread group per (stream, token).  grid (hc, n_tok), threads HY4_FLASH_TG.
kernel void hy4_flash_engram_gate(device const float* x [[buffer(0)]], device const float* kv [[buffer(1)]],
                            device const float* egate [[buffer(2)]], device float* dot_out [[buffer(3)]],
                            device float* gate_out [[buffer(4)]], constant Hy4FlashEngramArgs& a [[buffer(5)]],
                            uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HY4_FLASH_TG_SG];
    const uint i = tgp.x, t = tgp.y;
    device const float* xr = x + (ulong)t * a.x_stride + (ulong)i * a.n_embd;
    device const float* kr = kv + (ulong)t * a.kv_stride + (ulong)i * a.n_embd;
    device const float* gr = egate + (ulong)i * a.n_embd;
    float sx = 0.0f, sk = 0.0f;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) {
        sx += xr[d] * xr[d];
        sk += kr[d] * kr[d];
    }
    const float nx = precise::rsqrt(hy4_flash_tg_sum(sx, part, sg, lane) / float(a.n_embd) + a.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float nk = precise::rsqrt(hy4_flash_tg_sum(sk, part, sg, lane) / float(a.n_embd) + a.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) s += (xr[d] * nx) * (kr[d] * nk) * gr[d];
    const float dot = hy4_flash_tg_sum(s, part, sg, lane) / a.sqrt_n;
    if (tid == 0) {
        const float mag = precise::sqrt(max(fabs(dot), 1e-6f));
        const float sgn = dot < 0.0f ? -1.0f : 1.0f;   // step(-dot) * -2 + 1: dot == 0 gives +1
        dot_out[t * a.hc + i] = dot;
        gate_out[t * a.hc + i] = hy4_flash_sigmoid(mag * sgn);
    }
}

// delta and the injection, one thread group per (stream, token).  grid (hc, n_tok), threads HY4_FLASH_TG.
// The value row's norm is recomputed in each of the hc groups (3072 multiply-adds) rather than passed between
// dispatches: it keeps the value bitwise the same for every stream and costs one pass over 12 KB.
kernel void hy4_flash_engram_delta(device const float* x [[buffer(0)]], device const float* kv [[buffer(1)]],
                             device const float* pre_gamma [[buffer(2)]], device const float* hc_scale [[buffer(3)]],
                             device const float* gate [[buffer(4)]], device const float* any_valid [[buffer(5)]],
                             device float* delta [[buffer(6)]], device float* out [[buffer(7)]],
                             constant Hy4FlashEngramArgs& a [[buffer(8)]],
                             uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HY4_FLASH_TG_SG];
    const uint i = tgp.x, t = tgp.y;
    device const float* vr = kv + (ulong)t * a.kv_stride + (ulong)a.hc * a.n_embd;   // the value row is the last one
    float sv = 0.0f;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) sv += vr[d] * vr[d];
    const float nv = precise::rsqrt(hy4_flash_tg_sum(sv, part, sg, lane) / float(a.n_embd) + a.eps);
    const float g = gate[t * a.hc + i], keep = any_valid[t];
    const ulong o = (ulong)t * a.x_stride + (ulong)i * a.n_embd;
    device const float* sr = hc_scale + (ulong)i * a.n_embd;
    for (uint d = tid; d < a.n_embd; d += HY4_FLASH_TG) {
        const float dv = (vr[d] * nv * pre_gamma[d]) * (g + sr[d]) * keep;
        delta[o + d] = dv;
        out[o + d] = x[o + d] + dv;
    }
}
