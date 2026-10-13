#pragma OPENCL FP_CONTRACT OFF
#define IQ_WG 64
#define IQ_R 4
#define IQ_NCHUNK 40
#define IQ_BB 18

// block_q8_1 is 36 bytes = 9 uint: word 0 is {fp16 d, fp16 sum}, words 1..8 are the 32 int8.  So the
// activation buffer is read through one u32 view - the d is the low half of the first word - and the
// whole thing stays a single cl_mem bound to one argument.
inline float f16_to_f(uint h) {
    const uint s = (h >> 15u) & 1u;
    const uint e = (h >> 10u) & 0x1Fu;
    const uint m = h & 0x3FFu;
    if (e == 31u) return as_float((s << 31u) | 0x7F800000u | (m << 13u));
    if (e == 0u) {
        if (m == 0u) return as_float(s << 31u);
        const float v = (float) m * 5.9604644775390625e-08f;
        return s ? -v : v;
    }
    return as_float((s << 31u) | ((e + 112u) << 23u) | (m << 13u));
}

// f32 -> f16, round to nearest even, exactly as __float2half.  Written out rather than using the
// fp16 extension so the rounding is visible: q8_1_ds stores both the scale and the block sum this way.
inline uint f32_to_f16(float v) {
    const uint u = as_uint(v);
    const uint s = (u >> 16) & 0x8000u;
    const uint e = (u >> 23) & 0xFFu;
    const uint m = u & 0x7FFFFFu;
    if (e == 255u) return s | 0x7C00u | (m != 0u ? 0x200u : 0u);
    const int ne = (int) e - 127 + 15;
    if (ne >= 31) return s | 0x7C00u;
    if (ne <= -11) return s;
    if (ne <= 0) {
        const uint mm = m | 0x800000u;
        const uint sh = (uint) (14 - ne);
        uint r = mm >> sh;
        const uint rem = mm & ((1u << sh) - 1u);
        const uint halfway = 1u << (sh - 1u);      // not `half`: that is a type name in OpenCL
        if (rem > halfway || (rem == halfway && (r & 1u) != 0u)) ++r;
        return s | r;
    }
    uint r = m >> 13;
    const uint rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) != 0u)) {
        if (++r == 0x400u) { r = 0u; if (ne + 1 >= 31) return s | 0x7C00u; return s | ((uint) (ne + 1) << 10); }
    }
    return s | ((uint) ne << 10) | r;
}

inline uint rd16(__global const uchar* p, uint o) {
    return (uint) p[o] | ((uint) p[o + 1u] << 8);
}
inline uint rd32(__global const uchar* p, uint o) {
    return (uint) p[o] | ((uint) p[o + 1u] << 8) | ((uint) p[o + 2u] << 16) | ((uint) p[o + 3u] << 24);
}

inline uint unpack_ksigns(uint v) {
    const uint p = (uint) popcount(v & 0xFFu) & 1u;
    return (v & 0xFFu) ^ (p << 7u);
}

// vcmpne4/vsub4 as explicit byte loops: this driver mis-lowers the branchless bit trick in bit 31 of the
// mask (ocl_iq3.cpp), which turns a negative weight positive.  Measured bit-exact against __vcmpne4.
inline int vcmpne4(int a, int b) {
    const uint t = as_uint(a) ^ as_uint(b);
    uint r = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (((t >> (8 * i)) & 0xFFu) != 0u) r |= 0xFFu << (8 * i);
    return as_int(r);
}

inline int vsub4(int a, int b) {
    const uint ua = as_uint(a), ub = as_uint(b);
    const uint kHigh = 0x80808080u;
    return as_int(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}

// __dp4a without cl_khr_integer_dot_product (this device lacks it): mad24 x4 with sign-extended bytes,
// bit-exact against __dp4a and the fastest form the driver has (ocl_dot.cpp, 0/65536 mismatches).
inline int dp4x8(int a, int b, int c) {
    int s = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        s = mad24((a << (24 - 8 * i)) >> 24, (b << (24 - 8 * i)) >> 24, s);
    return s;
}

inline int pack4(int b0, int b1, int b2, int b3) {
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

// q8_1_finite: the block scale and sum are clamped to the largest finite half before being rounded to
// one, so a massive-activation dimension gives an approximate product instead of inf * 0 = NaN (#606).
inline float f16_max_clamp(float v) {
    return fabs(v) > 65504.0f ? copysign(65504.0f, v) : v;
}

// q = round(x/d) clamped to +-127, 0 when the block is all zeros.  OpenCL's round() is ties-away-from-zero,
// which is roundf on the CUDA side; rint() would be ties-to-even and differs on an exact half.
inline int q8_1_quant(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = round(xi / d);
    return (int) (q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q));
}

// q8_1_store (iq_kernels.cu): amax and sum come from an xor butterfly over the warp, offsets 16,8,4,2,1.
// The max is order-insensitive; the sum is not, and it is stored, so the tree has to be this one.  Every
// lane ends up with the same value.
inline void q8_1_store(float h, uint lane, __global uint* ds, __global uchar* qs) {
    __local float lmem[64];
    __local float lsc[64];
    lmem[lane] = fabs(h);
    lsc[lane] = h;
    for (uint o = 16u; o > 0u; o >>= 1u) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float va = lmem[lane ^ o];
        const float vs = lsc[lane ^ o];
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[lane] = fmax(lmem[lane], va);
        lsc[lane] = lsc[lane] + vs;
    }
    const float d = f16_max_clamp(lmem[lane] / 127.0f);
    qs[lane] = (uchar) q8_1_quant(h, d, lmem[lane]);
    if (lane == 0u) ds[0] = f32_to_f16(d) | (f32_to_f16(f16_max_clamp(lsc[lane])) << 16u);
}

// `wt` is the arena holding every resident expert blob; `ent_off[e]` locates entry e's expert in it,
// `ent_act[e]` the activation row that entry multiplies (the token's q8_1 for gate/up, e itself for the
// down pass, where the activation is the entry's own SwiGLU output).  out row = e * out_stride + row.
__kernel __attribute__((reqd_work_group_size(IQ_WG, 1, 1)))
void iq_row_dbg(__global const uchar* wt, __global const uint* xq,
            __global const uint* t2, __global const uint* t3, __global const uchar* kv,
            __global float* outv, __global const ulong* ent_off, __global const uint* ent_act,
            const uint n_entries, const uint bpe, const uint n_row_e,
            const uint wrow_bytes, const uint w_extra, const uint act_blk_row, const uint out_stride,
            __global float* dbg)
{
    const uint lane = get_local_id(0);
    if (get_group_id(0) != 0u) return;
    const uint e    = get_group_id(0) / bpe;
    if (e >= n_entries) return;                      // the grid is exact; the guard is per-group uniform
    const uint row0 = (get_group_id(0) - e * bpe) * (uint) IQ_R;

    __global const uchar* W  = wt + ent_off[e] + w_extra;
    __global const uint*  XW = xq + (size_t) ent_act[e] * act_blk_row * 9u;

    float acc[IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) acc[r] = 0.0f;

    for (uint k = lane; k < (uint) IQ_NCHUNK; k += (uint) IQ_WG) {

        const uint kbx = k >> 1u, m = k & 1u;
        const uint xo = kbx * 9u;
        const float dx = f16_to_f(XW[xo] & 0xFFFFu);
        const uint xw0 = XW[xo + 1u + 2u * m];
        const uint xw1 = XW[xo + 2u + 2u * m];
        const uint xw4 = XW[xo + 5u + 2u * m];
        const uint xw5 = XW[xo + 6u + 2u * m];

        #pragma unroll
        for (int r = 0; r < IQ_R; ++r) {
            const uint row = min(row0 + (uint) r, n_row_e - 1u);
            __global const uchar* wr = W + (size_t) row * (size_t) wrow_bytes;
            const uint base = kbx * (uint) IQ_BB;

            const uint qa = rd32(wr, base + 2u + 8u * m);
            const uint qb = rd32(wr, base + 2u + 8u * m + 4u);
            int g[4];
            g[0] = pack4((int) (char) kv[qa & 0x0Fu], (int) (char) kv[(qa >> 8) & 0x0Fu],
                         (int) (char) kv[(qa >> 16) & 0x0Fu], (int) (char) kv[(qa >> 24) & 0x0Fu]);
            g[1] = pack4((int) (char) kv[(qa >> 4) & 0x0Fu], (int) (char) kv[(qa >> 12) & 0x0Fu],
                         (int) (char) kv[(qa >> 20) & 0x0Fu], (int) (char) kv[(qa >> 28) & 0x0Fu]);
            g[2] = pack4((int) (char) kv[qb & 0x0Fu], (int) (char) kv[(qb >> 8) & 0x0Fu],
                         (int) (char) kv[(qb >> 16) & 0x0Fu], (int) (char) kv[(qb >> 24) & 0x0Fu]);
            g[3] = pack4((int) (char) kv[(qb >> 4) & 0x0Fu], (int) (char) kv[(qb >> 12) & 0x0Fu],
                         (int) (char) kv[(qb >> 20) & 0x0Fu], (int) (char) kv[(qb >> 28) & 0x0Fu]);
            int si = 0;
            si = dp4x8(g[0], (int) xw0, si);   // l = 0, low nibbles  <-> q8[iqs + 0]
            si = dp4x8(g[1], (int) xw4, si);   // l = 0, high nibbles <-> q8[iqs + 4]
            si = dp4x8(g[2], (int) xw1, si);   // l = 1, low nibbles  <-> q8[iqs + 1]
            si = dp4x8(g[3], (int) xw5, si);   // l = 1, high nibbles <-> q8[iqs + 5]
            const float dw = f16_to_f(rd16(wr, base));

            // --use_fast_math makes the CUDA accumulation a mul.f32 then an fma.rn.f32; dw*dx rounds on
            // its own and the accumulate rounds once.  FP_CONTRACT OFF keeps that split.
            acc[r] = fma(dw * dx, (float) si, acc[r]);
            if (r == 0) {
                dbg[lane * 8u + 0u] = (float) si;
                dbg[lane * 8u + 1u] = dw;
                dbg[lane * 8u + 2u] = dx;
                dbg[lane * 8u + 3u] = (float) kbx;
                dbg[lane * 8u + 4u] = (float) k;
                dbg[lane * 8u + 5u] = as_float(qa);
                dbg[lane * 8u + 6u] = as_float(qb);
                dbg[lane * 8u + 7u] = acc[r];
            }
        }
    }

    // The reference tree, spelled out: IQ_WG/2 .. 4, 2, 1.  Every row is reduced at once so the
    // barriers are paid once per work-group instead of once per row.
    __local float lred[IQ_WG * IQ_R];
    #pragma unroll
    for (int r = 0; r < IQ_R; ++r) lred[lane * IQ_R + r] = acc[r];
    barrier(CLK_LOCAL_MEM_FENCE);
    dbg[512u + lane] = acc[0];
    dbg[576u + lane] = acc[1];
    for (uint off = IQ_WG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) {
            #pragma unroll
            for (int r = 0; r < IQ_R; ++r)
                lred[lane * IQ_R + r] = lred[lane * IQ_R + r] + lred[(lane + off) * IQ_R + r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0u) {
        #pragma unroll
        for (int r = 0; r < IQ_R; ++r)
            if (row0 + (uint) r < n_row_e)
                outv[(size_t) e * out_stride + row0 + (uint) r] = lred[r];
    }
}
