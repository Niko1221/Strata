#pragma OPENCL FP_CONTRACT OFF

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

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void swiglu_q8_1(__global const float* gu, __global uchar* hq, const uint n_ff)
{
    const uint b    = get_group_id(0);           // flat q8_1 block over every entry
    const uint lane = get_local_id(0);
    const uint bpe  = n_ff >> 5u;
    const uint e    = b / bpe;
    const uint blk  = b - e * bpe;
    __global const float* gp = gu + (size_t) e * 2u * n_ff + blk * 32u;
    const float g = gp[lane];
    const float u = gp[n_ff + lane];
    // __expf is native_exp's counterpart; the quantiser is insensitive to which of the two is used
    // (ocl_swiglu.cpp: 0 code differences over 8 reduction/exp combinations), so the fast one is right.
    const float h = (g / (1.0f + native_exp(-g))) * u;
    q8_1_store(h, lane, (__global uint*) (hq + (size_t) b * 36u), hq + (size_t) b * 36u + 4u);
}
