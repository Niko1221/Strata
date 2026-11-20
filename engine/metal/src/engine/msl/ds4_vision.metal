// metal/src/engine/msl/ds4_vision.metal - DeepSeek-V4.1's vision tower on Metal.
//
// The kernels are the Metal form of src/dsv41/ds4_vision.cu's eight CUDA kernels (patchify, add_rows, rmsnorm, qkv_rope,
// softmax, silu_mul, unfold, bias_act), with the same arithmetic and the same BF16 rounding points, plus one tiled GEMM that
// takes the place of the cuBLAS calls (BF16 operands, FP32 accumulation, FP32 or BF16 result).
//
// Credit: the way a DeepSeek-V4 vision tower is laid out on Metal (which steps round to BF16, the 2-D rotary on q and k, the
// aligner's unfold order, a polynomial erf because MSL has none) follows antirez/ds4 (MIT), commit 890898cf,
// metal/deepseek4_vision.metal, metal/glm53_vision.metal and metal/glm53_bf16.metal.  Copyright (c) 2026 The ds4.c authors,
// (c) 2023-2026 The ggml authors, (c) 2023 DeepSeek.  See third_party/antirez-ds4/LICENSE and NOTICE.md.
// The tiled GEMM and the host side are Strata's own.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// every pointer is a device address (strata::gpu keeps all allocations in one residency set)
static inline float vis_bf(ushort b) { return as_type<float>(uint(b) << 16); }
static inline ushort vis_bfr(float f) {   // F32 -> BF16 bits, round to nearest even; a NaN stays a NaN
    const uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
    return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// the block reduction of ds4_vision.cu: xor shuffles inside a warp, then the first warp over the warps' values
static inline float vis_block_reduce(float v, bool is_max, threadgroup float* red, ushort tid, ushort nthreads) {
    for (ushort o = 16; o > 0; o >>= 1) {
        const float u = simd_shuffle_xor(v, o);
        v = is_max ? fmax(v, u) : v + u;
    }
    const ushort lane = tid & 31, w = tid >> 5, nw = (nthreads + 31) >> 5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[w] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nw ? red[lane] : (is_max ? -FLT_MAX : 0.0f);
    for (ushort o = 16; o > 0; o >>= 1) {
        const float u = simd_shuffle_xor(r, o);
        r = is_max ? fmax(r, u) : r + u;
    }
    return r;   // every thread of the group has it (the first warp's value is the same in all warps: each reduces red[])
}

// erf: MSL has none.  erf(x) = 2/sqrt(pi) exp(-x^2) sum_n 2^n x^(2n+1) / (1 3 5 ... (2n+1)): positive terms only, so no cancellation.
static inline float vis_erf(float x) {
    const float a = fabs(x);
    if (a >= 4.0f) return x < 0.0f ? -1.0f : 1.0f;
    const float a2 = a * a;
    float term = a, sum = a;
    for (int n = 1; n < 100; ++n) {
        term *= 2.0f * a2 / float(2 * n + 1);
        sum += term;
        if (term < sum * 1.0e-8f) break;
    }
    const float r = 1.1283791670955126f * precise::exp(-a2) * sum;
    return x < 0.0f ? -r : r;
}

struct VisPatchArgs {
    device const uchar* rgb;
    device ushort* out;
    int width, gw, p;
};
// patch i of the gh x gw grid: its 3 x p x p values, channel-major.  One group a patch.
kernel void vis_patchify(constant VisPatchArgs& a [[buffer(0)]], uint gi [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int pp = a.p * a.p;
    for (int e = tid; e < 3 * pp; e += nt) {
        const int c = e / pp, r = e % pp;
        const int y = (int(gi) / a.gw) * a.p + r / a.p, x = (int(gi) % a.gw) * a.p + r % a.p;
        const float u = float(a.rgb[(ulong(y) * ulong(a.width) + ulong(x)) * 3ul + ulong(c)]);
        a.out[ulong(gi) * ulong(3 * pp) + ulong(e)] = vis_bfr(((u / 255.0f) - 0.5f) / 0.5f);
    }
}

struct VisAddArgs {
    device float* x;
    device const float* y;      // 0: none
    device const ushort* bias;  // 0: none
    int ldy, dim;
};
// x[r] += y[r] (if any) + bias (if any).  One group a row.
kernel void vis_add_rows(constant VisAddArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    for (int e = tid; e < a.dim; e += nt) {
        float v = a.x[ulong(r) * ulong(a.dim) + ulong(e)];
        if (a.y != nullptr) v += a.y[ulong(r) * ulong(a.ldy) + ulong(e)];
        if (a.bias != nullptr) v += vis_bf(a.bias[e]);
        a.x[ulong(r) * ulong(a.dim) + ulong(e)] = v;
    }
}

struct VisNormArgs {
    device const float* x;
    device const ushort* w;
    device ushort* out;
    int dim;
    float eps;
};
kernel void vis_rmsnorm(constant VisNormArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float* xr = a.x + ulong(r) * ulong(a.dim);
    float ss = 0.0f;
    for (int e = tid; e < a.dim; e += nt) ss = fma(xr[e], xr[e], ss);
    ss = vis_block_reduce(ss, false, red, tid, nt);
    const float rs = rsqrt(ss / float(a.dim) + a.eps);
    for (int e = tid; e < a.dim; e += nt) a.out[ulong(r) * ulong(a.dim) + ulong(e)] = vis_bfr(vis_bf(a.w[e]) * (xr[e] * rs));
}

struct VisRopeArgs {
    device const float* y;
    device const ushort* bias;
    device const float* inv_freq;
    device ushort* qkv;
    int r0, gw, heads, hd;
};
// a piece's rows of wqkv's output (+ bias, rounded to BF16) into qkv, q and k rotated.  One group a row.
kernel void vis_qkv_rope(constant VisRopeArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int gr = a.r0 + int(r), dim = a.heads * a.hd, half_ = a.hd / 2, nf = a.hd / 4;
    const int row = gr / a.gw, col = gr % a.gw;
    device const float* yr = a.y + ulong(r) * ulong(3 * dim);
    device ushort* o = a.qkv + ulong(gr) * ulong(3 * dim);
    for (int i = tid; i < 2 * a.heads * half_; i += nt) {
        const int base = (i / half_) * a.hd, d = i % half_;
        const float x1 = vis_bf(vis_bfr(yr[base + d] + vis_bf(a.bias[base + d])));
        const float x2 = vis_bf(vis_bfr(yr[base + d + half_] + vis_bf(a.bias[base + d + half_])));
        const float ang = float(d < nf ? row : col) * a.inv_freq[d % nf];
        const float sn = precise::sin(ang), cs = precise::cos(ang);
        o[base + d] = vis_bfr(x1 * cs - x2 * sn);
        o[base + d + half_] = vis_bfr(x2 * cs + x1 * sn);
    }
    for (int e = 2 * dim + tid; e < 3 * dim; e += nt) o[e] = vis_bfr(yr[e] + vis_bf(a.bias[e]));
}

struct VisSoftmaxArgs {
    device const float* s;
    device ushort* p;
    int n;
    float scale;   // applied to the scores first (the attention scale when the GEMM did not fold it in; exact: a power of two)
};
constant int kSmPer = 20;   // a row of up to nt * 20 keys in registers (512 x 20 = 10240)
kernel void vis_softmax(constant VisSoftmaxArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float* sr = a.s + ulong(r) * ulong(a.n);
    device ushort* pr = a.p + ulong(r) * ulong(a.n);
    float v[kSmPer];
    float m = -FLT_MAX;
    for (int k = 0; k < kSmPer; ++k) {
        const int e = int(tid) + k * int(nt);
        v[k] = e < a.n ? sr[e] * a.scale : -FLT_MAX;
        m = fmax(m, v[k]);
    }
    m = vis_block_reduce(m, true, red, tid, nt);
    float sum = 0.0f;
    for (int k = 0; k < kSmPer; ++k) {
        const int e = int(tid) + k * int(nt);
        v[k] = e < a.n ? precise::exp(v[k] - m) : 0.0f;
        sum += v[k];
    }
    const float inv = 1.0f / vis_block_reduce(sum, false, red, tid, nt);
    for (int k = 0; k < kSmPer; ++k) {
        const int e = int(tid) + k * int(nt);
        if (e < a.n) pr[e] = vis_bfr(v[k] * inv);
    }
}

struct VisSiluArgs {
    device const float* y;
    device ushort* mid;
    int inter;
};
// mid = silu(gate) * up, gate | up the halves of w1's row.  Grid (ceil(inter / 256), rows).
kernel void vis_silu_mul(constant VisSiluArgs& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    const int e = int(g.x) * 256 + int(tid);
    if (e >= a.inter) return;
    const ulong r = g.y;
    const float gt = vis_bf(vis_bfr(a.y[r * ulong(2 * a.inter) + ulong(e)]));
    const float up = vis_bf(vis_bfr(a.y[r * ulong(2 * a.inter) + ulong(a.inter + e)]));
    a.mid[r * ulong(a.inter) + ulong(e)] = vis_bfr(vis_bf(vis_bfr(gt / (1.0f + precise::exp(-gt)))) * up);
}

struct VisUnfoldArgs {
    device const ushort* xn;
    device ushort* un;
    int gh, gw, m, dim, lw;
};
// the aligner's input: block (bi, bj) of the m x m patches as one row, value c * m * m + kh * m + kw, zero past the grid.  One group a row.
kernel void vis_unfold(constant VisUnfoldArgs& a [[buffer(0)]], uint l [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]], ushort nt [[threads_per_threadgroup]]) {
    const int bi = int(l) / a.lw, bj = int(l) % a.lw, mm = a.m * a.m;
    for (int f = tid; f < a.dim * mm; f += nt) {
        const int c = f / mm, k = f % mm, y = bi * a.m + k / a.m, x = bj * a.m + k % a.m;
        a.un[ulong(l) * ulong(a.dim * mm) + ulong(f)] =
            (y < a.gh && x < a.gw) ? a.xn[(ulong(y) * ulong(a.gw) + ulong(x)) * ulong(a.dim) + ulong(c)] : ushort(0);
    }
}

struct VisBiasActArgs {
    device const float* y;
    device const ushort* bias;
    device ushort* out;
    int n, gelu;
};
// out = act(bf16(y + bias)): GELU (erf) or none.  Grid (ceil(n / 256), rows).
kernel void vis_bias_act(constant VisBiasActArgs& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    const int e = int(g.x) * 256 + int(tid);
    if (e >= a.n) return;
    const ulong r = g.y;
    float v = vis_bf(vis_bfr(a.y[r * ulong(a.n) + ulong(e)] + vis_bf(a.bias[e])));
    if (a.gelu != 0) v = 0.5f * v * (1.0f + vis_erf(v * 0.70710678118654752f));
    a.out[r * ulong(a.n) + ulong(e)] = vis_bfr(v);
}

// ---- the GEMM: Y[r][o] = alpha * sum_k X[r][k] * W(o, k), BF16 operands, FP32 accumulation.
//   X: [rows][in] with row stride ldx.   W: flags bit 0 clear: [out][in] with row stride ldw (k contiguous);
//   bit 0 set: [in][out] with row stride ldw (o contiguous).   Y: [rows][out], row stride ldy, FP32 (bit 1 clear) or BF16 (set).
// A 64 x 64 tile a group (4 simdgroups of 32 x 32), k in steps of 32, operands staged as F32 in threadgroup memory.
struct VisGemmArgs {
    device const ushort* X;
    device const ushort* W;
    device uchar* Y;
    uint rows, in, out, ldx, ldw, ldy;
    float alpha;
    uint flags;
};
kernel void vis_gemm(constant VisGemmArgs& a [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[4096];   // As [64][32] then Bs [32][64]; reused for the result tile [64][64]
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + 2048;
    const uint o0 = tgp.x * 64u, r0 = tgp.y * 64u;
    const uint sr = (uint(sg) >> 1) * 32u, so = (uint(sg) & 1u) * 32u;
    const bool wn = (a.flags & 1u) != 0u;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < a.in; k0 += 32u) {
        for (uint idx = tid; idx < 2048u; idx += 128u) {
            const uint r = idx >> 5, kk = idx & 31u;
            const uint gr = r0 + r, gk = k0 + kk;
            As[idx] = (gr < a.rows && gk < a.in) ? vis_bf(a.X[ulong(gr) * ulong(a.ldx) + ulong(gk)]) : 0.0f;
        }
        if (!wn) {
            for (uint idx = tid; idx < 2048u; idx += 128u) {
                const uint o = idx >> 5, kk = idx & 31u;
                const uint go = o0 + o, gk = k0 + kk;
                Bs[kk * 64u + o] = (go < a.out && gk < a.in) ? vis_bf(a.W[ulong(go) * ulong(a.ldw) + ulong(gk)]) : 0.0f;
            }
        } else {
            for (uint idx = tid; idx < 2048u; idx += 128u) {
                const uint kk = idx >> 6, o = idx & 63u;
                const uint go = o0 + o, gk = k0 + kk;
                Bs[idx] = (go < a.out && gk < a.in) ? vis_bf(a.W[ulong(gk) * ulong(a.ldw) + ulong(go)]) : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 4u; ++k8) {
            simdgroup_float8x8 fa[4], fb[4];
            for (uint i = 0; i < 4u; ++i) simdgroup_load(fa[i], As + (sr + i * 8u) * 32u + k8 * 8u, 32);
            for (uint j = 0; j < 4u; ++j) simdgroup_load(fb[j], Bs + k8 * 8u * 64u + so + j * 8u, 64);
            for (uint i = 0; i < 4u; ++i)
                for (uint j = 0; j < 4u; ++j) simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4u; ++i)
        for (uint j = 0; j < 4u; ++j) simdgroup_store(acc[i][j], smem + (sr + i * 8u) * 64u + so + j * 8u, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool bf = (a.flags & 2u) != 0u;
    for (uint idx = tid; idx < 4096u; idx += 128u) {
        const uint r = idx >> 6, o = idx & 63u;
        const uint gr = r0 + r, go = o0 + o;
        if (gr >= a.rows || go >= a.out) continue;
        const float v = a.alpha * smem[idx];
        if (bf) ((device ushort*) a.Y)[ulong(gr) * ulong(a.ldy) + ulong(go)] = vis_bfr(v);
        else ((device float*) a.Y)[ulong(gr) * ulong(a.ldy) + ulong(go)] = v;
    }
}

// (reference, off by default) the same GEMM one output a thread with a sequential FP32 FMA chain: another accumulation order to measure
// how much of a difference to another implementation is the BF16 roundings' sensitivity to summation order
kernel void vis_gemm_ref(constant VisGemmArgs& a [[buffer(0)]], uint2 tgp [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    const uint o = tgp.x * 32u + uint(tid), r = tgp.y;
    if (o >= a.out || r >= a.rows) return;
    const bool wn = (a.flags & 1u) != 0u;
    float acc = 0.0f;
    if ((a.flags & 4u) == 0u) {
        for (uint k = 0; k < a.in; ++k) {
            const float x = vis_bf(a.X[ulong(r) * ulong(a.ldx) + ulong(k)]);
            const float w = wn ? vis_bf(a.W[ulong(k) * ulong(a.ldw) + ulong(o)]) : vis_bf(a.W[ulong(o) * ulong(a.ldw) + ulong(k)]);
            acc = fma(x, w, acc);
        }
    } else {   // chunks of 64 summed apart, then added (a blocked order, as a library GEMM has)
        for (uint k0 = 0; k0 < a.in; k0 += 64u) {
            float part = 0.0f;
            const uint k1 = min(k0 + 64u, a.in);
            for (uint k = k0; k < k1; ++k) {
                const float x = vis_bf(a.X[ulong(r) * ulong(a.ldx) + ulong(k)]);
                const float w = wn ? vis_bf(a.W[ulong(k) * ulong(a.ldw) + ulong(o)]) : vis_bf(a.W[ulong(o) * ulong(a.ldw) + ulong(k)]);
                part = fma(x, w, part);
            }
            acc += part;
        }
    }
    const float v = a.alpha * acc;
    if ((a.flags & 2u) != 0u) ((device ushort*) a.Y)[ulong(r) * ulong(a.ldy) + ulong(o)] = vis_bfr(v);
    else ((device float*) a.Y)[ulong(r) * ulong(a.ldy) + ulong(o)] = v;
}
