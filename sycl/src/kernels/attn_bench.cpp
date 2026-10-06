// attn_bench: the prompt path's QSA attention fallback (qsa_decode_attn_batch, INT8 KV) against variants of it, at the
// prompt path's shapes: 32 queries a launch, 24 heads over 2 KV heads, 2,048 selected cells each from a long context.
// Variant times show where the kernel's time goes; outputs are checked against the engine's kernel.
//   attn_bench [reps=20] [context cells=131072] [variants...]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
namespace k = strata::kernels;
constexpr int HD = 256, G = 12, NKV = 2, PS = 4, NQ = 32, SEL = 2048;
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

struct Pools { const int8_t* kq; const int8_t* vq; const uint16_t* ks; const uint16_t* vs; const int32_t* pt; };

// VAR 0: the engine's kernel, copied (INT8 KV). 1: no scores (K not read, all scores 0). 2: no values (V not read).
// 3: scores from the chunk's K staged in local memory as floats, one work-item per (cell, head) dot, no shuffles.
template <int VAR, int CH>
void chunk_kernel(sycl::nd_item<1> it, const float* q, Pools p, const int32_t* ids, int n_chunks, float scale,
                  float* part_acc, float* part_m, float* part_l, float* sq, float* sp, long long* srow, float* sk) {
    constexpr int T = 256;
    auto sg = it.get_sub_group();
    const int SGS = (int) sg.get_local_range()[0], W = T / SGS;
    const int g = (int) it.get_group(0);
    const int chunk = g % n_chunks, kvh = (g / n_chunks) % NKV, qi = g / (n_chunks * NKV);
    const int t = (int) it.get_local_id(0), lane = (int) sg.get_local_id()[0], warp = (int) sg.get_group_id()[0];
    q += (size_t) qi * (NKV * G) * HD;
    ids += (size_t) qi * SEL;
    const int c0 = chunk * CH, n_here = std::min(CH, SEL - c0);
    const int slot = (qi * NKV + kvh) * n_chunks + chunk;
    for (int i = t; i < G * HD; i += T) sq[i] = q[(size_t) (kvh * G) * HD + i];
    for (int ci = t; ci < CH; ci += T) if (true) {
        const int t = ci;
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = p.pt[cell / PS];
            if (page >= 0) r = (page * NKV + kvh) * PS + (cell % PS);
        }
        srow[t] = r;
    }
    it.barrier(sycl::access::fence_space::local_space);
    if constexpr (VAR == 14) {
        // two work-items per cell (CH = 128): heads [0, 6) and [6, 12) - every work-item scores
        constexpr int HH = G / 2;
        const int c = t % CH, hb = (t / CH) * HH;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int h = 0; h < HH; ++h) sp[(hb + h) * CH + c] = -FLT_MAX;
        } else {
            float acc_s[HH];
#pragma unroll
            for (int h = 0; h < HH; ++h) acc_s[h] = 0.0f;
            const int8_t* kr = p.kq + srow[c] * HD;
            const uint16_t* ksr = p.ks + srow[c] * (HD / k::KV_Q8_GROUP);
            for (int d = 0; d < HD; d += 16) {
                const sycl::int4 raw = *reinterpret_cast<const sycl::int4*>(kr + d);
                const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
                const float sc = h2f(ksr[d / k::KV_Q8_GROUP]);
                float kf[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) kf[j] = (float) b[j] * sc;
#pragma unroll
                for (int h = 0; h < HH; ++h) {
                    const float* qr = sq + (hb + h) * HD + d;
#pragma unroll
                    for (int j = 0; j < 16; ++j) acc_s[h] = sycl::fma(kf[j], qr[j], acc_s[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < HH; ++h) sp[(hb + h) * CH + c] = acc_s[h] * scale;
        }
    } else if constexpr (VAR == 4 || VAR == 6 || VAR == 7 || VAR == 8 || VAR == 9 || VAR == 10 || VAR == 11 || VAR == 12 || VAR == 13 || VAR == 15 || VAR == 16 || VAR == 17 || VAR == 18 || VAR == 19 || VAR == 20) {
        for (int c = t; c < CH; c += T) {
            if (c >= n_here || srow[c] < 0) {
                for (int h = 0; h < G; ++h) sp[h * CH + c] = -FLT_MAX;
                continue;
            }
            float acc_s[G];
#pragma unroll
            for (int h = 0; h < G; ++h) acc_s[h] = 0.0f;
            const int8_t* kr = p.kq + srow[c] * HD;
            const uint16_t* ksr = p.ks + srow[c] * (HD / k::KV_Q8_GROUP);
            for (int d = 0; d < HD; d += 16) {
                const sycl::int4 raw = *reinterpret_cast<const sycl::int4*>(kr + d);
                const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
                const float sc = h2f(ksr[d / k::KV_Q8_GROUP]);
                float kf[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) kf[j] = (float) b[j] * sc;
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float* qr = sq + h * HD + d;
#pragma unroll
                    for (int j = 0; j < 16; ++j) acc_s[h] = sycl::fma(kf[j], qr[j], acc_s[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h) sp[h * CH + c] = acc_s[h] * scale;
        }
    } else if constexpr (VAR == 3) {
        // stage K as floats: 16 work-items per cell row (16 bytes each)
        for (int i = t; i < CH * 16; i += T) {
            const int c = i >> 4, seg = i & 15;
            float* dst = sk + c * (HD + 4) + seg * 16;
            if (c < n_here && srow[c] >= 0) {
                const sycl::int4 raw = *reinterpret_cast<const sycl::int4*>(p.kq + srow[c] * HD + seg * 16);
                const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
                const float sc = h2f(p.ks[srow[c] * (HD / k::KV_Q8_GROUP) + seg * 16 / k::KV_Q8_GROUP]);
#pragma unroll
                for (int j = 0; j < 16; ++j) dst[j] = (float) b[j] * sc;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int task = t; task < CH * G; task += T) {
            const int c = task % CH, h = task / CH;
            if (c >= n_here || srow[c] < 0) { sp[h * CH + c] = -FLT_MAX; continue; }
            const float* kr = sk + c * (HD + 4);
            const float* qr = sq + h * HD;
            float s = 0.0f;
#pragma unroll 8
            for (int d = 0; d < HD; d += 4) {
                const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(kr + d);
                const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(qr + d);
                s += a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
            }
            sp[h * CH + c] = s * scale;
        }
    } else {
        for (int c = warp; c < CH; c += W) {
            if (c >= n_here || srow[c] < 0) { if (lane < G) sp[lane * CH + c] = -FLT_MAX; continue; }
            if constexpr (VAR == 1) { if (lane < G) sp[lane * CH + c] = 0.0f; continue; }
            float k8[8];
            {
                const int8_t* codes = p.kq + srow[c] * HD + lane * 8;
                const float sc = h2f(p.ks[srow[c] * (HD / k::KV_Q8_GROUP) + lane * 8 / k::KV_Q8_GROUP]);
                const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2*>(codes);
                const int8_t* cc = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) k8[j] = (float) cc[j] * sc;
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 qa = *reinterpret_cast<const sycl::float4*>(&sq[h * HD + lane * 8]);
                const sycl::float4 qb = *reinterpret_cast<const sycl::float4*>(&sq[h * HD + lane * 8 + 4]);
                float s = k8[0] * qa.x() + k8[1] * qa.y() + k8[2] * qa.z() + k8[3] * qa.w() + k8[4] * qb.x() +
                          k8[5] * qb.y() + k8[6] * qb.z() + k8[7] * qb.w();
                if constexpr (VAR == 5) {
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) s += sycl::permute_group_by_xor(sg, s, o);
                } else {
                    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                }
                if (lane == 0) sp[h * CH + c] = s * scale;
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int h = warp; h < G; h += W) {
        float m = -FLT_MAX;
        for (int c = lane; c < CH; c += SGS) m = sycl::fmax(m, sp[h * CH + c]);
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float l = 0.0f;
        for (int c = lane; c < CH; c += SGS) {
            const float e = (c < n_here && srow[c] >= 0) ? sycl::native::exp(sp[h * CH + c] - m) : 0.0f;
            sp[h * CH + c] = e;
            l += e;
        }
        l = sycl::reduce_over_group(sg, l, sycl::plus<float>());
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    it.barrier(sycl::access::fence_space::local_space);
    if constexpr (VAR == 6 || VAR == 7) {
        // values: work-item t owns dims [4*(t%64), +4) for cells cg, cg+4, ... (cg = t/64); the four cell groups'
        // partial sums meet in local memory (sk), then each work-item sums 12 of the 3,072 (head, dim) outputs
        const int dg = t & 63, cg = t >> 6, d0 = dg * 4;
        float a4[G][4];
#pragma unroll
        for (int h = 0; h < G; ++h) for (int j = 0; j < 4; ++j) a4[h][j] = 0.0f;
        for (int c = cg; c < n_here; c += 4) {
            if (srow[c] < 0) continue;
            const float sc = h2f(p.vs[srow[c] * (HD / k::KV_Q8_GROUP) + d0 / k::KV_Q8_GROUP]);
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(p.vq + srow[c] * HD + d0);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = (float) (int8_t) (raw >> (8 * j)) * sc;
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float w = sp[h * CH + c];
#pragma unroll
                for (int j = 0; j < 4; ++j) a4[h][j] = sycl::fma(w, v[j], a4[h][j]);
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) for (int j = 0; j < 4; ++j) sk[(cg * G + h) * HD + d0 + j] = a4[h][j];
        it.barrier(sycl::access::fence_space::local_space);
        for (int i = t; i < G * HD; i += T) {
            const float v = sk[i] + sk[G * HD + i] + sk[2 * G * HD + i] + sk[3 * G * HD + i];
            part_acc[(size_t) slot * G * HD + i] = v;
        }
        return;
    }
    if constexpr (VAR == 8) {
        // values: work-item t owns dim t; the weights of 4 cells per head in one float4 local load (masked cells
        // have weight 0 and a row of -1: read as zero)
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.0f;
        for (int c = 0; c < n_here; c += 4) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const long long r = (c + j < n_here) ? srow[c + j] : -1;
                v[j] = r >= 0 ? (float) p.vq[r * HD + t] * h2f(p.vs[r * (HD / k::KV_Q8_GROUP) + t / k::KV_Q8_GROUP]) : 0.0f;
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 w = *reinterpret_cast<const sycl::float4*>(&sp[h * CH + c]);
                acc[h] = sycl::fma(w.x(), v[0], acc[h]);
                acc[h] = sycl::fma(w.y(), v[1], acc[h]);
                acc[h] = sycl::fma(w.z(), v[2], acc[h]);
                acc[h] = sycl::fma(w.w(), v[3], acc[h]);
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        return;
    }
    if constexpr (VAR == 16) {
        // values: per-cell V row pointers and scales in local memory once; 4 cells' loads issued together
        const int8_t** vrow = reinterpret_cast<const int8_t**>(sk);          // [CH] pointers (nullptr: masked)
        float* vsc = sk + 2 * CH;                                            // [CH][4] scales
        for (int c = t; c < CH; c += T) {
            const long long r = c < n_here ? srow[c] : -1;
            vrow[c] = r >= 0 ? p.vq + r * HD : nullptr;
#pragma unroll
            for (int g = 0; g < 4; ++g) vsc[c * 4 + g] = r >= 0 ? h2f(p.vs[r * (HD / k::KV_Q8_GROUP) + g]) : 0.0f;
        }
        it.barrier(sycl::access::fence_space::local_space);
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.0f;
        const int g4 = t / k::KV_Q8_GROUP;
        int c = 0;
        for (; c + 4 <= n_here; c += 4) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int8_t* rp = vrow[c + j];
                v[j] = rp ? (float) rp[t] * vsc[(c + j) * 4 + g4] : 0.0f;
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const sycl::float4 w = *reinterpret_cast<const sycl::float4*>(&sp[h * CH + c]);
                acc[h] = sycl::fma(w.x(), v[0], acc[h]);
                acc[h] = sycl::fma(w.y(), v[1], acc[h]);
                acc[h] = sycl::fma(w.z(), v[2], acc[h]);
                acc[h] = sycl::fma(w.w(), v[3], acc[h]);
            }
        }
        for (; c < n_here; ++c) {
            const int8_t* rp = vrow[c];
            if (!rp) continue;
            const float v = (float) rp[t] * vsc[c * 4 + g4];
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h * CH + c], v, acc[h]);
        }
#pragma unroll
        for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        return;
    }
    if constexpr (VAR == 19) {
        // values on XMX: P (12 heads + 4 zero rows, fp16) x V (CH cells x 256 dims, fp16, packed pairs of cells) ->
        // 16 x 256 fp32. Sub-group w (of 16, 16 lanes) owns dims [16w, 16w + 16). sk: [V packed][P16][C tile per sg]
        namespace jm = sycl::ext::oneapi::experimental::matrix;
        sycl::half* vp = reinterpret_cast<sycl::half*>(sk);                  // [CH/2][256][2]
        sycl::half* p16 = vp + (size_t) CH * HD;                              // [16][CH]
        float* ct = reinterpret_cast<float*>(p16 + 16 * CH);                  // [16 sg][16][16]
        for (int c = 0; c < CH; c += 2) {                                     // work-item t = dim t, two cells a pair
            float v0 = 0.0f, v1 = 0.0f;
            const long long r0 = c < n_here ? srow[c] : -1, r1 = c + 1 < n_here ? srow[c + 1] : -1;
            if (r0 >= 0) v0 = (float) p.vq[r0 * HD + t] * h2f(p.vs[r0 * (HD / k::KV_Q8_GROUP) + t / k::KV_Q8_GROUP]);
            if (r1 >= 0) v1 = (float) p.vq[r1 * HD + t] * h2f(p.vs[r1 * (HD / k::KV_Q8_GROUP) + t / k::KV_Q8_GROUP]);
            vp[(size_t) (c / 2) * (HD * 2) + t * 2] = sycl::half(v0);
            vp[(size_t) (c / 2) * (HD * 2) + t * 2 + 1] = sycl::half(v1);
        }
        for (int i = t; i < 16 * CH; i += T) {
            const int h = i / CH, c = i % CH;
            p16[i] = sycl::half(h < G ? sp[h * CH + c] : 0.0f);
        }
        it.barrier(sycl::access::fence_space::local_space);
        auto lptr = [](auto* x) { return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(x); };
        const int n0 = warp * 16;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 16, 16> C;
        jm::joint_matrix_fill(sg, C, 0.0f);
        for (int k0 = 0; k0 < CH; k0 += 16) {
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 16, 16, jm::layout::row_major> A;
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::ext_intel_packed> B;
            jm::joint_matrix_load(sg, A, lptr(p16 + k0), CH);
            jm::joint_matrix_load(sg, B, lptr(vp + (size_t) (k0 / 2) * (HD * 2) + n0 * 2), HD * 2);
            jm::joint_matrix_mad(sg, C, A, B, C);
        }
        float* myc = ct + warp * 256;
        jm::joint_matrix_store(sg, C, lptr(myc), 16, jm::layout::row_major);
        sycl::group_barrier(sg);
        for (int i = lane; i < G * 16; i += SGS) {
            const int h = i / 16, d = i % 16;
            part_acc[((size_t) slot * G + h) * HD + n0 + d] = myc[h * 16 + d];
        }
        return;
    }
    if constexpr (VAR == 17 || VAR == 18) {
        // values with sub-group block reads: the sub-group's 32 (17) dims of a V row in one read from one address;
        // 18: 4 dims per work-item (int32 block read), heads split in 4 groups of 3 so 12 sums stay in registers
        namespace syex = sycl::ext::oneapi::experimental;
        float* vsc = sk;                                                     // [CH][4] scales
        for (int i = t; i < CH * 4; i += T) {
            const int c = i >> 2, g4 = i & 3;
            const long long r = c < n_here ? srow[c] : -1;
            vsc[i] = r >= 0 ? h2f(p.vs[r * (HD / k::KV_Q8_GROUP) + g4]) : 0.0f;
        }
        it.barrier(sycl::access::fence_space::local_space);
        if constexpr (VAR == 17) {
            const int base = warp * SGS;                                     // this sub-group's dims [base, base + SGS)
            float acc[G];
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = 0.0f;
            const int g4 = t / k::KV_Q8_GROUP;
            for (int c = 0; c < n_here; ++c) {
                const long long r = srow[c];
                if (r < 0) continue;
                int8_t v8;
                syex::group_load(sg, p.vq + r * HD + base, v8);
                const float v = (float) v8 * vsc[c * 4 + g4];
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h * CH + c], v, acc[h]);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + base + lane] = acc[h];
        } else {
            // 8 sub-groups: sub-group w covers dims [(w % 2) * 128, +128) as 32 lanes x 4, heads [(w / 2) * 3, +3)
            const int dbase = (warp & 1) * 128, h0 = (warp >> 1) * 3, d0 = dbase + lane * 4, g4 = d0 / k::KV_Q8_GROUP;
            float a[3][4];
#pragma unroll
            for (int h = 0; h < 3; ++h) for (int j = 0; j < 4; ++j) a[h][j] = 0.0f;
            for (int c = 0; c < n_here; ++c) {
                const long long r = srow[c];
                if (r < 0) continue;
                int32_t w4;
                syex::group_load(sg, reinterpret_cast<const int32_t*>(p.vq + r * HD + dbase), w4);
                const float sc = vsc[c * 4 + g4];
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = (float) (int8_t) (w4 >> (8 * j)) * sc;
#pragma unroll
                for (int h = 0; h < 3; ++h) {
                    const float w = sp[(h0 + h) * CH + c];
#pragma unroll
                    for (int j = 0; j < 4; ++j) a[h][j] = sycl::fma(w, v[j], a[h][j]);
                }
            }
#pragma unroll
            for (int h = 0; h < 3; ++h)
                *reinterpret_cast<sycl::float4*>(&part_acc[((size_t) slot * G + h0 + h) * HD + d0]) =
                    sycl::float4(a[h][0], a[h][1], a[h][2], a[h][3]);
        }
        return;
    }
    if constexpr (VAR == 15) {
        // values with the V scales of the chunk's cells cached in local memory (sk: [CH][4] floats)
        for (int i = t; i < CH * 4; i += T) {
            const int c = i >> 2, g4 = i & 3;
            const long long r = c < n_here ? srow[c] : -1;
            sk[i] = r >= 0 ? h2f(p.vs[r * (HD / k::KV_Q8_GROUP) + g4]) : 0.0f;
        }
        it.barrier(sycl::access::fence_space::local_space);
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.0f;
        const int g4 = t / k::KV_Q8_GROUP;
        for (int c = 0; c < n_here; ++c) {
            const long long r = srow[c];
            if (r < 0) continue;
            const float v = (float) p.vq[r * HD + t] * sk[c * 4 + g4];
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h * CH + c], v, acc[h]);
        }
#pragma unroll
        for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        return;
    }
    if constexpr (VAR == 12) {
        // values (a): work-item t owns dims [4*(t%64), +4) of heads [3*(t/64), +3): one 4-byte V load per cell
        const int dg = t & 63, hg = t >> 6, d0 = dg * 4, h0 = hg * 3;
        float a[3][4];
#pragma unroll
        for (int h = 0; h < 3; ++h) for (int j = 0; j < 4; ++j) a[h][j] = 0.0f;
        for (int c = 0; c < n_here; ++c) {
            const long long r = srow[c];
            if (r < 0) continue;
            const float sc = h2f(p.vs[r * (HD / k::KV_Q8_GROUP) + d0 / k::KV_Q8_GROUP]);
            const uint32_t raw = *reinterpret_cast<const uint32_t*>(p.vq + r * HD + d0);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = (float) (int8_t) (raw >> (8 * j)) * sc;
#pragma unroll
            for (int h = 0; h < 3; ++h) {
                const float w = sp[(h0 + h) * CH + c];
#pragma unroll
                for (int j = 0; j < 4; ++j) a[h][j] = sycl::fma(w, v[j], a[h][j]);
            }
        }
#pragma unroll
        for (int h = 0; h < 3; ++h) {
            sycl::float4 o(a[h][0], a[h][1], a[h][2], a[h][3]);
            *reinterpret_cast<sycl::float4*>(&part_acc[((size_t) slot * G + h0 + h) * HD + d0]) = o;
        }
        return;
    }
    if constexpr (VAR == 13) {
        // values (b): the chunk's V rows staged in local memory (16-byte loads) as floats with their scales applied
        float* sv = sk;   // [CH][HD]
        for (int i = t; i < CH * 16; i += T) {
            const int c = i >> 4, seg = i & 15;
            float* dst = sv + c * HD + seg * 16;
            const long long r = c < n_here ? srow[c] : -1;
            if (r >= 0) {
                const sycl::int4 raw = *reinterpret_cast<const sycl::int4*>(p.vq + r * HD + seg * 16);
                const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
                const float sc = h2f(p.vs[r * (HD / k::KV_Q8_GROUP) + seg * 16 / k::KV_Q8_GROUP]);
#pragma unroll
                for (int j = 0; j < 16; ++j) dst[j] = (float) b[j] * sc;
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) dst[j] = 0.0f;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.0f;
        for (int c = 0; c < n_here; ++c) {
            const float v = sv[c * HD + t];
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h * CH + c], v, acc[h]);
        }
#pragma unroll
        for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        return;
    }
    if constexpr (VAR == 9 || VAR == 10 || VAR == 11) {
        // diagnosis: 9 = no V loads (v from the index), 10 = no weight loads (w constant), 11 = no values at all
        float acc[G];
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = 0.0f;
        if constexpr (VAR != 11)
            for (int c = 0; c < n_here; ++c) {
                if (srow[c] < 0) continue;
                float v;
                if constexpr (VAR == 9) v = (float) (c ^ t) * 1e-3f;
                else v = (float) p.vq[srow[c] * HD + t] * h2f(p.vs[srow[c] * (HD / k::KV_Q8_GROUP) + t / k::KV_Q8_GROUP]);
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float w = VAR == 10 ? 0.5f + 0.01f * h : sp[h * CH + c];
                    acc[h] = sycl::fma(w, v, acc[h]);
                }
            }
#pragma unroll
        for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
        return;
    }
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    if constexpr (VAR != 2) {
        for (int c = 0; c < n_here; ++c) {
            if (srow[c] < 0) continue;
            const float sc = h2f(p.vs[srow[c] * (HD / k::KV_Q8_GROUP) + t / k::KV_Q8_GROUP]);
            const float v = (float) p.vq[srow[c] * HD + t] * sc;
#pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = sycl::fma(sp[h * CH + c], v, acc[h]);
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int VAR, int CH>
void run_variant(sycl::queue& q, const float* dq, Pools p, const int32_t* dids, float* scratch, float* out) {
    const int n_chunks = (SEL + CH - 1) / CH;
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) NQ * NKV * n_chunks * G * HD;
    float* part_l = part_m + (size_t) NQ * NKV * n_chunks * G;
    const float scale = 1.0f / std::sqrt((float) HD);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sq(G * HD, h), sp(G * CH, h),
            sk(VAR == 3 ? CH * (HD + 4) : (VAR == 6 || VAR == 7) ? 4 * G * HD : VAR == 13 ? CH * HD : VAR == 15 ? CH * 4 : VAR == 16 ? CH * 6 : (VAR == 17 || VAR == 18) ? CH * 4 : VAR == 19 ? (CH * HD / 2 + 8 * CH + 16 * 256) : 1, h);
        sycl::local_accessor<long long, 1> srow(CH, h);
        auto body = [=](sycl::nd_item<1> it) {
            chunk_kernel<VAR, CH>(it, dq, p, dids, n_chunks, scale, part_acc, part_m, part_l,
                                  sq.get_multi_ptr<sycl::access::decorated::no>().get(),
                                  sp.get_multi_ptr<sycl::access::decorated::no>().get(),
                                  srow.get_multi_ptr<sycl::access::decorated::no>().get(),
                                  sk.get_multi_ptr<sycl::access::decorated::no>().get());
        };
        const sycl::nd_range<1> nr((size_t) NQ * NKV * n_chunks * 256, 256);
        if constexpr (VAR == 7 || VAR == 19 || VAR == 20) h.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] { body(it); });
        else h.parallel_for(nr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] { body(it); });
    });
    // merge: (query, head) x 256 dims
    q.parallel_for(sycl::nd_range<1>((size_t) NQ * NKV * G * HD, HD), [=](sycl::nd_item<1> it) {
        const int grp = (int) it.get_group(0), d = (int) it.get_local_id(0);
        const int qi = grp / (NKV * G), hh = grp % (NKV * G), kvh = hh / G, hl = hh % G;
        float M = -FLT_MAX;
        for (int c = 0; c < n_chunks; ++c) M = sycl::fmax(M, part_m[((qi * NKV + kvh) * n_chunks + c) * G + hl]);
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int slot = (qi * NKV + kvh) * n_chunks + c;
            const float m = part_m[slot * G + hl];
            if (m == -FLT_MAX) continue;
            const float w = sycl::native::exp(m - M);
            L = sycl::fma(part_l[slot * G + hl], w, L);
            acc = sycl::fma(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
        }
        out[((size_t) qi * NKV * G + hh) * HD + d] = L > 0.0f ? acc / L : 0.0f;
    });
}

// the gather floor: every selected cell's K and V rows (and scales) read and summed, no attention math.
// One work-group per (query, KV head, 128 cells); 16 work-items per cell row, 16 bytes each.
static void gather_only(sycl::queue& q, Pools p, const int32_t* ids, float* sink) {
    constexpr int CH = 128;
    const int n_chunks = SEL / CH;
    q.parallel_for(sycl::nd_range<1>((size_t) NQ * NKV * n_chunks * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
        const int chunk = g % n_chunks, kvh = (g / n_chunks) % NKV, qi = g / (n_chunks * NKV);
        int acc = 0;
        for (int i = t; i < CH * 16; i += 256) {
            const int c = i >> 4, seg = i & 15;
            const int cell = ids[(size_t) qi * SEL + chunk * CH + c];
            const long long r = ((long long) p.pt[cell / PS] * NKV + kvh) * PS + (cell % PS);
            const sycl::int4 a = *reinterpret_cast<const sycl::int4*>(p.kq + r * HD + seg * 16);
            const sycl::int4 b = *reinterpret_cast<const sycl::int4*>(p.vq + r * HD + seg * 16);
            acc += a.x() ^ b.y() ^ a.z() ^ b.w() ^ (seg < 4 ? (int) p.ks[r * 4 + seg] + (int) p.vs[r * 4 + seg] : 0);
        }
        if (acc == 0x7fffffff) sink[0] = 1.0f;
    });
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    const int64_t ctx = argc > 2 ? std::atoll(argv[2]) : 131072;
    std::vector<std::string> want;
    for (int i = 3; i < argc; ++i) want.push_back(argv[i]);
    sycl::queue* q = &dpct::get_in_order_queue();
    std::printf("device: %s, context %lld cells, %d queries x %d cells\n",
                q->get_device().get_info<sycl::info::device::name>().c_str(), (long long) ctx, NQ, SEL);
    std::mt19937 rng(5);
    const int64_t pages = ctx / PS, rows = pages * NKV * PS;
    std::vector<int8_t> hk((size_t) rows * HD), hv((size_t) rows * HD);
    std::vector<uint16_t> hks((size_t) rows * HD / k::KV_Q8_GROUP), hvs(hks.size());
    std::uniform_int_distribution<int> i8(-127, 127);
    for (auto& x : hk) x = (int8_t) i8(rng);
    for (auto& x : hv) x = (int8_t) i8(rng);
    std::uniform_real_distribution<float> us(0.002f, 0.02f);
    for (auto& x : hks) x = f2h(us(rng));
    for (auto& x : hvs) x = f2h(us(rng));
    std::vector<int32_t> hpt((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) hpt[(size_t) i] = (int32_t) i;
    // each query: 512 distinct 4-cell blocks, ascending; neighbouring queries share about a third (a prompt chunk)
    std::vector<int32_t> hids((size_t) NQ * SEL);
    std::vector<int32_t> base(512);
    std::uniform_int_distribution<int64_t> blk(0, pages - 1);
    for (auto& b : base) b = (int32_t) blk(rng);
    for (int qi = 0; qi < NQ; ++qi) {
        std::vector<int32_t> bl = base;
        for (auto& b : bl) if (rng() % 3 != 0) b = (int32_t) blk(rng);
        std::sort(bl.begin(), bl.end());
        for (size_t i = 1; i < bl.size(); ++i) if (bl[i] <= bl[i - 1]) bl[i] = (int32_t) std::min<int64_t>(bl[i - 1] + 1, pages - 1);
        for (int j = 0; j < 512; ++j) for (int c = 0; c < 4; ++c) hids[(size_t) qi * SEL + j * 4 + c] = bl[j] * 4 + c;
    }
    std::vector<float> hq((size_t) NQ * NKV * G * HD);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hq) x = nd(rng);
    std::vector<int32_t> hst((size_t) NQ * k::kStepCount);
    for (int qi = 0; qi < NQ; ++qi) {
        hst[qi * k::kStepCount + k::kStepPos] = (int32_t) ctx - 1;
        hst[qi * k::kStepCount + k::kStepNKv] = (int32_t) ctx;
        hst[qi * k::kStepCount + k::kStepNBid] = (int32_t) (ctx / 4);
        hst[qi * k::kStepCount + k::kStepWidth] = SEL;
    }
    auto up = [&](auto& v) { auto* d = sycl::malloc_device<std::remove_reference_t<decltype(v[0])>>(v.size(), *q); q->memcpy(d, v.data(), v.size() * sizeof(v[0])).wait(); return d; };
    int8_t* dk = up(hk); int8_t* dv = up(hv); uint16_t* dks = up(hks); uint16_t* dvs = up(hvs);
    int32_t* dpt = up(hpt); int32_t* dids = up(hids); float* dq = up(hq); int32_t* dst = up(hst);
    k::QsaShapes s = k::qsa_real_shapes();
    k::QsaAttnPools pools;
    pools.k_q = dk; pools.v_q = dv; pools.k_scale = dks; pools.v_scale = dvs; pools.page_table = dpt;
    const size_t scratch_n = (size_t) NQ * k::qsa_decode_attn_scratch_floats(SEL, s);
    float* scratch = sycl::malloc_device<float>(scratch_n * 2 + (size_t) NQ * NKV * 64 * G * HD * 2, *q);
    float* out_ref = sycl::malloc_device<float>(hq.size(), *q);
    float* out_v = sycl::malloc_device<float>(hq.size(), *q);
    auto time = [&](auto&& fn) {
        fn(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double t_ref = time([&] { k::qsa_decode_attn_batch(dq, pools, dids, dst, SEL, s, scratch, out_ref, NQ, q); });
    std::vector<float> r(hq.size()), v(hq.size());
    q->memcpy(r.data(), out_ref, r.size() * 4).wait();
    const double bytes = (double) NQ * NKV * SEL * (HD * 2 + 2 * 4 * 2);   // K and V codes + scales gathered
    std::printf("engine kernel: %.3f ms per 32 queries (%.0f GB/s of K/V gathered)\n", t_ref, bytes / t_ref / 1e6);
    Pools p{dk, dv, dks, dvs, dpt};
    auto check = [&](const char* name, double t, bool exact) {
        q->memcpy(v.data(), out_v, v.size() * 4).wait();
        double emax = 0, rmax = 0;
        for (size_t i = 0; i < r.size(); ++i) { emax = std::max(emax, (double) std::fabs(r[i] - v[i])); rmax = std::max(rmax, (double) std::fabs(r[i])); }
        std::printf("  %-34s %.3f ms (%.2fx)%s max err %.2e of %.2e\n", name, t, t_ref / t, exact ? "," : " [timing only],", emax, rmax);
    };
    auto on = [&](const char* n) { return want.empty() || std::find(want.begin(), want.end(), std::string(n)) != want.end(); };
    if (on("copy")) check("copy of the engine kernel", time([&] { run_variant<0, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("noscores")) check("no scores (K not read)", time([&] { run_variant<1, 64>(*q, dq, p, dids, scratch, out_v); }), false);
    if (on("novalues")) check("no values (V not read)", time([&] { run_variant<2, 64>(*q, dq, p, dids, scratch, out_v); }), false);
    if (on("slmk")) check("K staged in SLM, item per (cell,head)", time([&] { run_variant<3, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("slmk32")) check("same, 32-cell chunks", time([&] { run_variant<3, 32>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("copy128")) check("engine kernel, 128-cell chunks", time([&] { run_variant<0, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("butterfly")) check("engine order, native shuffles", time([&] { run_variant<5, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("percell")) check("thread per cell, 64-cell chunks", time([&] { run_variant<4, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("percell128")) check("thread per cell, 128-cell chunks", time([&] { run_variant<4, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vec128")) check("per cell + 4-dim values, 128 cells", time([&] { run_variant<6, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vec64")) check("per cell + 4-dim values, 64 cells", time([&] { run_variant<6, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vec128sg16")) check("same, 128 cells, sub-group 16", time([&] { run_variant<7, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("w4_128")) check("per cell + 4-cell weight loads, 128", time([&] { run_variant<8, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("w4_64")) check("per cell + 4-cell weight loads, 64", time([&] { run_variant<8, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("w4_256")) check("per cell + 4-cell weight loads, 256", time([&] { run_variant<8, 256>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("gather")) {
        const double t = time([&] { gather_only(*q, p, dids, out_v); });
        std::printf("  %-34s %.3f ms (%.0f GB/s): the floor for this gather\n", "K/V rows read, no math", t, bytes / t / 1e6);
    }
    if (on("d_nov")) check("per cell, values without V loads", time([&] { run_variant<9, 128>(*q, dq, p, dids, scratch, out_v); }), false);
    if (on("d_now")) check("per cell, values without weight loads", time([&] { run_variant<10, 128>(*q, dq, p, dids, scratch, out_v); }), false);
    if (on("d_novals")) check("per cell, no values phase", time([&] { run_variant<11, 128>(*q, dq, p, dids, scratch, out_v); }), false);
    if (on("v4h3_128")) check("per cell + 4-byte V, 3 heads/item, 128", time([&] { run_variant<12, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("v4h3_64")) check("per cell + 4-byte V, 3 heads/item, 64", time([&] { run_variant<12, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vslm_64")) check("per cell + V staged in SLM, 64", time([&] { run_variant<13, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vslm_128")) check("per cell + V staged in SLM, 128", time([&] { run_variant<13, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("split2")) check("two items per cell (6 heads each), 128", time([&] { run_variant<14, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vscale")) check("per cell + V scales in local memory", time([&] { run_variant<15, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("vtable")) check("per cell + V pointer table, 4 loads", time([&] { run_variant<16, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("blk1")) check("per cell + sub-group block reads (1 B)", time([&] { run_variant<17, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("blk4")) check("per cell + block reads (4 B), 3 heads/item", time([&] { run_variant<18, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("xmxold")) {   // the tree's XMX prompt attention (opt-in STRATA_PROMPT_ATTN_XMX=1), 32 queries a call
        setenv("STRATA_PROMPT_ATTN_XMX", "1", 1);
        bool ok = k::qsa_prompt_attn_batch(dq, pools, dids, dst, SEL, s, out_v, NQ, q);
        if (!ok) std::printf("  existing XMX kernel refused the call\n");
        else check("existing XMX prompt attention (v2)", time([&] { k::qsa_prompt_attn_batch(dq, pools, dids, dst, SEL, s, out_v, NQ, q); }), true);
    }
    if (on("xv64")) check("per cell + XMX values (fp16), 64", time([&] { run_variant<19, 64>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("xv128")) check("per cell + XMX values (fp16), 128", time([&] { run_variant<19, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("pc_sg16")) check("thread per cell at sub-group 16, 128", time([&] { run_variant<20, 128>(*q, dq, p, dids, scratch, out_v); }), true);
    if (on("percell256")) check("thread per cell, 256-cell chunks", time([&] { run_variant<4, 256>(*q, dq, p, dids, scratch, out_v); }), true);
    return 0;
}
