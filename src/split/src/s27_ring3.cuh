// The 27B's sums across the cards (included by s27_impl.hpp): the ring of kernels.cu's k_ring, with words of three fp16
// values and a 16-bit tag instead of two and a 32-bit one. The box's GPU 0 is on the other socket from 1 and 2, and a
// sum's time grows with its bytes (~0.6 ms a window row for 128 sums at two values a word); a third fewer words.
// A 16-bit tag wraps every 65536 exchanges (256 windows): a word left in a slot from exactly that many exchanges back
// would pass for a new one. Slots are rewritten every exchange up to the window's rows, so only rows a long-unused
// wider window wrote are exposed: Engine::tags_room (s27_engine.cu) zeroes the buffers each 128 tags, so no word left is 256 tags old.
#pragma once

#include <cuda_fp16.h>

#include "kernels.hpp"
#include "model.hpp"

namespace fnx {

// a card's buffers: words in from the card before, finished words gathered; the next card's
struct R3 {
    int card = 0;
    unsigned long long *in = nullptr, *ag = nullptr, *next_in = nullptr, *next_ag = nullptr;
    unsigned* err = nullptr;
};

constexpr int R3_NT = (N + 2) / 3;                 // triples a row (the last padded)
constexpr int R3_CH = R3_NT / NCARD + 1;           // a chunk's triples, at most
constexpr int R3_STEPS = NCARD - 1;
constexpr int R3_BLOCKS = 40;
inline size_t r3_in_words() { return (size_t) 2 * R3_STEPS * MAXT * R3_CH; }
inline size_t r3_ag_words() { return (size_t) 2 * MAXT * R3_NT; }

__device__ __forceinline__ int r3_lo(int k) { return k * R3_NT / NCARD; }
__device__ __forceinline__ float r3_x(const float* x, int t, int i) { return i < N ? x[(size_t) t * N + i] : 0.f; }
__device__ __forceinline__ void r3_put(unsigned long long* p, float a, float b, float c, unsigned tag, unsigned* err) {
    if (!(fabsf(a) <= RING_MAX && fabsf(b) <= RING_MAX && fabsf(c) <= RING_MAX)) atomicOr(err, 2u);
    const unsigned long long w = (unsigned long long) __half_as_ushort(__float2half_rn(a)) |
                                 (unsigned long long) __half_as_ushort(__float2half_rn(b)) << 16 |
                                 (unsigned long long) __half_as_ushort(__float2half_rn(c)) << 32 | (unsigned long long) (tag & 0xffffu) << 48;
    *(volatile unsigned long long*) p = w;
}
__device__ __forceinline__ float3 r3_take(const unsigned long long* p, unsigned tag, unsigned* err) {
    const long long t0 = clock64();
    unsigned long long w;
    for (;;) {
        w = *(const volatile unsigned long long*) p;
        if ((unsigned) (w >> 48) == (tag & 0xffffu)) break;
        if (clock64() - t0 > 4000000000LL) { atomicOr(err, 1u); break; }
    }
    return make_float3(__half2float(__ushort_as_half((unsigned short) w)), __half2float(__ushort_as_half((unsigned short) (w >> 16))),
                       __half2float(__ushort_as_half((unsigned short) (w >> 32))));
}

static __global__ void __launch_bounds__(256) k_ring3(R3 r, float* x, int T, const unsigned* tagp, int call) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const int n = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x, g = r.card, set = tag & 1;
    const size_t in_set = (size_t) R3_STEPS * MAXT * R3_CH, ag_set = (size_t) MAXT * R3_NT;
    unsigned long long* in = r.in + set * in_set;
    unsigned long long* nin = r.next_in + set * in_set;
    unsigned long long* ag = r.ag + set * ag_set;
    unsigned long long* nag = r.next_ag + set * ag_set;
    auto slot = [&](unsigned long long* b, int s, int t, int i) { return b + ((size_t) s * MAXT + t) * R3_CH + i; };
    {   // step 0: my part of chunk g to g+1
        const int lo = r3_lo(g), len = r3_lo(g + 1) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, j = 3 * (lo + e % len);
            r3_put(slot(nin, 0, t, e % len), r3_x(x, t, j), r3_x(x, t, j + 1), r3_x(x, t, j + 2), tag, r.err);
        }
    }
#pragma unroll
    for (int s = 1; s < R3_STEPS; ++s) {   // step s: chunk g-s arrived with s parts; add mine, on to g+1
        const int c = (g + NCARD - s) % NCARD, lo = r3_lo(c), len = r3_lo(c + 1) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len, j = 3 * (lo + i);
            const float3 a = r3_take(slot(in, s - 1, t, i), tag, r.err);
            r3_put(slot(nin, s, t, i), a.x + r3_x(x, t, j), a.y + r3_x(x, t, j + 1), a.z + r3_x(x, t, j + 2), tag, r.err);
        }
    }
    {   // step NCARD - 1: chunk g+1 with the other cards' parts; add mine: finished. Keep it, send it on
        const int c = (g + 1) % NCARD, lo = r3_lo(c), len = r3_lo(c + 1) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len, j = 3 * (lo + i);
            const float3 a = r3_take(slot(in, R3_STEPS - 1, t, i), tag, r.err);
            const float v0 = a.x + r3_x(x, t, j), v1 = a.y + r3_x(x, t, j + 1), v2 = a.z + r3_x(x, t, j + 2);
            r3_put(ag + (size_t) t * R3_NT + lo + i, v0, v1, v2, tag, r.err);
            r3_put(nag + (size_t) t * R3_NT + lo + i, v0, v1, v2, tag, r.err);
        }
    }
#pragma unroll
    for (int f = 1; f < R3_STEPS; ++f) {   // chunk g-f+1 arrived finished from g-1: forward it
        const int c = (g + NCARD - f + 1) % NCARD, lo = r3_lo(c), len = r3_lo(c + 1) - lo;
        for (int e = tid; e < T * len; e += n) {
            const int t = e / len, i = e % len;
            const float3 v = r3_take(ag + (size_t) t * R3_NT + lo + i, tag, r.err);
            r3_put(nag + (size_t) t * R3_NT + lo + i, v.x, v.y, v.z, tag, r.err);
        }
    }
    for (int e = tid; e < T * R3_NT; e += n) {   // everything back into x
        const int t = e / R3_NT, j = 3 * (e % R3_NT);
        const float3 v = r3_take(ag + e, tag, r.err);
        x[(size_t) t * N + j] = v.x;
        if (j + 1 < N) x[(size_t) t * N + j + 1] = v.y;
        if (j + 2 < N) x[(size_t) t * N + j + 2] = v.z;
    }
}

}  // namespace fnx
