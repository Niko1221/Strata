// src/kernels/cuda/kv_q2.cu - see include/strata/kernels/kv_q2.hpp. 2-bit KV with Walsh-Hadamard rotation and
// PER-CHANNEL scales (KIVI's split, arXiv 2402.02750).
//
// Same page-table addressing, host-copy write and staging-pool write as kv_q4.cu, so KV streaming combines with it
// exactly as it does with q4_0. What is different is the unit: q4_0's block is 32 VALUES of one cell, so one cell
// is one self-contained 18-byte record; q2's block is 32 CELLS of one dimension, so one cell is two bits inside a
// 10-byte record shared by its whole group.
//
// ======================== The group, the page table and the race ========================
//
// A group is 32 consecutive cells OF THE POOL being written:
//     host copy / fully resident pool : the logical cell, group = pos / 32
//     a streamed pool's VRAM slots    : the resident cell, group = (slot_page * page_size + pos % page_size) / 32
// so the page table's indirection never splits a group, and a group is 32/page_size = 8 consecutive pages.
//
// THE RACE. Four cells' codes live in one byte. Two cells of the same group and the same dimension writing that
// byte is a read-modify-write race. The three append shapes handle it by granularity, not by locking:
//
//   * Decode (kv_append_q2_step, n_steps = 1) appends ONE cell per launch, on one stream, in increasing position
//     order. Two appends never run concurrently, so the byte has one writer at a time - that is why per-cell
//     decode is fine without any locking. It still writes the byte with an atomic, because the group's other
//     cells put their bits in the same byte in EARLIER launches and a plain store would erase them.
//     Call sites: layer.cpp's decode append, verify.cpp's per-token fallback, mtp.cpp's per-row drafter append.
//   * Prefill (kv_append_q2) and the batch appends (kv_append_q2_steps) are GROUP-GRANULAR: the grid is over
//     groups, one thread per dimension, and a group the call covers completely writes its 10-byte slot ONCE,
//     plain stores, no atomics. Call sites: prefill.cpp's prompt append and the drafter's prompt copy,
//     verify.cpp's accepted-window batch, mtp.cpp's drafter batch.
//   * A PARTIAL group - the tail of a prompt chunk (a chunk is not a multiple of 32 cells, so the range's first
//     and last group are partial), a batch's tail, the drafter ring's accepted window - takes the atomic path:
//     atomicAnd(~mask) then atomicOr(bits), which touches only this cell's two bits. Two writers of the same byte
//     with disjoint bit ranges cannot clobber each other, so a batch spanning a group boundary is safe.
//
// THE SCALE. The group's fp16 scale is the RMS of the 32 values of that dimension, and only a call that has all
// 32 cells can compute it. The rule in one sentence: a complete group's scale is recomputed from its 32 source
// values; a partial group keeps the scale the group already carries; and a group with no scale yet takes the
// full-group RMS with the previous group's mean square standing in for the cells not yet written, falling back
// to the RMS of the cells the call has when no previous scale is reachable race-free.
//
// The "partial keeps the stored scale" half is what makes a group straddling a prompt-chunk boundary come out
// right: prompt chunks are not multiples of 32 cells, so the chunk that starts mid-group would otherwise give
// that group a second scale and leave the first half's codes quantized against the first one. The first call to
// touch the group sets the scale; the calls that fill the rest keep it. Deterministic, because appends on a
// stream are ordered.
//
// THE SCALE'S ADDITION. The RMS is summed as ss += x*x over the group's cells in cell order - not a shuffle
// butterfly. The host reference in src/kernels/kv_q2_parity.cpp (quant_kivi) sums the same way in the same order,
// which is what makes the gate's bitwise check possible: an ulp in the scale flips a code at a threshold. The
// scale is stored as fp16 and read back with f32_from_f16 before multiplying, exactly as the reader does.

#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q2: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// head_dim 256 is the Hadamard transform size. page_size must line up with a 32-cell group: it divides 32 (the
// engine uses 4, so eight pages per group) or it is a multiple of it. Anything in between splits a group across
// pages, and the page table's indirection would then move half a group.
void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform size)\n", what);
        std::exit(1);
    }
}
void need_page(const QsaShapes& s, const char* what) {
    const long long p = s.page_size;
    if (p <= 0 || (p % Q2_0 != 0 && Q2_0 % p != 0)) {
        std::fprintf(stderr, "%s: page_size %lld does not line up with a 32-cell group\n", what, (long long) p);
        std::exit(1);
    }
}

/// The whole slot of a COMPLETE group: one thread owns it, so plain stores and no atomics. `pack` is the eight
/// code bytes in a 64-bit register (byte j at bits [8j, +8)), which keeps the packing register-only.
__device__ __forceinline__ void q2c_put_whole(uint8_t* pool, long long group, int h, int d, int kv_heads,
                                              int head_dim, uint16_t sbits, unsigned long long pack) {
    block_q2c_0* blk = q2c_slot(pool, group, h, d, kv_heads, head_dim);
    blk->d = sbits;
#pragma unroll
    for (int j = 0; j < Q2_0 / 4; ++j) blk->qs[j] = (uint8_t) ((pack >> (8 * j)) & 0xFFu);
}

/// One cell of a PARTIAL group: this cell's two bits of the group's byte for this dimension, and nothing else in
/// it. atomicAnd clears exactly those two bits, atomicOr sets them, so a concurrent writer of another cell of the
/// same byte (a batch that spans a group boundary) cannot lose its code. `write_scale` is set only when the group
/// had no scale at all, so a group's scale is written once and every cell of it reads back against one scale.
__device__ __forceinline__ void q2c_put_cell(uint8_t* pool, long long group, int idx, int h, int d, int kv_heads,
                                             int head_dim, int code, bool write_scale, uint16_t sbits) {
    block_q2c_0* blk = q2c_slot(pool, group, h, d, kv_heads, head_dim);
    const int shift = 2 * (idx & 3);
    // CUDA has no 8-bit atomics, and a 10-byte slot's code bytes are not 4-byte aligned (qs starts at offset 2,
    // and slots are 10 bytes apart), so the byte's containing WORD is found by masking the address.  The mask
    // touches only this byte's two bits, so the neighbouring bytes in the same word - another dimension's scale
    // low byte, a neighbouring code byte - keep their bits through the atomic.
    uint8_t* p = &blk->qs[idx >> 2];
    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    unsigned* word = reinterpret_cast<unsigned*>(addr & ~uintptr_t(3));
    const unsigned at = 8u * (unsigned) (addr & 3u) + (unsigned) shift;
    const unsigned mask = 0x3u << at;
    atomicAnd(word, ~mask);
    atomicOr(word, (unsigned) code << at);
    if (write_scale) blk->d = sbits;
}

// GROUP-GRANULAR append. One block = one (group, KV head, plane), head_dim threads = one thread per dimension.
// A thread walks the call's cells, keeps the ones that fall in its group, and writes that dimension's slot of the
// group: the scale and the eight code bytes, once, for a complete group.
//
// `pos0 >= 0` is the prompt path (T consecutive cells from pos0, source row = cell - pos0). `pos0 < 0` is the
// batch path: the positions come from the step rows on the DEVICE (graph-capturable: no host arithmetic, and the
// grid is sized from n_steps, which is a launch-time constant), and the source row is the step index.
__global__ void kv_append_q2_group_kernel(uint8_t* __restrict__ k_q2, uint8_t* __restrict__ v_q2,
                                          const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                          int step_stride, int n_steps, int64_t pos0, const float* __restrict__ kcur,
                                          const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                          int planes, KvHostPools host, KvHostPools stage, bool v3level,
                                          int use_prev) {
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const int h = blockIdx.x, d = threadIdx.x;
    const long long g0 = pos0 >= 0 ? pos0 / Q2_0 : (long long) __ldg(step + kStepPos) / Q2_0;
    const long long g = g0 + blockIdx.y;
    const int row_stride = kv_heads * head_dim;
    const float* src = is_v ? vcur : kcur;
    // The pool whose group index is the LOGICAL group g: the host copy when there is one (it is identity-layout
    // and every append writes it), else the staging pool, else the VRAM pool - which for a fully resident state
    // is identity-layout too, so its group g is the logical one.
    const uint8_t* sp = host.k_q2 != nullptr ? (is_v ? host.v_q2 : host.k_q2)
                  : (stage.k_q2 != nullptr ? (is_v ? stage.v_q2 : stage.k_q2)
                                           : (is_v ? v_q2 : k_q2));

    // PASS 1 - how many of the group's cells this call has, and their squared sum, in cell order. The order is
    // the host reference's (quant_kivi in the parity gate) sums in: an ulp in the scale flips a code at a
    // threshold, so the two have to add the same 32 numbers in the same order.
    //
    // __fmul_rn / __fadd_rn, not `ss += v * v`: nvcc contracts that to an FMA (fmad is on by default), and
    // fmaf(v, v, ss) is not ss + v*v. The host reference multiplies and adds separately, and MSVC never
    // contracts, so the explicit pair is what makes the two agree bit for bit.
    int n = 0;
    float ss = 0.0f;
    for (int i = 0; i < n_steps; ++i) {
        const long long p = pos0 >= 0 ? pos0 + i : (long long) __ldg(step + (long long) i * step_stride + kStepPos);
        if (p / Q2_0 != g) continue;
        const float v = src[(long long) i * row_stride + h * head_dim + d];
        ss = __fadd_rn(ss, __fmul_rn(v, v));
        ++n;
    }
    if (n <= 0) return;

    // The scale. A complete group is the RMS of its 32 values. A partial group keeps the scale the group already
    // carries (see the header comment): the prompt chunk that starts mid-group must not hand the group a second
    // scale behind the first half's codes. A group with no scale yet takes the FULL-GROUP RMS with the previous
    // group's mean square standing in for the cells not yet written:
    //     s = sqrtf((ss + (32 - n) * prev * prev) / 32)
    // - a 12-cell RMS is a noisy estimate of a channel, and this keeps the scale on the full-group definition
    // the reader divides by. With no usable previous scale it falls back to the RMS of the cells the call has.
    // `prev` is taken race-free in every case:
    //   * single-cell append (use_prev): appends are ordered on one stream, so the stored scale of group g-1
    //     is stable - borrow it exactly.
    //   * batch, group g-1 complete in THIS launch: its block writes f16(sqrt(ss_prev/32)) in this launch, so
    //     the pool must not be read - compute exactly that value from this call's cells instead.
    //   * batch, group g-1 not in this launch: its stored scale is stable (no block of this launch touches it).
    //   * batch, group g-1 partial in this launch: a partial group with a stored scale keeps it and never
    //     rewrites it, so a stored scale > 0 is stable (the speculative-verify case: a window spanning the
    //     boundary). A partial g-1 with NO stored scale would be writing a fresh one in this launch - not
    //     reachable for in-order appends (its earlier cells would already carry a scale); if it were, the
    //     read sees 0 (the fallback) or that fresh scale, either a valid estimate, never a torn value.
    //   Single-cell appends use the same formula with n = 1 (s is then ~97% prev): one rule for both paths.
    float s = sqrtf(ss * (1.0f / (float) n));
    bool fresh = false;                              // does this call write the group's scale?
    if (n < Q2_0) {
        const float stored = q2c_scale(sp, g, h, d, kv_heads, head_dim);
        if (stored > 0.0f) s = stored;
        else {
            fresh = true;
            float prev = 0.0f;
            if (g > 0) {
                if (use_prev) prev = q2c_scale(sp, g - 1, h, d, kv_heads, head_dim);
                else {
                    int n_prev = 0;
                    float ss_prev = 0.0f;
                    for (int i = 0; i < n_steps; ++i) {
                        const long long p = pos0 >= 0
                                                ? pos0 + i
                                                : (long long) __ldg(step + (long long) i * step_stride + kStepPos);
                        if (p / Q2_0 != g - 1) continue;
                        const float v = src[(long long) i * row_stride + h * head_dim + d];
                        ss_prev = __fadd_rn(ss_prev, __fmul_rn(v, v));
                        ++n_prev;
                    }
                    prev = n_prev == Q2_0
                               ? f32_from_f16(f16_from_f32(sqrtf(ss_prev * (1.0f / (float) Q2_0))))
                               : q2c_scale(sp, g - 1, h, d, kv_heads, head_dim);
                }
            }
            if (prev > 0.0f)
                s = sqrtf(__fmul_rn(__fadd_rn(ss, __fmul_rn((float) (Q2_0 - n), __fmul_rn(prev, prev))),
                                    1.0f / (float) Q2_0));
        }
    }
    const uint16_t sbits = f16_from_f32(s);
    const bool whole = (n == Q2_0);
    const bool lm3 = is_v && v3level;

    // PASS 2 - the codes, packed. `pack` is indexed by the LOGICAL in-group index (the host copy and the staging
    // pool are identity layouts), `dpack` by the RESIDENT one (the VRAM slots).
    unsigned long long pack = 0ull, dpack = 0ull;
    long long dgrp = -1;
    bool same_group = true;
    int resident = 0;
    for (int i = 0; i < n_steps; ++i) {
        const long long p = pos0 >= 0 ? pos0 + i : (long long) __ldg(step + (long long) i * step_stride + kStepPos);
        if (p / Q2_0 != g) continue;
        const float v = src[(long long) i * row_stride + h * head_dim + d];
        const unsigned long long code = (unsigned long long) (lm3 ? q2_code_lm3(v, s) : q2_code_lm4(v, s));
        const int li = (int) (p % Q2_0);
        pack |= code << (8 * (li >> 2) + 2 * (li & 3));
        const long long page = (long long) __ldg(table + p / page_size);
        if (page < 0) continue;
        ++resident;
        const long long pc = page * page_size + (p % page_size);
        const int di = (int) (pc % Q2_0);
        if (dgrp < 0) dgrp = pc / Q2_0;
        else if (pc / Q2_0 != dgrp) same_group = false;
        dpack |= code << (8 * (di >> 2) + 2 * (di & 3));
    }

    // A COMPLETE group whose resident cells form one group (the streamed case: kv_stream.cu grants residency a
    // group at a time) writes each dimension's 10-byte slot ONCE, plain stores, no atomics. Anything partial - a
    // prompt chunk's first and last group, a batch's tail, the drafter ring - goes cell by cell with the
    // bit-masked atomics, which is what makes a batch spanning a group boundary safe.
    if (resident > 0) {
        uint8_t* pool = is_v ? v_q2 : k_q2;
        if (whole && same_group) q2c_put_whole(pool, dgrp, h, d, kv_heads, head_dim, sbits, dpack);
        else
            for (int i = 0; i < n_steps; ++i) {
                const long long p = pos0 >= 0 ? pos0 + i
                                              : (long long) __ldg(step + (long long) i * step_stride + kStepPos);
                if (p / Q2_0 != g) continue;
                const long long page = (long long) __ldg(table + p / page_size);
                if (page < 0) continue;
                const long long pc = page * page_size + (p % page_size);
                const float v = src[(long long) i * row_stride + h * head_dim + d];
                const int code = lm3 ? q2_code_lm3(v, s) : q2_code_lm4(v, s);
                q2c_put_cell(pool, pc / Q2_0, (int) (pc % Q2_0), h, d, kv_heads, head_dim, code, fresh, sbits);
            }
    }
    if (whole) {
        if (host.k_q2 != nullptr)
            q2c_put_whole(is_v ? host.v_q2 : host.k_q2, g, h, d, kv_heads, head_dim, sbits, pack);
        if (stage.k_q2 != nullptr)
            q2c_put_whole(is_v ? stage.v_q2 : stage.k_q2, g, h, d, kv_heads, head_dim, sbits, pack);
    } else {
        for (int i = 0; i < n_steps; ++i) {
            const long long p = pos0 >= 0 ? pos0 + i : (long long) __ldg(step + (long long) i * step_stride + kStepPos);
            if (p / Q2_0 != g) continue;
            const int li = (int) (p % Q2_0);
            const float v = src[(long long) i * row_stride + h * head_dim + d];
            const int code = lm3 ? q2_code_lm3(v, s) : q2_code_lm4(v, s);
            if (host.k_q2 != nullptr)
                q2c_put_cell(is_v ? host.v_q2 : host.k_q2, g, li, h, d, kv_heads, head_dim, code, fresh, sbits);
            if (stage.k_q2 != nullptr)
                q2c_put_cell(is_v ? stage.v_q2 : stage.k_q2, g, li, h, d, kv_heads, head_dim, code, fresh, sbits);
        }
    }
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths). One thread per
// (cell, kv head, dimension): the thread's slot is (group, head, dim), and its cell's code is the two bits at
// 2*(idx%4) of qs[idx/4]. Consecutive threads walk consecutive dimensions, i.e. consecutive 10-byte slots.
__global__ void kv_gather_q2_kernel(const uint8_t* __restrict__ k_q2, const uint8_t* __restrict__ v_q2,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch,
                                    bool v3level) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const long long total = n_ids * kv_heads * head_dim;
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int d = (int) (i % head_dim);
    const int h = (int) ((i / head_dim) % kv_heads);
    const long long id = i / ((long long) head_dim * kv_heads);
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long pc = page * page_size + (cell % page_size);
    const long long g = pc / Q2_0;
    const int idx = (int) (pc % Q2_0);
    const block_q2c_0* kb = q2c_slot(k_q2, g, h, d, kv_heads, head_dim);
    const block_q2c_0* vb = q2c_slot(v_q2, g, h, d, kv_heads, head_dim);
    const int j = idx >> 2, shift = 2 * (idx & 3);
    const int kc = (kb->qs[j] >> shift) & 3;
    const int vc = (vb->qs[j] >> shift) & 3;
    const long long dst = (id * kv_heads + h) * head_dim + d;
    k_scratch[dst] = f16_from_f32(f32_from_f16(kb->d) * q2_lm4_level(kc));
    v_scratch[dst] = f16_from_f32(f32_from_f16(vb->d) * (v3level ? q2_lm3_level(vc) : q2_lm4_level(vc)));
}

}  // namespace

// The batch append: GROUP-GRANULAR over the groups the steps' positions touch. The grid over-provisions (n_steps
// consecutive cells touch at most n_steps/32 + 1 groups) and a block whose group holds none of them returns, so
// the launch size never depends on the positions - which are device state, and the graphs bake launch sizes in.
void kv_append_q2_steps(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host, bool v3level) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q2");
    need_page(s, "kv_append_q2");
    const int planes = (v_q2 == nullptr || (v_q2 == k_q2 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (n_steps + 1), (unsigned) planes);
    kv_append_q2_group_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
        k_q2, v_q2, page_table, step, step_stride, n_steps, -1, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, planes, host ? *host : KvHostPools{}, KvHostPools{}, v3level, n_steps == 1 ? 1 : 0);
    check("kv_append_q2 launch");
}

// The decode append: one cell, which is a partial group - the atomic path, and the group keeps its scale.
void kv_append_q2_step(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host, bool v3level) {
    kv_append_q2_steps(k_q2, v_q2, page_table, step, 0, 1, kcur, vcur, s, stream, host, v3level);
}

// The prompt path: T consecutive cells from pos0, group-granular. The range's first and last group are partial
// when the chunk is not a multiple of 32 cells (it usually is not) and take the atomic path.
void kv_append_q2(uint8_t* k_q2, uint8_t* v_q2, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage,
                  bool v3level) {
    if (T <= 0) return;
    need_256(s, "kv_append_q2");
    need_page(s, "kv_append_q2");
    const int planes = (v_q2 == nullptr || (v_q2 == k_q2 && V == K)) ? 1 : 2;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) ((T + Q2_0 - 1) / Q2_0 + 1), (unsigned) planes);
    kv_append_q2_group_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
        k_q2, v_q2, page_table, nullptr, 0, (int) T, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, planes, h, st, v3level, 0);
    check("kv_append_q2 batch launch");
}

void kv_gather_q2_step(const uint8_t* k_q2, const uint8_t* v_q2, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, bool v3level) {
    if (max_ids <= 0) return;
    const int64_t total = max_ids * s.n_head_kv * s.head_dim;
    const int rows = 256;
    const unsigned num_blocks = (unsigned) ((total + rows - 1) / rows);
    kv_gather_q2_kernel<<<dim3(num_blocks), rows, 0, (cudaStream_t) stream>>>(
        k_q2, v_q2, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch, v3level);
    check("kv_gather_q2 launch");
}

}  // namespace strata::kernels
