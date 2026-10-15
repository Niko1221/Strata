// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block

/// Blocks per residency unit - see kv_stream_unit_blocks() in the header, which states the rule. Short name
/// here so the kernels below read as they did.
int blocks_per_unit(const QsaShapes& s, int fmt) { return kv_stream_unit_blocks(s, fmt); }

// The per-UNIT byte runs of the (up to four) pool arrays: unit u of array i is bytes [u * len, (u + 1) * len),
// where a unit is `unit` blocks (see blocks_per_unit).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
    int unit;   // blocks per unit
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    r.unit = blocks_per_unit(s, fmt);
    if (fmt == kKvHybrid) {   // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.k_scale; r.dst[1] = (uint8_t*) slots.k_scale; r.len[1] = scales;
        r.src[2] = (const uint8_t*) host.v_q4;    r.dst[2] = (uint8_t*) slots.v_q4;    r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvQ2) {   // q2 (kv_q2.hpp): the same 80 B per cell and head as the per-token layout, and
                                 // the same two contiguous runs - per GROUP of 32/page_size pages, which is the
                                 // unit whose scale is shared (see blocks_per_unit)
        const int bytes = rows * (int) kv_q2_bytes_per_head((int) s.head_dim) * r.unit;
        r.src[0] = (const uint8_t*) host.k_q2; r.dst[0] = (uint8_t*) slots.k_q2; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q2; r.dst[1] = (uint8_t*) slots.v_q2; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvQ2K) {  // q2k (`--kv q2k`): K's q2 per-channel run and V's q4_0 run, in ONE unit of
                                 // blocks_per_unit() blocks. The unit is the q2 group, because K's scale is shared
                                 // by it (see blocks_per_unit); V's run is 8 self-contained q4_0 blocks per head,
                                 // so it is contiguous at the same granularity and needs nothing of its own.
        const int kb = rows * (int) kv_q2_bytes_per_head((int) s.head_dim) * r.unit;
        const int vb = rows * (int) kv_q4_bytes_per_head((int) s.head_dim) * r.unit;
        r.src[0] = (const uint8_t*) host.k_q2; r.dst[0] = (uint8_t*) slots.k_q2; r.len[0] = kb;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = vb;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, const int32_t* __restrict__ ids,
                                                     const int32_t* __restrict__ steps, int n_q, int cap, int page_size,
                                                     int gpb) {
    __shared__ int s_nmiss, s_lookups, s_cut;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] + 1;
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; }
    __syncthreads();
    // Residency is granted per UNIT: a block for every format whose block is self-contained, a GROUP of
    // 32/page_size blocks for q2 (its per-channel scale is shared by the whole group, so the group is resident
    // together or not at all). gpb == 1 - every format but q2 - leaves each loop below exactly as it was.
    const int ng = (int) (m.n_slots / gpb);
    // 1. hits take this epoch and their reference bit; a missing unit is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = threadIdx.x; i < width; i += RT) {
            const int u = qi[i] / page_size / gpb;
            if (i > 0 && qi[i - 1] / page_size / gpb == u) continue;   // ids are ascending: one lookup per unit
            ++lookups;
            const int b = u * gpb;                                      // the unit's first block decides residency
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl / gpb] = epoch;
                m.slot_ref[sl / gpb] = 1;
            } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
                m.miss_block[atomicAdd(&s_nmiss, 1)] = u;
            }
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // (#783, stuchapin909) most threads see none: skip the shared atomic
    __syncthreads();
    // 2. one victim unit per miss: a clock sweep from the hand. A unit this call uses (stamp == epoch) is never
    //    taken; a referenced one loses its bit as the hand passes it and is taken on the next pass. The sweep
    //    walks UNIT indices, so a q2 group always lands on one aligned run of gpb slots.
    const int need = s_nmiss, n = ng;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; the copy kernel fills the slots. A unit owns gpb consecutive slots starting at
    //    sg * gpb, so every block of the unit resolves, and the unit evicted from those slots loses all of them.
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int u = m.miss_block[k];
        if (k >= placed) {                                       // overflow: never happens with a legal n_slots
            for (int j = 0; j < gpb; ++j)
                if ((long long) u * gpb + j < m.n_blocks) m.page_table[u * gpb + j] = -1;
            continue;
        }
        const int sg = m.miss_slot[k];
        const int old = m.slot_block[sg];
        if (old >= 0)
            for (int j = 0; j < gpb; ++j)
                if ((long long) old * gpb + j < m.n_blocks) m.page_table[old * gpb + j] = -1;
        m.slot_block[sg] = u;
        m.slot_stamp[sg] = epoch;
        m.slot_ref[sg] = 1;
        for (int j = 0; j < gpb; ++j)
            if ((long long) u * gpb + j < m.n_blocks) m.page_table[u * gpb + j] = sg * gpb + j;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

// One block per missed UNIT (grid-stride): copy its runs from the host copy into its slots, 16 B per thread.
// miss_block / miss_slot hold unit indices and r.len is per unit, so this is unchanged by the unit being a
// group (q2) rather than a block (every other format).
__global__ void copy_kernel(KvStreamMap m, Runs r) {
    const int need = m.ctl[2];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + sl * r.len[a]);
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) dst[i] = src[i];
        }
    }
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

// The drafter ring's static table. With a unit of gpb blocks the ring is a ring of slot GROUPS: block i lands in
// slot (i/gpb % ng)*gpb + i%gpb, so a group's blocks stay in one aligned run. gpb == 1 gives i % n_slots, the
// table every other format has always had.
__global__ void ring_kernel(int32_t* table, long long n_blocks, long long n_slots, int gpb) {
    const long long ng = n_slots / gpb;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks;
         i += (long long) gridDim.x * blockDim.x)
        table[i] = (int32_t) ((i / gpb) % ng * gpb + i % gpb);
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    if (fmt == kKvQ2) return rows * kv_q2_bytes_per_head((int) s.head_dim) * 2;
    // q2k: K's 80 B per cell and head + V's 144 = 224, 448 per cell for 2 KV heads (q2: 320, q4_0: 576)
    if (fmt == kKvQ2K)
        return rows * (kv_q2_bytes_per_head((int) s.head_dim) + kv_q4_bytes_per_head((int) s.head_dim));
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    reset_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m);
    check("reset");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // One sweep step looks at RT consecutive slot GROUPS `(hand + thread) % (n_slots / gpb)`; with fewer groups
    // than that, two threads see the same group and may both take it for two different misses. Every format but
    // q2 has one group per slot, so this is the guard this function always had - gpb == 1 makes it read
    // `n_slots < RT`. The engine never streams the main layers with fewer than qsa_kv_resident_min() /
    // page_size = 5,120 slots, so for q2 (8 slots per group) the ring has to be rounded up: layer.cpp's kv_plan
    // does that, and this is the guard that says so.
    const int gpb = blocks_per_unit(s, fmt);
    if (m.n_slots / gpb < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots at %lld per group is fewer groups than the resolve block "
                             "(%d): the clock sweep would take a group twice\n",
                     (long long) m.n_slots, (long long) gpb, RT);
        std::exit(1);
    }
    resolve_kernel<<<1, RT, 0, (cudaStream_t) stream>>>(m, ids, steps, (int) n_q, (int) cap, (int) s.page_size, gpb);
    check("resolve");
    copy_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(m, runs_of(slots, host, fmt, s));
    check("copy");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream, int gpb) {
    ring_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(page_table, n_blocks, n_slots, gpb);
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    const int64_t unit = r.unit, nu = n_slots / unit;
    // Whole units from the first aligned unit at or below b0 (restoring a little more than asked is harmless:
    // the ring is refilled from the authoritative host copy). unit == 1 is the loop this function had.
    for (int64_t u = b0 / unit; u * unit < b1; ++u) {
        const int64_t sl = u % nu, run = std::min<int64_t>((b1 - u * unit + unit - 1) / unit, nu - sl);
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + u * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        u += run - 1;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    // Both sides are the identity layout, so a contiguous byte range is exactly the blocks' bytes whatever the
    // residency unit is: n_blocks blocks = n_blocks * (len / unit) bytes.
    // A q2/q2k unit is a GROUP of 8 blocks, and half a group is not a byte range the format means (its slots
    // are group-major), so round the block count UP to whole units first. Safe: n_blocks <= n_pages, and
    // layer.cpp's kv_plan rounds n_pages up to a multiple of the unit, so the rounded range is inside both
    // pools. unit == 1 leaves nb == n_blocks - the copy every other format makes.
    const int64_t nb = (n_blocks + r.unit - 1) / r.unit * r.unit;
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (nb * (r.len[a] / r.unit)), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream) {
    if (b1 <= b0) return;
    const Runs r = runs_of(stage, host, fmt, s);   // src: the host copy, dst: the staging pool (identity layout both)
    // Whole units only. The staging pool holds a q2/q2k GROUP, never a slice of one (kv_stage_from_host staged
    // it from the host in whole units), so expand [b0, b1) down/up to unit boundaries before copying: b0 down
    // to a multiple of the unit, b1 up to one (b1 <= n_pages, which kv_plan rounds up to a multiple of the
    // unit, so the rounded end is inside the pool). unit == 1 leaves both bounds exactly as they were.
    const int64_t u0 = b0 / r.unit * r.unit;
    const int64_t u1 = (b1 + r.unit - 1) / r.unit * r.unit;
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync((void*) (r.src[a] + u0 * (r.len[a] / r.unit)), r.dst[a] + u0 * (r.len[a] / r.unit),
                            (size_t) ((u1 - u0) * (r.len[a] / r.unit)), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("unstage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
