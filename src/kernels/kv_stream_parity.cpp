// src/kernels/kv_stream_parity.cpp - KV streaming (kv_stream.hpp) against a fully resident pool (GPU, no model).
//
// Decodes a synthetic sequence twice: into a fully resident pool with the identity page table, and into a streamed
// layer (host copy + a small VRAM slot pool that must evict constantly). Every few cells a batch of 1-8 queries
// with selections shaped like the indexer's (whole 4-cell blocks, a partial threshold block, the tail block, half of
// them recent) is attended in both; `kv_stream_resolve` runs before the streamed attention. Checks:
//   1. the attention outputs are BITWISE equal (the streamed readers see exactly the resident values);
//   2. the residency map is consistent after every call (slot_block and page_table invert each other);
//   3. no call overflowed, and the hit/miss counters add up;
//   4. a ring (the MTP drafter's layout) restored from the host copy reads the same values as the resident pool;
//   5. STRATA_FUSE_KVRESOLVE: a second streamed layer resolved with the copy in the resolve block
//      (kv_stream_resolve_with(..., true)): its attention BITWISE equal to the resident pool's, its map consistent,
//      no overflow, the same lookups.  Its map itself is not compared with the first layer's: the resolve's miss list
//      is filled by a shared-memory atomicAdd, so the block <-> victim pairing (and from then on which slots the hits
//      stamp) varies from run to run on either path - STRATA_KV_PARITY_CONTROL=1 resolves the second layer with the
//      copy kernel too and counts the calls after which the two maps differ.
//   6. STRATA_FUSE_QSAWAVE: two more streamed layers resolved together by kv_stream_resolve_rows (one block per
//      layer, layer z resolving query z of the batch; copy kernel and fused copy on alternate batches), each layer's
//      attention of its query BITWISE equal to the resident pool's, maps consistent, no overflow.
// INT8, FP16, Q4_0 (PR #21) and hybrid K8V4 pools; K8V4's appends are the engine's folded calls (layer.cpp), its host
// copy written through kv_hybrid_k_half / kv_hybrid_v_half.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T> T* halloc(size_t n) {   // pinned, mapped; returns the device pointer
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer(&d, h, 0), "devptr");
    return (T*) d;
}

struct Pools {   // one K/V pool set of `pages` pages
    k::KvHostPools p;   // reused as a plain pointer bundle
    void alloc(int64_t pages, const k::QsaShapes& s, int fmt, bool host) {
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size;
        if (fmt == k::kKvHybrid) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvQ4) {
            const size_t b = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
            p.k_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
            p.v_q4 = host ? halloc<uint8_t>(b) : dalloc<uint8_t>(b);
        } else if (fmt == k::kKvInt8) {
            p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.v_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
            p.k_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
            p.v_scale = host ? halloc<uint16_t>(rows * 4) : dalloc<uint16_t>(rows * 4);
        } else {
            p.k_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
            p.v_pool = host ? halloc<uint16_t>(rows * s.head_dim) : dalloc<uint16_t>(rows * s.head_dim);
        }
    }
    // every allocated array, as (device pointer, bytes) - alloc's sizes
    std::vector<std::pair<const void*, size_t>> arrays(int64_t pages, const k::QsaShapes& s) const {
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size, q4 = rows * k::kv_q4_bytes_per_head((int) s.head_dim);
        std::vector<std::pair<const void*, size_t>> a;
        if (p.k_pool) { a.push_back({p.k_pool, rows * s.head_dim * 2}); a.push_back({p.v_pool, rows * s.head_dim * 2}); }
        if (p.k_q) a.push_back({p.k_q, rows * s.head_dim});
        if (p.v_q) a.push_back({p.v_q, rows * s.head_dim});
        if (p.k_scale) a.push_back({p.k_scale, rows * 8});
        if (p.v_scale) a.push_back({p.v_scale, rows * 8});
        if (p.k_q4) a.push_back({p.k_q4, q4});
        if (p.v_q4) a.push_back({p.v_q4, q4});
        return a;
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale;
        a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4; a.page_table = table;
        return a;
    }
};

void append(const Pools& pl, const int32_t* table, const int32_t* step, const float* kc, const float* vc,
            const k::QsaShapes& s, int fmt, const k::KvHostPools* host) {
    if (fmt == k::kKvHybrid) {   // exactly as qsa_layer: each half folded onto its own pool, host copy by halves
        const k::KvHostPools hk = host ? k::kv_hybrid_k_half(*host) : k::KvHostPools{},
                             hv = host ? k::kv_hybrid_v_half(*host) : k::KvHostPools{};
        k::kv_append_q8_step(pl.p.k_q, pl.p.k_q, pl.p.k_scale, pl.p.k_scale, table, step, kc, kc, s, nullptr,
                             host ? &hk : nullptr);
        k::kv_append_q4_step(pl.p.v_q4, pl.p.v_q4, table, step, vc, vc, s, nullptr, host ? &hv : nullptr);
    } else if (fmt == k::kKvQ4) k::kv_append_q4_step(pl.p.k_q4, pl.p.v_q4, table, step, kc, vc, s, nullptr, host);
    else if (fmt == k::kKvInt8) k::kv_append_q8_step(pl.p.k_q, pl.p.v_q, pl.p.k_scale, pl.p.v_scale, table, step, kc, vc, s, nullptr, host);
    else k::kv_append_step(pl.p.k_pool, pl.p.v_pool, table, step, kc, vc, s, nullptr, host);
}

// A selection like qsa_block_topk's: ascending cells, whole blocks, the tail block's cells, and one partial block.
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1), recent(std::max<int64_t>(0, n_bid - 2048), n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(rng() % 2 ? recent(rng) : any(rng));
    int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool run(int fmt) {
    const char* name = fmt == k::kKvHybrid ? "k8v4" : fmt == k::kKvQ4 ? "q4_0" : fmt == k::kKvInt8 ? "int8" : "fp16";
    k::QsaShapes s = k::qsa_real_shapes();
    const int64_t N = 40000, n_blocks = (N + 3) / 4, n_slots = 8 * 516 + 700;   // must evict: slots < blocks
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s), NQ = 8;
    const int64_t H = s.n_head_kv, D = s.head_dim, NH = s.n_head;
    std::mt19937 rng(7 + 4 * fmt);
    std::normal_distribution<float> nd(0.f, 1.f);

    Pools ref, slots, host, slots2, slots3[2];
    ref.alloc(n_blocks, s, fmt, false);
    slots.alloc(n_slots, s, fmt, false);
    slots2.alloc(n_slots, s, fmt, false);
    for (Pools& p3 : slots3) p3.alloc(n_slots, s, fmt, false);
    host.alloc(n_blocks, s, fmt, true);
    int32_t* ident = dalloc<int32_t>(n_blocks);
    {
        std::vector<int32_t> t(n_blocks);
        for (int64_t i = 0; i < n_blocks; ++i) t[i] = (int32_t) i;
        ck(cudaMemcpy(ident, t.data(), n_blocks * 4, cudaMemcpyHostToDevice), "ident");
    }
    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>(n_blocks);
    m.slot_block = dalloc<int32_t>(n_slots); m.slot_stamp = dalloc<int32_t>(n_slots); m.slot_ref = dalloc<int32_t>(n_slots);
    m.miss_block = dalloc<int32_t>(n_slots); m.miss_slot = dalloc<int32_t>(n_slots);
    m.ctl = dalloc<int32_t>(k::kKvCtlInts);
    m.n_blocks = n_blocks; m.n_slots = n_slots;
    k::kv_stream_reset(m, nullptr);
    k::KvStreamMap m2 = m;   // 5: the fused resolve's own map
    m2.page_table = dalloc<int32_t>(n_blocks);
    m2.slot_block = dalloc<int32_t>(n_slots); m2.slot_stamp = dalloc<int32_t>(n_slots); m2.slot_ref = dalloc<int32_t>(n_slots);
    m2.miss_block = dalloc<int32_t>(n_slots); m2.miss_slot = dalloc<int32_t>(n_slots);
    m2.ctl = dalloc<int32_t>(k::kKvCtlInts);
    k::kv_stream_reset(m2, nullptr);
    k::KvStreamMap m3[2] = {m, m};   // 6: two layers resolved in one kv_stream_resolve_rows launch
    for (k::KvStreamMap& mm : m3) {
        mm.page_table = dalloc<int32_t>(n_blocks);
        mm.slot_block = dalloc<int32_t>(n_slots); mm.slot_stamp = dalloc<int32_t>(n_slots); mm.slot_ref = dalloc<int32_t>(n_slots);
        mm.miss_block = dalloc<int32_t>(n_slots); mm.miss_slot = dalloc<int32_t>(n_slots);
        mm.ctl = dalloc<int32_t>(k::kKvCtlInts);
        k::kv_stream_reset(mm, nullptr);
    }
    int rows_bad = 0, rows_calls = 0;
    int fused_bad = 0, pairing_moved = 0;
    static const bool control = std::getenv("STRATA_KV_PARITY_CONTROL") != nullptr;
    auto down8 = [](const void* x, size_t n) {
        std::vector<uint8_t> a(n);
        ck(cudaMemcpy(a.data(), x, n, cudaMemcpyDeviceToHost), "down");
        return a;
    };

    float* kc = dalloc<float>(H * D);
    float* vc = dalloc<float>(H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    int32_t* steps = dalloc<int32_t>(NQ * k::kStepCount);
    int32_t* ids = dalloc<int32_t>(NQ * cap);
    float* q = dalloc<float>(NQ * NH * D);
    const uint64_t scr = k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch = dalloc<float>(NQ * scr);
    float* out_ref = dalloc<float>(NQ * NH * D);
    float* out_str = dalloc<float>(NQ * NH * D);
    std::vector<float> hk(H * D), hv(H * D), hq(NQ * NH * D), a((size_t) NQ * NH * D), b2((size_t) NQ * NH * D);
    std::vector<int32_t> pt(n_blocks), sb(n_slots);

    int batches = 0, bad = 0;
    for (int64_t pos = 0; pos < N; ++pos) {
        for (auto& x : hk) x = nd(rng) * ((pos % 17 == 0) ? 30.f : 1.f);
        for (auto& x : hv) x = nd(rng);
        const int32_t st[4] = {(int32_t) pos, (int32_t) (pos + 1), (int32_t) ((pos + 1) / 4), 0};
        ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "v");
        ck(cudaMemcpy(step, st, sizeof(st), cudaMemcpyHostToDevice), "step");
        append(ref, ident, step, kc, vc, s, fmt, nullptr);
        append(slots, m.page_table, step, kc, vc, s, fmt, &host.p);
        append(slots2, m2.page_table, step, kc, vc, s, fmt, &host.p);   // (the host copy gets the same bytes twice)
        for (int z = 0; z < 2; ++z) append(slots3[z], m3[z].page_table, step, kc, vc, s, fmt, &host.p);
        if (pos % 131 != 130 && pos != N - 1) continue;
        // a batch: n_q queries at the last n_q positions (as a verify window), each with its own selection
        const int n_q = 1 + (int) (rng() % NQ);
        std::vector<int32_t> hids((size_t) NQ * cap, 0), hst(NQ * 4, 0);
        for (int t = 0; t < n_q; ++t) {
            const int64_t p = std::max<int64_t>(0, pos - (n_q - 1 - t)), n_kv = p + 1;
            const int64_t width = std::min<int64_t>(n_kv, cap);
            const std::vector<int32_t> sel = selection(n_kv, width, rng);
            if ((int64_t) sel.size() != width) { std::fprintf(stderr, "selection size %zu != %lld\n", sel.size(), (long long) width); return false; }
            std::copy(sel.begin(), sel.end(), hids.begin() + (size_t) t * cap);
            hst[t * 4 + 0] = (int32_t) p; hst[t * 4 + 1] = (int32_t) n_kv; hst[t * 4 + 2] = (int32_t) (n_kv / 4);
            hst[t * 4 + 3] = (int32_t) width;
        }
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, n_q, nullptr);
        k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, fmt, ids, steps, n_q, cap, s, nullptr);
        k::qsa_decode_attn_batch(q, slots.attn(m.page_table), ids, steps, cap, s, scratch, out_str, n_q, nullptr);
        k::kv_stream_resolve_with(m2, slots2.attn(m2.page_table), host.p, fmt, ids, steps, n_q, cap, s, nullptr, !control);
        ck(cudaDeviceSynchronize(), "batch");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "b");
        if (std::memcmp(a.data(), b2.data(), (size_t) n_q * NH * D * 4) != 0) {
            if (bad++ < 5) std::fprintf(stderr, "  %s pos %lld n_q %d: streamed attention differs\n", name, (long long) pos, n_q);
        }
        // the map inverts itself
        ck(cudaMemcpy(pt.data(), m.page_table, n_blocks * 4, cudaMemcpyDeviceToHost), "pt");
        ck(cudaMemcpy(sb.data(), m.slot_block, n_slots * 4, cudaMemcpyDeviceToHost), "sb");
        int64_t resident = 0;
        for (int64_t bk = 0; bk < n_blocks; ++bk) {
            if (pt[bk] < -1 || pt[bk] >= n_slots || (pt[bk] >= 0 && sb[pt[bk]] != bk)) {
                if (bad++ < 5) std::fprintf(stderr, "  map broken at block %lld (table %d)\n", (long long) bk, pt[bk]);
                break;
            }
            resident += pt[bk] >= 0;
        }
        for (int64_t sl = 0; sl < n_slots; ++sl)
            if (sb[sl] >= 0 && pt[sb[sl]] != sl) { if (bad++ < 5) std::fprintf(stderr, "  slot %lld not in the table\n", (long long) sl); break; }
        {   // 5. the fused resolve's layer: its own attention against the resident pool's, its map consistent
            k::qsa_decode_attn_batch(q, slots2.attn(m2.page_table), ids, steps, cap, s, scratch, out_str, n_q, nullptr);
            ck(cudaDeviceSynchronize(), "fused batch");
            ck(cudaMemcpy(b2.data(), out_str, (size_t) n_q * NH * D * 4, cudaMemcpyDeviceToHost), "b2");
            bool same = std::memcmp(a.data(), b2.data(), (size_t) n_q * NH * D * 4) == 0;
            const std::vector<uint8_t> t1 = down8(m.page_table, n_blocks * 4), t2 = down8(m2.page_table, n_blocks * 4);
            const std::vector<uint8_t> s2 = down8(m2.slot_block, n_slots * 4);
            if (std::memcmp(t1.data(), t2.data(), t1.size()) != 0) ++pairing_moved;
            const int32_t* p2 = (const int32_t*) t2.data();
            const int32_t* q2 = (const int32_t*) s2.data();
            for (int64_t bk = 0; same && bk < n_blocks; ++bk)
                if (p2[bk] < -1 || p2[bk] >= n_slots || (p2[bk] >= 0 && q2[p2[bk]] != bk)) same = false;
            if (!same && fused_bad++ < 5)
                std::fprintf(stderr, "  %s pos %lld n_q %d: fused resolve + copy: attention or map wrong\n", name, (long long) pos, n_q);
        }
        if (n_q >= 2) {   // 6. layers m3[0], m3[1] resolve queries 0 and 1 in one launch, then attend them
            k::KvResolveRows kr;
            for (int z = 0; z < 2; ++z) {
                kr.m[z] = m3[z];
                kr.slots[z] = slots3[z].attn(m3[z].page_table);
                kr.host[z] = host.p;
                kr.row[z] = z;
            }
            kr.n = 2;
            k::kv_stream_resolve_rows(kr, fmt, ids, steps, cap, s, nullptr, (rows_calls & 1) == 1);
            ++rows_calls;
            for (int z = 0; z < 2; ++z) {
                k::qsa_decode_attn_batch(q + (size_t) z * NH * D, slots3[z].attn(m3[z].page_table), ids + (size_t) z * cap,
                                         steps + (size_t) z * k::kStepCount, cap, s, scratch, out_str, 1, nullptr);
                ck(cudaDeviceSynchronize(), "rows batch");
                ck(cudaMemcpy(b2.data(), out_str, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "b3");
                bool same = std::memcmp(a.data() + (size_t) z * NH * D, b2.data(), (size_t) NH * D * 4) == 0;
                const std::vector<uint8_t> t3 = down8(m3[z].page_table, n_blocks * 4), s3 = down8(m3[z].slot_block, n_slots * 4);
                const int32_t* p3 = (const int32_t*) t3.data();
                const int32_t* q3 = (const int32_t*) s3.data();
                for (int64_t bk = 0; same && bk < n_blocks; ++bk)
                    if (p3[bk] < -1 || p3[bk] >= n_slots || (p3[bk] >= 0 && q3[p3[bk]] != bk)) same = false;
                if (!same && rows_bad++ < 5)
                    std::fprintf(stderr, "  %s pos %lld layer %d: resolve rows: attention or map wrong\n", name, (long long) pos, z);
            }
        }
        ++batches;
    }
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    std::printf("  %s: %d batches, %llu block lookups, %llu misses (%.1f%% hit), overflow %d, %d failures\n",
                name, batches, (unsigned long long) c.lookups, (unsigned long long) c.misses,
                c.lookups ? 100.0 * (double) (c.lookups - c.misses) / (double) c.lookups : 0.0, (int) c.overflow, bad);
    if (c.overflow || c.calls != (uint64_t) batches || c.misses == 0 || c.misses >= c.lookups) ++bad;
    const k::KvStreamCounters c2 = k::kv_stream_counters(m2);
    if (c2.overflow || c2.lookups != c.lookups || c2.calls != c.calls) ++fused_bad;
    std::printf("  %s %s: attention %s in %d calls, %llu misses (first layer %llu), overflow %d; the two maps differed "
                "after %d calls\n", name, control ? "second layer, copy kernel (control)" : "fused resolve + copy (STRATA_FUSE_KVRESOLVE)",
                fused_bad ? "WRONG" : "bitwise", batches, (unsigned long long) c2.misses, (unsigned long long) c.misses,
                (int) c2.overflow, pairing_moved);
    bad += fused_bad;
    for (int z = 0; z < 2; ++z) if (k::kv_stream_counters(m3[z]).overflow) ++rows_bad;
    std::printf("  %s resolve rows (STRATA_FUSE_QSAWAVE): 2 layers per launch, %d launches (copy kernel / fused copy "
                "alternating), attention %s\n", name, rows_calls, rows_bad ? "WRONG" : "bitwise");
    bad += rows_bad;

    // 5b. the fused copy's worst case, timed (not a check): every map reset, one query of full width - every block a miss
    {
        std::vector<int32_t> hids((size_t) cap), hst = {(int32_t) (N - 1), (int32_t) N, (int32_t) (N / 4), (int32_t) cap};
        for (int64_t i = 0; i < cap; ++i) hids[i] = (int32_t) (N - cap + i);
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), 16, cudaMemcpyHostToDevice), "steps");
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev");
        float ms[2][2] = {};
        for (int f = 0; f < 2; ++f)
            for (int rep = 0; rep < 2; ++rep) {   // rep 1: the timed one; rep 0 warms up (and the hits path after it)
                const k::KvStreamMap& mm = f ? m2 : m;
                const Pools& sp = f ? slots2 : slots;
                k::kv_stream_reset(mm, nullptr);
                ck(cudaEventRecord(e0, nullptr), "rec");
                k::kv_stream_resolve_with(mm, sp.attn(mm.page_table), host.p, fmt, ids, steps, 1, cap, s, nullptr, f == 1);
                ck(cudaEventRecord(e1, nullptr), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                ck(cudaEventElapsedTime(&ms[f][rep], e0, e1), "elapsed");
            }
        std::printf("  %s all-miss call (%lld blocks): resolve + copy kernel %.1f us, fused %.1f us\n", name,
                    (long long) ((cap + 3) / 4), ms[0][1] * 1e3, ms[1][1] * 1e3);
    }

    // 4. a ring over the same host copy: blocks [b1 - R, b1) restored, read through `b % R`
    {
        const int64_t R = 1500, b1 = n_blocks, b0 = b1 - R;
        Pools ring;
        ring.alloc(R, s, fmt, false);
        int32_t* rt = dalloc<int32_t>(n_blocks);
        k::kv_ring_table(rt, n_blocks, R, nullptr);
        k::kv_ring_restore(ring.attn(rt), host.p, fmt, b0, b1, R, s, nullptr);
        const int64_t width = cap;
        std::vector<int32_t> hids((size_t) cap), hst = {(int32_t) (N - 1), (int32_t) N, (int32_t) (N / 4), (int32_t) width};
        for (int64_t i = 0; i < width; ++i) hids[i] = (int32_t) (N - width + i);   // the window's last cells
        ck(cudaMemcpy(ids, hids.data(), hids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), 16, cudaMemcpyHostToDevice), "steps");
        k::qsa_decode_attn_batch(q, ref.attn(ident), ids, steps, cap, s, scratch, out_ref, 1, nullptr);
        k::qsa_decode_attn_batch(q, ring.attn(rt), ids, steps, cap, s, scratch, out_str, 1, nullptr);
        ck(cudaDeviceSynchronize(), "ring");
        ck(cudaMemcpy(a.data(), out_ref, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b2.data(), out_str, (size_t) NH * D * 4, cudaMemcpyDeviceToHost), "b");
        const bool ok = std::memcmp(a.data(), b2.data(), (size_t) NH * D * 4) == 0;
        std::printf("  %s ring restore: %s\n", name, ok ? "identical" : "DIFFERS");
        if (!ok) ++bad;
    }
    return bad == 0;
}
}  // namespace

int main() {
    std::printf("kv_stream_parity: streamed vs resident KV, bitwise\n");
    const bool a = run(k::kKvInt8), b = run(k::kKvF16), c = run(k::kKvQ4), d = run(k::kKvHybrid);
    if (!a || !b || !c || !d) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
