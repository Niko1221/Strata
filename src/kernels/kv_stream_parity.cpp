// src/kernels/kv_stream_parity.cpp - KV streaming (kv_stream.hpp) against a fully resident pool (GPU, no model).
//
// Decodes a synthetic sequence twice: into a fully resident pool with the identity page table, and into a streamed
// layer (host copy + a small VRAM slot pool that must evict constantly). Every few cells a batch of 1-8 queries
// with selections shaped like the indexer's (whole 4-cell blocks, a partial threshold block, the tail block, half of
// them recent) is attended in both; `kv_stream_resolve` runs before the streamed attention. Checks:
//   1. the attention outputs are BITWISE equal (the streamed readers see exactly the resident values);
//   2. the residency map is consistent after every call (slot_block and page_table invert each other);
//   3. no call overflowed, and the hit/miss counters add up;
//   4. a ring (the MTP drafter's layout) restored from the host copy reads the same values as the resident pool.
// INT8, FP16, Q4_0 (PR #21) and hybrid K8V4 pools; K8V4's appends are the engine's folded calls (layer.cpp), its host
// copy written through kv_hybrid_k_half / kv_hybrid_v_half.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/core/kv_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

    Pools ref, slots, host;
    ref.alloc(n_blocks, s, fmt, false);
    slots.alloc(n_slots, s, fmt, false);
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
        ++batches;
    }
    const k::KvStreamCounters c = k::kv_stream_counters(m);
    std::printf("  %s: %d batches, %llu block lookups, %llu misses (%.1f%% hit), overflow %d, %d failures\n",
                name, batches, (unsigned long long) c.lookups, (unsigned long long) c.misses,
                c.lookups ? 100.0 * (double) (c.lookups - c.misses) / (double) c.lookups : 0.0, (int) c.overflow, bad);
    if (c.overflow || c.calls != (uint64_t) batches || c.misses == 0 || c.misses >= c.lookups) ++bad;

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

// Shared and private lanes execute the same appends, resolves and attention calls. The shared side
// exercises the production KvPool's pinned allocation, physical copies and fixed device-table uploads.
bool shared_branches(int fmt, int64_t C, int chunk_shift) {
    using strata::core::KvChunkMap;
    using strata::core::KvPool;
    const auto s=k::qsa_real_shapes();
    // The resolve window stays independent of the host chunk size.
    constexpr int64_t slots=1024, width=8;
    const int64_t cells=2*C+5, logical=std::max<int64_t>(3,3*slots*s.page_size/C), blocks=logical*C/s.page_size;
    strata::core::ModelGeometry g;
    g.n_head=s.n_head; g.n_head_kv=s.n_head_kv; g.head_dim=s.head_dim;
    std::array<strata::core::SessionState,4> sessions;
    std::array<strata::core::QsaState,4> states;
    for (size_t n=0;n<4;++n) {
        auto& st=states[n];
        st.kv_mode=1; st.kv_int8=fmt==k::kKvInt8; st.kv_q4=fmt==k::kKvQ4; st.kv_hybrid=fmt==k::kKvHybrid;
        st.max_cells=logical*C; st.n_pages=blocks; st.n_slots=slots;
        sessions[n].max_cells=logical*C; sessions[n].qsa_alloc=1; sessions[n].qsa_states=&st;
    }
    KvPool shared;
    shared.set_share_prefix(true);
    std::string error;
    if (!shared.init(g,{{&sessions[0],-1}},14*C,error,C)) {
        std::fprintf(stderr,"shared branch pool init: %s\n",error.c_str()); return false;
    }
    if (shared.chunk_cells()!=C) {
        std::fprintf(stderr,"shared branch pool did not use the configured chunk size\n"); return false;
    }
    std::array<const int32_t*,4> stable{},stable_device{};
    for (size_t n=0;n<4;++n) {
        if (!shared.attach(sessions[n],error)) {
            std::fprintf(stderr,"shared branch pool attach: %s\n",error.c_str()); return false;
        }
        stable[n]=states[n].host.chunk_host; stable_device[n]=states[n].host.chunk;
    }
    KvChunkMap private_map;
    private_map.init(12);
    std::array<KvChunkMap::Lane,4> plain{};
    std::array<int32_t*,4> private_table{};
    std::vector<void*> device;
    auto alloc=[&](size_t bytes) {
        auto* p=dalloc<uint8_t>(bytes); device.push_back(p); return p;
    };
    Pools shared_host,private_host,shared_slots,private_slots;
    shared_host.p=states[0].host; // borrowed process-lifetime pinned storage; never free it as test memory
    private_host.alloc(13*C/s.page_size,s,fmt,true);
    shared_slots.alloc(slots,s,fmt,false); private_slots.alloc(slots,s,fmt,false);
    auto arrays=[&](const Pools& p) {
        const size_t rows=C*s.n_head_kv, q4=rows*k::kv_q4_bytes_per_head((int)s.head_dim);
        return std::array<std::pair<void*,size_t>,8>{{
            {p.p.k_pool,rows*s.head_dim*2},{p.p.v_pool,rows*s.head_dim*2},
            {p.p.k_q,rows*s.head_dim},{p.p.v_q,rows*s.head_dim},
            {p.p.k_scale,rows*(s.head_dim/64)*2},{p.p.v_scale,rows*(s.head_dim/64)*2},
            {p.p.k_q4,q4},{p.p.v_q4,q4}}};
    };
    auto copy=[&](const Pools& from,int32_t src,const Pools& to,int32_t dst) {
        auto a=arrays(from),b=arrays(to);
        for (size_t part=0;part<a.size();++part) if (a[part].first)
            ck(cudaMemcpy(static_cast<uint8_t*>(b[part].first)+size_t(dst)*b[part].second,
                          static_cast<uint8_t*>(a[part].first)+size_t(src)*a[part].second,
                          a[part].second,cudaMemcpyDefault),"prefix physical copy");
    };
    auto upload=[&](const KvChunkMap& map,KvChunkMap::Lane lane,int32_t* dst) {
        ck(cudaMemcpy(dst,map.table(lane),logical*sizeof(int32_t),cudaMemcpyHostToDevice),"prefix table upload");
    };
    for (size_t n=0;n<4;++n) {
        plain[n]=private_map.add_lane(logical);
        private_table[n]=reinterpret_cast<int32_t*>(alloc(logical*sizeof(int32_t)));
    }
    bool ok=shared.prepare_write(sessions[0],0,cells);
    for (auto lane:plain) ok=private_map.reserve(lane,3) && ok;
    auto make_map=[&]() {
        k::KvStreamMap m;
        m.page_table=reinterpret_cast<int32_t*>(alloc(blocks*4));
        m.slot_block=reinterpret_cast<int32_t*>(alloc(slots*4));
        m.slot_stamp=reinterpret_cast<int32_t*>(alloc(slots*4));
        m.slot_ref=reinterpret_cast<int32_t*>(alloc(slots*4));
        m.miss_block=reinterpret_cast<int32_t*>(alloc(slots*4));
        m.miss_slot=reinterpret_cast<int32_t*>(alloc(slots*4));
        m.ctl=reinterpret_cast<int32_t*>(alloc(k::kKvCtlInts*4));
        m.n_blocks=blocks; m.n_slots=slots;
        k::kv_stream_reset(m,nullptr);
        return m;
    };
    auto sm=make_map(),pm=make_map();
    const size_t kv=s.n_head_kv*s.head_dim, query=s.n_head*s.head_dim;
    auto* kc=reinterpret_cast<float*>(alloc(kv*4));
    auto* vc=reinterpret_cast<float*>(alloc(kv*4));
    auto* step=reinterpret_cast<int32_t*>(alloc(k::kStepCount*4));
    auto* ids=reinterpret_cast<int32_t*>(alloc(width*4));
    auto* q=reinterpret_cast<float*>(alloc(query*4));
    auto* scratch=reinterpret_cast<float*>(alloc(k::qsa_decode_attn_scratch_floats(width,s)*4));
    auto* so=reinterpret_cast<float*>(alloc(query*4));
    auto* po=reinterpret_cast<float*>(alloc(query*4));
    std::vector<float> hk(kv),hv(kv),hq(query,0.125f),a(query),b(query);
    ck(cudaMemcpy(q,hq.data(),query*4,cudaMemcpyHostToDevice),"branch queries");
    auto host=[&](const Pools& p,int32_t* table) {
        auto h=p.p; h.chunk=table; h.chunk_shift=chunk_shift; return h;
    };
    auto append_pair=[&](size_t lane,int64_t pos,int salt) {
        for (size_t j=0;j<kv;++j) {
            hk[j]=float(int((j*7+pos*3+salt)%127)-63)/32.f;
            hv[j]=float(int((j*11+pos+salt*3)%101)-50)/16.f;
        }
        const int32_t st[4]={int32_t(pos),int32_t(pos+1),int32_t((pos+1)/4),0};
        ck(cudaMemcpy(kc,hk.data(),kv*4,cudaMemcpyHostToDevice),"branch K");
        ck(cudaMemcpy(vc,hv.data(),kv*4,cudaMemcpyHostToDevice),"branch V");
        ck(cudaMemcpy(step,st,sizeof(st),cudaMemcpyHostToDevice),"branch append step");
        const auto& sh=states[lane].host;
        auto ph=host(private_host,private_table[lane]);
        append(shared_slots,sm.page_table,step,kc,vc,s,fmt,&sh);
        append(private_slots,pm.page_table,step,kc,vc,s,fmt,&ph);
    };
    upload(private_map,plain[0],private_table[0]);
    for (int64_t pos=0;pos<cells;++pos) append_pair(0,pos,0);
    ck(cudaDeviceSynchronize(),"branch base");
    auto prefix=shared.retain(states[0],cells);
    ok=bool(prefix) && ok;
    if (!prefix) return false;
    for (size_t n=1;n<4;++n) {
        ok=shared.share(sessions[0],sessions[n],cells) && ok;
        for (int c=0;c<3;++c) copy(private_host,private_map.table(plain[0])[c],private_host,private_map.table(plain[n])[c]);
        ok=states[n].host.chunk_host[0]==states[0].host.chunk_host[0] &&
           states[n].host.chunk_host[1]==states[0].host.chunk_host[1] &&
           states[n].host.chunk_host[2]!=states[0].host.chunk_host[2] && ok;
        upload(private_map,plain[n],private_table[n]);
    }
    prefix.reset();
    auto compare=[&](size_t n) {
        auto sh=arrays(shared_host),ph=arrays(private_host);
        for (size_t part=0;part<sh.size();++part) if (sh[part].first) {
            const size_t stride=sh[part].second/C;
            for (int64_t c=0;c<3;++c) {
                const size_t bytes=size_t(std::min<int64_t>(C,cells-c*C))*stride;
                std::vector<uint8_t> x(bytes),y(bytes);
                ck(cudaMemcpy(x.data(),static_cast<uint8_t*>(sh[part].first)+
                              size_t(states[n].host.chunk_host[c])*sh[part].second,bytes,cudaMemcpyDefault),"shared bytes");
                ck(cudaMemcpy(y.data(),static_cast<uint8_t*>(ph[part].first)+
                              size_t(private_map.table(plain[n])[c])*ph[part].second,bytes,cudaMemcpyDefault),"private bytes");
                ok=(x==y) && ok;
            }
        }
        // Force both readers through host resolve, not stale residency left by the preceding branch.
        k::kv_stream_reset(sm,nullptr); k::kv_stream_reset(pm,nullptr);
        const int32_t selected[width]={0,1,int32_t(C-2),int32_t(C-1),int32_t(C),int32_t(C+1),int32_t(2*C),int32_t(2*C+4)};
        const int32_t st[4]={int32_t(cells-1),int32_t(cells),int32_t(cells/4),int32_t(width)};
        ck(cudaMemcpy(ids,selected,sizeof(selected),cudaMemcpyHostToDevice),"branch selection");
        ck(cudaMemcpy(step,st,sizeof(st),cudaMemcpyHostToDevice),"branch attend step");
        const auto& hs=states[n].host;
        auto hp=host(private_host,private_table[n]);
        k::kv_stream_resolve(sm,shared_slots.attn(sm.page_table),hs,fmt,ids,step,1,width,s,nullptr);
        k::kv_stream_resolve(pm,private_slots.attn(pm.page_table),hp,fmt,ids,step,1,width,s,nullptr);
        k::qsa_decode_attn_batch(q,shared_slots.attn(sm.page_table),ids,step,width,s,scratch,so,1,nullptr);
        k::qsa_decode_attn_batch(q,private_slots.attn(pm.page_table),ids,step,width,s,scratch,po,1,nullptr);
        ck(cudaDeviceSynchronize(),"branch resolve and attend");
        ck(cudaMemcpy(a.data(),so,query*4,cudaMemcpyDeviceToHost),"shared attention");
        ck(cudaMemcpy(b.data(),po,query*4,cudaMemcpyDeviceToHost),"private attention");
        ok=std::memcmp(a.data(),b.data(),query*4)==0 && !k::kv_stream_counters(sm).overflow &&
           !k::kv_stream_counters(pm).overflow && ok;
    };
    for (size_t n=0;n<4;++n) compare(n);
    for (size_t n=0;n<4;++n) {
        const int32_t tail=states[n].host.chunk_host[2];
        ok=shared.prepare_write(sessions[n],C-2,C+7) && ok;
        ok=states[n].host.chunk_host[2]==tail && ok;
        // Same rewind/speculative-window execution schedule; distinct candidates straddle a chunk boundary.
        k::kv_stream_reset(sm,nullptr); k::kv_stream_reset(pm,nullptr);
        for (int64_t pos=C-2;pos<C+7;++pos) append_pair(n,pos,int(n)+1);
        ck(cudaDeviceSynchronize(),"branch candidates");
        // Include untouched siblings after every branch write: delayed shared corruption cannot hide.
        for (size_t sibling=0;sibling<4;++sibling) compare(sibling);
    }
    for (size_t n=0;n<4;++n) {
        ok=stable[n]==states[n].host.chunk_host && stable_device[n]==states[n].host.chunk && ok;
        shared.detach(sessions[n]); private_map.shrink(plain[n],0);
    }
    ok=shared.free_cells()==shared.total_cells() && private_map.free_chunks()==private_map.n_chunks() && ok;
    for (auto* p:device) ck(cudaFree(p),"branch free");
    for (const Pools* p:{&private_host,&shared_slots,&private_slots})
        for (const auto& [ptr,bytes]:arrays(*p)) if (ptr) {
            (void)bytes;
            cudaPointerAttributes attr{}; ck(cudaPointerGetAttributes(&attr,ptr),"branch pointer");
            if (attr.hostPointer) ck(cudaFreeHost(attr.hostPointer),"branch host free");
            else ck(cudaFree(ptr),"branch pool free");
        }
    std::printf("  format %d, %lld-cell chunks, four shared branches: %s\n",
                fmt,(long long)C,ok ? "bitwise identical" : "DIFFERS");
    return ok;
}
}  // namespace

int main() {
    std::printf("kv_stream_parity: streamed vs resident KV, bitwise\n");
    const bool a = run(k::kKvInt8), b = run(k::kKvF16), c = run(k::kKvQ4), d = run(k::kKvHybrid);
    if (!a || !b || !c || !d) ++g_fail;
    for (const auto [cells,shift] : {std::pair<int64_t,int>{1024,8},{4096,10}})
        for (int fmt:{k::kKvF16,k::kKvInt8,k::kKvQ4,k::kKvHybrid})
            if (!shared_branches(fmt,cells,shift)) ++g_fail;
    if (!shared_branches(k::kKvF16,8192,11)) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
