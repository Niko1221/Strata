// src/kernels/cpu/q2_lut.cpp - see include/strata/kernels/cpu/q2_lut.hpp: the switch, the stable ranges, the
// mirrors and the per-thread tables in front of the kernels of q2_lut_avx1.cpp / q2_lut_snb.cpp.  Baseline x86-64:
// nothing here needs more than SSE2, the kernels are called only behind q2_lut_enabled() (AVX), and the F16C one
// only where CPUID says F16C.
//
// Mirrors.  Keyed by the matrix's address; made tile by tile (16 rows) by the threads that need the tiles: a thread
// claims an empty tile (0 -> 1), repacks it, publishes it (2); a tile another thread is repacking it waits for.
// Nobody falls back to q2_rows_any while a mirror is incomplete: the two kernels sum floats in different orders, so
// a fallback would make a row's last bits depend on thread timing.  Mirror memory comes from 256 MiB chunks (one
// mapping per matrix would run into vm.max_map_count: a Flash-Next arena has 73,728 matrices) and is first touched
// by the repacking thread, the one that computes those rows.
//
// Tables.  Per thread, a few entries keyed by the ActQ's address AND its bytes: an entry is reused only when
// q[0..k) is identical (memcmp), so an ActQ refilled in place (the pool's per-expert a2) can never meet stale
// tables.  The gate/up activation is the same for every expert of a layer, so a thread builds its tables once per
// layer and token.
#include "strata/kernels/cpu/q2_lut.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "q2_lut_kernels.hpp"

#include <emmintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace strata::kernels::cpu {
namespace {

using namespace q2lut;

bool has_f16c() {   // CPUID.1:ECX[29]
    static const bool f = [] {
#if defined(_MSC_VER)
        int r[4];
        __cpuid(r, 1);
        return ((unsigned) r[2] >> 29 & 1u) != 0;
#else
        unsigned a, b, c, d;
        return __get_cpuid(1, &a, &b, &c, &d) && (c >> 29 & 1u) != 0;
#endif
    }();
    return f;
}

struct Mirror {
    int rows = 0, nblk = 0, ntiles = 0;
    uint8_t* data = nullptr;
    std::unique_ptr<std::atomic<uint8_t>[]> state;   // per tile: 0 empty, 1 being repacked, 2 ready
};

struct Registry {
    std::mutex mu;
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;   // [lo, hi)
    std::unordered_map<const uint8_t*, std::unique_ptr<Mirror>> mirrors;
    std::vector<std::unique_ptr<uint8_t[]>> chunks;
    size_t chunk_used = 0, chunk_cap = 0;
    uint64_t mirror_bytes = 0;

    uint8_t* carve(size_t bytes) {   // under mu
        bytes = (bytes + 63) & ~(size_t) 63;
        if (chunks.empty() || chunk_used + bytes > chunk_cap) {
            chunk_cap = std::max<size_t>(bytes, (size_t) 256 << 20);
            chunks.emplace_back(new uint8_t[chunk_cap + 64]);   // not touched here: the repacking threads first-touch it
            chunk_used = 0;
        }
        uint8_t* base = (uint8_t*) (((uintptr_t) chunks.back().get() + 63) & ~(uintptr_t) 63);
        uint8_t* p = base + chunk_used;
        chunk_used += bytes;
        return p;
    }
};

Registry& reg() {
    static Registry r;
    return r;
}

// Rows 16t .. 16t+15 of the GGUF matrix at w into the mirror (all nblk blocks).  GGUF Q2_0 block: fp16 scale, then
// 16 code bytes, value 4v+j = (byte v >> 2j) & 3 - so byte v's low nibble is already the index of the pair (4v, 4v+1)
// and its high nibble that of (4v+2, 4v+3): the mirror's index byte for row lane is GGUF's code byte unchanged, and
// the repack is a transpose of 16 rows' bytes into the lanes.
void repack_tile(const uint8_t* w, size_t row_bytes, int rows, int nblk, int t, uint8_t* mir) {
    uint8_t* tile = mir + (size_t) t * nblk * REC;
    std::memset(tile, 0, (size_t) nblk * REC);
    for (int lane = 0; lane < TR; ++lane) {
        const int r = t * TR + lane;
        if (r >= rows) break;
        for (int b = 0; b < nblk; ++b) {
            const uint8_t* blk = w + (size_t) r * row_bytes + (size_t) b * BB;
            uint8_t* rec = tile + (size_t) b * REC;
            std::memcpy(rec + 2 * lane, blk, 2);
            for (int v = 0; v < 16; ++v) rec[2 * TR + 16 * v + lane] = blk[2 + v];
        }
    }
}

// The mirror of the matrix at w, or nullptr when w is not in a registered range.
Mirror* mirror_for(const uint8_t* w, size_t row_bytes, int nblk, int rows) {
    Registry& R = reg();
    std::lock_guard<std::mutex> lk(R.mu);
    const uintptr_t lo = (uintptr_t) w, hi = lo + row_bytes * (size_t) rows;
    bool stable = false;
    for (const auto& rg : R.ranges)
        if (lo >= rg.first && hi <= rg.second) { stable = true; break; }
    if (!stable) return nullptr;
    auto it = R.mirrors.find(w);
    if (it != R.mirrors.end()) {
        Mirror* m = it->second.get();
        return m->rows == rows && m->nblk == nblk ? m : nullptr;   // the same address read with another shape: decline
    }
    auto m = std::make_unique<Mirror>();
    m->rows = rows;
    m->nblk = nblk;
    m->ntiles = (rows + TR - 1) / TR;
    const size_t bytes = (size_t) m->ntiles * nblk * REC;
    m->data = R.carve(bytes);
    m->state.reset(new std::atomic<uint8_t>[(size_t) m->ntiles]);
    for (int t = 0; t < m->ntiles; ++t) m->state[(size_t) t].store(0, std::memory_order_relaxed);
    R.mirror_bytes += bytes;
    Mirror* p = m.get();
    R.mirrors.emplace(w, std::move(m));
    return p;
}

// Makes tiles [t0, t1) ready: repacks the empty ones this thread claims, waits for the ones others repack.
void ensure_tiles(Mirror& m, const uint8_t* w, size_t row_bytes, int t0, int t1) {
    bool pending = false;
    for (int t = t0; t < t1; ++t) {
        std::atomic<uint8_t>& s = m.state[(size_t) t];
        if (s.load(std::memory_order_acquire) == 2) continue;
        uint8_t empty = 0;
        if (s.compare_exchange_strong(empty, 1, std::memory_order_acq_rel)) {
            repack_tile(w, row_bytes, m.rows, m.nblk, t, m.data);
            s.store(2, std::memory_order_release);
        } else {
            pending = true;
        }
    }
    if (pending)
        for (int t = t0; t < t1; ++t)
            while (m.state[(size_t) t].load(std::memory_order_acquire) != 2) _mm_pause();
}

// This thread's tables for activation `a` of width k.
struct TableEntry {
    const ActQ* a = nullptr;
    int k = 0;
    uint64_t used = 0;
    std::vector<int8_t> q;
    std::vector<uint8_t> mem;
    uint8_t* tbl = nullptr;
};

const uint8_t* tables_for(const ActQ& a, int k) {
    thread_local std::vector<TableEntry> cache(2 * MAXT + 2);
    thread_local uint64_t clock = 0;
    ++clock;
    TableEntry* victim = &cache[0];
    for (TableEntry& e : cache) {
        if (e.a == &a && e.k == k && std::memcmp(e.q.data(), a.q, (size_t) k) == 0) {
            e.used = clock;
            return e.tbl;
        }
        if (e.used < victim->used) victim = &e;
    }
    TableEntry& e = *victim;
    e.a = &a;
    e.k = k;
    e.used = clock;
    e.q.assign(a.q, a.q + k);
    e.mem.resize((size_t) (k / 2) * TBL + 64);
    e.tbl = (uint8_t*) (((uintptr_t) e.mem.data() + 63) & ~(uintptr_t) 63);
    build_tables(a.q, k, e.tbl);
    return e.tbl;
}

}  // namespace

bool q2_lut_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_Q2_LUT");
        if (e == nullptr) return false;
        if (std::strcmp(e, "force") == 0) return cpu_avx1_ok();
        return e[0] == '1' && cpu_avx1_ok() && !cpu_avx2_ok();
    }();
    return on;
}

void q2_lut_register_stable(const void* base, uint64_t bytes) {
    if (base == nullptr || bytes == 0) return;
    Registry& R = reg();
    std::lock_guard<std::mutex> lk(R.mu);
    R.ranges.emplace_back((uintptr_t) base, (uintptr_t) base + bytes);
}

void q2_lut_unregister_stable(const void* base) {
    Registry& R = reg();
    std::lock_guard<std::mutex> lk(R.mu);
    auto it = std::find_if(R.ranges.begin(), R.ranges.end(),
                           [&](const std::pair<uintptr_t, uintptr_t>& r) { return r.first == (uintptr_t) base; });
    if (it == R.ranges.end()) return;
    const uintptr_t lo = it->first, hi = it->second;
    R.ranges.erase(it);
    for (auto m = R.mirrors.begin(); m != R.mirrors.end();) {
        const uintptr_t k = (uintptr_t) m->first;
        if (k >= lo && k < hi) {
            R.mirror_bytes -= (uint64_t) m->second->ntiles * m->second->nblk * REC;
            m = R.mirrors.erase(m);
        } else {
            ++m;
        }
    }
    if (R.mirrors.empty()) {   // chunks are shared between mirrors: freed only when none is left
        R.chunks.clear();
        R.chunk_used = R.chunk_cap = 0;
        R.mirror_bytes = 0;
    }
}

bool q2_lut_rows(const uint8_t* w, size_t row_bytes, int nblocks, int rows, const ActQ* const* a, int nt,
                 float* const* out, int r0, int r1) {
    if (!q2_lut_enabled() || nt < 1 || nt > MAXT || rows <= 0 || row_bytes != (size_t) nblocks * BB) return false;
    Mirror* m = mirror_for(w, row_bytes, nblocks, rows);
    if (m == nullptr) return false;
    r0 = std::max(r0, 0);
    r1 = std::min(r1, rows);
    if (r0 >= r1) return true;
    ensure_tiles(*m, w, row_bytes, r0 / TR, (r1 + TR - 1) / TR);
    const uint8_t* tbl[MAXT];
    for (int t = 0; t < nt; ++t) tbl[t] = tables_for(*a[t], nblocks * QK);
    if (has_f16c()) rows_f16c(m->data, nblocks, tbl, a, nt, out, r0, r1);
    else rows_nof16c(m->data, nblocks, tbl, a, nt, out, r0, r1);
    return true;
}

uint64_t q2_lut_mirror_bytes() {
    Registry& R = reg();
    std::lock_guard<std::mutex> lk(R.mu);
    return R.mirror_bytes;
}

}  // namespace strata::kernels::cpu
