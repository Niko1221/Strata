// CPU-only checks of the shared KV pool's logic (--kv-pool-tokens), no GPU and no CUDA:
//   * KvChunkMap: a fresh pool hands out chunk 0, 1, 2 .., a failed reserve changes nothing, shrink gives the tail back, swap exchanges two
//     lanes' tables in place, every chunk is free or held exactly once (a random walk), the rest of a table is trash;
//   * KvHostPools::block_host / contiguous / for_each_piece: a session's block b lands in the table's chunk, and a byte
//     range is cut where a chunk ends (every byte checked against the block mapping);
//   * kv_pool_eviction_plan: who gives way when the pool is full, and that nothing is given back when it would not help.
#include "strata/core/kv_chunks.hpp"
#include "strata/core/kv_pool_policy.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using strata::core::KvChunkMap;
using strata::core::KvPoolLane;
using strata::kernels::KvHostPools;

namespace {

int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                     \
            ++failures;                                                                              \
        }                                                                                            \
    } while (0)

std::vector<int32_t> table_of(const KvChunkMap& m, KvChunkMap::Lane l) {
    return std::vector<int32_t>(m.table(l), m.table(l) + m.logical(l));
}

void chunk_map_basics() {
    KvChunkMap m;
    CHECK(!m.active());
    m.init(8);
    CHECK(m.active() && m.free_chunks() == 8 && m.trash() == 8);
    const auto a = m.add_lane(4), b = m.add_lane(4), c = m.add_lane(6);
    CHECK(table_of(m, a) == std::vector<int32_t>(4, 8));
    CHECK(m.held(a) == 0);

    bool changed = false;
    CHECK(m.reserve(a, 3, &changed) && changed);
    CHECK((table_of(m, a) == std::vector<int32_t>{0, 1, 2, 8}));   // the lowest chunks first
    CHECK(m.held(a) == 3 && m.free_chunks() == 5);
    const uint64_t v = m.version();
    CHECK(m.reserve(a, 2, &changed) && !changed && m.version() == v);   // already held: nothing changes
    CHECK(m.reserve(a, 3, &changed) && !changed);

    CHECK(m.reserve(b, 2, &changed) && changed);
    CHECK((table_of(m, b) == std::vector<int32_t>{3, 4, 8, 8}));

    // not enough free chunks: false and nothing changed (not even partly)
    const auto before_c = table_of(m, c);
    const int64_t free_before = m.free_chunks();
    const uint64_t ver_before = m.version();
    CHECK(!m.reserve(c, 6, &changed) && !changed);   // needs 6, 3 are free
    CHECK(table_of(m, c) == before_c && m.held(c) == 0 && m.free_chunks() == free_before && m.version() == ver_before);
    CHECK(m.reserve(c, 3) && m.free_chunks() == 0);
    CHECK(!m.reserve(b, 3));                                   // the pool is empty

    // a request past the table is clamped to it
    KvChunkMap n;
    n.init(10);
    const auto d = n.add_lane(3);
    CHECK(n.reserve(d, 99) && n.held(d) == 3 && n.free_chunks() == 7);

    // shrink gives the tail back (the table's entries past it name the trash chunk again)
    CHECK(m.shrink(a, 1));
    CHECK((table_of(m, a) == std::vector<int32_t>{0, 8, 8, 8}));
    CHECK(m.held(a) == 1 && m.free_chunks() == 2);
    CHECK(!m.shrink(a, 1) && !m.shrink(a, 5));   // nothing to give back
    CHECK(m.reserve(b, 4));                      // b takes what a gave back (chunks 1 and 2)
    CHECK((table_of(m, b) == std::vector<int32_t>{3, 4, 1, 2}));
    CHECK(m.shrink(b, 0) && m.held(b) == 0 && table_of(m, b) == std::vector<int32_t>(4, 8) && m.free_chunks() == 4);
    CHECK(m.reserve(a, 2) && m.held(a) == 2 && m.free_chunks() == 3);
    CHECK(table_of(m, a)[0] == 0 && table_of(m, a)[1] >= 1 && table_of(m, a)[1] <= 4);   // one of b's

    // swap: tables and held counts exchange, the buffers stay where they are
    const int32_t* pa = m.table(a);
    const int32_t* pc = m.table(c);
    CHECK(!m.swap(a, a));
    CHECK(!m.swap(a, c));   // different logical sizes
    const auto e = m.add_lane(4);
    CHECK(m.reserve(e, 1));
    const auto ta = table_of(m, a), te = table_of(m, e);
    const int64_t ha = m.held(a), he = m.held(e);
    CHECK(m.swap(a, e));
    CHECK(table_of(m, a) == te && table_of(m, e) == ta && m.held(a) == he && m.held(e) == ha);
    CHECK(m.table(a) == pa && m.table(c) == pc);   // pointers the kernels hold are still valid
}

// every chunk is free or held by exactly one lane, and what a lane does not hold names the trash chunk
void chunk_map_invariants(const KvChunkMap& m, const std::vector<KvChunkMap::Lane>& lanes) {
    std::multiset<int32_t> seen;
    int64_t held = 0;
    for (const auto l : lanes) {
        held += m.held(l);
        for (int64_t i = 0; i < m.logical(l); ++i) {
            const int32_t c = m.table(l)[i];
            if (i < m.held(l)) {
                CHECK(c >= 0 && c < m.n_chunks());
                seen.insert(c);
            } else {
                CHECK(c == m.trash());
            }
        }
    }
    CHECK(held + m.free_chunks() == m.n_chunks());
    for (int32_t c = 0; c < (int32_t) m.n_chunks(); ++c) CHECK(seen.count(c) <= 1);
    CHECK((int64_t) seen.size() == held);
}

void chunk_map_random_walk() {
    std::mt19937 rng(1011);
    for (int round = 0; round < 20; ++round) {
        KvChunkMap m;
        m.init(1 + (int64_t) (rng() % 40));
        std::vector<KvChunkMap::Lane> lanes;
        const int n_lanes = 2 + (int) (rng() % 5);
        for (int i = 0; i < n_lanes; ++i) lanes.push_back(m.add_lane(8));
        for (int step = 0; step < 400; ++step) {
            const auto l = lanes[rng() % lanes.size()];
            switch (rng() % 4) {
            case 0: {
                const int64_t before_free = m.free_chunks(), held = m.held(l);
                const int64_t need = (int64_t) (rng() % 10);
                const int64_t want = std::min<int64_t>(std::max(need, held), 8);
                const bool ok = m.reserve(l, need);
                CHECK(ok == (want - held <= before_free));
                if (!ok) CHECK(m.held(l) == held && m.free_chunks() == before_free);
                break;
            }
            case 1: m.shrink(l, (int64_t) (rng() % 9)); break;
            case 2: m.swap(l, lanes[rng() % lanes.size()]); break;
            default: m.shrink(l, 0); break;
            }
            chunk_map_invariants(m, lanes);
        }
    }
}

// The reference for a session's byte p of an array with `bb` bytes per block, chunks of 1 << shift blocks.
size_t mapped_byte(const std::vector<int32_t>& table, int shift, size_t bb, size_t p) {
    const size_t block = p / bb, in = p % bb;
    const size_t phys = ((size_t) table[block >> shift] << shift) | (block & ((1u << shift) - 1));
    return phys * bb + in;
}

void host_pools_mapping() {
    const std::vector<int32_t> table = {5, 2, 7};
    KvHostPools h;
    h.chunk_host = table.data();
    h.chunk_shift = 2;   // 4 blocks per chunk
    CHECK(h.block_host(0) == 20 && h.block_host(3) == 23 && h.block_host(4) == 8 && h.block_host(6) == 10 &&
          h.block_host(11) == 31);
    CHECK(h.contiguous(0, 100) == 4 && h.contiguous(1, 100) == 3 && h.contiguous(3, 100) == 1 &&
          h.contiguous(4, 2) == 2 && h.contiguous(5, 100) == 3);
    const KvHostPools identity;   // no table: the identity layout, one piece
    CHECK(identity.block_host(77) == 77 && identity.contiguous(3, 9) == 9);

    for (const size_t bb : {size_t(1), size_t(7), size_t(256), size_t(1056)}) {
        const size_t total = 12 * bb;   // 12 blocks: 3 chunks
        for (size_t at = 0; at < total; at += 1 + total / 13)
            for (size_t n : {size_t(0), size_t(1), bb, 3 * bb + 1, total - at}) {
                if (at + n > total) continue;
                std::vector<int> hit(n, 0);
                size_t next_done = 0;
                bool ok = h.for_each_piece(bb, at, n, [&](size_t off, size_t done, size_t len) {
                    CHECK(done == next_done && len > 0);   // in order, no gaps
                    next_done = done + len;
                    // a piece never crosses a chunk of the session
                    const size_t first = (at + done) / bb, last = (at + done + len - 1) / bb;
                    CHECK((first >> 2) == (last >> 2));
                    for (size_t i = 0; i < len; ++i) {
                        CHECK(off + i == mapped_byte(table, 2, bb, at + done + i));
                        ++hit[done + i];
                    }
                    return true;
                });
                CHECK(ok && next_done == n);
                for (const int c : hit) CHECK(c == 1);
                // a callback that fails stops the walk
                int calls = 0;
                ok = h.for_each_piece(bb, at, n, [&](size_t, size_t, size_t) { ++calls; return false; });
                CHECK(n == 0 ? ok : (!ok && calls == 1));
            }
    }
    // without a table the bytes are one piece at `at`
    int pieces = 0;
    CHECK(identity.for_each_piece(16, 40, 200, [&](size_t off, size_t done, size_t len) {
        ++pieces;
        return off == 40 && done == 0 && len == 200;
    }) && pieces == 1);
}

void eviction_plan() {
    std::vector<int> v;
    // a free pool needs nobody
    CHECK(strata::core::kv_pool_eviction_plan(100, 0, 50, {}, v) && v.empty());
    CHECK(strata::core::kv_pool_eviction_plan(10, 40, 50, {}, v) && v.empty());   // what the lane holds counts
    // nothing to give: false
    CHECK(!strata::core::kv_pool_eviction_plan(10, 0, 50, {}, v) && v.empty());
    // lowest rank first, then the one used longest ago, then the lower slot; stop as soon as it is enough
    const std::vector<KvPoolLane> lanes = {
        {true, 1, 5, 10},    // 0: a cached conversation, used at 5
        {true, 0, 9, 10},    // 1: holds nothing worth keeping
        {true, 1, 2, 10},    // 2: a cached conversation, used at 2 (longer ago than 0)
        {true, 2, 1, 10},    // 3: a prompt read that gave way: last
        {false, 0, 0, 10},   // 4: active - never
        {true, 1, 2, 10},    // 5: ties with 2: the lower slot first
    };
    CHECK(strata::core::kv_pool_eviction_plan(0, 0, 25, lanes, v));
    CHECK((v == std::vector<int>{1, 2, 5}));
    CHECK(strata::core::kv_pool_eviction_plan(0, 0, 10, lanes, v) && (v == std::vector<int>{1}));
    CHECK(strata::core::kv_pool_eviction_plan(5, 0, 10, lanes, v) && (v == std::vector<int>{1}));
    CHECK(strata::core::kv_pool_eviction_plan(0, 0, 50, lanes, v));
    CHECK((v == std::vector<int>{1, 2, 5, 0, 3}));   // every slot that can give way, the read that gave way last
    // all the slots that can give together would not cover it: nobody is asked
    CHECK(!strata::core::kv_pool_eviction_plan(0, 0, 51, lanes, v) && v.empty());
    CHECK(!strata::core::kv_pool_eviction_plan(0, 0, 60, lanes, v) && v.empty());
    CHECK(strata::core::kv_pool_eviction_plan(3, 0, 53, lanes, v) && v.size() == 5);
}

}  // namespace

int main() {
    chunk_map_basics();
    chunk_map_random_walk();
    host_pools_mapping();
    eviction_plan();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("kv_chunks_test: ok");
    return 0;
}
