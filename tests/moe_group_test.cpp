// CPU-only test of the expert grouping.  The device kernel k_moe_group runs the same phases (moe_group.hpp), so this checks
// the logic the GPU uses against the host reference group_pairs.
//   g++ -std=c++17 -I../src/qwen36 moe_group_test.cpp -o moe_group_test && ./moe_group_test
#include <cstdio>
#include <random>
#include "moe_group.hpp"
using namespace q36;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

// phases A, B, C as the device runs them (A with plain increments instead of atomics)
static void device_like(std::vector<int> ids, int B, int k, int ne, unsigned long long base, unsigned long long stride,
                        std::vector<unsigned long long>& grp_ptr, std::vector<int>& grp_start, int& n_groups,
                        std::vector<int>& tok, std::vector<int>& dst, std::vector<int>& pos, int& bad) {
    const int np = B * k;
    std::vector<int> cnt(ne, 0), off(ne, 0);
    bad = 0;
    for (int p = 0; p < np; ++p) {
        if (ids[p] < 0 || ids[p] >= ne) { ids[p] = 0; bad = 1; }
        ++cnt[ids[p]];
    }
    const int cap = std::min(ne, np);
    grp_ptr.assign(cap, 0);
    grp_start.assign(cap + 1, 0);
    tok.assign(np, -1); dst.assign(np, -1); pos.assign(np, -1);
    group_offsets(cnt.data(), ne, base, stride, grp_ptr.data(), grp_start.data(), &n_groups, off.data());
    for (int e = 0; e < ne; ++e) if (cnt[e]) group_scatter(e, ids.data(), np, k, off[e], tok.data(), dst.data(), pos.data());
}

static void compare(const std::vector<int>& ids, int B, int k, int ne) {
    const int np = B * k;
    std::vector<int> cnt, off, cur, perm(np), pos_ref(np);
    group_pairs(ids.data(), B, k, ne, cnt, off, cur, perm.data(), pos_ref.data());
    std::vector<unsigned long long> gp; std::vector<int> gs, tok, dst, pos; int ng = 0, bad = 0;
    device_like(ids, B, k, ne, 1000, 7, gp, gs, ng, tok, dst, pos, bad);
    CHECK(!bad);
    CHECK(tok == perm);                     // ent_tok == perm
    CHECK(pos == pos_ref);                  // pos identical
    for (int q = 0; q < np; ++q) CHECK(dst[q] == q);
    int active = 0;
    for (int e = 0; e < ne; ++e) active += cnt[e] > 0;
    CHECK(ng == active);
    CHECK(ng <= std::min(ne, np));
    CHECK(gs[0] == 0 && gs[ng] == np);
    int g = 0;
    for (int e = 0; e < ne; ++e) {
        if (!cnt[e]) continue;
        CHECK(gp[g] == 1000ull + (unsigned long long)e * 7);
        CHECK(gs[g] == off[e] && gs[g + 1] == off[e + 1]);
        ++g;
    }
    // every pair is read back from a row written by an entry of its own expert
    for (int p = 0; p < np; ++p) {
        const int q = pos[p], e = ids[p];
        CHECK(q >= off[e] && q < off[e + 1]);
        CHECK(tok[q] == p / k);
    }
}

int main() {
    std::mt19937 rng(123);
    // random routing, several shapes (decode B=1, prefill up to 128 tokens, small and Qwen3.6-sized expert counts)
    for (int it = 0; it < 300; ++it) {
        const int B = 1 + (int)(rng() % 128), k = 1 + (int)(rng() % 8), ne = 8 + (int)(rng() % 249);
        std::vector<int> ids(B * k);
        for (int b = 0; b < B; ++b) {                       // top-k picks distinct experts per token
            std::vector<int> pool(ne);
            for (int e = 0; e < ne; ++e) pool[e] = e;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j = 0; j < k && j < ne; ++j) ids[b * k + j] = pool[j];
        }
        compare(ids, B, k, ne);
    }
    { std::vector<int> ids(8 * 128, 5); for (int p = 0; p < 1024; ++p) ids[p] = 5 + (p % 8); compare(ids, 128, 8, 256); }   // 8 hot experts
    { std::vector<int> ids = {3, 3, 3, 3}; compare(ids, 1, 4, 16); }                                                          // same expert repeated
    { std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7}; compare(ids, 1, 8, 256); }                                           // decode: 8 groups of 1
    { std::vector<int> ids = {255, 0}; compare(ids, 1, 2, 256); }                                                           // extremes
    // an invalid id is sanitised and reported (the device sets an error flag instead of throwing)
    { std::vector<int> ids = {0x7fffffff, 4}; std::vector<unsigned long long> gp; std::vector<int> gs, tok, dst, pos; int ng, bad;
      device_like(ids, 1, 2, 16, 0, 1, gp, gs, ng, tok, dst, pos, bad); CHECK(bad == 1); CHECK(ng == 2); }
    std::printf(fails ? "moe_group_test: %d FAILED\n" : "moe_group_test: all passed (%d failures)\n", fails);
    return fails != 0;
}
