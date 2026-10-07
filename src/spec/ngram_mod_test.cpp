// src/spec/ngram_mod_test.cpp - the ngram-mod table's unit tests (CPU only, no model).
//
//   ngram_mod_test            unit tests
//   ngram_mod_test --simulate [--match N] [--min M] [--k K] FILE.ids ...
//
// The simulation replays text as if a model had produced it: each step drafts up to K tokens from the n-gram table,
// accepts the prefix that matches the true next tokens, and commits that prefix plus one model token (the verify
// pass's bonus), exactly the loop the engine runs. tokens/step is the ngram-mod drafter's ceiling speedup on that
// text when a K+1-token verify costs the same as a 1-token step. The llama.cpp hash semantics are asserted against
// a straight reimplementation of PR #19164's, not trusted.
#include "strata/spec/ngram_mod.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using strata::spec::NgramMod;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

// the reference hash, straight from llama.cpp's common_ngram_mod::idx
size_t ref_idx(const std::vector<int32_t>& key, size_t entries) {
    uint64_t h = 0;
    for (size_t i = 0; i < key.size(); ++i) h = h * 6364136223846793005ull + (uint64_t) (int64_t) key[i];
    return (size_t) (h % entries);
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

void unit_tests() {
    {   // the same key always lands in the same cell, and it is llama.cpp's cell
        NgramMod m(3, 1024);
        const int32_t key[] = {7, 8, 9};
        const size_t i0 = m.index(key);
        check(i0 == ref_idx({7, 8, 9}, 1024), "index matches the reference hash");
        for (int r = 0; r < 8; ++r) check(m.index(key) == i0, "index is stable over repeats");
        // a different key hashes to the reference cell of that key (a constant hash would fail this)
        const int32_t other[] = {9, 8, 7};
        check(m.index(other) == ref_idx({9, 8, 7}, 1024), "index is the reference hash for another key too");
        check(m.index(key) == m.index(key), "index is pure");
    }
    {   // A B C -> D: after add, get reads D
        NgramMod m(3, 1024);
        const int32_t abc[] = {1, 2, 3, 4};
        m.add(abc);                                  // 1 2 3 -> 4
        check(m.get(abc) == 4, "get returns the stored continuation");
        check(m.used() == 1, "one occupied cell after one add");
    }
    {   // a key never added reads EMPTY
        NgramMod m(3, 1024);
        const int32_t xyz[] = {5, 6, 7};
        check(m.get(xyz) == NgramMod::EMPTY, "missing key reads EMPTY");
    }
    {   // an overwrite to the same cell replaces the value (the modulo-table contract)
        NgramMod m(2, 1024);
        const int32_t k1[] = {1, 2, 3};
        m.add(k1);
        check(m.get(k1) == 3, "first mapping stored");
        // find a colliding key by search (small table, the LCG is fixed): any key with the same index
        size_t target = m.index(k1);
        bool found = false;
        for (int32_t a = 100; a < 4000 && !found; ++a)
            for (int32_t b = 100; b < 4000 && !found; ++b) {
                const int32_t k2[] = {a, b, 77};
                if (m.index(k2) == target && !(a == k1[0] && b == k1[1])) {
                    m.add(k2);
                    check(m.get(k2) == 77, "colliding key stores its own continuation");
                    check(m.get(k1) == 77, "the collision overwrote the earlier value");
                    check(m.used() == 1, "an overwrite does not grow the used count");
                    found = true;
                }
            }
        check(found, "a colliding key exists in a 1024-cell table");
    }
    {   // EMPTY is not a token id: adding and getting token 0 works (0 is a real token)
        NgramMod m(1, 1024);
        const int32_t zero[] = {0, 0};
        m.add(zero);
        check(m.get(zero) == 0, "token 0 round-trips (distinct from EMPTY)");
    }
    {   // reset clears every cell and the used count
        NgramMod m(3, 1024);
        const int32_t abc[] = {1, 2, 3, 4};
        m.add(abc);
        m.reset();
        check(m.get(abc) == NgramMod::EMPTY, "reset empties the cells");
        check(m.used() == 0, "reset clears the used count");
        check(m.occupancy() == 0.0, "occupancy reads 0 after reset");
    }
    {   // occupancy counts cells, not adds: many adds to the same cell stay one used cell
        NgramMod m(1, 16);
        for (int32_t t = 0; t < 8; ++t) { const int32_t kv[] = {t, t + 1}; m.add(kv); }
        check(m.used() == 8, "distinct single-token keys occupy distinct cells");
        for (int32_t r = 0; r < 4; ++r) { const int32_t kv[] = {3, 99}; m.add(kv); }
        check(m.used() == 8, "re-adding a key leaves the used count alone");
    }
    {   // n = 1: single-token keys work (the degenerate ngram)
        NgramMod m(1, 64);
        const int32_t kv[] = {42, 43};
        m.add(kv);
        check(m.get(kv) == 43, "n=1 maps one token to its continuation");
        const int32_t other[] = {7, 0};
        check(m.get(other) == NgramMod::EMPTY, "n=1 misses an unknown token");
    }
    {   // a tiny table (4 cells) still answers, collisions included - the occupancy-reset tests build on this
        NgramMod m(2, 4);
        int stored = 0;
        for (int32_t a = 0; a < 32; ++a) {
            const int32_t kv[] = {a, a + 1, a + 2};
            const size_t before = m.used();
            m.add(kv);
            stored += m.used() > before ? 1 : 0;
        }
        check(stored <= 4 && stored >= 1, "a 4-cell table bounds the used count");
        check(m.occupancy() > 0.0, "occupancy reflects the writes");
    }
    std::printf("ngram_mod unit tests: %s\n", g_fail ? "FAILED" : "OK");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        // filled in with the drafter (commit 2): the same replay suffix_drafter_test --simulate runs
        std::fprintf(stderr, "ngram_mod_test: --simulate needs the drafter\n");
        return 2;
    }
    unit_tests();
    return g_fail ? 1 : 0;
}
