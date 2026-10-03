// moe_group.hpp - groups the (token, slot) pairs of a batch by expert so each expert runs once on all its tokens.
// CUDA-free: unit-tested on a CPU (tests/moe_group_test.cpp).
#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace q36 {

// ids[b*k + j] = expert of token b, slot j.  Output:
//   off[e] .. off[e+1]  : the sorted positions of expert e (off has ne+1 entries)
//   perm[q]             : token of the q-th sorted pair            (input rows to gather)
//   pos[b*k + j]        : sorted position of pair (b, j)           (where to read its result back)
inline void group_pairs(const int* ids, int B, int k, int ne, std::vector<int>& cnt, std::vector<int>& off,
                        std::vector<int>& cur, int* perm, int* pos) {
    cnt.assign(ne, 0);
    off.assign(ne + 1, 0);
    cur.assign(ne, 0);
    const int np = B * k;
    for (int p = 0; p < np; ++p) {
        if (ids[p] < 0 || ids[p] >= ne) throw std::runtime_error("router returned an invalid expert id");
        ++cnt[ids[p]];
    }
    for (int e = 0; e < ne; ++e) { off[e + 1] = off[e] + cnt[e]; cur[e] = off[e]; }
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < k; ++j) {
            const int p = b * k + j, q = cur[ids[p]]++;
            perm[q] = b;
            pos[p] = q;
        }
}

}  // namespace q36
