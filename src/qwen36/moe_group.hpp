// moe_group.hpp - groups the (token, slot) pairs of a batch by expert so each expert runs once on all its tokens.
// CUDA-free apart from the Q36_HD marker: the same functions run on a CPU in tests/moe_group_test.cpp and inside the
// device kernel k_moe_group (qwen36_kernels.cu), so what the test checks is the code the GPU runs.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA__)
#define Q36_HD __host__ __device__
#else
#define Q36_HD
#endif

namespace q36 {

// ------------------------------------------------------------------------------------------------------------------
// Host-side grouping (layers whose experts the grouped kernel cannot take, and the CPU reference).
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

// ------------------------------------------------------------------------------------------------------------------
// Device-shaped grouping for strata::kernels::native_expert_grouped.  Three phases separated by a barrier on the
// device (the CPU test runs them one after the other):
//   A  cnt[e] = number of pairs routed to e                  (atomics on the device; sanitises invalid ids to 0)
//   B  group_offsets: one thread turns the counts into the group list - only the experts that have pairs, in
//      ascending expert order: grp_ptr[g] = blob address of the group's expert, grp_start[g .. g+1) = its entries
//   C  group_scatter: one thread per expert writes that expert's entries in ascending pair order
// The result is identical to group_pairs: entry q is pair p = (token b, slot j); ent_tok[q] = b, ent_dst[q] = q (the
// row of the output the entry writes) and pos[p] = q (where the combine step reads pair p back).

// Phase B (single thread).  off[e] = first entry of expert e (only meaningful where cnt[e] > 0).
Q36_HD inline void group_offsets(const int* cnt, int ne, unsigned long long blob_base, unsigned long long blob_stride,
                                 unsigned long long* grp_ptr, int* grp_start, int* n_groups, int* off) {
    int g = 0, o = 0;
    for (int e = 0; e < ne; ++e) {
        off[e] = o;
        if (cnt[e] > 0) {
            grp_ptr[g] = blob_base + (unsigned long long)e * blob_stride;
            grp_start[g] = o;
            ++g;
        }
        o += cnt[e];
    }
    grp_start[g] = o;
    *n_groups = g;
}

// Phase C for ONE expert e (called by the thread that owns e).
Q36_HD inline void group_scatter(int e, const int* ids, int np, int k, int start, int* ent_tok, int* ent_dst, int* pos) {
    int q = start;
    for (int p = 0; p < np; ++p)
        if (ids[p] == e) {
            ent_tok[q] = p / k;
            ent_dst[q] = q;
            pos[p] = q;
            ++q;
        }
}

}  // namespace q36
