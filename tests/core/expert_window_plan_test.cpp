// tests/core/expert_window_plan_test.cpp - the verify window's GPU plan (Plan v0.3 P6), without a GPU.
//
// `detail::window_gpu_plan` (the O(n) planner `expert_pool_dispatch_multi` publishes from) is compared with
// a verbatim copy of the O(n^2) planner it replaced, on targeted and on randomized windows.  Every output is
// compared exactly: the per-entry kinds, the VRAM and PCIe group lists (pointers, starts, dst/tok entries),
// the counts and the DMA source list, because the GPU consumes all of them positionally.
//
// The corners that matter:
//   - duplicates spread across tokens (group entries must stay in routing order),
//   - the PCIe share's "last m misses in routing order" boundary, with unpinnable experts on both sides of it,
//   - the staging cap (64 and P.staging_cap) cutting the share,
//   - peer/helper-held experts (no miss, no PCIe, kind 2 / left at -1),
//   - out-of-range ids (planned, then refused by the pool loop later),
//   - pcie_mode 0/1/2 (staging addresses vs device aliases),
//   - per-slot cache offsets vs uniform slots.
#include "strata/core/expert_source.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using strata::core::GpuPlanSink;
using strata::core::detail::WindowGpuPlanInput;
using strata::core::detail::window_gpu_plan;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------- the rig ----------------------------------------------------------------

constexpr int64_t kMaxN = 128;

struct Stub {
    const int32_t* host_res = nullptr;   // n_expert entries
    int64_t n_expert = 0;
    std::vector<uint8_t> peer;           // [e] != 0: the peer GPU holds it
    std::vector<uint8_t> helper;         // [e] != 0: a helper GPU holds it
    std::vector<uint8_t> pinned;         // [e] != 0: the blob is page-locked
    const uint8_t* blob_base = nullptr;  // pinned_blob(e) = blob_base + e * blob_bytes
    const uint8_t* alias_base = nullptr; // device_alias(e) = alias_base + e * blob_bytes
    int64_t blob_bytes = 0;
};

bool stub_peer_has(void* ctx, int32_t e) { return ((Stub*) ctx)->peer[(size_t) e] != 0; }
bool stub_helper_holds(void* ctx, int32_t e) { return ((Stub*) ctx)->helper[(size_t) e] != 0; }
const uint8_t* stub_pinned_blob(void* ctx, int32_t e) {
    const Stub& s = *(const Stub*) ctx;
    return s.pinned[(size_t) e] ? s.blob_base + (size_t) e * (size_t) s.blob_bytes : nullptr;
}
const uint8_t* stub_device_alias(void* ctx, int32_t e) {
    const Stub& s = *(const Stub*) ctx;
    return s.alias_base + (size_t) e * (size_t) s.blob_bytes;
}

// A sink with host arrays, sized for the largest window.
struct Sink {
    int32_t counts[3];
    int32_t start[kMaxN + 1];
    int32_t start2[kMaxN + 1];
    int32_t dst[kMaxN];
    int32_t tok[kMaxN];
    unsigned long long ptr[kMaxN];
    unsigned long long ptr2[kMaxN];
    GpuPlanSink P;
    Sink(int64_t cap, int64_t staging_cap, int pcie_mode, unsigned long long staging) {
        std::memset(this, 0, sizeof(*this) - sizeof(P));
        P.counts = counts;
        P.start = start;
        P.dst = dst;
        P.tok = tok;
        P.ptr = ptr;
        P.ptr2 = ptr2;
        P.start2 = start2;
        P.staging = staging;
        P.staging_cap = staging_cap;
        P.cap = cap;
        P.pcie_mode = pcie_mode;
    }
};

WindowGpuPlanInput make_input(const int32_t* ids, int64_t n, int64_t k, Stub& stub, int pcie_num,
                              bool pcie_layer, const uint8_t* cache_base, int64_t cache_blob,
                              const uint64_t* cache_slot_off, int64_t* pcie_experts) {
    WindowGpuPlanInput in;
    in.ids = ids;
    in.n = n;
    in.k = k;
    in.n_expert = stub.n_expert;
    in.host_res = stub.host_res;
    in.peer_has = stub_peer_has;
    in.helper_holds = stub_helper_holds;
    in.pinned_blob = stub_pinned_blob;
    in.device_alias = stub_device_alias;
    in.ctx = &stub;
    in.pcie_num = pcie_num;
    in.pcie_layer = pcie_layer;
    in.cache_base = cache_base;
    in.cache_blob = cache_blob;
    in.cache_slot_off = cache_slot_off;
    in.pcie_experts = pcie_experts;
    return in;
}

// ------------------------------------------------------- the replaced O(n^2) planner -------------------------------------------------------
// Verbatim from expert_pool_dispatch_multi before the O(n) rewrite, reading the same input through the same
// predicates, writing the same outputs.  If this drifts from what 0.1.x shipped, the test compares nothing.
void reference_plan(const WindowGpuPlanInput& in, GpuPlanSink& P, int32_t* kind, int64_t blob_bytes,
                    const uint8_t** dma_src) {
    const int64_t n = in.n, k = in.k;
    const int32_t* ids = in.ids;
    int64_t distinct[kMaxN], first_of[kMaxN];
    int nd = 0, nmiss = 0;
    auto helper_holds = [&](int32_t e) { return in.helper_holds != nullptr && in.helper_holds(in.ctx, e); };
    for (int64_t i = 0; i < n; ++i) {
        first_of[i] = i;
        for (int64_t j = 0; j < i; ++j)
            if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
        if (first_of[i] == i) {
            distinct[nd++] = i;
            const int32_t e = ids[i];
            if (e >= 0 && e < in.n_expert && in.host_res[(size_t) e] < 0 &&
                !(in.peer_has != nullptr && in.peer_has(in.ctx, e)) && !helper_holds(e)) ++nmiss;
        }
    }
    const bool pcie_ok = in.pcie_num > 0 && in.pcie_layer;
    const int m = pcie_ok ? (nmiss * in.pcie_num) >> 8 : 0;
    int miss_rank = 0, groups = 0, entries = 0, fetches = 0;
    int64_t pcie_i0[64];
    for (int q = 0; q < nd; ++q) {
        const int64_t i0 = distinct[q];
        const int32_t e = ids[i0];
        int kd = -1;
        unsigned long long ptr = 0;
        if (e >= 0 && e < in.n_expert) {
            const int32_t slot = in.host_res[(size_t) e];
            if (slot >= 0) {
                kd = 0;
                ptr = (unsigned long long) (in.cache_base + (in.cache_slot_off ? (size_t) in.cache_slot_off[slot]
                                                                               : (size_t) slot * (size_t) in.cache_blob));
            } else if (in.peer_has != nullptr && in.peer_has(in.ctx, e)) {
                kd = 2;
            } else if (helper_holds(e)) {
            } else {
                if (miss_rank >= nmiss - m && fetches < P.staging_cap && fetches < 64) {
                    const uint8_t* src = in.pinned_blob(in.ctx, e);
                    if (src != nullptr) {
                        kd = 1;
                        dma_src[fetches] = src;
                        pcie_i0[fetches] = i0;
                        ++fetches;
                    }
                }
                ++miss_rank;
            }
        }
        for (int64_t i = i0; i < n; ++i)
            if (first_of[i] == i0) kind[i] = kd;
        if (kd != 0) continue;
        P.ptr[groups] = ptr;
        P.start[groups] = entries;
        for (int64_t i = i0; i < n; ++i)
            if (first_of[i] == i0) {
                P.dst[entries] = (int32_t) i;
                P.tok[entries] = (int32_t) (i / k);
                ++entries;
            }
        ++groups;
    }
    P.start[groups] = entries;
    for (int q = 0; q < fetches; ++q) {
        const int64_t i0 = pcie_i0[q];
        P.ptr2[q] = P.pcie_mode != 0 ? (unsigned long long) in.device_alias(in.ctx, ids[i0])
                                     : P.staging + (unsigned long long) q * (unsigned long long) blob_bytes;
        P.start2[q] = entries;
        for (int64_t i = i0; i < n; ++i)
            if (first_of[i] == i0) {
                P.dst[entries] = (int32_t) i;
                P.tok[entries] = (int32_t) (i / k);
                ++entries;
            }
        if (in.pcie_experts != nullptr) *in.pcie_experts += 1;
    }
    P.start2[fetches] = entries;
    P.counts[0] = groups;
    P.counts[1] = entries;
    P.counts[2] = fetches;
}

// -------------------------------------------------------------- comparison --------------------------------------------------------------

bool same_plan(const GpuPlanSink& A, const GpuPlanSink& B, const int32_t* kind_a, const int32_t* kind_b,
               const uint8_t* const* dma_a, const uint8_t* const* dma_b, int64_t n, std::string& why) {
    for (int64_t i = 0; i < n; ++i)
        if (kind_a[i] != kind_b[i]) { why = "kind[" + std::to_string(i) + "]"; return false; }
    for (int c = 0; c < 3; ++c)
        if (A.counts[c] != B.counts[c]) { why = "counts[" + std::to_string(c) + "]"; return false; }
    const int groups = A.counts[0], entries = A.counts[1], fetches = A.counts[2];
    for (int g = 0; g <= groups; ++g) {
        if (A.start[g] != B.start[g]) { why = "start[" + std::to_string(g) + "]"; return false; }
        if (g < groups && A.ptr[g] != B.ptr[g]) { why = "ptr[" + std::to_string(g) + "]"; return false; }
    }
    for (int q = 0; q <= fetches; ++q) {
        if (A.start2[q] != B.start2[q]) { why = "start2[" + std::to_string(q) + "]"; return false; }
        if (q < fetches && A.ptr2[q] != B.ptr2[q]) { why = "ptr2[" + std::to_string(q) + "]"; return false; }
        if (q < fetches && dma_a[q] != dma_b[q]) { why = "dma_src[" + std::to_string(q) + "]"; return false; }
    }
    for (int i = 0; i < entries; ++i) {
        if (A.dst[i] != B.dst[i]) { why = "dst[" + std::to_string(i) + "]"; return false; }
        if (A.tok[i] != B.tok[i]) { why = "tok[" + std::to_string(i) + "]"; return false; }
    }
    return true;
}

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
uint64_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// One window through both planners; returns false (and fills `why`) on any difference.
bool run_case(const char* name, int64_t n_tok, int64_t k, int64_t n_expert, double resident_frac,
              double pinned_frac, bool with_peer, bool with_helper, int pcie_num, bool pcie_layer,
              int64_t staging_cap, int pcie_mode, bool slot_off, double dup_frac, bool invalid_ids) {
    const int64_t n = n_tok * k;
    Stub stub;
    stub.n_expert = n_expert;
    stub.blob_bytes = 4096;
    std::vector<uint8_t> blob_mem((size_t) n_expert * 4096), alias_mem((size_t) n_expert * 4096),
        cache_mem((size_t) n_expert * 4096);
    stub.blob_base = blob_mem.data();
    stub.alias_base = alias_mem.data();
    std::vector<int32_t> host_res((size_t) n_expert);
    std::vector<uint64_t> offs((size_t) n_expert);
    stub.peer.assign((size_t) n_expert, 0);
    stub.helper.assign((size_t) n_expert, 0);
    stub.pinned.assign((size_t) n_expert, 0);
    for (int64_t e = 0; e < n_expert; ++e) {
        const double r = (rnd() % 1000) / 1000.0;
        if (r < resident_frac)
            host_res[(size_t) e] = (int32_t) (e % 97);   // some slot
        else if (with_peer && r < resident_frac + 0.08) {
            host_res[(size_t) e] = -1;
            stub.peer[(size_t) e] = 1;
        } else if (with_helper && r < resident_frac + 0.16) {
            host_res[(size_t) e] = -1;
            stub.helper[(size_t) e] = 1;
        } else {
            host_res[(size_t) e] = -1;
            if ((rnd() % 1000) / 1000.0 < pinned_frac) stub.pinned[(size_t) e] = 1;
        }
        offs[(size_t) e] = (uint64_t) e * 3331;   // irregular per-slot offsets
    }
    stub.host_res = host_res.data();
    std::vector<int32_t> ids((size_t) n);
    const int64_t dup_pool = dup_frac > 0 ? 1 + (int64_t) (n_expert * dup_frac) : n_expert;
    for (int64_t i = 0; i < n; ++i) {
        ids[(size_t) i] = (int32_t) (rnd() % (uint64_t) (dup_pool < n_expert ? dup_pool : n_expert));
        if (invalid_ids && (rnd() % 32) == 0)
            ids[(size_t) i] = (rnd() & 1) ? -1 : (int32_t) n_expert + (int32_t) (rnd() % 7);
    }
    Sink A(kMaxN, staging_cap, pcie_mode, 0x5000000), B(kMaxN, staging_cap, pcie_mode, 0x5000000);
    int32_t kind_a[kMaxN], kind_b[kMaxN];
    const uint8_t* dma_a[64];
    const uint8_t* dma_b[64];
    int64_t pcie_a = 0, pcie_b = 0;
    WindowGpuPlanInput in = make_input(ids.data(), n, k, stub, pcie_num, pcie_layer, cache_mem.data(), 4096,
                                       slot_off ? offs.data() : nullptr, &pcie_a);
    WindowGpuPlanInput in_b = in;
    in_b.pcie_experts = &pcie_b;
    window_gpu_plan(in, A.P, kind_a, stub.blob_bytes, dma_a);
    reference_plan(in_b, B.P, kind_b, stub.blob_bytes, dma_b);
    std::string why;
    const bool same = same_plan(A.P, B.P, kind_a, kind_b, dma_a, dma_b, n, why) && pcie_a == pcie_b;
    check(same, std::string(name) + (same ? "" : " (first difference: " + why + ")"));
    return same;
}

}  // namespace

int main() {
    std::printf("expert_window_plan_test: the O(n) window planner against the O(n^2) one it replaced\n");
    int ran = 0;
    auto c = [&](const char* name, int64_t n_tok, int64_t k, int64_t n_expert, double res, double pin,
                 bool peer, bool helper, int pcie_num, bool pcie_layer, int64_t staging_cap, int pcie_mode,
                 bool slot_off, double dup, bool invalid) {
        run_case(name, n_tok, k, n_expert, res, pin, peer, helper, pcie_num, pcie_layer, staging_cap, pcie_mode,
                 slot_off, dup, invalid);
        ++ran;
    };
    // targeted corners
    c("one token, all misses, no PCIe", 1, 10, 512, 0.0, 0.0, false, false, 79, true, 16, 0, false, 0.0, false);
    c("all experts resident (no miss at all)", 8, 10, 512, 1.0, 0.0, false, false, 79, true, 16, 0, false, 0.0, false);
    c("all misses pinned, full share (pcie_num 255)", 8, 10, 512, 0.0, 1.0, false, false, 255, true, 64, 0, false, 0.0, false);
    c("all misses, PCIe off (pcie_num 0)", 8, 10, 512, 0.0, 1.0, false, false, 0, true, 64, 0, false, 0.0, false);
    c("all misses, layer refuses PCIe", 8, 10, 512, 0.0, 1.0, false, false, 79, false, 64, 0, false, 0.0, false);
    c("staging cap cuts the share", 8, 10, 512, 0.0, 1.0, false, false, 255, true, 3, 0, false, 0.0, false);
    c("unpinnable misses inside the share", 8, 10, 512, 0.0, 0.5, false, false, 200, true, 64, 0, false, 0.0, false);
    c("one distinct expert for the whole window", 8, 10, 4, 0.0, 1.0, false, false, 255, true, 64, 0, false, 0.0, false);
    c("every entry distinct (n distinct)", 4, 10, 512, 0.3, 0.5, false, false, 79, true, 64, 0, false, 0.0, false);
    c("heavy duplicates, boundary through a duplicate run", 8, 10, 6, 0.1, 0.7, false, false, 128, true, 64, 0, false, 0.5, false);
    c("peer holds some misses", 8, 10, 512, 0.2, 0.5, true, false, 128, true, 64, 0, false, 0.0, false);
    c("a helper GPU holds some misses", 8, 10, 512, 0.2, 0.5, false, true, 128, true, 64, 0, false, 0.0, false);
    c("peer and helper together", 8, 10, 512, 0.2, 0.5, true, true, 128, true, 64, 0, false, 0.1, false);
    c("out-of-range ids mixed in", 8, 10, 64, 0.3, 0.5, false, false, 128, true, 64, 0, false, 0.0, true);
    c("pcie_mode 1 (device aliases)", 8, 10, 512, 0.1, 1.0, false, false, 255, true, 64, 1, false, 0.0, false);
    c("pcie_mode 2 (copy kernel, aliases)", 8, 10, 512, 0.1, 1.0, false, false, 255, true, 64, 2, false, 0.0, false);
    c("per-slot cache offsets", 8, 10, 512, 0.6, 0.5, false, false, 128, true, 64, 0, true, 0.0, false);
    c("k = 1", 8, 1, 512, 0.3, 0.5, false, false, 128, true, 64, 0, false, 0.0, false);
    c("k = 2, tiny expert count", 8, 2, 8, 0.25, 0.5, true, true, 128, true, 64, 0, false, 0.3, false);
    c("m = 0 with misses (small nmiss)", 1, 2, 512, 0.0, 1.0, false, false, 79, true, 64, 0, false, 0.0, false);
    // randomized sweep
    for (int t = 0; t < 4000; ++t) {
        const int64_t n_tok = 1 + (int64_t) (rnd() % 8);
        const int64_t k = 1 + (int64_t) (rnd() % 10);
        const int64_t n_expert = 1 + (int64_t) (rnd() % 512);
        run_case("randomized window", n_tok, k, n_expert, (rnd() % 1001) / 1000.0, (rnd() % 1001) / 1000.0,
                 (rnd() & 1) != 0, (rnd() & 1) != 0, (int) (rnd() % 256), (rnd() & 1) != 0,
                 (int64_t) (rnd() % 65), (int) (rnd() % 3), (rnd() & 1) != 0, (rnd() % 1001) / 1000.0,
                 (rnd() % 4) == 0);
        ++ran;
    }
    std::printf("  %d cases\n", ran);
    if (g_fail != 0) {
        std::printf("expert_window_plan_test: %d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("expert_window_plan_test: all ok\n");
    return 0;
}
