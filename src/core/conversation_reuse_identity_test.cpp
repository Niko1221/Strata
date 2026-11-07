// src/core/conversation_reuse_identity_test.cpp - the byte-identity half of the reuse contract.
//
// A capture may keep a prefix's K/V bytes instead of recopying them (conversation_kv_save's
// unchanged_tokens).  The snapshot API contract says:
//
//   A nonzero unchanged_tokens is valid only for storage retained from an image actually restored into this
//   session, bounded by every subsequent rewrite.  Equal token IDs alone do not establish that its K/V
//   bytes are unchanged.
//
// The bookkeeping half of that - which bytes are claimed unchanged, and where the claim stops - is already
// covered in conversation_cache_test.cpp (retain / limit_reuse / take_reuse).  This file covers the half
// that is not: whether a prefix's bytes SURVIVE a rewrite, and whether two different prefixes can be
// told apart.  A claim of "unchanged" is only meaningful if the bytes really are unchanged.
//
// The engine-level half of the question - two independent computations of the same token prefix, through
// different chunkings, producing the same bytes - needs a real model and is measured with
// STRATA_STATE_HASH=1, not here.  What this file can decide is whether the claim is safe in the shapes the
// cache actually produces: a tail rewrite leaves the prefix alone, a rewrite from the start drops the whole
// donation, and a different prefix gives different bytes.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

// One session's K/V state, with the fixture's own buffers.
struct State {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device;
    std::array<void*, 5> sources{};
    std::array<size_t, 5> sizes{};

    template <class T> void alloc(T*& p, size_t n) {
        if (!n) return;
        void* raw = nullptr;
        cuda_check(cudaMalloc(&raw, n));
        device.push_back(raw);
        p = static_cast<T*>(raw);
    }
    State() {
        auto& st = state;
        st.kv_mode = 0; st.kv_int8 = false; st.kv_q4 = false; st.kv_hybrid = false;
        st.n_pages = 32; st.n_slots = st.n_pages;
        st.max_cells = st.n_pages * 4;
        st.idx_pooled_rows = st.max_cells / 4 + 2;
        g.n_layers = 4; g.n_expert = 256;
        g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
        const size_t per = g.head_dim * 2;   // F16 K/V
        const size_t rows = st.max_cells * g.n_head_kv;
        sizes = {rows * per, rows * per, 0, 0, size_t(st.idx_pooled_rows) * g.idx_key_dim * 4};
        alloc(st.k_pool, sizes[0]);
        alloc(st.v_pool, sizes[1]);
        alloc(st.idx_pooled, sizes[4]);
        sources[0] = st.k_pool; sources[1] = st.v_pool; sources[4] = st.idx_pooled;
        alloc(st.page_table, st.n_pages * 4);
    }
    size_t per_token() const { return sizes[0] / size_t(state.max_cells); }
    // Write `tokens` tokens' worth of bytes with a given salt.  The byte at an offset depends only on the
    // offset and the salt, so the same token prefix always gets the same bytes.
    void write_prefix(uint8_t salt, int64_t tokens) {
        for (size_t i = 0; i < sources.size(); ++i) {
            if (!sizes[i]) continue;
            const size_t n = std::min(sizes[i], per_token() * size_t(tokens));
            if (!n) continue;
            std::vector<uint8_t> data(n);
            for (size_t j = 0; j < n; ++j) data[j] = (uint8_t)(salt + i * 31 + j * 7 + j / 257);
            cuda_check(cudaMemcpy(sources[i], data.data(), n, cudaMemcpyDefault));
        }
    }
    // Rewrite the bytes from `first_dirty` on: what a rewind and a replay do to the tail.
    void rewrite_tail(uint8_t salt, int64_t first_dirty) {
        for (size_t i = 0; i < sources.size(); ++i) {
            if (!sizes[i]) continue;
            const size_t offset = std::min(sizes[i], per_token() * size_t(first_dirty));
            if (sizes[i] <= offset) continue;
            cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i]) + offset, salt, sizes[i] - offset));
        }
    }
    ~State() { for (void* p : device) cudaFree(p); }
};

// The bytes of an image's first `upto` tokens, read back from the host image.  `per_token` is how many
// bytes of a buffer one token takes (the K/V buffers are cell-major: the cells come first).
bool image_prefix_equals(const ConversationKv& a, const ConversationKv& b, int64_t upto, size_t per_token,
                         std::string& why) {
    const std::array<const ConversationBuffer*, 5> A = {&a.k, &a.v, &a.k_scale, &a.v_scale, &a.pooled};
    const std::array<const ConversationBuffer*, 5> B = {&b.k, &b.v, &b.k_scale, &b.v_scale, &b.pooled};
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i]->empty() && B[i]->empty()) continue;
        if (A[i]->empty() || B[i]->empty()) { why = "one side's buffer is empty"; return false; }
        const size_t n = std::min(A[i]->size(), B[i]->size());
        // The indexer's pooled rows are not cell-major: they are rewritten as a whole, so compare them whole.
        const size_t take = (i == 4) ? n : std::min(n, per_token * size_t(upto));
        if (!take) continue;
        std::vector<uint8_t> x(take), y(take);
        if (!A[i]->read(x.data(), 0, take) || !B[i]->read(y.data(), 0, take)) { why = "read failed"; return false; }
        if (std::memcmp(x.data(), y.data(), take) != 0) { why = "the prefix bytes differ"; return false; }
    }
    return true;
}
}  // namespace

int main() {
    std::string err;

    // ---- 1. A tail rewrite leaves the prefix alone, and changes the tail ---------------------------
    // This is what the unchanged_tokens claim rests on: the bytes before the first rewrite are still the
    // ones the session holds, so a capture may keep them.
    {
        State a, b;
        a.write_prefix(3, 32);
        b.write_prefix(3, 32);
        ConversationKv whole, before;
        check(conversation_kv_save(whole, a.state, a.g, 32, true, err), "save the whole state");
        check(conversation_kv_save(before, b.state, b.g, 32, true, err), "save the state before the rewrite");
        b.rewrite_tail(0x5A, 20);
        ConversationKv after;
        check(conversation_kv_save(after, b.state, b.g, 32, true, err), "save after the tail rewrite");
        std::string why;
        check(image_prefix_equals(before, after, 20, b.per_token(), why),
              "a tail rewrite leaves the prefix bytes alone");
        if (!why.empty()) std::fprintf(stderr, "  (%s)\n", why.c_str());
        // The bytes past the rewrite must differ, or the bound would be meaningless.
        const size_t per = b.per_token();
        std::vector<uint8_t> x(per), y(per);
        check(before.k.read(x.data(), per * 24, per), "read the tail before the rewrite");
        check(after.k.read(y.data(), per * 24, per), "read the tail after the rewrite");
        check(std::memcmp(x.data(), y.data(), per) != 0, "a tail rewrite changes the tail bytes");
    }

    // ---- 2. A different prefix gives different bytes (the negative control) -------------------------
    // Without this, case 1 would prove nothing: identical bytes could just mean the comparison is too
    // coarse to see a difference.
    {
        State a, c;
        a.write_prefix(7, 24);
        c.write_prefix(0x33, 24);
        ConversationKv ia, ic;
        check(conversation_kv_save(ia, a.state, a.g, 24, true, err), "save A");
        check(conversation_kv_save(ic, c.state, c.g, 24, true, err), "save C");
        std::string why;
        check(!image_prefix_equals(ia, ic, 24, a.per_token(), why),
              "a different prefix gives different K/V bytes");
    }

    // ---- 3. The claim's bound: where "unchanged" stops ----------------------------------------------
    {
        ConversationCache cache(1 << 20, 4);
        std::vector<ConversationKv> layers(2);
        layers[0].k.resize(1024, 1);
        layers[1].k.resize(1024, 2);
        cache.retain(std::move(layers), 16);
        // The caller's bound: limit_reuse(read_from), read_from being where this request rewrites.
        cache.limit_reuse(6);
        auto bounded = cache.take_reuse();
        check(bounded.unchanged_tokens == 6, "the claim stops at the first rewrite");
        check(bounded.kv.size() == 2, "the donated buffers come with it");
        // A rewrite from the start drops the whole donation: nothing is claimed unchanged.
        cache.retain(std::move(bounded.kv), 6);
        cache.limit_reuse(0);
        check(cache.take_reuse().kv.empty(), "a rewrite from the start drops every donation");
        // And a later continuation cannot widen a claim that was already cut.
        cache.retain(std::move(bounded.kv), 6);
        cache.limit_reuse(3);
        cache.limit_reuse(9);
        auto narrowed = cache.take_reuse();
        check(narrowed.unchanged_tokens == 3, "a later continuation cannot undo a rewind's dirty boundary");
    }

    // ---- 4. The claim survives the capture: the kept bytes are the donated ones --------------------
    // A capture with a nonzero unchanged_tokens must keep the donated prefix bytes, not the device's.
    // The API is written for exactly this: the image is the donated one (it already holds the K/V), and
    // the save fills the rest from the device while leaving the claimed prefix alone.
    {
        State a;
        a.write_prefix(5, 24);
        ConversationKv donated;
        check(conversation_kv_save(donated, a.state, a.g, 24, true, err), "save the donated image");
        // The device's state is rewritten from token 12 on: the capture must not take those bytes.
        a.rewrite_tail(0x77, 12);
        ConversationKv kept = donated;   // the donated buffers, as the capture's reuse
        size_t reused = 0;
        if (!conversation_kv_save(kept, a.state, a.g, 24, true, err, 16, &reused))
            std::fprintf(stderr, "  (save with unchanged=16 failed: %s)\n", err.c_str());
        check(err.empty(), "capture with a nonzero unchanged prefix");
        check(reused > 0, "the capture reports the bytes it did not copy");
        std::string why;
        check(image_prefix_equals(donated, kept, 16, a.per_token(), why),
              "the kept prefix bytes are the donated ones");
        if (!why.empty()) std::fprintf(stderr, "  (%s)\n", why.c_str());
        // Past the claimed prefix the capture takes the device's bytes, so they must differ.
        const size_t per = a.per_token();
        std::vector<uint8_t> x(per), y(per);
        check(donated.k.read(x.data(), per * 20, per), "read the donated tail");
        check(kept.k.read(y.data(), per * 20, per), "read the captured tail");
        check(std::memcmp(x.data(), y.data(), per) != 0, "past the claimed prefix the capture takes the device's bytes");
    }

    std::printf("conversation_reuse_identity_test: %d checks passed\n", checks);
    return 0;
}
