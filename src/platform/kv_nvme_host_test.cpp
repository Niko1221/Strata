// src/platform/kv_nvme_host_test.cpp - the NVMe cold tier's FORMAT and RESUME rules, on the host, with NO CUDA
// context (docs/nvme-kv-cache-design.md step 4: the fixtures C1 / C2 / C3 / C4 owed).
//
// WHAT THIS IS.  `nvme_dump_at`, `nvme_restore` and `KvNvmeStore` are the tier, and until now the only tests that
// exercise them are three shell scripts that need a 24 GB model and the RTX 4090.  This fixture drives those same
// functions with a synthetic session whose "device" arrays are ordinary host buffers, so the envelope's rules -
// the turn-boundary key, the image filter, the pooled-row count, the spare-row re-publish, the version and
// geometry refusals, the drafter-ring window - are checkable on a machine running no model at all.
//
// HOW THE DEVICE IS STOOD IN.  The tier touches the device in five ways: `cudaMemcpy` for the indexer / gdn / ple
// segments, `cudaMemcpyAsync` in the drafter-ring refill, `kv_stream_reset`'s kernel launch, a
// `cudaDeviceSynchronize` before the writes and one after them.  The target links with `-Wl,--wrap=...` - the
// shared core's own `conversation_transfer_test` uses the same technique - and the wrappers below turn the copies
// into plain `memcpy`, let either sync be made to fail on a chosen call number, and MODEL the runtime's last-error
// state (returned once, then cleared) instead of suppressing it.  `main` also forces `CUDA_VISIBLE_DEVICES=-1`
// before any CUDA call, so this fixture cannot reach a GPU even if it is run by hand on a busy machine.
//
// WHAT IT PROVES: the bytes and the decisions.  Every segment the tier writes and reads, the arithmetic that
// sizes them, the key a snapshot is stored under, the match that promotes one, which refusal fires first, and -
// since step 5 - WHICH KIND of failure each one is: the recoverable class, which provably makes no CUDA call and
// writes nothing, versus the transfer class, which provably leaves a half-written session and must not be
// recovered from.  It also models the CUDA last-error state, so a tier that handles a copy failure without
// consuming the error is caught here rather than by the next kernel launch.
//
// WHAT IT CANNOT PROVE, and no host fixture can:
//   * that the arrays are really device memory or that `cudaMemcpy` moves them - the wrappers replace it, so a
//     wrong `cudaMemcpyKind` would pass here;
//   * that `kv_stream_reset` refills the streamed layers' slots: its launch is a no-op, so the fixture asserts
//     nothing about a page table;
//   * whether a real failed H2D copy leaves a STICKY context error or a benign one.  That is the question the
//     failure contract refuses to guess at: the tier classifies the failure and stops, rather than claiming the
//     device is fine.  Only a GPU run can show what a real failure does, and the GPU is held by a live engine;
//   * that a promoted session generates the same TOKENS as the session that was dumped.  That oracle is
//     `STRATA_STATE_HASH` in tools/nvme_p0_test.sh, and the fingerprint lives in the program layer;
//   * a real model's geometry, a real tokenizer, or a conversation over HTTP.

#include "strata/platform/kv_nvme.hpp"

#include "strata/core/conversation_snapshot.hpp"   // conversation_state_sizes, ConversationCheckpoint
#include "strata/kernels/kv_stream.hpp"            // KvFormat, kKvCtlInts, the ring refill the restore performs
#include "strata/kernels/qsa.hpp"                  // qsa_real_shapes, qsa_pooled_rows

#include <cuda_runtime.h>

#include <cstring>

// ---- the host stand-in for the device (the target links -Wl,--wrap=<name>; see the file header) ----
//
// THE CUDA ERROR STATE IS MODELLED, NOT SUPPRESSED.  `cudaGetLastError()` returns the last error AND CLEARS it,
// and that is load-bearing for the failure contract (C7): a tier that handles a copy failure without reading the
// error leaves it for the NEXT `cudaGetLastError()` in the engine - and on the serve loop's drop-and-restart path
// that next call is `kv_stream_reset`'s `check()` (kv_stream.cu:199-202), which EXITS the process.  So these
// wrappers record what a failed copy set and hand it back exactly once, the way CUDA does.  A restore that
// consumed its own error reads back "no error"; one that did not leaves a pending error this fixture can see.
namespace {
int copy_calls = 0, sync_calls = 0;
int fail_copy = 0, fail_sync = 0;              // the Nth cudaMemcpy / cudaDeviceSynchronize fails (0 = never)
cudaError_t pending_error = cudaSuccess;       // what a real `cudaGetLastError()` would be reporting
void reset_faults() { copy_calls = sync_calls = fail_copy = fail_sync = 0; pending_error = cudaSuccess; }
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    ++copy_calls;
    if (fail_copy != 0 && copy_calls == fail_copy) { pending_error = cudaErrorInvalidValue; return pending_error; }
    if (!dst || !src) { pending_error = cudaErrorInvalidValue; return pending_error; }
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    // Never injected: a failure here goes to `kv_ring_restore`'s `check("ring restore")` (kv_stream.cu:223-234),
    // which exits the process - the fixture would end instead of asserting.  It is asserted in the contract text
    // instead: a residency refill that cannot launch is fatal by construction, not by choice.
    if (!dst || !src) { pending_error = cudaErrorInvalidValue; return pending_error; }
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    ++sync_calls;
    if (fail_sync != 0 && sync_calls == fail_sync) { pending_error = cudaErrorUnknown; return pending_error; }
    return cudaSuccess;
}
/// Faithful: returns the pending error AND clears it.  `kv_stream_reset` launches a real kernel, which with no
/// device only SETS an error; because this wrapper answers for the runtime's own state, `check()` sees what the
/// tier left behind - which is exactly the thing C7 turns on.
extern "C" cudaError_t __wrap_cudaGetLastError() {
    const cudaError_t e = pending_error;
    pending_error = cudaSuccess;
    return e;
}

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace strata::core;
using strata::platform::NvmeEntry;
using strata::platform::NvmeHeader;

namespace {

int checks = 0, refusals = 0, fatalities = 0;
std::string last_error;

void ck(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
void ck_eq(int64_t got, int64_t want, const char* what) {
    ++checks;
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s (got %lld, want %lld)\n", what, (long long) got, (long long) want);
        std::exit(1);
    }
}
/// THE FLOOR EVERY FAILURE PATH MUST CLEAR: the tier reported its own failure, so it must have CONSUMED the CUDA
/// error it handled.  A pending error here is a restore that would make the next `cudaGetLastError()` in the
/// engine - `kv_stream_reset`'s `check()`, which exits - report this failure as if it had happened there.
/// (`pinned.cu:170-183` is this tree paying for that exact trap before.)
void ck_no_pending_error(const char* what) {
    ++checks;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s left a CUDA error pending (%s) for the next caller to misread\n",
                     what, cudaGetErrorString(e));
        std::exit(1);
    }
}
/// Runs a dump or restore that is EXPECTED to fail, and keeps its message so the caller can assert WHICH refusal
/// happened.  "restore returned false" cannot tell a version refusal from a corrupt-file refusal from a null.
template <class Fn>
void expect_fail(Fn&& fn, const char* must_name) {
    std::string err;
    const bool ok = fn(err);
    ++checks;
    if (ok) {
        std::fprintf(stderr, "FAIL: expected a refusal naming '%s', but the call succeeded\n", must_name);
        std::exit(1);
    }
    if (err.find(must_name) == std::string::npos) {
        std::fprintf(stderr, "FAIL: the refusal for '%s' was: %s\n", must_name, err.c_str());
        std::exit(1);
    }
    last_error = err;
    ++refusals;
}

// ================================ the synthetic session ================================

const strata::kernels::QsaShapes SHP = strata::kernels::qsa_real_shapes();   // page_size 4, idx_block 4
constexpr int64_t MAX_CELLS = 64;                       // 16 pages; 18 pooled rows (kv_plan's max_cells/4 + 2)
constexpr int64_t L_BOUNDARY = 10;                      // NOT idx_block-aligned: 2 completed blocks + a spare
constexpr int64_t L_CONSUMED = 26;                      // the boundary plus 16 generated / hidden-reasoning tokens
const int64_t SPARE_ROW = L_BOUNDARY / SHP.idx_block;   // 2: the row a boundary snapshot holds stale
constexpr int64_t DRAFT_CELLS = 28;                     // the drafter's ring: 7 host pages
constexpr int64_t DRAFT_SLOTS = 2;                      // ... and 2 VRAM slots

/// A byte pattern naming the array AND the block a byte came from, so every copy is checked for PROVENANCE
/// (which block landed in which slot) rather than for "not zero".  Never 0: 0 is the untouched sentinel.
uint8_t block_tag(int array_id, int64_t block) { return (uint8_t) (1 + ((array_id * 37 + block * 11) % 250)); }
void tag_blocks(std::vector<uint8_t>& b, int array_id, size_t block_bytes) {
    for (size_t blk = 0; blk * block_bytes < b.size(); ++blk)
        for (size_t i = 0; i < block_bytes && blk * block_bytes + i < b.size(); ++i)
            b[blk * block_bytes + i] = block_tag(array_id, (int64_t) blk);
}
bool slot_holds(const std::vector<uint8_t>& slots, int array_id, size_t block_bytes, int64_t slot, int64_t block) {
    for (size_t i = 0; i < block_bytes && (size_t) slot * block_bytes + i < slots.size(); ++i)
        if (slots[(size_t) slot * block_bytes + i] != block_tag(array_id, block)) return false;
    return true;
}

void set_row(std::vector<uint8_t>& pooled, int64_t row, int64_t idx_dim, float v) {
    ((float*) pooled.data())[row * idx_dim] = v;
}
float row_value(const std::vector<uint8_t>& pooled, int64_t row, int64_t idx_dim) {
    return ((const float*) pooled.data())[row * idx_dim];
}

struct Store {
    std::vector<uint8_t> k, v, ks, vs;                        // the AUTHORITATIVE host copy
    std::vector<uint8_t> slot_k, slot_v, slot_ks, slot_vs;    // the "VRAM" pages (a ring's slots)
    std::vector<uint8_t> pooled, tail, dead, bpos;            // the indexer's device arrays
    std::vector<int32_t> page_table, slot_block, slot_stamp, slot_ref, miss_block, miss_slot, ctl;
};

/// Bytes one block (page) of one array holds: [page][kv_head][page_size][head_dim], or the scale run.
int64_t block_bytes(int array_id) {
    const int64_t rows = SHP.n_head_kv * SHP.page_size;
    return array_id < 2 ? rows * SHP.head_dim : rows * (SHP.head_dim / 64) * 2;
}

struct Session {
    ModelGeometry g;
    ConversationStateSizes z;
    std::vector<Store> stores;
    std::vector<QsaState> layers;
    std::vector<uint8_t> gdn, ple;
    SessionState ss;
    Store draft_store;
    QsaState draft;

    Session() {
        // A small model with the REAL QSA widths, so the tier's granules and the shared core's geometry key are
        // the ones a live engine would use.  GDN is sized down because this fixture never runs a layer.
        g.n_embd = 2560; g.n_layers = 8; g.qsa_interval = 4;                 // 2 QSA layers, 6 GDN layers
        g.ssm_state_size = 2; g.ssm_k_heads = 1; g.ssm_v_heads = 2;
        g.ssm_d_conv = 4; g.ssm_conv_channels = 8; g.ssm_value_dim = 1;
        g.n_head = 24; g.n_head_kv = 2; g.head_dim = 256; g.idx_q_heads = 4; g.idx_key_dim = 128;
        g.hc = 4; g.hc_lr = 320; g.n_expert = 512; g.n_ff = 640;
        std::string err;
        if (!conversation_state_sizes(g, z, err)) {
            std::fprintf(stderr, "sizing refused: %s\n", err.c_str());
            std::exit(1);
        }

        const int64_t n_qsa = g.n_qsa_layers(), pages = MAX_CELLS / SHP.page_size;
        stores.resize((size_t) n_qsa);
        layers.resize((size_t) n_qsa);
        for (int64_t i = 0; i < n_qsa; ++i) {
            Store& s = stores[(size_t) i];
            s.k.assign((size_t) MAX_CELLS * SHP.n_head_kv * SHP.head_dim, 0);
            s.v.assign(s.k.size(), 0);
            s.ks.assign((size_t) MAX_CELLS * SHP.n_head_kv * (SHP.head_dim / 64) * 2, 0);
            s.vs.assign(s.ks.size(), 0);
            s.pooled.assign((size_t) (MAX_CELLS / SHP.idx_block + 2) * g.idx_key_dim * 4, 0);
            s.tail.assign(z.tail, 0); s.dead.assign(z.dead, 0); s.bpos.assign(z.block_pos, 0);
            s.page_table.assign((size_t) pages, -1);
            s.slot_block.assign((size_t) pages, -1); s.slot_stamp.assign((size_t) pages, -1);
            s.slot_ref.assign((size_t) pages, 0);
            s.miss_block.assign((size_t) pages, 0); s.miss_slot.assign((size_t) pages, 0);
            s.ctl.assign(strata::kernels::kKvCtlInts, 0);
            wire(layers[(size_t) i], s, MAX_CELLS, pages, pages, MAX_CELLS / SHP.idx_block + 2, false);
        }
        {   // the drafter: a RING (kv_mode 2), so the restore's `refill_drafter_ring` has a window to refill
            Store& s = draft_store;
            const int64_t dpages = DRAFT_CELLS / SHP.page_size;
            s.k.assign((size_t) DRAFT_CELLS * SHP.n_head_kv * SHP.head_dim, 0);
            s.v.assign(s.k.size(), 0);
            s.ks.assign((size_t) DRAFT_CELLS * SHP.n_head_kv * (SHP.head_dim / 64) * 2, 0);
            s.vs.assign(s.ks.size(), 0);
            s.slot_k.assign((size_t) DRAFT_SLOTS * SHP.page_size * SHP.n_head_kv * SHP.head_dim, 0);
            s.slot_v.assign(s.slot_k.size(), 0);
            s.slot_ks.assign((size_t) DRAFT_SLOTS * SHP.page_size * SHP.n_head_kv * (SHP.head_dim / 64) * 2, 0);
            s.slot_vs.assign(s.slot_ks.size(), 0);
            s.pooled.assign(2 * g.idx_key_dim * 4, 0);   // a ring has no indexer: 2 rows, as `kv_plan` sizes it
            s.tail.assign(z.tail, 0); s.dead.assign(z.dead, 0); s.bpos.assign(z.block_pos, 0);
            s.page_table.assign((size_t) dpages, 0);
            wire(draft, s, DRAFT_CELLS, dpages, DRAFT_SLOTS, 2, true);
        }
        gdn.assign(z.gdn, 0);
        ple.assign(z.ple, 0);

        ss.max_cells = MAX_CELLS;
        ss.gdn_state = (float*) gdn.data();
        ss.ple_hist = (float*) ple.data();
        ss.qsa_states = layers.data();
    }

    static void wire(QsaState& st, Store& s, int64_t max_cells, int64_t n_pages, int64_t n_slots,
                     int64_t pooled_rows, bool ring) {
        st.kv_int8 = true;
        st.kv_mode = ring ? 2 : 1;
        st.max_cells = max_cells; st.n_pages = n_pages; st.n_slots = n_slots; st.idx_pooled_rows = pooled_rows;
        st.host.k_q = (int8_t*) s.k.data(); st.host.v_q = (int8_t*) s.v.data();
        st.host.k_scale = (uint16_t*) s.ks.data(); st.host.v_scale = (uint16_t*) s.vs.data();
        if (ring) {   // the pools a reader sees, which is what the ring refill copies INTO
            st.k_q = (int8_t*) s.slot_k.data(); st.v_q = (int8_t*) s.slot_v.data();
            st.k_scale = (uint16_t*) s.slot_ks.data(); st.v_scale = (uint16_t*) s.slot_vs.data();
        }
        st.page_table = s.page_table.data();
        st.map.page_table = s.page_table.data(); st.map.n_blocks = n_pages; st.map.n_slots = n_slots;
        st.map.slot_block = s.slot_block.data(); st.map.slot_stamp = s.slot_stamp.data();
        st.map.slot_ref = s.slot_ref.data(); st.map.ctl = s.ctl.data();
        st.map.miss_block = s.miss_block.data(); st.map.miss_slot = s.miss_slot.data();
        st.idx_pooled = (float*) s.pooled.data(); st.idx_tail = (float*) s.tail.data();
        st.idx_dead = (float*) s.dead.data(); st.idx_block_pos = (int32_t*) s.bpos.data();
    }

    /// The live indexer, in the state a turn-boundary dump actually finds it: `dead` is a value NO pooled row
    /// has, the completed rows hold their own keys, and the SPARE row at `L / idx_block` holds what a block
    /// completed by tokens PAST the boundary wrote - the stale value C3 has to overwrite.
    void seed_indexer(float dead_value, int64_t completed_blocks, float stale_spare) {
        for (size_t i = 0; i < stores.size(); ++i) {
            Store& s = stores[i];
            const int64_t rows = (int64_t) s.pooled.size() / (g.idx_key_dim * 4);
            for (int64_t r = 0; r < rows; ++r) set_row(s.pooled, r, g.idx_key_dim, 1000.0f + (float) r);
            for (int64_t b = 0; b < completed_blocks; ++b) set_row(s.pooled, b, g.idx_key_dim, 1000.0f + (float) b);
            set_row(s.pooled, SPARE_ROW, g.idx_key_dim, stale_spare);
            set_row(s.dead, 0, g.idx_key_dim, dead_value);
            std::fill(s.tail.begin(), s.tail.end(), (uint8_t) (0x30 + i));
            *(int32_t*) s.bpos.data() = (int32_t) ((L_CONSUMED / SHP.idx_block) * SHP.idx_block);  // past L
        }
    }
    /// A session that has never seen this snapshot: nothing already holds the value a restore has to publish.
    void clear_indexer() {
        for (Store& s : stores) {
            std::fill(s.pooled.begin(), s.pooled.end(), 0);
            std::fill(s.tail.begin(), s.tail.end(), 0);
            std::fill(s.dead.begin(), s.dead.end(), 0);
            std::fill(s.bpos.begin(), s.bpos.end(), 0);
        }
        std::fill(gdn.begin(), gdn.end(), 0);
        std::fill(ple.begin(), ple.end(), 0);
    }

    /// Tags every KV array block-by-block, so a copy can be traced to the block it came from.
    void tag_kv() {
        int array = 0;
        for (Store& s : stores) {
            tag_blocks(s.k, array++, (size_t) block_bytes(0));
            tag_blocks(s.v, array++, (size_t) block_bytes(0));
            tag_blocks(s.ks, array++, (size_t) block_bytes(2));
            tag_blocks(s.vs, array++, (size_t) block_bytes(2));
        }
        draft_k_tag = array; tag_blocks(draft_store.k, array++, (size_t) block_bytes(0));
        tag_blocks(draft_store.v, array++, (size_t) block_bytes(0));
        draft_ks_tag = array; tag_blocks(draft_store.ks, array++, (size_t) block_bytes(2));
        tag_blocks(draft_store.vs, array++, (size_t) block_bytes(2));
    }
    /// Which tag id `tag_kv` gave the drafter's K and its scale array (so the ring checks name them, not literals).
    int draft_k_array() const { return draft_k_tag; }
    int draft_ks_array() const { return draft_ks_tag; }

    // ---- the "did the restore touch anything?" oracle (the shared core's fixture has the same one) ----
    /// Fill every buffer the tier could ever write with a sentinel, so "untouched" is a byte-level claim.  Never 0:
    /// 0 is what a fresh session holds and what several other checks here assert.
    void poison() {
        auto fill = [](std::vector<uint8_t>& b) { std::fill(b.begin(), b.end(), SENTINEL); };
        for (Store& s : stores) { fill(s.k); fill(s.v); fill(s.ks); fill(s.vs); fill(s.pooled);
                                   fill(s.tail); fill(s.dead); fill(s.bpos); }
        fill(draft_store.k); fill(draft_store.v); fill(draft_store.ks); fill(draft_store.vs);
        fill(draft_store.pooled); fill(draft_store.tail); fill(draft_store.dead); fill(draft_store.bpos);
        fill(draft_store.slot_k); fill(draft_store.slot_v); fill(draft_store.slot_ks); fill(draft_store.slot_vs);
        fill(gdn); fill(ple);
        ss.ple_prev[0] = -1;
        ss.ple_prev[1] = -1;
    }
    static bool is_sentinel(const std::vector<uint8_t>& b) {
        return std::all_of(b.begin(), b.end(), [](uint8_t x) { return x == SENTINEL; });
    }
    /// Every array the tier writes is still the sentinel - i.e. the failure happened before it wrote anything.
    bool untouched() const {
        for (const Store& s : stores)
            if (!(is_sentinel(s.k) && is_sentinel(s.v) && is_sentinel(s.ks) && is_sentinel(s.vs) &&
                  is_sentinel(s.pooled) && is_sentinel(s.tail) && is_sentinel(s.dead) && is_sentinel(s.bpos)))
                return false;
        if (!(is_sentinel(draft_store.k) && is_sentinel(draft_store.v) && is_sentinel(draft_store.ks) &&
              is_sentinel(draft_store.vs) && is_sentinel(draft_store.pooled) && is_sentinel(draft_store.tail) &&
              is_sentinel(draft_store.dead) && is_sentinel(draft_store.bpos))) return false;
        if (!(is_sentinel(draft_store.slot_k) && is_sentinel(draft_store.slot_v) &&
              is_sentinel(draft_store.slot_ks) && is_sentinel(draft_store.slot_vs))) return false;
        return is_sentinel(gdn) && is_sentinel(ple) && ss.ple_prev[0] == -1 && ss.ple_prev[1] == -1;
    }
    /// The PLE window is the last thing `nvme_restore` writes, so it says "the apply pass finished" in a form the
    /// store's own API can be checked against (the store does not hand back `L`).
    int64_t ple_prev_last() const { return ss.ple_prev[1]; }

    static constexpr uint8_t SENTINEL = 0xA5;

  private:
    int draft_k_tag = -1, draft_ks_tag = -1;
};

/// One pooled ROW still holding the poison sentinel - the row-level form of "this write did not happen".
bool row_is_sentinel(const std::vector<uint8_t>& pooled, int64_t row, int64_t idx_dim) {
    const uint8_t* p = pooled.data() + (size_t) row * (size_t) idx_dim * 4;
    return std::all_of(p, p + (size_t) idx_dim * 4, [](uint8_t x) { return x == Session::SENTINEL; });
}

// ================================ the envelope, read back as bytes ================================

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

/// The layout an L-token snapshot MUST have, computed HERE rather than read from the tier, so a change to the
/// tier's segment arithmetic shows up as a disagreement.  `rows` is a parameter so the fixture can ask what the
/// file would look like under a different pooled-row count.
struct Layout {
    int64_t idx_dim = 0;
    size_t ids = 0, imgs = 0, gdn = 0, ple = 0, kv = 0, pooled = 0, tail = 0, dead = 0, bpos = 0, drafter = 0, end = 0;
    size_t pooled_at(size_t layer, int64_t row) const {
        return strata::platform::kNvmeHeaderBytes + ids + imgs + gdn + ple +
               layer * (kv + pooled + tail + dead + bpos) + kv + (size_t) row * (size_t) idx_dim * 4;
    }
};
Layout layout_of(const Session& S, int64_t L, int64_t n_imgs, int64_t rows) {
    Layout x;
    x.idx_dim = S.g.idx_key_dim;
    const int64_t pages = (L + SHP.page_size - 1) / SHP.page_size;
    x.ids = (size_t) L * 4;
    x.imgs = (size_t) n_imgs * sizeof(ConversationImageKey);
    x.gdn = S.z.gdn;
    x.ple = S.z.ple;
    for (int a = 0; a < 4; ++a) x.kv += (size_t) pages * block_bytes(a);
    x.pooled = (size_t) rows * S.g.idx_key_dim * 4;
    x.tail = S.z.tail; x.dead = S.z.dead; x.bpos = S.z.block_pos;
    const int64_t mL = std::min<int64_t>(L, DRAFT_CELLS);
    const int64_t mp = (mL + SHP.page_size - 1) / SHP.page_size;
    for (int a = 0; a < 4; ++a) x.drafter += (size_t) mp * block_bytes(a);
    x.end = strata::platform::kNvmeHeaderBytes + x.ids + x.imgs + x.gdn + x.ple +
            (size_t) S.g.n_qsa_layers() * (x.kv + x.pooled + x.tail + x.dead + x.bpos) + x.drafter;
    return x;
}

std::vector<int32_t> ids_of(int64_t n) {
    std::vector<int32_t> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (int32_t) (100 + i);
    return v;
}

/// The boundary's checkpoint: its ids, its pictures (the live list filtered below `L`), and a running state that
/// is NOT the live session's - except `dead`, which is copied from the live indexer exactly as
/// `conversation_checkpoint_save` copies it, because `dead` is the cell-0 key and constant for the sequence
/// (qsa.cu:187).  That is C3's premise, and it is what makes the re-publish checkable: the file's `dead` is the
/// value the spare row must end up holding, and the row the dump read is a DIFFERENT one.
ConversationCheckpoint boundary_checkpoint(const Session& S, int32_t block_pos_at_boundary) {
    ConversationCheckpoint cp;
    cp.ids = ids_of(L_BOUNDARY);
    cp.imgs = {{3, 0xAAAA}};                       // imgs_below(live_imgs, 10): the picture at 14 is NOT in it
    cp.gdn.assign(S.z.gdn, 0xC1);
    cp.ple.assign(S.z.ple, 0xC2);
    cp.tails.assign((size_t) S.g.n_qsa_layers() * S.z.tail, 0xC3);
    cp.dead.assign((size_t) S.g.n_qsa_layers() * S.z.dead, 0);
    for (size_t i = 0; i < (size_t) S.g.n_qsa_layers(); ++i)
        std::memcpy(cp.dead.data() + i * S.z.dead, S.stores[i].dead.data(), S.z.dead);
    cp.block_pos.assign((size_t) S.g.n_qsa_layers() * S.z.block_pos, 0);
    for (size_t i = 0; i < (size_t) S.g.n_qsa_layers(); ++i)
        std::memcpy(cp.block_pos.data() + i * S.z.block_pos, &block_pos_at_boundary, sizeof block_pos_at_boundary);
    return cp;
}

// ================================ fixture 1: the turn boundary, and the promote it makes possible ================================

void fixture_turn_boundary(const std::string& dir) {
    Session S;
    const int64_t n_qsa = S.g.n_qsa_layers();
    const std::vector<int32_t> consumed = ids_of(L_CONSUMED), boundary = ids_of(L_BOUNDARY);
    // The live conversation's pictures: one INSIDE the boundary, one at token 14 - inside the consumed prefix, at
    // or past the boundary.  That is the exact shape that hid both image bugs (`6585361`, `49d0b7d`).
    const std::vector<ConversationImageKey> live_imgs = {{3, 0xAAAA}, {14, 0xBBBB}};
    const int32_t boundary_block = (int32_t) ((L_BOUNDARY / SHP.idx_block) * SHP.idx_block);   // 8

    // The live indexer: `dead` a value no pooled row has, and the spare row at L/idx_block holding the key of a
    // block completed by tokens PAST the boundary.
    const float DEAD = 987654.0f, STALE = 555.0f;
    S.seed_indexer(DEAD, L_CONSUMED / SHP.idx_block, STALE);
    for (const Store& s : S.stores)
        for (int64_t r = 0; r < (int64_t) s.pooled.size() / (S.g.idx_key_dim * 4); ++r)
            ck(row_value(s.pooled, r, S.g.idx_key_dim) != DEAD,
               "C3's precondition: idx_dead differs from every pooled row");
    S.tag_kv();
    for (Store& s : S.stores) std::fill(s.tail.begin(), s.tail.end(), (uint8_t) 0xE7);   // the LIVE tails
    const ConversationCheckpoint cp = boundary_checkpoint(S, boundary_block);

    std::string err;
    strata::platform::KvNvmeStore store;
    ck(store.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), err), "the store opens");
    // the serve loop's own call shape: consumed ids, live pictures, and the boundary as the key
    ck(store.dump(S.ss, S.draft, S.g, consumed, live_imgs, true, &cp, err), "the turn-boundary dump succeeds");
    ck(store.size() == 1, "one snapshot");
    const NvmeEntry& e = store.entries()[0];
    ck_eq(e.L, L_BOUNDARY, "C1: the KEY is the turn boundary, not the consumed length");
    ck(e.ids == boundary, "the key excludes the model's generated and hidden reasoning tokens");
    ck(e.imgs == cp.imgs, "the entry carries the boundary's pictures, not the live conversation's");
    ck(e.imgs.size() == 1 && e.imgs[0].start == 3, "the picture at token 14 is not in the snapshot");

    const int64_t rows = strata::kernels::qsa_pooled_rows(L_BOUNDARY, SHP);
    ck_eq(rows, SPARE_ROW + 1, "C4: qsa_pooled_rows = the completed rows plus the spare row");
    const std::vector<uint8_t> file = slurp(e.path);
    const Layout x = layout_of(S, L_BOUNDARY, 1, rows);
    ck_eq((int64_t) file.size(), (int64_t) x.end + 8, "the file is the layout the format says it is");
    uint32_t magic = 0, version = 0;
    std::memcpy(&magic, file.data(), 4); std::memcpy(&version, file.data() + 4, 4);
    ck_eq(magic, (int64_t) NvmeHeader{}.magic, "magic");
    ck_eq(version, (int64_t) strata::platform::kNvmeFormatVersion, "format version");
    int64_t file_L = 0;
    std::memcpy(&file_L, file.data() + 8, 8);
    ck_eq(file_L, L_BOUNDARY, "L at offset 8");
    std::vector<int32_t> file_ids((size_t) L_BOUNDARY);
    std::memcpy(file_ids.data(), file.data() + strata::platform::kNvmeHeaderBytes, (size_t) L_BOUNDARY * 4);
    ck(file_ids == boundary, "the ids start exactly at kNvmeHeaderBytes - the offset the shell oracles read");
    for (size_t layer = 0; layer < (size_t) n_qsa; ++layer) {
        float spare = 0, r0 = 0, r1 = 0;
        std::memcpy(&spare, file.data() + x.pooled_at(layer, SPARE_ROW), 4);
        std::memcpy(&r0, file.data() + x.pooled_at(layer, 0), 4);
        std::memcpy(&r1, file.data() + x.pooled_at(layer, 1), 4);
        ck(spare == STALE, "the snapshot carries the STALE spare row a boundary dump can only produce");
        ck(r0 == 1000.0f && r1 == 1001.0f, "the completed block rows are the live ones");
        float file_dead = 0;   // the segment order is kv, pooled, tail, dead, block_pos
        std::memcpy(&file_dead, file.data() + x.pooled_at(layer, rows) + x.tail, 4);
        ck(file_dead == DEAD, "the file's `dead` segment is the boundary's spare key - the value the row must become");
    }
    {   // the wider formula the tier rejected (`L / idx_block + 2`) would put layer 1's pooled segment elsewhere
        const Layout wide = layout_of(S, L_BOUNDARY, 1, rows + 1);
        float at_wide = 0;
        std::memcpy(&at_wide, file.data() + wide.pooled_at(1, 0), 4);
        ck(at_wide != 1000.0f, "the file is NOT the '+2 rows' layout: layer 1's pooled row 0 sits one row earlier");
    }

    // ---- restore into a FRESH session, and check what the tier leaves the indexer in ----
    Session R;
    R.clear_indexer();
    ck(row_value(R.stores[0].pooled, SPARE_ROW, S.g.idx_key_dim) != DEAD,
       "the fresh session's spare row is not `dead` before the restore, so the re-publish is observable");
    ck(store.restore(e, R.ss, R.draft, R.g, err) == strata::core::ConversationRestore::restored,
       "the snapshot restores");
    ck_eq(R.ss.ple_prev[1], boundary.back(), "the PLE window is the boundary's last token");
    for (int64_t i = 0; i < n_qsa; ++i) {
        const Store& s = R.stores[(size_t) i];
        ck(row_value(s.pooled, SPARE_ROW, S.g.idx_key_dim) == DEAD,
           "C3: pooled row L/idx_block is `dead` again after the restore");
        ck(row_value(s.pooled, 0, S.g.idx_key_dim) == 1000.0f &&
           row_value(s.pooled, 1, S.g.idx_key_dim) == 1001.0f, "the re-publish did not clobber the completed rows");
        ck(row_value(s.pooled, SPARE_ROW + 1, S.g.idx_key_dim) == 0.0f,
           "the row PAST the spare was never written: the segment is [0, L/idx_block + 1), as C4 settled");
        int32_t bp = 0;
        std::memcpy(&bp, s.bpos.data(), 4);
        ck_eq(bp, boundary_block, "C5: block_pos is the BOUNDARY's block (8), not the live array's (24)");
        ck(row_value(s.dead, 0, S.g.idx_key_dim) == DEAD, "dead is the boundary checkpoint's spare key");
        ck(std::equal(s.tail.begin(), s.tail.end(), cp.tails.begin() + (size_t) i * S.z.tail),
           "C5: the tail is the boundary checkpoint's, not the live device's");
    }
    ck(std::equal(R.gdn.begin(), R.gdn.end(), cp.gdn.begin()), "the GDN state is the boundary's");
    ck(std::equal(R.ple.begin(), R.ple.end(), cp.ple.begin()), "the PLE history is the boundary's");
    const int64_t pages = (L_BOUNDARY + SHP.page_size - 1) / SHP.page_size;
    ck(std::equal(R.stores[0].k.begin(), R.stores[0].k.begin() + pages * block_bytes(0), S.stores[0].k.begin()),
       "the authoritative host KV copy round-trips byte for byte");
    ck(std::equal(R.draft_store.k.begin(), R.draft_store.k.begin() + pages * block_bytes(0),
                  S.draft_store.k.begin()), "the drafter's host copy round-trips");

    // C6: the drafter's ring, refilled by the adapter.  b1 is the window the adapter RESTORED -
    // ceil(min(L, max_cells) / page_size) = 3 - not the snapshot's L, which would give 7 and land block 6 in
    // slot 0.  The LOWER bound (b1 - n_slots) is not separately observable here: with more blocks than slots the
    // ring's last writer wins, so a wider lower bound ends in the same slots.  That is stated, not glossed over.
    const int64_t b1 = (std::min<int64_t>(L_BOUNDARY, DRAFT_CELLS) + SHP.page_size - 1) / SHP.page_size;
    ck_eq(b1, 3, "the refill window ends at the restored prefix, not at L");
    ck(slot_holds(R.draft_store.slot_k, S.draft_k_array(), (size_t) block_bytes(0), 1, 1), "slot 1 holds block 1");
    ck(slot_holds(R.draft_store.slot_k, S.draft_k_array(), (size_t) block_bytes(0), 0, 2),
       "slot 0 holds block 2 (the ring map is block % n_slots)");
    ck(slot_holds(R.draft_store.slot_ks, S.draft_ks_array(), (size_t) block_bytes(2), 0, 2),
       "the scale arrays follow the same map");
    ck(!slot_holds(R.draft_store.slot_k, S.draft_k_array(), (size_t) block_bytes(0), 0, 6),
       "the refill did not run to ceil(L / page_size): block 6 is NOT in slot 0");

    // ---- the PROMOTE: the next request re-sends the prompt WITHOUT the model's reasoning tokens ----
    std::vector<int64_t> next_request;
    for (int32_t t : boundary) next_request.push_back(t);
    for (int64_t i = 0; i < 6; ++i) next_request.push_back(900 + i);          // the client's new turn
    const std::vector<ConversationImageKey> next_imgs = {{3, 0xAAAA}, {16, 0xCCCC}};   // a new picture, past the key
    const NvmeEntry* best = strata::platform::kv_nvme_match(store.entries(), next_request, next_imgs, true, 0);
    ck(best != nullptr && best->L == L_BOUNDARY, "C1: the next request PROMOTES the boundary snapshot");
    ck(best != nullptr && best->path == e.path, "and promotes THIS snapshot");
    ck(strata::platform::kv_nvme_match(store.entries(), next_request, next_imgs, false, 0) == nullptr,
       "a different control-vector state does not promote");
    ck(strata::platform::kv_nvme_match(store.entries(), next_request, {{3, 0x9999}}, true, 0) == nullptr,
       "a different picture inside the key does not promote");
    ck(strata::platform::kv_nvme_match(store.entries(), std::vector<int64_t>(boundary.begin(), boundary.end()),
                                       next_imgs, true, 0) == nullptr,
       "an entry as long as the request cannot promote (the last token starts the verify window)");
    ck(strata::platform::kv_nvme_match(store.entries(), next_request, next_imgs, true, L_BOUNDARY) == nullptr,
       "an entry no longer than what the session already holds cannot promote");

    // THE NEGATIVE CONTROL that makes the promote meaningful: the SAME session dumped without a boundary is keyed
    // on the consumed ids, and the same request cannot reach it.
    strata::platform::KvNvmeStore full;
    ck(full.open(dir + "/full", S.g, strata::core::qsa_kv_format(S.layers[0]), err), "the full-state store opens");
    ck(full.dump(S.ss, S.draft, S.g, consumed, live_imgs, true, nullptr, err), "the full-consumed dump succeeds");
    ck_eq(full.entries()[0].L, L_CONSUMED, "a dump without a boundary is keyed on the consumed length");
    ck(strata::platform::kv_nvme_match(full.entries(), next_request, next_imgs, true, 0) == nullptr,
       "C1: a consumed-state snapshot can never match a client that drops the reasoning tokens");
    std::vector<int64_t> same_history(consumed.begin(), consumed.end());
    for (int64_t i = 0; i < 3; ++i) same_history.push_back(900 + i);
    ck(strata::platform::kv_nvme_match(full.entries(), same_history, live_imgs, true, 0) != nullptr,
       "and it does match a request that replays the whole consumed history - the control is not vacuous");
}

// ================================ fixture 2: the image filter ================================

void fixture_image_filter(const std::string& dir) {
    Session S;
    S.seed_indexer(1.0f, L_CONSUMED / SHP.idx_block, 2.0f);
    const std::vector<int32_t> boundary = ids_of(L_BOUNDARY);
    const ConversationCheckpoint cp = boundary_checkpoint(S, 8);
    const std::string path = dir + "/snap.bin";

    expect_fail([&](std::string& e) {
        return strata::platform::nvme_dump_at(path.c_str(), S.ss, S.draft, S.g, boundary,
                                              {{3, 1}, {L_BOUNDARY, 2}}, true, &cp, e);
    }, "not inside the 10-token prefix");
    ck(last_error.find("start 10") != std::string::npos, "the refusal names the image's start");
    expect_fail([&](std::string& e) {
        return strata::platform::nvme_dump_at(path.c_str(), S.ss, S.draft, S.g, boundary, {{L_CONSUMED, 2}}, true,
                                              &cp, e);
    }, "not inside the 10-token prefix");
    expect_fail([&](std::string& e) {
        return strata::platform::nvme_dump_at(path.c_str(), S.ss, S.draft, S.g, boundary, {{-1, 2}}, true, &cp, e);
    }, "not inside the 10-token prefix");
    std::string err;
    ck(strata::platform::nvme_dump_at(path.c_str(), S.ss, S.draft, S.g, boundary, cp.imgs, true, &cp, err),
       "the boundary's own filtered list is accepted");
}

// ================================ fixture 3: the refusals, and the pooled-row guard ================================

void fixture_refusals(const std::string& dir) {
    Session S;
    S.seed_indexer(7.0f, L_CONSUMED / SHP.idx_block, 5.0f);
    S.tag_kv();
    const std::vector<int32_t> boundary = ids_of(L_BOUNDARY);
    std::string err;
    const std::string path = dir + "/snap.bin";
    ck(strata::platform::nvme_dump(path.c_str(), S.ss, S.draft, S.g, boundary, {}, true, err), "a dump to read back");
    const std::vector<uint8_t> good = slurp(path);

    auto restore_from = [&](const std::vector<uint8_t>& bytes, Session& into, std::string& e) {
        std::ofstream f(path, std::ios::binary);
        f.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        f.close();
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        bool cvec = false;
        int64_t L = 0;
        return strata::platform::nvme_restore(path.c_str(), into.ss, into.draft, into.g, ids, imgs, cvec, L, e) ==
               strata::core::ConversationRestore::restored;
    };

    {   // a version-2 file is refused BY NAME, before any segment is walked
        std::vector<uint8_t> v2 = good;
        uint32_t old = 2;
        std::memcpy(v2.data() + 4, &old, 4);
        Session R;
        expect_fail([&](std::string& e) { return restore_from(v2, R, e); }, "version 2");
        ck(last_error.find("this build writes version 3") != std::string::npos,
           "the refusal says which version this build writes");
    }
    {   // another geometry: the shared core's key, one field off (the last of its 18 int64s, at offset 32+17*8)
        std::vector<uint8_t> bad = good;
        bad[32 + 17 * 8] ^= 0xFF;
        Session R;
        expect_fail([&](std::string& e) { return restore_from(bad, R, e); }, "geometry/format mismatch");
    }
    {   // a flipped payload byte: integrity, and nothing applied
        std::vector<uint8_t> bad = good;
        bad[strata::platform::kNvmeHeaderBytes + 40 + 16 + 100] ^= 0xFF;   // inside the PLE segment
        Session R;
        expect_fail([&](std::string& e) { return restore_from(bad, R, e); }, "integrity check failed");
    }
    {   // a truncated file
        std::vector<uint8_t> bad = good;
        bad.resize(bad.size() - 4096);
        Session R;
        expect_fail([&](std::string& e) { return restore_from(bad, R, e); }, "truncated snapshot");
    }
    {   // a live pooled array too small for the snapshot: refuse, naming BOTH counts (no silent short segment)
        Session T;
        T.seed_indexer(7.0f, 1, 1.0f);
        for (size_t i = 0; i < T.stores.size(); ++i) {
            T.stores[i].pooled.assign(2 * T.g.idx_key_dim * 4, 0);   // 2 rows; L/idx_block + 1 = 3
            T.layers[i].idx_pooled = (float*) T.stores[i].pooled.data();
            T.layers[i].idx_pooled_rows = 2;
        }
        expect_fail([&](std::string& e) {
            return strata::platform::nvme_dump((dir + "/small.bin").c_str(), T.ss, T.draft, T.g, boundary, {},
                                               true, e);
        }, "pooled rows");
        ck(last_error.find("needs 3") != std::string::npos && last_error.find("holds 2") != std::string::npos,
           "the refusal names the count it needs and the count the engine has");
    }
    {   // a checkpoint that does not fit this engine
        Session T;
        T.seed_indexer(7.0f, 1, 1.0f);
        ConversationCheckpoint cp = boundary_checkpoint(T, 8);
        cp.gdn.assign(T.z.gdn + 1, 1);   // one byte too many
        expect_fail([&](std::string& e) {
            return strata::platform::nvme_dump_at((dir + "/cp.bin").c_str(), T.ss, T.draft, T.g, cp.ids, {}, true,
                                                  &cp, e);
        }, "checkpoint does not fit");
    }
}

#ifndef _WIN32
/// Runs `fn` with the C stderr redirected into a temp file and returns what it wrote.  The store's scan message is
/// the ONLY thing an operator holding a directory of unreadable snapshots sees, so the fixture reads it rather
/// than assuming it.
template <class Fn>
std::string capture_stderr(Fn&& fn) {
    const std::string p = "/tmp/kv-nvme-stderr-" + std::to_string((long) ::getpid()) + ".txt";
    const int saved = dup(2);
    if (saved < 0) { fn(); return {}; }
    std::fflush(stderr);
    if (std::freopen(p.c_str(), "w", stderr) == nullptr) { ::close(saved); fn(); return {}; }
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    ::close(saved);
    std::ifstream f(p);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
#endif

// ================================ fixture 4: THE FAILURE CONTRACT (collision C7) ================================
//
// THE QUESTION: which failures may a caller recover from, and which ones mean the process must stop.  Our tier
// used to answer "all of them, by dropping the snapshot and re-reading the prompt"; the shared core answers
// "a transfer failure is fatal".  This fixture decides between them with the tier's own behaviour rather than by
// argument, and it does it with the CUDA error state MODELLED (see the wrappers above) because the whole question
// turns on what a failed copy leaves behind for the next caller.
//
// THE SHAPE OF `nvme_restore`, which is what makes the two classes distinguishable at all:
//   validation pass  - file read, header, geometry, segment walk, layout end, payload digest.  No CUDA call.
//   pre-apply sync   - the device must answer before a byte is written (the shared core does the same,
//                      conversation_state.cpp:258).
//   apply pass       - a LOOP over the recorded segments.  A failure in its middle is a HALF-WRITTEN session.
//   residency pass   - kv_stream_reset per layer, the spare-row re-publish, the drafter ring refill.
//   post-apply sync  - the device took everything above.  This is the proof a clean reset needs.
void fixture_failure_contract(const std::string& dir) {
    using Restore = strata::core::ConversationRestore;
    const std::string path = dir + "/snap.bin";

    Session S;
    S.seed_indexer(11.0f, L_CONSUMED / SHP.idx_block, 12.0f);
    S.tag_kv();
    std::string err;
    reset_faults();
    ck(strata::platform::nvme_dump(path.c_str(), S.ss, S.draft, S.g, ids_of(L_BOUNDARY), {}, true, err),
       "the failure fixture has a snapshot to mutate");
    const std::vector<uint8_t> good = slurp(path);
    // cudaMemcpy calls the apply pass makes, in order: gdn, ple, then per QSA layer pooled / tail / dead /
    // block_pos, then one spare-row re-publish per layer.  The KV segments are plain memcpy into the pinned host
    // copies and are NOT counted - which is itself part of the contract's evidence.
    const int device_applies = 2 + (int) S.g.n_qsa_layers() * 5;

    struct Res {
        Restore kind = Restore::restored;
        std::string err;
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        bool cvec = false;
        int64_t L = 0;
    };
    auto restore_bytes = [&](const std::vector<uint8_t>& bytes, Session& into, const std::string& p) -> Res {
        std::ofstream f(p, std::ios::binary);
        f.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        f.close();
        Res r;
        r.kind = strata::platform::nvme_restore(p.c_str(), into.ss, into.draft, into.g, r.ids, r.imgs, r.cvec, r.L,
                                                r.err);
        return r;
    };

    // ---- CLASS 1: every refusal the tier can make BEFORE the apply pass.  Recoverable, and provably so. ----
    auto expect_recoverable = [&](const std::vector<uint8_t>& bytes, const char* must_name, const char* label) {
        reset_faults();
        Session R;
        R.poison();
        const Res r = restore_bytes(bytes, R, path);
        ++refusals;
        ck(r.kind == Restore::invalid, label);
        if (r.err.find(must_name) == std::string::npos) {
            std::fprintf(stderr, "FAIL: the refusal for '%s' was: %s\n", must_name, r.err.c_str());
            std::exit(1);
        }
        ck_eq(copy_calls, 0, "a refused snapshot made not one cudaMemcpy");
        ck_eq(sync_calls, 0, "a refused snapshot made not one CUDA call at all");
        ck(R.untouched(), "the refused restore left every session buffer at its sentinel");
        ck(r.ids.empty() && r.imgs.empty(), "the caller is not handed a prefix from a refused snapshot");
        ck_no_pending_error(must_name);
        last_error = r.err;
    };
    {
        std::vector<uint8_t> b = good; std::memcpy(b.data(), "XXXX", 4);
        expect_recoverable(b, "bad magic", "a foreign file is refused");
    }
    {
        std::vector<uint8_t> b = good; uint32_t old = 2; std::memcpy(b.data() + 4, &old, 4);
        expect_recoverable(b, "version 2", "a stale format version is refused");
    }
    {
        std::vector<uint8_t> b = good; b[32 + 17 * 8] ^= 0xFF;
        expect_recoverable(b, "geometry/format mismatch", "another geometry is refused");
    }
    {
        std::vector<uint8_t> b = good; int64_t huge = 1LL << 40; std::memcpy(b.data() + 8, &huge, 8);
        expect_recoverable(b, "malformed header sizes", "a header that would size an impossible read is refused");
    }
    {
        std::vector<uint8_t> b = good; b.resize(b.size() - 4096);
        expect_recoverable(b, "truncated snapshot", "a truncated file is refused");
    }
    {
        std::vector<uint8_t> b = good; b.resize(b.size() + 8, 0x5A);   // trailing junk: the walk ends short of it
        expect_recoverable(b, "layout mismatch", "a file the segment walk cannot account for is refused");
    }
    {
        std::vector<uint8_t> b = good; b[strata::platform::kNvmeHeaderBytes + 40 + 16 + 100] ^= 0xFF;
        expect_recoverable(b, "integrity check failed", "a flipped payload byte is refused");
    }
    {   // a live engine too small for the snapshot: refused by the same guard the dump uses
        Session T;
        T.seed_indexer(7.0f, 1, 1.0f);
        for (size_t i = 0; i < T.stores.size(); ++i) {
            T.stores[i].pooled.assign(2 * T.g.idx_key_dim * 4, 0);   // 2 rows; this snapshot needs 3
            T.layers[i].idx_pooled = (float*) T.stores[i].pooled.data();
            T.layers[i].idx_pooled_rows = 2;
        }
        T.poison();
        reset_faults();
        const Res r = restore_bytes(good, T, path);
        ++refusals;
        ck(r.kind == Restore::invalid, "a snapshot too large for the live arrays is refused, not clamped");
        ck(r.err.find("needs 3") != std::string::npos && r.err.find("holds 2") != std::string::npos,
           "and the refusal names both counts");
        ck_eq(copy_calls, 0, "the too-large snapshot performed no cudaMemcpy");
        ck(T.untouched(), "and wrote nothing");
        ck_no_pending_error("the pooled-rows refusal");
    }
    {   // a null target buffer is a refusal, not a transfer (the shared core validates the same thing first)
        Session T;
        T.seed_indexer(7.0f, 1, 1.0f);
        T.layers[0].host.k_q = nullptr;
        T.poison();
        reset_faults();
        const Res r = restore_bytes(good, T, path);
        ++refusals;
        ck(r.kind == Restore::invalid, "a missing target buffer is refused before the first copy");
        ck(r.err.find("null host KV array") != std::string::npos, "and says which array is missing");
        ck_eq(copy_calls, 0, "a refused target buffer means no copy was attempted");
        ck(T.untouched(), "and nothing was written");
        ck_no_pending_error("the null-array refusal");
    }

    // ---- CLASS 2: a CUDA failure at or after the first write.  Fatal, and half-applied by construction. ----
    auto expect_fatal = [&](int which_copy, int which_sync, const char* must_name, const char* label) {
        reset_faults();
        fail_copy = which_copy;
        fail_sync = which_sync;
        Session R;
        R.poison();
        const Res r = restore_bytes(good, R, path);
        ++fatalities;
        ck(r.kind == Restore::transfer_failed, label);
        if (r.err.find(must_name) == std::string::npos) {
            std::fprintf(stderr, "FAIL: the transfer failure for '%s' was: %s\n", label, r.err.c_str());
            std::exit(1);
        }
        // THE FLOOR: the tier handled the failure, so it must have consumed the error it caused.  If it did not,
        // the next `cudaGetLastError()` in the engine - `kv_stream_reset`'s `check()`, which exits the process -
        // reports THIS failure as though it had happened there.
        ck_no_pending_error(label);
        last_error = r.err;
    };
    {   // the device is already unusable BEFORE a byte is written: fatal, and nothing applied
        expect_fatal(0, 1, "before the apply pass", "a failed pre-apply synchronize is a transfer failure");
        ck_eq(copy_calls, 0, "the pre-apply sync failed before any copy was attempted");
        Session R;
        R.poison();
        reset_faults();
        fail_sync = 1;
        restore_bytes(good, R, path);
        ck(R.untouched(), "a pre-apply sync failure wrote nothing: the session is provably intact");
    }
    {   // the FIRST device segment fails: fatal, and (because it is first) nothing applied yet
        expect_fatal(1, 0, "host-to-device transfer failed for the gdn", "the first copy failing is a transfer failure");
        ck_eq(copy_calls, 1, "exactly one cudaMemcpy happened before the tier stopped");
        Session R;
        R.poison();
        reset_faults();
        fail_copy = 1;
        restore_bytes(good, R, path);
        ck(R.untouched(), "and it was the first write, so nothing landed");
    }
    {   // THE HEADLINE: a failure in the MIDDLE of the apply pass leaves a HALF-WRITTEN session.  This is the
        // same shape the shared core's own fixture asserts for its restore
        // (`conversation_validation_test.cpp:167-169`: "partial CUDA transfer failure is fatal, not an
        // invalid-image fallback" / "fault fixture genuinely produced partial state").
        expect_fatal(2, 0, "host-to-device transfer failed for the ple", "a copy failing mid-apply is fatal");
        ck_eq(copy_calls, 2, "the tier stopped at the second device segment");
        Session R;
        R.poison();
        reset_faults();
        fail_copy = 2;
        restore_bytes(good, R, path);
        ck(!Session::is_sentinel(R.gdn), "the GDN state WAS written");
        ck(Session::is_sentinel(R.ple), "the PLE history was not");
        ck(Session::is_sentinel(R.stores[0].pooled), "nor was the first layer's indexer");
        ck(!R.untouched(), "the session is genuinely half-applied - which is why no clean reset is claimed here");
    }
    {   // deeper into the pass: more of the session is already the snapshot's
        expect_fatal(5, 0, "host-to-device transfer failed for the dead", "a late copy failing is fatal");
        Session R;
        R.poison();
        reset_faults();
        fail_copy = 5;
        restore_bytes(good, R, path);
        ck(!Session::is_sentinel(R.gdn) && !Session::is_sentinel(R.ple), "gdn and ple landed");
        ck(!Session::is_sentinel(R.stores[0].pooled) && !Session::is_sentinel(R.stores[0].tail),
           "layer 0's pooled and tail landed");
        ck(Session::is_sentinel(R.stores[1].bpos) && Session::is_sentinel(R.stores[1].pooled),
           "layer 1 did not: the mix is the whole point");
    }
    {   // the LAST device write of the residency pass fails: everything else is in, the invariant is not
        expect_fatal(device_applies, 0, "spare pooled row", "the spare-row re-publish failing is fatal");
        ck_eq(copy_calls, device_applies, "it is the last cudaMemcpy the tier makes");
        Session R;
        R.poison();
        reset_faults();
        fail_copy = device_applies;
        restore_bytes(good, R, path);
        ck(!Session::is_sentinel(R.stores[1].dead), "every other segment landed");
        // The pooled SEGMENT (rows [0, L/idx_block + 1)) landed, so the spare row holds the STALE value a
        // boundary dump can only produce, while `dead` holds the value the row must be given.  The invariant the
        // writers maintain at every block completion (`pooled[n_bid] == dead`, qsa.cu:213) is the one thing
        // missing - the session looks restored and is one invariant short of usable.
        ck(row_value(R.stores[1].pooled, SPARE_ROW, S.g.idx_key_dim) == 12.0f,
           "the spare row still holds the snapshot's STALE value");
        ck(row_value(R.stores[1].dead, 0, S.g.idx_key_dim) == 11.0f, "while `dead` holds the value it must become");
        ck(row_value(R.stores[0].pooled, SPARE_ROW, S.g.idx_key_dim) == 11.0f,
           "layer 0 got its re-publish: the half-applied state is per layer, not all-or-nothing");
    }
    {   // every write succeeded and the device will not confirm them: still fatal, because nothing proves they landed
        expect_fatal(0, 2, "after the apply pass", "a failed post-apply synchronize is a transfer failure");
        ck_eq(copy_calls, device_applies, "every segment was copied");
        ck_eq(sync_calls, 2, "and the tier stopped at the second sync");
        Session R;
        R.poison();
        reset_faults();
        fail_sync = 2;
        restore_bytes(good, R, path);
        ck(!R.untouched(), "the session holds the snapshot but no confirmation - the proof a reset needs is absent");
    }
    {   // the same file with a healthy device: restored, two syncs, every write accounted for
        reset_faults();
        Session R;
        R.poison();
        const Res r = restore_bytes(good, R, path);
        ck(r.kind == Restore::restored, "the same snapshot restores when nothing is injected");
        ck_eq(copy_calls, device_applies, "every device segment was copied");
        ck_eq(sync_calls, 2, "one sync before the writes and one after: the contract's two proofs");
        ck(!R.untouched(), "and the session holds it");
        ck(row_value(R.stores[0].pooled, SPARE_ROW, S.g.idx_key_dim) == 11.0f,
           "the spare row is the re-published one, as fixture 1 asserts");
        ck_no_pending_error("the successful restore");
    }

    // ---- the store reports the same classes, and adds one of its own ----
    {
        const std::string sdir = dir + "/store";
        strata::platform::KvNvmeStore store;
        reset_faults();
        ck(store.open(sdir, S.g, strata::core::qsa_kv_format(S.layers[0]), err), "the store opens");
        ck(store.dump(S.ss, S.draft, S.g, ids_of(L_BOUNDARY), {}, true, nullptr, err), "the store dumps");
        ck_eq(store.size(), 1, "one entry");
        const NvmeEntry& e = store.entries()[0];

        {   // a transfer failure through the store is the same class as through nvme_restore
            reset_faults();
            fail_copy = 2;
            Session R;
            R.poison();
            ck(store.restore(e, R.ss, R.draft, R.g, err) == Restore::transfer_failed,
               "the store reports a transfer failure exactly as the tier does");
            ck(err.find("host-to-device transfer failed") != std::string::npos,
               "and carries the tier's message, not a generic one");
            ck(!R.untouched(), "and the session it left is half-applied");
            ck_no_pending_error("the store's transfer failure");
        }
        {   // a corrupt file through the store is the recoverable class, with no CUDA call at all
            std::vector<uint8_t> bad = slurp(e.path);
            bad[strata::platform::kNvmeHeaderBytes + 8] ^= 0xFF;
            std::ofstream f(e.path, std::ios::binary);
            f.write((const char*) bad.data(), (std::streamsize) bad.size());
            f.close();
            reset_faults();
            Session R;
            R.poison();
            ck(store.restore(e, R.ss, R.draft, R.g, err) == Restore::invalid,
               "a corrupt stored snapshot is the recoverable class");
            ck(err.find("integrity check failed") != std::string::npos, "named by the tier's own reason");
            ck_eq(copy_calls, 0, "with no cudaMemcpy");
            ck(R.untouched(), "and nothing written");
        }
        {   // THE TOCTOU THE STORE ADDS: the file no longer matches its index, but it APPLIED CLEANLY.  The tier's
            // final sync succeeded, which is the proof the contract demands of a recovery - so this one IS
            // recoverable, and the caller may drop the entry and re-read the prompt.
            reset_faults();
            ck(strata::platform::nvme_dump(e.path.c_str(), S.ss, S.draft, S.g, ids_of(8), {}, true, err),
               "a second, shorter snapshot is written over the stored file");
            Session R;
            R.poison();
            reset_faults();   // count only what the RESTORE does
            ck(store.restore(e, R.ss, R.draft, R.g, err) == Restore::invalid,
               "a stale index is the recoverable class, not a fatal one");
            ck(err.find("entry changed under us") != std::string::npos, "and says what disagreed");
            ck_eq(copy_calls, device_applies, "the snapshot was FULLY applied");
            ck_eq(sync_calls, 2, "including the final sync: the device answered after the last write");
            ck_eq(R.ple_prev_last(), 107, "the session holds the NEW snapshot completely (its last token)");
            ck_no_pending_error("the stale-index case");
        }
    }

    // ---- a failed DUMP publishes nothing, which is why the dump side needs no failure contract ----
    {
        const std::string ddir = dir + "/dumpfail";
        strata::platform::KvNvmeStore store;
        reset_faults();
        ck(store.open(ddir, S.g, strata::core::qsa_kv_format(S.layers[0]), err), "the dump-failure store opens");
        Session P;
        P.seed_indexer(13.0f, L_CONSUMED / SHP.idx_block, 14.0f);
        P.poison();
        reset_faults();
        fail_copy = 1;
        ck(!store.dump(P.ss, P.draft, P.g, ids_of(L_BOUNDARY), {}, true, nullptr, err),
           "a dump whose device read fails returns false");
        ck(err.find("nvme_dump: gdn") != std::string::npos, "and names the segment it could not read");
        ck_eq(store.size(), 0, "no entry was added");
        int files = 0;
        for (const auto& de : fs::directory_iterator(ddir))
            if (de.path().filename().string().rfind("kv-", 0) == 0) ++files;
        ck_eq(files, 0, "and no partial file was left for the next scan to admit");
        ck(P.untouched(), "a dump only reads: the session it failed on is byte-for-byte what it was");
        ck_no_pending_error("the failed dump");
        reset_faults();
        ck(store.dump(P.ss, P.draft, P.g, ids_of(L_BOUNDARY), {}, true, nullptr, err),
           "the same dump succeeds once the device answers - a dump failure is not a sticky condition");
    }
}

// ================================ fixture 5: a store full of snapshots this build cannot read ================================
//
// STEP 3'S OPERATOR CASE.  `/local/strata/kvstore` holds 113 GB of version-2 snapshots and this build writes
// version 3.  The question the contract has to answer is whether that is a fatal misconfiguration or a
// recoverable condition - and the answer has to come from what the code actually does at startup.
void fixture_stale_store(const std::string& dir) {
    using Restore = strata::core::ConversationRestore;
    Session S;
    S.seed_indexer(21.0f, L_CONSUMED / SHP.idx_block, 22.0f);
    S.tag_kv();
    std::string err;
    const std::string snap = dir + "/good.bin";
    reset_faults();
    ck(strata::platform::nvme_dump(snap.c_str(), S.ss, S.draft, S.g, ids_of(L_BOUNDARY), {}, true, err),
       "the stale-store fixture has a v3 snapshot to copy");
    const std::vector<uint8_t> good = slurp(snap);

    const std::string v2dir = dir + "/v2store";
    for (int i = 1; i <= 3; ++i) {
        std::vector<uint8_t> v2 = good;
        uint32_t old = 2;
        std::memcpy(v2.data() + 4, &old, 4);
        std::ofstream f(v2dir + "/kv-1-" + std::to_string(i) + ".bin", std::ios::binary);
        f.write((const char*) v2.data(), (std::streamsize) v2.size());
        f.close();
    }

    strata::platform::KvNvmeStore store;
    bool opened = false;
#ifndef _WIN32
    const std::string noise = capture_stderr([&] {
        opened = store.open(v2dir, S.g, strata::core::qsa_kv_format(S.layers[0]), err);
    });
#else
    opened = store.open(v2dir, S.g, strata::core::qsa_kv_format(S.layers[0]), err);
    const std::string noise;
#endif
    ck(opened, "a store of snapshots this build cannot read is NOT a startup failure: the server still starts");
    ck(err.empty(), "and it is not reported as an error");
    ck_eq(store.size(), 0, "none of its files become entries, so nothing can be promoted from it");
    ck_eq((int64_t) store.total_bytes(), 0, "and none of its bytes count against the cap");
    ck(noise.find("format version 2") != std::string::npos, "the operator is told the version the store holds");
    ck(noise.find("this build writes version 3") != std::string::npos, "and the version this build writes");
    ck(noise.find("refuses older files") != std::string::npos, "and that they are refused, not converted");
    ck(noise.find("every request re-prefills") != std::string::npos, "and that the consequence is a re-prefill");
    ck(noise.find("re-dump them with the binary that wrote them") != std::string::npos,
       "and the first operator action: re-dump them with the binary that wrote them");
    ck(noise.find("remove them and let the store rebuild") != std::string::npos,
       "and the second: delete the store and let it rebuild. Not left ambiguous");
    int files = 0;
    for (const auto& de : fs::directory_iterator(v2dir))
        if (de.path().filename().string().rfind("kv-", 0) == 0) ++files;
    ck_eq(files, 3, "the stale files STAY on disk: a tier that cannot read them does not delete an operator's data");

    // THE CONSEQUENCE FOR THE NEXT REQUEST: no entry, so no promote, so a full re-prefill.  The same request that
    // fixture 1 shows promoting a v3 entry is the one used here, so the null result is the store's, not the
    // match's.
    const std::vector<int32_t> boundary = ids_of(L_BOUNDARY);
    std::vector<int64_t> request(boundary.begin(), boundary.end());
    for (int64_t i = 0; i < 6; ++i) request.push_back(900 + i);
    // The request's only picture is in the NEW message (token 16, past the stored prefix), so the entry's empty
    // image list is the right one to match against: `kv_nvme_match` filters the request's pictures below the
    // ENTRY's length, exactly as `checkpoint_at` filters them when it keys a checkpoint.
    const std::vector<ConversationImageKey> req_imgs = {{16, 0xCCCC}};
    ck(strata::platform::kv_nvme_match(store.entries(), request, req_imgs, true, 0) == nullptr,
       "a stale store re-prefills: the recoverable condition, proven by the match returning nothing");

    {   // the control that keeps the check above from being vacuous: the SAME files at version 3 do become entries
        const std::string v3dir = dir + "/v3store";
        for (int i = 1; i <= 3; ++i) {
            std::ofstream f(v3dir + "/kv-1-" + std::to_string(i) + ".bin", std::ios::binary);
            f.write((const char*) good.data(), (std::streamsize) good.size());
            f.close();
        }
        strata::platform::KvNvmeStore live;
        reset_faults();
        ck(live.open(v3dir, S.g, strata::core::qsa_kv_format(S.layers[0]), err), "the same store at version 3 opens");
        ck_eq(live.size(), 3, "and every file becomes an entry");
        ck(strata::platform::kv_nvme_match(live.entries(), request, req_imgs, true, 0) != nullptr,
           "and the same request promotes from it: the skip is the version, not the match");
        // and a v2 file read directly (not through a scan) is the recoverable class, with nothing written
        Session R;
        R.poison();
        reset_faults();
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        bool cvec = false;
        int64_t L = 0;
        ck(strata::platform::nvme_restore((v2dir + "/kv-1-1.bin").c_str(), R.ss, R.draft, R.g, ids, imgs, cvec, L,
                                         err) == Restore::invalid,
           "reading one stale file directly is the recoverable class too");
        ck_eq(copy_calls, 0, "with no cudaMemcpy");
        ck(R.untouched(), "and nothing written");
    }
}

// ================================ fixture 6: WHAT A DUMP REPORTS (TierActivity) ================================
//
// docs/nvme-kv-cache-web-design.md §3 wants the tier's own facts as a RETURN VALUE, not as a prose line.  Every
// number asserted here is one the store already knew - its entry bytes, its cap victims, its exact-match skip -
// so the check is always REPORT vs THE STORE'S OWN BOOKS: a counter that drifts from what the store did fails
// here rather than on the page.  No decision is re-litigated: the idempotent skip, the supersede rule and the
// "the last entry is kept even over the cap" policy are the ones fixtures 1-5 already exercise; this fixture
// only asks what the call said while it did them.
void fixture_activity(const std::string& dir) {
    using strata::platform::TierActivity;
    Session S;
    S.seed_indexer(41.0f, L_CONSUMED / SHP.idx_block, 42.0f);
    S.tag_kv();
    std::string err;
    const int kvf = strata::core::qsa_kv_format(S.layers[0]);

    {   // A WRITE, THEN THE IDEMPOTENT RE-DUMP OF IT, THEN THE SUPERSEDE OF A GROWN CONVERSATION
        strata::platform::KvNvmeStore store;
        ck(store.open(dir + "/report", S.g, kvf, err), "the report store opens");
        TierActivity w;
        ck(store.dump(S.ss, S.draft, S.g, ids_of(L_BOUNDARY), {}, true, nullptr, err, &w), "a snapshot is dumped");
        ck(!w.skipped, "a real write does not report a skip");
        ck_eq((int64_t) w.written, (int64_t) store.entries()[0].bytes, "written is the entry the store now holds");
        std::error_code ec;
        ck_eq((int64_t) w.written, (int64_t) fs::file_size(store.entries()[0].path, ec),
              "and the size of the file it just wrote");
        ck_eq(w.dropped, 0, "nothing was superseded");
        ck_eq(w.evicted, 0, "and nothing evicted (no cap set)");
        const uint64_t first_bytes = store.entries()[0].bytes;

        TierActivity r;
        ck(store.dump(S.ss, S.draft, S.g, ids_of(L_BOUNDARY), {}, true, nullptr, err, &r),
           "the same state is dumped again");
        ck(r.skipped, "the exact-match skip reports itself");
        ck_eq((int64_t) r.written, 0, "an idempotent re-dump writes ZERO bytes");
        ck_eq(r.dropped, 0, "and supersedes nothing");
        ck_eq(r.evicted, 0, "and evicts nothing");
        ck_eq(store.size(), 1, "the store still holds the ONE snapshot it already had");
        ck_eq((int64_t) store.total_bytes(), (int64_t) first_bytes, "and its books did not move");

        TierActivity g;
        ck(store.dump(S.ss, S.draft, S.g, ids_of(L_CONSUMED), {}, true, nullptr, err, &g),
           "the conversation grows and is dumped again");
        ck(!g.skipped, "a grown conversation is not a skip");
        ck_eq(g.dropped, 1, "the previous head of THIS process was superseded");
        ck_eq((int64_t) g.dropped_bytes, (int64_t) first_bytes, "with the superseded entry's own bytes");
        ck_eq((int64_t) g.written, (int64_t) store.entries()[0].bytes, "written is the NEW entry's bytes");
        ck_eq(store.size(), 1, "a conversation stays one snapshot");
        ck_eq((int64_t) store.total_bytes(), (int64_t) store.entries()[0].bytes, "and the books hold only it");
    }
    {   // THE CAP: the dump that pushes the store over the cap reports the eviction its enforce_cap caused
        const std::string cdir = dir + "/cap";
        strata::platform::KvNvmeStore store;
        ck(store.open(cdir, S.g, kvf, err), "the cap store opens");
        TierActivity a1;
        ck(store.dump(S.ss, S.draft, S.g, ids_of(L_BOUNDARY), {}, true, nullptr, err, &a1),
           "the first conversation dumps");
        const uint64_t stale_bytes = store.entries()[0].bytes;
        ck_eq(a1.evicted, 0, "a store under its cap evicts nothing");
        ::sleep(1);   // mtime resolution is SECONDS: make the age order the LRU rule reads real, not incidental
        store.set_cap_bytes((int64_t) stale_bytes);   // a second snapshot cannot fit beside the first
        std::vector<int32_t> other = ids_of(L_BOUNDARY);
        other[(size_t) L_BOUNDARY - 1] = 709;   // a DIFFERENT conversation: not an extension, so not superseded
        TierActivity a2;
        ck(store.dump(S.ss, S.draft, S.g, other, {}, true, nullptr, err, &a2), "the second conversation dumps");
        ck_eq(a2.dropped, 0, "and supersedes nothing: the supersede rule is untouched by the cap");
        ck_eq(a2.evicted, 1, "the cap dropped exactly one entry");
        ck_eq((int64_t) a2.evicted_bytes, (int64_t) stale_bytes, "and reports the victim's own bytes");
        ck_eq((int64_t) a2.written, (int64_t) store.entries()[0].bytes, "written is still only this call's write");
        ck_eq(store.size(), 1, "the store is back to one snapshot");
        ck(store.entries()[0].ids == other, "the victim was the OLDER entry: the LRU rule is unchanged");
        ck_eq((int64_t) store.total_bytes(), (int64_t) store.entries()[0].bytes, "and the books match what is left");

        // THE LAST ENTRY IS KEPT EVEN OVER THE CAP - reported as ZERO evictions, never as an emptied store.
        store.set_cap_bytes(1);
        std::vector<int32_t> grown = other;
        grown.push_back(900);   // the same conversation grown: the supersede leaves ONE entry, over the cap
        TierActivity a3;
        ck(store.dump(S.ss, S.draft, S.g, grown, {}, true, nullptr, err, &a3),
           "a dump under a 1-byte cap still writes");
        ck_eq(a3.dropped, 1, "the supersede still happened");
        ck_eq(a3.evicted, 0, "the cap evicted nothing: the last entry is kept even over the cap");
        ck_eq(store.size(), 1, "the store was not emptied");
        ck((int64_t) store.total_bytes() > 1, "even though it sits over the cap - the documented policy");
    }
}

/// The row count itself, at the boundaries that matter - including the non-aligned one C3 lives at.
void fixture_pooled_rows() {
    ck_eq(strata::kernels::qsa_pooled_rows(0, SHP), 0, "an empty prefix owns no rows");
    ck_eq(strata::kernels::qsa_pooled_rows(1, SHP), 1, "one token owns the spare row only");
    ck_eq(strata::kernels::qsa_pooled_rows(3, SHP), 1, "an incomplete block owns the spare row only");
    ck_eq(strata::kernels::qsa_pooled_rows(4, SHP), 2, "one completed block plus the spare");
    ck_eq(strata::kernels::qsa_pooled_rows(8, SHP), 3, "two completed blocks plus the spare");
    ck_eq(strata::kernels::qsa_pooled_rows(10, SHP), 3, "a NON-aligned boundary: 2 completed rows plus the spare");
    ck_eq(strata::kernels::qsa_pooled_rows(11, SHP), 3, "the count does not move until a block completes");
    ck(strata::kernels::qsa_pooled_rows(MAX_CELLS, SHP) <= MAX_CELLS / SHP.idx_block + 2,
       "the count always fits the array kv_plan allocates");
}

}  // namespace

int main() {
#ifndef _WIN32
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1);   // before any CUDA call: this fixture must not reach a busy GPU
#endif
    const std::string root = "/tmp/kv-nvme-host-test-" + std::to_string((long) ::getpid());
    std::error_code ec;
    fs::remove_all(root, ec);
    for (const char* sub : {"/boundary", "/boundary/full", "/images", "/refusals", "/failure", "/failure/store",
                            "/failure/dumpfail", "/stale", "/stale/v2store", "/stale/v3store"})
        fs::create_directories(root + sub, ec);

    fixture_pooled_rows();
    fixture_turn_boundary(root + "/boundary");
    fixture_image_filter(root + "/images");
    fixture_refusals(root + "/refusals");
    fixture_failure_contract(root + "/failure");
    fixture_stale_store(root + "/stale");
    fixture_activity(root + "/activity");

    fs::remove_all(root, ec);
    std::printf("kv_nvme_host_test: %d checks passed (%d refusals, %d transfer failures asserted); no CUDA context, "
                "no model\n",
                checks, refusals, fatalities);
    return 0;
}
