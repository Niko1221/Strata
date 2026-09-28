// src/platform/kv_nvme_host_test.cpp - the NVMe cold tier's FORMAT and RESUME rules, on the host, with NO CUDA
// context (docs/nvme-kv-cache-convergence.md step 4: the fixtures C1 / C2 / C3 / C4 owed).
//
// WHAT THIS IS.  `nvme_dump_at`, `nvme_restore` and `KvNvmeStore` are the tier, and until now the only tests that
// exercise them are three shell scripts that need a 24 GB model and the RTX 4090.  This fixture drives those same
// functions with a synthetic session whose "device" arrays are ordinary host buffers, so the envelope's rules -
// the turn-boundary key, the image filter, the pooled-row count, the spare-row re-publish, the version and
// geometry refusals, the drafter-ring window - are checkable on a machine running no model at all.
//
// HOW THE DEVICE IS STOOD IN.  The tier touches the device in four ways: `cudaMemcpy` for the indexer / gdn / ple
// segments, `cudaMemcpyAsync` in the drafter-ring refill, `kv_stream_reset`'s kernel launch, and a final
// `cudaDeviceSynchronize`.  The target links with `-Wl,--wrap=...` - the shared core's own
// `conversation_transfer_test` uses the same technique - and the wrappers below turn the first two into plain
// `memcpy`, the launch into a no-op and the sync into success.  `main` also forces `CUDA_VISIBLE_DEVICES=-1`
// before any CUDA call, so this fixture cannot reach a GPU even if it is run by hand on a busy machine.
//
// WHAT IT PROVES: the bytes and the decisions.  Every segment the tier writes and reads, the arithmetic that
// sizes them, the key a snapshot is stored under, the match that promotes one, and which refusal fires first.
//
// WHAT IT CANNOT PROVE, and no host fixture can:
//   * that the arrays are really device memory or that `cudaMemcpy` moves them - the wrappers replace it, so a
//     wrong `cudaMemcpyKind` would pass here;
//   * that `kv_stream_reset` refills the streamed layers' slots: its launch is a no-op, so the fixture asserts
//     nothing about a page table;
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
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (!dst || !src) return cudaErrorInvalidValue;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    if (!dst || !src) return cudaErrorInvalidValue;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() { return cudaSuccess; }
/// `kv_stream_reset` launches a kernel.  With no device the launch only SETS an error, and `check()` in
/// kv_stream.cu exits the process on it.  Reporting "no error" is what lets the tier run at all - and it is also
/// why this fixture asserts nothing about what that kernel would have done to the page table.
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }

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

int checks = 0, refusals = 0;
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

  private:
    int draft_k_tag = -1, draft_ks_tag = -1;
};

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
    ck(store.restore(e, R.ss, R.draft, R.g, err), "the snapshot restores");
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
        return strata::platform::nvme_restore(path.c_str(), into.ss, into.draft, into.g, ids, imgs, cvec, L, e);
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
    for (const char* sub : {"/boundary", "/boundary/full", "/images", "/refusals"})
        fs::create_directories(root + sub, ec);

    fixture_pooled_rows();
    fixture_turn_boundary(root + "/boundary");
    fixture_image_filter(root + "/images");
    fixture_refusals(root + "/refusals");

    fs::remove_all(root, ec);
    std::printf("kv_nvme_host_test: %d checks passed (%d refusals asserted); no CUDA context, no model\n",
                checks, refusals);
    return 0;
}
