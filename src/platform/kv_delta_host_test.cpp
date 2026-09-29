// src/platform/kv_delta_host_test.cpp - the delta tier's RECORD FAMILY on the host (Phase 1:
// docs/nvme-delta-cache-handoff.md §6).  No CUDA context, no model.
//
// WHAT THIS IS.  Phase 1 is the foundation the byte-identity invariant (§5.2) stands on: the three record
// structs pinned by static_asserts, the pure byte math that cuts the v3 snapshot's segments into chunk and
// State payloads, and the chunk/State record I/O with footer verification.  This fixture checks each piece
// against arithmetic computed HERE rather than read from the tier, so a change to either shows up as a
// disagreement.  The one cross-check that looks past Phase 1: sealed chunks + the State record must add up to
// the v3 payload's own segment arithmetic, because that sum is what §5.2's byte-identity oracle will `cmp`.
//
// WHAT IT CANNOT PROVE: that the writer's CUDA copies put the right bytes in the slices (Phase 2's oracle),
// that a real SSD persists on fsync (no host fixture can), or that the manifest scan admits the right files
// (Phase 4).

#include "strata/platform/kv_delta.hpp"

#include "strata/core/conversation_snapshot.hpp"  // conversation_state_sizes
#include "strata/kernels/qsa.hpp"                 // qsa_real_shapes, qsa_pooled_rows

#include <cuda_runtime.h>

#include <cstring>

// ---- the host stand-in for the device (the target links -Wl,--wrap=<name>; see the file header) ----
// Copied from kv_nvme_host_test.cpp: Phase 1 needs only that the symbols EXIST (the record I/O makes no CUDA
// call, and the fixture asserts none), but Phase 2's writer drives wrapped copies, so the stand-in moves in now
// rather than being re-copied then.  The error state is MODELLED, not suppressed, for the same reason the v3
// fixture models it: a tier that handles a failure without consuming it must be caught HERE.
namespace {
int copy_calls = 0, sync_calls = 0;
int fail_copy = 0, fail_sync = 0;
cudaError_t pending_error = cudaSuccess;
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
    if (!dst || !src) { pending_error = cudaErrorInvalidValue; return pending_error; }
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    ++sync_calls;
    if (fail_sync != 0 && sync_calls == fail_sync) { pending_error = cudaErrorUnknown; return pending_error; }
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() {
    const cudaError_t e = pending_error;
    pending_error = cudaSuccess;
    return e;
}

#include <algorithm>
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

/// THE FLOOR EVERY FAILURE PATH MUST CLEAR (kv_nvme_host_test's rule): the tier reported its own failure, so it
/// must have consumed the CUDA error it handled - a pending error here would be misreported by the next caller.
void ck_no_pending_error(const char* what) {
    ++checks;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s left a CUDA error pending (%s) for the next caller to misread\n",
                     what, cudaGetErrorString(e));
        std::exit(1);
    }
}

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
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- the fixture: the same small geometry the v3 host fixture uses, with the REAL QSA widths ----

constexpr int64_t L_BOUNDARY = 10;   // NOT idx_block-aligned: sealed(10) = 8, a 2-token ragged tail

struct Fixture {
    ModelGeometry g;
    ConversationStateSizes z;
    strata::platform::DeltaShapes sh;
    std::vector<QsaState> layers;
    SessionState ss;
    QsaState mtp;
    std::vector<uint8_t> scratch;   // backs the "host arrays" (sizing reads only width and nullness)

    Fixture(bool drop_one_drafter_array = false) {
        g.n_embd = 2560; g.n_layers = 8; g.qsa_interval = 4;   // 2 QSA layers, 6 GDN layers
        g.ssm_state_size = 2; g.ssm_k_heads = 1; g.ssm_v_heads = 2;
        g.ssm_d_conv = 4; g.ssm_conv_channels = 8; g.ssm_value_dim = 1;
        g.n_head = 24; g.n_head_kv = 2; g.head_dim = 256; g.idx_q_heads = 4; g.idx_key_dim = 128;
        g.hc = 4; g.hc_lr = 320; g.n_expert = 512; g.n_ff = 640;
        std::string err;
        if (!conversation_state_sizes(g, z, err)) {
            std::fprintf(stderr, "sizing refused: %s\n", err.c_str());
            std::exit(1);
        }
        sh = strata::platform::delta_shapes();

        layers.resize((size_t) g.n_qsa_layers());
        ss.qsa_states = layers.data();
        ss.max_cells = 64;
        scratch.assign(4096, 0);
        uint8_t* p = scratch.data();
        for (QsaState& st : layers) {
            st.kv_int8 = true;
            st.kv_mode = 1;
            st.max_cells = 64;
            st.host.k_q = (int8_t*) p; st.host.v_q = (int8_t*) p;
            st.host.k_scale = (uint16_t*) p; st.host.v_scale = (uint16_t*) p;
        }
        ss.gdn_state = (float*) p;
        ss.ple_hist = (float*) p;
        mtp.kv_int8 = true;
        mtp.kv_mode = 2;
        mtp.max_cells = 64;
        mtp.host.k_q = (int8_t*) p; mtp.host.v_q = (int8_t*) p;
        mtp.host.k_scale = (uint16_t*) p; mtp.host.v_scale = (uint16_t*) p;
        if (drop_one_drafter_array) mtp.host.k_scale = nullptr;   // the v3 dump skips it; the delta tier must too
    }

    // ---- the arithmetic the fixture computes ITSELF, from the walk's primitives ----
    // Bytes one KV page of array `k` holds: n_head_kv * page_size * row_width (int8: head_dim; scale: (d/64)*2).
    int64_t page_bytes(int k) const {
        const int64_t w = k < 2 ? g.head_dim : (g.head_dim / 64) * 2;
        return g.n_head_kv * sh.shapes.page_size * w;
    }
    int64_t drafter_page_bytes() const {   // all four arrays, one page each
        return page_bytes(0) + page_bytes(1) + page_bytes(2) + page_bytes(3);
    }
    int64_t per_layer_chunk_bytes() const {   // one page of each array + one pooled row
        return page_bytes(0) + page_bytes(1) + page_bytes(2) + page_bytes(3) + g.idx_key_dim * 4;
    }
};

// ================================ the shapes and the byte math ================================

void fixture_shapes_and_slices() {
    Fixture F;
    ck_eq(F.sh.shapes.page_size, 4, "the real artifact's page size");
    ck_eq(F.sh.block, 4, "BLOCK = lcm(page_size, idx_block) = 4 with the real shapes");
    ck_eq(F.sh.rows_per_chunk, 1, "one sealed pooled row per chunk");

    // sealed(): the largest BLOCK multiple not exceeding T - the ragged tail is what is left over
    ck_eq(strata::platform::delta_sealed(0, F.sh), 0, "an empty boundary seals nothing");
    ck_eq(strata::platform::delta_sealed(3, F.sh), 0, "below one BLOCK everything is tail");
    ck_eq(strata::platform::delta_sealed(4, F.sh), 4, "exactly one BLOCK seals fully");
    ck_eq(strata::platform::delta_sealed(L_BOUNDARY, F.sh), 8, "a non-aligned boundary leaves a 2-token tail");

    // one sealed chunk: 1 page of each of the 4 arrays per layer, 1 pooled row per layer, then the drafter page
    const int64_t chunk0 = strata::platform::delta_chunk_payload_bytes(F.ss, F.mtp, F.g, F.sh, 0);
    ck_eq(chunk0, (int64_t) F.g.n_qsa_layers() * F.per_layer_chunk_bytes() + F.drafter_page_bytes(),
          "a chunk's payload is the per-layer page slices plus the pooled row plus the drafter page");
    ck_eq(chunk0, 13696, "and with this geometry that is 13696 bytes (4 tokens, ~26 KB/token as the handoff says)");

    // a chunk starting at or past the drafter's ring contributes no drafter pages (the T <= max_cells gate keeps
    // this out of production; the rule itself is §5.4's "only while a < max_cells")
    {
        Fixture G;
        G.mtp.max_cells = 4;
        const int64_t past = strata::platform::delta_chunk_payload_bytes(G.ss, G.mtp, G.g, G.sh, 4);
        ck_eq(past, (int64_t) G.g.n_qsa_layers() * G.per_layer_chunk_bytes(),
               "a chunk past the drafter ring carries no drafter pages");
    }
    // a null drafter array is skipped exactly as the v3 dump skips it (and as mtp_host counts it)
    {
        Fixture G(/*drop_one_drafter_array=*/true);
        const int64_t got = strata::platform::delta_chunk_payload_bytes(G.ss, G.mtp, G.g, G.sh, 0);
        ck_eq(got, (int64_t) G.g.n_qsa_layers() * G.per_layer_chunk_bytes() + G.drafter_page_bytes() - G.page_bytes(2),
              "a null drafter array contributes nothing to the chunk payload");
    }

    // THE PHASE-1 FORM OF §5.2: sealed chunks + the State record must add up to the v3 payload's own segment
    // arithmetic for the same boundary - ids/imgs excluded (the manifest carries those, not the chunks).
    {
        Fixture G;
        const int64_t T = L_BOUNDARY, S = strata::platform::delta_sealed(T, G.sh);
        const int64_t chunks = strata::platform::delta_chunk_payload_bytes(G.ss, G.mtp, G.g, G.sh, 0) +
                               strata::platform::delta_chunk_payload_bytes(G.ss, G.mtp, G.g, G.sh, S);
        const int64_t state = strata::platform::delta_state_payload_bytes(G.ss, G.mtp, G.g, G.z, G.sh, T);
        const int64_t pages3 = (T + G.sh.shapes.page_size - 1) / G.sh.shapes.page_size;      // v3: ceil(L/page)
        const int64_t rows3 = strata::kernels::qsa_pooled_rows(T, G.sh.shapes);              // v3: pooled rows
        const int64_t drafter_pages3 = pages3;   // T <= max_cells: the ring covers the whole prefix
        int64_t v3_body = 0;
        for (int64_t i = 0; i < G.g.n_qsa_layers(); ++i)
            v3_body += pages3 * (G.page_bytes(0) + G.page_bytes(1) + G.page_bytes(2) + G.page_bytes(3)) +
                       rows3 * G.g.idx_key_dim * 4 + (int64_t) (G.z.tail + G.z.dead + G.z.block_pos);
        v3_body += drafter_pages3 * G.drafter_page_bytes();
        ck_eq(chunks + state - ((int64_t) G.z.gdn + (int64_t) G.z.ple), v3_body,
              "sealed chunks + the State record add up to the v3 payload (the byte-identity precondition)");
    }

    // the State record's own arithmetic, at the boundaries that matter
    {
        Fixture G;
        const int64_t per_layer = G.page_bytes(0) + G.page_bytes(1) + G.page_bytes(2) + G.page_bytes(3);
        // T = 10, S = 8: one tail page per array, the spare row only, plus the checkpoint's three blobs
        int64_t want10 = (int64_t) G.z.gdn + (int64_t) G.z.ple;
        for (int64_t i = 0; i < G.g.n_qsa_layers(); ++i)
            want10 += per_layer + G.g.idx_key_dim * 4 + (int64_t) (G.z.tail + G.z.dead + G.z.block_pos);
        want10 += per_layer;   // the drafter's tail page
        ck_eq(strata::platform::delta_state_payload_bytes(G.ss, G.mtp, G.g, G.z, G.sh, 10), want10,
              "the T=10 state is one tail page + the spare row + the running state per layer");
        // T = 8, S = 8: no tail pages at all, but the spare row is ALWAYS in the state (qsa_pooled_rows(8) - 2 = 1)
        int64_t want8 = (int64_t) G.z.gdn + (int64_t) G.z.ple;
        for (int64_t i = 0; i < G.g.n_qsa_layers(); ++i)
            want8 += G.g.idx_key_dim * 4 + (int64_t) (G.z.tail + G.z.dead + G.z.block_pos);
        ck_eq(strata::platform::delta_state_payload_bytes(G.ss, G.mtp, G.g, G.z, G.sh, 8), want8,
              "a block-aligned boundary's state carries no tail pages but still the spare row");
        // T = 3, S = 0: everything is tail (one page) and the spare row is the only pooled row
        int64_t want3 = (int64_t) G.z.gdn + (int64_t) G.z.ple;
        for (int64_t i = 0; i < G.g.n_qsa_layers(); ++i)
            want3 += per_layer + G.g.idx_key_dim * 4 + (int64_t) (G.z.tail + G.z.dead + G.z.block_pos);
        want3 += per_layer;
        ck_eq(strata::platform::delta_state_payload_bytes(G.ss, G.mtp, G.g, G.z, G.sh, 3), want3,
              "a sub-BLOCK boundary's state is all tail plus the running state");
    }
}

// ================================ chunk records: round-trip and refusals ================================

void fixture_chunk_records(const std::string& dir) {
    const std::string chunks = dir + "/chunks";
    fs::create_directories(chunks);
    const uint64_t key = 0x1122334455667788ull;
    std::vector<uint8_t> payload(1000);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t) (i * 7 + 3);

    strata::platform::DeltaChunkHeader h;
    h.key = key; h.a = 4; h.b = 8; h.layers = 2; h.payload_bytes = (int64_t) payload.size();
    const std::string path = chunks + "/" + strata::platform::delta_key_name(key) + ".bin";
    ck(strata::platform::delta_key_name(key) == "1122334455667788", "the key's name is 16 lowercase hex digits");
    std::string err;
    ck(strata::platform::delta_write_chunk(path, h, payload.data(), payload.size(), err),
       ("a chunk writes: " + err).c_str());

    const std::vector<uint8_t> file = slurp(path);
    ck_eq((int64_t) file.size(), (int64_t) (strata::platform::kDeltaChunkHeaderBytes + payload.size() + 8),
          "the file is header + payload + an 8-byte footer");
    ck(!fs::exists(path + ".tmp"), "the temp name is gone once the rename landed");

    {   // the round-trip: byte-identical payload, and no temp file left behind
        std::vector<uint8_t> back;
        ck(strata::platform::delta_read_chunk(path, key, 4, 8, back, err), ("the chunk reads back: " + err).c_str());
        ck(back == payload, "the payload round-trips byte-identically");
    }
    {   // a wrong expected key is refused BY NAME, before the digest is even computed
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_chunk(path, key + 1, 4, 8, back, e);
        }, "key mismatch");
    }
    {   // a wrong range likewise - the caller's manifest says which chunk belongs where
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_chunk(path, key, 0, 4, back, e);
        }, "range mismatch");
    }
    {   // a foreign magic is a LAYOUT fact, reported before any integrity verdict
        std::vector<uint8_t> file2 = file;
        std::memcpy(file2.data(), "XXXX", 4);
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) file2.data(), (std::streamsize) file2.size());
        f.close();
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_chunk(path, key, 4, 8, back, e);
        }, "bad magic");
    }
    {   // a truncated tail fails the SIZE check (a layout fact), not the digest
        std::vector<uint8_t> file2 = file;
        file2.resize(file2.size() - 5);
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) file2.data(), (std::streamsize) file2.size());
        f.close();
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_chunk(path, key, 4, 8, back, e);
        }, "truncated or oversized");
    }
    {   // a flipped payload byte passes every layout check and fails ONLY the digest
        std::vector<uint8_t> file2 = file;
        file2[strata::platform::kDeltaChunkHeaderBytes + 40] ^= 0xFF;
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) file2.data(), (std::streamsize) file2.size());
        f.close();
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_chunk(path, key, 4, 8, back, e);
        }, "integrity check failed");
    }
    {   // a header that disagrees with its own payload is refused at WRITE time - a caller bug, not a disk state
        strata::platform::DeltaChunkHeader bad = h;
        bad.payload_bytes = 999;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_write_chunk(chunks + "/bad.bin", bad, payload.data(), payload.size(), e);
        }, "payload bytes");
    }
}

// ================================ state records: round-trip and refusals ================================

void fixture_state_records(const std::string& dir) {
    const std::string states = dir + "/states";
    fs::create_directories(states);
    std::vector<uint8_t> payload(777);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t) (i * 13 + 1);
    const uint64_t tag = 0xFEEDFACE12345678ull;

    std::string err;
    uint64_t key = 0;
    ck(strata::platform::delta_write_state(states, tag, payload.data(), payload.size(), key, err),
       ("a state record writes: " + err).c_str());
    ck(key == strata::platform::delta_state_key(tag, payload.data(), payload.size()),
       "the key is the payload FNV-1a seeded with the conversation's tag");
    ck(fs::exists(states + "/" + strata::platform::delta_key_name(key) + ".bin"),
       "the file sits under its content-addressed name");

    {   // the round-trip
        std::vector<uint8_t> back;
        ck(strata::platform::delta_read_state(states + "/" + strata::platform::delta_key_name(key) + ".bin", tag,
                                              key, back, err),
           ("the state reads back: " + err).c_str());
        ck(back == payload, "the state payload round-trips byte-identically");
    }
    {   // a different tag makes the same payload a DIFFERENT state record (cvec/geometry are key material, §5.7)
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_state(states + "/" + strata::platform::delta_key_name(key) + ".bin",
                                                      tag ^ 1, key, back, e);
        }, "key mismatch");
    }
    {   // a flipped payload byte: the footer (the bare payload hash) catches it first
        const std::string p = states + "/" + strata::platform::delta_key_name(key) + ".bin";
        std::vector<uint8_t> file2 = slurp(p);
        file2[strata::platform::kDeltaStateHeaderBytes + 10] ^= 0xFF;
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        f.write((const char*) file2.data(), (std::streamsize) file2.size());
        f.close();
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_state(p, tag, key, back, e);
        }, "integrity check failed");
    }
    {   // truncated: a layout refusal, before the digest
        const std::string p = states + "/trunc.bin";
        std::vector<uint8_t> file2 = slurp(states + "/" + strata::platform::delta_key_name(key) + ".bin");
        file2.resize(file2.size() - 9);
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        f.write((const char*) file2.data(), (std::streamsize) file2.size());
        f.close();
        std::vector<uint8_t> back;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_read_state(p, tag, key, back, e);
        }, "truncated or oversized");
    }
}

// ================================ the synthetic session (copied from kv_nvme_host_test.cpp) ================================

const strata::kernels::QsaShapes SHP = strata::kernels::qsa_real_shapes();   // page_size 4, idx_block 4
constexpr int64_t MAX_CELLS = 64;                       // 16 pages; the QSA layers' array
constexpr int64_t DRAFT_CELLS = 28;                     // the drafter's ring: 7 host pages; the delta gate is T <= 28
constexpr int64_t DRAFT_SLOTS = 2;

uint8_t block_tag(int array_id, int64_t block) { return (uint8_t) (1 + ((array_id * 37 + block * 11) % 250)); }
void tag_blocks(std::vector<uint8_t>& b, int array_id, size_t block_bytes) {
    for (size_t blk = 0; blk * block_bytes < b.size(); ++blk)
        for (size_t i = 0; i < block_bytes && blk * block_bytes + i < b.size(); ++i)
            b[blk * block_bytes + i] = block_tag(array_id, (int64_t) blk);
}

struct Store {
    std::vector<uint8_t> k, v, ks, vs;
    std::vector<uint8_t> slot_k, slot_v, slot_ks, slot_vs;
    std::vector<uint8_t> pooled, tail, dead, bpos;
    std::vector<int32_t> page_table, slot_block, slot_stamp, slot_ref, miss_block, miss_slot, ctl;
};

int64_t block_bytes(int array_id) {
    const int64_t rows = SHP.n_head_kv * SHP.page_size;
    return array_id < 2 ? rows * SHP.head_dim : rows * (SHP.head_dim / 64) * 2;
}

std::vector<int32_t> ids_of(int64_t n) {
    std::vector<int32_t> v((size_t) n);
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (int32_t) (100 + i);
    return v;
}

struct Session {
    ModelGeometry g;
    ConversationStateSizes z;
    strata::platform::DeltaShapes sh = strata::platform::delta_shapes();
    std::vector<Store> stores;
    std::vector<QsaState> layers;
    std::vector<uint8_t> gdn, ple;
    SessionState ss;
    Store draft_store;
    QsaState draft;

    Session() {
        g.n_embd = 2560; g.n_layers = 8; g.qsa_interval = 4;
        g.ssm_state_size = 2; g.ssm_k_heads = 1; g.ssm_v_heads = 2;
        g.ssm_d_conv = 4; g.ssm_conv_channels = 8; g.ssm_value_dim = 1;
        g.n_head = 24; g.n_head_kv = 2; g.head_dim = 256; g.idx_q_heads = 4; g.idx_key_dim = 128;
        g.hc = 4; g.hc_lr = 320; g.n_expert = 512; g.n_ff = 640;
        std::string err;
        if (!conversation_state_sizes(g, z, err)) { std::fprintf(stderr, "sizing refused: %s\n", err.c_str()); std::exit(1); }
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
        {   // the drafter: a ring (kv_mode 2), host copy authoritative
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
            s.pooled.assign(2 * g.idx_key_dim * 4, 0);
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
        if (ring) {
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

    void seed_indexer(float dead_value, int64_t completed_blocks, float stale_spare) {
        for (size_t i = 0; i < stores.size(); ++i) {
            Store& s = stores[i];
            const int64_t rows = (int64_t) s.pooled.size() / (g.idx_key_dim * 4);
            for (int64_t r = 0; r < rows; ++r) ((float*) s.pooled.data())[r * g.idx_key_dim] = 1000.0f + (float) r;
            for (int64_t b = 0; b < completed_blocks; ++b) ((float*) s.pooled.data())[b * g.idx_key_dim] = 1000.0f + (float) b;
            ((float*) s.pooled.data())[(L_BOUNDARY / SHP.idx_block) * g.idx_key_dim] = stale_spare;
            ((float*) s.dead.data())[0] = dead_value;
            std::fill(s.tail.begin(), s.tail.end(), (uint8_t) (0x30 + i));
            *(int32_t*) s.bpos.data() = (int32_t) ((L_BOUNDARY / SHP.idx_block) * SHP.idx_block);
        }
    }

    void tag_kv() {
        int array = 0;
        for (Store& s : stores) {
            tag_blocks(s.k, array++, (size_t) block_bytes(0));
            tag_blocks(s.v, array++, (size_t) block_bytes(0));
            tag_blocks(s.ks, array++, (size_t) block_bytes(2));
            tag_blocks(s.vs, array++, (size_t) block_bytes(2));
        }
        tag_blocks(draft_store.k, array++, (size_t) block_bytes(0));
        tag_blocks(draft_store.v, array++, (size_t) block_bytes(0));
        tag_blocks(draft_store.ks, array++, (size_t) block_bytes(2));
        tag_blocks(draft_store.vs, array++, (size_t) block_bytes(2));
    }
};

ConversationCheckpoint boundary_checkpoint(const Session& S, int64_t T, int32_t block_pos_at_boundary) {
    ConversationCheckpoint cp;
    cp.ids = ids_of(T);
    cp.imgs = {{T > 3 ? 3 : T - 1, 0xAAAA}};   // a picture INSIDE the prefix (T=3 has no room for start 3)
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

// ================================ THE ORACLE: reassemble a delta store into the v3 image ================================
//
// Built HERE with arithmetic computed independently of the tier, so a disagreement is the test's verdict.  The
// layout is the v3 envelope's (kv_nvme_host_test.cpp's Layout), filled from: the manifest (header fields, ids,
// imgs), the sealed chunks in key order (their payload is the v3 segments' own bytes for [a, a+BLOCK)), and the
// head State record (the ragged tail + the running state).
std::vector<uint8_t> reassemble(const Session& S, const std::string& delta_dir, const std::string& manifest_path,
                                uint64_t weights_fp) {
    using namespace strata::platform;
    std::string err;
    DeltaManifestHeader h;
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<DeltaChunkRef> refs;
    ck(delta_read_manifest(manifest_path, h, ids, imgs, refs, err), ("the manifest reads: " + err).c_str());
    const uint64_t tag = delta_tag(S.g, h.kv_format, h.cvec != 0, weights_fp, h.block);
    std::vector<uint8_t> state;
    ck(delta_read_state(delta_dir + "/states/" + delta_key_name(h.state_key) + ".bin", tag, h.state_key, state, err),
       ("the state reads: " + err).c_str());

    const int64_t T = h.L, page = h.page_size, blk = h.block, ib = h.idx_block;
    const int64_t sealed = (T / blk) * blk;
    const int64_t pagesT = (T + page - 1) / page;
    const int64_t rowsT = strata::kernels::qsa_pooled_rows(T, SHP);
    const int64_t pagesC = blk / page, rowsC = blk / ib;
    const int64_t nL = S.g.n_qsa_layers();
    const int64_t idx4 = S.g.idx_key_dim * 4;
    const int64_t pb[4] = {block_bytes(0), block_bytes(1), block_bytes(2), block_bytes(3)};
    const bool has_ple = S.ss.ple_hist != nullptr;

    // state payload offsets
    const int64_t st_tail_pages = pagesT - sealed / page, st_tail_rows = rowsT - sealed / ib;
    const int64_t st_layer = st_tail_pages * (pb[0] + pb[1] + pb[2] + pb[3]) + st_tail_rows * idx4 +
                             (int64_t) (S.z.tail + S.z.dead + S.z.block_pos);
    const int64_t st_drafter = (int64_t) S.z.gdn + (has_ple ? (int64_t) S.z.ple : 0) + nL * st_layer;
    // chunk payload offsets
    const int64_t ch_layer = pagesC * (pb[0] + pb[1] + pb[2] + pb[3]) + rowsC * idx4;
    const int64_t ch_drafter = nL * ch_layer;

    const size_t total = (size_t) (strata::platform::kNvmeHeaderBytes + (size_t) T * 4 +
                                   imgs.size() * sizeof(ConversationImageKey) + (size_t) S.z.gdn +
                                   (has_ple ? (size_t) S.z.ple : 0) +
                                   (size_t) nL * ((size_t) pagesT * (pb[0] + pb[1] + pb[2] + pb[3]) +
                                                  (size_t) rowsT * idx4 + S.z.tail + S.z.dead + S.z.block_pos) +
                                   (size_t) pagesT * (pb[0] + pb[1] + pb[2] + pb[3]) + 8);
    std::vector<uint8_t> buf(total, 0);

    NvmeHeader v3;
    v3.L = T;
    v3.n_imgs = (int64_t) imgs.size();
    v3.cvec = h.cvec;
    v3.kv_format = h.kv_format;
    v3.geometry = h.geometry;
    v3.page_size = h.page_size; v3.idx_block = h.idx_block; v3.max_cells = h.max_cells;
    v3.mtp_host = h.mtp_host;
    std::memcpy(buf.data(), &v3, sizeof v3);
    size_t at = sizeof v3;
    std::memcpy(buf.data() + at, ids.data(), ids.size() * 4); at += ids.size() * 4;
    if (!imgs.empty()) { std::memcpy(buf.data() + at, imgs.data(), imgs.size() * sizeof(ConversationImageKey)); at += imgs.size() * sizeof(ConversationImageKey); }
    std::memcpy(buf.data() + at, state.data(), S.z.gdn); at += S.z.gdn;   // gdn
    if (has_ple) { std::memcpy(buf.data() + at, state.data() + S.z.gdn, S.z.ple); at += S.z.ple; }   // ple

    auto layer_base = [&](int64_t i) { return at + (size_t) i * ((size_t) pagesT * (pb[0] + pb[1] + pb[2] + pb[3]) + (size_t) rowsT * idx4 + S.z.tail + S.z.dead + S.z.block_pos); };
    for (int64_t i = 0; i < nL; ++i) {
        const size_t lb = layer_base(i);
        int64_t v3_prefix = 0, ch_prefix = 0, st_prefix = 0;   // each walk keeps its OWN cumulative prefix
        for (int k = 0; k < 4; ++k) {
            const size_t dest = lb + (size_t) v3_prefix;
            int64_t written = 0;
            for (int64_t j = 0; j < (int64_t) refs.size(); ++j) {
                std::vector<uint8_t> payload;
                ck(delta_read_chunk(delta_dir + "/chunks/" + delta_key_name(refs[(size_t) j].key) + ".bin",
                                    refs[(size_t) j].key, j * blk, (j + 1) * blk, payload, err),
                   ("chunk " + std::to_string(j) + " reads: " + err).c_str());
                const size_t src = (size_t) (i * ch_layer + ch_prefix);
                std::memcpy(buf.data() + dest + (size_t) written, payload.data() + src, (size_t) (pagesC * pb[k]));
                written += pagesC * pb[k];
            }
            const size_t ssrc = (size_t) (S.z.gdn + (has_ple ? (int64_t) S.z.ple : 0) + i * st_layer + st_prefix);
            std::memcpy(buf.data() + dest + (size_t) written, state.data() + ssrc, (size_t) (st_tail_pages * pb[k]));
            written += st_tail_pages * pb[k];
            ck_eq(written, pagesT * pb[k], "the array's pages add up to the v3 segment");
            v3_prefix += pagesT * pb[k];
            ch_prefix += pagesC * pb[k];
            st_prefix += st_tail_pages * pb[k];
        }
        // pooled rows: the chunks' sealed rows, then the state's tail rows (the in-progress block + the spare)
        {
            const size_t dest = lb + (size_t) (pagesT * (pb[0] + pb[1] + pb[2] + pb[3]));
            int64_t written = 0;
            for (int64_t j = 0; j < (int64_t) refs.size(); ++j) {
                std::vector<uint8_t> payload;
                ck(delta_read_chunk(delta_dir + "/chunks/" + delta_key_name(refs[(size_t) j].key) + ".bin",
                                    refs[(size_t) j].key, j * blk, (j + 1) * blk, payload, err), err.c_str());
                const size_t src = (size_t) (i * ch_layer + pagesC * (pb[0] + pb[1] + pb[2] + pb[3]));
                std::memcpy(buf.data() + dest + (size_t) written, payload.data() + src, (size_t) (rowsC * idx4));
                written += rowsC * idx4;
            }
            const size_t ssrc = (size_t) (S.z.gdn + (has_ple ? (int64_t) S.z.ple : 0) + i * st_layer +
                                          st_tail_pages * (pb[0] + pb[1] + pb[2] + pb[3]));
            std::memcpy(buf.data() + dest + (size_t) written, state.data() + ssrc, (size_t) (st_tail_rows * idx4));
            written += st_tail_rows * idx4;
            ck_eq(written, rowsT * idx4, "the pooled rows add up to the v3 segment");
        }
        const size_t run = lb + (size_t) (pagesT * (pb[0] + pb[1] + pb[2] + pb[3]) + rowsT * idx4);
        std::memcpy(buf.data() + run, state.data() + (size_t) (S.z.gdn + (has_ple ? (int64_t) S.z.ple : 0) + i * st_layer +
                                                          st_tail_pages * (pb[0] + pb[1] + pb[2] + pb[3]) + st_tail_rows * idx4),
                    S.z.tail);
        std::memcpy(buf.data() + run + S.z.tail,
                    state.data() + (size_t) (S.z.gdn + (has_ple ? (int64_t) S.z.ple : 0) + i * st_layer +
                                             st_tail_pages * (pb[0] + pb[1] + pb[2] + pb[3]) + st_tail_rows * idx4 + (int64_t) S.z.tail),
                    S.z.dead);
        std::memcpy(buf.data() + run + S.z.tail + S.z.dead,
                    state.data() + (size_t) (S.z.gdn + (has_ple ? (int64_t) S.z.ple : 0) + i * st_layer +
                                             st_tail_pages * (pb[0] + pb[1] + pb[2] + pb[3]) + st_tail_rows * idx4 +
                                             (int64_t) S.z.tail + (int64_t) S.z.dead),
                    S.z.block_pos);
    }
    // the drafter: the chunks' pages for [0, S), then the state's tail pages
    {
        const size_t dbase = layer_base(nL);
        int64_t v3_prefix = 0, ch_prefix = 0, st_prefix = 0;
        for (int k = 0; k < 4; ++k) {   // all four drafter arrays non-null in this fixture
            const size_t dest = dbase + (size_t) v3_prefix;
            int64_t written = 0;
            for (int64_t j = 0; j < (int64_t) refs.size(); ++j) {
                std::vector<uint8_t> payload;
                ck(delta_read_chunk(delta_dir + "/chunks/" + delta_key_name(refs[(size_t) j].key) + ".bin",
                                    refs[(size_t) j].key, j * blk, (j + 1) * blk, payload, err), err.c_str());
                const size_t src = (size_t) ch_drafter + (size_t) ch_prefix;
                std::memcpy(buf.data() + dest + (size_t) written, payload.data() + src, (size_t) (pagesC * pb[k]));
                written += pagesC * pb[k];
            }
            const size_t ssrc = (size_t) st_drafter + (size_t) st_prefix;
            std::memcpy(buf.data() + dest + (size_t) written, state.data() + ssrc, (size_t) (st_tail_pages * pb[k]));
            written += st_tail_pages * pb[k];
            ck_eq(written, pagesT * pb[k], "the drafter array's pages add up to the v3 segment");
            v3_prefix += pagesT * pb[k];
            ch_prefix += pagesC * pb[k];
            st_prefix += st_tail_pages * pb[k];
        }
    }
    // the digest covers the PAYLOAD only - the 8 bytes this buffer reserves for the footer itself are not hashed
    // (nvme_restore's rule: hash [sizeof(NvmeHeader), at), where at is the walk end BEFORE the footer)
    const uint64_t digest = nvme_fnv1a(kNvmeFnvBasis, buf.data() + sizeof v3, buf.size() - sizeof v3 - 8);
    std::memcpy(buf.data() + buf.size() - 8, &digest, 8);
    return buf;
}


// ================================ fixture: the reader (Phase 3) ================================

bool all_zero(const std::vector<uint8_t>& b) { return std::all_of(b.begin(), b.end(), [](uint8_t x) { return x == 0; }); }

void fixture_reader(const std::string& root) {
    using Restore = strata::core::ConversationRestore;
    const uint64_t WFP = 0xDEADBEEF12345678ull;

    // one delta store, dumped at T=10; the same session dumped to a v3 file for the cross-check
    Session S;
    S.seed_indexer(987654.0f, L_BOUNDARY / SHP.idx_block, 555.0f);
    S.tag_kv();
    const ConversationCheckpoint cp = boundary_checkpoint(S, L_BOUNDARY, 8);
    const std::string dir = root + "/reader";
    std::string err;
    const bool d_ok = strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp,
                                                      WFP, 7, 1, err);
    ck(d_ok, ("the reader's store dumps: " + err).c_str());
    const std::string v3_path = root + "/reader-v3.bin";
    const bool v3_ok = strata::platform::nvme_dump_at(v3_path.c_str(), S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, err);
    ck(v3_ok, ("the reader's v3 control writes: " + err).c_str());

    strata::platform::NvmeEntry e;
    e.path = dir + "/log-7-1.manifest";
    e.ids = cp.ids;
    e.imgs = cp.imgs;
    e.L = L_BOUNDARY;
    e.cvec = true;
    e.kind = 1;

    {   // THE CROSS-CHECK: nvme_restore(file) and delta_restore(store) leave IDENTICAL sessions
        Session R1, R2;
        std::vector<int32_t> ids1, ids2;
        std::vector<ConversationImageKey> imgs1, imgs2;
        bool cvec1 = false, cvec2 = false;
        int64_t L1 = 0, L2 = 0;
        ck(strata::platform::nvme_restore(v3_path.c_str(), R1.ss, R1.draft, R1.g, ids1, imgs1, cvec1, L1, err) ==
               Restore::restored, ("the v3 restore of the control file: " + err).c_str());
        ck(strata::platform::delta_restore(e, R2.ss, R2.draft, R2.g, WFP, ids2, imgs2, cvec2, L2, err) ==
               Restore::restored, ("the delta restore: " + err).c_str());
        ck_eq(L1, L_BOUNDARY, "the v3 restore's length");
        ck_eq(L2, L_BOUNDARY, "the delta restore's length");
        ck(ids1 == ids2 && imgs1 == imgs2 && cvec1 == cvec2, "the two restores hand back the same prefix and images");
        for (int64_t i = 0; i < S.g.n_qsa_layers(); ++i) {
            const Store& a = R1.stores[(size_t) i];
            const Store& b = R2.stores[(size_t) i];
            ck(a.k == b.k && a.v == b.v && a.ks == b.ks && a.vs == b.vs, "the KV host copies are identical");
            ck(a.pooled == b.pooled && a.tail == b.tail && a.dead == b.dead && a.bpos == b.bpos,
               "the indexer state is identical");
        }
        ck(R1.gdn == R2.gdn && R1.ple == R2.ple, "the running state is identical");
        ck(R1.draft_store.k == R2.draft_store.k && R1.draft_store.ks == R2.draft_store.ks,
           "the drafter's host copy is identical");
        ck(R1.ss.ple_prev[0] == R2.ss.ple_prev[0] && R1.ss.ple_prev[1] == R2.ss.ple_prev[1],
           "and the PLE window");
        // and the delta restore is a REAL restore: the prefix actually landed in the fresh session
        const int64_t pages = (L_BOUNDARY + SHP.page_size - 1) / SHP.page_size;
        ck(std::equal(R2.stores[0].k.begin(), R2.stores[0].k.begin() + pages * block_bytes(0), S.stores[0].k.begin()),
           "the restored KV equals the dumper's, page for page");
        ck(((const float*) R2.stores[0].dead.data())[0] == 987654.0f, "the dead key is the checkpoint's");
        ck(((const float*) R2.stores[0].pooled.data())[(L_BOUNDARY / SHP.idx_block) * S.g.idx_key_dim] == 987654.0f,
           "C3: the spare row was re-published to the dead key");
    }

    auto restore_fresh = [&](Session& R) -> std::pair<Restore, std::string> {
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        bool cvec = false;
        int64_t L = 0;
        std::string e2;
        const Restore r = strata::platform::delta_restore(e, R.ss, R.draft, R.g, WFP, ids, imgs, cvec, L, e2);
        return {r, e2};
    };
    auto expect_invalid = [&](const char* must_name, const char* label) {
        reset_faults();
        Session R;   // fresh: every array zero - "untouched" is the all-zero claim
        const auto r = restore_fresh(R);
        ++refusals;
        ck(r.first == Restore::invalid, label);
        if (r.second.find(must_name) == std::string::npos) {
            std::fprintf(stderr, "FAIL: the refusal for '%s' was: %s\n", must_name, r.second.c_str());
            std::exit(1);
        }
        ck_eq(copy_calls, 0, "a refused delta restore made not one cudaMemcpy");
        ck_eq(sync_calls, 0, "and not one CUDA call at all");
        bool zero = all_zero(R.gdn) && all_zero(R.ple);
        for (const Store& s : R.stores)
            zero = zero && all_zero(s.k) && all_zero(s.v) && all_zero(s.ks) && all_zero(s.vs) &&
                   all_zero(s.pooled) && all_zero(s.tail) && all_zero(s.dead) && all_zero(s.bpos);
        ck(zero, "and wrote nothing to the session");
        ck_no_pending_error(label);
        last_error = r.second;
    };

    {   // a MISSING chunk file: the refusal names the chunk (its content-addressed name is in the path)
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<strata::platform::DeltaChunkRef> refs;
        strata::platform::DeltaManifestHeader m;
        ck(strata::platform::delta_read_manifest(e.path, m, ids, imgs, refs, err), "the manifest reads");
        const std::string victim =
            dir + "/chunks/" + strata::platform::delta_key_name(refs[1].key) + ".bin";
        const std::string kept = victim + ".kept";
        fs::rename(victim, kept);
        expect_invalid("chunks/", "a missing chunk is the recoverable class");
        ck(last_error.find(strata::platform::delta_key_name(refs[1].key)) != std::string::npos,
           "and the refusal names the chunk's key");
        fs::rename(kept, victim);
    }
    {   // a TRUNCATED chunk
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<strata::platform::DeltaChunkRef> refs;
        strata::platform::DeltaManifestHeader m;
        ck(strata::platform::delta_read_manifest(e.path, m, ids, imgs, refs, err), "the manifest reads");
        const std::string victim = dir + "/chunks/" + strata::platform::delta_key_name(refs[0].key) + ".bin";
        const std::vector<uint8_t> good = slurp(victim);
        std::ofstream f(victim, std::ios::binary | std::ios::trunc);
        f.write((const char*) good.data(), (std::streamsize) (good.size() - 5));
        f.close();
        expect_invalid("truncated or oversized", "a truncated chunk is the recoverable class");
        { std::ofstream f2(victim, std::ios::binary | std::ios::trunc);
          f2.write((const char*) good.data(), (std::streamsize) good.size()); }
    }
    {   // a corrupt MANIFEST footer: the manifest is untrustworthy, refused before any chunk is read
        const std::string p = e.path;
        const std::vector<uint8_t> clean = slurp(p);
        std::vector<uint8_t> bad = clean;
        bad[bad.size() - 1] ^= 0xFF;
        { std::ofstream f(p, std::ios::binary | std::ios::trunc);
          f.write((const char*) bad.data(), (std::streamsize) bad.size()); }
        expect_invalid("integrity check failed", "a corrupt manifest footer is the recoverable class");
        { std::ofstream f(p, std::ios::binary | std::ios::trunc);
          f.write((const char*) clean.data(), (std::streamsize) clean.size()); }   // repaired: the store re-dumps
    }
    {   // a weight-set mismatch: match-time refusal, re-checked at restore
        reset_faults();
        Session R;
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        bool cvec = false;
        int64_t L = 0;
        std::string e2;
        ck(strata::platform::delta_restore(e, R.ss, R.draft, R.g, WFP + 1, ids, imgs, cvec, L, e2) ==
               Restore::invalid, "a different weight set is refused");
        ck(e2.find("different weight set") != std::string::npos, "and says which rule fired");
        ck_eq(copy_calls, 0, "with no CUDA call");
    }
    // and the SAME manifest restores again once the file is fixed - a refusal is not a sticky condition
    {
        Session R;
        const auto r = restore_fresh(R);
        ck(r.first == Restore::restored, "the manifest restores again after the corruption is repaired");
    }
}

int count_files(const std::string& dir, const char* prefix) {
    int n = 0;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec))
        if (de.path().filename().string().rfind(prefix, 0) == 0) ++n;
    return n;
}

/// Records under their REAL (content-addressed) names only - the `.tmp-*` residue a crash leaves is not one.
int count_real(const std::string& dir) {
    int n = 0;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec)) {
        const std::string name = de.path().filename().string();
        if (name.rfind(".tmp-", 0) != 0) ++n;
    }
    return n;
}

// ================================ fixture: THE byte-identity oracle (§5.2) ================================

void fixture_byte_identity(const std::string& root) {
    const uint64_t WFP = 0xDEADBEEF12345678ull;   // any fixed weight-set fingerprint - it rides in the manifest
    const int64_t Ts[] = {3, 4, 5, 10, 18};   // sub-BLOCK, exactly BLOCK, BLOCK+1, mid, several BLOCKs
    for (int64_t T : Ts) {
        Session S;
        S.seed_indexer(987654.0f, T / SHP.idx_block, 555.0f);
        S.tag_kv();
        const int32_t bp = (int32_t) ((T / SHP.idx_block) * SHP.idx_block);
        const ConversationCheckpoint cp = boundary_checkpoint(S, T, bp);
        const std::string v3_path = root + "/v3-" + std::to_string(T) + ".bin";
        const std::string dir = root + "/delta-" + std::to_string(T);
        std::string err;
        const bool v3_ok = strata::platform::nvme_dump_at(v3_path.c_str(), S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, err);
        ck(v3_ok, ("the v3 dump writes for T=" + std::to_string(T) + ": " + err).c_str());
        const bool d_ok = strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp,
                                                          WFP, 1, 1, err);
        ck(d_ok, ("the delta dump writes for T=" + std::to_string(T) + ": " + err).c_str());
        const std::vector<uint8_t> want = slurp(v3_path);
        const std::vector<uint8_t> got = reassemble(S, dir, dir + "/log-1-1.manifest", WFP);
        ck(got.size() == want.size(),
           ("T=" + std::to_string(T) + ": the reassembly is the v3 image's size (got " +
            std::to_string(got.size()) + ", want " + std::to_string(want.size()) + ")").c_str());
        if (got != want) {
            for (size_t i = 0; i < got.size() && i < want.size(); ++i)
                if (got[i] != want[i]) {
                    std::fprintf(stderr, "FAIL: T=%lld: first differing byte at offset %zu (got %02x, want %02x)\n",
                                 (long long) T, i, got[i], want[i]);
                    std::fprintf(stderr, "  size %zu; z.gdn %zu z.ple %zu z.tail %zu z.dead %zu z.bpos %zu\n",
                                 want.size(), S.z.gdn, S.z.ple, S.z.tail, S.z.dead, S.z.block_pos);
                    std::exit(1);
                }
        }
        ck(got == want, ("T=" + std::to_string(T) + ": the reassembled image is BYTE-IDENTICAL to nvme_dump_at's").c_str());
    }
}

// ================================ fixture: growth, idempotence, forks, failure, crash hooks ================================

/// Dumps `T` into a fresh delta dir and returns the head the manifest describes (what the next dump appends to).
strata::platform::DeltaHead dump_head(const Session& S, int64_t T, const std::string& dir, int64_t seq,
                                      const strata::platform::DeltaHead* prev, uint64_t wfp) {
    const ConversationCheckpoint cp = boundary_checkpoint(S, T, (int32_t) ((T / SHP.idx_block) * SHP.idx_block));
    std::string err;
    ck(strata::platform::delta_dump_at(prev, dir, S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, wfp, 7, seq, err),
       ("the delta dump writes at T=" + std::to_string(T) + ": " + err).c_str());
    strata::platform::DeltaHead head;
    strata::platform::DeltaManifestHeader h;
    std::vector<ConversationImageKey> imgs;
    std::vector<strata::platform::DeltaChunkRef> refs;
    ck(strata::platform::delta_read_manifest(dir + "/log-7-" + std::to_string(seq) + ".manifest", h, head.ids, imgs,
                                             refs, err),
       ("the manifest reads back: " + err).c_str());
    head.L = h.L;
    head.path = dir + "/log-7-" + std::to_string(seq) + ".manifest";
    return head;
}

void fixture_writer_semantics(const std::string& root) {
    const uint64_t WFP = 0xDEADBEEF12345678ull;

    {   // GROWTH: dump at T1, extend the conversation, dump at T2 - only NEW sealed chunks hit the disk
        const std::string dir = root + "/growth";
        Session S;
        S.seed_indexer(1.0f, 4, 2.0f);
        S.tag_kv();
        const strata::platform::DeltaHead h1 = dump_head(S, 10, dir, 1, nullptr, WFP);
        ck_eq(h1.L, 10, "the head is the boundary it was dumped at");
        ck_eq(count_files(dir + "/chunks", ""), 2, "sealed(10) = 8 = two chunks on a fresh store");
        // the extended session: the SAME arrays (the KV below T1 is untouched by the generation that followed),
        // a boundary at 18
        const strata::platform::DeltaHead h2 = dump_head(S, 18, dir, 2, &h1, WFP);
        ck_eq(h2.L, 18, "the second head is the grown boundary");
        ck_eq(count_files(dir + "/chunks", ""), 4, "sealed(18) = 16 = four chunks, of which two are the first dump's");
        ck(count_files(dir, "log-") == 1, "the previous head's manifest was unlinked: one head per conversation");
        // the byte-identity oracle ALSO holds for the grown head (the §5.2 invariant is per boundary, not per dump)
        const std::string v3_path = root + "/growth-v3.bin";
        const ConversationCheckpoint cp = boundary_checkpoint(S, 18, (int32_t) (16));
        std::string err;
        const bool v3_ok = strata::platform::nvme_dump_at(v3_path.c_str(), S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, err);
        ck(v3_ok, ("the v3 dump of the grown boundary writes: " + err).c_str());
        const std::vector<uint8_t> want = slurp(v3_path);
        const std::vector<uint8_t> got = reassemble(S, dir, dir + "/log-7-2.manifest", WFP);
        ck(got == want, "the GROWN conversation's reassembly is byte-identical to nvme_dump_at's");
    }
    {   // IDEMPOTENCE at the same T: no new chunk files, and the manifest body is unchanged
        const std::string dir = root + "/idempotent";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        const strata::platform::DeltaHead h1 = dump_head(S, 10, dir, 1, nullptr, WFP);
        const int before = count_files(dir + "/chunks", "");
        const strata::platform::DeltaHead h2 = dump_head(S, 10, dir, 2, &h1, WFP);
        ck_eq(h2.L, 10, "the re-dump is keyed at the same boundary");
        ck_eq(count_files(dir + "/chunks", ""), before, "an idempotent re-dump writes ZERO new chunk files");
        ck(count_files(dir, "log-") == 1, "and leaves exactly one head");
    }
    {   // FORK: same prefix, different suffix - the shared chunks are the SAME files, both heads valid
        const std::string dir = root + "/fork";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        const strata::platform::DeltaHead h1 = dump_head(S, 10, dir, 1, nullptr, WFP);
        // the two chunks the parent head sealed, by name - the fork must REUSE exactly these files
        std::vector<std::string> parent_chunks;
        std::error_code ec;
        for (const auto& de : fs::directory_iterator(dir + "/chunks", ec))
            parent_chunks.push_back(de.path().filename().string());
        ck_eq((int64_t) parent_chunks.size(), 2, "the parent sealed two chunks");
        // the fork: a second conversation branching at the boundary - new ids share the first 10 tokens
        std::vector<int32_t> forked = ids_of(14);          // the SAME first 10 tokens, then a different tail
        for (int64_t i = 10; i < 14; ++i) forked[(size_t) i] = 700 + (int32_t) i;
        ConversationCheckpoint cp;
        cp.ids = forked;
        cp.imgs = {};
        cp.gdn.assign(S.z.gdn, 0xD1); cp.ple.assign(S.z.ple, 0xD2);
        cp.tails.assign((size_t) S.g.n_qsa_layers() * S.z.tail, 0xD3);
        cp.dead.assign((size_t) S.g.n_qsa_layers() * S.z.dead, 0);
        cp.block_pos.assign((size_t) S.g.n_qsa_layers() * S.z.block_pos, 0);
        std::string err;
        const bool fork_ok = strata::platform::delta_dump_at(&h1, dir, S.ss, S.draft, S.g, forked, {}, true, &cp, WFP, 7, 3, err);
        ck(fork_ok, ("the fork's dump writes: " + err).c_str());
        ck_eq(count_files(dir + "/chunks", ""), 3,
              "sealed(14) = 12 = three chunks; the two below the fork point are the SAME files the parent used");
        for (const std::string& name : parent_chunks)
            ck(fs::exists(dir + "/chunks/" + name), ("the parent's chunk " + name + " is still on disk - shared, not rewritten").c_str());
        // the fork's manifest references the parent's chunks BY KEY for the shared prefix (derived from the ids,
        // never stored - which is exactly why sharing needs no parent references)
        strata::platform::DeltaManifestHeader m;
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<strata::platform::DeltaChunkRef> rf;
        ck(strata::platform::delta_read_manifest(dir + "/log-7-3.manifest", m, ids, imgs, rf, err),
           "the fork's manifest reads");
        ck_eq((int64_t) rf.size(), 3, "three chunk refs");
        for (int j = 0; j < 2; ++j)
            ck(fs::exists(dir + "/chunks/" + strata::platform::delta_key_name(rf[(size_t) j].key) + ".bin"),
               "the fork's chunk ref names a file that existed before the fork");
        ck(count_files(dir, "log-") == 1,
           "the fork superseded the process's previous head (a strict-prefix extension IS the growth case; the store's "
           "own head tracking is what distinguishes conversations, and both tiers' chunks survive either way)");
    }
    {   // DUMP-SIDE FAILURE: a failing device copy aborts with NO manifest and NO head move
        const std::string dir = root + "/dumpfail";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        reset_faults();
        fail_copy = 1;   // the first pooled-rows D2H copy fails, before any file is written
        const ConversationCheckpoint cp = boundary_checkpoint(S, 10, 8);
        std::string err;
        ck(!strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, WFP, 7, 1, err),
           "a dump whose device read fails returns false");
        ck(err.find("pooled rows") != std::string::npos, "and names the copy that failed");
        ck_eq(count_files(dir, "log-"), 0, "no manifest was written");
        ck_eq(count_files(dir + "/chunks", ""), 0, "and no chunk either: the copy fails before the first write");
        ++checks;
        if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr, "FAIL: the failed dump left a CUDA error pending\n"); std::exit(1); }
        // and the same dump succeeds once the device answers - a dump failure is not a sticky condition
        reset_faults();
        const bool retry_ok = strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, WFP, 7, 1, err);
        ck(retry_ok, ("the retry succeeds: " + err).c_str());
    }
    {   // REFUSALS: everything nvme_dump_at refuses, the delta tier refuses too - plus its own gate
        const std::string dir = root + "/refusals";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        const ConversationCheckpoint cp = boundary_checkpoint(S, 10, 8);
        std::string err;
        expect_fail([&](std::string& e) {
            return strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, cp.ids, {}, true, nullptr, 0, 1, 1, e);
        }, "turn boundaries only");
        expect_fail([&](std::string& e) {   // T past the drafter ring: the fallback message, and the caller falls back
            const ConversationCheckpoint cp30 = boundary_checkpoint(S, 30, 28);
            return strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, ids_of(30), {}, true,
                                                   &cp30, 0, 1, 1, e);
        }, "exceeds the drafter ring");
        ConversationCheckpoint split = boundary_checkpoint(S, 10, 8);
        split.stage_parts.resize(1);   // a layer-split session is inert for both tiers
        expect_fail([&](std::string& e) {
            return strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, split.ids, {}, true, &split, 0, 1, 1, e);
        }, "layer-split");
        expect_fail([&](std::string& e) {   // an image at the boundary itself: outside the prefix
            return strata::platform::delta_dump_at(nullptr, dir, S.ss, S.draft, S.g, cp.ids, {{10, 0xB}}, true, &cp, 0, 1, 1, e);
        }, "not inside the 10-token prefix");
    }
    {   // THE CRASH HOOKS: each Ck leaves exactly the disk state the matrix says, and never a torn head
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        // the crash hooks fire on a GROWN dump (T=10 -> T=18): the second write of a growing conversation, which
        // is the scenario the matrix's supersede rows describe
        const ConversationCheckpoint cp = boundary_checkpoint(S, 18, 16);
        auto dump_with = [&](const char* at, const std::string& dir, const strata::platform::DeltaHead* prev,
                             int64_t seq) {
            if (at) setenv("STRATA_DELTA_FAIL_AT", at, 1); else unsetenv("STRATA_DELTA_FAIL_AT");
            std::string err;
            const bool ok = strata::platform::delta_dump_at(prev, dir, S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp,
                                                            WFP, 7, seq, err);
            unsetenv("STRATA_DELTA_FAIL_AT");
            return std::make_pair(ok, err);
        };
        {   // C1: a .tmp-* residue, nothing under a real name
            const std::string dir = root + "/c1";
            const auto r = dump_with("C1", dir, nullptr, 1);
            ck(!r.first, "C1 aborts the dump");
            ck_eq(count_real(dir + "/chunks"), 0, "C1 leaves no chunk under its real name");
            ck_eq(count_files(dir + "/chunks", ".tmp-"), 1, "C1 leaves exactly the temp the sweep will reclaim");
            ck_eq(count_files(dir, "log-"), 0, "and no manifest");
        }
        {   // C2: the first chunk is durable, nothing else
            const std::string dir = root + "/c2";
            const auto r = dump_with("C2", dir, nullptr, 1);
            ck(!r.first, "C2 aborts the dump");
            ck_eq(count_real(dir + "/chunks"), 1, "C2 leaves the first chunk durable");
            ck_eq(count_files(dir + "/chunks", ".tmp-"), 0, "and no temp");
            ck_eq(count_files(dir, "log-"), 0, "and no manifest");
        }
        {   // C3: chunks + state, still no manifest
            const std::string dir = root + "/c3";
            const auto r = dump_with("C3", dir, nullptr, 1);
            ck(!r.first, "C3 aborts the dump");
            ck_eq(count_real(dir + "/chunks"), 4, "C3 leaves all the new sealed chunks (sealed(18) = 16 = four)");
            ck_eq(count_real(dir + "/states"), 1, "and the state record");
            ck_eq(count_files(dir, "log-"), 0, "and still no manifest");
        }
        {   // C4 - THE INTERESTING ROW: the new head is committed, the old head STILL EXISTS; both reassemble
            const std::string dir = root + "/c4";
            const strata::platform::DeltaHead h1 = dump_head(S, 10, dir, 1, nullptr, WFP);
            const auto r = dump_with("C4", dir, &h1, 2);
            ck(!r.first, "C4 aborts the dump after the new manifest is durable");
            ck_eq(count_files(dir, "log-"), 2, "BOTH heads are on disk: the new head was complete before the old vanished");
            // byte-identity for both heads, each against its own boundary
            {
                const std::string v3a = root + "/c4-v3a.bin", v3b = root + "/c4-v3b.bin";
                const ConversationCheckpoint cp10 = boundary_checkpoint(S, 10, 8);
                const ConversationCheckpoint cp18 = boundary_checkpoint(S, 18, 16);
                std::string err;
                const bool a_ok = strata::platform::nvme_dump_at(v3a.c_str(), S.ss, S.draft, S.g, cp10.ids, cp10.imgs, true, &cp10, err);
                ck(a_ok, ("C4's v3 control a writes: " + err).c_str());
                const bool b_ok = strata::platform::nvme_dump_at(v3b.c_str(), S.ss, S.draft, S.g, cp18.ids, cp18.imgs, true, &cp18, err);
                ck(b_ok, ("C4's v3 control b writes: " + err).c_str());
                if (!(reassemble(S, dir, dir + "/log-7-1.manifest", WFP) == slurp(v3a))) { std::fprintf(stderr, "FAIL: C4: the OLD head no longer reassembles byte-identically\n"); std::exit(1); } ++checks;
                if (!(reassemble(S, dir, dir + "/log-7-2.manifest", WFP) == slurp(v3b))) { std::fprintf(stderr, "FAIL: C4: the NEW head does not reassemble byte-identically\n"); std::exit(1); } ++checks;
            }
        }
        {   // C5: the old head is gone; the new head is the only one - the final state
            const std::string dir = root + "/c5";
            const strata::platform::DeltaHead h1 = dump_head(S, 10, dir, 1, nullptr, WFP);
            const auto r = dump_with("C5", dir, &h1, 2);
            ck(!r.first, "C5 aborts the dump after the old head was unlinked");
            ck_eq(count_files(dir, "log-"), 1, "exactly one head: the new one");
            ck(!fs::exists(dir + "/log-7-1.manifest"), "the old head's manifest is gone");
        }
    }
}

}  // namespace

int main() {
#ifndef _WIN32
    setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    const std::string root = "/tmp/kv-delta-host-test-" + std::to_string((long) ::getpid());
    std::error_code ec;
    fs::remove_all(root, ec);

    fixture_shapes_and_slices();
    fixture_chunk_records(root);
    fixture_state_records(root);

    // the claim the file header makes, asserted: Phase 1's record I/O is pure file work - not one CUDA call.
    ck_eq(copy_calls, 0, "the record family made not one cudaMemcpy");
    ck_eq(sync_calls, 0, "and not one cudaDeviceSynchronize");
    ck(cudaGetLastError() == cudaSuccess, "and left no CUDA error pending");
    reset_faults();   // from here the WRITER's device reads count: Phase 2's pooled rows go through cudaMemcpy

    fixture_byte_identity(root);
    fixture_writer_semantics(root);
    fixture_reader(root);

    fs::remove_all(root, ec);
    std::printf("kv_delta_host_test: %d checks passed; no CUDA context, no model\n", checks);
    return 0;
}
