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
#include <random>
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

/// THE DISK'S OWN ANSWER to what a sweep did: every record under `delta_dir`'s chunks/ and states/, counted and
/// summed.  A `TierActivity` sweep report is checked against the DIFFERENCE of two calls to this, so the counter
/// is measured against the records that actually left the store rather than against arithmetic restated here.
std::pair<int, uint64_t> delta_records(const std::string& delta_dir) {
    int n = 0;
    uint64_t bytes = 0;
    std::error_code ec;
    for (const char* sub : {"/chunks", "/states"})
        for (const fs::directory_entry& de : fs::directory_iterator(delta_dir + sub, ec)) {
            if (ec) break;
            if (!de.is_regular_file()) continue;
            ++n;
            bytes += (uint64_t) de.file_size(ec);
        }
    return {n, bytes};
}

/// Bytes the files directly under `dir` hold: used to name the chunk files a grown dump REUSES, which are
/// therefore NOT that turn's write.
uint64_t dir_bytes(const std::string& dir) {
    uint64_t bytes = 0;
    std::error_code ec;
    for (const fs::directory_entry& de : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (de.is_regular_file()) bytes += (uint64_t) de.file_size(ec);
    }
    return bytes;
}

// ================================ fixture: the store (Phase 4) ================================

void fixture_store(const std::string& root) {
    {   // MIXED-DIR SCAN: v3 snapshots and delta manifests coexist; each store scans its own family into the
        // shared NvmeEntry vocabulary with the right kind
        const std::string dir = root + "/mixed";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        const ConversationCheckpoint cp = boundary_checkpoint(S, 10, 8);
        strata::platform::KvNvmeStore v3;
        std::string err;
        strata::platform::KvNvmeStore nov3;
        ck(v3.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), err), "the v3 store opens");
        ck(v3.dump(S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, err), "the v3 half dumps");
        strata::platform::KvDeltaStore delta;
        ck(delta.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
           ("the delta store opens beside it: " + err).c_str());
        ck(delta.dump(S.ss, S.draft, S.g, cp.ids, cp.imgs, true, &cp, err), "the delta half dumps");
        ck_eq((int64_t) v3.size(), 1, "one v3 entry");
        ck_eq((int64_t) delta.size(), 1, "one delta entry");
        ck_eq((int64_t) v3.entries()[0].kind, 0, "the snapshot's kind is 0");
        ck_eq((int64_t) delta.entries()[0].kind, 1, "the manifest's kind is 1");
        ck(delta.entries()[0].path.find("/delta/") != std::string::npos, "the delta entry lives under delta/");
    }
    {   // SUPERSEDE: a growing conversation stays ONE head - the old manifest unlinked, chunks kept
        const std::string dir = root + "/supersede/delta";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        strata::platform::KvDeltaStore delta;
        std::string err;
        strata::platform::KvNvmeStore nov3;
        ck(delta.open(root + "/supersede", S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
           "the store opens");
        const ConversationCheckpoint cp10 = boundary_checkpoint(S, 10, 8);
        strata::platform::TierActivity a10;
        ck(delta.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &cp10, err, &a10), "dump at 10");
        ck(!a10.skipped, "a first dump is not a skip");
        ck_eq(a10.dropped, 0, "with no previous head to supersede");
        ck_eq((int64_t) a10.written, (int64_t) delta.entries()[0].bytes,
              "written is the whole conversation this turn put on disk");
        const std::string first_head = delta.entries()[0].path;
        const uint64_t first_bytes = delta.entries()[0].bytes;
        const uint64_t reused_bytes = dir_bytes(dir + "/chunks");   // the chunks the grown dump must NOT re-count
        const ConversationCheckpoint cp18 = boundary_checkpoint(S, 18, 16);
        strata::platform::TierActivity a18;
        ck(delta.dump(S.ss, S.draft, S.g, ids_of(18), {}, true, &cp18, err, &a18), "dump at 18");
        ck_eq((int64_t) delta.size(), 1, "still one conversation");
        ck(!fs::exists(first_head), "the old head's manifest was unlinked");
        ck_eq(count_files(root + "/supersede/delta/chunks", ""), 4, "and the chunks accumulated (2 + 2 new)");
        // THE REPORT, against the books and the disk: `dropped` is the head the writer really superseded, and
        // `written` is THIS turn's write - the same number the "appended N chunks" line prints - not the size of
        // the conversation the turn produced.
        ck_eq(a18.dropped, 1, "the grown dump superseded this process's previous head");
        ck_eq((int64_t) a18.dropped_bytes, (int64_t) first_bytes, "and reports that entry's own bytes");
        ck_eq((int64_t) a18.written,
              (int64_t) delta.entries()[0].bytes - (int64_t) reused_bytes,
              "written is the new manifest + State + ONLY the chunks this turn sealed");
        ck((int64_t) a18.written < (int64_t) delta.entries()[0].bytes,
           "so one turn's write is smaller than the conversation it belongs to");
    }
    {   // THE P2-2 TEST: fork sharing survives eviction.  Two conversations share most of their chunks (a cross-
        // restart fork: the second store instance's head tracking is empty); evicting the first must NOT delete
        // the shared chunks - the naive "evict = delete the conversation's chunks" bug dies here.
        const std::string dir = root + "/forkshare";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        std::string err;
        strata::platform::KvNvmeStore nov3;
        {
            strata::platform::KvDeltaStore first;   // "process 1"
            ck(first.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "process 1 opens");
            const ConversationCheckpoint fa = boundary_checkpoint(S, 10, 8);
            ck(first.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &fa, err), "A dumps at 10");
        }
        std::vector<std::string> shared_names;
        for (const auto& de : fs::directory_iterator(dir + "/delta/chunks"))
            shared_names.push_back(de.path().filename().string());
        ck_eq((int64_t) shared_names.size(), 2, "A sealed two chunks");
        std::vector<int32_t> forked = ids_of(14);          // B: A's prefix, then a different tail
        for (int64_t i = 10; i < 14; ++i) forked[(size_t) i] = 700 + (int32_t) i;
        {
            strata::platform::KvDeltaStore second;   // "process 2": a restart - nothing superseded
            ck(second.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "process 2 opens");
            ck_eq((int64_t) second.size(), 1, "the scan sees A");
            const ConversationCheckpoint fb = boundary_checkpoint(S, 14, 12);
            strata::platform::TierActivity ab;
            ck(second.dump(S.ss, S.draft, S.g, forked, {}, true, &fb, err, &ab), "B dumps");
            ck_eq((int64_t) second.size(), 2, "A and B are BOTH live now");
            // THE P2-2 RULE ON THE REPORT, not just on the byte totals: a fork superseded nothing, so it must not
            // claim a drop it did not make (the same rule that keeps A's entry in the store's index).
            ck_eq(ab.dropped, 0, "a fork's dump reports NO supersede");
            ck_eq((int64_t) ab.dropped_bytes, 0, "and no bytes dropped with it");
            ck(!ab.skipped, "it is not a skip either: it wrote a new head");
            ck_eq((int64_t) ab.written, (int64_t) second.entries()[1].bytes,
                  "written is all of B's records: with no previous head there is nothing to reuse");
        }
        strata::platform::KvDeltaStore both;
        ck(both.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "both reopen");
        ck_eq((int64_t) both.size(), 2, "both manifests scanned");
        // evict A: cap = A's bytes exactly (A is the oldest mtime)
        const strata::platform::NvmeEntry* a = nullptr;
        for (const strata::platform::NvmeEntry& e : both.entries()) if (e.ids.size() == 10) a = &e;
        ck(a != nullptr, "A is in the store");
        const uint64_t cap = both.total_bytes() - a->bytes;
        const uint64_t a_bytes = a->bytes;   // the victim's bytes, read BEFORE the eviction erases the entry
        const auto rec_before = delta_records(dir + "/delta");
        const strata::platform::TierActivity ev = kv_delta_enforce_cap(nov3, both, (int64_t) cap);
        const auto rec_after = delta_records(dir + "/delta");
        ck_eq((int64_t) both.size(), 1, "A was evicted, B stands");
        ck_eq(ev.evicted, 1, "the cap reports exactly one eviction");
        ck_eq((int64_t) ev.evicted_bytes, (int64_t) a_bytes, "with the victim entry's own bytes");
        ck_eq(ev.swept, rec_before.first - rec_after.first,
              "and the sweep it ran reports the records that actually left the disk");
        ck_eq((int64_t) ev.swept_bytes, (int64_t) (rec_before.second - rec_after.second), "with their bytes");
        ck_eq(ev.swept, 1, "A's State record is the only orphan: both of its chunks are shared with B");
        ck_eq((int64_t) both.sweep().swept, 0, "and a second sweep reclaims nothing new");
        for (const std::string& name : shared_names)
            ck(fs::exists(dir + "/delta/chunks/" + name),
               ("the shared chunk " + name + " SURVIVED A's eviction (B references it)").c_str());
        {   // evict B too: a THIRD conversation gives the never-empty policy something to keep
            const ConversationCheckpoint fc = boundary_checkpoint(S, 20, 16);
            std::vector<int32_t> third = ids_of(20);
            for (int64_t i = 14; i < 20; ++i) third[(size_t) i] = 800 + (int32_t) i;   // extends nothing live
            strata::platform::TierActivity ac;
            ck(both.dump(S.ss, S.draft, S.g, third, {}, true, &fc, err, &ac), "C dumps");
            ck_eq(ac.dropped, 0, "C extends no live head, so it supersedes nothing");
            const strata::platform::NvmeEntry* b = nullptr;
            for (const strata::platform::NvmeEntry& e : both.entries()) if (e.ids.size() == 14) b = &e;
            ck(b != nullptr, "B is still in the store");
            const uint64_t b_bytes = b->bytes;
            const auto rec2_before = delta_records(dir + "/delta");
            const strata::platform::TierActivity ev2 = kv_delta_enforce_cap(nov3, both, 1);   // 1 byte: evict all
            const auto rec2_after = delta_records(dir + "/delta");
            ck_eq((int64_t) both.size(), 1, "B evicted (C stands: the policy keeps the last entry)");
            ck_eq(ev2.evicted, 1, "one eviction, reported");
            ck_eq((int64_t) ev2.evicted_bytes, (int64_t) b_bytes, "counted with B's bytes");
            ck_eq(ev2.swept, rec2_before.first - rec2_after.first,
                  "and the sweep's count is the records it reclaimed from the disk");
            ck_eq((int64_t) ev2.swept_bytes, (int64_t) (rec2_before.second - rec2_after.second), "with their bytes");
            ck_eq(ev2.swept, 1, "that record is B's State record: every chunk B sealed is below C's boundary too");
            ck_eq((int64_t) count_files(dir + "/delta/chunks", ""), 5,
                  "and with A and B gone, only C's own five chunks survive the sweep (sealed(20) = 20 = 5)");
        }
    }
    {   // TWO-TIER SINGLE CAP: eviction order is global oldest-mtime; the accounting counts both tiers
        const std::string dir = root + "/twotier";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        std::string err;
        strata::platform::KvNvmeStore nov3;
        strata::platform::KvNvmeStore v3;
        strata::platform::KvDeltaStore delta;
        ck(v3.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), err), "v3 opens");
        ck(delta.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "delta opens");
        const ConversationCheckpoint t10 = boundary_checkpoint(S, 10, 8);
        ck(v3.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &t10, err), "v3 dumps (oldest)");
        ::sleep(1);   // mtime resolution is seconds: make the age order real
        ck(delta.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &t10, err), "delta dumps");
        const uint64_t total = v3.total_bytes() + delta.total_bytes();
        const uint64_t v3_bytes = v3.entries()[0].bytes;   // the victim: the globally oldest entry
        const strata::platform::TierActivity ev = kv_delta_enforce_cap(v3, delta, (int64_t) total - 1);
        ck_eq((int64_t) (v3.size() + delta.size()), 1, "one conversation was evicted across the two tiers");
        ck_eq((int64_t) delta.size(), 1, "and it was the V3 one (the globally oldest mtime)");
        ck_eq((int64_t) v3.size(), 0, "(the delta entry stands)");
        ck_eq(ev.evicted, 1, "the combined cap reports one eviction");
        ck_eq((int64_t) ev.evicted_bytes, (int64_t) v3_bytes,
              "counted from the tier the victim came from (the v3 snapshot's bytes)");
        ck_eq(ev.swept, 0, "and the sweep reclaimed nothing: no delta manifest was unlinked");
    }
    {   // BOUNDARY: a cap smaller than one conversation empties down to the last entry, and the store still opens
        const std::string dir = root + "/boundary";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        std::string err;
        strata::platform::KvNvmeStore nov3;
        {
            strata::platform::KvDeltaStore d;
            ck(d.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "the store opens");
            const ConversationCheckpoint b10 = boundary_checkpoint(S, 10, 8);
            const ConversationCheckpoint b18 = boundary_checkpoint(S, 18, 16);
            ck(d.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &b10, err), "A");
            ck(d.dump(S.ss, S.draft, S.g, ids_of(18), {}, true, &b18, err), "B");
        }
        strata::platform::KvDeltaStore d;
        ck(d.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "the store reopens");
        const strata::platform::TierActivity ev = kv_delta_enforce_cap(nov3, d, 1);   // 1 byte: evict everything
        ck_eq((int64_t) d.size(), 1, "down to the last entry (never empty)");
        ck_eq(ev.evicted, 0, "reported as NO eviction: the never-empty policy removed nothing, it did not empty");
        ck_eq(ev.swept, 0, "and the sweep it ran had nothing left to reclaim (open already swept the supersede)");
        strata::platform::KvDeltaStore again;
        ck(again.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
           "and the store still opens and scans clean");
        ck_eq((int64_t) again.size(), 1, "with the survivor as an entry");
    }
    {   // RECENCY: an idempotent re-dump refreshes mtime, so the ACTIVE conversation is never the eviction victim
        const std::string dir = root + "/recency";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        std::string err;
        strata::platform::KvNvmeStore nov3;
        strata::platform::KvDeltaStore d;
        ck(d.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "the store opens");
        const ConversationCheckpoint r10 = boundary_checkpoint(S, 10, 8);
        const ConversationCheckpoint r18 = boundary_checkpoint(S, 18, 16);
        ck(d.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &r10, err), "A (will go stale)");
        ::sleep(1);
        ck(d.dump(S.ss, S.draft, S.g, ids_of(18), {}, true, &r18, err), "B (the active one)");
        strata::platform::TierActivity rid;
        ck(d.dump(S.ss, S.draft, S.g, ids_of(18), {}, true, &r18, err, &rid), "B re-dumped (idempotent)");
        ck(rid.skipped, "the re-dump reports the exact-match skip");
        ck_eq((int64_t) rid.written, 0, "and wrote ZERO bytes - it only refreshed recency");
        ck_eq(rid.dropped, 0, "and superseded nothing");
        const uint64_t total = d.total_bytes();
        const auto rec_before = delta_records(dir + "/delta");
        const strata::platform::TierActivity ev = kv_delta_enforce_cap(nov3, d, (int64_t) total - 1);
        const auto rec_after = delta_records(dir + "/delta");
        ck_eq((int64_t) d.size(), 1, "one was evicted");
        ck_eq((int64_t) d.entries()[0].L, 18, "and it was A: the re-dump kept B's mtime fresh");
        // What the cap was ALLOWED to do here: B's dump superseded A's manifest, so the store already held the
        // one entry it must keep - the report says zero evictions rather than pretending otherwise.  The sweep,
        // though, has work: A's State record is the orphan its supersede left behind.
        ck_eq(ev.evicted, 0, "no eviction: the last entry is kept even over the cap");
        ck_eq(ev.swept, rec_before.first - rec_after.first,
              "and the sweep's count is the record it actually reclaimed");
        ck_eq((int64_t) ev.swept_bytes, (int64_t) (rec_before.second - rec_after.second), "with its bytes");
        ck_eq(ev.swept, 1, "that record is A's orphan State record");
    }
    {   // P7 AT THE STORE LEVEL: an externally deleted chunk degrades to refuse-and-drop, and drop() works
        const std::string dir = root + "/p7";
        Session S;
        S.seed_indexer(1.0f, 2, 2.0f);
        S.tag_kv();
        std::string err;
        strata::platform::KvNvmeStore nov3;
        strata::platform::KvDeltaStore d;
        ck(d.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "the store opens");
        const ConversationCheckpoint p10 = boundary_checkpoint(S, 10, 8);
        ck(d.dump(S.ss, S.draft, S.g, ids_of(10), {}, true, &p10, err), "dumped");
        const strata::platform::NvmeEntry e = d.entries()[0];
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<strata::platform::DeltaChunkRef> refs;
        strata::platform::DeltaManifestHeader m;
        ck(strata::platform::delta_read_manifest(e.path, m, ids, imgs, refs, err), "the manifest reads");
        fs::remove(dir + "/delta/chunks/" + strata::platform::delta_key_name(refs[0].key) + ".bin");
        Session R;
        ck(d.restore(e, R.ss, R.draft, R.g, err) == strata::core::ConversationRestore::invalid,
           "the externally-deleted chunk refuses the restore");
        d.drop(e);
        ck_eq((int64_t) d.size(), 0, "and the store drops the dead entry");
    }
}

// ================================ the differential fuzz (the playback workhorse) ================================
//
// ops = {extend, fork at a random prefix, new conversation, re-dump same T, crash@Ck (random), evict to a random
// fraction, external-delete a random file, restart}; a reference model tracks the live manifests and each one's
// EXPECTED reassembly bytes (the v3 dump of the same boundary, taken when the model last saw it).  After EVERY
// op: (a) the on-disk conversation set is the model's live set; (b) every manifest whose referenced records all
// exist reassembles byte-identically, and one whose records were externally deleted refuses (P7); (c) after a
// sweep, the files are EXACTLY the referenced set.
//
// BUDGET: the ctest default is 25 seeds x 40 ops (~20 s); the handoff's full sweep - 1000 seeds x 200 ops,
// 37,899,123 checks - was run green with KV_DELTA_FUZZ_SEEDS=1000 KV_DELTA_FUZZ_OPS=200 and is the form to run
// before touching the writer or the sweep again.
//
// WHAT IT CANNOT PROVE: the SSD actually persisted on fsync (a power-cut property - the posture degrades, never
// corrupts: a torn chunk fails its digest to invalid -> refuse -> re-prefill); and the wrapped-CUDA limits the
// kv_nvme_host_test header states.

struct FuzzConv {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> expected;   // the v3 image bytes the reassembly must produce
    bool defective = false;          // an external delete broke a record under it: restore must refuse (P7)
};

void fixture_fuzz(const std::string& root) {
    const long seeds = std::getenv("KV_DELTA_FUZZ_SEEDS") ? (long) std::atoll(std::getenv("KV_DELTA_FUZZ_SEEDS")) : 25;
    const long ops_n = std::getenv("KV_DELTA_FUZZ_OPS") ? (long) std::atoll(std::getenv("KV_DELTA_FUZZ_OPS")) : 40;
    Session S;   // ONE static session: the arrays never change, so a boundary's checkpoint (and the expected v3
                 // image for its ids) is a deterministic function of the boundary length
    S.seed_indexer(987654.0f, 6, 555.0f);
    S.tag_kv();
    strata::platform::KvNvmeStore nov3;   // the enforce helper needs an lvalue; the fuzz's cap is delta-only
    const uint64_t sfp = strata::platform::kv_delta_weights_fp({"model.gguf"});   // the STORE's fingerprint (the
    // model file does not exist here, so the fp is the path+size hash - the same value the manifests carry)

    std::vector<FuzzConv> live;
    strata::platform::KvDeltaStore store;
    std::string dir, err;
    auto checkpoint_for = [&](int64_t T) {
        return boundary_checkpoint(S, T, (int32_t) ((T / SHP.idx_block) * SHP.idx_block));
    };
    // the reference v3 image for a conversation state (deterministic, computed independently by the tier)
    auto reference_image = [&](const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs) {
        const std::string v3 = dir + "/.ref-v3";
        ConversationCheckpoint cp = checkpoint_for((int64_t) ids.size());
        cp.ids = ids;   // THE ENGINE'S CONTRACT: the dump keys on the CHECKPOINT's ids - they are one thing
        ck(strata::platform::nvme_dump_at(v3.c_str(), S.ss, S.draft, S.g, ids, imgs, true, &cp, err),
           "the reference v3 dump succeeds");
        return slurp(v3);
    };
    // sync the model to a store's scanned entries: survivors keep their expected bytes; new (post-crash) ones
    // get them computed; the crash's lost in-memory state is exactly what a relaunch re-derives from disk
    auto sync_model = [&](const strata::platform::KvDeltaStore& st) {
        std::vector<FuzzConv> next;
        for (const strata::platform::NvmeEntry& e2 : st.entries()) {
            bool found = false;
            for (const FuzzConv& f : live)
                if ((size_t) e2.L == f.ids.size() && std::equal(f.ids.begin(), f.ids.end(), e2.ids.begin())) {
                    next.push_back(f);
                    found = true;
                    break;
                }
            if (!found) {
                FuzzConv f;
                f.ids = e2.ids;
                f.imgs = e2.imgs;
                f.expected = reference_image(e2.ids, e2.imgs);
                next.push_back(std::move(f));
            }
        }
        live = std::move(next);
    };

    // THE PROCESS MODEL of the store's head: the previous dump of THIS store instance, and the supersede rule
    // it applies (the previous head whose ids the new key extends is unlinked; a fork from a non-head survives).
    std::vector<int32_t> last_dump_ids;
    auto do_dump = [&](FuzzConv f) {
        ConversationCheckpoint cp = checkpoint_for((int64_t) f.ids.size());
        cp.ids = f.ids;   // the checkpoint IS the boundary: its ids are the conversation's ids (a fork's tail
        f.imgs = cp.imgs; // lives in the ids; a checkpoint disagreeing with its key is a caller bug)
        // mirror the store's IDEMPOTENT SKIP: an exact match refreshes recency and returns BEFORE the supersede
        // bookkeeping - the process's previous dump (the supersede hint) is untouched, because the skip never
        // became a dump.  The model's hint must follow, or the model supersedes a manifest the store kept.
        bool idempotent = false;
        for (const strata::platform::NvmeEntry& e2 : store.entries())
            if (e2.L == (int64_t) f.ids.size() && e2.cvec && e2.imgs == f.imgs &&
                std::equal(f.ids.begin(), f.ids.end(), e2.ids.begin()))
                idempotent = true;
        ck(store.dump(S.ss, S.draft, S.g, f.ids, f.imgs, true, &cp, err), "the fuzz dump succeeds");
        f.expected = reference_image(f.ids, f.imgs);
        // defective STAYS: a re-dump REUSES the shared chunks - a record an external delete removed is still
        // gone, the new manifest references it just the same, and the tier's refuse is still the right answer
        if (!idempotent) {
            // the supersede: the previous head whose ids the new key extends loses its manifest
            if (!last_dump_ids.empty() && last_dump_ids.size() <= f.ids.size() &&
                std::equal(last_dump_ids.begin(), last_dump_ids.end(), f.ids.begin())) {
                std::vector<FuzzConv> kept;
                for (const FuzzConv& g2 : live)
                    if (!(g2.ids.size() == last_dump_ids.size() &&
                          std::equal(last_dump_ids.begin(), last_dump_ids.end(), g2.ids.begin())))
                        kept.push_back(g2);
                live = std::move(kept);
            }
            last_dump_ids = f.ids;
        }
        // upsert by ids
        bool replaced = false;
        for (FuzzConv& g2 : live)
            if (g2.ids.size() == f.ids.size() && std::equal(f.ids.begin(), f.ids.end(), g2.ids.begin())) {
                g2 = f;
                replaced = true;
            }
        if (!replaced) live.push_back(f);
    };

    const char* crash_opts[] = {"C1", "C2", "C3", "C4", "C5"};
    for (long seed = 0; seed < seeds; ++seed) {
        std::mt19937 rng((unsigned) (seed + 1));
        auto rnd = [&](int64_t lo, int64_t hi) { return (int64_t) (rng() % (uint64_t) (hi - lo + 1)) + lo; };
        dir = root + "/fuzz/" + std::to_string(seed);
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
        store = strata::platform::KvDeltaStore();
        ck(store.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err), "the fuzz store opens");
        live.clear();

        for (long op = 0; op < ops_n; ++op) {
            const int kind = (int) (rng() % 8);
            if (kind == 0 && !live.empty()) {   // EXTEND a random live conversation
                const size_t c = (size_t) rnd(0, (int64_t) live.size() - 1);
                FuzzConv f = live[c];
                if ((int64_t) f.ids.size() + 8 > 28) continue;   // the delta path's gate: T <= the drafter ring
                const int64_t grow = rnd(1, 8);
                for (int64_t i = 0; i < grow; ++i) f.ids.push_back(700 + (int32_t) ((f.ids.size() * 13 + i) % 500));
                do_dump(f);
            } else if (kind == 1 && !live.empty()) {   // FORK at a random prefix of a random conversation
                const size_t c = (size_t) rnd(0, (int64_t) live.size() - 1);
                const int64_t p = rnd(1, (int64_t) live[c].ids.size());
                FuzzConv f;
                f.ids.assign(live[c].ids.begin(), live[c].ids.begin() + (int) p);
                const int64_t tail = rnd(0, 28 - p > 8 ? 8 : 28 - p);   // keep the fork inside the ring too
                for (int64_t i = 0; i < tail; ++i) f.ids.push_back(900 + (int32_t) ((p * 7 + i) % 400));
                if (f.ids.empty()) continue;
                do_dump(f);   // an in-epoch fork that extended the process's head REPLACED it (§5.14)
            } else if (kind == 2) {   // a NEW conversation: ids that extend nothing
                FuzzConv f;
                const int64_t n = rnd(5, 28);
                for (int64_t i = 0; i < n; ++i) f.ids.push_back(100 + (int32_t) ((seed * 31 + op * 7 + i) % 900));
                do_dump(std::move(f));
            } else if (kind == 3 && !live.empty()) {   // RE-DUMP the same T (idempotent)
                const size_t c = (size_t) rnd(0, (int64_t) live.size() - 1);
                do_dump(live[c]);
            } else if (kind == 4) {   // CRASH at a random point, then relaunch
                setenv("STRATA_DELTA_FAIL_AT", crash_opts[rng() % 5], 1);
                {
                    FuzzConv f;
                    f.ids = ids_of(rnd(5, 28));
                    const ConversationCheckpoint cp = checkpoint_for((int64_t) f.ids.size());
                    f.imgs = cp.imgs;
                    std::string derr;
                    const bool ok = store.dump(S.ss, S.draft, S.g, f.ids, f.imgs, true, &cp, derr);
                    if (ok) {
                        // nothing new to write (all chunks reused) AND the hook fired late: the dump committed.
                        // Model it exactly like a successful dump (C1/C2 only fire on a NEW chunk).
                        do_dump(f);
                    } else {
                        ck(derr.find("STRATA_DELTA_FAIL_AT") != std::string::npos, "the crash hook reported itself");
                    }
                }
                unsetenv("STRATA_DELTA_FAIL_AT");
                store = strata::platform::KvDeltaStore();   // the relaunch
                last_dump_ids.clear();                      // a new process has no previous dump of its own
                ck(store.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
                   "the relaunched store opens clean");
                sync_model(store);   // C4's both-heads row included: the scan sees whatever the disk holds
            } else if (kind == 5 && !live.empty()) {   // EVICT to a random fraction of the current bytes
                const uint64_t total = store.total_bytes();
                kv_delta_enforce_cap(nov3, store, (int64_t) (total * (uint64_t) rnd(1, 90) / 100));
                // if the eviction dropped the process's HEAD entry, the store's supersede hint went with it
                // (drop() clears last_ids_/last_path_) - the model's hint must follow
                if (!last_dump_ids.empty()) {
                    bool head_there = false;
                    for (const strata::platform::NvmeEntry& e2 : store.entries())
                        if ((size_t) e2.L == last_dump_ids.size() &&
                            std::equal(last_dump_ids.begin(), last_dump_ids.end(), e2.ids.begin()))
                            head_there = true;
                    if (!head_there) last_dump_ids.clear();
                }
                sync_model(store);
            } else if (kind == 6) {   // EXTERNAL DELETE of a random record file (P7's disk-level form)
                std::vector<fs::path> files;
                for (const char* sub : {"/delta/chunks", "/delta/states"})
                    for (const fs::directory_entry& de : fs::directory_iterator(dir + sub, ec))
                        if (de.is_regular_file() && de.path().filename().string().rfind(".tmp-", 0) != 0)
                            files.push_back(de.path());
                for (const fs::directory_entry& de : fs::directory_iterator(dir + "/delta", ec))
                    if (de.is_regular_file() && de.path().filename().string().rfind("log-", 0) == 0)
                        files.push_back(de.path());
                if (files.empty()) continue;
                const fs::path victim = files[rng() % files.size()];
                const std::string key = victim.filename().string().rfind("log-", 0) == 0
                                            ? std::string()
                                            : victim.filename().string().substr(0, victim.filename().string().size() - 4);
                fs::remove(victim, ec);
                if (key.empty()) {   // a manifest: its conversation is gone from the disk
                    store = strata::platform::KvDeltaStore();
                    last_dump_ids.clear();
                    ck(store.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
                       "the store reopens after a manifest delete");
                    sync_model(store);
                } else {   // a chunk or state: every manifest referencing it becomes defective (restore refuses, P7)
                    for (FuzzConv& f : live) {
                        bool touches = false;
                        for (const strata::platform::NvmeEntry& e2 : store.entries())
                            if ((size_t) e2.L == f.ids.size() && std::equal(f.ids.begin(), f.ids.end(), e2.ids.begin())) {
                                std::vector<int32_t> ids2;
                                std::vector<ConversationImageKey> imgs2;
                                std::vector<strata::platform::DeltaChunkRef> refs2;
                                strata::platform::DeltaManifestHeader m2;
                                if (strata::platform::delta_read_manifest(e2.path, m2, ids2, imgs2, refs2, err) &&
                                    (strata::platform::delta_key_name(m2.state_key) == key ||
                                     std::any_of(refs2.begin(), refs2.end(), [&](const strata::platform::DeltaChunkRef& r) {
                                         return strata::platform::delta_key_name(r.key) == key;
                                     })))
                                    touches = true;
                            }
                        if (touches) f.defective = true;
                    }
                }
            } else {   // RESTART: a fresh store instance on the same dir (the head tracking resets)
                store = strata::platform::KvDeltaStore();
                last_dump_ids.clear();
                ck(store.open(dir, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
                   "the restarted store opens clean");
            }

            // ---- the per-op assertions ----
            std::vector<std::string> on_disk;
            for (const fs::directory_entry& de : fs::directory_iterator(dir + "/delta", ec))
                if (de.path().filename().string().rfind("log-", 0) == 0) on_disk.push_back(de.path().string());

            // two manifests may hold the SAME ids (two processes each dumped this boundary - the store admits
            // both; they are interchangeable); the conversation SET is what must agree
            {
                std::map<std::string, int> disk_set, live_set;
                for (const std::string& mp2 : on_disk) {
                    strata::platform::DeltaManifestHeader m2;
                    std::vector<int32_t> ids2;
                    std::vector<ConversationImageKey> imgs2;
                    std::vector<strata::platform::DeltaChunkRef> refs2;
                    if (!strata::platform::delta_read_manifest(mp2, m2, ids2, imgs2, refs2, err)) continue;
                    disk_set[std::string((const char*) ids2.data(), ids2.size() * 4)]++;
                }
                for (const FuzzConv& f2 : live)
                    live_set[std::string((const char*) f2.ids.data(), f2.ids.size() * 4)]++;
                ck_eq((int64_t) disk_set.size(), (int64_t) live_set.size(),
                      "the fuzz: on-disk conversations == live conversations");
                for (const auto& kv2 : live_set)
                    ck(disk_set.count(kv2.first) != 0, "the fuzz: every live conversation has a head on disk");
            }
            for (const std::string& mp : on_disk) {
                strata::platform::DeltaManifestHeader m;
                std::vector<int32_t> ids2;
                std::vector<ConversationImageKey> imgs2;
                std::vector<strata::platform::DeltaChunkRef> refs2;
                ck(strata::platform::delta_read_manifest(mp, m, ids2, imgs2, refs2, err), "the fuzz manifest reads");
                bool matched = false;
                for (FuzzConv& f : live)
                    if ((size_t) m.L == f.ids.size() && std::equal(f.ids.begin(), f.ids.end(), ids2.begin())) {
                        matched = true;
                        // BROKEN is a property of the DISK, computed fresh each op: an external delete removed a
                        // record this manifest references.  (A flag carried on the model would go wrong in both
                        // directions: a superseded manifest stops referencing the deleted record, while a re-dump
                        // REUSES the hole - the referenced-but-missing set is the only truth.)
                        bool broken = !fs::exists(dir + "/delta/states/" +
                                                  strata::platform::delta_key_name(m.state_key) + ".bin");
                        for (const strata::platform::DeltaChunkRef& r : refs2)
                            broken = broken ||
                                     !fs::exists(dir + "/delta/chunks/" +
                                                 strata::platform::delta_key_name(r.key) + ".bin");
                        if (!broken) {   // (b) the reassembly is byte-identical to the model's expected image
                            const std::vector<uint8_t> got = reassemble(S, dir + "/delta", mp, sfp);
                            ck(got == f.expected, "the fuzz: a live manifest reassembles byte-identically");
                        } else {   // P7: the broken record degrades to refuse-and-drop
                            strata::platform::NvmeEntry e2;
                            e2.path = mp;
                            e2.kind = 1;
                            Session R;
                            std::vector<int32_t> rids;
                            std::vector<ConversationImageKey> rimgs;
                            bool rc = false;
                            int64_t rl = 0;
                            std::string rerr;
                            const strata::core::ConversationRestore got2 =
                                strata::platform::delta_restore(e2, R.ss, R.draft, R.g, sfp, rids, rimgs, rc, rl, rerr);
                            ck(got2 == strata::core::ConversationRestore::invalid,
                               "the fuzz: a broken conversation refuses (never converts, never corrupts)");
                        }
                    }
                ck(matched, "the fuzz: every on-disk manifest is a live conversation");
            }
            // (c) every chunk/state file is either referenced by a live manifest or REMOVABLE BY SWEEP - the
            // disjunction the doc states.  Supersede garbage between sweeps is the second disjunct, so the
            // property with content is (d): after a sweep, the files are EXACTLY the referenced set.
            // (d) a sweep removes EXACTLY the unreferenced set - checked on a THROWAWAY COPY of the directory,
            // so the check itself never disturbs the process's own head tracking (the sweep at open is the real
            // store's behaviour; here the copy's sweep is the assertion)
            {
                const std::string copy = dir + ".sweepcheck";
                fs::remove_all(copy, ec);
                fs::copy(dir, copy, fs::copy_options::recursive, ec);
                strata::platform::KvDeltaStore sw;
                ck(sw.open(copy, S.g, strata::core::qsa_kv_format(S.layers[0]), {"model.gguf"}, err),
                   "the fuzz: the sweep-check copy opens");
                std::map<std::string, bool> referenced;
                for (const strata::platform::NvmeEntry& e2 : sw.entries()) {
                    strata::platform::DeltaManifestHeader m;
                    std::vector<int32_t> ids2;
                    std::vector<ConversationImageKey> imgs2;
                    std::vector<strata::platform::DeltaChunkRef> refs2;
                    ck(strata::platform::delta_read_manifest(e2.path, m, ids2, imgs2, refs2, err),
                       "the fuzz: the sweep's manifests read");
                    referenced[strata::platform::delta_key_name(m.state_key)] = true;
                    for (const strata::platform::DeltaChunkRef& r : refs2)
                        referenced[strata::platform::delta_key_name(r.key)] = true;
                }
                for (const char* sub : {"/chunks", "/states"})
                    for (const fs::directory_entry& de : fs::directory_iterator(copy + "/delta" + sub, ec)) {
                        const std::string name = de.path().filename().string();
                        ck(referenced.count(name.substr(0, name.size() - 4)) != 0,
                           "the fuzz: after a sweep, every file is referenced - exactly the orphans were removed");
                    }
                fs::remove_all(copy, ec);
            }
            if (live.size() > 6) {   // keep the model small: drop the oldest when the store grows past 6
                const strata::platform::NvmeEntry* oldest = &store.entries()[0];
                for (const strata::platform::NvmeEntry& e2 : store.entries())
                    if (e2.mtime < oldest->mtime) oldest = &e2;
                const bool was_head = !last_dump_ids.empty() &&
                                      (size_t) oldest->L == last_dump_ids.size() &&
                                      std::equal(last_dump_ids.begin(), last_dump_ids.end(), oldest->ids.begin());
                store.drop(*oldest);
                if (was_head) last_dump_ids.clear();   // the store's hint went with the dropped head
                sync_model(store);
            }
        }
        fs::remove_all(dir, ec);
    }
    std::fprintf(stderr, "fuzz: %ld seeds x %ld ops clean\n", seeds, ops_n);
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
    fixture_store(root);
    fixture_fuzz(root);

    fs::remove_all(root, ec);
    std::printf("kv_delta_host_test: %d checks passed; no CUDA context, no model\n", checks);
    return 0;
}
