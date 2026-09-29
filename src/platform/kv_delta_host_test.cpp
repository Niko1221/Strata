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

int checks = 0;

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
    reset_faults();

    fs::remove_all(root, ec);
    std::printf("kv_delta_host_test: %d checks passed; no CUDA context, no model\n", checks);
    return 0;
}
