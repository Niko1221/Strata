// src/platform/kv_delta.cpp - see include/strata/platform/kv_delta.hpp.
#include "strata/platform/kv_delta.hpp"

#include "strata/kernels/qsa.hpp"  // qsa_real_shapes, qsa_pooled_rows

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <numeric>

#include <filesystem>

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

namespace fs = std::filesystem;

namespace strata::platform {

DeltaShapes delta_shapes() {
    DeltaShapes sh;
    sh.shapes = strata::kernels::qsa_real_shapes();   // the same source the v3 walk reads - never a second copy
    // lcm, not max and not a tuning constant: a sealed chunk must not straddle a KV page (the slice is then a
    // plain contiguous range of the pinned host array) nor an indexer block (the pooled rows stay whole rows).
    sh.block = std::lcm(sh.shapes.page_size, sh.shapes.idx_block);
    sh.rows_per_chunk = sh.block / sh.shapes.idx_block;
    return sh;
}

int64_t delta_chunk_payload_bytes(const SessionState& ss, const QsaState& mtp, const ModelGeometry& g,
                                  const DeltaShapes& sh, int64_t a) {
    const int64_t pages = sh.block / sh.shapes.page_size;   // sealed chunks are page-aligned by construction
    const int64_t page_kv_bytes = g.n_head_kv * sh.shapes.page_size;   // x the array's row width w
    int64_t bytes = 0;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        const QsaState& st = ss.qsa_states[i];
        for (int k = 0; k < nvme_kv_array_count(st); ++k)
            bytes += pages * page_kv_bytes * nvme_kv_host_array(st, g.head_dim, k).w;
        bytes += sh.rows_per_chunk * g.idx_key_dim * 4;   // pooled rows are fp32 (idx_key_dim, 4 bytes)
    }
    if (a < mtp.max_cells) {   // §5.4: the drafter's cells are covered only while the chunk starts inside the ring
        for (int k = 0; k < nvme_kv_array_count(mtp); ++k) {
            const NvmeKvArr ka = nvme_kv_host_array(mtp, g.head_dim, k);
            if (!ka.p) continue;   // the v3 dump skips a null drafter array and so does the count it records
            bytes += pages * page_kv_bytes * ka.w;
        }
    }
    return bytes;
}

int64_t delta_state_payload_bytes(const SessionState& ss, const QsaState& mtp, const ModelGeometry& g,
                                  const ConversationStateSizes& z, const DeltaShapes& sh, int64_t t) {
    const int64_t s = delta_sealed(t, sh);
    const int64_t tail_pages = (t + sh.shapes.page_size - 1) / sh.shapes.page_size - s / sh.shapes.page_size;
    const int64_t tail_rows = strata::kernels::qsa_pooled_rows(t, sh.shapes) - s / sh.shapes.idx_block;
    const int64_t page_kv_bytes = g.n_head_kv * sh.shapes.page_size;
    int64_t bytes = (int64_t) z.gdn + (ss.ple_hist ? (int64_t) z.ple : 0);
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        const QsaState& st = ss.qsa_states[i];
        for (int k = 0; k < nvme_kv_array_count(st); ++k)
            bytes += tail_pages * page_kv_bytes * nvme_kv_host_array(st, g.head_dim, k).w;
        bytes += tail_rows * g.idx_key_dim * 4;   // the in-progress block and the spare row are IN the state
        bytes += (int64_t) (z.tail + z.dead + z.block_pos);   // the checkpoint's running state, one slice per layer
    }
    for (int k = 0; k < nvme_kv_array_count(mtp); ++k) {
        const NvmeKvArr ka = nvme_kv_host_array(mtp, g.head_dim, k);
        if (!ka.p) continue;
        bytes += tail_pages * page_kv_bytes * ka.w;
    }
    return bytes;
}

namespace {

long pid() {
#ifdef _WIN32
    return (long) _getpid();
#else
    return (long) ::getpid();
#endif
}

/// One temp-file counter per process: two records written in the same process never share a temp name, and
/// across processes the pid does it.  The temp name is DERIVED from the target (`<target>.tmp`), so two
/// processes writing DIFFERENT chunks never collide at all - the §5.15 single-writer rule stays honest without
/// a shared counter.
bool write_record(const std::string& path, const void* header, size_t header_bytes, const void* payload,
                  size_t payload_bytes, uint64_t footer, std::string& err) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "kv-delta: open " + tmp + ": " + std::strerror(errno); return false; }
    bool ok = std::fwrite(header, 1, header_bytes, f) == header_bytes &&
              (payload_bytes == 0 || std::fwrite(payload, 1, payload_bytes, f) == payload_bytes) &&
              std::fwrite(&footer, 1, sizeof footer, f) == sizeof footer;
    if (ok) ok = std::fflush(f) == 0;
#ifndef _WIN32
    if (ok) ok = ::fsync(::fileno(f)) == 0;   // crash consistency: a record under its real name is durable
#else
    if (ok) ok = ::_commit(::fileno(f)) == 0;
#endif
    std::fclose(f);
    if (!ok) { err = "kv-delta: write " + tmp; std::remove(tmp.c_str()); return false; }
    std::error_code ec;
    fs::rename(tmp, path, ec);   // atomic: the content-addressed name never holds a partial record
    if (ec) { err = "kv-delta: rename " + tmp + " -> " + path + ": " + ec.message(); return false; }
    return true;
}

/// The footer is the plain FNV-1a over the payload - the same hash the v3 payload footer uses, seeded the same.
inline uint64_t payload_footer(const void* payload, size_t n) {
    return nvme_fnv1a(kNvmeFnvBasis, payload, n);
}

}  // namespace

std::string delta_key_name(uint64_t key) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long) key);
    return buf;
}

bool delta_write_chunk(const std::string& path, const DeltaChunkHeader& h, const void* payload,
                       size_t payload_bytes, std::string& err) {
    if (h.magic != kDeltaChunkMagic || h.version != kDeltaFormatVersion) {
        err = "kv-delta: chunk " + path + ": refusing to write a header with a foreign magic/version";
        return false;
    }
    if (h.payload_bytes != (int64_t) payload_bytes) {
        // a header that disagrees with its own payload is a caller bug: the digest would happily cover either
        err = "kv-delta: chunk " + path + ": header says " + std::to_string(h.payload_bytes) + " payload bytes, the caller handed " +
              std::to_string(payload_bytes);
        return false;
    }
    const uint64_t footer = payload_footer(payload, payload_bytes);
    return write_record(path, &h, sizeof h, payload, payload_bytes, footer, err);
}

bool delta_read_chunk(const std::string& path, uint64_t expected_key, int64_t expected_a, int64_t expected_b,
                      std::vector<uint8_t>& payload, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "kv-delta: chunk " + path + ": open: " + std::strerror(errno); return false; }
    if (std::fseek(f, 0, SEEK_END) != 0) { err = "kv-delta: chunk " + path + ": seek"; std::fclose(f); return false; }
    const long long fsize = ftello(f);
    std::rewind(f);
    DeltaChunkHeader h;
    payload.clear();
    if (fsize < (long long) (sizeof h + sizeof(uint64_t)) || fsize > (long long) 4 << 30) {
        // a stray huge file must not become an allocation; a short one is not a chunk
        err = "kv-delta: chunk " + path + ": not a chunk (size)";
        std::fclose(f);
        return false;
    }
    if (std::fread(&h, 1, sizeof h, f) != sizeof h) {
        err = "kv-delta: chunk " + path + ": truncated header";
        std::fclose(f);
        return false;
    }
    // LAYOUT AND IDENTITY FACTS FIRST (the diagnostic-ordering rule): magic, version, which chunk this is
    // supposed to be, and whether the file's own size agrees with its header - only then is the digest verdict
    // meaningful, because a layout-drifted read would otherwise report as "corrupt".
    if (h.magic != kDeltaChunkMagic) {
        err = "kv-delta: chunk " + path + ": not a delta chunk (bad magic)";
        std::fclose(f);
        return false;
    }
    if (h.version != kDeltaFormatVersion) {
        err = "kv-delta: chunk " + path + ": format version " + std::to_string(h.version) + ", this build writes " +
              std::to_string(kDeltaFormatVersion) + " - refusing (nothing converts it)";
        std::fclose(f);
        return false;
    }
    if (h.key != expected_key) {
        err = "kv-delta: chunk " + path + ": key mismatch (file " + delta_key_name(h.key) + ", manifest " +
              delta_key_name(expected_key) + ")";
        std::fclose(f);
        return false;
    }
    if (h.a != expected_a || h.b != expected_b || h.b <= h.a) {
        err = "kv-delta: chunk " + path + ": range mismatch (file [" + std::to_string(h.a) + ", " +
              std::to_string(h.b) + "), manifest [" + std::to_string(expected_a) + ", " +
              std::to_string(expected_b) + "))";
        std::fclose(f);
        return false;
    }
    if (h.payload_bytes < 0 || fsize != (long long) (sizeof h + (uint64_t) h.payload_bytes + sizeof(uint64_t))) {
        err = "kv-delta: chunk " + path + ": truncated or oversized (" + std::to_string(fsize) + " bytes, header says " +
              std::to_string(sizeof h + (uint64_t) h.payload_bytes + sizeof(uint64_t)) + ")";
        std::fclose(f);
        return false;
    }
    payload.resize((size_t) h.payload_bytes);
    if (h.payload_bytes && std::fread(payload.data(), 1, payload.size(), f) != payload.size()) {
        err = "kv-delta: chunk " + path + ": truncated payload";
        std::fclose(f);
        return false;
    }
    uint64_t digest = 0;
    if (std::fread(&digest, 1, sizeof digest, f) != sizeof digest) {
        err = "kv-delta: chunk " + path + ": truncated footer";
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    if (digest != payload_footer(payload.data(), payload.size())) {
        err = "kv-delta: chunk " + path + ": integrity check failed (corrupt chunk)";
        payload.clear();
        return false;
    }
    return true;
}

bool delta_write_state(const std::string& dir, uint64_t tag, const void* payload, size_t payload_bytes,
                       uint64_t& key_out, std::string& err) {
    DeltaStateHeader h;
    h.payload_bytes = (int64_t) payload_bytes;
    key_out = delta_state_key(tag, payload, payload_bytes);
    const std::string path = dir + "/" + delta_key_name(key_out) + ".bin";
    return write_record(path, &h, sizeof h, payload, payload_bytes, payload_footer(payload, payload_bytes), err);
}

bool delta_read_state(const std::string& path, uint64_t tag, uint64_t expected_key, std::vector<uint8_t>& payload,
                      std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "kv-delta: state " + path + ": open: " + std::strerror(errno); return false; }
    if (std::fseek(f, 0, SEEK_END) != 0) { err = "kv-delta: state " + path + ": seek"; std::fclose(f); return false; }
    const long long fsize = ftello(f);
    std::rewind(f);
    DeltaStateHeader h;
    payload.clear();
    if (fsize < (long long) (sizeof h + sizeof(uint64_t)) || fsize > (long long) 4 << 30) {
        err = "kv-delta: state " + path + ": not a state record (size)";
        std::fclose(f);
        return false;
    }
    if (std::fread(&h, 1, sizeof h, f) != sizeof h) {
        err = "kv-delta: state " + path + ": truncated header";
        std::fclose(f);
        return false;
    }
    if (h.magic != kDeltaStateMagic) {
        err = "kv-delta: state " + path + ": not a delta state record (bad magic)";
        std::fclose(f);
        return false;
    }
    if (h.version != kDeltaFormatVersion) {
        err = "kv-delta: state " + path + ": format version " + std::to_string(h.version) + ", this build writes " +
              std::to_string(kDeltaFormatVersion) + " - refusing (nothing converts it)";
        std::fclose(f);
        return false;
    }
    if (h.payload_bytes < 0 || fsize != (long long) (sizeof h + (uint64_t) h.payload_bytes + sizeof(uint64_t))) {
        err = "kv-delta: state " + path + ": truncated or oversized (" + std::to_string(fsize) + " bytes, header says " +
              std::to_string(sizeof h + (uint64_t) h.payload_bytes + sizeof(uint64_t)) + ")";
        std::fclose(f);
        return false;
    }
    payload.resize((size_t) h.payload_bytes);
    if (h.payload_bytes && std::fread(payload.data(), 1, payload.size(), f) != payload.size()) {
        err = "kv-delta: state " + path + ": truncated payload";
        std::fclose(f);
        return false;
    }
    uint64_t digest = 0;
    if (std::fread(&digest, 1, sizeof digest, f) != sizeof digest) {
        err = "kv-delta: state " + path + ": truncated footer";
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    // the footer is the bare payload hash; the IDENTITY check (does this payload hash, seeded with the
    // conversation's tag, to the key the manifest named?) rides on top of it
    if (digest != payload_footer(payload.data(), payload.size())) {
        err = "kv-delta: state " + path + ": integrity check failed (corrupt state record)";
        payload.clear();
        return false;
    }
    if (tag != 0 && expected_key != 0 && delta_state_key(tag, payload.data(), payload.size()) != expected_key) {
        err = "kv-delta: state " + path + ": key mismatch (this state record is not the one the manifest names)";
        payload.clear();
        return false;
    }
    return true;
}

}  // namespace strata::platform
