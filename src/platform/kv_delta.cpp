// src/platform/kv_delta.cpp - see include/strata/platform/kv_delta.hpp.
#include "strata/platform/kv_delta.hpp"

#include "strata/kernels/qsa.hpp"  // qsa_real_shapes, qsa_pooled_rows

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
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

/// **CONSUME THE CUDA ERROR THIS FUNCTION JUST HANDLED** (kv_nvme.cpp's consume_cuda_error discipline): an
/// error left sticky makes the NEXT `cudaGetLastError()` in the engine - `kv_stream_reset`'s `check()`, which
/// exits - report this failure as if it had happened there.
inline void consume_cuda_error() { (void) cudaGetLastError(); }

/// The deterministic crash hook (the crash matrix's C1..C5; C6 is the store's in-memory step).  Unset - the
/// default, and the only shipped configuration - the hook does nothing at all.
enum class FailAt { kNone, kC1, kC2, kC3, kC4, kC5 };
FailAt fail_at_env() {
    // read LIVE, not cached: a test process flips the variable between dumps to walk the crash points one by one
    const char* e = std::getenv("STRATA_DELTA_FAIL_AT");
    if (!e) return FailAt::kNone;
    if (!std::strcmp(e, "C1")) return FailAt::kC1;
    if (!std::strcmp(e, "C2")) return FailAt::kC2;
    if (!std::strcmp(e, "C3")) return FailAt::kC3;
    if (!std::strcmp(e, "C4")) return FailAt::kC4;
    if (!std::strcmp(e, "C5")) return FailAt::kC5;
    return FailAt::kNone;
}
std::string fail_at_err(const char* what) {
    return std::string("kv-delta: ") + what + " aborted by STRATA_DELTA_FAIL_AT (injected crash point)";
}

long pid_of() {
#ifdef _WIN32
    return (long) _getpid();
#else
    return (long) ::getpid();
#endif
}

/// The temp name follows the §5.15 convention - `<dir>/.tmp-<pid>-<seq>` - so a scan can ignore the WHOLE class
/// by its prefix, and two processes never share a temp even when writing the same content-addressed chunk.
/// `leave_temp_only` is the C1 crash point: the bytes are written and the file CLOSED but never fsynced and
/// never renamed, so the residue is a `.tmp-*` the scan ignores and the sweep reclaims.
bool write_record(const std::string& path, const void* header, size_t header_bytes, const void* payload,
                  size_t payload_bytes, uint64_t footer, bool leave_temp_only, std::string& err) {
    static long counter = 0;
    const std::string tmp =
        fs::path(path).parent_path().string() + "/.tmp-" + std::to_string(pid_of()) + "-" + std::to_string(counter++);
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) { err = "kv-delta: open " + tmp + ": " + std::strerror(errno); return false; }
    bool ok = std::fwrite(header, 1, header_bytes, f) == header_bytes &&
              (payload_bytes == 0 || std::fwrite(payload, 1, payload_bytes, f) == payload_bytes) &&
              std::fwrite(&footer, 1, sizeof footer, f) == sizeof footer;
    if (ok && !leave_temp_only) ok = std::fflush(f) == 0;
#ifndef _WIN32
    if (ok && !leave_temp_only) ok = ::fsync(::fileno(f)) == 0;   // crash consistency: a record under its real name is durable
#else
    if (ok && !leave_temp_only) ok = ::_commit(::fileno(f)) == 0;
#endif
    std::fclose(f);
    if (leave_temp_only) { err = fail_at_err("the chunk write"); return false; }   // C1: the temp STAYS, nothing renamed
    if (!ok) {
        err = "kv-delta: write " + tmp;
        std::remove(tmp.c_str());
        return false;
    }
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
    return write_record(path, &h, sizeof h, payload, payload_bytes, footer, false, err);
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
    return write_record(path, &h, sizeof h, payload, payload_bytes, payload_footer(payload, payload_bytes), false,
                        err);
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

// ================================ keys (§5.7) and manifests (§5.6) ================================

uint64_t delta_tag(const ModelGeometry& g, int kv_format, bool cvec, uint64_t weights_fp, int64_t block) {
    // the material, in the order §5.7 lists it: the shared core's 18-field geometry key VERBATIM (its native
    // bytes - the same bytes NvmeHeader writes), then the format, the control-vector state, the weight-set
    // fingerprint, and BLOCK.  A conversation stored with any of these different never shares a chunk.
    const std::array<int64_t, 18> geo = strata::core::conversation_geometry_key(g);
    const int32_t cv = cvec ? 1 : 0;
    uint64_t h = nvme_fnv1a(kNvmeFnvBasis, geo.data(), geo.size() * sizeof(int64_t));
    h = nvme_fnv1a(h, &kv_format, sizeof kv_format);
    h = nvme_fnv1a(h, &cv, sizeof cv);
    h = nvme_fnv1a(h, &weights_fp, sizeof weights_fp);
    return nvme_fnv1a(h, &block, sizeof block);
}

uint64_t delta_chunk_key(uint64_t tag, const int32_t* ids, int64_t blocks) {
    // c_j = FNV1a(c_{j-1}, ids[(j-1)*BLOCK, j*BLOCK)); the caller asks for c_{j+1} of chunk j by passing j+1.
    // BLOCK is not a parameter: the chain is defined over the BLOCK-sized runs of the conversation's ids, and
    // every caller derives BLOCK from the same delta_shapes().
    const int64_t block = delta_shapes().block;
    uint64_t h = tag;
    for (int64_t j = 0; j < blocks; ++j) h = nvme_fnv1a(h, ids + (size_t) (j * block), (size_t) block * sizeof(int32_t));
    return h;
}

bool delta_write_manifest(const std::string& path, const DeltaManifestHeader& h, const std::vector<int32_t>& ids,
                          const std::vector<ConversationImageKey>& imgs, const std::vector<DeltaChunkRef>& chunks,
                          std::string& err) {
    if (h.magic != kDeltaManifestMagic || h.version != kDeltaFormatVersion) {
        err = "kv-delta: manifest " + path + ": refusing to write a header with a foreign magic/version";
        return false;
    }
    if (h.L != (int64_t) ids.size() || h.n_imgs != (int64_t) imgs.size() || h.n_chunks != (int64_t) chunks.size()) {
        err = "kv-delta: manifest " + path + ": header counts disagree with the body it is given";
        return false;
    }
    std::vector<uint8_t> body;
    body.reserve(ids.size() * sizeof(int32_t) + imgs.size() * sizeof(ConversationImageKey) +
                 chunks.size() * sizeof(DeltaChunkRef));
    body.insert(body.end(), (const uint8_t*) ids.data(), (const uint8_t*) (ids.data() + ids.size()));
    if (!imgs.empty())
        body.insert(body.end(), (const uint8_t*) imgs.data(), (const uint8_t*) (imgs.data() + imgs.size()));
    if (!chunks.empty())
        body.insert(body.end(), (const uint8_t*) chunks.data(),
                    (const uint8_t*) (chunks.data() + chunks.size()));
    const uint64_t footer = payload_footer(body.data(), body.size());
    return write_record(path, &h, sizeof h, body.data(), body.size(), footer, false, err);
}

bool delta_read_manifest(const std::string& path, DeltaManifestHeader& h, std::vector<int32_t>& ids,
                         std::vector<ConversationImageKey>& imgs, std::vector<DeltaChunkRef>& chunks,
                         std::string& err) {
    ids.clear(); imgs.clear(); chunks.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "kv-delta: manifest " + path + ": open: " + std::strerror(errno); return false; }
    if (std::fseek(f, 0, SEEK_END) != 0) { err = "kv-delta: manifest " + path + ": seek"; std::fclose(f); return false; }
    const long long fsize = ftello(f);
    std::rewind(f);
    if (fsize < (long long) (sizeof h + sizeof(uint64_t)) || fsize > (long long) 1 << 31) {
        err = "kv-delta: manifest " + path + ": not a manifest (size)";
        std::fclose(f);
        return false;
    }
    if (std::fread(&h, 1, sizeof h, f) != sizeof h) {
        err = "kv-delta: manifest " + path + ": truncated header";
        std::fclose(f);
        return false;
    }
    // LAYOUT FACTS FIRST: what the header claims must fit the file it arrived in, before any digest verdict.
    const uint64_t id_bytes = h.L < 0 ? 0 : (uint64_t) h.L * sizeof(int32_t);
    const uint64_t img_bytes = h.n_imgs < 0 ? 0 : (uint64_t) h.n_imgs * sizeof(ConversationImageKey);
    const uint64_t ref_bytes = h.n_chunks < 0 ? 0 : (uint64_t) h.n_chunks * sizeof(DeltaChunkRef);
    if (h.magic != kDeltaManifestMagic) {
        err = "kv-delta: manifest " + path + ": not a delta manifest (bad magic)";
        std::fclose(f);
        return false;
    }
    if (h.version != kDeltaFormatVersion) {
        err = "kv-delta: manifest " + path + ": format version " + std::to_string(h.version) +
              ", this build writes " + std::to_string(kDeltaFormatVersion) + " - refusing (nothing converts it)";
        std::fclose(f);
        return false;
    }
    if (h.L < 1 || h.n_imgs < 0 || h.n_chunks < 0 || h.block < 1 ||
        fsize != (long long) (sizeof h + id_bytes + img_bytes + ref_bytes + sizeof(uint64_t))) {
        err = "kv-delta: manifest " + path + ": malformed header sizes (" + std::to_string(fsize) +
              " bytes, header says " + std::to_string(sizeof h + id_bytes + img_bytes + ref_bytes + 8) + ")";
        std::fclose(f);
        return false;
    }
    if (h.n_chunks != h.L / h.block) {
        err = "kv-delta: manifest " + path + ": " + std::to_string(h.n_chunks) + " chunks for a " +
              std::to_string(h.L) + "-token boundary at block " + std::to_string(h.block) + " - refusing";
        std::fclose(f);
        return false;
    }
    std::vector<uint8_t> body((size_t) (id_bytes + img_bytes + ref_bytes));
    if (!body.empty() && std::fread(body.data(), 1, body.size(), f) != body.size()) {
        err = "kv-delta: manifest " + path + ": truncated body";
        std::fclose(f);
        return false;
    }
    uint64_t digest = 0;
    if (std::fread(&digest, 1, sizeof digest, f) != sizeof digest) {
        err = "kv-delta: manifest " + path + ": truncated footer";
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    if (digest != payload_footer(body.data(), body.size())) {
        err = "kv-delta: manifest " + path + ": integrity check failed (corrupt manifest)";
        return false;
    }
    size_t at = 0;
    ids.resize((size_t) h.L);
    std::memcpy(ids.data(), body.data() + at, id_bytes);
    at += (size_t) id_bytes;
    imgs.resize((size_t) h.n_imgs);
    if (h.n_imgs) {   // the image records are 16 bytes but not always 8-aligned: memcpy, never a cast
        std::memcpy(imgs.data(), body.data() + at, img_bytes);
        at += (size_t) img_bytes;
    }
    chunks.resize((size_t) h.n_chunks);
    for (int64_t j = 0; j < h.n_chunks; ++j) {
        DeltaChunkRef r;
        std::memcpy(&r, body.data() + at, sizeof r);
        at += sizeof r;
        // a chunk reference starts where its index says it must - the sealed lattice is not negotiable
        if (r.a != j * h.block) {
            err = "kv-delta: manifest " + path + ": chunk " + std::to_string(j) + " starts at " +
                  std::to_string(r.a) + ", not at its block boundary " + std::to_string(j * h.block);
            return false;
        }
        chunks[(size_t) j] = r;
    }
    return true;
}

// ================================ the writer (§5.9) ================================

bool delta_dump_at(const DeltaHead* prev, const std::string& dir, const SessionState& ss, const QsaState& mtp_state,
                   const ModelGeometry& g, const std::vector<int32_t>& ids,
                   const std::vector<ConversationImageKey>& imgs, bool cvec, const ConversationCheckpoint* at,
                   uint64_t weights_fp, int64_t dump_pid, int64_t seq, std::string& err) {
    const FailAt fail_at = fail_at_env();
    // The delta tier writes TURN BOUNDARIES only: without a checkpoint there is no boundary to key on and no
    // single-source running state, and that dump is the v3 path's job (the serve loop routes it there).
    if (!at) { err = "kv-delta: the delta tier writes turn boundaries only - no boundary checkpoint was given"; return false; }
    if (at && !at->stage_parts.empty()) {
        // the layer-split rule, verbatim in spirit from nvme_dump_at: the envelope carries the primary stage only
        err = "kv-delta: a layer-split session's later stages are not snapshot-able - refusing rather than "
              "writing incomplete";
        return false;
    }
    const int64_t T = (int64_t) at->ids.size();
    if (T < 1) { err = "kv-delta: empty boundary"; return false; }
    if (T != (int64_t) ids.size()) { err = "kv-delta: the boundary checkpoint's ids and the given ids disagree"; return false; }
    if (g.n_qsa_layers() > 0 && ss.qsa_states[0].kv_mode == 0) {
        err = "kv-delta: KV is fully resident (kv_mode 0) - the delta tier, like v3, needs the streamed host copy";
        return false;
    }
    DeltaShapes sh = delta_shapes();
    ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(g, z, err)) { err = "kv-delta: " + err; return false; }
    if (at->ids.size() != ids.size() ||
        at->gdn.size() != z.gdn ||
        at->tails.size() != z.tail * (size_t) g.n_qsa_layers() ||
        at->dead.size() != z.dead * (size_t) g.n_qsa_layers() ||
        at->block_pos.size() != z.block_pos * (size_t) g.n_qsa_layers() ||
        (!at->ple.empty() && at->ple.size() != z.ple)) {
        err = "kv-delta: turn-boundary checkpoint does not fit this engine";
        return false;
    }
    for (const ConversationImageKey& im : imgs)
        if (im.start < 0 || im.start >= T) {
            err = "kv-delta: an image record (start " + std::to_string(im.start) + ") is not inside the " +
                  std::to_string(T) + "-token prefix the manifest is keyed by";
            return false;
        }
    if (T > mtp_state.max_cells) {
        // the drafter ring wraps past max_cells; wrap-aware chunking is out of scope (§5.15) - fall back to v3
        err = "nvme delta: boundary " + std::to_string(T) + " exceeds the drafter ring (" +
              std::to_string(mtp_state.max_cells) + ") - whole snapshot";
        return false;
    }
    {
        // the same pooled-capacity guard nvme_dump_at applies: a prefix the live array cannot index is refused,
        // never written short
        const int64_t rows = strata::kernels::qsa_pooled_rows(T, sh.shapes);
        if (rows > ss.qsa_states[0].idx_pooled_rows) {
            err = "kv-delta: pooled rows: a " + std::to_string(T) + "-token prefix needs " + std::to_string(rows) +
                  " indexer pooled rows, this engine's array holds " + std::to_string(ss.qsa_states[0].idx_pooled_rows) +
                  " - refusing";
            return false;
        }
    }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        for (int k = 0; k < nvme_kv_array_count(ss.qsa_states[i]); ++k)
            if (!nvme_kv_host_array(ss.qsa_states[i], g.head_dim, k).p) {
                err = "kv-delta: null host KV array";
                return false;
            }

    const int64_t BLOCK = sh.block;
    const int64_t S = delta_sealed(T, sh);
    const int64_t n_chunks = S / BLOCK;
    const int64_t reuse = prev && prev->L > 0 && prev->L <= T && (int64_t) prev->ids.size() == prev->L &&
                                  std::equal(prev->ids.begin(), prev->ids.end(), ids.begin())
                              ? prev->L / BLOCK
                              : 0;   // every sealed chunk fully covered by the previous head; a fork reuses none

    const uint64_t tag = delta_tag(g, strata::core::qsa_kv_format(ss.qsa_states[0]), cvec, weights_fp, BLOCK);

    const std::string chunks_dir = dir + "/chunks", states_dir = dir + "/states";
    std::error_code ec;
    fs::create_directories(chunks_dir, ec);
    fs::create_directories(states_dir, ec);

    // ---- the chunk payload walker: the sealed chunk covering [a, a+BLOCK) is the v3 segments' own bytes for
    // that token range, in the v3 walk's order (§5.4).  Host KV slices are memcpy off the pinned arrays; the
    // pooled rows come off the device with ONE cudaMemcpy per layer (consume_cuda_error on failure).
    const int64_t pages_per_chunk = BLOCK / sh.shapes.page_size;
    std::vector<uint8_t> chunk_buf((size_t) delta_chunk_payload_bytes(ss, mtp_state, g, sh, 0));
    auto fill_chunk = [&](int64_t a) -> bool {
        size_t off = 0;
        for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
            const QsaState& st = ss.qsa_states[i];
            for (int k = 0; k < nvme_kv_array_count(st); ++k) {
                const NvmeKvArr ka = nvme_kv_host_array(st, g.head_dim, k);
                const size_t bytes = (size_t) pages_per_chunk * (size_t) (g.n_head_kv * sh.shapes.page_size * ka.w);
                std::memcpy(chunk_buf.data() + off, (const uint8_t*) ka.p + (size_t) ((a / sh.shapes.page_size) * g.n_head_kv * sh.shapes.page_size * ka.w), bytes);
                off += bytes;
            }
            const size_t pb = (size_t) sh.rows_per_chunk * (size_t) g.idx_key_dim * 4;
            if (pb && cudaMemcpy(chunk_buf.data() + off, st.idx_pooled + (size_t) ((a / sh.shapes.idx_block) * g.idx_key_dim),
                                 pb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                consume_cuda_error();
                err = "kv-delta: device-to-host copy failed for layer " + std::to_string(i) + "'s pooled rows";
                return false;
            }
            off += pb;
        }
        if (a < mtp_state.max_cells) {
            for (int k = 0; k < nvme_kv_array_count(mtp_state); ++k) {
                const NvmeKvArr ka = nvme_kv_host_array(mtp_state, g.head_dim, k);
                if (!ka.p) continue;
                const size_t bytes = (size_t) pages_per_chunk * (size_t) (g.n_head_kv * sh.shapes.page_size * ka.w);
                std::memcpy(chunk_buf.data() + off, (const uint8_t*) ka.p + (size_t) ((a / sh.shapes.page_size) * g.n_head_kv * sh.shapes.page_size * ka.w), bytes);
                off += bytes;
            }
        }
        return true;
    };

    std::vector<DeltaChunkRef> refs((size_t) n_chunks);
    for (int64_t j = 0; j < n_chunks; ++j) {
        refs[(size_t) j].key = delta_chunk_key(tag, ids.data(), j + 1);
        refs[(size_t) j].a = j * BLOCK;
    }
    // The FIRST new chunk is where C1/C2 fire (a dump reusing everything has no chunk step to crash in).
    const int64_t first_new = reuse < n_chunks ? reuse : -1;
    for (int64_t j = reuse; j < n_chunks; ++j) {
        if (!fill_chunk(j * BLOCK)) return false;
        DeltaChunkHeader ch;
        ch.key = refs[(size_t) j].key;
        ch.a = j * BLOCK;
        ch.b = (j + 1) * BLOCK;
        ch.layers = g.n_qsa_layers();
        ch.payload_bytes = (int64_t) chunk_buf.size();
        const std::string path = chunks_dir + "/" + delta_key_name(ch.key) + ".bin";
        if (fail_at == FailAt::kC1 && j == first_new) {
            // C1: the temp is written and closed, never fsynced, never renamed - the residue the sweep reclaims
            const uint64_t footer = nvme_fnv1a(kNvmeFnvBasis, chunk_buf.data(), chunk_buf.size());
            write_record(path, &ch, sizeof ch, chunk_buf.data(), chunk_buf.size(), footer, true, err);
            return false;
        }
        if (!delta_write_chunk(path, ch, chunk_buf.data(), chunk_buf.size(), err)) return false;
        if (fail_at == FailAt::kC2 && j == first_new) { err = fail_at_err("the chunk write"); return false; }
    }

    // ---- the State record: the ragged tail + the boundary checkpoint's running state (§5.5).  Every source is
    // the checkpoint's blob - the C5 rule - except the pooled rows and a missing ple blob, which are device
    // reads exactly where the v3 dump makes them.
    const int64_t tail_pages = (T + sh.shapes.page_size - 1) / sh.shapes.page_size - S / sh.shapes.page_size;
    const int64_t tail_rows = strata::kernels::qsa_pooled_rows(T, sh.shapes) - S / sh.shapes.idx_block;
    std::vector<uint8_t> state_buf((size_t) delta_state_payload_bytes(ss, mtp_state, g, z, sh, T));
    {
        size_t off = 0;
        std::memcpy(state_buf.data() + off, at->gdn.data(), z.gdn);
        off += z.gdn;
        if (ss.ple_hist) {
            if (!at->ple.empty()) {
                std::memcpy(state_buf.data() + off, at->ple.data(), z.ple);
            } else if (z.ple && cudaMemcpy(state_buf.data() + off, ss.ple_hist, z.ple, cudaMemcpyDeviceToHost) != cudaSuccess) {
                consume_cuda_error();
                err = "kv-delta: device-to-host copy failed for the ple history";
                return false;
            }
            off += z.ple;
        }
        for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
            const QsaState& st = ss.qsa_states[i];
            for (int k = 0; k < nvme_kv_array_count(st); ++k) {
                const NvmeKvArr ka = nvme_kv_host_array(st, g.head_dim, k);
                const size_t bytes = (size_t) tail_pages * (size_t) (g.n_head_kv * sh.shapes.page_size * ka.w);
                std::memcpy(state_buf.data() + off, (const uint8_t*) ka.p + (size_t) ((S / sh.shapes.page_size) * g.n_head_kv * sh.shapes.page_size * ka.w), bytes);
                off += bytes;
            }
            const size_t pb = (size_t) tail_rows * (size_t) g.idx_key_dim * 4;
            if (pb && cudaMemcpy(state_buf.data() + off, st.idx_pooled + (size_t) ((S / sh.shapes.idx_block) * g.idx_key_dim),
                                 pb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                consume_cuda_error();
                err = "kv-delta: device-to-host copy failed for layer " + std::to_string(i) + "'s tail pooled rows";
                return false;
            }
            off += pb;
            std::memcpy(state_buf.data() + off, at->tails.data() + (size_t) i * z.tail, z.tail);
            off += z.tail;
            std::memcpy(state_buf.data() + off, at->dead.data() + (size_t) i * z.dead, z.dead);
            off += z.dead;
            std::memcpy(state_buf.data() + off, at->block_pos.data() + (size_t) i * z.block_pos, z.block_pos);
            off += z.block_pos;
        }
        for (int k = 0; k < nvme_kv_array_count(mtp_state); ++k) {
            const NvmeKvArr ka = nvme_kv_host_array(mtp_state, g.head_dim, k);
            if (!ka.p) continue;
            const size_t bytes = (size_t) tail_pages * (size_t) (g.n_head_kv * sh.shapes.page_size * ka.w);
            std::memcpy(state_buf.data() + off, (const uint8_t*) ka.p + (size_t) ((S / sh.shapes.page_size) * g.n_head_kv * sh.shapes.page_size * ka.w), bytes);
            off += bytes;
        }
    }
    uint64_t state_key = 0;
    if (!delta_write_state(states_dir, tag, state_buf.data(), state_buf.size(), state_key, err)) return false;
    if (fail_at == FailAt::kC3) { err = fail_at_err("the state write"); return false; }

    // ---- the manifest: the head move IS the commit (§5.9 step 6).  Everything above is durable before it; the
    // previous head is unlinked only after it.
    DeltaManifestHeader h;
    h.L = T;
    h.block = BLOCK;
    h.n_chunks = n_chunks;
    h.n_imgs = (int64_t) imgs.size();
    h.cvec = cvec ? 1 : 0;
    h.kv_format = strata::core::qsa_kv_format(ss.qsa_states[0]);
    h.page_size = sh.shapes.page_size;
    h.idx_block = sh.shapes.idx_block;
    h.max_cells = ss.qsa_states[0].max_cells;
    h.geometry = strata::core::conversation_geometry_key(g);
    h.weights_fp = weights_fp;
    h.state_key = state_key;
    h.pid = dump_pid;
    h.seq = seq;
    h.mtp_host = 0;   // counted below, exactly as the v3 dump counts it: non-null drafter arrays
    for (int k = 0; k < nvme_kv_array_count(mtp_state); ++k) h.mtp_host += nvme_kv_host_array(mtp_state, g.head_dim, k).p ? 1 : 0;
    char name[64];
    std::snprintf(name, sizeof name, "log-%ld-%ld.manifest", (long) dump_pid, (long) seq);
    const std::string manifest_path = dir + "/" + name;
    if (!delta_write_manifest(manifest_path, h, ids, imgs, refs, err)) return false;
    if (fail_at == FailAt::kC4) { err = fail_at_err("the manifest write"); return false; }
    if (prev && !prev->path.empty() && prev->path != manifest_path) {
        fs::remove(prev->path, ec);   // the supersede: the old head's exclusive chunks become sweepable garbage
        if (ec) ec.clear();
    }
    if (fail_at == FailAt::kC5) { err = fail_at_err("the head move"); return false; }
    return true;
}

}  // namespace strata::platform
