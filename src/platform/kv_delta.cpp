// src/platform/kv_delta.cpp - see include/strata/platform/kv_delta.hpp.
#include "strata/platform/kv_delta.hpp"

#include "strata/kernels/qsa.hpp"  // qsa_real_shapes, qsa_pooled_rows

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>

#include <filesystem>
#include <sys/stat.h>

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
    sh.span = sh.block * kDeltaBlocksPerChunk;
    sh.rows_per_chunk = sh.block / sh.shapes.idx_block;
    return sh;
}

int64_t delta_chunk_payload_bytes(const SessionState& ss, const QsaState& mtp, const ModelGeometry& g,
                                  const DeltaShapes& sh, int64_t a) {
    const int64_t pages = sh.span / sh.shapes.page_size;   // sealed chunks are page-aligned by construction
    const int64_t page_kv_bytes = g.n_head_kv * sh.shapes.page_size;   // x the array's row width w
    int64_t bytes = 0;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        const QsaState& st = ss.qsa_states[i];
        for (int k = 0; k < nvme_kv_array_count(st); ++k)
            bytes += pages * page_kv_bytes * nvme_kv_host_array(st, g.head_dim, k).w;
        bytes += (sh.span / sh.shapes.idx_block) * g.idx_key_dim * 4;   // the chunk's whole span of pooled rows
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

/// The restore's own instrumentation (the restore-perf handoff's Phase 0): STRATA_DELTA_RESTORE_TIMING=1 makes
/// delta_restore print ONE line - manifest scan / chunk read+digest / assemble / apply / rss peak - so the
/// parallel-read (A) and streaming (B) work is measured, not felt.  Unset - the default, and the only shipped
/// configuration - nothing is printed.  Read live, like STRATA_DELTA_FAIL_AT.
bool delta_timing_on() {
    const char* e = std::getenv("STRATA_DELTA_RESTORE_TIMING");
    return e && e[0] && e[0] != '0';
}

/// The process's high-water RSS in MB, and the RSS at the moment of the call in MB (VmHWM / VmRSS).  The pair
/// is what makes the restore's OWN staging visible even inside the engine: the peak minus the entry RSS is the
/// transient the promote added.  0/0 when /proc is not there (Windows hosts) - the line then still carries the
/// phase timings.
void delta_rss_mb(uint64_t& hwm_mb, uint64_t& rss_mb) {
    hwm_mb = rss_mb = 0;
#ifndef _WIN32
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        long long kb = 0;
        if (std::sscanf(line, "VmHWM: %lld", &kb) == 1) hwm_mb = (uint64_t) (kb / 1024);
        else if (std::sscanf(line, "VmRSS: %lld", &kb) == 1) rss_mb = (uint64_t) (kb / 1024);
    }
    std::fclose(f);
#endif
}

using DeltaClock = std::chrono::steady_clock;
inline double delta_ms_since(DeltaClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(DeltaClock::now() - t0).count();
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
    // fdatasync, NOT fsync: the record's PAYLOAD is what the digest (and so the crash posture) needs durable;
    // the rename that publishes it is ordered by the manifest's own sync (§5.9: chunks before the manifest).
    // A full fsync per record forces an XFS journal commit per 61 KB chunk - measured on the live store at
    // 263 fsyncs/s with the device 87% utilized while a big turn's ~2,000-chunk append drained, stalling the
    // serve loop for the length of the dump. fdatasync keeps the crash contract (a torn record fails its
    // digest to `invalid` -> refuse -> re-prefill) at a fraction of the journal traffic.
    if (ok && !leave_temp_only) ok = ::fdatasync(::fileno(f)) == 0;
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

/// seconds since the epoch (the LRU clock) - kv_nvme.cpp's file_mtime, which the anon namespace there does not
/// export; the delta tier's mtime must be the SAME clock the v3 entries use for the cross-tier LRU to mean one
/// thing, and it is: seconds, from stat.
int64_t file_mtime_of(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
    return (int64_t) st.st_mtime;
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
    // the chunk span this manifest was written with (0 = the pre-grouping layout: one block per chunk)
    const int64_t k = h.blocks_per_chunk > 0 ? h.blocks_per_chunk : 1;
    if (h.n_chunks != h.L / (h.block * k)) {
        err = "kv-delta: manifest " + path + ": " + std::to_string(h.n_chunks) + " chunks for a " +
              std::to_string(h.L) + "-token boundary at block " + std::to_string(h.block) + " x " +
              std::to_string(k) + " - refusing";
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
        if (r.a != j * h.block * k) {
            err = "kv-delta: manifest " + path + ": chunk " + std::to_string(j) + " starts at " +
                  std::to_string(r.a) + ", not at its chunk boundary " + std::to_string(j * h.block * k);
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
    const int64_t SPAN = sh.span;                     // tokens per sealed chunk (kDeltaBlocksPerChunk blocks)
    const int64_t S = delta_sealed(T, sh);
    const int64_t n_chunks = S / SPAN;
    const int64_t reuse = prev && prev->L > 0 && prev->L <= T && (int64_t) prev->ids.size() == prev->L &&
                                  std::equal(prev->ids.begin(), prev->ids.end(), ids.begin())
                              ? prev->L / SPAN
                              : 0;   // every sealed chunk fully covered by the previous head; a fork reuses none

    const uint64_t tag = delta_tag(g, strata::core::qsa_kv_format(ss.qsa_states[0]), cvec, weights_fp, BLOCK);

    const std::string chunks_dir = dir + "/chunks", states_dir = dir + "/states";
    std::error_code ec;
    fs::create_directories(chunks_dir, ec);
    fs::create_directories(states_dir, ec);

    // ---- the chunk payload walker: the sealed chunk covering [a, a+BLOCK) is the v3 segments' own bytes for
    // that token range, in the v3 walk's order (§5.4).  Host KV slices are memcpy off the pinned arrays; the
    // pooled rows come off the device with ONE cudaMemcpy per layer (consume_cuda_error on failure).
    const int64_t pages_per_chunk = SPAN / sh.shapes.page_size;
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
            const size_t pb = (size_t) ((sh.span / sh.shapes.idx_block) * g.idx_key_dim * 4);   // the chunk's span of rows
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
        // chunk j covers the token span [j*SPAN, (j+1)*SPAN); its key is the id chain THROUGH ITS LAST TOKEN
        refs[(size_t) j].key = delta_chunk_key(tag, ids.data(), (j + 1) * kDeltaBlocksPerChunk);
        refs[(size_t) j].a = j * SPAN;
    }
    // The FIRST new chunk is where C1/C2 fire (a dump reusing everything has no chunk step to crash in).
    const int64_t first_new = reuse < n_chunks ? reuse : -1;
    for (int64_t j = reuse; j < n_chunks; ++j) {
        if (!fill_chunk(j * SPAN)) return false;
        DeltaChunkHeader ch;
        ch.key = refs[(size_t) j].key;
        ch.a = j * SPAN;
        ch.b = (j + 1) * SPAN;
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
    h.blocks_per_chunk = kDeltaBlocksPerChunk;   // the field's old reserved slot, now carrying the grouping
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

// ================================ the reader (§5.10) ================================

strata::core::ConversationRestore delta_restore(const NvmeEntry& e, SessionState& ss, QsaState& mtp_state,
                                                const ModelGeometry& g, uint64_t weights_fp,
                                                std::vector<int32_t>& ids, std::vector<ConversationImageKey>& imgs,
                                                bool& cvec, int64_t& L, std::string& err, uint64_t* image_bytes) {
    using Restore = strata::core::ConversationRestore;
    const DeltaClock::time_point t_start = DeltaClock::now();
    uint64_t rss_entry_mb = 0, hwm0_mb = 0;
    if (delta_timing_on()) delta_rss_mb(hwm0_mb, rss_entry_mb);   // the engine's own RSS, before the promote
    ids.clear();
    imgs.clear();
    // ---- step 1: the manifest.  Every refusal in this function happens before the first CUDA call, so it
    // provably writes nothing to the session (P6); only nvme_restore_image can report `transfer_failed`.
    DeltaManifestHeader h;
    std::vector<DeltaChunkRef> refs;
    if (!delta_read_manifest(e.path, h, ids, imgs, refs, err)) return Restore::invalid;
    DeltaShapes sh = delta_shapes();
    ConversationStateSizes z;
    if (!strata::core::conversation_state_sizes(g, z, err)) { err = "kv-delta: " + err; return Restore::invalid; }
    if (h.geometry != strata::core::conversation_geometry_key(g) ||
        h.kv_format != strata::core::qsa_kv_format(ss.qsa_states[0]) ||
        h.page_size != sh.shapes.page_size || h.idx_block != sh.shapes.idx_block ||
        h.block != sh.block || h.max_cells > ss.qsa_states[0].max_cells) {
        err = "kv-delta: manifest " + e.path + ": geometry/format mismatch (refusing to convert)";
        return Restore::invalid;
    }
    if (h.weights_fp != weights_fp) {
        // the match-time rule (§5.8), re-checked: a manifest of another weight set is not a candidate, ever
        err = "kv-delta: manifest " + e.path + ": belongs to a different weight set";
        return Restore::invalid;
    }
    int64_t mtp_arrays = 0;
    for (int k = 0; k < nvme_kv_array_count(mtp_state); ++k)
        mtp_arrays += nvme_kv_host_array(mtp_state, g.head_dim, k).p ? 1 : 0;
    if (h.mtp_host != mtp_arrays) {
        err = "kv-delta: manifest " + e.path + ": drafter arrays " + std::to_string(mtp_arrays) +
              " != manifest " + std::to_string(h.mtp_host);
        return Restore::invalid;
    }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        for (int k = 0; k < nvme_kv_array_count(ss.qsa_states[i]); ++k)
            if (!nvme_kv_host_array(ss.qsa_states[i], g.head_dim, k).p) {
                err = "kv-delta: null host KV array";
                return Restore::invalid;
            }

    const double t_manifest = delta_ms_since(t_start);
    // ---- step 2: every chunk and the State record, fully validated (existence, key/range, file size, footer
    // digest, identity via the tag) BEFORE anything is assembled.  A payload must also equal what THIS
    // geometry's slice math says - the shapes were checked, but the arrays' formats are the live engine's word.
    const uint64_t tag = delta_tag(g, h.kv_format, h.cvec != 0, weights_fp, sh.block);
    // the chunk span this manifest was written with (0 = the pre-grouping layout: one block per chunk)
    const int64_t K = h.blocks_per_chunk > 0 ? h.blocks_per_chunk : 1;
    const int64_t T = h.L, S = delta_sealed(T, sh);
    const int64_t pages_per_chunk = sh.block * K / sh.shapes.page_size;
    const std::string dir = fs::path(e.path).parent_path().string();
    std::vector<std::vector<uint8_t>> chunk_payloads(refs.size());
    for (int64_t j = 0; j < (int64_t) refs.size(); ++j) {
        const std::string path = dir + "/chunks/" + delta_key_name(refs[(size_t) j].key) + ".bin";
        std::vector<uint8_t>& payload = chunk_payloads[(size_t) j];
        if (!delta_read_chunk(path, refs[(size_t) j].key, refs[(size_t) j].a, refs[(size_t) j].a + sh.block * K,
                              payload, err))
            return Restore::invalid;   // the read's message already names the chunk and the reason
        const int64_t want = delta_chunk_payload_bytes(ss, mtp_state, g, sh, refs[(size_t) j].a);
        if (payload.size() != (size_t) want) {
            err = "kv-delta: chunk " + path + ": payload " + std::to_string(payload.size()) +
                  " bytes, this geometry's slice math says " + std::to_string(want);
            return Restore::invalid;
        }
    }
    const std::string state_path = dir + "/states/" + delta_key_name(h.state_key) + ".bin";
    std::vector<uint8_t> state;
    if (!delta_read_state(state_path, tag, h.state_key, state, err)) return Restore::invalid;
    {
        const int64_t want = delta_state_payload_bytes(ss, mtp_state, g, z, sh, T);
        if (state.size() != (size_t) want) {
            err = "kv-delta: state " + state_path + ": payload " + std::to_string(state.size()) +
                  " bytes, this geometry's slice math says " + std::to_string(want);
            return Restore::invalid;
        }
    }

    const double t_chunks = delta_ms_since(t_start);
    // ---- step 3: assemble the EXACT v3 image in one buffer (the whole-file staging the v3 restore already
    // does - the measured, accepted C10 cost), interleaving chunk and State slices per the v3 walk's order.
    const bool has_ple = ss.ple_hist != nullptr;
    const int64_t pagesT = (T + sh.shapes.page_size - 1) / sh.shapes.page_size;
    const int64_t rowsT = strata::kernels::qsa_pooled_rows(T, sh.shapes);
    const int64_t idx4 = g.idx_key_dim * 4;
    const int n_arrays = nvme_kv_array_count(ss.qsa_states[0]);
    int64_t widths[4] = {0, 0, 0, 0};
    for (int k = 0; k < n_arrays; ++k) widths[k] = nvme_kv_host_array(ss.qsa_states[0], g.head_dim, k).w;
    const int64_t page_kv = g.n_head_kv * sh.shapes.page_size;
    const int64_t st_pages = pagesT - S / sh.shapes.page_size;   // the State record's tail-page count per array
    const int64_t st_rows = rowsT - S / sh.shapes.idx_block;
    // per-layer strides, chunk side and state side; the drafter's arrays ride after the layers in BOTH
    int64_t ch_arrays = 0, st_arrays = 0, dr_slice = 0;
    for (int k = 0; k < n_arrays; ++k) {
        ch_arrays += pages_per_chunk * page_kv * widths[k];
        st_arrays += st_pages * page_kv * widths[k];
        dr_slice += pages_per_chunk * page_kv * widths[k];
    }
    const int64_t ch_stride = ch_arrays + (sh.span / sh.shapes.idx_block) * idx4;   // the chunk's span of rows
    const int64_t st_stride = st_arrays + st_rows * idx4 + (int64_t) (z.tail + z.dead + z.block_pos);
    const int64_t st_drafter_base = (int64_t) z.gdn + (has_ple ? (int64_t) z.ple : 0) + g.n_qsa_layers() * st_stride;
    int64_t arr_total = 0;
    for (int k = 0; k < n_arrays; ++k) arr_total += pagesT * page_kv * widths[k];
    const size_t payload_bytes = sizeof(NvmeHeader) + (size_t) T * 4 +
                                 imgs.size() * sizeof(ConversationImageKey) + z.gdn + (has_ple ? z.ple : 0) +
                                 (size_t) g.n_qsa_layers() *
                                     (size_t) (arr_total + rowsT * idx4 + (int64_t) (z.tail + z.dead + z.block_pos)) +
                                 (size_t) arr_total +   // the drafter: T <= max_cells, so the ring covers the prefix
                                 sizeof(uint64_t);
    std::vector<uint8_t> buf(payload_bytes, 0);
    // the RAM this promote stages, for the record (the KV line's staging_bytes).  Reported from the size of the
    // buffer, which is what the C10 cost IS; nothing here changes what the buffer holds.
    if (image_bytes) *image_bytes = payload_bytes;

    NvmeHeader v3;
    v3.L = T;
    v3.n_imgs = (int64_t) imgs.size();
    v3.cvec = h.cvec;
    v3.kv_format = h.kv_format;
    v3.geometry = h.geometry;
    v3.page_size = h.page_size; v3.idx_block = h.idx_block; v3.max_cells = h.max_cells;
    v3.mtp_host = h.mtp_host;
    size_t at = 0;
    auto put = [&](const void* p, size_t n) { std::memcpy(buf.data() + at, p, n); at += n; };
    put(&v3, sizeof v3);
    put(ids.data(), (size_t) T * 4);
    if (!imgs.empty()) put(imgs.data(), imgs.size() * sizeof(ConversationImageKey));
    put(state.data(), z.gdn);                       // gdn: the State record's first slice
    if (has_ple) put(state.data() + z.gdn, z.ple);  // then ple, when the session has PLE history
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        const size_t ch_base = (size_t) (i * ch_stride);
        const size_t st_base = (size_t) (z.gdn + (has_ple ? (int64_t) z.ple : 0) + i * st_stride);
        int64_t ch_off = 0, st_off = 0;
        for (int k = 0; k < n_arrays; ++k) {
            const size_t slice = (size_t) (pages_per_chunk * page_kv * widths[k]);
            for (int64_t j = 0; j < (int64_t) refs.size(); ++j)
                put(chunk_payloads[(size_t) j].data() + ch_base + (size_t) ch_off, slice);
            put(state.data() + st_base + (size_t) st_off, (size_t) (st_pages * page_kv * widths[k]));
            ch_off += (int64_t) slice;
            st_off += (int64_t) (st_pages * page_kv * widths[k]);
        }
        const size_t rows_slice = (size_t) ((sh.span / sh.shapes.idx_block) * idx4);
        for (int64_t j = 0; j < (int64_t) refs.size(); ++j)
            put(chunk_payloads[(size_t) j].data() + ch_base + (size_t) ch_off, rows_slice);
        put(state.data() + st_base + (size_t) st_off, (size_t) (st_rows * idx4));
        st_off += (int64_t) (st_rows * idx4);
        put(state.data() + st_base + (size_t) st_off, z.tail);   st_off += (int64_t) z.tail;
        put(state.data() + st_base + (size_t) st_off, z.dead);   st_off += (int64_t) z.dead;
        put(state.data() + st_base + (size_t) st_off, z.block_pos);
    }
    {   // the drafter: the chunks' pages for [0, S) plus the State record's tail pages, per non-null array.
        // TWO offsets: the chunk's drafter section strides by the CHUNK's per-array slice (span/page pages),
        // while the State's drafter section strides by the TAIL's per-array slice (st_pages pages) - one
        // running offset was correct only when a chunk was ONE block (the strides then coincided), and the
        // K-grouping turned the difference into silently wrong scale-array bytes.
        const size_t ch_drafter = (size_t) (g.n_qsa_layers() * ch_stride);
        int64_t ch_dr = 0, st_dr = 0;
        for (int k = 0; k < n_arrays; ++k) {
            const size_t csz = (size_t) (pages_per_chunk * page_kv * widths[k]);
            const size_t ssz = (size_t) (st_pages * page_kv * widths[k]);
            for (int64_t j = 0; j < (int64_t) refs.size(); ++j)
                put(chunk_payloads[(size_t) j].data() + ch_drafter + (size_t) ch_dr, csz);
            put(state.data() + st_drafter_base + (size_t) st_dr, ssz);
            ch_dr += (int64_t) csz;
            st_dr += (int64_t) ssz;
        }
    }
    // the walk must land exactly on the footer - an assembly bug here would otherwise hide behind the digest
    // check as a mysterious "corrupt" verdict, so it refuses loudly instead
    if (at + sizeof(uint64_t) != payload_bytes) {
        err = "kv-delta: internal: assembled " + std::to_string(at) + " payload bytes, sized " +
              std::to_string(payload_bytes - sizeof(uint64_t));
        return Restore::invalid;
    }
    // the digest covers the payload only, never the header nor the footer itself (the v3 restore's own rule)
    const uint64_t digest = nvme_fnv1a(kNvmeFnvBasis, buf.data() + sizeof(NvmeHeader),
                                       payload_bytes - sizeof(NvmeHeader) - sizeof(uint64_t));
    std::memcpy(buf.data() + payload_bytes - sizeof digest, &digest, sizeof digest);

    const double t_assemble = delta_ms_since(t_start);
    // ---- step 4: the EXISTING validation+apply pass, unchanged - layout walk, drift diagnostics, digest,
    // apply, the STATE_HASH gate; the failure classes are its own (§5.10 step 4)
    const strata::core::ConversationRestore rc =
        nvme_restore_image(buf.data(), buf.size(), ss, mtp_state, g, ids, imgs, cvec, L, err);
    if (delta_timing_on()) {
        // one line, all phases: read-back volume for context, then ms per phase, then the process's peak RSS
        // (VmHWM) beside the RSS the process held at entry, so the promote's OWN transient is the difference.
        uint64_t read_back = state.size();
        for (const std::vector<uint8_t>& p : chunk_payloads) read_back += p.size();
        uint64_t hwm_mb = 0, now_mb = 0;
        delta_rss_mb(hwm_mb, now_mb);
        std::fprintf(stderr,
                     "strata serve: kv-delta restore timing: manifest %.1f ms, read+digest %.1f ms, "
                     "assemble %.1f ms, apply %.1f ms, rss peak %llu MB (entry %llu MB), T %lld, %lld chunks, "
                     "%.2f GiB read\n",
                     t_manifest, t_chunks - t_manifest, t_assemble - t_chunks,
                     delta_ms_since(t_start) - t_assemble, (unsigned long long) hwm_mb,
                     (unsigned long long) rss_entry_mb, (long long) h.L, (long long) refs.size(),
                     (double) read_back / (double) (1LL << 30));
    }
    return rc;
}


// ================================ the store (§5.11-§5.12) ================================

uint64_t kv_delta_weights_fp(const std::vector<std::string>& model_files) {
    // path bytes || size (int64) || first 64 KiB || last 64 KiB, per file, in the order the engine loads them
    uint64_t h = kNvmeFnvBasis;
    for (const std::string& file : model_files) {
        h = nvme_fnv1a(h, file.data(), file.size());
        std::error_code ec;
        const uint64_t size = (uint64_t) fs::file_size(file, ec);
        h = nvme_fnv1a(h, &size, sizeof size);
        FILE* f = std::fopen(file.c_str(), "rb");
        if (!f) continue;   // an unreadable shard fingerprints as path+size only - still unique enough to refuse
        char window[1 << 16];
        const size_t head = (size_t) std::fread(window, 1, sizeof window, f);
        h = nvme_fnv1a(h, window, head);
        if (std::fseek(f, 0, SEEK_END) == 0) {
            const long long sz = ftello(f);
            const long long tail_start = sz > (long long) sizeof window ? sz - (long long) sizeof window : (long long) head;
            if (sz > 0 && std::fseek(f, tail_start, SEEK_SET) == 0) {
                const size_t tail = (size_t) std::fread(window, 1, (size_t) (sz - tail_start), f);
                h = nvme_fnv1a(h, window, tail);
            }
        }
        std::fclose(f);
    }
    return h;
}

bool KvDeltaStore::open(const std::string& v3_dir, const ModelGeometry& g, int kv_format,
                        const std::vector<std::string>& model_files, std::string& err) {
    dir_ = v3_dir + "/delta";
    // each open claims a block of manifest numbers, so two instances in ONE process never rename a manifest
    // over another instance's file (across processes the pid already separates them)
    static long instances = 0;
    seq_base_ = ++instances * 1000000;
    seq_ = 0;
    weights_fp_ = kv_delta_weights_fp(model_files);
    const DeltaShapes sh = delta_shapes();
    std::error_code ec;
    fs::create_directories(dir_ + "/chunks", ec);
    fs::create_directories(dir_ + "/states", ec);
    if (ec) { err = "kv-delta: create " + dir_ + ": " + ec.message(); return false; }
    // one pass over chunks/ and states/: file sizes by key, and the tier's total byte count (including garbage
    // the coming sweep will remove - the cap must not lie about the disk)
    std::map<std::string, uint64_t> chunk_bytes, state_bytes;
    total_ = 0;
    for (const char* sub : {"/chunks", "/states"}) {
        std::map<std::string, uint64_t>& map = std::string(sub) == "/chunks" ? chunk_bytes : state_bytes;
        for (const fs::directory_entry& de : fs::directory_iterator(dir_ + sub, ec)) {
            if (ec) break;
            if (!de.is_regular_file()) continue;
            const std::string name = de.path().filename().string();
            const uint64_t sz = (uint64_t) de.file_size(ec);
            total_ += sz;
            if (name.rfind(".tmp-", 0) == 0) continue;      // a crash's residue: counted (then swept), never a record
            if (name.size() > 4) map[name.substr(0, name.size() - 4)] = sz;   // strip .bin
        }
    }
    size_t foreign = 0, other_weights = 0;
    for (const fs::directory_entry& de : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        const std::string name = de.path().filename().string();
        if (!de.is_regular_file() || name.rfind("log-", 0) != 0) continue;
        DeltaManifestHeader h;
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<DeltaChunkRef> refs;
        if (!delta_read_manifest(de.path().string(), h, ids, imgs, refs, err)) { ++foreign; continue; }
        // the same refuse-never-convert rule the v3 scan applies: another format/shape is left on disk and
        // skipped - and a weight-set mismatch is said OUT LOUD, once, because every conversation in it will
        // re-prefill for as long as this binary serves (§5.8)
        if (h.geometry != strata::core::conversation_geometry_key(g) || h.kv_format != kv_format ||
            h.page_size != sh.shapes.page_size || h.idx_block != sh.shapes.idx_block || h.block != sh.block) {
            ++foreign; continue;
        }
        if (h.weights_fp != weights_fp_) { ++other_weights; continue; }
        NvmeEntry e;
        e.path = de.path().string();
        e.ids = std::move(ids);
        e.imgs = std::move(imgs);
        e.L = h.L;
        e.cvec = h.cvec != 0;
        e.kind = 1;
        e.mtime = file_mtime_of(e.path);
        e.bytes = (uint64_t) fs::file_size(de.path(), ec);
        if (state_bytes.count(delta_key_name(h.state_key))) e.bytes += state_bytes.at(delta_key_name(h.state_key));
        std::vector<uint64_t> keys;
        keys.reserve(refs.size());
        for (const DeltaChunkRef& r : refs) {
            keys.push_back(r.key);
            auto it = chunk_bytes.find(delta_key_name(r.key));
            if (it != chunk_bytes.end()) e.bytes += it->second;
        }
        total_ += e.bytes;
        entry_chunks_.push_back(std::move(keys));
        entry_states_.push_back(h.state_key);
        entries_.push_back(std::move(e));
    }
    if (foreign)
        std::fprintf(stderr, "strata serve: kv-delta: %zu unreadable/foreign manifest(s) skipped in %s\n",
                     foreign, dir_.c_str());
    if (other_weights)
        std::fprintf(stderr,
                     "strata serve: kv-delta: %zu stored conversations belong to a different weight set - skipped\n",
                     other_weights);
    sweep();   // §5.12: sweep at open, after the scan
    return true;
}

bool KvDeltaStore::dump(const SessionState& ss, const QsaState& mtp_state, const ModelGeometry& g,
                        const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec,
                        const ConversationCheckpoint* at, std::string& err, TierActivity* act) {
    if (!at || at->ids.empty()) { err = "kv-delta: the delta tier dumps turn boundaries only"; return false; }
    const std::vector<int32_t>& key = at->ids;
    const std::vector<ConversationImageKey>& stored = at->imgs;
    // idempotent: this exact state is already the head - refresh recency, write nothing
    for (NvmeEntry& e : entries_)
        if (e.L == (int64_t) key.size() && e.cvec == cvec && e.imgs.size() == stored.size() &&
            std::equal(key.begin(), key.end(), e.ids.begin()) && std::equal(stored.begin(), stored.end(), e.imgs.begin())) {
            e.mtime = (int64_t) ::time(nullptr);
            if (act) act->skipped = true;   // the head already holds this state: recency refreshed, nothing written
            return true;
        }
    // this process's previous head, if the new ids extend it (a fork or a rewrite reuses none - the writer
    // checks; the store's bookkeeping below may only drop the old ENTRY when the writer really superseded)
    DeltaHead prev;
    bool superseded = false;
    if (!last_ids_.empty() && last_ids_.size() <= key.size() &&
        std::equal(last_ids_.begin(), last_ids_.end(), key.begin())) {
        prev.L = (int64_t) last_ids_.size();
        prev.ids = last_ids_;
        prev.path = last_path_;
        superseded = true;
    }
    const int64_t prev_T = prev.L;
    const int64_t reused = prev_T / delta_shapes().span;   // whole sealed chunks the previous head covers
    const long seq = seq_base_ + seq_++;
    const std::string path = dir_ + "/log-" + std::to_string(pid_of()) + "-" + std::to_string(seq) + ".manifest";
    if (!delta_dump_at(prev.L ? &prev : nullptr, dir_, ss, mtp_state, g, key, stored, cvec, at, weights_fp_,
                       pid_of(), seq, err)) {
        return false;   // no manifest move happened; the only residue is content-addressed garbage (§5.13)
    }
    // register the entry, with the byte count the ON-DISK records actually have (manifest + state + its chunks)
    DeltaManifestHeader h;
    std::vector<int32_t> rids;
    std::vector<ConversationImageKey> rimgs;
    std::vector<DeltaChunkRef> refs;
    if (!delta_read_manifest(path, h, rids, rimgs, refs, err)) {
        err = "kv-delta: the manifest the writer just committed does not read back: " + err;
        return false;
    }
    NvmeEntry e;
    e.path = path;
    e.ids = key;
    e.imgs = stored;
    e.L = (int64_t) key.size();
    e.cvec = cvec;
    e.kind = 1;
    e.mtime = (int64_t) ::time(nullptr);
    std::error_code ec;
    e.bytes = (uint64_t) fs::file_size(path, ec);
    uint64_t appended = e.bytes;   // the manifest rewrite is this turn's write too
    {
        std::error_code ec2;
        const uint64_t state_bytes = (uint64_t) fs::file_size(dir_ + "/states/" + delta_key_name(h.state_key) + ".bin", ec2);
        e.bytes += state_bytes;
        appended += state_bytes;   // + the State record
    }
    for (size_t j = 0; j < refs.size(); ++j) {
        std::error_code ec2;
        const uint64_t sz = (uint64_t) fs::file_size(dir_ + "/chunks/" + delta_key_name(refs[j].key) + ".bin", ec2);
        if (ec2) continue;
        e.bytes += sz;
        if ((int64_t) j >= reused) appended += sz;   // only the chunks THIS turn sealed are this turn's write cost
    }
    // the supersede: the writer unlinked the previous head's manifest; drop its entry and bytes from the books.
    // ONLY on a real supersede - a fork's (or rewrite's) dump left the previous head's manifest ON DISK, and its
    // entry must stay: the store's list is the index of what the disk holds, not of what this process dumped last.
    if (superseded) {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].path == last_path_) {
                if (act) { ++act->dropped; act->dropped_bytes += entries_[i].bytes; }
                total_ -= entries_[i].bytes;
                entries_.erase(entries_.begin() + (long) i);
                entry_chunks_.erase(entry_chunks_.begin() + (long) i);
                entry_states_.erase(entry_states_.begin() + (long) i);
                break;
            }
    }
    total_ += e.bytes;
    last_ids_ = key;
    last_path_ = path;
    {
        std::vector<uint64_t> keys;
        keys.reserve(refs.size());
        for (const DeltaChunkRef& r : refs) keys.push_back(r.key);
        entry_chunks_.push_back(std::move(keys));
        entry_states_.push_back(h.state_key);
    }
    entries_.push_back(std::move(e));
    // the instrumentation line the oracles (and the endurance report) read: this turn's write volume
    std::fprintf(stderr, "strata serve: nvme delta: appended %lld chunks (%.1f MiB) T %lld->%lld\n",
                 (long long) ((int64_t) refs.size() - reused), (double) appended / (double) (1 << 20),
                 (long long) prev_T, (long long) e.L);
    if (act) act->written = appended;   // THE SAME NUMBER the line above prints, so a counter is checkable
                                        // against a line the oracles can already grep
    return true;
}

strata::core::ConversationRestore KvDeltaStore::restore(const NvmeEntry& e, SessionState& ss, QsaState& mtp_state,
                                                        const ModelGeometry& g, std::string& err) {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    bool cvec = false;
    int64_t L = 0;
    last_image_bytes_ = 0;   // a refusal before step 3 staged nothing
    const strata::core::ConversationRestore r =
        delta_restore(e, ss, mtp_state, g, weights_fp_, ids, imgs, cvec, L, err, &last_image_bytes_);
    if (r != strata::core::ConversationRestore::restored) return r;
    if (L != e.L || cvec != e.cvec || imgs != e.imgs || ids != e.ids) {
        // the file disagreed with the index the scan built - the v3 store's TOCTOU rule, same class: the image
        // applied cleanly and the final sync succeeded, which IS the proof a clean reset needs
        err = "kv-delta: entry changed under us";
        return strata::core::ConversationRestore::invalid;
    }
    return strata::core::ConversationRestore::restored;
}

void KvDeltaStore::drop(const NvmeEntry& e) {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].path == e.path) {
            std::error_code ec;
            fs::remove(entries_[i].path, ec);
            total_ -= entries_[i].bytes;
            if (entries_[i].path == last_path_) { last_ids_.clear(); last_path_.clear(); }
            entries_.erase(entries_.begin() + (long) i);
            entry_chunks_.erase(entry_chunks_.begin() + (long) i);
            entry_states_.erase(entry_states_.begin() + (long) i);
            return;
        }
}

TierActivity KvDeltaStore::sweep() {
    TierActivity act;
    // mark: the union of everything the LIVE MANIFESTS reference - read from the DISK, not from the in-memory
    // entry list, because the disk can be AHEAD of it: a dump that failed after its manifest was renamed (the
    // crash matrix's C4/C5) left a committed head this instance never registered, and sweeping against the
    // stale marks would delete a live conversation's chunks.  The union is computed BEFORE any unlink - a chunk
    // shared with a live manifest is never garbage, which is what makes eviction safe with forks around.
    std::map<std::string, bool> mark;
    std::error_code ec;
    for (const fs::directory_entry& de : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        if (!de.is_regular_file() || de.path().filename().string().rfind("log-", 0) != 0) continue;
        DeltaManifestHeader h;
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<DeltaChunkRef> refs;
        std::string merr;   // an unreadable manifest is not a mark source; its reason is not this sweep's news
        if (!delta_read_manifest(de.path().string(), h, ids, imgs, refs, merr)) continue;
        mark[delta_key_name(h.state_key)] = true;
        for (const DeltaChunkRef& r : refs) mark[delta_key_name(r.key)] = true;
    }
    size_t swept = 0;
    uint64_t bytes = 0;
    uint64_t kept = 0;
    for (const char* sub : {"/chunks", "/states"}) {
        for (const fs::directory_entry& de : fs::directory_iterator(dir_ + sub, ec)) {
            if (ec) break;
            if (!de.is_regular_file()) continue;
            const std::string name = de.path().filename().string();
            const bool temp = name.rfind(".tmp-", 0) == 0;
            const std::string key = (!temp && name.size() > 4) ? name.substr(0, name.size() - 4) : name;
            const uint64_t sz = (uint64_t) de.file_size(ec);
            if (temp || !mark[key]) {
                bytes += sz;
                fs::remove(de.path(), ec);
                ++swept;
            } else {
                kept += sz;
            }
        }
    }
    // RECOMPUTE the tier's total from the disk instead of subtracting the swept bytes: the supersede already
    // subtracted the old head's entry (whose bytes counted the records the sweep now deletes), so a plain
    // subtraction would take the same file off the books TWICE - the total drifted low by ~the State record's
    // size per superseded turn, and a low total makes the cap UNDER-evict (the disk grows past --kv-nvme-max).
    // The walk above is the disk's truth; the manifests ride on top of it.
    total_ = kept;
    for (const fs::directory_entry& de : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        if (de.is_regular_file() && de.path().filename().string().rfind("log-", 0) == 0)
            total_ += (uint64_t) de.file_size(ec);
    }
    act.swept = (int64_t) swept;
    act.swept_bytes = bytes;
    if (swept)
        std::fprintf(stderr, "strata serve: kv-delta: swept %zu orphan chunks (%.2f GiB)\n",
                     swept, (double) bytes / (double) (1LL << 30));
    return act;
}

TierActivity kv_delta_enforce_cap(KvNvmeStore& v3, KvDeltaStore& delta, int64_t cap_bytes) {
    TierActivity act;
    // the last entry is kept even over the cap (never empty the store) - the v3 store's own documented policy
    while (cap_bytes > 0 && v3.total_bytes() + delta.total_bytes() > (uint64_t) cap_bytes &&
           v3.size() + delta.size() > 1) {
        const NvmeEntry* oldest = nullptr;
        bool from_delta = false;
        for (const NvmeEntry& e : v3.entries())
            if (!oldest || e.mtime < oldest->mtime) { oldest = &e; from_delta = false; }
        for (const NvmeEntry& e : delta.entries())
            if (!oldest || e.mtime < oldest->mtime) { oldest = &e; from_delta = true; }
        if (!oldest) break;
        const uint64_t victim_bytes = oldest->bytes;   // read it BEFORE the drop erases the entry
        if (from_delta) delta.drop(*oldest);
        else v3.drop(*oldest);
        ++act.evicted;
        act.evicted_bytes += victim_bytes;
    }
    const TierActivity swept = delta.sweep();   // §5.12: at cap pressure, AFTER eviction
    act.swept = swept.swept;
    act.swept_bytes = swept.swept_bytes;
    return act;
}

}  // namespace strata::platform
