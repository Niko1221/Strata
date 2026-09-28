// src/platform/kv_nvme.cpp - see include/strata/platform/kv_nvme.hpp.
#include "strata/platform/kv_nvme.hpp"

#include "strata/core/conversation_snapshot.hpp"   // conversation_state_sizes: the shared running-state byte counts
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <type_traits>

#include <filesystem>
#include <sys/stat.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

#include "strata/kernels/kv_stream.hpp"   // KvFormat, kv_stream_reset
#include "strata/kernels/kv_q4.hpp"       // kv_q4_bytes_per_head
#include "strata/kernels/qsa.hpp"         // qsa_real_shapes

namespace strata::platform {

// The envelope's image segment is one 16-byte (start, hash) record per image.  The plumbing now names the shared
// core's type instead of std::pair<int64_t, uint64_t>; the BYTES must not move in this step, and this is what
// proves they did not (docs/nvme-kv-cache-convergence.md step 2).  sizeof/alignof alone would not prove it: they
// say the record is still 16 bytes, not that `start` is still the first eight of them.  A standard-layout class
// lays its non-static data members out in declaration order, so pinning that here makes the record's own layout
// explicit - a field added, retyped OR REORDERED fails the build.  (The pair comparison is the tripwire against
// the width the v2 files were actually written with, not a claim about std::pair's layout.)
static_assert(std::is_standard_layout_v<strata::core::ConversationImageKey> &&
              sizeof(strata::core::ConversationImageKey) == sizeof(std::pair<int64_t, uint64_t>) &&
              alignof(strata::core::ConversationImageKey) == alignof(std::pair<int64_t, uint64_t>),
              "the shared image key must keep the v2 envelope's 16-byte (start, hash) image record");

namespace {

namespace fs = std::filesystem;

/// The byte counts the envelope is laid out with.  The running-state ones come from the SHARED CORE
/// (`strata::core::conversation_state_sizes`) - this file no longer carries a second copy of those formulas
/// (docs/nvme-kv-cache-convergence.md step 2).  `page_size` / `idx_block` are the granules the segment walk
/// needs, read from the same `qsa_real_shapes()` the shared core reads.
struct Sizes {
    int64_t page_size = 0, idx_block = 0;
    strata::core::ConversationStateSizes state;   // gdn / ple / tail / dead / block_pos bytes
};

/// THE POOLED-ROW COUNT, stated once and used by both the dump and the restore (collision C4, settled on the
/// SHARED CORE's formula - `conversation_snapshot.cpp:45`): the completed block rows [0, L/idx_block) plus the
/// SPARE row at L/idx_block.  The spare row is not padding - qsa.cu:213 and native_qsa_indexer.cu:93 keep
/// `pooled[n_bid]` equal to `dead`, and qsa.cu:260 and native_qsa_score.cu:74 read row n_bid straight out of the
/// pool, so a snapshot that stopped at the completed rows would leave the block in progress scored with a stale
/// key.  Our old `L / idx_block + 2` wrote one row MORE than that: what the live array happens to hold at dump
/// time, and unreachable at resume, because every pooled reader gates on n_bid (qsa.cu:253 `b > n_bid`,
/// qsa_select.cu:33 - which reads `dead` for `b == n_bid`, so its highest pooled read is n_bid - 1,
/// native_qsa_score.cu:74 `row <= full`).  The writer does touch row n_bid+1 when a block completes
/// (qsa.cu:213 seeds it with `dead`), so a stale value left there by a shorter snapshot is overwritten before
/// anything can read it.
///
/// A live array too small for the snapshot REFUSES, as their `conversation_kv_validate` does
/// (`conversation_snapshot.cpp:72`); our old `min(..., idx_pooled_rows)` clamp wrote a SHORT segment that the
/// restore then reported as layout drift.
bool snapshot_pooled_rows(int64_t L, const Sizes& z, const strata::core::QsaState& st, int64_t& rows,
                          std::string& err) {
    rows = L / z.idx_block + 1;
    if (rows > st.idx_pooled_rows) {
        err = "pooled rows: a " + std::to_string(L) + "-token prefix needs " + std::to_string(rows) +
              " indexer pooled rows, this engine's array holds " + std::to_string(st.idx_pooled_rows) +
              " - refusing";
        return false;
    }
    return true;
}

bool sizes_of(const strata::core::ModelGeometry& g, Sizes& z, std::string& err) {
    const strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    z.page_size = s.page_size;
    z.idx_block = s.idx_block;
    // The shared core can REFUSE to size a geometry (a non-positive field, an overflow in the byte product).  It
    // says so with a string naming which one, and that string used to be dropped on the floor here: a refused
    // sizing left every count at zero, and the envelope was then laid out and walked with zero-length segments.
    if (!strata::core::conversation_state_sizes(g, z.state, err)) {
        err = "kv-nvme: " + err;
        return false;
    }
    return true;
}

// bytes per (cell, head) row of one KV array, for the state's format
int64_t row_bytes(const strata::core::QsaState& st, int64_t head_dim) {
    if (st.kv_q4) return (int64_t) strata::kernels::kv_q4_bytes_per_head((int) head_dim);
    if (st.kv_int8) return head_dim;                 // 1 byte per value
    return head_dim * 2;                             // fp16
}
int64_t scale_row_bytes(int64_t head_dim) { return (head_dim / 64) * 2; }   // int8 fp16 scale per 64

// The KV arrays of a state as (pinned-host pointer, row bytes) pairs, in a fixed order.  The host copy is the
// source of truth (streamed / ring mode); the contiguous prefix covering `n_pages` whole pages is dumped.
struct KvArr { void* p = nullptr; int64_t w = 0; };

KvArr kv_host_arrays(const strata::core::QsaState& st, int64_t head_dim, int which) {
    KvArr a;
    if (st.kv_q4) {
        a.p = which == 0 ? (void*) st.host.k_q4 : (void*) st.host.v_q4;
        a.w = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) head_dim);
    } else if (st.kv_int8) {
        if (which == 0) { a.p = st.host.k_q; a.w = head_dim; }
        else if (which == 1) { a.p = st.host.v_q; a.w = head_dim; }
        else if (which == 2) { a.p = st.host.k_scale; a.w = scale_row_bytes(head_dim); }
        else { a.p = st.host.v_scale; a.w = scale_row_bytes(head_dim); }
    } else {
        a.p = which == 0 ? (void*) st.host.k_pool : (void*) st.host.v_pool;
        a.w = head_dim * 2;
    }
    return a;
}
int kv_array_count(const strata::core::QsaState& st) { return (st.kv_q4 || !st.kv_int8) ? 2 : 4; }

bool wr(FILE* f, const void* p, size_t n) { return n == 0 || std::fwrite(p, 1, n, f) == n; }

uint64_t fnv1a_up(uint64_t h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= q[i]; h *= 1099511628211ull; }
    return h;
}

// a hashing writer: everything after the header is hashed as it is written, and the digest rides at
// the end of the file - a flipped byte anywhere in the payload fails the restore's integrity check
struct HashWr {
    FILE* f = nullptr;
    uint64_t h = 1469598103934665603ull;
    bool wr(const void* p, size_t n) {
        if (n == 0) return true;
        h = fnv1a_up(h, p, n);
        return std::fwrite(p, 1, n, f) == n;
    }
};

/// 64-bit file position (long is 32-bit on Windows, where a 3.3 GiB f16 snapshot would overflow ftell)
long long ftell64(FILE* f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return ftello(f);
#endif
}

long pid() {
#ifdef _WIN32
    return (long) _getpid();
#else
    return (long) ::getpid();
#endif
}

/// seconds since the epoch (the LRU clock; file_clock would not be comparable across the two write paths)
int64_t file_mtime(const std::string& path) {
#ifdef _WIN32
    struct _stat st;
    if (::_stat(path.c_str(), &st) != 0) return 0;
#else
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
#endif
    return (int64_t) st.st_mtime;
}

}  // namespace

bool nvme_dump_at(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                  const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
                  const strata::core::ConversationCheckpoint* at, std::string& err) {
    // L is the SNAPSHOT length: for a turn-boundary snapshot the running state comes from the checkpoint's blobs
    // (the state AT L), the KV/pooled/dead arrays are truncated to L (their contents below L are untouched by
    // the generation that followed).  `block_pos` is NOT truncated and is not "per-token": it is one int32 per
    // QSA layer of device-internal scratch - the pooling kernel writes the completed block's first-cell position
    // into it (qsa.cu:216, native_qsa_indexer.cu:94) and the rotation reads it back on the device
    // (qsa.hpp:199-204).  That is why C5 was settled in favour of the CHECKPOINT's copy: at a boundary the live
    // `block_pos` names a block COMPLETED BY TOKENS PAST the boundary, so the live read would write a running-
    // state value that does not describe the prefix the file is keyed by.  `dead` is the cell-0 key (qsa.cu:187,
    // written only when pos == 0) and is constant for the sequence, so the two copies agree there - but the pair
    // is written from ONE source, which is what lets the envelope say "every running-state byte is the state at
    // L".  conversation_state_sizes sizes block_pos at sizeof(int32_t), so the segment did not move; only its
    // source did, and the version bump is what stops a v2 file being read as if it had been written at L.
    const bool at_boundary = at != nullptr;
    Sizes z;
    if (!sizes_of(g, z, err)) return false;   // never lay the envelope out with zeroed byte counts
    const int64_t L = (int64_t) ids.size();
    if (L < 1) { err = "nvme_dump: empty session"; return false; }
    if (g.n_qsa_layers() > 0 && ss.qsa_states[0].kv_mode == 0) {
        err = "nvme_dump: KV is fully resident (kv_mode 0) - run with --kv-resident (streamed) so the host copy exists";
        return false;
    }
    if (at_boundary && (at->ids.size() != ids.size() ||
                        at->gdn.size() != z.state.gdn ||
                        at->tails.size() != z.state.tail * (size_t) g.n_qsa_layers() ||
                        at->dead.size() != z.state.dead * (size_t) g.n_qsa_layers() ||
                        at->block_pos.size() != z.state.block_pos * (size_t) g.n_qsa_layers() ||
                        (!at->ple.empty() && at->ple.size() != z.state.ple))) {
        // the checkpoint is the shared core's, so its blobs are checked against the shared core's byte counts
        // rather than trusted because the caller handed over three raw pointers
        err = "nvme_dump: turn-boundary checkpoint does not fit this engine";
        return false;
    }
    // Every image record must lie INSIDE the prefix this snapshot is keyed by.  The resume match compares the next
    // request's images below `L` against this segment, so a picture at or past `L` describes a token the snapshot
    // does not hold and makes the file unmatchable; a negative one is a caller that never filtered at all.
    for (const strata::core::ConversationImageKey& im : imgs)
        if (im.start < 0 || im.start >= L) {
            err = "nvme_dump: an image record (start " + std::to_string(im.start) +
                  ") is not inside the " + std::to_string(L) + "-token prefix the snapshot is keyed by";
            return false;
        }
    FILE* f = std::fopen(path, "wb");
    if (!f) { err = std::string("nvme_dump: open ") + path; return false; }

    NvmeHeader h;
    h.L = L;
    h.n_imgs = (int64_t) imgs.size();
    h.cvec = cvec ? 1 : 0;
    h.kv_format = strata::core::qsa_kv_format(ss.qsa_states[0]);
    h.n_qsa = g.n_qsa_layers(); h.n_gdn = g.n_gdn_layers(); h.n_head_kv = g.n_head_kv; h.head_dim = g.head_dim;
    h.idx_dim = g.idx_key_dim; h.page_size = z.page_size; h.idx_block = z.idx_block;
    h.max_cells = ss.qsa_states[0].max_cells;
    if (!wr(f, &h, sizeof h)) { err = "nvme_dump: header"; std::fclose(f); return false; }
    HashWr hw{f};
    if (!hw.wr(ids.data(), ids.size() * sizeof(int32_t))) { err = "nvme_dump: ids"; std::fclose(f); return false; }
    if (!imgs.empty() && !hw.wr(imgs.data(), imgs.size() * sizeof(strata::core::ConversationImageKey))) {
        err = "nvme_dump: imgs"; std::fclose(f); return false;
    }

    // running state (device -> temp host -> file)
    std::vector<uint8_t> tmp;
    auto dump_dev = [&](const void* dptr, size_t bytes) {
        tmp.resize(bytes);
        if (bytes && cudaMemcpy(tmp.data(), dptr, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        return hw.wr(tmp.data(), bytes);
    };
    if (at_boundary) {
        if (!hw.wr(at->gdn.data(), z.state.gdn)) { err = "nvme_dump: gdn"; std::fclose(f); return false; }
    } else if (!dump_dev(ss.gdn_state, z.state.gdn)) { err = "nvme_dump: gdn"; std::fclose(f); return false; }
    if (ss.ple_hist) {
        // a session with no PLE history checkpoints no ple blob either: fall back to the live device array
        const bool have_blob = at_boundary && !at->ple.empty();
        if (!(have_blob ? hw.wr(at->ple.data(), z.state.ple) : dump_dev(ss.ple_hist, z.state.ple))) {
            err = "nvme_dump: ple"; std::fclose(f); return false;
        }
    }

    const int64_t n_pages = (L + z.page_size - 1) / z.page_size;
    int64_t pooled_rows = 0;
    if (!snapshot_pooled_rows(L, z, ss.qsa_states[0], pooled_rows, err)) {
        err = "nvme_dump: " + err;
        std::fclose(f);
        return false;
    }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        const strata::core::QsaState& st = ss.qsa_states[i];
        for (int a = 0; a < kv_array_count(st); ++a) {
            KvArr ka = kv_host_arrays(st, g.head_dim, a);
            if (!ka.p) { err = "nvme_dump: null host KV array"; std::fclose(f); return false; }
            const size_t bytes = (size_t) n_pages * (size_t) g.n_head_kv * (size_t) z.page_size * (size_t) ka.w;
            if (!hw.wr(ka.p, bytes)) { err = "nvme_dump: kv"; std::fclose(f); return false; }
        }
        if (!dump_dev(st.idx_pooled, (size_t) pooled_rows * g.idx_key_dim * 4)) { err = "nvme_dump: pooled"; std::fclose(f); return false; }
        if (at_boundary) {
            // the tail AT L: the checkpoint's per-layer tail blob (the state as of the boundary)
            if (!hw.wr(at->tails.data() + (size_t) i * z.state.tail, z.state.tail)) { err = "nvme_dump: tail"; std::fclose(f); return false; }
        } else if (!dump_dev(st.idx_tail, z.state.tail)) { err = "nvme_dump: tail"; std::fclose(f); return false; }
        // C5: the envelope owns the CHECKPOINT's copies.  With a boundary, every running-state segment above is
        // already taken from it, so `dead` / `block_pos` follow the same rule instead of reaching for the live
        // device arrays; without one, the live arrays are the state at L and are the only source there is.
        if (at_boundary) {
            if (!hw.wr(at->dead.data() + (size_t) i * z.state.dead, z.state.dead)) {
                err = "nvme_dump: dead"; std::fclose(f); return false;
            }
            if (!hw.wr(at->block_pos.data() + (size_t) i * z.state.block_pos, z.state.block_pos)) {
                err = "nvme_dump: block_pos"; std::fclose(f); return false;
            }
        } else if (!dump_dev(st.idx_dead, z.state.dead) || !dump_dev(st.idx_block_pos, z.state.block_pos)) {
            err = "nvme_dump: dead/block_pos"; std::fclose(f); return false;
        }
    }

    // MTP drafter host KV copy (ring): cells [0, min(L, max_cells)); the header records how many arrays went out
    {
        const int64_t mL = std::min<int64_t>(L, mtp_state.max_cells);
        const int64_t mp = (mL + z.page_size - 1) / z.page_size;
        int64_t wrote = 0;
        for (int a = 0; a < kv_array_count(mtp_state); ++a) {
            KvArr ka = kv_host_arrays(mtp_state, g.head_dim, a);
            if (!ka.p) continue;
            const size_t bytes = (size_t) mp * (size_t) g.n_head_kv * (size_t) z.page_size * (size_t) ka.w;
            if (!hw.wr(ka.p, bytes)) { err = "nvme_dump: mtp kv"; std::fclose(f); return false; }
            ++wrote;
        }
        // the count went into the header, which is already written: rewrite just that field
        const long off = (long) ((const uint8_t*) &h.mtp_host - (const uint8_t*) &h);
        if (std::fseek(f, off, SEEK_SET) != 0 || !wr(f, &wrote, sizeof wrote) ||
            std::fseek(f, 0, SEEK_END) != 0) { err = "nvme_dump: mtp_host"; std::fclose(f); return false; }
    }

    const uint64_t digest = hw.h;
    if (!wr(f, &digest, sizeof digest)) { err = "nvme_dump: footer"; std::fclose(f); return false; }
    std::fflush(f);
#ifndef _WIN32
    ::fsync(::fileno(f));   // crash consistency: a DONE dump survives a power cut
#else
    ::_commit(::fileno(f));
#endif
    std::fclose(f);
    return true;
}  // nvme_dump_at

bool nvme_dump(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
               const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
               const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec, std::string& err) {
    return nvme_dump_at(path, ss, mtp_state, g, ids, imgs, cvec, nullptr, err);
}

bool nvme_restore(const char* path, strata::core::SessionState& ss, strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, std::vector<int32_t>& ids,
                  std::vector<strata::core::ConversationImageKey>& imgs, bool& cvec, int64_t& L, std::string& err) {
    Sizes z;
    if (!sizes_of(g, z, err)) return false;   // the walk must not be sized with zeroed byte counts either
    FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("nvme_restore: open ") + path; return false; }
    if (std::fseek(f, 0, SEEK_END) != 0) { err = "nvme_restore: seek"; std::fclose(f); return false; }
    const long long fsize = ftell64(f);
    std::rewind(f);
    // a stray huge file (or a directory opened by mistake) must not become an allocation
    if (fsize < (long long) sizeof(NvmeHeader) || fsize > (long long) 64 << 30) {
        err = "nvme_restore: not a snapshot (size)"; std::fclose(f); return false;
    }
    // ATOMIC: the whole snapshot is read and validated before anything is applied, so a truncated file fails
    // without touching the session.
    std::vector<uint8_t> buf((size_t) fsize);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        err = "nvme_restore: read"; std::fclose(f); return false;
    }
    std::fclose(f);
    size_t at = sizeof(NvmeHeader);
    bool bad = false;
    auto take = [&](size_t n) -> const uint8_t* {
        if (bad || n > buf.size() - at) { bad = true; return nullptr; }
        const uint8_t* p = buf.data() + at;
        at += n;
        return p;
    };

    NvmeHeader h;
    std::memcpy(&h, buf.data(), sizeof h);
    if (h.magic != NvmeHeader{}.magic) { err = "nvme_restore: not a strata NVMe snapshot (bad magic)"; return false; }
    if (h.version != NvmeHeader{}.version) {
        err = "nvme_restore: snapshot is format version " + std::to_string(h.version) +
              ", this build writes version " + std::to_string(NvmeHeader{}.version) +
              " - refusing (an older snapshot is re-dumped by the engine that wrote it; nothing converts it)";
        return false;
    }
    if (h.n_qsa != g.n_qsa_layers() || h.n_gdn != g.n_gdn_layers() || h.n_head_kv != g.n_head_kv ||
        h.head_dim != g.head_dim || h.idx_dim != g.idx_key_dim || h.page_size != z.page_size ||
        h.idx_block != z.idx_block || h.kv_format != strata::core::qsa_kv_format(ss.qsa_states[0]) ||
        h.max_cells > ss.qsa_states[0].max_cells) {
        err = "nvme_restore: geometry/format mismatch (refusing to convert)"; return false;
    }
    // the counts are only trusted once they fit the file (a corrupt header must not size an allocation)
    if (h.L < 1 || (size_t) h.L * sizeof(int32_t) + sizeof(NvmeHeader) > buf.size() ||
        h.n_imgs < 0 ||
        (size_t) h.n_imgs * sizeof(strata::core::ConversationImageKey) + (size_t) h.L * sizeof(int32_t) + sizeof(NvmeHeader) > buf.size()) {
        err = "nvme_restore: malformed header sizes"; return false;
    }
    L = h.L;
    const int32_t* idp = (const int32_t*) take((size_t) L * sizeof(int32_t));
    // the imgs segment is 16-byte valued but not always 8-byte aligned (offset 104 + 4*L): memcpy, never a cast
    const void* imgp = h.n_imgs ? take((size_t) h.n_imgs * sizeof(strata::core::ConversationImageKey)) : nullptr;

    // ---- walk the rest, recording the applies; nothing is written until the walk succeeds ----
    struct Apply { void* dst; const void* src; size_t bytes; bool device; };
    std::vector<Apply> applies;
    auto seg = [&](void* dst, size_t bytes, bool device) {
        const uint8_t* p = take(bytes);
        if (p) applies.push_back({dst, p, bytes, device});
    };
    seg(ss.gdn_state, z.state.gdn, true);
    if (ss.ple_hist) seg(ss.ple_hist, z.state.ple, true);
    const int64_t n_pages = (L + z.page_size - 1) / z.page_size;
    int64_t pooled_rows = 0;
    if (!snapshot_pooled_rows(L, z, ss.qsa_states[0], pooled_rows, err)) { err = "nvme_restore: " + err; return false; }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        strata::core::QsaState& st = ss.qsa_states[i];
        for (int a = 0; a < kv_array_count(st); ++a) {
            KvArr ka = kv_host_arrays(st, g.head_dim, a);
            if (!ka.p) { err = "nvme_restore: null host KV array"; return false; }
            seg(ka.p, (size_t) n_pages * (size_t) g.n_head_kv * (size_t) z.page_size * (size_t) ka.w, false);
        }
        seg(st.idx_pooled, (size_t) pooled_rows * g.idx_key_dim * 4, true);
        seg(st.idx_tail, z.state.tail, true);
        seg(st.idx_dead, z.state.dead, true);
        seg(st.idx_block_pos, z.state.block_pos, true);
    }
    int64_t mtp_arrays = 0;
    {
        const int64_t mL = std::min<int64_t>(L, mtp_state.max_cells);
        const int64_t mp = (mL + z.page_size - 1) / z.page_size;
        for (int a = 0; a < kv_array_count(mtp_state); ++a) {
            KvArr ka = kv_host_arrays(mtp_state, g.head_dim, a);
            if (!ka.p) continue;
            seg(ka.p, (size_t) mp * (size_t) g.n_head_kv * (size_t) z.page_size * (size_t) ka.w, false);
            ++mtp_arrays;
        }
    }
    // ORDER MATTERS: decide layout first, integrity second - a layout-drifted file (an engine upgrade
    // changed a sizing formula) would otherwise misreport as "corrupt"
    if (bad) { err = "nvme_restore: truncated snapshot"; return false; }
    if (buf.size() < at + sizeof(uint64_t)) {
        err = "nvme_restore: layout mismatch (walk end " + std::to_string(at) + " past payload of a "
              + std::to_string(buf.size()) + "-byte file: idx_pooled_rows / PLE / drafter ring changed?) - refusing";
        return false;
    }
    if (at != buf.size() - sizeof(uint64_t)) {
        err = "nvme_restore: layout mismatch (walk end " + std::to_string(at) + ", file payload "
              + std::to_string(buf.size() - sizeof(uint64_t)) + ": idx_pooled_rows / PLE / drafter ring changed?) - refusing";
        return false;
    }
    if (mtp_arrays != h.mtp_host || (h.n_imgs && !imgp) || !idp) {
        err = "nvme_restore: layout mismatch (drafter arrays " + std::to_string(mtp_arrays) + " != header "
              + std::to_string(h.mtp_host) + ") - refusing";
        return false;
    }
    uint64_t digest = 0;
    std::memcpy(&digest, buf.data() + buf.size() - sizeof digest, sizeof digest);
    // the digest covers the PAYLOAD only (the header is written unhashed before the hasher exists, and its
    // geometry fields are validated field-by-field): hash [sizeof(NvmeHeader), at)
    const uint64_t expect = fnv1a_up(1469598103934665603ull, buf.data() + sizeof(NvmeHeader), at - sizeof(NvmeHeader));
    if (digest != expect) { err = "nvme_restore: integrity check failed (corrupt snapshot)"; return false; }

    // ---- everything validated: apply ----
    ids.assign(idp, idp + L);
    if (imgp) {
        imgs.resize((size_t) h.n_imgs);
        std::memcpy(imgs.data(), imgp, (size_t) h.n_imgs * sizeof(strata::core::ConversationImageKey));
    }
    cvec = h.cvec != 0;
    for (const Apply& a : applies) {
        if (a.device) {
            if (cudaMemcpy(a.dst, a.src, a.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "nvme_restore: H2D"; return false;
            }
        } else {
            std::memcpy(a.dst, a.src, a.bytes);
        }
    }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        strata::kernels::kv_stream_reset(ss.qsa_states[i].map, nullptr);   // refill slots from the host copy on demand

    // the PLE token window, oldest first (as checkpoint_restore leaves it)
    ss.ple_prev[0] = L >= 2 ? ids[(size_t) L - 2] : -1;
    ss.ple_prev[1] = L >= 1 ? ids[(size_t) L - 1] : -1;

    if (cudaDeviceSynchronize() != cudaSuccess) { err = "nvme_restore: sync"; return false; }
    return true;
}

// ================================ the store ================================

bool KvNvmeStore::open(const std::string& dir, const strata::core::ModelGeometry& g, int kv_format, std::string& err) {
    dir_ = dir;
    fmt_ = kv_format;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { err = "kv-nvme: create " + dir + ": " + ec.message(); return false; }
    const strata::kernels::QsaShapes shp = strata::kernels::qsa_real_shapes();
    size_t skipped = 0, stale = 0; uint32_t stale_version = 0;
    try {
        for (const fs::directory_entry& de : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!de.is_regular_file() || de.path().filename().string().rfind("kv-", 0) != 0) continue;
            NvmeHeader h;
            std::vector<int32_t> ids;
            std::vector<strata::core::ConversationImageKey> imgs;
            uint64_t fbytes = 0;
            try {
                std::ifstream f(de.path(), std::ios::binary);
                if (!f.read((char*) &h, sizeof h) || h.magic != NvmeHeader{}.magic) { ++skipped; continue; }
                if (h.version != NvmeHeader{}.version) {
                    // a format version this build cannot read is a DIFFERENT kind of skip from a malformed file:
                    // it is the whole store, and the operator needs the version it found to know which binary
                    // still serves it
                    ++stale; stale_version = h.version;
                    continue;
                }
                const uint64_t fb = (uint64_t) de.file_size(ec);
                // the full geometry tag (restore checks it again): another format/shape is left on disk, never converted
                if (h.kv_format != fmt_ || h.n_qsa != g.n_qsa_layers() || h.n_gdn != g.n_gdn_layers() ||
                    h.n_head_kv != g.n_head_kv || h.head_dim != g.head_dim || h.idx_dim != g.idx_key_dim ||
                    h.page_size != shp.page_size || h.idx_block != shp.idx_block ||
                    h.L < 1 || h.n_imgs < 0 ||
                    fb < sizeof(NvmeHeader) + (uint64_t) h.L * 4 +
                        (uint64_t) h.n_imgs * sizeof(strata::core::ConversationImageKey) + sizeof(uint64_t)) {
                    ++skipped; continue;   // no room for the integrity footer
                }
                ids.assign((size_t) h.L, 0);
                if (!f.read((char*) ids.data(), (size_t) h.L * sizeof(int32_t))) { ++skipped; continue; }
                // the image records follow the ids, in the order nvme_dump_at wrote them.  The resume match needs
                // them: an entry without them looks exactly like a snapshot of a conversation that had no pictures,
                // which is the one thing its image comparison can never satisfy for a session that had one.
                imgs.resize((size_t) h.n_imgs);
                if (!imgs.empty() &&
                    !f.read((char*) imgs.data(),
                            (std::streamsize) imgs.size() * sizeof(strata::core::ConversationImageKey))) {
                    ++skipped; continue;
                }
                fbytes = fb;
            } catch (...) { ++skipped; continue; }   // a corrupt store file must never take the server down
            NvmeEntry e;
            e.path = de.path().string();
            e.ids = std::move(ids);
            e.imgs = std::move(imgs);
            e.L = h.L;
            e.cvec = h.cvec != 0;
            e.bytes = fbytes;
            e.mtime = file_mtime(e.path);
            total_ += e.bytes;
            entries_.push_back(std::move(e));
        }
    } catch (const std::exception& ex) { err = std::string("kv-nvme: scan: ") + ex.what(); return false; }
    if (skipped) std::fprintf(stderr, "strata serve: kv-nvme: %zu malformed/foreign snapshot(s) skipped in %s\n", skipped, dir.c_str());
    if (stale)
        std::fprintf(stderr, "strata serve: kv-nvme: %zu snapshot(s) of format version %u in %s: this build writes "
                             "version %u and refuses older files (they stay on disk; re-dump them with the binary "
                             "that wrote them)\n", stale, stale_version, dir.c_str(), NvmeHeader{}.version);
    enforce_cap();
    return true;
}

bool KvNvmeStore::dump(const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                       const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                       const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
                       const strata::core::ConversationCheckpoint* at, std::string& err) {
    // the KEY is the matchable prefix (the turn boundary) when one is given - that is what the next request
    // replays; the full consumed state includes the model's hidden reasoning tokens, which a chat client
    // re-sending history will never reproduce.  The boundary and its running state arrive as ONE shared
    // checkpoint, so the key and the blobs cannot disagree about which point in the conversation is stored.
    const std::vector<int32_t>& key = at ? at->ids : ids;
    // THE IMAGES DESCRIBE THE SAME POINT AS THE KEY.  `imgs` is the LIVE list - every picture the session holds,
    // including the ones in the message being answered now, whose `start` is at or past the boundary.  The
    // checkpoint's own list is the filtered one (`checkpoint_at` builds it as imgs_below(req_imgs, c.ids.size())),
    // and it is the list the NEXT request will recompute with imgs_below(req_imgs, e.L).  Passing the live list
    // wrote pictures the snapshot's prefix does not contain, so an entry for any conversation that had a picture
    // could never be matched - neither by the resume loop nor by the idempotent skip below.
    const std::vector<strata::core::ConversationImageKey>& stored = at ? at->imgs : imgs;
    if (key.empty()) { err = "kv-nvme: empty session"; return false; }
    // exact match: this state is already stored - refresh its recency and skip the write (imgs too: same pad-token
    // ids with different pictures are a different session)
    for (NvmeEntry& e : entries_)
        if (e.L == (int64_t) key.size() && e.cvec == cvec && e.imgs.size() == stored.size() &&
            std::equal(key.begin(), key.end(), e.ids.begin()) && std::equal(stored.begin(), stored.end(), e.imgs.begin())) {
            e.mtime = (int64_t) ::time(nullptr);
            return true;
        }
    // Supersede: only THIS process's previous dump, and only when it is a strict prefix of the new key (the same
    // conversation grown).  A general "drop any stored prefix" would be WRONG: a branched conversation shares the
    // prefix without extending it, and its entry is the only cache its own continuations can match - so the cap,
    // not supersession, bounds cross-restart accumulation (review P2-2, deferred with this rationale).
    if (!last_ids_.empty() && last_ids_.size() < key.size() &&
        std::equal(last_ids_.begin(), last_ids_.end(), key.begin())) {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].path == last_path_) {
                std::error_code ec;
                fs::remove(last_path_, ec);
                total_ -= entries_[i].bytes;
                entries_.erase(entries_.begin() + (long) i);
                break;
            }
        last_ids_.clear();
        last_path_.clear();
    }
    char name[64];
    std::snprintf(name, sizeof name, "kv-%ld-%ld.bin", pid(), seq_++);
    const std::string path = dir_ + "/" + name;
    if (!nvme_dump_at(path.c_str(), ss, mtp_state, g, key, stored, cvec, at, err)) {
        std::error_code ec;
        fs::remove(path, ec);   // a failed dump must not leave a partial file for the next scan to admit
        return false;
    }
    std::error_code ec;
    NvmeEntry e;
    e.path = path;
    e.ids = key;
    e.L = (int64_t) key.size();
    e.cvec = cvec;
    e.imgs = stored;   // the entry must carry what the file holds, or the match loops compare against nothing
    e.bytes = (uint64_t) fs::file_size(path, ec);
    e.mtime = (int64_t) ::time(nullptr);
    total_ += e.bytes;
    last_ids_ = key;
    last_path_ = path;
    entries_.push_back(std::move(e));
    enforce_cap();
    return true;
}

bool KvNvmeStore::restore(const NvmeEntry& e, strata::core::SessionState& ss, strata::core::QsaState& mtp_state,
                          const strata::core::ModelGeometry& g, std::string& err) {
    int64_t L = 0;
    bool cvec = false;
    std::vector<int32_t> ids;
    std::vector<strata::core::ConversationImageKey> imgs;
    if (!nvme_restore(e.path.c_str(), ss, mtp_state, g, ids, imgs, cvec, L, err)) return false;
    if (L != e.L || cvec != e.cvec || imgs != e.imgs) { err = "kv-nvme: entry changed under us"; return false; }
    return true;
}

void KvNvmeStore::drop(const NvmeEntry& e) {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].path == e.path) {
            std::error_code ec;
            fs::remove(entries_[i].path, ec);
            total_ -= entries_[i].bytes;
            if (entries_[i].path == last_path_) { last_ids_.clear(); last_path_.clear(); }
            entries_.erase(entries_.begin() + (long) i);
            return;
        }
}

void KvNvmeStore::enforce_cap() {
    // The last entry is kept even over the cap (never empty the store); documented policy, see the review notes.
    while (cap_ > 0 && total_ > (uint64_t) cap_ && entries_.size() > 1) {
        size_t oldest = 0;
        for (size_t i = 1; i < entries_.size(); ++i)
            if (entries_[i].mtime < entries_[oldest].mtime) oldest = i;
        const std::string path = entries_[oldest].path;
        std::error_code ec;
        fs::remove(path, ec);
        total_ -= entries_[oldest].bytes;
        if (path == last_path_) { last_ids_.clear(); last_path_.clear(); }
        entries_.erase(entries_.begin() + (long) oldest);
    }
}

}  // namespace strata::platform
