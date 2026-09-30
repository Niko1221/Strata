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
// proves they did not (docs/nvme-kv-cache-design.md step 2).  sizeof/alignof alone would not prove it: they
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

/// THE POOLED-ROW COUNT the envelope writes (collision C4).  The FORMULA is not here: it is
/// `strata::kernels::qsa_pooled_rows`, the one the shared core's snapshot sizing and the STRATA_STATE_HASH
/// fingerprint also use, so the file's pooled segment and the fingerprint's pooled span cannot drift apart
/// (collision C8).  What this function adds is the tier's own rule: a live array too small for the snapshot
/// REFUSES, as their `conversation_kv_validate` does (`conversation_snapshot.cpp:72`); our old
/// `min(..., idx_pooled_rows)` clamp wrote a SHORT segment that the restore then reported as layout drift.
bool snapshot_pooled_rows(int64_t L, const Sizes& z, const strata::core::QsaState& st, int64_t& rows,
                          std::string& err) {
    rows = strata::kernels::qsa_pooled_rows(L, z.shapes);
    if (rows > st.idx_pooled_rows) {
        err = "pooled rows: a " + std::to_string(L) + "-token prefix needs " + std::to_string(rows) +
              " indexer pooled rows, this engine's array holds " + std::to_string(st.idx_pooled_rows) +
              " - refusing";
        return false;
    }
    return true;
}

bool sizes_of(const strata::core::ModelGeometry& g, Sizes& z, std::string& err) {
    z.shapes = strata::kernels::qsa_real_shapes();
    // The shared core can REFUSE to size a geometry (a non-positive field, an overflow in the byte product).  It
    // says so with a string naming which one, and that string used to be dropped on the floor here: a refused
    // sizing left every count at zero, and the envelope was then laid out and walked with zero-length segments.
    if (!strata::core::conversation_state_sizes(g, z.state, err)) {
        err = "kv-nvme: " + err;
        return false;
    }
    return true;
}

/// C6: THE DRAFTER'S RING IS REFILLED BY THE ADAPTER, right after it applies a snapshot - not by whichever
/// call site happened to remember.  The collapse condition is written in docs/nvme-kv-cache-design.md: this
/// is the right home while the adapter reads straight into the pinned pools, because the refill is part of the
/// residency contract for bytes it just wrote; it folds into `conversation_kv_restore` (which already calls
/// `kv_ring_restore`, conversation_snapshot.cpp:171) the moment the adapter adopts that wholesale.
///
/// The blocks are the ones the serve loop's own refill computed: [b1 - n_slots, b1) below the restored prefix.
/// Those are exactly the slots a LONGER previous turn could have clobbered, because the ring table is static -
/// `block -> block % n_slots` (kv_stream.hpp:76-79) - so with more blocks than slots, blocks past b1 wrap onto
/// the slots that belong to blocks below it.  Blocks past b1 are deliberately left alone: the drafter's attention
/// reads only cells below the one it is writing (`n_kv = pos + 1`, layer.cpp:813) and the request continues from
/// L, so every cell past L is written before anything can read it.  The adapter's drafter copy covers
/// [0, min(L, max_cells)), so the refill is clamped to the same window it restored.
void refill_drafter_ring(strata::core::QsaState& st, const strata::core::ModelGeometry& g, const Sizes& z, int64_t L) {
    if (st.kv_mode != 2 || L <= 0 || st.n_slots <= 0) return;   // only a ring needs a ring refill
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t b1 = (std::min<int64_t>(L, st.max_cells) + z.shapes.page_size - 1) / z.shapes.page_size;
    const int64_t b0 = std::max<int64_t>(0, b1 - st.n_slots);
    strata::kernels::kv_ring_restore(strata::core::qsa_attn_pools(st), st.host, strata::core::qsa_kv_format(st),
                                     b0, b1, st.n_slots, s, nullptr);
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

/// **CONSUME THE CUDA ERROR THIS FUNCTION JUST HANDLED, OR IT LIES ABOUT SOMETHING ELSE LATER.**
/// `cudaGetLastError()` returns the last error AND CLEARS it; until something reads it, the error state is
/// sticky.  This tree has already paid for that trap once: `pinned.cu:170-183` caught a `cudaHostRegister`
/// failure, handled it (the arena is simply not pinned), and the NEXT `cudaGetLastError()` in the engine -
/// `gr_read`'s launch check - then reported "out of memory" for kernels that allocate nothing.  `graph.cpp:110-118`
/// states the same rule for a capture: an error left on the stream makes EndCapture succeed with a broken graph.
///
/// On the tier's failure paths the next `cudaGetLastError()` is not some distant call: the serve loop's fallback
/// runs `session_zero` (generate.cpp:2992) -> `qsa_state_zero` (layer.cpp:657) -> `kv_stream_reset` (layer.cpp:676)
/// -> `check("reset")` (kv_stream.cu:199-202), which EXITS the process on whatever it finds.  A restore that
/// failed to copy and left its error pending therefore aborts the re-prefill with a message naming the reset.
///
/// Consuming the error does NOT make a transfer failure recoverable - a sticky context error (an illegal access,
/// a failed launch) comes straight back from the next call - it makes the REPORT true, and it is the floor below
/// which no clean reset is worth discussing.
void consume_cuda_error() { (void) cudaGetLastError(); }

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

uint64_t nvme_weights_fp(const std::vector<std::string>& model_files) {
    // path bytes || size (int64) || first 64 KiB || last 64 KiB, per file, in the order the engine loads them.
    // §5.8's formula, verbatim: this is the delta tier's old kv_delta_weights_fp body, MOVED rather than
    // rewritten, because two copies of a fingerprint are two chances to disagree about what a store is keyed by.
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

// The two helpers the delta tier reuses, defined here at namespace scope over the anonymous-namespace originals:
// thin forwarders, so the v3 walk's own code paths are untouched and the delta tier cannot grow a second copy
// of the array walk or the hash.
uint64_t nvme_fnv1a(uint64_t h, const void* p, size_t n) { return fnv1a_up(h, p, n); }

NvmeKvArr nvme_kv_host_array(const strata::core::QsaState& st, int64_t head_dim, int which) {
    const KvArr a = kv_host_arrays(st, head_dim, which);
    return {a.p, a.w};
}

int nvme_kv_array_count(const strata::core::QsaState& st) { return kv_array_count(st); }

bool nvme_dump_at(const char* path, const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                  const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
                  const strata::core::ConversationCheckpoint* at, uint64_t weights_fp, std::string& err) {
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
    if (at_boundary && !at->stage_parts.empty()) {
        // A layer-split session's later stages hold their own running state (0.1.21), and the envelope carries
        // the PRIMARY stage only: a snapshot written from a split engine could never be restored, because the
        // later stages' blobs are not in the file and the restore has nothing to put back into them.  Refuse
        // rather than grow the format for it (docs/nvme-kv-cache-design.md, the layer-split rule).
        err = "nvme_dump: a layer-split session's later stages are not snapshot-able - the envelope carries the "
              "primary stage only, so a split engine's snapshot is refused rather than written incomplete";
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
    // qsa_kv_key, not qsa_kv_format: the KEY must be able to name every format the engine can hold, and
    // qsa_kv_format refuses a hybrid state outright (a hybrid layout has no block-mover walk - see its comment).
    h.kv_format = strata::core::qsa_kv_key(ss.qsa_states[0]);
    h.weights_fp = weights_fp;   // v4: the weights these bytes WERE computed from
    h.geometry = strata::core::conversation_geometry_key(g);   // the shared core's key, not a projection of it
    h.page_size = z.shapes.page_size; h.idx_block = z.shapes.idx_block; h.max_cells = ss.qsa_states[0].max_cells;
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
        if (bytes && cudaMemcpy(tmp.data(), dptr, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            consume_cuda_error();   // the dump reports its own failure; do not leave the error for the next caller
            return false;
        }
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

    const int64_t n_pages = (L + z.shapes.page_size - 1) / z.shapes.page_size;
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
            const size_t bytes = (size_t) n_pages * (size_t) g.n_head_kv * (size_t) z.shapes.page_size * (size_t) ka.w;
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
        const int64_t mp = (mL + z.shapes.page_size - 1) / z.shapes.page_size;
        int64_t wrote = 0;
        for (int a = 0; a < kv_array_count(mtp_state); ++a) {
            KvArr ka = kv_host_arrays(mtp_state, g.head_dim, a);
            if (!ka.p) continue;
            const size_t bytes = (size_t) mp * (size_t) g.n_head_kv * (size_t) z.shapes.page_size * (size_t) ka.w;
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
               const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec, uint64_t weights_fp,
               std::string& err) {
    return nvme_dump_at(path, ss, mtp_state, g, ids, imgs, cvec, nullptr, weights_fp, err);
}
strata::core::ConversationRestore nvme_restore(const char* path, strata::core::SessionState& ss,
                                               strata::core::QsaState& mtp_state, const strata::core::ModelGeometry& g,
                                               uint64_t weights_fp, std::vector<int32_t>& ids,
                                               std::vector<strata::core::ConversationImageKey>& imgs, bool& cvec,
                                               int64_t& L, std::string& err) {
    using Restore = strata::core::ConversationRestore;
    FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("nvme_restore: open ") + path; return Restore::invalid; }
    if (std::fseek(f, 0, SEEK_END) != 0) { err = "nvme_restore: seek"; std::fclose(f); return Restore::invalid; }
    const long long fsize = ftell64(f);
    std::rewind(f);
    // a stray huge file (or a directory opened by mistake) must not become an allocation
    if (fsize < (long long) sizeof(NvmeHeader) || fsize > (long long) 64 << 30) {
        err = "nvme_restore: not a snapshot (size)"; std::fclose(f); return Restore::invalid;
    }
    // VALIDATION IS ATOMIC: the whole snapshot is read and validated before anything is applied, so a truncated
    // or corrupt file fails without touching the session.  (The APPLY pass in nvme_restore_image is not atomic -
    // it is a loop, and a CUDA failure in its middle is `transfer_failed` for that reason.)
    std::vector<uint8_t> buf((size_t) fsize);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        err = "nvme_restore: read"; std::fclose(f); return Restore::invalid;
    }
    std::fclose(f);
    return nvme_restore_image(buf.data(), buf.size(), ss, mtp_state, g, weights_fp, ids, imgs, cvec, L, err);
}

// The restore pass proper, on an IN-MEMORY image: everything after nvme_restore's whole-file read, line for
// line - the extraction exists so the delta tier (which assembles the v3 image from chunks + a State record)
// can run the EXISTING validation+apply pass unchanged instead of growing a second one.  No logic moved.
strata::core::ConversationRestore nvme_restore_image(const uint8_t* data, size_t n, strata::core::SessionState& ss,
                                                     strata::core::QsaState& mtp_state,
                                                     const strata::core::ModelGeometry& g, uint64_t weights_fp,
                                                     std::vector<int32_t>& ids,
                                                     std::vector<strata::core::ConversationImageKey>& imgs, bool& cvec,
                                                     int64_t& L, std::string& err) {
    using Restore = strata::core::ConversationRestore;
    // Every refusal below that returns `invalid` happens before the first `Apply` runs, so it provably writes
    // nothing to the session.  Every `transfer_failed` is a CUDA copy or a sync: it happens at or after the first
    // apply, and the tier cannot prove the context still answers.
    Sizes z;
    if (!sizes_of(g, z, err)) return Restore::invalid;   // the walk must not be sized with zeroed byte counts either
    // the size cap lives in the file wrapper AND here: an in-memory image has no other guard
    if (n < sizeof(NvmeHeader) + sizeof(uint64_t) || n > (size_t) 64 << 30) {
        err = "nvme_restore: not a snapshot (size)"; return Restore::invalid;
    }
    size_t at = sizeof(NvmeHeader);
    bool bad = false;
    auto take = [&](size_t count) -> const uint8_t* {
        if (bad || count > n - at) { bad = true; return nullptr; }
        const uint8_t* p = data + at;
        at += count;
        return p;
    };

    NvmeHeader h;
    std::memcpy(&h, data, sizeof h);
    if (h.magic != NvmeHeader{}.magic) { err = "nvme_restore: not a strata NVMe snapshot (bad magic)"; return Restore::invalid; }
    if (h.version != NvmeHeader{}.version) {
        err = "nvme_restore: snapshot is format version " + std::to_string(h.version) +
              ", this build writes version " + std::to_string(NvmeHeader{}.version) +
              " - refusing (an older snapshot is re-dumped by the engine that wrote it; nothing converts it)";
        return Restore::invalid;
    }
    // THE FORMAT KEY IS CHECKED FIRST, and named on its own, because "these bytes are another KV layout" and
    // "these bytes are another model" are different operator actions.  `qsa_kv_key` - never `qsa_kv_format`,
    // which REFUSES a hybrid state outright (a hybrid layout has no block-mover walk; see its comment) - is the
    // live engine's own key, so a file is never compared to the engine with a different function than the one
    // that wrote it.  k8v4 is a real value here: a store written under --kv k8v4 is keyed apart from every
    // other format rather than filed as fp16 (816 B/cell read as 1,056).
    const int live_kv_format = strata::core::qsa_kv_key(ss.qsa_states[0]);
    if (h.kv_format != live_kv_format) {
        err = "nvme_restore: KV format mismatch (snapshot written as format " + std::to_string(h.kv_format) +
              ", this engine holds format " + std::to_string(live_kv_format) +
              ") - refusing to convert; that is another layout, not a slower copy of this one";
        return Restore::invalid;
    }
    // v4: THE WEIGHT SET.  Same geometry, same prompt, same segment layout - and KV of a different network.  The
    // integrity digest cannot see this: it proves only that the file was not corrupted on the way to disk, and
    // no arithmetic over the bytes could tell one network's K/V from another's.  So the file has to SAY, and a
    // v3 file (which has no such field) is refused by version above rather than read here as "fingerprint 0" -
    // which is exactly the value a foreign store would otherwise have been accepted on.
    // This refusal is the recoverable class and provably writes nothing: it is before the first CUDA call, so the
    // caller's answer is to drop the snapshot and re-prefill.
    if (h.weights_fp != weights_fp) {
        err = "nvme_restore: weight-set mismatch (snapshot belongs to different weights) - refusing";
        return Restore::invalid;
    }
    if (h.geometry != strata::core::conversation_geometry_key(g) ||
        h.page_size != z.shapes.page_size || h.idx_block != z.shapes.idx_block ||
        h.max_cells > ss.qsa_states[0].max_cells) {
        err = "nvme_restore: geometry/format mismatch (refusing to convert)"; return Restore::invalid;
    }
    // the counts are only trusted once they fit the file (a corrupt header must not size an allocation)
    if (h.L < 1 || (size_t) h.L * sizeof(int32_t) + sizeof(NvmeHeader) > n ||
        h.n_imgs < 0 ||
        (size_t) h.n_imgs * sizeof(strata::core::ConversationImageKey) + (size_t) h.L * sizeof(int32_t) + sizeof(NvmeHeader) > n) {
        err = "nvme_restore: malformed header sizes"; return Restore::invalid;
    }
    L = h.L;
    const int32_t* idp = (const int32_t*) take((size_t) L * sizeof(int32_t));
    // the imgs segment is 16-byte valued but not always 8-byte aligned (offset 104 + 4*L): memcpy, never a cast
    const void* imgp = h.n_imgs ? take((size_t) h.n_imgs * sizeof(strata::core::ConversationImageKey)) : nullptr;

    // ---- walk the rest, recording the applies; nothing is written until the walk succeeds ----
    using Apply = NvmeRestoreApply;   // the type lives in the header: the delta tier's streaming path (B2) builds
                                      // its own list of these and hands it to nvme_restore_apply
    std::vector<Apply> applies;
    auto seg = [&](void* dst, size_t bytes, bool device, const char* what) {
        const uint8_t* p = take(bytes);
        if (p) applies.push_back({dst, p, bytes, device, what});
    };
    seg(ss.gdn_state, z.state.gdn, true, "gdn");
    if (ss.ple_hist) seg(ss.ple_hist, z.state.ple, true, "ple");
    const int64_t n_pages = (L + z.shapes.page_size - 1) / z.shapes.page_size;
    int64_t pooled_rows = 0;
    if (!snapshot_pooled_rows(L, z, ss.qsa_states[0], pooled_rows, err)) {
        err = "nvme_restore: " + err;
        return Restore::invalid;
    }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        strata::core::QsaState& st = ss.qsa_states[i];
        for (int a = 0; a < kv_array_count(st); ++a) {
            KvArr ka = kv_host_arrays(st, g.head_dim, a);
            // A null target buffer is a REFUSAL, not a transfer: their core validates the same thing before its
            // first copy (`conversation_snapshot.cpp:104-106`, "missing target state buffer").
            if (!ka.p) { err = "nvme_restore: null host KV array"; return Restore::invalid; }
            seg(ka.p, (size_t) n_pages * (size_t) g.n_head_kv * (size_t) z.shapes.page_size * (size_t) ka.w, false,
                "kv");
        }
        seg(st.idx_pooled, (size_t) pooled_rows * g.idx_key_dim * 4, true, "pooled");
        seg(st.idx_tail, z.state.tail, true, "tail");
        seg(st.idx_dead, z.state.dead, true, "dead");
        seg(st.idx_block_pos, z.state.block_pos, true, "block_pos");
    }
    int64_t mtp_arrays = 0;
    {
        const int64_t mL = std::min<int64_t>(L, mtp_state.max_cells);
        const int64_t mp = (mL + z.shapes.page_size - 1) / z.shapes.page_size;
        for (int a = 0; a < kv_array_count(mtp_state); ++a) {
            KvArr ka = kv_host_arrays(mtp_state, g.head_dim, a);
            if (!ka.p) continue;
            seg(ka.p, (size_t) mp * (size_t) g.n_head_kv * (size_t) z.shapes.page_size * (size_t) ka.w, false,
                "drafter kv");
            ++mtp_arrays;
        }
    }
    // ORDER MATTERS: decide layout first, integrity second - a layout-drifted file (an engine upgrade
    // changed a sizing formula) would otherwise misreport as "corrupt"
    if (bad) { err = "nvme_restore: truncated snapshot"; return Restore::invalid; }
    if (n < at + sizeof(uint64_t)) {
        err = "nvme_restore: layout mismatch (walk end " + std::to_string(at) + " past payload of a "
              + std::to_string(n) + "-byte file: idx_pooled_rows / PLE / drafter ring changed?) - refusing";
        return Restore::invalid;
    }
    if (at != n - sizeof(uint64_t)) {
        err = "nvme_restore: layout mismatch (walk end " + std::to_string(at) + ", file payload "
              + std::to_string(n - sizeof(uint64_t)) + ": idx_pooled_rows / PLE / drafter ring changed?) - refusing";
        return Restore::invalid;
    }
    if (mtp_arrays != h.mtp_host || (h.n_imgs && !imgp) || !idp) {
        err = "nvme_restore: layout mismatch (drafter arrays " + std::to_string(mtp_arrays) + " != header "
              + std::to_string(h.mtp_host) + ") - refusing";
        return Restore::invalid;
    }
    uint64_t digest = 0;
    std::memcpy(&digest, data + n - sizeof digest, sizeof digest);
    // the digest covers the PAYLOAD only (the header is written unhashed before the hasher exists, and its
    // geometry fields are validated field-by-field): hash [sizeof(NvmeHeader), at)
    const uint64_t expect = fnv1a_up(1469598103934665603ull, data + sizeof(NvmeHeader), at - sizeof(NvmeHeader));
    if (digest != expect) { err = "nvme_restore: integrity check failed (corrupt snapshot)"; return Restore::invalid; }

    // everything validated: the apply pass is `nvme_restore_apply` below - extracted line-for-line from what
    // used to be this function's tail (the restore-perf handoff's B1), and called with the walk's own applies.
    // Nothing of the choreography moved: it reads the same bytes and makes the same decisions in the same order.
    return nvme_restore_apply(applies, ss, mtp_state, g, z, idp, L, imgp, h.n_imgs, h.cvec != 0, ids, imgs, cvec,
                              err);
}

// The apply pass itself, on an ALREADY-VALIDATED applies list (R1: the v3 restore path's bytes and decisions
// are frozen - this is the same code, reached through one more function boundary; the delta tier's streaming
// restore (B2) reaches the same code with its own list).
strata::core::ConversationRestore nvme_restore_apply(const std::vector<NvmeRestoreApply>& applies,
                                                     strata::core::SessionState& ss,
                                                     strata::core::QsaState& mtp_state,
                                                     const strata::core::ModelGeometry& g, const Sizes& z,
                                                     const int32_t* idp, int64_t L, const void* imgp,
                                                     int64_t n_imgs, bool cvec_flag, std::vector<int32_t>& ids,
                                                     std::vector<strata::core::ConversationImageKey>& imgs,
                                                     bool& cvec, std::string& err) {
    using Restore = strata::core::ConversationRestore;
    using Apply = NvmeRestoreApply;   // the name the moved body was written with
    // ---- everything validated: apply.  THE APPLY PASS BEGINS WITH A SYNC, exactly as their
    // `conversation_snapshot_restore` does (`conversation_state.cpp:258`, and their fixture asserts a failure here
    // mutates nothing: `conversation_validation_test.cpp:158-161`).  It proves the device answers BEFORE a single
    // byte is written, so a context that is already unusable fails as `transfer_failed` with the session
    // untouched, rather than half-applying and failing later.
    //
    // TEST-ONLY FAULT INJECTION (docs/nvme-kv-cache-design.md §5.5): STRATA_TEST_FAIL_CUDA names one thing to
    // break - the pre-apply synchronize ("sync"), the spare-row re-publish ("spare"), or one apply-pass segment
    // by name ("gdn", "ple", "pooled", "tail", "dead", "block_pos").  The named transfer is SKIPPED and its
    // failure branch taken, so the fatal transfer_failed path can be EXECUTED on a healthy device, which no host
    // fixture can arrange (their copies are memcpy).  Copies before the named one have already been applied, so
    // the session is exactly the half-applied state the contract says must not be recovered from.  Unset - the
    // default, and the only shipped configuration - the hook does nothing at all.
    const char* fail_env = std::getenv("STRATA_TEST_FAIL_CUDA");
    const std::string fail_seg = fail_env ? fail_env : "";
    if (fail_seg == "sync" || cudaDeviceSynchronize() != cudaSuccess) {
        err = "nvme_restore: device synchronize before the apply pass";
        if (fail_seg != "sync") consume_cuda_error();
        return Restore::transfer_failed;
    }

    ids.assign(idp, idp + L);
    if (imgp) {
        imgs.resize((size_t) n_imgs);
        std::memcpy(imgs.data(), imgp, (size_t) n_imgs * sizeof(strata::core::ConversationImageKey));
    }
    cvec = cvec_flag;
    for (const Apply& a : applies) {
        if (a.device) {
            if (fail_seg == a.what) {   // test-only: skip the named transfer, take its failure branch
                err = std::string("nvme_restore: host-to-device transfer failed for the ") + a.what +
                      " segment (" + std::to_string(a.bytes) + " bytes) [STRATA_TEST_FAIL_CUDA]";
                return Restore::transfer_failed;
            }
            if (cudaMemcpy(a.dst, a.src, a.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = std::string("nvme_restore: host-to-device transfer failed for the ") + a.what +
                      " segment (" + std::to_string(a.bytes) + " bytes)";
                consume_cuda_error();
                return Restore::transfer_failed;
            }
        } else {
            std::memcpy(a.dst, a.src, a.bytes);
        }
    }
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        strata::kernels::kv_stream_reset(ss.qsa_states[i].map, nullptr);   // refill slots from the host copy on demand
        // C3: re-publish the SPARE pooled row, exactly as their `conversation_checkpoint_restore` does
        // (`conversation_state.cpp:186`).  The row at `L / idx_block` is the spare for the block IN PROGRESS,
        // and a turn-boundary snapshot's copy of it is stale by construction: tokens past the boundary completed
        // that block and the pooling kernel rewrote the row with the completed block's key (qsa.cu:211) before the
        // dump read it.  At resume `n_bid = n_kv / idx_block = L / idx_block` (qsa.hpp:156, layer.cpp:813), and
        // `qsa_index_kernel` scores row `b == n_bid` straight out of the pool (qsa.cu:253, 260) as does the native
        // scorer (native_qsa_score.cu:74 `row <= full`) - so the masking the `b == n_bid` path in qsa_select.cu:35
        // does (it reads `dead` instead) is NOT enough: the invariant `pooled[n_bid] == dead`, which the writers
        // maintain at every block completion (qsa.cu:213, native_qsa_indexer.cu:93), has to be restored too.
        // For a full-L dump the row already equals `dead`, so this is a no-op there.
        const int64_t row = L / z.shapes.idx_block;   // < idx_pooled_rows: snapshot_pooled_rows checked it
        if (fail_seg == "spare" ||
            cudaMemcpy(ss.qsa_states[i].idx_pooled + (size_t) row * g.idx_key_dim,
                       ss.qsa_states[i].idx_dead, z.state.dead, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = std::string("nvme_restore: host-to-device transfer failed while re-publishing the spare pooled row") +
                  (fail_seg == "spare" ? " [STRATA_TEST_FAIL_CUDA]" : "");
            if (fail_seg != "spare") consume_cuda_error();
            return Restore::transfer_failed;
        }
    }
    refill_drafter_ring(mtp_state, g, z, L);   // C6: the drafter's ring, by the tier that just wrote its host copy

    // the PLE token window, oldest first (as checkpoint_restore leaves it)
    ss.ple_prev[0] = L >= 2 ? ids[(size_t) L - 2] : -1;
    ss.ple_prev[1] = L >= 1 ? ids[(size_t) L - 1] : -1;

    // THE PROOF A RECOVERY NEEDS.  Everything above is written; this sync is what shows the device took it.  A
    // failure here is still `transfer_failed` - the writes happened and nothing shows they landed - while a
    // SUCCESS here is the one piece of evidence that lets a caller reset the session and re-read the prompt
    // (docs/nvme-kv-cache-design.md §5): the context answered after the last write.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = "nvme_restore: device synchronize after the apply pass";
        consume_cuda_error();
        return Restore::transfer_failed;
    }
    return Restore::restored;
}

// ================================ the store ================================

bool KvNvmeStore::open(const std::string& dir, const strata::core::ModelGeometry& g, int kv_format,
                       const std::vector<std::string>& model_files, std::string& err) {
    dir_ = dir;
    fmt_ = kv_format;
    weights_fp_ = nvme_weights_fp(model_files);   // once per process, like the delta tier's own open
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { err = "kv-nvme: create " + dir + ": " + ec.message(); return false; }
    const strata::kernels::QsaShapes shp = strata::kernels::qsa_real_shapes();
    size_t skipped = 0, stale = 0, foreign_weights = 0; uint32_t stale_version = 0;
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
                // THE WEIGHT SET, AT SCAN TIME.  A same-geometry snapshot of another model's conversation is
                // not a candidate: saying so here means it never becomes an entry, never reaches the match, and
                // never holds a byte against this store's cap - the three ways a foreign file could otherwise
                // keep the tier looking alive while serving nothing from it.
                if (h.weights_fp != weights_fp_) { ++foreign_weights; continue; }
                const uint64_t fb = (uint64_t) de.file_size(ec);
                // the full geometry key (restore checks it again): another format/shape is left on disk, never converted
                if (h.kv_format != fmt_ || h.geometry != strata::core::conversation_geometry_key(g) ||
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
    // ITS OWN LINE, because this is the case none of the others explains: the files pass every other check and
    // are still not promoted, and the only reason is that they hold another network's KV.  Sharing one store
    // directory across model swaps is a normal thing to do, so a silent skip here reads as a broken tier.
    if (foreign_weights)
        std::fprintf(stderr, "strata serve: kv-nvme: %zu snapshot(s) in %s belong to a DIFFERENT weight set (same "
                             "geometry, different weights) and are never promoted - every such request re-prefills. "
                             "They stay on disk for the build that wrote them; delete them if the model has "
                             "changed for good\n",
                     foreign_weights, dir.c_str());
    if (stale)
        std::fprintf(stderr, "strata serve: kv-nvme: %zu snapshot(s) of format version %u in %s: this build writes "
                             "version %u and refuses older files. They stay on disk and are skipped, so nothing can "
                             "be promoted from them and every request re-prefills - to keep them working, re-dump "
                             "them with the binary that wrote them; to stop the skip, remove them and let the store "
                             "rebuild\n",
                     stale, stale_version, dir.c_str(), NvmeHeader{}.version);
    enforce_cap();
    return true;
}

bool KvNvmeStore::dump(const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                       const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                       const std::vector<strata::core::ConversationImageKey>& imgs, bool cvec,
                       const strata::core::ConversationCheckpoint* at, std::string& err, TierActivity* act) {
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
            if (act) act->skipped = true;   // the store's bytes are unchanged: this call wrote nothing
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
                if (act) { ++act->dropped; act->dropped_bytes += entries_[i].bytes; }
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
    if (!nvme_dump_at(path.c_str(), ss, mtp_state, g, key, stored, cvec, at, weights_fp_, err)) {
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
    if (act) act->written = e.bytes;   // the snapshot this call wrote, as the file on disk sizes it
    total_ += e.bytes;
    last_ids_ = key;
    last_path_ = path;
    entries_.push_back(std::move(e));
    const TierActivity cap = enforce_cap();
    if (act) { act->evicted += cap.evicted; act->evicted_bytes += cap.evicted_bytes; }
    return true;
}

strata::core::ConversationRestore KvNvmeStore::restore(const NvmeEntry& e, strata::core::SessionState& ss,
                                              strata::core::QsaState& mtp_state,
                                              const strata::core::ModelGeometry& g, std::string& err) {
    int64_t L = 0;
    bool cvec = false;
    std::vector<int32_t> ids;
    std::vector<strata::core::ConversationImageKey> imgs;
    const strata::core::ConversationRestore r =
        nvme_restore(e.path.c_str(), ss, mtp_state, g, weights_fp_, ids, imgs, cvec, L, err);
    if (r != strata::core::ConversationRestore::restored) return r;
    if (L != e.L || cvec != e.cvec || imgs != e.imgs) {
        // The file disagreed with the index the scan built - a TOCTOU on the store, not on the device.  It is the
        // RECOVERABLE class on the tier's own evidence: `nvme_restore` applied every segment and its final
        // `cudaDeviceSynchronize()` succeeded, so the context answered after the last write, and the session holds
        // a complete, digest-verified snapshot rather than a half-applied one.  The caller drops the entry and
        // re-reads the prompt; that is the clean reset the contract permits, and it is permitted precisely because
        // this path has the proof a clean reset needs.
        err = "kv-nvme: entry changed under us";
        return strata::core::ConversationRestore::invalid;
    }
    return strata::core::ConversationRestore::restored;
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

TierActivity KvNvmeStore::enforce_cap() {
    TierActivity act;
    // The last entry is kept even over the cap (never empty the store); documented policy, see the review notes.
    while (cap_ > 0 && total_ > (uint64_t) cap_ && entries_.size() > 1) {
        size_t oldest = 0;
        for (size_t i = 1; i < entries_.size(); ++i)
            if (entries_[i].mtime < entries_[oldest].mtime) oldest = i;
        const std::string path = entries_[oldest].path;
        std::error_code ec;
        fs::remove(path, ec);
        ++act.evicted;
        act.evicted_bytes += entries_[oldest].bytes;
        total_ -= entries_[oldest].bytes;
        if (path == last_path_) { last_ids_.clear(); last_path_.clear(); }
        entries_.erase(entries_.begin() + (long) oldest);
    }
    return act;
}

}  // namespace strata::platform
