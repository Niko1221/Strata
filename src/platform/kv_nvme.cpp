// src/platform/kv_nvme.cpp - see include/strata/platform/kv_nvme.hpp.
#include "strata/platform/kv_nvme.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>

#include <filesystem>
#include <sys/stat.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#endif

#include "strata/kernels/kv_stream.hpp"   // KvFormat, kv_stream_reset
#include "strata/kernels/kv_q4.hpp"       // kv_q4_bytes_per_head
#include "strata/kernels/ngram.hpp"       // NG_HIST, NG_HC_DIM
#include "strata/kernels/qsa.hpp"         // qsa_real_shapes

namespace strata::platform {

namespace {

namespace fs = std::filesystem;

struct Sizes {
    int64_t page_size = 0, idx_block = 0;
    size_t gdn = 0, ple = 0, tail = 0, dead = 0;   // bytes per the relevant scope
};

Sizes sizes_of(const strata::core::ModelGeometry& g) {
    const strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    Sizes z;
    z.page_size = s.page_size;
    z.idx_block = s.idx_block;
    z.gdn = (size_t) g.n_gdn_layers() *
            ((size_t) g.ssm_state_size * (size_t) g.ssm_v_heads * (size_t) g.ssm_state_size +
             (size_t) g.ssm_conv_channels * (size_t) (g.ssm_d_conv - 1)) *
            sizeof(float);
    z.ple = (size_t) strata::kernels::NG_HIST * (size_t) strata::kernels::NG_HC_DIM * sizeof(float);
    z.tail = (size_t) (s.idx_block - 1) * (size_t) g.idx_key_dim * sizeof(float);
    z.dead = (size_t) g.idx_key_dim * sizeof(float);
    return z;
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
                  const std::vector<std::pair<int64_t, uint64_t>>& imgs, bool cvec, NvmeRunning running,
                  std::string& err) {
    // L is the SNAPSHOT length: for a turn-boundary snapshot the running state comes from the caller's blobs
    // (the state AT L), the KV/pooled/dead arrays are truncated to L (their contents below L are untouched by
    // the generation that followed), and the per-token scratch (block_pos) rides along harmlessly.
    const bool at_boundary = running.gdn != nullptr;
    const Sizes z = sizes_of(g);
    const int64_t L = (int64_t) ids.size();
    if (L < 1) { err = "nvme_dump: empty session"; return false; }
    if (g.n_qsa_layers() > 0 && ss.qsa_states[0].kv_mode == 0) {
        err = "nvme_dump: KV is fully resident (kv_mode 0) - run with --kv-resident (streamed) so the host copy exists";
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
    if (!imgs.empty() && !hw.wr(imgs.data(), imgs.size() * sizeof(std::pair<int64_t, uint64_t>))) {
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
        if (!hw.wr(running.gdn, z.gdn)) { err = "nvme_dump: gdn"; std::fclose(f); return false; }
    } else if (!dump_dev(ss.gdn_state, z.gdn)) { err = "nvme_dump: gdn"; std::fclose(f); return false; }
    if (ss.ple_hist) {
        const bool have_blob = running.ple != nullptr;
        if (!(have_blob ? hw.wr(running.ple, z.ple) : dump_dev(ss.ple_hist, z.ple))) {
            err = "nvme_dump: ple"; std::fclose(f); return false;
        }
    }

    const int64_t n_pages = (L + z.page_size - 1) / z.page_size;
    const int64_t pooled_rows = std::min<int64_t>(L / z.idx_block + 2, ss.qsa_states[0].idx_pooled_rows);
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
            if (!hw.wr(running.tails + (size_t) i * z.tail, z.tail)) { err = "nvme_dump: tail"; std::fclose(f); return false; }
        } else if (!dump_dev(st.idx_tail, z.tail)) { err = "nvme_dump: tail"; std::fclose(f); return false; }
        if (!dump_dev(st.idx_dead, z.dead)) { err = "nvme_dump: dead"; std::fclose(f); return false; }
        if (!dump_dev(st.idx_block_pos, 4)) { err = "nvme_dump: block_pos"; std::fclose(f); return false; }
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
               const std::vector<std::pair<int64_t, uint64_t>>& imgs, bool cvec, std::string& err) {
    return nvme_dump_at(path, ss, mtp_state, g, ids, imgs, cvec, NvmeRunning{}, err);
}

bool nvme_restore(const char* path, strata::core::SessionState& ss, strata::core::QsaState& mtp_state,
                  const strata::core::ModelGeometry& g, std::vector<int32_t>& ids,
                  std::vector<std::pair<int64_t, uint64_t>>& imgs, bool& cvec, int64_t& L, std::string& err) {
    const Sizes z = sizes_of(g);
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
    if (h.magic != NvmeHeader{}.magic || h.version != 2) { err = "nvme_restore: bad header"; return false; }
    if (h.n_qsa != g.n_qsa_layers() || h.n_gdn != g.n_gdn_layers() || h.n_head_kv != g.n_head_kv ||
        h.head_dim != g.head_dim || h.idx_dim != g.idx_key_dim || h.page_size != z.page_size ||
        h.idx_block != z.idx_block || h.kv_format != strata::core::qsa_kv_format(ss.qsa_states[0]) ||
        h.max_cells > ss.qsa_states[0].max_cells) {
        err = "nvme_restore: geometry/format mismatch (refusing to convert)"; return false;
    }
    // the counts are only trusted once they fit the file (a corrupt header must not size an allocation)
    if (h.L < 1 || (size_t) h.L * sizeof(int32_t) + sizeof(NvmeHeader) > buf.size() ||
        h.n_imgs < 0 ||
        (size_t) h.n_imgs * sizeof(std::pair<int64_t, uint64_t>) + (size_t) h.L * sizeof(int32_t) + sizeof(NvmeHeader) > buf.size()) {
        err = "nvme_restore: malformed header sizes"; return false;
    }
    L = h.L;
    const int32_t* idp = (const int32_t*) take((size_t) L * sizeof(int32_t));
    // the imgs segment is 8-byte valued but not always 8-byte aligned (offset 104 + 4*L): memcpy, never a cast
    const void* imgp = h.n_imgs ? take((size_t) h.n_imgs * sizeof(std::pair<int64_t, uint64_t>)) : nullptr;

    // ---- walk the rest, recording the applies; nothing is written until the walk succeeds ----
    struct Apply { void* dst; const void* src; size_t bytes; bool device; };
    std::vector<Apply> applies;
    auto seg = [&](void* dst, size_t bytes, bool device) {
        const uint8_t* p = take(bytes);
        if (p) applies.push_back({dst, p, bytes, device});
    };
    seg(ss.gdn_state, z.gdn, true);
    if (ss.ple_hist) seg(ss.ple_hist, z.ple, true);
    const int64_t n_pages = (L + z.page_size - 1) / z.page_size;
    const int64_t pooled_rows = std::min<int64_t>(L / z.idx_block + 2, ss.qsa_states[0].idx_pooled_rows);
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) {
        strata::core::QsaState& st = ss.qsa_states[i];
        for (int a = 0; a < kv_array_count(st); ++a) {
            KvArr ka = kv_host_arrays(st, g.head_dim, a);
            if (!ka.p) { err = "nvme_restore: null host KV array"; return false; }
            seg(ka.p, (size_t) n_pages * (size_t) g.n_head_kv * (size_t) z.page_size * (size_t) ka.w, false);
        }
        seg(st.idx_pooled, (size_t) pooled_rows * g.idx_key_dim * 4, true);
        seg(st.idx_tail, z.tail, true);
        seg(st.idx_dead, z.dead, true);
        seg(st.idx_block_pos, 4, true);
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
    if (bad) { err = "nvme_restore: truncated snapshot"; return false; }
    if (buf.size() < at + sizeof(uint64_t)) { err = "nvme_restore: no integrity footer"; return false; }
    uint64_t digest = 0;
    std::memcpy(&digest, buf.data() + buf.size() - sizeof digest, sizeof digest);
    // the digest covers the PAYLOAD only (the header is written unhashed before the hasher exists, and its
    // geometry fields are validated field-by-field): hash [sizeof(NvmeHeader), at)
    const uint64_t expect = fnv1a_up(1469598103934665603ull, buf.data() + sizeof(NvmeHeader), at - sizeof(NvmeHeader));
    if (digest != expect) { err = "nvme_restore: integrity check failed (corrupt snapshot)"; return false; }
    // the file is complete but its layout differs: idx_pooled_rows, PLE presence or the drafter's ring size
    if (at != buf.size() - sizeof(uint64_t))
        err = "nvme_restore: layout mismatch (idx_pooled_rows / PLE / drafter ring?) - refusing";
    if (at != buf.size() - sizeof(uint64_t) || mtp_arrays != h.mtp_host || (h.n_imgs && !imgp) || !idp) return false;

    // ---- everything validated: apply ----
    ids.assign(idp, idp + L);
    if (imgp) {
        imgs.resize((size_t) h.n_imgs);
        std::memcpy(imgs.data(), imgp, (size_t) h.n_imgs * sizeof(std::pair<int64_t, uint64_t>));
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
    size_t skipped = 0;
    try {
        for (const fs::directory_entry& de : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!de.is_regular_file() || de.path().filename().string().rfind("kv-", 0) != 0) continue;
            NvmeHeader h;
            std::vector<int32_t> ids;
            uint64_t fbytes = 0;
            try {
                std::ifstream f(de.path(), std::ios::binary);
                if (!f.read((char*) &h, sizeof h) || h.magic != NvmeHeader{}.magic || h.version != 2) { ++skipped; continue; }
                const uint64_t fb = (uint64_t) de.file_size(ec);
                // the full geometry tag (restore checks it again): another format/shape is left on disk, never converted
                if (h.kv_format != fmt_ || h.n_qsa != g.n_qsa_layers() || h.n_gdn != g.n_gdn_layers() ||
                    h.n_head_kv != g.n_head_kv || h.head_dim != g.head_dim || h.idx_dim != g.idx_key_dim ||
                    h.page_size != shp.page_size || h.idx_block != shp.idx_block ||
                    h.L < 1 || h.n_imgs < 0 ||
                    fb < sizeof(NvmeHeader) + (uint64_t) h.L * 4 + (uint64_t) h.n_imgs * 16
                        + sizeof(uint64_t)) { ++skipped; continue; }   // no room for the integrity footer
                ids.assign((size_t) h.L, 0);
                if (!f.read((char*) ids.data(), (size_t) h.L * sizeof(int32_t))) { ++skipped; continue; }
                fbytes = fb;
            } catch (...) { ++skipped; continue; }   // a corrupt store file must never take the server down
            NvmeEntry e;
            e.path = de.path().string();
            e.ids = std::move(ids);
            e.L = h.L;
            e.cvec = h.cvec != 0;
            e.bytes = fbytes;
            e.mtime = file_mtime(e.path);
            total_ += e.bytes;
            entries_.push_back(std::move(e));
        }
    } catch (const std::exception& ex) { err = std::string("kv-nvme: scan: ") + ex.what(); return false; }
    if (skipped) std::fprintf(stderr, "strata serve: kv-nvme: %zu malformed/foreign snapshot(s) skipped in %s\n", skipped, dir.c_str());
    enforce_cap();
    return true;
}

bool KvNvmeStore::dump(const strata::core::SessionState& ss, const strata::core::QsaState& mtp_state,
                       const strata::core::ModelGeometry& g, const std::vector<int32_t>& ids,
                       const std::vector<std::pair<int64_t, uint64_t>>& imgs, bool cvec,
                       const std::vector<int32_t>* at_ids, NvmeRunning running, std::string& err) {
    // the KEY is the matchable prefix (the turn boundary) when one is given - that is what the next request
    // replays; the full consumed state includes the model's hidden reasoning tokens, which a chat client
    // re-sending history will never reproduce
    const std::vector<int32_t>& key = at_ids ? *at_ids : ids;
    if (key.empty()) { err = "kv-nvme: empty session"; return false; }
    // exact match: this state is already stored - refresh its recency and skip the write (imgs too: same pad-token
    // ids with different pictures are a different session)
    for (NvmeEntry& e : entries_)
        if (e.L == (int64_t) key.size() && e.cvec == cvec && e.imgs.size() == imgs.size() &&
            std::equal(key.begin(), key.end(), e.ids.begin()) && std::equal(imgs.begin(), imgs.end(), e.imgs.begin())) {
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
    if (!nvme_dump_at(path.c_str(), ss, mtp_state, g, key, imgs, cvec, running, err)) {
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
    std::vector<std::pair<int64_t, uint64_t>> imgs;
    if (!nvme_restore(e.path.c_str(), ss, mtp_state, g, ids, imgs, cvec, L, err)) return false;
    if (L != e.L || cvec != e.cvec) { err = "kv-nvme: entry changed under us"; return false; }
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
