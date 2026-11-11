// =============================================================================================
// [giphoo] l2-exp: conversation_file_v2.inc ? the L2 block-format toolkit (v2.1/v2.2/v2.3/M3.1)
//
// Included once by conversation_file.cpp where the original writer namespace ended; it reaches
// the host's anonymous-namespace plumbing (Out/In walkers, Publisher, RawFile, sat arithmetic)
// and keeps EVERY v2/l2 change in this one module.  The host file's upstream diff is minimal:
// three template signatures, this include, two dispatchers.  Design: docs/l2-blockformat-v2-design.md.
//
// Contents (all [giphoo] l2-exp):
//   format v2.1   block-compressed writer/reader (per-field segments, lz4 blocks, directory)
//   SAVEINC v2.2  the incremental save (checkpoint/field-segment reuse, in-place append, peek)
//   parked  v2.3  the streamed restore whose physical gate sees only transients (M3)
//   RESTOREPP     partial restore by checkpoint - reads proportional to the borrow (M3.1)
//   public entry  session_file_saveinc / saveinc_peek / read_parked / read_prefix
// =============================================================================================
// [giphoo] l2-exp v2: the block-compressed session format (docs/l2-blockformat-v2-design.md) needs lz4.
// Without the header the v2 code compiles out: writes stay v1 and v2 files are refused with a clear error.
#if __has_include(<lz4.h>)
#define STRATA_SESSION_V2_FORMAT 1
#include <lz4.h>
#endif

#ifdef STRATA_SESSION_V2_FORMAT
#include <atomic>
// =============================================================================================
// [giphoo] l2-exp v2.1: the block-compressed session format (design: docs/l2-blockformat-v2-design.md).
//
//   header 64 B: magic 'STRSESS\x02', version 2, header size 64, model and config fingerprints,
//                payload = the RAW (uncompressed) logical payload bytes, reserved 0, header hash.
//   body:        segment streams.  A segment is a run of blocks; a block is
//                [usize:u32][csize:u32][mode:u32] + csize bytes (mode 0 = lz4, 1 = stored).
//                Segment order is fixed:
//                  seg#0  kind 0  counts   - every scalar and size the parser needs, nothing else
//                  seg#1  kind 4  live     - live state fields as raw bytes (ids, imgs, gdn, ...)
//                  seg#2. kind 5  ckpt #i  - one segment per kept checkpoint, keep order
//                  then   kind 1  kv part  - one per (layer, part), layer-major part-minor
//                A segment's block list is an explicit index table in the directory, so an
//                incremental save (SAVEINC) can extend a segment with new blocks while its old
//                blocks stay where they are.  Checkpoint and live segments carry their ids hash
//                in the directory: SAVEINC matches kept checkpoints by that hash and reuses them.
//   directory:   layout=2, magic 'S2DIR', payload_raw, segment records {kind, layer, aux,
//                raw_total, ids_hash, n_blocks, block indexes...}, block records {offset, csize,
//                usize, mode, raw hash}, and a hash of the directory itself.
//   footer 16 B: magic 'STRSESS\x02' + dir_off, found from the end of the file.
//
//   Reads replay the logical model from the segments with every v1 limit checked in `counts`,
//   before any payload allocation: geometry, layer range, token counts, state and K/V part sizes.
//   Every block's raw bytes are hashed at write time and re-verified at decompression.
#include "conversation_file_l2.inc"   // [giphoo] L2 2.0: ???????? (hooks: strata::core::l2two::hook_*; ??? include, ?? cpp ???)

namespace {
constexpr char kMagicV2[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x02'};
constexpr char kDirMagic[8] = {'S', '2', 'D', 'I', 'R', '\0', '\0', '\x02'};
constexpr uint16_t kV2Layout = 2;
constexpr uint64_t kV2BlkDefault = 1ull << 20, kV2BlkMin = 64ull << 10, kV2BlkMax = 16ull << 20;

uint64_t v2_block_target() {
    const char* e = std::getenv("STRATA_SESSION_BLK");
    if (e && *e) {
        const long long v = std::strtoll(e, nullptr, 10);
        if (v > 0) return std::min<uint64_t>(std::max<uint64_t>((uint64_t) v, kV2BlkMin), kV2BlkMax);
    }
    return kV2BlkDefault;
}

bool session_v2_writes() {
    const char* e = std::getenv("STRATA_SESSION_V2");
    return e && *e && *e != '0';   // opt-in until M4 flips production; the reader takes both versions
}

// [giphoo] l2-exp v2.4: parallel block readahead for the restore path.  The blocks of a segment are
// independent (own offset/size/hash in the directory), so K workers can pread+decompress+verify
// ahead of the consumer.  0 = the serial single-pread path; the raidz1 array of NVMe devices
// delivers far more than one reader can pull through pread+lz4+hash in line.
int v2_readahead_depth() {
    static const int d = [] {
        const char* e = std::getenv("STRATA_V2_READAHEAD");
        if (e && *e) {
            const long v = std::strtol(e, nullptr, 10);
            if (v >= 0 && v <= 32) return (int) v;
        }
        return 0;
    }();
    return d;
}

template<class V> size_t v2_esz(const V&) { return sizeof(typename V::value_type); }
uint64_t v2_ids_hash(const std::vector<int32_t>& ids) {
    return ids.empty() ? 0 : session_hash64(ids.data(), ids.size() * sizeof(int32_t), 0);
}

struct V2BlockRec { uint64_t offset = 0; uint32_t csize = 0, usize = 0; uint32_t mode = 0; uint64_t hash = 0; };
struct V2SegRec {
    uint8_t kind = 0;        // 0 counts, 1 K/V part, 4 live, 5 checkpoint
    uint16_t layer = 0;      // K/V layer (kind 1)
    uint32_t aux = 0;        // part index (kind 1) or keep index (kind 5)
    uint64_t raw_total = 0, ids_hash = 0;
    std::vector<uint32_t> idx;   // explicit block index table (sparse-friendly)
};

size_t v2_img_part_size(const ConversationKv& k, size_t p) {
    switch (p) {
        case 0: return k.k.size();
        case 1: return k.v.size();
        case 2: return k.k_scale.size();
        case 3: return k.v_scale.size();
        default: return k.pooled.size();
    }
}
ConversationBuffer* v2_img_part(ConversationKv& k, size_t p) {
    switch (p) {
        case 0: return &k.k;
        case 1: return &k.v;
        case 2: return &k.k_scale;
        case 3: return &k.v_scale;
        default: return &k.pooled;
    }
}

// ---- file plumbing: the same segments read and write through a fresh temp file (full save) or an
// ---- existing file opened read/write (the incremental save appends in place)
struct V2Io {
    virtual bool read_at(uint64_t off, void* p, size_t n) = 0;
    virtual bool write_all(const void* p, size_t n) = 0;
    virtual bool sync() = 0;
    virtual std::string error() const = 0;
    virtual int pread_fd() const { return -1; }   // [giphoo] v2.4: >= 0 lets workers pread without shared state
    virtual int block_fd(uint64_t offset, uint64_t& local_offset) {
        local_offset = offset;
        return pread_fd();
    }
    virtual ~V2Io() = default;
};
struct V2FreshIo : V2Io {
    explicit V2FreshIo(RawFile& f) : f_(f) {}
    bool read_at(uint64_t off, void* p, size_t n) override { return f_.read_at(off, (uint8_t*) p, n); }
    bool write_all(const void* p, size_t n) override { return f_.write_all((const uint8_t*) p, n); }
    bool sync() override { return f_.sync(); }
    std::string error() const override { return f_.error(); }
    int pread_fd() const override { return f_.pread_fd(); }
private:
    RawFile& f_;
};
class V2RW : public V2Io {
public:
    // [giphoo] fix fd leak (2026-11-05 disk-full incident): V2RW owned fd_ but had no destructor,
    // so every early-return in saveinc_impl/saveinc_peek_impl leaked one fd per call; with the
    // gateway's evict/rename cycling inodes those fds held 600G+ of deleted-but-open files.
    V2RW() = default;
    ~V2RW() { close(); }                // close() is idempotent (fd_ < 0 returns true)
    V2RW(const V2RW&) = delete;         // the fd is owned: no copy semantics
    V2RW& operator=(const V2RW&) = delete;
    bool open_existing(const std::string& path, std::string& why) {
        fd_ = ::open(path.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
        if (fd_ < 0) { why = std::strerror(errno); return false; }
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_nlink != 1) {
            why = "not a regular file with one name";
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        size_ = (uint64_t) st.st_size;
        return true;
    }
    uint64_t size() const { return size_; }
    bool read_at(uint64_t off, void* p, size_t n) override {
        uint8_t* q = (uint8_t*) p;
        while (n) {
            const ssize_t r = ::pread(fd_, q, n, (off_t) off);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { err_ = r < 0 ? std::strerror(errno) : "file ends early"; return false; }
            q += r;
            n -= (size_t) r;
            off += (uint64_t) r;
        }
        return true;
    }
    bool write_all(const void* p, size_t n) override {
        // append at the tracked end: plain write() would land at the shared fd offset (0 after open)
        const uint8_t* q = (const uint8_t*) p;
        while (n) {
            const ssize_t w = ::pwrite(fd_, q, n, (off_t) size_);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { err_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            q += w;
            n -= (size_t) w;
            size_ += (uint64_t) w;
        }
        return true;
    }
    bool rewrite_at(uint64_t off, const void* p, size_t n) {
        const uint8_t* q = (const uint8_t*) p;
        while (n) {
            const ssize_t w = ::pwrite(fd_, q, n, (off_t) off);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { err_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            q += w;
            n -= (size_t) w;
            off += (uint64_t) w;
        }
        return true;
    }
    bool sync() override {
        if (::fsync(fd_) == 0) return true;
        err_ = std::strerror(errno);
        return false;
    }
    bool truncate_to(uint64_t n) {
        if (::ftruncate(fd_, (off_t) n) == 0) { size_ = n; return true; }
        err_ = std::strerror(errno);
        return false;
    }
    std::string error() const override { return err_; }
    bool close() {
        if (fd_ < 0) return true;
        const bool ok = ::close(fd_) == 0;
        fd_ = -1;
        return ok;
    }
    // [giphoo] v2.4.2: reopen for the post-verify rollback (the file is closed by then)
    bool reopen_rw(const std::string& path) {
        if (fd_ >= 0) return true;
        fd_ = ::open(path.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
        if (fd_ < 0) { err_ = std::strerror(errno); return false; }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { err_ = std::strerror(errno); ::close(fd_); fd_ = -1; return false; }
        size_ = (uint64_t) st.st_size;
        return true;
    }
private:
    int fd_ = -1;
    uint64_t size_ = 0;
    std::string err_;
};

// ---- write side: every field streams through a block compressor, nothing is materialised -----
class V2Out {
public:
    V2Out(V2Io* io, uint64_t blk) : io_(io), blk_(blk) {}
    bool ok() const { return ok_; }
    bool read_failed() const { return read_failed_; }
    uint64_t raw_total() const { return raw_n_; }
    uint64_t data_end() const { return off_; }             // where the directory will sit
    std::vector<V2SegRec>& segments() { return segs_; }
    std::vector<V2BlockRec>& blocks() { return blocks_; }
    void set_progress(SessionProgress p, uint64_t expected) { progress_ = p; expected_ = expected; }
    // [giphoo] saveinc: start from an existing file's block table, appending at `base`
    void adopt(std::vector<V2BlockRec> old_blocks, uint64_t base, uint64_t raw_total) {
        blocks_ = std::move(old_blocks);
        off_ = base;
        raw_n_ = raw_total;
    }

    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    void open_segment(uint8_t kind, uint16_t layer = 0, uint32_t aux = 0, uint64_t ids_hash = 0) {
        flush_block();
        V2SegRec s;
        s.kind = kind;
        s.layer = layer;
        s.aux = aux;
        s.ids_hash = ids_hash;
        segs_.push_back(s);
        cur_seg_ = segs_.size() - 1;
    }
    // [giphoo] saveinc: continue an existing segment - keep its identity and block list, then append
    void continue_segment(V2SegRec s) {
        flush_block();
        segs_.push_back(std::move(s));
        cur_seg_ = segs_.size() - 1;
    }
    // [giphoo] v2.4: the live-ids segment's hash must describe the ids AS OF THIS SAVE, not the
    // base file's era - a stale hash refuses the next extension (fail-closed) and blinds the
    // partial reader's live shortcut.  Stamp it right after the continuation is set up.
    void stamp_ids_hash(uint64_t h) { if (cur_seg_ < segs_.size()) segs_[cur_seg_].ids_hash = h; }
    void bytes(const void* p, size_t c) { sink(p, c); }
    void img_bytes(const ConversationBuffer& b, size_t from, size_t n) {
        if (!n) return;
        b.visit(from, n, [&](const uint8_t* q, size_t c, size_t) { sink(q, c); return ok_; });
    }
    void src_bytes(const SessionKvSource& s, size_t part, size_t from, size_t n) {
        if (!n) return;
        if (!s.read) { ok_ = false; read_failed_ = true; return; }
        std::vector<uint8_t> buf(std::min<size_t>(n, 1u << 20));
        for (size_t at = 0; at < n && ok_;) {
            const size_t c = std::min(buf.size(), n - at);
            if (!s.read(part, at + from, buf.data(), c)) { ok_ = false; read_failed_ = true; return; }
            sink(buf.data(), c);
            at += c;
        }
    }
    void finish() { flush_block(); }

private:
    void raw(const void* p, size_t c) {   // counts scalars
        if (!ok_ || !c) return;
        sink(p, c);
    }
    void sink(const void* p, size_t c) {
        if (!ok_ || !c) return;
        if (!io_) { raw_n_ = sat_add(raw_n_, c); return; }   // the sizing pass counts
        const uint8_t* q = static_cast<const uint8_t*>(p);
        while (c) {
            if (buf_used_ == buf_.size()) flush_block();
            const size_t room = std::min(c, buf_.size() - buf_used_);
            std::memcpy(buf_.data() + buf_used_, q, room);
            buf_used_ += room;
            q += room;
            c -= room;
        }
    }
    void flush_block() {
        if (!buf_used_) return;
        V2BlockRec r;
        r.usize = (uint32_t) buf_used_;
        r.mode = 1;
        r.csize = r.usize;
        r.hash = session_hash64(buf_.data(), buf_used_, 0);
        r.offset = off_;
        if (io_) {
            const uint32_t hdr[3] = {r.usize, r.csize, r.mode};
            ok_ = io_->write_all(hdr, 12) && io_->write_all(buf_.data(), r.csize);
        }
        off_ = sat_add(off_, 12 + (uint64_t) r.csize);
        blocks_.push_back(r);
        segs_[cur_seg_].idx.push_back((uint32_t) (blocks_.size() - 1));
        segs_[cur_seg_].raw_total = sat_add(segs_[cur_seg_].raw_total, buf_used_);
        raw_n_ = sat_add(raw_n_, buf_used_);
        buf_used_ = 0;
        // [giphoo] the progress contract is monotonic done <= total: an incremental save's raw_n_
        // starts at the adopted payload (the old file's), so the total follows the high-water mark
        if (progress_) {
            prog_hi_ = std::max(prog_hi_, raw_n_);
            progress_(prog_hi_, std::max(expected_, prog_hi_));
        }
    }
    V2Io* io_;
    uint64_t blk_, off_ = kHeader, raw_n_ = 0, expected_ = 0, prog_hi_ = 0;
    size_t cur_seg_ = 0, buf_used_ = 0;
    bool ok_ = true, read_failed_ = false;
    SessionProgress progress_;
    std::vector<uint8_t> buf_ = std::vector<uint8_t>((size_t) kV2BlkMax);
    std::vector<V2SegRec> segs_;
    std::vector<V2BlockRec> blocks_;
};

// counts: every scalar first, so the reader can pre-gate everything before any payload byte
void v2_put_counts(V2Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources,
                   size_t n_ck) {
    for (int64_t g : s.geometry) o.i64(g);
    o.i64(s.layer_lo); o.i64(s.layer_hi); o.u64(s.cvec ? 1 : 0);
    o.u64(n_ck);
    auto ck = [&](const ConversationCheckpoint& c) {
        o.u64(c.ids.size());
        o.u64(c.imgs.size());
        o.u64(c.used);
        o.u64(c.gdn.size() * v2_esz(c.gdn));
        o.u64(c.ple.size() * v2_esz(c.ple));
        o.u64(c.tails.size() * v2_esz(c.tails));
        o.u64(c.dead.size() * v2_esz(c.dead));
        o.u64(c.block_pos.size() * v2_esz(c.block_pos));
    };
    ck(s.live);
    for (size_t i = 0; i < n_ck; ++i) ck(i < s.checkpoints.size() ? s.checkpoints[i] : ConversationCheckpoint{});
    const size_t n_layers = sources ? sources->size() : s.kv.size();
    o.u64(n_layers);
    for (size_t l = 0; l < n_layers; ++l) {
        if (sources) {
            const auto& k = (*sources)[l];
            put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
            for (size_t p = 0; p < 5; ++p) o.u64(k.sizes[p]);
        } else {
            const auto& k = s.kv[l];
            put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
            for (size_t p = 0; p < 5; ++p) o.u64(v2_img_part_size(k, p));
        }
    }
}

// [giphoo] each state field is its own segment (kind 4 live / kind 5 checkpoint; layer = the
// checkpoint keep index, aux = the field index): every field grows append-only, so an incremental
// save extends each field's segment at its own tail and the streams never shift out of place.
enum { kV2FIds = 0, kV2FImgs, kV2FGdn, kV2FPle, kV2FTails, kV2FDead, kV2FBlockPos, kV2FCount };
void v2_stream_state(V2Out& o, const ConversationCheckpoint& c, uint8_t kind, uint16_t owner) {
    const ConversationBuffer* none = nullptr;   // the fields are plain vectors, streamed as bytes
    (void) none;
    o.open_segment(kind, owner, kV2FIds, v2_ids_hash(c.ids));
    o.bytes(c.ids.data(), c.ids.size() * v2_esz(c.ids));
    o.open_segment(kind, owner, kV2FImgs);
    o.bytes(c.imgs.data(), c.imgs.size() * v2_esz(c.imgs));
    o.open_segment(kind, owner, kV2FGdn);
    o.bytes(c.gdn.data(), c.gdn.size() * v2_esz(c.gdn));
    o.open_segment(kind, owner, kV2FPle);
    o.bytes(c.ple.data(), c.ple.size() * v2_esz(c.ple));
    o.open_segment(kind, owner, kV2FTails);
    o.bytes(c.tails.data(), c.tails.size() * v2_esz(c.tails));
    o.open_segment(kind, owner, kV2FDead);
    o.bytes(c.dead.data(), c.dead.size() * v2_esz(c.dead));
    o.open_segment(kind, owner, kV2FBlockPos);
    o.bytes(c.block_pos.data(), c.block_pos.size() * v2_esz(c.block_pos));
}

void v2_put_stream(V2Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources, size_t n_ck) {
    o.open_segment(0);
    v2_put_counts(o, s, sources, n_ck);
    v2_stream_state(o, s.live, 4, 0);
    for (size_t i = 0; i < n_ck; ++i) {
        const ConversationCheckpoint empty{};
        const ConversationCheckpoint& c = i < s.checkpoints.size() ? s.checkpoints[i] : empty;
        v2_stream_state(o, c, 5, (uint16_t) i);
    }
    if (sources) {
        for (size_t l = 0; l < sources->size(); ++l) {
            const auto& k = (*sources)[l];
            for (size_t p = 0; p < 5; ++p) {
                if (!k.sizes[p]) continue;
                o.open_segment(1, (uint16_t) l, (uint32_t) p);
                o.src_bytes(k, p, 0, k.sizes[p]);
            }
        }
    } else {
        for (size_t l = 0; l < s.kv.size(); ++l) {
            const auto& k = s.kv[l];
            for (size_t p = 0; p < 5; ++p) {
                const size_t n = v2_img_part_size(k, p);
                if (!n) continue;
                o.open_segment(1, (uint16_t) l, (uint32_t) p);
                o.img_bytes(*v2_img_part(const_cast<ConversationKv&> (k), p), 0, n);
            }
        }
    }
    o.finish();
}

void v2_header_bytes(uint8_t* h, const SessionFileIdentity& id, uint64_t payload) {
    std::memcpy(h, kMagicV2, 8);
    const uint32_t version = 2, size = kHeader;
    std::memcpy(h + 8, &version, 4);
    std::memcpy(h + 12, &size, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    std::memcpy(h + 32, &payload, 8);
    std::memset(h + 40, 0, 16);   // reserved
    const uint64_t hh = session_hash64(h, 56, 0);
    std::memcpy(h + 56, &hh, 8);
}

void v2_dir_bytes(const std::vector<V2SegRec>& segs, const std::vector<V2BlockRec>& blocks, uint64_t payload,
                  std::vector<uint8_t>& d) {
    auto w8 = [&](const void* p, size_t n) { const uint8_t* q = (const uint8_t*) p; d.insert(d.end(), q, q + n); };
    auto w16 = [&](uint16_t v) { w8(&v, 2); };
    auto w32 = [&](uint32_t v) { w8(&v, 4); };
    auto w64 = [&](uint64_t v) { w8(&v, 8); };
    w8(kDirMagic, 8);
    w16(kV2Layout);
    w64(payload);
    w32((uint32_t) segs.size());
    w64(blocks.size());
    for (const auto& s : segs) {
        d.push_back(s.kind);
        d.push_back(0);
        w16(s.layer);
        w32(s.aux);
        w64(s.raw_total);
        w64(s.ids_hash);
        w32((uint32_t) s.idx.size());
        for (uint32_t i : s.idx) w32(i);
    }
    for (const auto& b : blocks) {
        w64(b.offset);
        w32(b.csize);
        w32(b.usize);
        w32(b.mode);
        w64(b.hash);
    }
    const uint64_t dh = session_hash64(d.data(), d.size(), 0);
    w64(dh);
}

bool v2_verify_structure(const std::string& path, std::string& error);   // [giphoo] v2.4: defined past the readers

bool write_v2_impl(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                   const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                   SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;
    if (has_stage_parts(image)) { error = "session file: layer-split state cannot be saved"; return false; }
    if (sources && !image.kv.empty()) { error = "session file: K/V given both in the image and as sources"; return false; }
    const uint64_t blk = v2_block_target();
    V2Out sizing(nullptr, blk);
    v2_put_stream(sizing, image, sources, image.checkpoints.size());
    if (!sizing.ok() || sizing.raw_total() > (uint64_t) std::numeric_limits<size_t>::max() - (64ull << 20)) {
        error = "session file: the state is too large for this build";
        return false;
    }
    const uint64_t payload = sizing.raw_total();
    st.error = SessionError::io;
    if (opt.min_free_bytes) {
        const uint64_t n_blocks_est = payload / kV2BlkMin + 64;
        const uint64_t worst_total = sat_add(sat_add(sat_add(sat_add(kHeader, payload), sat_mul(n_blocks_est, 12 + 44)), 16), 1u << 16);
        uint64_t avail = 0;
        if (!free_space(path, avail)) { error = "session file: cannot read the free disk space for " + path; return false; }
        if (avail < worst_total || avail - worst_total < opt.min_free_bytes) {
            st.error = SessionError::storage;
            error = "session file: not enough disk space (" + std::to_string(worst_total >> 20) + " MiB plus a reserve of " +
                    std::to_string(opt.min_free_bytes >> 20) + " MiB needed, " + std::to_string(avail >> 20) + " MiB free)";
            return false;
        }
    }
    Publisher pub;
    if (!pub.open(path, error)) { st.error = pub.kind(); return false; }
    RawFile raw;
    if (!pub.create_temp(raw, error)) {
        st.error = raw.kind();
        error = "session file: cannot create a temporary file beside " + path + ": " + error;
        return false;
    }
    V2FreshIo io(raw);
    if (opt.progress) opt.progress(0, payload);
    V2Out out(&io, blk);
    if (opt.progress) out.set_progress(opt.progress, payload);
    uint8_t h[kHeader];
    v2_header_bytes(h, id, payload);
    bool ok = io.write_all(h, kHeader);
    if (ok) {
        v2_put_stream(out, image, sources, image.checkpoints.size());
        ok = out.ok();
    }
    uint64_t file_total = 0;
    if (ok) {
        std::vector<uint8_t> d;
        v2_dir_bytes(out.segments(), out.blocks(), payload, d);
        std::vector<uint8_t> ft(16);
        const uint64_t dir_off = out.data_end();
        std::memcpy(ft.data(), kMagicV2, 8);
        std::memcpy(ft.data() + 8, &dir_off, 8);
        ok = io.write_all(d.data(), d.size()) && io.write_all(ft.data(), 16);
        file_total = dir_off + d.size() + 16;
    }
    if (ok && opt.phase) opt.phase("flush", file_total);
    if (ok) ok = io.sync();
    const bool closed = raw.close();
    ok = ok && closed;
    if (ok) {
        if (opt.phase) opt.phase("publish", file_total);
        std::string why;
        if (pub.publish(why, opt.durable, st, nullptr)) {
            std::string verr;
            if (!v2_verify_structure(path, verr)) {
                st.error = SessionError::invalid;
                error = "session file: written but " + verr;
                return false;
            }
            st.error = SessionError::none;
            bytes = (size_t) file_total;
            return true;
        }
        error = "session file: writing " + path + " failed" + (why.empty() ? std::string() : ": " + why);
        if (st.published) error += " (the new file has already replaced the old one; its folder entry may not survive a power loss)";
        return false;
    }
    error = "session file: writing " + path + " failed: " +
            (out.read_failed() ? "K/V source read" : (error.empty() ? io.error() : error));
    st.error = out.read_failed() ? SessionError::io : raw.kind();
    return false;
}

bool write_v2_guarded(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                      const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                      SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = write_v2_impl(path, image, sources, id, bytes, error, opt, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while writing " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

// ---- read side --------------------------------------------------------------------------------
bool read_v2_common(V2Io& io, uint64_t size, uint64_t payload, const SessionFileIdentity& id, SavedConversation& image,
                    size_t& bytes, std::string& error, const SessionReadLimits& limits,
                    const std::function<bool(uint64_t, std::string&)>* fits, SessionStatus& st);

struct V2Dir {
    uint64_t payload_raw = 0, dir_off = 0;
    std::vector<V2SegRec> segments;
    std::vector<V2BlockRec> blocks;
};

bool v2_read_dir(V2Io& f, uint64_t size, V2Dir& d, std::string& error) {
    if (size < kHeader + 16 + 8 + 2 + 8 + 4 + 8 + 8) { error = "session file: too small for the v2 layout"; return false; }
    uint8_t foot[16];
    if (!f.read_at(size - 16, foot, 16)) { error = "session file: footer read error: " + f.error(); return false; }
    if (std::memcmp(foot, kMagicV2, 8) != 0) { error = "session file: bad v2 footer"; return false; }
    std::memcpy(&d.dir_off, foot + 8, 8);
    const uint64_t dir_len = size - 16 - d.dir_off;
    if (d.dir_off < kHeader || dir_len < 8 + 2 + 8 + 4 + 8 + 8 + 8) { error = "session file: invalid directory offset"; return false; }
    if (dir_len > (uint64_t) 64 << 20) { error = "session file: directory too large"; return false; }
    std::vector<uint8_t> buf((size_t) dir_len);
    if (!f.read_at(d.dir_off, buf.data(), (size_t) dir_len)) { error = "session file: directory read error: " + f.error(); return false; }
    uint64_t dh = 0;
    std::memcpy(&dh, buf.data() + dir_len - 8, 8);
    if (dh != session_hash64(buf.data(), dir_len - 8, 0)) { error = "session file: directory checksum mismatch"; return false; }
    size_t at = 0;
    auto take = [&](void* p, size_t n) -> bool {
        if (at + n > dir_len - 8) { error = "session file: directory ends early"; return false; }
        std::memcpy(p, buf.data() + at, n);
        at += n;
        return true;
    };
    char magic[8];
    uint16_t layout = 0;
    uint32_t n_seg = 0;
    uint64_t n_blk = 0;
    if (!take(magic, 8) || std::memcmp(magic, kDirMagic, 8) != 0) { error = "session file: bad directory magic"; return false; }
    if (!take(&layout, 2) || layout != kV2Layout) {
        error = "session file: unsupported session layout " + std::to_string(layout);
        return false;
    }
    if (!take(&d.payload_raw, 8) || !take(&n_seg, 4) || !take(&n_blk, 8)) return false;
    if (!n_seg || n_seg > 65536 || n_blk > size / 12) { error = "session file: invalid directory counts"; return false; }
    d.segments.resize(n_seg);
    for (auto& s : d.segments) {
        uint8_t pad = 1;
        uint32_t n_idx = 0;
        if (!take(&s.kind, 1) || !take(&pad, 1) || !take(&s.layer, 2) || !take(&s.aux, 4) || !take(&s.raw_total, 8) ||
            !take(&s.ids_hash, 8) || !take(&n_idx, 4))
            return false;
        if ((s.kind != 0 && s.kind != 1 && s.kind != 4 && s.kind != 5) || pad || n_idx > n_blk) {
            error = "session file: invalid segment record";
            return false;
        }
        s.idx.resize(n_idx);
        for (auto& i : s.idx) {
            if (!take(&i, 4)) return false;
            if (i >= n_blk) { error = "session file: segment block index out of range"; return false; }
        }
    }
    d.blocks.resize(n_blk);
    for (auto& b : d.blocks) {
        if (!take(&b.offset, 8) || !take(&b.csize, 4) || !take(&b.usize, 4) || !take(&b.mode, 4) || !take(&b.hash, 8)) return false;
        if (b.mode > 1 || !b.usize || b.usize > kV2BlkMax || (b.mode == 1 && b.csize != b.usize) ||
            b.offset < kHeader || b.offset + 12 + b.csize > d.dir_off) {
            error = "session file: invalid block record";
            return false;
        }
    }
    uint64_t counted = 0;
    for (const auto& s : d.segments) {
        for (uint32_t i = 0; i < s.idx.size(); ++i) counted += d.blocks[s.idx[i]].usize;
        if (counted > d.payload_raw) { error = "session file: directory blocks exceed the payload"; return false; }
    }
    return true;
}

class V2SegReader {
public:
    struct ReadMeter {
        std::atomic<uint64_t> payload{0};
        std::atomic<uint64_t> blocks{0};
    };
    V2SegReader(V2Io& f, const V2Dir& d, const V2SegRec& s, std::string& error, ReadMeter* meter = nullptr)
        : f_(f), d_(d), s_(s), error_(error), depth_(v2_readahead_depth()), meter_(meter) {}
    V2SegReader(const V2SegReader&) = delete;              // futures do not copy; the vector moves
    V2SegReader& operator=(const V2SegReader&) = delete;
    V2SegReader(V2SegReader&&) = default;                  // references rebind, deque<future> moves
    bool read(uint8_t* p, size_t n) {
        while (n) {
            if (at_ == have_ && !next_block()) return false;
            const size_t c = std::min(n, have_ - at_);
            std::memcpy(p, cur_.data() + at_, c);
            at_ += c;
            p += c;
            n -= c;
            served_ += c;
            if (meter_) meter_->payload.fetch_add(c, std::memory_order_relaxed);
        }
        return true;
    }
    uint64_t served() const { return served_; }

private:
    // one decoded, verified block; the worker holds everything it needs by value
    // (fd + record pointer), so the reader may relocate while futures are in flight
    struct PreBlock {
        std::vector<uint8_t> data;
        std::string err;
        bool ok = false;
    };
    // header (12 bytes) and payload are contiguous in the file: one read
    static PreBlock fetch_fd(int fd, const V2BlockRec* r, ReadMeter* meter) {
        PreBlock b;
        std::vector<uint8_t> raw(12u + r->csize);
        size_t got = 0;
        while (got < raw.size()) {
            const ssize_t n = ::pread(fd, raw.data() + got, raw.size() - got, (off_t) (r->offset + got));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { b.err = "session file: block read error"; return b; }
            got += (size_t) n;
        }
        if (meter) meter->blocks.fetch_add(raw.size(), std::memory_order_relaxed);
        return finish(raw, r);
    }
    static PreBlock fetch_io(V2Io* f, const V2BlockRec* r, ReadMeter* meter) {
        PreBlock b;
        std::vector<uint8_t> raw(12u + r->csize);
        if (!f->read_at(r->offset, raw.data(), raw.size())) {
            b.err = "session file: block read error: " + f->error();
            return b;
        }
        if (meter) meter->blocks.fetch_add(raw.size(), std::memory_order_relaxed);
        return finish(raw, r);
    }
    static PreBlock finish(std::vector<uint8_t>& raw, const V2BlockRec* r) {
        PreBlock b;
        uint32_t usize = 0, csize = 0, mode = 0;
        std::memcpy(&usize, raw.data(), 4);
        std::memcpy(&csize, raw.data() + 4, 4);
        std::memcpy(&mode, raw.data() + 8, 4);
        if (usize != r->usize || csize != r->csize || mode != r->mode) {
            b.err = "session file: block header disagrees with the directory";
            return b;
        }
        b.data.resize(r->usize);
        if (r->mode != 1) { b.err = "session file: compressed blocks are not supported in this build"; return b; }
        std::memcpy(b.data.data(), raw.data() + 12, r->csize);
        if (session_hash64(b.data.data(), b.data.size(), 0) != r->hash) {
            b.err = "session file: block checksum mismatch";
            return b;
        }
        b.ok = true;
        return b;
    }
    bool next_block() {
        if (pos_ >= s_.idx.size()) { error_ = "session file: segment data ends early"; return false; }
        if (depth_ <= 0) {
            const V2BlockRec& r = d_.blocks[s_.idx[pos_]];
            PreBlock b = fetch_io(&f_, &r, meter_);
            if (!b.ok) { error_ = b.err; return false; }
            cur_ = std::move(b.data);
            have_ = cur_.size();
            at_ = 0;
            ++pos_;
            return true;
        }
        while (fut_.size() < (size_t) depth_ && launched_ < s_.idx.size()) {
            const V2BlockRec* r = &d_.blocks[s_.idx[launched_++]];
            uint64_t local_offset = r->offset;
            const int fd = f_.block_fd(r->offset, local_offset);
            if (fd < 0) {
                fut_.emplace_back(std::async(std::launch::async, [this, r] { return fetch_io(&f_, r, meter_); }));
            } else {
                V2BlockRec local = *r;
                local.offset = local_offset;
                fut_.emplace_back(std::async(std::launch::async, [fd, local, this] { return fetch_fd(fd, &local, meter_); }));
            }
        }
        if (fut_.empty()) { error_ = "session file: segment data ends early"; return false; }
        PreBlock b = fut_.front().get();
        fut_.pop_front();
        if (!b.ok) { error_ = b.err; return false; }
        cur_ = std::move(b.data);
        have_ = cur_.size();
        at_ = 0;
        ++pos_;
        return true;
    }
    V2Io& f_;
    const V2Dir& d_;
    const V2SegRec& s_;
    std::string& error_;
    std::vector<uint8_t> cur_;
    size_t at_ = 0, have_ = 0, pos_ = 0, launched_ = 0;
    uint64_t served_ = 0;
    int depth_ = 0;
    V2SegReader::ReadMeter* meter_ = nullptr;
    std::deque<std::future<PreBlock>> fut_;
};

struct V2Counts {
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = false;
    uint64_t n_ck = 0;
    struct Ck { uint64_t ids = 0, imgs = 0, used = 0, gdn = 0, ple = 0, tails = 0, dead = 0, block_pos = 0; };
    std::vector<Ck> cks;
    uint64_t n_layers = 0;
    struct Layer { int64_t f[7]; uint64_t parts[5]; };
    std::vector<Layer> layers;
};

struct V2In {
    V2Io* file;
    const V2Dir* dir;
    size_t next_seg = 0;
    uint64_t left;
    std::string& error;
    const SessionReadLimits& limits;
    V2SegReader::ReadMeter* meter = nullptr;
    bool fail(const std::string& m) { if (error.empty()) error = "session file: " + m; return false; }
    bool raw(void* p, size_t c) {
        if (c > left) return fail("payload ends early");
        if (c && !meta->read((uint8_t*) p, c)) return false;
        left -= c;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    bool count(uint64_t& n, uint64_t limit = UINT64_MAX) {
        if (!u64(n)) return false;
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " + std::to_string(limit));
        return true;
    }
    bool take_seg(uint8_t want_kind, uint16_t layer, uint32_t aux, V2SegReader*& out) {
        if (next_seg >= dir->segments.size()) return fail("payload ends early");
        const V2SegRec& s = dir->segments[next_seg];
        if (s.kind != want_kind || s.layer != layer || s.aux != aux)
            return fail("payload order disagrees with the directory");
        if (s.raw_total > left) return fail("payload ends early");
        ++next_seg;
        readers_.emplace_back(*file, *dir, s, error, meter);
        out = &readers_.back();
        return true;
    }
    std::vector<V2SegReader> readers_;
    V2SegReader* meta = nullptr;
};

bool v2_get_counts(V2In& in, V2Counts& c) {
    V2SegReader* m = nullptr;
    if (!in.take_seg(0, 0, 0, m)) return false;
    in.meta = m;
    for (auto& g : c.geometry) if (!in.i64(g)) return false;
    uint64_t cv = 0;
    if (!in.i64(c.layer_lo) || !in.i64(c.layer_hi) || !in.u64(cv)) return false;
    if (cv > 1) return in.fail("invalid cvec flag");
    c.cvec = cv == 1;
    if (in.limits.geometry && *in.limits.geometry != c.geometry)
        return in.fail("saved with another model geometry than this runtime's");
    if (in.limits.layer_range &&
        (in.limits.layer_range->first != c.layer_lo || in.limits.layer_range->second != c.layer_hi))
        return in.fail("saved with another layer range than this runtime's");
    if (!in.count(c.n_ck, in.limits.max_checkpoints)) return false;
    c.cks.resize((size_t) c.n_ck + 1);   // + 1: the live state rides in the same table
    for (auto& k : c.cks) {
        if (!in.count(k.ids, in.limits.max_tokens) || !in.count(k.imgs, in.limits.max_tokens) ||
            !in.u64(k.used) || !in.u64(k.gdn) || !in.u64(k.ple) || !in.u64(k.tails) || !in.u64(k.dead) ||
            !in.u64(k.block_pos))
            return false;
        const auto& ms = in.limits.max_state_bytes;
        if (k.gdn > ms[0] || k.ple > ms[1] || k.tails > ms[2] || k.dead > ms[3] || k.block_pos > ms[4])
            return in.fail("state array exceeds this engine's limit");
    }
    if (!in.count(c.n_layers, in.limits.max_kv_layers)) return false;
    c.layers.resize((size_t) c.n_layers);
    for (auto& l : c.layers) {
        for (auto& v : l.f) if (!in.i64(v)) return false;
        if (l.f[0] < INT32_MIN || l.f[0] > INT32_MAX) return in.fail("invalid K/V format");
        for (auto& p : l.parts) {
            if (!in.u64(p)) return false;
            const size_t li = &l - c.layers.data();
            const uint64_t lim = li < in.limits.max_kv_bytes.size() ? in.limits.max_kv_bytes[li][&p - l.parts] : UINT64_MAX;
            if (p > lim) return in.fail("K/V part exceeds this engine's limit " + std::to_string(lim));
        }
    }
    return true;
}

bool v2_read_state(V2In& in, uint8_t kind, uint16_t owner, ConversationCheckpoint& c, const V2Counts::Ck& n) {
    V2SegReader* r = nullptr;
    auto take = [&](auto& v, uint64_t elems, size_t esz, uint32_t field) -> bool {
        if (!in.take_seg(kind, owner, field, r)) return false;
        v.resize((size_t) elems);
        const uint64_t bytes = elems * esz;
        if (bytes && !r->read((uint8_t*) v.data(), (size_t) bytes)) {
            if (in.error.empty()) in.error = "session file: state segment ends early";
            return false;
        }
        in.left -= bytes;
        return true;
    };
    c.used = n.used;
    return take(c.ids, n.ids, sizeof(int32_t), kV2FIds) &&
           take(c.imgs, n.imgs, sizeof(ConversationImageKey), kV2FImgs) && take(c.gdn, n.gdn, 1, kV2FGdn) &&
           take(c.ple, n.ple, 1, kV2FPle) && take(c.tails, n.tails, 1, kV2FTails) &&
           take(c.dead, n.dead, 1, kV2FDead) && take(c.block_pos, n.block_pos, 1, kV2FBlockPos);
}

// [giphoo] v2.4: post-write structural verification.  Re-opens the published file and walks the
// SAME take_seg sequence the readers use - counts, live fields, checkpoint fields, non-empty K/V
// parts - without reading payload blocks (~ms).  A file that fails this would fail every restore;
// catching it at save time keeps the gateway's index pointing at the last good archive instead of
// discovering the corruption minutes or hours later as a cold read.
bool v2_verify_loaded(V2Io& io, const V2Dir& dir, std::string& error) {
    SessionReadLimits limits;
    std::string e2;
    V2In in{&io, &dir, 0, dir.payload_raw, e2, limits};
    V2Counts c;
    if (!v2_get_counts(in, c)) { error = "verify: counts: " + e2; return false; }
    if (in.next_seg != 1) { error = "verify: the counts are not the first segment"; return false; }
    V2SegReader* r = nullptr;
    for (int f = 0; f < kV2FCount; ++f)
        if (!in.take_seg(4, 0, (uint32_t) f, r)) { error = "verify: live field " + std::to_string(f) + ": " + e2; return false; }
    for (uint64_t i = 0; i < c.n_ck && i < 256; ++i)
        for (int f = 0; f < kV2FCount; ++f)
            if (!in.take_seg(5, (uint16_t) i, (uint32_t) f, r)) { error = "verify: checkpoint " + std::to_string(i) + " field " + std::to_string(f) + ": " + e2; return false; }
    for (size_t l = 0; l < c.layers.size(); ++l)
        for (size_t p2 = 0; p2 < 5; ++p2) {
            if (!c.layers[l].parts[p2]) continue;
            if (!in.take_seg(1, (uint16_t) l, (uint32_t) p2, r)) { error = "verify: K/V layer " + std::to_string(l) + " part " + std::to_string(p2) + ": " + e2; return false; }
        }
    if (in.next_seg != dir.segments.size()) {
        error = "verify: " + std::to_string(dir.segments.size() - in.next_seg) + " trailing segments after the walk";
        return false;
    }
    return true;
}

bool v2_verify_structure(const std::string& path, std::string& error) {
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size)) { error = "verify: cannot open: " + raw.error(); return false; }
    uint8_t h[kHeader];
    if (!raw.read_at(0, h, kHeader)) { error = "verify: header read"; return false; }
    if (std::memcmp(h, kMagicV2, 8) != 0) { error = "verify: not the v2 magic"; return false; }
    uint32_t version = 0;
    uint64_t payload = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&payload, h + 32, 8);
    if (version != 2) { error = "verify: unsupported version"; return false; }
    V2FreshIo io(raw);
    V2Dir dir;
    if (!v2_read_dir(io, size, dir, error)) { error = "verify: " + error; return false; }
    return v2_verify_loaded(io, dir, error);
}

bool get_payload_v2(V2In& in, const V2Counts& c, SavedConversation& s) {
    s.geometry = c.geometry;
    s.layer_lo = c.layer_lo;
    s.layer_hi = c.layer_hi;
    s.cvec = c.cvec;
    if (!v2_read_state(in, 4, 0, s.live, c.cks[0])) return false;
    s.checkpoints.resize((size_t) c.n_ck);
    for (size_t i = 0; i < c.n_ck; ++i)
        if (!v2_read_state(in, 5, (uint16_t) i, s.checkpoints[i], c.cks[i + 1])) return false;
    V2SegReader* r = nullptr;
    s.kv.resize((size_t) c.n_layers);
    for (size_t l = 0; l < c.n_layers; ++l) {
        auto& k = s.kv[l];
        k.format = (int) c.layers[l].f[0];
        k.cells = c.layers[l].f[1];
        k.heads = c.layers[l].f[2];
        k.head_dim = c.layers[l].f[3];
        k.page_size = c.layers[l].f[4];
        k.pooled_rows = c.layers[l].f[5];
        k.idx_dim = c.layers[l].f[6];
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t n = c.layers[l].parts[p];
            if (!n) continue;
            if (!in.take_seg(1, (uint16_t) l, (uint32_t) p, r)) return false;
            ConversationBuffer* b = v2_img_part(k, p);
            *b = {};
            b->resize((size_t) n);
            if (!b->visit(0, (size_t) n, [&](uint8_t* q, size_t cn, size_t) { return r->read(q, cn); })) return false;
            in.left -= n;
        }
    }
    return true;
}

// [giphoo] l2-exp v2.3 (M3): the parked reader.  A v2 file's counts carry every size up front, so
// the pool's budget is asked BEFORE any payload byte moves, and the physical RAM gate sees only the
// streaming transients (one decompressed block and the small vectors) instead of the whole image -
// the image lands in pool-managed RAM, which the budget already bounds.  This is the read-side key
// to "L2 ????": the old path asked the physical gate for image+floor (2.6G+2.5G) with ~3-4G free.
bool read_v2_parked(RawFile& raw, uint64_t size, const uint8_t* h, const SessionFileIdentity& id,
                    SavedConversation& image, size_t& bytes, std::string& error, const SessionReadLimits& limits,
                    const std::function<bool(uint64_t image_bytes, std::string& why)>& fits, SessionStatus& st);

bool read_v2_impl(RawFile& raw, uint64_t size, const uint8_t* h, const SessionFileIdentity& id, SavedConversation& image,
                  size_t& bytes, std::string& error, const SessionReadLimits& limits, SessionStatus& st) {
    st.error = SessionError::invalid;
    uint32_t version = 0, hsize = 0;
    uint64_t model = 0, config = 0, payload = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8);
    std::memcpy(&config, h + 24, 8);
    std::memcpy(&payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8);
    std::memcpy(&r1, h + 48, 8);
    std::memcpy(&hh, h + 56, 8);
    if (hh != session_hash64(h, 56, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (version != 2) { error = "session file: unsupported version " + std::to_string(version); return false; }
    if (hsize != kHeader || r0 || r1) { error = "session file: invalid header fields"; return false; }
    if (model != id.model) { error = "session file: saved with another model (model fingerprint differs)"; return false; }
    if (config != id.config) { error = "session file: saved with another engine configuration (config fingerprint differs)"; return false; }
    V2FreshIo io(raw);
    return read_v2_common(io, size, payload, id, image, bytes, error, limits, nullptr, st);
}

// the shared v2 walk; `fits` (when given) switches on the parked semantics: no whole-image
// physical admit, a pool-budget ask after counts, and a light transient-only floor check
bool read_v2_loaded_common(V2Io& io, uint64_t size, const V2Dir& dir, uint64_t payload,
                           const SessionFileIdentity& id, SavedConversation& image, size_t& bytes,
                           std::string& error, const SessionReadLimits& limits,
                           const std::function<bool(uint64_t, std::string&)>* fits, SessionStatus& st) {
    (void) id;
    if (dir.payload_raw != payload) { st.error = SessionError::invalid; error = "session file: payload size disagrees with the directory"; return false; }
    if (fits) {
        // transients only: one decompressed block pair, the small vectors, a margin
        const uint64_t light = sat_add(sat_add(kV2BlkMax * 2, (uint64_t) 1 << 20), session_read_peak_bytes(0));
        if (limits.admit) {
            std::string why;
            if (!limits.admit(light, why)) {
                st.error = SessionError::memory;
                error = "session file: " + why;
                return false;
            }
        }
    } else if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_peak_bytes(payload), why)) {
            st.error = SessionError::memory;
            error = "session file: " + why;
            return false;
        }
    }
    SavedConversation parsed;
    V2In in{&io, &dir, 0, payload, error, limits};
    V2Counts counts;
    if (!v2_get_counts(in, counts)) { st.error = SessionError::invalid; return false; }
    if (fits) {
        // every size is known now: ids+imgs+state live and per checkpoint, then the K/V parts
        uint64_t total = 0;
        for (const auto& k : counts.cks)
            total = sat_add(sat_add(sat_add(sat_add(total, k.ids * 4), k.imgs * 16),
                                    sat_add(sat_add(k.gdn, k.ple), sat_add(k.tails, k.dead))),
                            k.block_pos);
        for (const auto& l : counts.layers)
            for (size_t p = 0; p < 5; ++p) total = sat_add(total, l.parts[p]);
        std::string why;
        if (!(*fits)(total, why)) {
            st.error = SessionError::storage;
            error = "session file: " + why;
            return false;
        }
    }
    if (!get_payload_v2(in, counts, parsed)) { st.error = SessionError::invalid; return false; }
    if (in.left) { st.error = SessionError::invalid; error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    if (in.next_seg != dir.segments.size()) {
        st.error = SessionError::invalid;
        error = "session file: the directory has unconsumed segments";
        return false;
    }
    image = std::move(parsed);
    bytes = (size_t) size;
    st.error = SessionError::none;
    return true;
}

bool read_v2_common(V2Io& io, uint64_t size, uint64_t payload, const SessionFileIdentity& id, SavedConversation& image,
                    size_t& bytes, std::string& error, const SessionReadLimits& limits,
                    const std::function<bool(uint64_t, std::string&)>* fits, SessionStatus& st) {
    V2Dir dir;
    if (!v2_read_dir(io, size, dir, error)) { st.error = SessionError::invalid; return false; }
    return read_v2_loaded_common(io, size, dir, payload, id, image, bytes, error, limits, fits, st);
}

bool read_v2_parked(RawFile& raw, uint64_t size, const uint8_t* h, const SessionFileIdentity& id,
                    SavedConversation& image, size_t& bytes, std::string& error, const SessionReadLimits& limits,
                    const std::function<bool(uint64_t image_bytes, std::string& why)>& fits, SessionStatus& st) {
    // the header checks mirror read_v2_impl; the walk itself is the shared common body
    st.error = SessionError::invalid;
    uint32_t version = 0, hsize = 0;
    uint64_t model = 0, config = 0, payload = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8);
    std::memcpy(&config, h + 24, 8);
    std::memcpy(&payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8);
    std::memcpy(&r1, h + 48, 8);
    std::memcpy(&hh, h + 56, 8);
    if (hh != session_hash64(h, 56, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (version != 2) { error = "session file: unsupported version " + std::to_string(version); return false; }
    if (hsize != kHeader || r0 || r1) { error = "session file: invalid header fields"; return false; }
    if (model != id.model) { error = "session file: saved with another model (model fingerprint differs)"; return false; }
    if (config != id.config) { error = "session file: saved with another engine configuration (config fingerprint differs)"; return false; }
    V2FreshIo io(raw);
    return read_v2_common(io, size, payload, id, image, bytes, error, limits, &fits, st);
}

// ---- partial restore by checkpoint (M3.1) ------------------------------------------------------
// [giphoo] Load only what a borrower needs: the deepest file checkpoint whose ids hash-match
// the incoming prompt's prefix becomes the pool entry's live (kv truncated to its page-aligned
// position).  Matching costs two small reads (directory + counts) and hash-vs-hash; the K/V
// segments are read sequentially for exactly the prefix bytes, so only leading blocks move.
bool read_prefix_loaded(V2Io& io, uint64_t size, const V2Dir& dir, const SessionFileIdentity& id,
                        const std::vector<int32_t>& prompt, SavedConversation& image, size_t& bytes,
                        std::string& error, const SessionReadLimits& limits,
                        const std::function<bool(uint64_t, std::string&)>& fits, SessionStatus& st,
                        bool& live_would_match, SessionPrefixStats* prefix_stats) {
    (void) id;
    live_would_match = false;
    st.error = SessionError::invalid;
    if (prefix_stats) {
        const bool v3_archive = prefix_stats->v3_archive;
        *prefix_stats = {};
        prefix_stats->v3_archive = v3_archive;
        prefix_stats->archive_bytes = size;
    }
    V2SegReader::ReadMeter meter;
    SessionReadLimits open_limits;   // counts parse stays open; the caller's limits gate the image
    std::string e2;
    V2In in{&io, &dir, 0, dir.payload_raw, e2, open_limits, &meter};
    V2Counts c;
    if (!v2_get_counts(in, c)) { error = "session file: " + e2; return false; }
    // geometry gates before anything is allocated
    if (limits.geometry && *limits.geometry != c.geometry) {
        error = "session file: saved with another model geometry than this runtime's";
        return false;
    }
    if (limits.layer_range &&
        (limits.layer_range->first != c.layer_lo || limits.layer_range->second != c.layer_hi)) {
        error = "session file: saved with another layer range than this runtime's";
        return false;
    }
    // checkpoint inventory: keep-index -> ids_hash (aux 0) and field segments
    const V2SegRec* ck_ids_hash_seg[256] = {};
    const V2SegRec* ck_field[256][kV2FCount] = {};
    const V2SegRec* live_field[kV2FCount] = {};
    const V2SegRec* part_seg[256][5] = {};   // [layer][part] (layer bounded by sanity below)
    for (const auto& s : dir.segments) {
        if (s.kind == 5 && s.layer < 256) {
            if (s.aux == kV2FIds) {
                ck_ids_hash_seg[s.layer] = &s;
                ck_field[s.layer][kV2FIds] = &s;
            } else if (s.aux < kV2FCount) {
                ck_field[s.layer][s.aux] = &s;
            }
        } else if (s.kind == 4 && s.layer == 0 && s.aux < kV2FCount) {
            live_field[s.aux] = &s;
        } else if (s.kind == 1 && s.layer < 256 && s.aux < 5) {
            part_seg[s.layer][s.aux] = &s;
        }
    }
    // the live state itself matches the prompt prefix -> the caller is better off restoring in full
    if (live_field[kV2FIds] && c.cks[0].ids && c.cks[0].ids <= prompt.size() &&
        session_hash64(prompt.data(), (size_t) c.cks[0].ids * sizeof(int32_t), 0) ==
            live_field[kV2FIds]->ids_hash)
        live_would_match = true;
    if (prefix_stats) prefix_stats->live_would_match = live_would_match;
    // deepest checkpoint whose ids hash-match the prompt prefix: token depth wins, keep index only
    // breaks an exact-length tie (the keep set is policy-ordered, not depth-ordered)
    int64_t pick = -1;
    uint64_t pick_len = 0;
    for (uint64_t i = 0; i < c.n_ck && i < 256; ++i) {
        const uint64_t len = c.cks[(size_t) i + 1].ids;
        if (!len || len > prompt.size() || !ck_ids_hash_seg[i] || !ck_field[i][kV2FIds]) continue;
        if (session_hash64(prompt.data(), (size_t) len * sizeof(int32_t), 0) ==
            ck_ids_hash_seg[i]->ids_hash) {
            if (len > pick_len) {
                pick = (int64_t) i;
                pick_len = len;
            }
        }
    }
    if (prefix_stats) prefix_stats->checkpoint_tokens = pick_len;
    if (pick < 0) {
        error = "session file: no checkpoint of this file is a prefix of the prompt";
        return false;
    }
    const auto& picked = c.cks[(size_t) pick + 1];
    const uint64_t P = pick_len;
    // The prefix plan is known from counts/directory: ask the pool for the image BEFORE any
    // checkpoint field or K/V payload block moves.  A refusal then costs only metadata reads.
    auto prefix_sizes = [&](size_t l, uint64_t (&sizes)[5]) -> bool {
        const auto& hdr = c.layers[l];
        const uint64_t cells_full = (uint64_t) hdr.f[1];
        const int64_t page = hdr.f[4] > 0 ? hdr.f[4] : 1;
        const uint64_t cells_P = (P + (uint64_t) page - 1) / (uint64_t) page * (uint64_t) page;
        const uint64_t row_bytes = (uint64_t) hdr.f[6] * 4;
        const uint64_t pooled_full = hdr.parts[4];
        uint64_t rows_P = 0;
        if (pooled_full && row_bytes) {
            const uint64_t rows_full = pooled_full / row_bytes;
            const uint64_t upto_full = c.cks[0].ids;
            const uint64_t idx_block = rows_full > 1 && upto_full ? upto_full / (rows_full - 1) : 0;
            rows_P = idx_block ? P / idx_block + 1 : rows_full;
            rows_P = std::min<uint64_t>(rows_P, rows_full);
        }
        sizes[4] = rows_P * row_bytes;
        if (cells_full)
            for (int p = 0; p < 4; ++p) sizes[p] = cells_P * (hdr.parts[p] / cells_full);
        for (int p = 0; p < 5; ++p)
            if (sizes[p] && !part_seg[l][p]) return false;
        return true;
    };
    uint64_t planned_image_bytes = sat_add(picked.ids * sizeof(int32_t), picked.imgs * sizeof(ConversationImageKey));
    planned_image_bytes = sat_add(planned_image_bytes, picked.gdn);
    planned_image_bytes = sat_add(planned_image_bytes, picked.ple);
    planned_image_bytes = sat_add(planned_image_bytes, picked.tails);
    planned_image_bytes = sat_add(planned_image_bytes, picked.dead);
    planned_image_bytes = sat_add(planned_image_bytes, picked.block_pos);
    for (size_t l = 0; l < c.n_layers; ++l) {
        uint64_t sizes[5] = {};
        if (!prefix_sizes(l, sizes)) {
            error = "session file: the file lacks a K/V part segment";
            return false;
        }
        for (uint64_t n : sizes) planned_image_bytes = sat_add(planned_image_bytes, n);
    }
    if (prefix_stats) prefix_stats->image_bytes = planned_image_bytes;
    if (fits) {
        std::string why;
        if (!fits(planned_image_bytes, why)) {
            st.error = SessionError::storage;
            error = "session file: " + why;
            return false;
        }
    }
    // the picked checkpoint becomes the image's live: its seven field segments, whole
    SavedConversation out;
    out.geometry = c.geometry;
    out.layer_lo = c.layer_lo;
    out.layer_hi = c.layer_hi;
    out.cvec = c.cvec;
    out.live.used = picked.used;
    for (int f = 0; f < kV2FCount; ++f) {
        if (!ck_field[pick][f]) { error = "session file: the file lacks a checkpoint field segment"; return false; }
        V2SegReader rf(io, dir, *ck_field[pick][f], error, &meter);
        auto take = [&](auto& v, uint64_t bytes) -> bool {
            using E = typename std::remove_reference_t<decltype(v)>::value_type;
            if (bytes % sizeof(E)) return false;
            v.resize((size_t) (bytes / sizeof(E)));
            return bytes == 0 || rf.read((uint8_t*) v.data(), (size_t) bytes);
        };
        const bool okf = f == kV2FIds ? take(out.live.ids, picked.ids * sizeof(int32_t)) :
                          f == kV2FImgs ? take(out.live.imgs, picked.imgs * sizeof(ConversationImageKey)) :
                          f == kV2FGdn ? take(out.live.gdn, picked.gdn) :
                          f == kV2FPle ? take(out.live.ple, picked.ple) :
                          f == kV2FTails ? take(out.live.tails, picked.tails) :
                          f == kV2FDead ? take(out.live.dead, picked.dead) :
                                          take(out.live.block_pos, picked.block_pos);
        if (!okf) { error = "session file: checkpoint segment ends early"; return false; }
    }
    // K/V parts truncated to the checkpoint's page-aligned position
    out.kv.resize((size_t) c.n_layers);
    uint64_t image_bytes = 0;
    for (size_t l = 0; l < c.n_layers; ++l) {
        const auto& hdr = c.layers[l];
        ConversationKv& k = out.kv[l];
        k.format = (int) hdr.f[0];
        k.heads = hdr.f[2];
        k.head_dim = hdr.f[3];
        k.page_size = hdr.f[4];
        k.idx_dim = hdr.f[6];
        uint64_t sizes_P[5] = {};
        if (!prefix_sizes(l, sizes_P)) {
            error = "session file: the file lacks a K/V part segment";
            return false;
        }
        const uint64_t page = hdr.f[4] > 0 ? (uint64_t) hdr.f[4] : 1;
        k.cells = (int64_t) ((P + page - 1) / page * page);
        const uint64_t row_bytes = (uint64_t) k.idx_dim * 4;
        k.pooled_rows = (int64_t) (row_bytes ? sizes_P[4] / row_bytes : 0);
        for (int p = 0; p < 5; ++p) {
            const uint64_t n = sizes_P[p];
            ConversationBuffer* b = v2_img_part(k, (size_t) p);
            *b = {};
            if (!n) continue;
            V2SegReader r(io, dir, *part_seg[l][p], error, &meter);
            b->resize((size_t) n);
            if (!b->visit(0, (size_t) n, [&](uint8_t* q, size_t cn, size_t) { return r.read(q, cn); })) {
                error = "session file: K/V part segment ends early";
                return false;
            }
            image_bytes += n;
        }
    }
    image_bytes += (uint64_t) out.live.ids.size() * 4 + out.live.imgs.size() * 16 +
                   out.live.gdn.size() + out.live.ple.size() + out.live.tails.size() +
                   out.live.dead.size() + out.live.block_pos.size();
    image = std::move(out);
    bytes = (size_t) size;
    if (prefix_stats) {
        prefix_stats->checkpoint_hit = true;
        prefix_stats->image_bytes = image_bytes;
        prefix_stats->payload_read_bytes = meter.payload.load(std::memory_order_relaxed);
        prefix_stats->block_read_bytes = meter.blocks.load(std::memory_order_relaxed);
        prefix_stats->live_would_match = live_would_match;
    }
    st.error = SessionError::none;
    return true;
}

bool read_prefix_impl(const std::string& path, const SessionFileIdentity& id,
                      const std::vector<int32_t>& prompt, SavedConversation& image, size_t& bytes,
                      std::string& error, const SessionReadLimits& limits,
                      const std::function<bool(uint64_t, std::string&)>& fits, SessionStatus& st,
                      bool& live_would_match, SessionPrefixStats* prefix_stats) {
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size)) {
        st.error = raw.refused() ? SessionError::invalid : raw.kind();
        error = "session file: " + path + ": " + raw.error();
        return false;
    }
    uint8_t h[kHeader];
    if (!raw.read_at(0, h, kHeader)) { st.error = raw.kind(); error = "session file: header read error"; return false; }
    if (std::memcmp(h, kMagicV2, 8) != 0) { error = "session file: the partial read needs the v2 format"; return false; }
    uint32_t version = 0;
    uint64_t model = 0, config = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&model, h + 16, 8);
    std::memcpy(&config, h + 24, 8);
    if (version != 2 || model != id.model || config != id.config) {
        error = "session file: saved with another model or configuration";
        return false;
    }
    V2FreshIo io(raw);
    V2Dir dir;
    if (!v2_read_dir(io, size, dir, error)) return false;
    return read_prefix_loaded(io, size, dir, id, prompt, image, bytes, error, limits, fits, st,
                              live_would_match, prefix_stats);
}

// ---- the incremental save (M2.2) --------------------------------------------------------------
uint64_t saveinc_compact_env(const char* key, uint64_t fallback) {
    const char* e = std::getenv(key);
    if (!e || !*e) return fallback;
    const long long value = std::strtoll(e, nullptr, 10);
    return value >= 0 ? (uint64_t) value : fallback;
}

bool saveinc_compaction_values(uint64_t physical, uint64_t logical, std::string& reason) {
    const uint64_t file_limit = saveinc_compact_env("STRATA_SAVEINC_COMPACT_FILE_BYTES", 12ull << 30);
    const uint64_t slack = saveinc_compact_env("STRATA_SAVEINC_COMPACT_SLACK_BYTES", 1ull << 30);
    const uint64_t ratio_x100 = saveinc_compact_env("STRATA_SAVEINC_COMPACT_RATIO_X100", 200);
    const bool absolute = file_limit > 0 && physical >= file_limit;
    const bool amplified = ratio_x100 > 0 && physical > logical && physical - logical > slack &&
                           (long double) physical > (long double) logical * (long double) ratio_x100 / 100.0L;
    if (!absolute && !amplified) return false;
    reason = "saveinc: compaction required (physical=" + std::to_string(physical) +
             ", logical=" + std::to_string(logical) + ", limit=" + std::to_string(file_limit) + ")";
    return true;
}

bool saveinc_compaction_check(const std::string& path, std::string& reason) {
    reason.clear();
    V2RW f;
    std::string why;
    if (!f.open_existing(path, why)) return false;
    uint8_t h[kHeader];
    if (!f.read_at(0, h, kHeader) || std::memcmp(h, kMagicV2, 8) != 0) return false;
    V2Dir dir;
    std::string error;
    if (!v2_read_dir(f, f.size(), dir, error)) return false;
    return saveinc_compaction_values(f.size(), dir.payload_raw, reason);
}

// Appends to the existing v2.1 file in place: reused checkpoint segments and the K/V part prefixes
// keep their blocks untouched; the live tail, the missing checkpoints, the part tails, the counts
// block and the directory are appended.  The header is rewritten in place afterwards.  A crash
// mid-append leaves a file whose footer no longer matches (the reader refuses it fail-closed); the
// conversation itself is still live in the engine, so the next save rebuilds the file.
bool saveinc_impl(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                  const SessionFileIdentity& id, const std::vector<SessionSaveIncSlot>& keep, size_t& bytes,
                  std::string& error, const SessionWriteOptions& opt, SessionStatus& st) {
    st = {};

    st.error = SessionError::invalid;
    if (has_stage_parts(meta)) { error = "saveinc: layer-split state cannot be saved"; return false; }
    if (!meta.kv.empty()) { error = "saveinc: meta carries K/V"; return false; }
    V2RW f;
    std::string why;
    if (!f.open_existing(path, why)) { error = "saveinc: cannot open " + path + ": " + why; return false; }
    uint8_t h[kHeader];
    if (!f.read_at(0, h, kHeader)) { error = "saveinc: header read: " + f.error(); return false; }
    if (std::memcmp(h, kMagicV2, 8) != 0) { error = "saveinc: not a v2 session file"; return false; }
    {
        uint32_t version = 0;
        uint64_t model = 0, config = 0;
        std::memcpy(&version, h + 8, 4);
        std::memcpy(&model, h + 16, 8);
        std::memcpy(&config, h + 24, 8);
        if (version != 2 || model != id.model || config != id.config) {
            error = "saveinc: the file belongs to another model or configuration";
            return false;
        }
    }
    V2Dir old;
    if (!v2_read_dir(f, f.size(), old, error)) { error = "saveinc: " + error; return false; }
    if (saveinc_compaction_values(f.size(), old.payload_raw, why)) { error = why; return false; }
    // counts of the old file
    SessionReadLimits open_limits;
    std::string e2;
    V2In pin{&f, &old, 0, old.payload_raw, e2, open_limits};
    V2Counts oc;
    if (!v2_get_counts(pin, oc)) { error = "saveinc: " + e2; return false; }
    if (pin.next_seg != 1) { error = "saveinc: unexpected directory order"; return false; }
    // 1. the live ids must extend the old file's live ids exactly
    const uint64_t old_ids = oc.cks[0].ids;
    const ConversationCheckpoint& live = meta.live;
    if (old_ids > live.ids.size()) { error = "saveinc: the live ids shrank"; return false; }
    const V2SegRec* old_live_ids = nullptr;
    for (const auto& seg : old.segments)
        if (seg.kind == 4 && seg.layer == 0 && seg.aux == kV2FIds) { old_live_ids = &seg; break; }
    if (!old_live_ids) { error = "saveinc: the old file has no live ids segment"; return false; }
    if (old_ids) {
        const uint64_t hprev = session_hash64(live.ids.data(), (size_t) old_ids * sizeof(int32_t), 0);
        if (hprev != old_live_ids->ids_hash) { error = "saveinc: the live ids diverge from the file"; return false; }
    }
    // 2. the keep plan: meta.checkpoints holds exactly the fresh ones, keep order rules the file
    const V2SegRec* old_ck[256][kV2FCount] = {};
    for (const auto& s : old.segments)
        if (s.kind == 5 && s.layer < 256 && s.aux < kV2FCount) old_ck[s.layer][s.aux] = &s;
    size_t n_fresh = 0;
    for (const auto& slot : keep) {
        const bool r = slot.reuse >= 0 && slot.fresh < 0, fr = slot.fresh >= 0 && slot.reuse < 0;
        if (r == fr) { error = "saveinc: a keep slot must be exactly reuse or fresh"; return false; }
        if (r) {
            bool complete = slot.reuse < 256;
            for (int f2 = 0; f2 < kV2FCount && complete; ++f2) complete = old_ck[slot.reuse][f2] != nullptr;
            if (!complete) {
                error = "saveinc: a keep slot reuses an absent old checkpoint";
                return false;
            }
        }
        if (fr) ++n_fresh;
    }
    if (meta.checkpoints.size() != n_fresh) {
        error = "saveinc: meta.checkpoints must hold exactly the fresh keep slots";
        return false;
    }
    for (const auto& slot : keep)
        if (slot.fresh >= 0 && (size_t) slot.fresh >= meta.checkpoints.size()) {
            error = "saveinc: a fresh keep slot points outside meta.checkpoints";
            return false;
        }
    // 3. per-part growth (pooled: the moving spare row is re-sent)
    if (kv.size() != oc.layers.size()) { error = "saveinc: the K/V layer count changed"; return false; }
    for (size_t l = 0; l < kv.size(); ++l) {
        const auto& k = kv[l];
        const auto& ol = oc.layers[l];
        // identity fields only: format, heads, head_dim, page_size, idx_dim.  cells (the page-aligned
        // token count) and pooled_rows GROW with the conversation - their growth is the part deltas.
        const int64_t nf[7] = {(int64_t) k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim};
        for (int a : {0, 2, 3, 4, 6})
            if (nf[a] != ol.f[a]) {
                error = "saveinc: the K/V header of layer " + std::to_string(l) + " changed";
                return false;
            }
        if (k.cells < ol.f[1] || k.pooled_rows < ol.f[5]) {
            error = "saveinc: the K/V extent of layer " + std::to_string(l) + " shrank";
            return false;
        }
        const uint64_t row = (uint64_t) k.idx_dim * 4;
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t osz = ol.parts[p], nsz = k.sizes[p];
            const uint64_t floor_sz = p == 4 && osz >= row ? osz - row : osz;
            if (nsz < floor_sz) { error = "saveinc: a K/V part shrank"; return false; }
        }
    }
    // 4. the appended image: counts + live tail + fresh checkpoints + part tails + directory
    const uint64_t blk = v2_block_target();
    const uint64_t base = f.size();
    V2Out out(&f, blk);
    out.adopt(old.blocks, base, old.payload_raw);
    if (opt.progress) out.set_progress(opt.progress, old.payload_raw);   // grows with every appended byte
    // counts (all-new scalars)
    out.open_segment(0);
    for (int64_t g : meta.geometry) out.i64(g);
    out.i64(meta.layer_lo); out.i64(meta.layer_hi); out.u64(meta.cvec ? 1 : 0);
    out.u64(keep.size());
    auto put_ck_counts = [&](const ConversationCheckpoint& c) {
        out.u64(c.ids.size());
        out.u64(c.imgs.size());
        out.u64(c.used);
        out.u64(c.gdn.size() * v2_esz(c.gdn));
        out.u64(c.ple.size() * v2_esz(c.ple));
        out.u64(c.tails.size() * v2_esz(c.tails));
        out.u64(c.dead.size() * v2_esz(c.dead));
        out.u64(c.block_pos.size() * v2_esz(c.block_pos));
    };
    // [giphoo] L2 2.0 ??? (1/4): split ?? + counts (????? conversation_file_l2.inc)
    const bool l2_split_ = strata::core::l2two::hook_split(meta.live, oc.cks[0]);
    strata::core::l2two::hook_put_counts(put_ck_counts, l2_split_, meta.live, oc.cks[0]);
    for (size_t i = 0; i < keep.size(); ++i) {
        const auto& slot = keep[i];
        if (slot.reuse >= 0) {
            // the reused checkpoint's counts come from the old counts table (keep index + 1)
            const auto& k = oc.cks[(size_t) slot.reuse + 1];
            out.u64(k.ids); out.u64(k.imgs); out.u64(k.used);
            out.u64(k.gdn); out.u64(k.ple); out.u64(k.tails); out.u64(k.dead); out.u64(k.block_pos);
        } else {
            put_ck_counts(meta.checkpoints[(size_t) slot.fresh]);
        }
    }
    out.u64(kv.size());
    for (size_t l = 0; l < kv.size(); ++l) {
        const auto& k = kv[l];
        put_kv_header(out, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        for (size_t p = 0; p < 5; ++p) out.u64(k.sizes[p]);
    }
    // live tail: each field's segment continues at its own tail (fields grow append-only)
    const auto& olc = oc.cks[0];
    const V2SegRec* old_live_field[kV2FCount] = {};
    for (const auto& seg : old.segments)
        if (seg.kind == 4 && seg.layer == 0 && seg.aux < kV2FCount) old_live_field[seg.aux] = &seg;
    auto live_field = [&](uint32_t field, const void* np, uint64_t nf, uint64_t of, uint64_t esz) -> bool {
        const V2SegRec* oseg = old_live_field[field];
        if (oseg) {
            out.continue_segment(*oseg);
            if (nf > of) out.bytes((const uint8_t*) np + of * esz, (size_t) ((nf - of) * esz));
        } else {
            out.open_segment(4, 0, field, field == kV2FIds ? v2_ids_hash(live.ids) : 0);
            out.bytes(np, (size_t) (nf * esz));
        }
        return true;
    };
    // [giphoo] L2 2.0 ??? (2/4): split ????????? + ids_hash ??? (?? state ??)
    live_field(kV2FIds, live.ids.data(), strata::core::l2two::hook_field(l2_split_, live.ids.size(), olc.ids), olc.ids, sizeof(int32_t));
    if (!l2_split_)
        out.stamp_ids_hash(v2_ids_hash(live.ids));   // [giphoo] v2.4: the hash now covers the CURRENT ids
    live_field(kV2FImgs, live.imgs.data(), strata::core::l2two::hook_field(l2_split_, live.imgs.size(), olc.imgs), olc.imgs, sizeof(ConversationImageKey));
    live_field(kV2FGdn, live.gdn.data(), strata::core::l2two::hook_field(l2_split_, live.gdn.size(), olc.gdn), olc.gdn, 1);
    live_field(kV2FPle, live.ple.data(), strata::core::l2two::hook_field(l2_split_, live.ple.size(), olc.ple), olc.ple, 1);
    live_field(kV2FTails, live.tails.data(), strata::core::l2two::hook_field(l2_split_, live.tails.size(), olc.tails), olc.tails, 1);
    live_field(kV2FDead, live.dead.data(), strata::core::l2two::hook_field(l2_split_, live.dead.size(), olc.dead), olc.dead, 1);
    live_field(kV2FBlockPos, live.block_pos.data(), strata::core::l2two::hook_field(l2_split_, live.block_pos.size(), olc.block_pos), olc.block_pos, 1);
    // checkpoints: a reused one keeps all seven field segments untouched; a fresh one streams them
    for (size_t i = 0; i < keep.size(); ++i) {
        const auto& slot = keep[i];
        if (slot.reuse >= 0) {
            for (int f2 = 0; f2 < kV2FCount; ++f2) {
                // [giphoo] v2.4.1: the keep position owns the segment's index.  A continued record carried
                // the OLD file's layer number; under checkpoint rotation (the oldest dropped, a fresh one
                // appended) the reuse index and the keep position disagree, and the emitted directory held
                // kind-5 segments out of the ascending order every reader (and the post-write verifier)
                // walks - "checkpoint N field 0: payload order disagrees" (08:40 first catch).
                V2SegRec rs = *old_ck[slot.reuse][f2];
                rs.layer = (uint16_t) i;
                out.continue_segment(std::move(rs));
            }
        } else {
            const ConversationCheckpoint& c = meta.checkpoints[(size_t) slot.fresh];
            v2_stream_state(out, c, 5, (uint16_t) i);
        }
    }
    // K/V part tails.  Parts 0-3 grow append-only: keep the old blocks, stream [osz, nsz).
    // Pooled (part 4) rewrites its moving spare row: the stream boundary is one row back, and
    // old blocks are kept only up to that boundary - the boundary usually falls inside the old
    // tail block, so that block is dropped and its kept half re-streamed from the source.
    for (size_t l = 0; l < kv.size(); ++l) {
        const auto& k = kv[l];
        const uint64_t row = (uint64_t) k.idx_dim * 4;
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t osz = oc.layers[l].parts[p], nsz = k.sizes[p];
            const uint64_t want = p == 4 && osz >= row ? osz - row : osz;   // the stream boundary
            if (!osz && !nsz) continue;
            const V2SegRec* old_seg = nullptr;
            for (const auto& s : old.segments)
                if (s.kind == 1 && s.layer == l && s.aux == p) { old_seg = &s; break; }
            if (osz && !old_seg) { error = "saveinc: the old file lacks a K/V part segment"; return false; }
            uint64_t keep = 0;
            size_t keep_blocks = 0;
            V2SegRec seg;
            if (old_seg) {
                seg = *old_seg;
                for (uint32_t bi : old_seg->idx) {
                    const uint64_t u = old.blocks[bi].usize;
                    if (keep + u <= want) {
                        keep += u;
                        ++keep_blocks;
                    } else {
                        break;
                    }
                }
                seg.idx.resize(keep_blocks);
                seg.raw_total = keep;
            } else {
                seg.kind = 1;
                seg.layer = (uint16_t) l;
                seg.aux = (uint32_t) p;
            }
            out.continue_segment(seg);
            out.src_bytes(k, p, keep, nsz - keep);
        }
    }
    out.finish();
    if (!out.ok()) {
        // the file grew with partial blocks: cut it back to the state before this attempt
        f.truncate_to(base);
        f.close();
        error = "saveinc: writing " + path + " failed: " +
                (out.read_failed() ? "K/V source read" : f.error());
        st.error = out.read_failed() ? SessionError::io : SessionError::io;
        return false;
    }
    // directory + footer + header rewrite.  The payload is the NEW segments' sum - the adopted
    // raw_total counts the old file whole, including the replaced counts block and dropped blocks.
    uint64_t payload_new = 0;
    for (const auto& seg : out.segments()) payload_new = sat_add(payload_new, seg.raw_total);
    std::vector<uint8_t> d;
    v2_dir_bytes(out.segments(), out.blocks(), payload_new, d);
    std::vector<uint8_t> ft(16);
    const uint64_t dir_off = out.data_end();
    std::memcpy(ft.data(), kMagicV2, 8);
    std::memcpy(ft.data() + 8, &dir_off, 8);
    uint8_t nh[kHeader];
    v2_header_bytes(nh, id, payload_new);
    bool ok = f.write_all(d.data(), d.size()) && f.write_all(ft.data(), 16) && f.rewrite_at(0, nh, kHeader);
    if (ok && opt.phase) opt.phase("flush", f.size());
    if (ok) ok = f.sync();
    const bool closed = f.close();
    ok = ok && closed;
    if (!ok) { st.error = SessionError::io; error = "saveinc: publishing " + path + " failed: " + f.error(); return false; }
    {
        std::string verr;
        if (!v2_verify_structure(path, verr)) {
            // [giphoo] v2.4.2: roll the file back to the pre-extension size - the extended
            // directory is already on disk and invalid; if the engine dies before the
            // fallback full save replaces the file, the archive would sit corrupt in the
            // index (761cbd/016161dd: two bugs' parting gift).  truncate restores the
            // last good state the reader can still parse.
            // the header was already rewritten with the new payload figure - restore the
            // original bytes too, or the rolled-back file would describe a directory it
            // no longer carries (old trailer at base-16 vs new header payload).
            if (!f.reopen_rw(path) || !f.truncate_to(base) || !f.rewrite_at(0, h, kHeader) || !f.sync()) {
                error = "saveinc: " + verr + " (rollback to the base failed: " + f.error() + ")";
                st.error = SessionError::io;
                return false;
            }
            error = "saveinc: " + verr;
            st.error = SessionError::invalid;
            return false;
        }
    }
    bytes = (size_t) f.size();
    st.error = SessionError::none;
    // [giphoo] L2 2.0 ??? (3/4): state ?????? L2-1 (????? conversation_file_l2.inc)
    strata::core::l2two::hook_post_saveinc(path, meta, id, opt, l2_split_);
    return true;
}

bool saveinc_guarded(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                     const SessionFileIdentity& id, const std::vector<SessionSaveIncSlot>& keep, size_t& bytes,
                     std::string& error, const SessionWriteOptions& opt, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = saveinc_impl(path, meta, kv, id, keep, bytes, error, opt, st);
    } catch (const std::bad_alloc&) {
        error = "saveinc: out of memory";
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

// what the engine needs to plan an incremental save
bool saveinc_peek_impl(const std::string& path, const SessionFileIdentity& id, SessionSaveIncPeek& peek,
                       std::string& error) {
    V2RW f;
    std::string why;
    if (!f.open_existing(path, why)) { error = "saveinc: cannot open " + path + ": " + why; return false; }
    uint8_t h[kHeader];
    if (!f.read_at(0, h, kHeader)) { error = "saveinc: header read: " + f.error(); return false; }
    if (std::memcmp(h, kMagicV2, 8) != 0) { error = "saveinc: not a v2 session file"; return false; }
    uint32_t version = 0;
    uint64_t model = 0, config = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&model, h + 16, 8);
    std::memcpy(&config, h + 24, 8);
    if (version != 2 || model != id.model || config != id.config) {
        error = "saveinc: the file belongs to another model or configuration";
        return false;
    }
    V2Dir dir;
    if (!v2_read_dir(f, f.size(), dir, error)) { error = "saveinc: " + error; return false; }
    SessionReadLimits open_limits;
    std::string e2;
    V2In in{&f, &dir, 0, dir.payload_raw, e2, open_limits};
    V2Counts c;
    if (!v2_get_counts(in, c)) { error = "saveinc: " + e2; return false; }
    peek = {};
    peek = {};
    peek.live_ids = c.cks[0].ids;
    peek.part_bytes.assign((size_t) c.n_layers, {});
    for (const auto& s : dir.segments) {
        if (s.kind == 4 && s.layer == 0 && s.aux == kV2FIds) peek.live_ids_hash = s.ids_hash;
        if (s.kind == 1 && s.layer < c.n_layers && s.aux < 5) peek.part_bytes[s.layer][s.aux] = s.raw_total;
    }
    peek.ckpt_hashes.assign((size_t) c.n_ck, 0);
    for (const auto& s : dir.segments)
        if (s.kind == 5 && s.layer < c.n_ck && s.aux == kV2FIds) peek.ckpt_hashes[s.layer] = s.ids_hash;
    return true;
}
#include "conversation_file_v3.inc"
} // namespace
#endif // STRATA_SESSION_V2_FORMAT

// ---- public entries (moved here from the host file: one toolkit, one home) --------------------
#ifdef STRATA_SESSION_V2_FORMAT
bool session_file_verify(const std::string& path, std::string& error) {
    if (v3_is_archive(path)) return v3_verify_open(path, error);
    return v2_verify_structure(path, error);
}

bool session_file_saveinc(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                          const SessionFileIdentity& id, const std::vector<SessionSaveIncSlot>& keep, size_t& bytes,
                          std::string& error, const SessionWriteOptions& options, SessionStatus* status) {
    if (v3_is_archive(path)) {
        SessionStatus st;
        bool ok = false;
        try {
            ok = v3_saveinc(path, meta, kv, id, keep, bytes, error, options, st);
        } catch (const std::bad_alloc&) {
            error = "saveinc: out of memory while writing the V3 archive";
            st.error = SessionError::memory;
        }
        if (status) *status = st;
        return ok;
    }
    return saveinc_guarded(path, meta, kv, id, keep, bytes, error, options, status);
}

bool session_file_saveinc_peek(const std::string& path, const SessionFileIdentity& id, SessionSaveIncPeek& peek,
                               std::string& error) {
    if (v3_is_archive(path)) return v3_peek(path, id, peek, error);
    return saveinc_peek_impl(path, id, peek, error);
}

bool session_file_saveinc_needs_compaction(const std::string& path, std::string& reason) {
    if (v3_is_archive(path)) return v3_saveinc_needs_compaction_open(path, reason);
    return saveinc_compaction_check(path, reason);
}
#endif

#ifndef STRATA_SESSION_V2_FORMAT
bool session_file_saveinc_needs_compaction(const std::string&, std::string& reason) {
    reason.clear();
    return false;
}
#endif

#ifdef STRATA_SESSION_V2_FORMAT
// [giphoo] l2-exp v2.3 (M3): the parked read.  v2 only; a v1 file is refused with `error` naming
// the ordinary read, so the caller can fall back.
// [giphoo] l2-exp M3.1: partial restore by checkpoint. `prompt` is the incoming request's token
// ids; the deepest file checkpoint whose ids hash-match its prefix becomes the image (kv cut to
// the checkpoint's page-aligned position). `live_would_match` = the file's live state itself is
// a prefix (the caller then prefers a full restore); no matching checkpoint -> false with the
// reason in `error`.  The reads stay proportional to the matched prefix, not the file.
bool session_file_read_prefix(const std::string& path, const SessionFileIdentity& id,
                              const std::vector<int32_t>& prompt, SavedConversation& image,
                              size_t& bytes, std::string& error, const SessionReadLimits& limits,
                              const std::function<bool(uint64_t image_bytes, std::string& why)>& fits,
                              SessionStatus* status, bool* live_would_match,
                              SessionPrefixStats* prefix_stats) {
    SessionStatus st;
    bool ok = false, live = false;
    try {
        if (v3_is_archive(path)) {
            if (prefix_stats) prefix_stats->v3_archive = true;
            ok = v3_read_prefix(path, id, prompt, image, bytes, error, limits, fits, st, live,
                                prefix_stats);
        } else {
            ok = read_prefix_impl(path, id, prompt, image, bytes, error, limits, fits, st, live,
                                  prefix_stats);
        }
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while partially reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    if (live_would_match) *live_would_match = live;
    if (prefix_stats) prefix_stats->live_would_match = live;
    return ok;
}

bool session_file_read_parked(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                              size_t& bytes, std::string& error, const SessionReadLimits& limits,
                              const std::function<bool(uint64_t image_bytes, std::string& why)>& fits,
                              SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        st.error = SessionError::invalid;
        if (v3_is_archive(path)) {
            const std::function<bool(uint64_t, std::string&)>* fits_ptr = &fits;
            ok = v3_read(path, id, image, bytes, error, limits, fits_ptr, st);
            if (status) *status = st;
            return ok;
        }
        RawFile raw;
        uint64_t size = 0;
        if (!raw.open_session(path, size)) {
            st.error = raw.refused() ? SessionError::invalid : raw.kind();
            error = "session file: " + path + ": " + raw.error();
            return false;
        }
        if (size < kHeader + 16) { error = "session file: size " + std::to_string(size) + " is below the minimum"; return false; }
        const uint64_t max_size = session_read_max_file_bytes(limits);
        FileSource f(raw);
        if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
        if (limits.progress) f.progress.emplace(limits.progress);
        f.expected = size;
        uint8_t h[kHeader];
        if (!f.read(h, kHeader)) { st.error = f.kind(); error = "session file: header read error: " + f.error(); return false; }
        if (std::memcmp(h, kMagicV2, 8) != 0) {
            error = "session file: the parked read needs the v2 format (use the ordinary read)";
            return false;
        }
        uint64_t payload = 0;
        std::memcpy(&payload, h + 32, 8);
        if (payload > max_size || payload > (uint64_t) std::numeric_limits<size_t>::max()) {
            error = "session file: logical payload " + std::to_string(payload) + " exceeds this engine's limit";
            return false;
        }
        ok = read_v2_parked(raw, size, h, id, image, bytes, error, limits, fits, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}
#endif
