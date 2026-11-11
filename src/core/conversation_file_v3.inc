// Coarse-generation session archives: immutable base/delta segments, independent
// checkpoints and an atomically replaced manifest. Included inside the V2
// anonymous namespace so the composite reader reuses its validated block parser.
namespace fs = std::filesystem;

constexpr char kMagicV3[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x03'};
constexpr size_t kV3ManifestMax = 1u << 20;

bool session_v3_writes() {
    const char* e = std::getenv("STRATA_SESSION_V3");
    return e && *e && *e != '0';
}

std::string v3_manifest_path(const std::string& path) { return (fs::path(path) / "manifest").string(); }
std::string v3_segment_path(const std::string& path, const std::string& name) {
    return (fs::path(path) / "segments" / name).string();
}

bool v3_is_archive(const std::string& path) {
    std::error_code ec;
    const fs::path root(path);
    return fs::is_directory(root, ec) && !fs::is_symlink(root, ec) &&
           fs::is_regular_file(root / "manifest", ec) && !fs::is_symlink(root / "manifest", ec);
}

bool v3_write_path(const std::string& path, const SessionWriteOptions& opt) {
    if (!opt.use_v3 || !session_v3_writes()) return false;
    std::error_code ec;
    const auto st = fs::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory) return true;
    if (ec) return false;
    return fs::is_directory(st) && !fs::is_symlink(st);
}

uint64_t v3_env_u64(const char* key, uint64_t fallback) {
    const char* e = std::getenv(key);
    if (!e || !*e) return fallback;
    const long long value = std::strtoll(e, nullptr, 10);
    return value >= 0 ? (uint64_t) value : fallback;
}

std::string v3_hex(uint64_t value) {
    char out[17];
    std::snprintf(out, sizeof out, "%016llx", (unsigned long long) value);
    return out;
}

struct V3Manifest {
    SessionFileIdentity id;
    uint64_t payload = 0, generation = 0, file_bytes = 0;
    std::vector<std::string> data;        // base first, then deltas in commit order
    std::vector<std::string> checkpoints; // current keep order
};

bool v3_leaf_name(const std::string& name) {
    if (name.empty() || name.size() > 128 || name[0] == '.' || name == ".." ||
        name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name.find('\0') != std::string::npos) return false;
    for (unsigned char c : name)
        if (!(c == '-' || c == '_' || c == '.' || (c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) return false;
    return true;
}

void v3_put16(std::vector<uint8_t>& out, uint16_t value) {
    out.insert(out.end(), (const uint8_t*) &value, (const uint8_t*) &value + 2);
}
void v3_put32(std::vector<uint8_t>& out, uint32_t value) {
    out.insert(out.end(), (const uint8_t*) &value, (const uint8_t*) &value + 4);
}
void v3_put_name(std::vector<uint8_t>& out, const std::string& name) {
    v3_put16(out, (uint16_t) name.size());
    out.insert(out.end(), name.begin(), name.end());
}

std::vector<uint8_t> v3_manifest_bytes(const V3Manifest& man) {
    std::vector<uint8_t> out(kHeader, 0);
    std::memcpy(out.data(), kMagicV3, 8);
    const uint32_t version = 3, hsize = kHeader;
    std::memcpy(out.data() + 8, &version, 4);
    std::memcpy(out.data() + 12, &hsize, 4);
    std::memcpy(out.data() + 16, &man.id.model, 8);
    std::memcpy(out.data() + 24, &man.id.config, 8);
    std::memcpy(out.data() + 32, &man.payload, 8);
    std::memcpy(out.data() + 40, &man.generation, 8);
    const uint64_t hh = session_hash64(out.data(), 56, 0);
    std::memcpy(out.data() + 56, &hh, 8);
    v3_put32(out, (uint32_t) man.data.size());
    v3_put32(out, (uint32_t) man.checkpoints.size());
    for (const auto& name : man.data) v3_put_name(out, name);
    for (const auto& name : man.checkpoints) v3_put_name(out, name);
    const uint64_t bh = session_hash64(out.data() + kHeader, out.size() - kHeader, 0);
    out.insert(out.end(), (const uint8_t*) &bh, (const uint8_t*) &bh + 8);
    return out;
}

bool v3_read_manifest(const std::string& path, const SessionFileIdentity& id, V3Manifest& man,
                      std::string& error, bool check_identity = true) {
    RawFile raw;
    uint64_t size = 0;
    const std::string mp = v3_manifest_path(path);
    if (!raw.open_session(mp, size)) {
        error = "session v3: cannot open manifest: " + raw.error();
        return false;
    }
    if (size < kHeader + 8 + 8 || size > kV3ManifestMax) {
        error = "session v3: invalid manifest size";
        return false;
    }
    std::vector<uint8_t> buf((size_t) size);
    if (!raw.read_at(0, buf.data(), buf.size())) {
        error = "session v3: manifest read: " + raw.error();
        return false;
    }
    uint32_t version = 0, hsize = 0;
    uint64_t r1 = 0, hh = 0, bh = 0;
    std::memcpy(&version, buf.data() + 8, 4);
    std::memcpy(&hsize, buf.data() + 12, 4);
    std::memcpy(&man.generation, buf.data() + 40, 8);
    std::memcpy(&r1, buf.data() + 48, 8);
    std::memcpy(&hh, buf.data() + 56, 8);
    std::memcpy(&bh, buf.data() + buf.size() - 8, 8);
    if (std::memcmp(buf.data(), kMagicV3, 8) != 0 || version != 3 || hsize != kHeader || r1 ||
        hh != session_hash64(buf.data(), 56, 0) ||
        bh != session_hash64(buf.data() + kHeader, buf.size() - kHeader - 8, 0)) {
        error = "session v3: manifest header or checksum mismatch";
        return false;
    }
    std::memcpy(&man.id.model, buf.data() + 16, 8);
    std::memcpy(&man.id.config, buf.data() + 24, 8);
    std::memcpy(&man.payload, buf.data() + 32, 8);
    if (check_identity && (man.id.model != id.model || man.id.config != id.config)) {
        error = "session v3: saved with another model or configuration";
        return false;
    }
    if (!man.generation) { error = "session v3: invalid generation"; return false; }
    size_t at = kHeader;
    auto take = [&](void* p, size_t n) {
        if (at + n > buf.size() - 8) return false;
        std::memcpy(p, buf.data() + at, n);
        at += n;
        return true;
    };
    uint32_t nd = 0, nc = 0;
    if (!take(&nd, 4) || !take(&nc, 4) || !nd || nd > 129 || nc > 256) {
        error = "session v3: invalid manifest counts";
        return false;
    }
    auto names = [&](std::vector<std::string>& out, uint32_t n) {
        out.clear();
        out.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint16_t len = 0;
            if (!take(&len, 2) || at + len > buf.size() - 8) return false;
            std::string name((const char*) buf.data() + at, len);
            at += len;
            if (!v3_leaf_name(name)) { error = "session v3: unsafe segment name"; return false; }
            out.push_back(std::move(name));
        }
        return true;
    };
    if (!names(man.data, nd) || !names(man.checkpoints, nc) || at != buf.size() - 8) {
        if (error.empty()) error = "session v3: manifest ends early or has trailing bytes";
        return false;
    }
    man.file_bytes = size;
    return true;
}

bool v3_write_manifest(const std::string& path, const V3Manifest& man, const SessionWriteOptions& opt,
                       std::string& error, SessionStatus& st) {
    const std::vector<uint8_t> bytes = v3_manifest_bytes(man);
    Publisher pub;
    const std::string mp = v3_manifest_path(path);
    if (!pub.open(mp, error)) { st.error = pub.kind(); return false; }
    RawFile raw;
    if (!pub.create_temp(raw, error)) { st.error = raw.kind(); return false; }
    bool ok = raw.write_all(bytes.data(), bytes.size());
    if (ok && opt.phase) opt.phase("flush", bytes.size());
    if (ok && opt.durable) ok = raw.sync();
    ok = raw.close() && ok;
    if (!ok) { st.error = raw.kind(); error = "session v3: manifest write: " + raw.error(); return false; }
    if (opt.phase) opt.phase("publish", bytes.size());
    std::string why;
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;
    if (const int e = injected(fault, "manifest_rename")) {
        error = "session v3: manifest publish: " + std::string(std::strerror(e));
        st.error = errno_kind(e);
        return false;
    }
    if (!pub.publish(why, opt.durable, st, fault)) {
        error = "session v3: manifest publish: " + why;
        return false;
    }
    return true;
}

using V3Emit = std::function<void(V2Out&)>;

uint64_t v3_emit_size(const V3Emit& emit) {
    V2Out sizing(nullptr, v2_block_target());
    emit(sizing);
    return sizing.raw_total();
}

bool v3_write_segment(const std::string& path, const std::string& name, const SessionFileIdentity& id,
                      const V3Emit& emit, const SessionWriteOptions& opt, std::vector<std::string>& created,
                      uint64_t& payload, uint64_t& file_bytes, std::string& error, SessionStatus& st) {
    payload = v3_emit_size(emit);
    const std::string sp = v3_segment_path(path, name);
    Publisher pub;
    if (!pub.open(sp, error)) { st.error = pub.kind(); return false; }
    RawFile raw;
    if (!pub.create_temp(raw, error)) { st.error = raw.kind(); return false; }
    V2FreshIo io(raw);
    V2Out out(&io, v2_block_target());
    uint8_t h[kHeader];
    v2_header_bytes(h, id, payload);
    bool ok = io.write_all(h, kHeader);
    if (ok) { emit(out); out.finish(); ok = out.ok(); }
    uint64_t total = 0;
    if (ok) {
        std::vector<uint8_t> d;
        v2_dir_bytes(out.segments(), out.blocks(), payload, d);
        std::vector<uint8_t> ft(16);
        const uint64_t dir_off = out.data_end();
        std::memcpy(ft.data(), kMagicV2, 8);
        std::memcpy(ft.data() + 8, &dir_off, 8);
        ok = io.write_all(d.data(), d.size()) && io.write_all(ft.data(), 16);
        total = dir_off + d.size() + 16;
    }
    if (ok && opt.phase) opt.phase("flush", total);
    if (ok && opt.durable) ok = io.sync();
    ok = raw.close() && ok;
    if (!ok) {
        st.error = out.read_failed() ? SessionError::io : raw.kind();
        error = "session v3: writing segment " + name + ": " +
                (out.read_failed() ? std::string("K/V source read") : io.error());
        return false;
    }
    if (opt.phase) opt.phase("publish", total);
    std::string why;
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;
    if (!pub.publish(why, opt.durable, st, fault)) {
        error = "session v3: publishing segment " + name + ": " + why;
        return false;
    }
    created.push_back(name);
    file_bytes = total;
    return true;
}

void v3_remove_segments(const std::string& path, const std::vector<std::string>& names) {
    std::error_code ec;
    for (const auto& name : names) if (v3_leaf_name(name)) fs::remove(v3_segment_path(path, name), ec);
}

std::vector<std::string> v3_all_names(const V3Manifest& man) {
    std::vector<std::string> out = man.data;
    out.insert(out.end(), man.checkpoints.begin(), man.checkpoints.end());
    return out;
}

void v3_cleanup_superseded(const std::string& path, const V3Manifest& old, const V3Manifest& current) {
    (void) old;
    std::vector<std::string> keep = v3_all_names(current);
    keep.push_back("counts-" + v3_hex(current.generation) + ".v2");
    std::error_code ec;
    const fs::path dir = fs::path(path) / "segments";
    for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->is_symlink(ec)) continue;
        const std::string name = it->path().filename().string();
        if (std::find(keep.begin(), keep.end(), name) == keep.end()) fs::remove(it->path(), ec);
    }
}

uint64_t v3_disk_bytes(const std::string& path) {
    std::error_code ec;
    if (!fs::is_directory(path, ec)) {
        const uint64_t n = fs::file_size(path, ec);
        return ec ? 0 : n;
    }
    uint64_t total = 0;
    for (fs::recursive_directory_iterator it(path, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || it->is_symlink(ec)) continue;
        total += it->file_size(ec);
        if (ec) return total;
    }
    return total;
}

struct V3Part {
    std::string name;
    RawFile raw;
    V2Dir dir;
    uint64_t size = 0, payload = 0, vbase = 0;
};

struct V3Archive : V2Io {
    V3Manifest man;
    std::vector<std::unique_ptr<V3Part>> parts;
    V2Dir composite;
    uint64_t total_bytes = 0;
    std::string err;

    const V3Part* find(const std::string& name) const {
        for (const auto& p : parts) if (p->name == name) return p.get();
        return nullptr;
    }
    bool read_at(uint64_t off, void* dst, size_t n) override {
        for (auto& p : parts) {
            if (off < p->vbase || off >= p->vbase + p->size) continue;
            if (off + n > p->vbase + p->size) { err = "virtual block crosses a segment file"; return false; }
            if (p->raw.read_at(off - p->vbase, (uint8_t*) dst, n)) return true;
            err = p->name + ": " + p->raw.error();
            return false;
        }
        err = "virtual block offset is not mapped";
        return false;
    }
    int block_fd(uint64_t off, uint64_t& local) override {
        for (auto& p : parts) {
            if (off < p->vbase || off >= p->vbase + p->size) continue;
            local = off - p->vbase;
            return p->raw.pread_fd();
        }
        return -1;
    }
    bool write_all(const void*, size_t) override { err = "read-only composite"; return false; }
    bool sync() override { return true; }
    std::string error() const override { return err; }
};

bool v3_add_part(V3Archive& archive, const std::string& root, const std::string& name,
                 const SessionFileIdentity& id, uint64_t& vbase, std::string& error) {
    auto part = std::make_unique<V3Part>();
    part->name = name;
    if (!part->raw.open_session(v3_segment_path(root, name), part->size)) {
        error = "session v3: cannot open segment " + name + ": " + part->raw.error();
        return false;
    }
    if (part->size < kHeader + 16) { error = "session v3: segment too small: " + name; return false; }
    uint8_t h[kHeader];
    if (!part->raw.read_at(0, h, kHeader)) { error = "session v3: segment header: " + name; return false; }
    uint32_t version = 0, hsize = 0;
    uint64_t model = 0, config = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8);
    std::memcpy(&config, h + 24, 8);
    std::memcpy(&part->payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8);
    std::memcpy(&r1, h + 48, 8);
    std::memcpy(&hh, h + 56, 8);
    if (std::memcmp(h, kMagicV2, 8) != 0 || version != 2 || hsize != kHeader || r0 || r1 ||
        hh != session_hash64(h, 56, 0) || model != id.model || config != id.config) {
        error = "session v3: segment identity or header mismatch: " + name;
        return false;
    }
    V2FreshIo io(part->raw);
    if (!v2_read_dir(io, part->size, part->dir, error)) {
        error = "session v3: " + name + ": " + error;
        return false;
    }
    if (part->dir.payload_raw != part->payload) {
        error = "session v3: segment payload disagrees with its directory: " + name;
        return false;
    }
    part->vbase = vbase;
    if (vbase > UINT64_MAX - part->size - 16) { error = "session v3: virtual address overflow"; return false; }
    vbase += part->size + 16;
    archive.total_bytes += part->size;
    archive.parts.push_back(std::move(part));
    return true;
}

bool v3_load_archive(const std::string& path, const SessionFileIdentity& id, V3Archive& archive,
                     std::string& error) {
    if (!v3_read_manifest(path, id, archive.man, error)) return false;
    uint64_t vbase = kHeader;
    const std::string counts_name = "counts-" + v3_hex(archive.man.generation) + ".v2";
    if (!v3_add_part(archive, path, counts_name, id, vbase, error)) return false;
    for (const auto& name : archive.man.data)
        if (!v3_add_part(archive, path, name, id, vbase, error)) return false;
    for (const auto& name : archive.man.checkpoints)
        if (!v3_add_part(archive, path, name, id, vbase, error)) return false;
    archive.total_bytes += archive.man.file_bytes;

    using Key = std::tuple<int, int, int>;
    struct Ref { size_t part, seg; };
    std::map<Key, std::vector<Ref>> refs;
    auto add_refs = [&](size_t pi, bool data) {
        const auto& segs = archive.parts[pi]->dir.segments;
        for (size_t si = 0; si < segs.size(); ++si) {
            const auto& s = segs[si];
            if (data && s.kind != 4 && s.kind != 1) {
                error = "session v3: data segment has an invalid role";
                return false;
            }
            if (!data && (s.kind != 0 || segs.size() != 1)) {
                error = "session v3: counts segment has an invalid role";
                return false;
            }
            const Key key{s.kind, s.layer, (int) s.aux};
            if (data && s.kind == 1 && s.aux == 4) refs[key] = {{pi, si}}; // pooled is a full replacement
            else refs[key].push_back({pi, si});
        }
        return true;
    };
    if (!add_refs(0, false)) return false;
    for (size_t i = 0; i < archive.man.data.size(); ++i)
        if (!add_refs(1 + i, true)) return false;

    archive.composite.payload_raw = archive.man.payload;
    auto emit = [&](const Key& key, int owner = -1) {
        auto found = refs.find(key);
        if (found == refs.end()) return;
        V2SegRec out;
        out.kind = (uint8_t) std::get<0>(key);
        out.layer = owner >= 0 ? (uint16_t) owner : (uint16_t) std::get<1>(key);
        out.aux = (uint32_t) std::get<2>(key);
        for (const auto& ref : found->second) {
            const auto& part = archive.parts[ref.part];
            const auto& seg = part->dir.segments[ref.seg];
            out.raw_total = sat_add(out.raw_total, seg.raw_total);
            out.ids_hash = seg.ids_hash;
            for (uint32_t bi : seg.idx) {
                V2BlockRec block = part->dir.blocks[bi];
                if (block.offset > UINT64_MAX - part->vbase) {
                    archive.err = "session v3: block address overflow";
                    return;
                }
                block.offset += part->vbase;
                out.idx.push_back((uint32_t) archive.composite.blocks.size());
                archive.composite.blocks.push_back(block);
            }
        }
        archive.composite.segments.push_back(std::move(out));
    };
    emit({0, 0, 0});
    for (int f = 0; f < kV2FCount; ++f) emit({4, 0, f});
    const size_t ck_base = 1 + archive.man.data.size();
    for (size_t i = 0; i < archive.man.checkpoints.size(); ++i) {
        const auto& part = archive.parts[ck_base + i];
        bool seen[kV2FCount] = {};
        for (size_t si = 0; si < part->dir.segments.size(); ++si) {
            const auto& seg = part->dir.segments[si];
            if (seg.kind != 5 || seg.layer != 0 || seg.aux >= kV2FCount || seen[seg.aux]) {
                error = "session v3: invalid checkpoint segment: " + part->name;
                return false;
            }
            seen[seg.aux] = true;
            refs[{5, 0, (int) seg.aux}] = {{ck_base + i, si}};
        }
        for (int f = 0; f < kV2FCount; ++f) {
            if (!seen[f]) { error = "session v3: incomplete checkpoint segment: " + part->name; return false; }
            emit({5, 0, f}, (int) i);
        }
    }
    for (const auto& [key, unused] : refs) {
        (void) unused;
        if (std::get<0>(key) == 1) emit(key);
    }
    if (archive.composite.segments.empty() || archive.composite.segments[0].kind != 0 || !archive.err.empty()) {
        error = archive.err.empty() ? "session v3: composite has no counts" : archive.err;
        return false;
    }
    uint64_t payload = 0;
    for (const auto& seg : archive.composite.segments) payload = sat_add(payload, seg.raw_total);
    if (payload != archive.man.payload) {
        error = "session v3: manifest payload does not match the composite segments";
        return false;
    }
    return true;
}

bool v3_counts(V3Archive& archive, V2Counts& counts, std::string& error) {
    SessionReadLimits open;
    V2In in{&archive, &archive.composite, 0, archive.composite.payload_raw, error, open};
    return v2_get_counts(in, counts);
}

bool v3_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
             size_t& bytes, std::string& error, const SessionReadLimits& limits,
             const std::function<bool(uint64_t, std::string&)>* fits, SessionStatus& st) {
    V3Archive archive;
    if (!v3_load_archive(path, id, archive, error)) { st.error = SessionError::invalid; return false; }
    const uint64_t max_size = session_read_max_file_bytes(limits);
    if (archive.man.payload > max_size || archive.man.payload > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: logical payload " + std::to_string(archive.man.payload) + " exceeds this engine's limit";
        st.error = SessionError::invalid;
        return false;
    }
    return read_v2_loaded_common(archive, archive.total_bytes, archive.composite, archive.man.payload,
                                 id, image, bytes, error, limits, fits, st);
}

bool v3_tail_needs_compaction(const V3Archive& archive, std::string& reason);

bool v3_verify(const std::string& path, const SessionFileIdentity& id, std::string& error) {
    V3Archive archive;
    if (!v3_load_archive(path, id, archive, error)) return false;
    return v2_verify_loaded(archive, archive.composite, error);
}

bool v3_peek(const std::string& path, const SessionFileIdentity& id, SessionSaveIncPeek& peek,
             std::string& error) {
    V3Archive archive;
    if (!v3_load_archive(path, id, archive, error)) { error = "saveinc: " + error; return false; }
    V2Counts counts;
    std::string e2;
    if (!v3_counts(archive, counts, e2)) { error = "saveinc: " + e2; return false; }
    peek = {};
    peek.live_ids = counts.cks[0].ids;
    peek.part_bytes.assign((size_t) counts.n_layers, {});
    peek.ckpt_hashes.assign((size_t) counts.n_ck, 0);
    for (const auto& seg : archive.composite.segments) {
        if (seg.kind == 4 && seg.layer == 0 && seg.aux == kV2FIds) peek.live_ids_hash = seg.ids_hash;
        if (seg.kind == 5 && seg.layer < counts.n_ck && seg.aux == kV2FIds) peek.ckpt_hashes[seg.layer] = seg.ids_hash;
        if (seg.kind == 1 && seg.layer < counts.n_layers && seg.aux < 5)
            peek.part_bytes[seg.layer][seg.aux] = seg.raw_total;
    }
    std::string tail_reason;
    peek.consolidate_tail = v3_tail_needs_compaction(archive, tail_reason);
    if (peek.consolidate_tail && !archive.man.data.empty()) {
        peek.base_part_bytes.assign((size_t) counts.n_layers, {});
        if (const V3Part* base = archive.find(archive.man.data[0])) {
            for (const auto& seg : base->dir.segments)
                if (seg.kind == 1 && seg.layer < counts.n_layers && seg.aux < 5)
                    peek.base_part_bytes[seg.layer][seg.aux] = seg.raw_total;
        }
    }
    return true;
}

bool v3_read_prefix(const std::string& path, const SessionFileIdentity& id,
                    const std::vector<int32_t>& prompt, SavedConversation& image, size_t& bytes,
                    std::string& error, const SessionReadLimits& limits,
                    const std::function<bool(uint64_t, std::string&)>& fits, SessionStatus& st,
                    bool& live_would_match, SessionPrefixStats* prefix_stats) {
    V3Archive archive;
    if (!v3_load_archive(path, id, archive, error)) { st.error = SessionError::invalid; return false; }
    return read_prefix_loaded(archive, archive.total_bytes, archive.composite, id, prompt, image, bytes,
                              error, limits, fits, st, live_would_match, prefix_stats);
}

void v3_put_kv(V2Out& out, const SavedConversation& image, const std::vector<SessionKvSource>* sources) {
    if (sources) {
        for (size_t l = 0; l < sources->size(); ++l) {
            const auto& k = (*sources)[l];
            for (size_t p = 0; p < 5; ++p) {
                if (!k.sizes[p]) continue;
                out.open_segment(1, (uint16_t) l, (uint32_t) p);
                out.src_bytes(k, p, 0, k.sizes[p]);
            }
        }
        return;
    }
    for (size_t l = 0; l < image.kv.size(); ++l) {
        const auto& k = image.kv[l];
        for (size_t p = 0; p < 5; ++p) {
            const size_t n = v2_img_part_size(k, p);
            if (!n) continue;
            out.open_segment(1, (uint16_t) l, (uint32_t) p);
            out.img_bytes(*v2_img_part(const_cast<ConversationKv&> (k), p), 0, n);
        }
    }
}

bool v3_prepare_root(const std::string& path, std::string& error, SessionStatus& st) {
    std::error_code ec;
    const fs::path root(path);
    if (fs::exists(root, ec)) {
        if (!fs::is_directory(root, ec) || fs::is_symlink(root, ec)) {
            error = "session v3: archive path is not a real directory";
            st.error = SessionError::invalid;
            return false;
        }
    } else if (!fs::create_directory(root, ec) && ec) {
        error = "session v3: create archive: " + ec.message();
        st.error = SessionError::io;
        return false;
    }
    if (!fs::create_directory(root / "segments", ec) && ec) {
        error = "session v3: create segments directory: " + ec.message();
        st.error = SessionError::io;
        return false;
    }
    return true;
}

bool v3_space_preflight(const std::string& path, uint64_t need, const SessionWriteOptions& opt,
                        std::string& error, SessionStatus& st) {
    if (!opt.min_free_bytes) return true;
    uint64_t avail = 0;
    if (!free_space(path, avail)) { error = "session v3: free-space probe failed"; st.error = SessionError::io; return false; }
    const uint64_t worst = sat_add(need, need / kV2BlkMin * 56 + (1u << 20));
    if (avail < worst || avail - worst < opt.min_free_bytes) {
        error = "session v3: not enough disk space for the new generation";
        st.error = SessionError::storage;
        return false;
    }
    return true;
}

uint64_t v3_next_generation(const V3Manifest* old) {
    if (old && old->generation < UINT64_MAX) return old->generation + 1;
    const uint64_t value = random64();
    return value ? value : 1;
}

bool v3_full_write(const std::string& path, const SavedConversation& image,
                   const std::vector<SessionKvSource>* sources, const SessionFileIdentity& id,
                   size_t& bytes, std::string& error, const SessionWriteOptions& opt, SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;
    if (has_stage_parts(image)) { error = "session v3: layer-split state cannot be saved"; return false; }
    if (sources && !image.kv.empty()) { error = "session v3: K/V given both in image and sources"; return false; }
    V3Manifest old;
    const bool had_old = v3_is_archive(path) && v3_read_manifest(path, id, old, error);
    if (v3_is_archive(path) && !had_old) { st.error = SessionError::invalid; return false; }
    if (!v3_prepare_root(path, error, st)) return false;
    V3Manifest current;
    current.id = id;
    current.generation = v3_next_generation(had_old ? &old : nullptr);
    const std::string suffix = v3_hex(current.generation);
    const std::string base = "base-" + suffix + ".v2";
    const std::string counts = "counts-" + suffix + ".v2";
    current.data = {base};

    const V3Emit emit_base = [&](V2Out& out) {
        v2_stream_state(out, image.live, 4, 0);
        v3_put_kv(out, image, sources);
    };
    uint64_t estimate = v3_emit_size(emit_base);
    const V3Emit emit_counts = [&](V2Out& out) {
        out.open_segment(0);
        v2_put_counts(out, image, sources, image.checkpoints.size());
    };
    estimate = sat_add(estimate, v3_emit_size(emit_counts));
    for (const auto& ck : image.checkpoints) {
        estimate = sat_add(estimate, v3_emit_size([&](V2Out& out) { v2_stream_state(out, ck, 5, 0); }));
    }
    if (!v3_space_preflight(path, estimate, opt, error, st)) return false;
    SessionWriteOptions part_opt = opt;
    part_opt.min_free_bytes = 0;
    std::vector<std::string> created;
    uint64_t file_bytes = 0, part_payload = 0;
    if (!v3_write_segment(path, base, id, emit_base, part_opt, created, part_payload, file_bytes, error, st)) {
        v3_remove_segments(path, created); return false;
    }
    current.payload = part_payload;
    for (size_t i = 0; i < image.checkpoints.size(); ++i) {
        const auto& ck = image.checkpoints[i];
        const std::string name = "ckpt-" + v3_hex(v2_ids_hash(ck.ids)) + "-" + suffix + ".v2";
        const V3Emit emit_ck = [&](V2Out& out) { v2_stream_state(out, ck, 5, 0); };
        if (!v3_write_segment(path, name, id, emit_ck, part_opt, created, part_payload, file_bytes, error, st)) {
            v3_remove_segments(path, created); return false;
        }
        current.checkpoints.push_back(name);
        current.payload = sat_add(current.payload, part_payload);
    }
    if (!v3_write_segment(path, counts, id, emit_counts, part_opt, created, part_payload, file_bytes, error, st)) {
        v3_remove_segments(path, created); return false;
    }
    current.payload = sat_add(current.payload, part_payload);
    if (!v3_write_manifest(path, current, opt, error, st)) {
        v3_remove_segments(path, created); return false;
    }
    std::string verify_error;
    if (!v3_verify(path, id, verify_error)) {
        SessionWriteOptions rollback_opt = opt;
        rollback_opt.fault = {};
        if (had_old) {
            std::string rollback_error;
            SessionStatus rollback_status;
            v3_write_manifest(path, old, rollback_opt, rollback_error, rollback_status);
        } else {
            std::error_code ec;
            fs::remove(v3_manifest_path(path), ec);
        }
        v3_remove_segments(path, created);
        error = "session v3: written but " + verify_error;
        st.error = SessionError::invalid;
        return false;
    }
    if (had_old) v3_cleanup_superseded(path, old, current);
    st.error = SessionError::none;
    bytes = (size_t) v3_disk_bytes(path);
    return true;
}

bool v3_tail_needs_compaction(const V3Archive& archive, std::string& reason) {
    const uint64_t max_deltas = v3_env_u64("STRATA_V3_MAX_DELTAS", 16);
    const uint64_t max_bytes = v3_env_u64("STRATA_V3_COMPACT_TAIL_BYTES", 1ull << 30);
    uint64_t tail = 0;
    for (size_t i = 1; i < archive.man.data.size(); ++i) {
        const V3Part* part = archive.find(archive.man.data[i]);
        if (part) tail = sat_add(tail, part->size);
    }
    const size_t deltas = archive.man.data.size() - 1;
    if ((max_deltas && deltas >= max_deltas) || (max_bytes && tail >= max_bytes)) {
        reason = "saveinc: v3 tail compaction required (deltas=" + std::to_string(deltas) +
                 ", bytes=" + std::to_string(tail) + ")";
        return true;
    }
    return false;
}

uint64_t v3_merged_saveinc_payload(const V3Manifest& current, const V3Archive& old,
                                   const V3Archive& delta, const SavedConversation& meta,
                                   const std::vector<SessionSaveIncSlot>& keep, uint64_t counts_payload) {
    auto data_part = [&](const std::string& name) -> const V3Part* {
        if (const V3Part* part = delta.find(name)) return part;
        return old.find(name);
    };
    uint64_t payload = counts_payload;
    for (size_t i = 0; i < current.data.size(); ++i) {
        const V3Part* part = data_part(current.data[i]);
        if (!part) continue;
        for (const auto& seg : part->dir.segments) {
            if (seg.kind == 1 && seg.aux == 4) {
                bool superseded = false;
                for (size_t j = i + 1; j < current.data.size() && !superseded; ++j) {
                    const V3Part* later = data_part(current.data[j]);
                    if (!later) continue;
                    for (const auto& candidate : later->dir.segments)
                        if (candidate.kind == 1 && candidate.layer == seg.layer && candidate.aux == 4) {
                            superseded = true;
                            break;
                        }
                }
                if (superseded) continue;
            }
            payload = sat_add(payload, seg.raw_total);
        }
    }
    for (size_t i = 0; i < keep.size(); ++i) {
        if (keep[i].reuse >= 0) {
            const V3Part* part = old.find(current.checkpoints[i]);
            if (part) payload = sat_add(payload, part->payload);
        } else {
            const auto& ck = meta.checkpoints[(size_t) keep[i].fresh];
            payload = sat_add(payload, v3_emit_size([&](V2Out& out) { v2_stream_state(out, ck, 5, 0); }));
        }
    }
    return payload;
}

bool v3_saveinc(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                const SessionFileIdentity& id, const std::vector<SessionSaveIncSlot>& keep, size_t& bytes,
                std::string& error, const SessionWriteOptions& opt, SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;
    if (has_stage_parts(meta)) { error = "saveinc: layer-split state cannot be saved"; return false; }
    if (!meta.kv.empty()) { error = "saveinc: meta carries K/V"; return false; }
    V3Archive archive;
    if (!v3_load_archive(path, id, archive, error)) { error = "saveinc: " + error; return false; }
    std::string tail_reason;
    const bool consolidate_tail = v3_tail_needs_compaction(archive, tail_reason);
    V2Counts old;
    std::string e2;
    if (!v3_counts(archive, old, e2)) { error = "saveinc: " + e2; return false; }
    const auto& live = meta.live;
    const auto& old_live = old.cks[0];
    if (old_live.ids > live.ids.size()) { error = "saveinc: the live ids shrank"; return false; }
    const V2SegRec* old_ids = nullptr;
    for (const auto& seg : archive.composite.segments)
        if (seg.kind == 4 && seg.layer == 0 && seg.aux == kV2FIds) old_ids = &seg;
    if (!old_ids) { error = "saveinc: old archive has no live ids"; return false; }
    if (old_live.ids && session_hash64(live.ids.data(), (size_t) old_live.ids * sizeof(int32_t), 0) != old_ids->ids_hash) {
        error = "saveinc: the live ids diverge from the archive";
        return false;
    }
    size_t fresh_count = 0;
    for (const auto& slot : keep) {
        const bool reuse = slot.reuse >= 0 && slot.fresh < 0;
        const bool fresh = slot.fresh >= 0 && slot.reuse < 0;
        if (reuse == fresh) { error = "saveinc: invalid V3 keep slot"; return false; }
        if (reuse && (size_t) slot.reuse >= archive.man.checkpoints.size()) {
            error = "saveinc: reused checkpoint is outside the old keep set"; return false;
        }
        if (fresh) {
            ++fresh_count;
            if ((size_t) slot.fresh >= meta.checkpoints.size()) {
                error = "saveinc: fresh checkpoint is outside meta"; return false;
            }
        }
    }
    if (fresh_count != meta.checkpoints.size()) {
        error = "saveinc: meta.checkpoints must contain exactly the fresh keep slots"; return false;
    }
    if (kv.size() != old.layers.size()) { error = "saveinc: K/V layer count changed"; return false; }
    for (size_t l = 0; l < kv.size(); ++l) {
        const auto& k = kv[l];
        const auto& ol = old.layers[l];
        const int64_t nf[7] = {(int64_t) k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim};
        for (int a : {0, 2, 3, 4, 6}) if (nf[a] != ol.f[a]) {
            error = "saveinc: K/V header changed at layer " + std::to_string(l); return false;
        }
        if (k.cells < ol.f[1] || k.pooled_rows < ol.f[5]) {
            error = "saveinc: K/V extent shrank at layer " + std::to_string(l); return false;
        }
        for (size_t p = 0; p < 5; ++p) if (k.sizes[p] < ol.parts[p]) {
            error = "saveinc: K/V part shrank at layer " + std::to_string(l); return false;
        }
    }

    V3Manifest current;
    current.id = id;
    current.generation = v3_next_generation(&archive.man);
    const std::string suffix = v3_hex(current.generation);
    const std::string delta = "delta-" + suffix + ".v2";
    const std::string counts_name = "counts-" + suffix + ".v2";
    std::array<uint64_t, kV2FCount> base_live{};
    std::vector<std::array<uint64_t, 5>> base_parts(kv.size());
    if (consolidate_tail) {
        const V3Part* base = archive.man.data.empty() ? nullptr : archive.find(archive.man.data[0]);
        if (!base) { error = "saveinc: V3 base segment is missing"; return false; }
        for (const auto& seg : base->dir.segments) {
            if (seg.kind == 4 && seg.layer == 0 && seg.aux < kV2FCount) {
                const uint64_t esz = seg.aux == kV2FIds ? sizeof(int32_t) :
                                     seg.aux == kV2FImgs ? sizeof(ConversationImageKey) : 1;
                base_live[seg.aux] = seg.raw_total / esz;
            }
            if (seg.kind == 1 && seg.layer < kv.size() && seg.aux < 5)
                base_parts[seg.layer][seg.aux] = seg.raw_total;
        }
        current.data = {archive.man.data[0]};
    } else {
        current.data = archive.man.data;
    }
    current.data.push_back(delta);
    const bool split = strata::core::l2two::hook_split(meta.live, old_live);
    const V3Emit emit_delta = [&](V2Out& out) {
        auto field = [&](uint32_t which, const void* ptr, uint64_t fresh, uint64_t current_old, size_t esz) {
            const uint64_t n = strata::core::l2two::hook_field(split, fresh, current_old);
            const uint64_t from = consolidate_tail ? base_live[which] : current_old;
            out.open_segment(4, 0, which);
            if (which == kV2FIds)
                out.stamp_ids_hash(session_hash64(live.ids.data(), (size_t) n * sizeof(int32_t), 0));
            if (n > from) out.bytes((const uint8_t*) ptr + from * esz, (size_t) ((n - from) * esz));
        };
        field(kV2FIds, live.ids.data(), live.ids.size(), old_live.ids, sizeof(int32_t));
        field(kV2FImgs, live.imgs.data(), live.imgs.size(), old_live.imgs, sizeof(ConversationImageKey));
        field(kV2FGdn, live.gdn.data(), live.gdn.size(), old_live.gdn, 1);
        field(kV2FPle, live.ple.data(), live.ple.size(), old_live.ple, 1);
        field(kV2FTails, live.tails.data(), live.tails.size(), old_live.tails, 1);
        field(kV2FDead, live.dead.data(), live.dead.size(), old_live.dead, 1);
        field(kV2FBlockPos, live.block_pos.data(), live.block_pos.size(), old_live.block_pos, 1);
        for (size_t l = 0; l < kv.size(); ++l) {
            const auto& k = kv[l];
            for (size_t p = 0; p < 5; ++p) {
                const uint64_t current_old = old.layers[l].parts[p], new_n = k.sizes[p];
                if (p < 4) {
                    const uint64_t from = consolidate_tail ? base_parts[l][p] : current_old;
                    if (new_n <= from) continue;
                    out.open_segment(1, (uint16_t) l, (uint32_t) p);
                    out.src_bytes(k, p, from, new_n - from);
                } else if (new_n) {
                    out.open_segment(1, (uint16_t) l, (uint32_t) p);
                    out.src_bytes(k, p, 0, new_n);
                }
            }
        }
    };
    const V3Emit emit_counts = [&](V2Out& out) {
        out.open_segment(0);
        for (int64_t g : meta.geometry) out.i64(g);
        out.i64(meta.layer_lo); out.i64(meta.layer_hi); out.u64(meta.cvec ? 1 : 0);
        out.u64(keep.size());
        auto put_ck = [&](const ConversationCheckpoint& ck) {
            out.u64(ck.ids.size()); out.u64(ck.imgs.size()); out.u64(ck.used);
            out.u64(ck.gdn.size()); out.u64(ck.ple.size()); out.u64(ck.tails.size());
            out.u64(ck.dead.size()); out.u64(ck.block_pos.size());
        };
        strata::core::l2two::hook_put_counts(put_ck, split, meta.live, old_live);
        for (const auto& slot : keep) {
            if (slot.reuse >= 0) {
                const auto& ck = old.cks[(size_t) slot.reuse + 1];
                out.u64(ck.ids); out.u64(ck.imgs); out.u64(ck.used); out.u64(ck.gdn);
                out.u64(ck.ple); out.u64(ck.tails); out.u64(ck.dead); out.u64(ck.block_pos);
            } else put_ck(meta.checkpoints[(size_t) slot.fresh]);
        }
        out.u64(kv.size());
        for (const auto& k : kv) {
            put_kv_header(out, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
            for (size_t p = 0; p < 5; ++p) out.u64(k.sizes[p]);
        }
    };
    uint64_t estimate = sat_add(v3_emit_size(emit_delta), v3_emit_size(emit_counts));
    for (const auto& ck : meta.checkpoints)
        estimate = sat_add(estimate, v3_emit_size([&](V2Out& out) { v2_stream_state(out, ck, 5, 0); }));
    if (!v3_space_preflight(path, estimate, opt, error, st)) return false;
    SessionWriteOptions part_opt = opt;
    part_opt.min_free_bytes = 0;
    std::vector<std::string> created;
    uint64_t file_bytes = 0, part_payload = 0;
    if (!v3_write_segment(path, delta, id, emit_delta, part_opt, created, part_payload, file_bytes, error, st)) {
        v3_remove_segments(path, created); return false;
    }
    current.payload = part_payload;
    for (const auto& name : archive.man.data) {
        const V3Part* part = archive.find(name);
        if (!part) { error = "saveinc: old data segment disappeared"; v3_remove_segments(path, created); return false; }
        current.payload = sat_add(current.payload, part->payload);
    }
    std::vector<std::string> fresh_names(meta.checkpoints.size());
    for (size_t i = 0; i < meta.checkpoints.size(); ++i) {
        const auto& ck = meta.checkpoints[i];
        fresh_names[i] = "ckpt-" + v3_hex(v2_ids_hash(ck.ids)) + "-" + suffix + ".v2";
        const V3Emit emit_ck = [&](V2Out& out) { v2_stream_state(out, ck, 5, 0); };
        if (!v3_write_segment(path, fresh_names[i], id, emit_ck, part_opt, created, part_payload, file_bytes, error, st)) {
            v3_remove_segments(path, created); return false;
        }
    }
    for (const auto& slot : keep) {
        if (slot.reuse >= 0) {
            const std::string& name = archive.man.checkpoints[(size_t) slot.reuse];
            current.checkpoints.push_back(name);
            const V3Part* part = archive.find(name);
            if (!part) { error = "saveinc: old checkpoint disappeared"; v3_remove_segments(path, created); return false; }
            current.payload = sat_add(current.payload, part->payload);
        } else {
            const auto& ck = meta.checkpoints[(size_t) slot.fresh];
            current.checkpoints.push_back(fresh_names[(size_t) slot.fresh]);
            current.payload = sat_add(current.payload, v3_emit_size([&](V2Out& out) { v2_stream_state(out, ck, 5, 0); }));
        }
    }
    if (!v3_write_segment(path, counts_name, id, emit_counts, part_opt, created, part_payload, file_bytes, error, st)) {
        v3_remove_segments(path, created); return false;
    }
    V3Archive delta_view;
    uint64_t delta_vbase = kHeader;
    if (!v3_add_part(delta_view, path, delta, id, delta_vbase, error)) {
        v3_remove_segments(path, created); return false;
    }
    current.payload = v3_merged_saveinc_payload(current, archive, delta_view, meta, keep, part_payload);
    if (!v3_write_manifest(path, current, opt, error, st)) {
        v3_remove_segments(path, created); return false;
    }
    std::string verify_error;
    if (!v3_verify(path, id, verify_error)) {
        SessionWriteOptions rollback_opt = opt;
        rollback_opt.fault = {};
        std::string rollback_error;
        SessionStatus rollback_status;
        v3_write_manifest(path, archive.man, rollback_opt, rollback_error, rollback_status);
        v3_remove_segments(path, created);
        error = "saveinc: written but " + verify_error;
        st.error = SessionError::invalid;
        return false;
    }
    v3_cleanup_superseded(path, archive.man, current);
    st.error = SessionError::none;
    bytes = (size_t) v3_disk_bytes(path);
    strata::core::l2two::hook_post_saveinc(path, meta, id, opt, split);
    return true;
}

bool v3_saveinc_needs_compaction(const std::string& path, const SessionFileIdentity& id, std::string& reason) {
    V3Archive archive;
    std::string error;
    if (!v3_load_archive(path, id, archive, error)) return false;
    return v3_tail_needs_compaction(archive, reason);
}

bool v3_verify_open(const std::string& path, std::string& error) {
    V3Manifest man;
    if (!v3_read_manifest(path, man.id, man, error, false)) return false;
    return v3_verify(path, man.id, error);
}

bool v3_saveinc_needs_compaction_open(const std::string& path, std::string& reason) {
    // V3 saveinc folds the delta chain back to its immutable base internally;
    // the caller must not bypass capture with a full save.
    (void) path;
    reason.clear();
    return false;
}
