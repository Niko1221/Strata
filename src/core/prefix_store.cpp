#include "strata/core/prefix_store.hpp"
#include "strata/core/conversation_memory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <new>
#include <system_error>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// file: magic, identity, cvec, ids | running state and K/V headers | K/V payloads (conversation_kv_layout order)
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'F', 'X', '0', '1'};

uint64_t key_of(const std::vector<int32_t>& ids, bool cvec) {
    uint64_t h = 1469598103934665603ull ^ (cvec ? 1 : 0);
    for (int32_t id : ids)
        for (int b = 0; b < 4; ++b) { h ^= (uint8_t) ((uint32_t) id >> (8 * b)); h *= 1099511628211ull; }
    return h;
}

struct File {
    std::FILE* f = nullptr;
    uint64_t left = 0;   // bytes a reader may still take: a corrupt length never allocates past the file
    File(const std::string& path, const char* mode) : f(std::fopen(path.c_str(), mode)) {
        std::error_code ec;
        if (f && mode[0] == 'r') left = fs::file_size(path, ec);
    }
    ~File() { if (f) std::fclose(f); }
    bool put(const void* p, size_t n) { return n == 0 || std::fwrite(p, 1, n, f) == n; }
    bool get(void* p, size_t n) {
        if (n > left || (n && std::fread(p, 1, n, f) != n)) return false;
        left -= n;
        return true;
    }
    template<class T> bool put_pod(const T& v) { return put(&v, sizeof v); }
    template<class T> bool get_pod(T& v) { return get(&v, sizeof v); }
    template<class T> bool put_vec(const std::vector<T>& v) {
        return put_pod<uint64_t>(v.size()) && put(v.data(), v.size() * sizeof(T));
    }
    template<class T> bool get_vec(std::vector<T>& v) {
        uint64_t n = 0;
        if (!get_pod(n) || n > left / sizeof(T)) return false;
        v.resize((size_t) n);
        return get(v.data(), v.size() * sizeof(T));
    }
    template<class T> bool skip_vec() {   // the index needs no running state
        uint64_t n = 0;
        if (!get_pod(n) || n > left / sizeof(T) || std::fseek(f, (long) (n * sizeof(T)), SEEK_CUR) != 0) return false;
        left -= n * sizeof(T);
        return true;
    }
    bool close() {   // flushed to the disk before the rename publishes it
        bool ok = std::fflush(f) == 0;
#ifdef _WIN32
        ok = ok && _commit(_fileno(f)) == 0;
#else
        ok = ok && fsync(fileno(f)) == 0;
#endif
        ok = std::fclose(f) == 0 && ok;
        f = nullptr;
        return ok;
    }
};

bool put_head(File& w, const std::string& identity, const std::vector<int32_t>& ids, bool cvec) {
    return w.put(kMagic, sizeof kMagic) && w.put_pod<uint64_t>(identity.size()) &&
           w.put(identity.data(), identity.size()) && w.put_pod<uint8_t>(cvec ? 1 : 0) && w.put_vec(ids);
}

bool get_head(File& r, const std::string& identity, std::vector<int32_t>& ids, bool& cvec) {
    char magic[sizeof kMagic];
    uint64_t n = 0;
    uint8_t c = 0;
    if (!r.get(magic, sizeof magic) || !std::equal(magic, magic + sizeof magic, kMagic) || !r.get_pod(n) ||
        n != identity.size()) return false;
    std::string id(identity.size(), '\0');
    if (!r.get(id.data(), id.size()) || id != identity || !r.get_pod(c) || c > 1 || !r.get_vec(ids) || ids.empty())
        return false;
    cvec = c != 0;
    return true;
}

bool put_meta(File& w, const SavedConversation& m, const std::vector<ConversationKvSizes>& sizes) {
    bool ok = w.put(m.geometry.data(), sizeof m.geometry) && w.put_pod(m.layer_lo) && w.put_pod(m.layer_hi);
    for (const auto* v : {&m.live.gdn, &m.live.ple, &m.live.tails, &m.live.dead, &m.live.block_pos})
        ok = ok && w.put_vec(*v);
    ok = ok && w.put_pod<uint64_t>(m.kv.size());
    for (size_t i = 0; ok && i < m.kv.size(); ++i) {
        const auto& k = m.kv[i];
        ok = w.put_pod<int32_t>(k.format) && w.put_pod(k.cells) && w.put_pod(k.heads) && w.put_pod(k.head_dim) &&
             w.put_pod(k.page_size) && w.put_pod(k.pooled_rows) && w.put_pod(k.idx_dim) &&
             w.put(sizes[i].data(), sizeof sizes[i]);
    }
    return ok;
}

bool get_meta(File& r, SavedConversation& m, std::vector<ConversationKvSizes>& sizes, bool skip_state = false) {
    bool ok = r.get(m.geometry.data(), sizeof m.geometry) && r.get_pod(m.layer_lo) && r.get_pod(m.layer_hi);
    for (auto* v : {&m.live.gdn, &m.live.ple, &m.live.tails, &m.live.dead, &m.live.block_pos})
        ok = ok && (skip_state ? r.skip_vec<uint8_t>() : r.get_vec(*v));
    uint64_t layers = 0;
    if (!ok || !r.get_pod(layers) || layers > 4096) return false;
    m.kv.resize((size_t) layers);
    sizes.resize((size_t) layers);
    uint64_t payload = 0;
    for (size_t i = 0; i < m.kv.size(); ++i) {
        auto& k = m.kv[i];
        int32_t format = 0;
        if (!r.get_pod(format) || !r.get_pod(k.cells) || !r.get_pod(k.heads) || !r.get_pod(k.head_dim) ||
            !r.get_pod(k.page_size) || !r.get_pod(k.pooled_rows) || !r.get_pod(k.idx_dim) ||
            !r.get(sizes[i].data(), sizeof sizes[i])) return false;
        k.format = format;
        for (size_t n : sizes[i]) payload += n;
    }
    return payload == r.left;   // the payloads, exactly: a partial file restores nothing
}
} // namespace

std::string PrefixStore::path(uint64_t key) const {
    char name[32];
    std::snprintf(name, sizeof name, "%016llx.pfx", (unsigned long long) key);
    return (fs::path(o_.dir) / name).string();
}

PrefixStore::Entry* PrefixStore::find(uint64_t key) {
    for (auto& e : entries_) if (e.key == key) return &e;
    return nullptr;
}

void PrefixStore::erase(uint64_t key) {
    std::error_code ec;
    fs::remove(path(key), ec);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.key == key; }),
                   entries_.end());
}

size_t PrefixStore::ram_bytes() const {
    size_t n = 0;
    for (const auto& e : entries_) n += e.ram_bytes;
    return n;
}

size_t PrefixStore::disk_bytes() const {
    size_t n = 0;
    for (const auto& e : entries_) n += e.file_bytes;
    return n;
}

// least recently used first, until `incoming` more fits the budget
void PrefixStore::trim_disk(size_t incoming) {
    while (!entries_.empty() && disk_bytes() + incoming > o_.disk_budget)
        erase(std::min_element(entries_.begin(), entries_.end(),
                               [](const Entry& a, const Entry& b) { return a.used < b.used; })->key);
}

bool PrefixStore::trim_ram(size_t incoming) {
    if (incoming > o_.ram_budget) return false;
    while (ram_bytes() + incoming > o_.ram_budget) {
        Entry* oldest = nullptr;
        for (auto& e : entries_)
            if (e.ram && (!oldest || e.used < oldest->used)) oldest = &e;
        oldest->ram.reset();
        oldest->ram_bytes = 0;
    }
    return true;
}

bool PrefixStore::open(const Options& options, std::string& log, std::string& error) {
    o_ = options;
    entries_.clear();
    if (!enabled()) return true;
    std::error_code ec;
    fs::create_directories(o_.dir, ec);
    if (ec || !fs::is_directory(o_.dir, ec)) {
        error = "prefix cache: cannot use directory " + o_.dir;
        o_.dir.clear();
        return false;
    }
    struct Found { Entry e; fs::file_time_type t; };
    std::vector<Found> found;
    size_t dropped = 0;
    for (const auto& d : fs::directory_iterator(o_.dir, ec)) {
        const auto p = d.path();
        if (p.extension() == ".tmp") { fs::remove(p, ec); ++dropped; continue; }
        if (p.extension() != ".pfx") continue;
        Entry e;
        bool ok = false;
        {
            File r(p.string(), "rb");
            SavedConversation m;
            std::vector<ConversationKvSizes> sizes;
            ok = r.f && get_head(r, o_.identity, e.ids, e.cvec) && get_meta(r, m, sizes, true);
        }
        e.key = key_of(e.ids, e.cvec);
        if (!ok || p.filename().string() != fs::path(path(e.key)).filename().string()) {
            fs::remove(p, ec);
            ++dropped;
            continue;
        }
        e.file_bytes = (size_t) fs::file_size(p, ec);
        found.push_back({std::move(e), fs::last_write_time(p, ec)});
    }
    std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
    for (auto& f : found) {
        f.e.used = ++clock_;
        entries_.push_back(std::move(f.e));
    }
    trim_disk(0);
    char text[160];
    std::snprintf(text, sizeof text, "%zu snapshots (%.1f MB) in %s, %zu other files deleted", entries_.size(),
                  (double) disk_bytes() / 1e6, o_.dir.c_str(), dropped);
    log = text;
    return true;
}

bool PrefixStore::contains(const std::vector<int32_t>& ids, bool cvec) const {
    for (const auto& e : entries_) if (e.cvec == cvec && e.ids == ids) return true;
    return false;
}

bool PrefixStore::capture(const ConversationCheckpoint& root, bool cvec, const SessionState& ss,
                          const ModelGeometry& g, const QsaState& draft, std::string& log, std::string& error) {
    if (!enabled() || contains(root.ids, cvec)) return true;
    const auto t0 = Clock::now();
    SavedConversation meta;
    std::vector<ConversationKvSizes> sizes;
    if (!conversation_prefix_header(meta, sizes, root, cvec, ss, g, draft, error)) return false;
    size_t bytes = root.bytes();
    for (const auto& z : sizes) for (size_t n : z) bytes += n;
    if (bytes > o_.disk_budget) {
        log = "skipped: larger than the disk budget";
        return true;
    }
    const uint64_t key = key_of(root.ids, cvec);
    if (find(key)) erase(key);   // a hash collision: the newer prefix replaces it
    trim_disk(bytes);
    const std::string final_path = path(key), tmp = final_path + ".tmp";
    std::error_code ec;
    bool ok = false;
    try {
        std::vector<uint8_t> bounce(o_.bounce);
        File w(tmp, "wb");
        ok = w.f && put_head(w, o_.identity, root.ids, cvec) && put_meta(w, meta, sizes) &&
             conversation_prefix_save_stream(ss, g, draft, (int64_t) root.ids.size(), bounce,
                                             [&](const uint8_t* p, size_t n) { return w.put(p, n); }, error) &&
             w.close();
        if (!ok && error.empty()) error = "prefix cache: writing " + tmp + " failed";
    } catch (const std::bad_alloc&) {
        error = "prefix cache: no memory for the write buffer";
    }
    if (ok) fs::rename(tmp, final_path, ec);
    if (!ok || ec) {
        fs::remove(tmp, ec);
        if (error.empty()) error = "prefix cache: publishing " + final_path + " failed";
        return false;
    }
    Entry e;
    e.key = key;
    e.ids = root.ids;
    e.cvec = cvec;
    e.file_bytes = (size_t) fs::file_size(final_path, ec);
    e.used = ++clock_;
    entries_.push_back(std::move(e));
    char text[160];
    std::snprintf(text, sizeof text, "saved %zu tokens (%.1f MB) in %.1f ms; snapshots=%zu disk=%.1f MB",
                  root.ids.size(), (double) entries_.back().file_bytes / 1e6,
                  std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), entries_.size(),
                  (double) disk_bytes() / 1e6);
    log = text;
    return true;
}

ConversationRestore PrefixStore::restore(uint64_t key, SessionState& ss, const ModelGeometry& g,
                                         const QsaState& draft, std::string& source, std::string& error) {
    Entry* e = find(key);
    if (!e) { error = "prefix cache: no such snapshot"; return ConversationRestore::invalid; }
    e->used = ++clock_;
    std::error_code ec;
    fs::last_write_time(path(key), fs::file_time_type::clock::now(), ec);   // the LRU order across restarts
    if (e->ram) {
        source = "ram";
        const auto r = conversation_snapshot_restore(*e->ram, ss, g, draft, error);
        if (r == ConversationRestore::invalid) erase(key);
        return r;
    }
    File r(path(key), "rb");
    auto drop = [&] {   // closed first: Windows does not delete an open file
        if (r.f) std::fclose(r.f);
        r.f = nullptr;
        erase(key);
    };
    SavedConversation m;
    std::vector<ConversationKvSizes> sizes;
    bool cvec = true;
    if (!r.f || !get_head(r, o_.identity, m.live.ids, cvec) || !get_meta(r, m, sizes) || m.live.ids != e->ids ||
        cvec != e->cvec) {
        error = "prefix cache: unreadable snapshot file";
        drop();
        return ConversationRestore::invalid;
    }
    m.cvec = cvec;
    size_t bytes = m.live.bytes() + m.kv.size() * sizeof(ConversationKv);
    for (const auto& z : sizes) for (size_t n : z) bytes += n;
    // the RAM tier when it has room and the machine has the memory; else straight from the file
    if (trim_ram(bytes) && conversation_memory_admit(conversation_available_memory(), bytes, o_.min_free)) {
        try {
            auto image = std::make_shared<SavedConversation>(std::move(m));
            bool ok = true;
            for (size_t i = 0; ok && i < image->kv.size(); ++i) {
                auto& k = image->kv[i];
                ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
                for (size_t j = 0; ok && j < 5; ++j) {
                    parts[j]->resize(sizes[i][j]);
                    ok = parts[j]->visit(0, sizes[i][j], [&](uint8_t* p, size_t n, size_t) { return r.get(p, n); });
                }
            }
            if (!ok) {
                error = "prefix cache: reading the snapshot file failed";
                drop();
                return ConversationRestore::invalid;
            }
            source = "disk>ram";
            const auto result = conversation_snapshot_restore(*image, ss, g, draft, error);
            if (result == ConversationRestore::invalid) { drop(); return result; }
            e->ram_bytes = image->bytes();
            e->ram = std::move(image);
            return result;
        } catch (const std::bad_alloc&) {
            error = "prefix cache: no memory to read the snapshot";
            return ConversationRestore::invalid;   // nothing written yet: read the prompt instead
        }
    }
    source = "disk";
    std::vector<uint8_t> bounce;
    try { bounce.resize(o_.bounce); } catch (const std::bad_alloc&) {
        error = "prefix cache: no memory for the read buffer";
        return ConversationRestore::invalid;
    }
    const auto result = conversation_prefix_restore_stream(m, sizes, ss, g, draft, bounce,
                                                           [&](uint8_t* p, size_t n) { return r.get(p, n); }, error);
    if (result == ConversationRestore::invalid) drop();
    return result;
}

} // namespace strata::core
