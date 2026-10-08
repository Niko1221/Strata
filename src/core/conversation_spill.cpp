// The spill directory's own format is only the sidecar: the conversation itself is an ordinary session file,
// written and read by conversation_file.cpp, so a spilled conversation is byte-for-byte what the slot save/restore
// API would write and the same model/config identity rules refuse a foreign file. The sidecar carries the token and
// image metadata (no K/V) so best() can match without reading the conversation.
#include "strata/core/conversation_spill.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <system_error>

namespace strata::core {
namespace {

constexpr char kMetaMagic[4] = {'S', 'C', 'S', 'M'};   // Strata conversation-spill metadata
constexpr uint32_t kMetaVersion = 1;
constexpr size_t kMaxSidecarEntries = 256;

// A small length-delimited writer/reader for the sidecar, with a trailing hash of the payload (the same
// SessionHasher the session file uses) so a truncated or edited sidecar is refused rather than trusted.
class MetaWriter {
public:
    explicit MetaWriter(std::ostream& out) : out_(out) {}
    bool raw(const void* p, size_t n) {
        out_.write(static_cast<const char*>(p), (std::streamsize) n);
        if (!out_) return false;
        hash_.update(p, n);
        return true;
    }
    bool u64(uint64_t v) { return raw(&v, sizeof v); }
    bool i64(int64_t v) { return raw(&v, sizeof v); }
    bool finish() { const uint64_t d = hash_.digest(); return raw(&d, sizeof d); }
private:
    std::ostream& out_;
    SessionHasher hash_{0};
};

class MetaReader {
public:
    MetaReader(std::istream& in, uint64_t file_bytes) : in_(in), file_bytes_(file_bytes) {}
    uint64_t remaining() const { return consumed_ <= file_bytes_ && file_bytes_ - consumed_ >= 8 ? file_bytes_ - consumed_ - 8 : 0; }
    bool raw(void* p, size_t n) {
        if (n > remaining()) return false;
        in_.read(static_cast<char*>(p), (std::streamsize) n);
        if (in_.gcount() != (std::streamsize) n) return false;
        hash_.update(p, n);
        consumed_ += n;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, sizeof v); }
    bool i64(int64_t& v) { return raw(&v, sizeof v); }
    bool finish() {
        if (consumed_ + 8 != file_bytes_) return false;
        uint64_t want = 0;
        in_.read(reinterpret_cast<char*>(&want), 8);
        if (in_.gcount() != 8) return false;
        return want == hash_.digest();
    }
private:
    std::istream& in_;
    uint64_t file_bytes_ = 0, consumed_ = 0;
    SessionHasher hash_{0};
};

bool valid_meta_chain(const ConversationCheckpoint& live, const std::vector<size_t>& checkpoint_lengths) {
    if (live.ids.empty()) return false;
    int64_t previous_image = -1;
    for (const auto& key : live.imgs) {
        if (key.start <= previous_image || key.start < 0 || (uint64_t) key.start >= live.ids.size()) return false;
        previous_image = key.start;
    }
    for (size_t length : checkpoint_lengths)
        if (length == 0 || length > live.ids.size()) return false;
    return true;
}

} // namespace

bool ConversationSpillCache::read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity,
                                          std::string& error) const {
    other_identity = false;
    std::error_code ec;
    const uint64_t file_bytes = std::filesystem::file_size(meta, ec);
    if (ec || file_bytes < 4 + 4 + 16 + 8) { error = "spill sidecar is missing or too short"; return false; }
    std::ifstream in(meta, std::ios::binary);
    if (!in) { error = "cannot open spill sidecar"; return false; }
    MetaReader r(in, file_bytes);
    char magic[4];
    uint64_t version = 0;
    uint64_t model = 0, config = 0;
    if (!r.raw(magic, 4) || std::memcmp(magic, kMetaMagic, 4) != 0 || !r.u64(version) || version != kMetaVersion ||
        !r.u64(model) || !r.u64(config)) {
        error = "unknown spill sidecar magic or version"; return false;
    }
    if (model != identity_.model || config != identity_.config) { other_identity = true; return false; }
    uint64_t cvec = 0;
    if (!r.u64(cvec) || cvec > 1) { error = "invalid cvec flag in sidecar"; return false; }
    entry.cvec = cvec != 0;
    uint64_t n = 0;
    if (!r.u64(n) || n > r.remaining() / 4 || n > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "invalid token count in sidecar"; return false;
    }
    try { entry.live_meta.ids.resize((size_t) n); } catch (const std::bad_alloc&) { error = "not enough RAM for sidecar"; return false; }
    for (auto& id : entry.live_meta.ids) { int64_t v = 0; if (!r.i64(v)) { error = "truncated sidecar ids"; return false; } id = (int32_t) v; }
    if (!r.u64(n) || n > r.remaining() / 16) { error = "invalid image count in sidecar"; return false; }
    try { entry.live_meta.imgs.resize((size_t) n); } catch (const std::bad_alloc&) { error = "not enough RAM for sidecar"; return false; }
    for (auto& k : entry.live_meta.imgs)
        if (!r.i64(k.start) || !r.u64(k.hash)) { error = "truncated sidecar images"; return false; }
    if (!r.u64(n) || n > kMaxSidecarEntries) { error = "invalid checkpoint count in sidecar"; return false; }
    entry.checkpoint_lengths.reserve((size_t) n);
    for (uint64_t i = 0; i < n; ++i) { uint64_t len = 0; if (!r.u64(len)) { error = "truncated sidecar checkpoints"; return false; } entry.checkpoint_lengths.push_back((size_t) len); }
    if (!r.finish()) { error = "spill sidecar checksum or length mismatch"; return false; }
    if (!valid_meta_chain(entry.live_meta, entry.checkpoint_lengths)) { error = "spill sidecar is not a valid prefix chain"; return false; }
    return true;
}

bool ConversationSpillCache::write_sidecar(const std::filesystem::path& meta, const Entry& entry, std::string& error) const {
    const std::filesystem::path temporary(meta.string() + ".tmp");
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { error = "cannot create spill sidecar"; return false; }
        MetaWriter w(out);
        if (!w.raw(kMetaMagic, 4) || !w.u64(kMetaVersion) || !w.u64(identity_.model) || !w.u64(identity_.config) ||
            !w.u64(entry.cvec ? 1 : 0) || !w.u64(entry.live_meta.ids.size())) {
            error = "cannot write spill sidecar"; return false;
        }
        for (int32_t id : entry.live_meta.ids) if (!w.i64(id)) { error = "cannot write spill sidecar ids"; return false; }
        if (!w.u64(entry.live_meta.imgs.size())) { error = "cannot write spill sidecar"; return false; }
        for (const auto& k : entry.live_meta.imgs) if (!w.i64(k.start) || !w.u64(k.hash)) { error = "cannot write spill sidecar images"; return false; }
        if (!w.u64(entry.checkpoint_lengths.size())) { error = "cannot write spill sidecar"; return false; }
        for (size_t len : entry.checkpoint_lengths) if (!w.u64(len)) { error = "cannot write spill sidecar checkpoints"; return false; }
        if (!w.finish()) { error = "cannot finish spill sidecar"; return false; }
        out.flush();
        if (!out) { error = "cannot flush spill sidecar"; return false; }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, meta, ec);
    if (ec) { std::filesystem::remove(temporary, ec); error = "cannot commit spill sidecar"; return false; }
    return true;
}

bool ConversationSpillCache::open(const std::filesystem::path& directory, SessionFileIdentity identity,
                                  uint64_t budget_bytes, std::string& error) {
    enabled_ = false;
    stale_files_wiped_ = 0;
    disk_evictions_ = 0;
    entries_.clear();
    bytes_ = 0;
    pinned_path_.clear();
    directory_ = directory;
    identity_ = identity;
    budget_ = budget_bytes;
    if (directory.empty() || !identity.model || !identity.config || !budget_) return true;
    try {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        if (ec || !std::filesystem::is_directory(directory_, ec) || ec) {
            error = "cannot create or inspect spill directory " + directory_.string(); return false;
        }
        std::vector<std::filesystem::path> metas;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) break;
            const std::string name = it->path().filename().string();
            if (name.rfind("strata-conv-", 0) != 0) continue;
            if (!std::filesystem::is_regular_file(status)) continue;
            if (it->path().extension() == ".meta") metas.push_back(it->path());
            else if (it->path().extension() == ".tmp") { std::filesystem::remove(it->path(), ec); if (!ec) ++stale_files_wiped_; ec.clear(); }
        }
        if (ec) { error = "cannot scan spill directory " + directory_.string(); return false; }
        std::sort(metas.begin(), metas.end());
        size_t stale = stale_files_wiped_;
        for (const auto& meta : metas) {
            Entry entry;
            entry.stem = meta;
            entry.stem.replace_extension();   // drop ".meta" -> the shared stem
            bool other_identity = false;
            std::string parse_error;
            if (!read_sidecar(meta, entry, other_identity, parse_error)) {
                if (!other_identity) { std::filesystem::remove(meta, ec); ec.clear(); std::filesystem::remove(entry.session_path(), ec); ec.clear(); ++stale; }
                continue;
            }
            std::error_code sec;
            const uint64_t session_bytes = std::filesystem::file_size(entry.session_path(), sec);
            if (sec) { std::filesystem::remove(meta, ec); ec.clear(); ++stale; continue; }   // session file gone: sidecar is useless
            entry.file_bytes = session_bytes;
            if (session_bytes > budget_) { std::filesystem::remove(meta, ec); ec.clear(); std::filesystem::remove(entry.session_path(), ec); ec.clear(); ++stale; continue; }
            entries_.push_back(std::move(entry));
            bytes_ += session_bytes;
            enforce_budget();
        }
        // session files with no sidecar left behind (a crash between the two writes) go too.
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) break;
            const std::string name = it->path().filename().string();
            if (name.rfind("strata-conv-", 0) != 0 || it->path().extension() != ".sess") continue;
            if (!std::filesystem::is_regular_file(status)) continue;
            std::filesystem::path meta = it->path();
            meta.replace_extension(".meta");
            if (std::filesystem::exists(meta, ec) && !ec) continue;
            std::filesystem::remove(it->path(), ec); ec.clear(); ++stale;
        }
        enabled_ = true;
        enforce_budget();
        stale_files_wiped_ = stale;
        return true;
    } catch (const std::bad_alloc&) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = "not enough RAM to index spill directory"; return false;
    } catch (const std::exception& e) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = std::string("cannot index spill directory: ") + e.what(); return false;
    }
}

bool ConversationSpillCache::load(const std::string& path, SavedConversation& image, const SessionReadLimits& limits,
                                  std::string& error) const {
    if (!enabled_) { error = "spill cache is disabled"; return false; }
    size_t bytes = 0;
    return session_file_read(path, identity_, image, bytes, error, limits);
}

bool ConversationSpillCache::spill(const SavedConversation& image, std::string& error) {
    if (!enabled_) return false;
    if (image.live.ids.empty() || !image.stage_images.empty()) { error = "spill needs a single-GPU conversation"; return false; }
    std::error_code ec;
    std::filesystem::path stem;
    do {
        stem = directory_ / ("strata-conv-" + std::to_string(++serial_));
    } while (std::filesystem::exists(stem.string() + ".sess", ec) || std::filesystem::exists(stem.string() + ".meta", ec));
    const std::string session = stem.string() + ".sess";
    size_t file_bytes = 0;
    SessionWriteOptions wo;
    wo.durable = true;
    SessionStatus st;
    if (!session_file_write(session, image, identity_, file_bytes, error, wo, &st)) {
        std::filesystem::remove(session, ec);
        return false;
    }
    if (file_bytes > budget_) { std::filesystem::remove(session, ec); error = "snapshot exceeds the disk cache limit"; return false; }
    Entry entry;
    entry.stem = stem;
    entry.file_bytes = file_bytes;
    entry.cvec = image.cvec;
    entry.live_meta.ids = image.live.ids;
    entry.live_meta.imgs = image.live.imgs;
    entry.checkpoint_lengths.reserve(image.checkpoints.size());
    for (const auto& c : image.checkpoints) entry.checkpoint_lengths.push_back(c.ids.size());
    if (!write_sidecar(stem.string() + ".meta", entry, error)) { std::filesystem::remove(session, ec); return false; }
    try {
        entries_.push_back(std::move(entry));
    } catch (const std::bad_alloc&) {
        std::filesystem::remove(session, ec); std::filesystem::remove(stem.string() + ".meta", ec);
        error = "not enough RAM to index conversation spill file"; return false;
    }
    bytes_ += file_bytes;
    enforce_budget();
    return true;
}

bool ConversationSpillCache::erase(const std::string& path, std::string& error) {
    auto found = std::find_if(entries_.begin(), entries_.end(),
                              [&](const Entry& entry) { return entry.session_path() == path; });
    if (found == entries_.end()) return true;
    std::error_code ec;
    std::filesystem::remove(found->session_path(), ec);
    if (ec) { error = "cannot remove spill file " + found->session_path(); return false; }
    std::filesystem::remove(found->meta_path(), ec);
    bytes_ -= found->file_bytes;
    if (pinned_path_ == path) pinned_path_.clear();
    entries_.erase(found);
    return true;
}

void ConversationSpillCache::pin(const std::string& path) { pinned_path_ = path; }

void ConversationSpillCache::unpin(const std::string& path) {
    if (pinned_path_ == path) { pinned_path_.clear(); enforce_budget(); }
}

size_t ConversationSpillCache::drop_superseded(const std::vector<int32_t>& ids,
                                               const std::vector<ConversationImageKey>& images,
                                               const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        const Entry& entry = entries_[i];
        size_t deepest = 0;
        for (size_t length : entry.checkpoint_lengths) deepest = std::max(deepest, length);
        auto same_saved_prefix = [&](size_t length, const std::vector<int32_t>& candidate_ids,
                                     const std::vector<ConversationImageKey>& candidate_images) {
            if (length == 0 || candidate_ids.size() != length || length > entry.live_meta.ids.size() ||
                !std::equal(entry.live_meta.ids.begin(), entry.live_meta.ids.begin() + (std::ptrdiff_t) length,
                            candidate_ids.begin())) return false;
            size_t a = 0, b = 0;
            while ((a < entry.live_meta.imgs.size() && entry.live_meta.imgs[a].start < (int64_t) length) ||
                   (b < candidate_images.size() && candidate_images[b].start < (int64_t) length)) {
                if (a == entry.live_meta.imgs.size() || entry.live_meta.imgs[a].start >= (int64_t) length ||
                    b == candidate_images.size() || candidate_images[b].start >= (int64_t) length ||
                    !(entry.live_meta.imgs[a] == candidate_images[b])) return false;
                ++a;
                ++b;
            }
            return true;
        };
        bool held = same_saved_prefix(deepest, ids, images);
        for (const auto& checkpoint : checkpoints)
            if (!held && same_saved_prefix(deepest, checkpoint.ids, checkpoint.imgs)) held = true;
        if (entry.cvec == cvec && deepest && held && entry.session_path() != pinned_path_) {
            std::error_code ec;
            std::filesystem::remove(entry.session_path(), ec);
            std::filesystem::remove(entry.meta_path(), ec);
            if (ec) { ++i; continue; }
            bytes_ -= entry.file_bytes;
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            ++dropped;
            continue;
        }
        ++i;
    }
    return dropped;
}

void ConversationSpillCache::enforce_budget() {
    for (size_t i = 0; (bytes_ > budget_ || entries_.size() > kMaxSidecarEntries) && i < entries_.size();) {
        if (entries_[i].session_path() == pinned_path_) { ++i; continue; }
        std::error_code ec;
        std::filesystem::remove(entries_[i].session_path(), ec);
        std::filesystem::remove(entries_[i].meta_path(), ec);
        if (ec) { ++i; continue; }
        bytes_ -= entries_[i].file_bytes;
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
        ++disk_evictions_;
    }
}

} // namespace strata::core
