// The spill directory's own format is only the sidecar: the conversation itself is an ordinary session file,
// written and read by conversation_file.cpp, so a spilled conversation is byte-for-byte what the slot save/restore
// API would write and the same model/config identity rules refuse a foreign file. The sidecar carries the token and
// image metadata (no K/V) so best() can match without reading the conversation.
//
// A layer-split conversation is one session file per stage (SavedConversation::stage_images), one joint sidecar.
// The scan never deletes; only enforce_budget() (the GC) does, and it leaves a conversation larger than the budget
// alone.
#include "strata/core/conversation_spill.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <system_error>

namespace strata::core {
namespace {

constexpr char kMetaMagic[4] = {'S', 'C', 'S', 'M'};   // Strata conversation-spill metadata
constexpr uint32_t kMetaVersion = 2;                   // v2 adds the stage count; v1 (single image) is still read
constexpr size_t kMaxSidecarEntries = 256;
constexpr char kFilePrefix[] = "strata-conv-";

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

// The shared stem of every file of a conversation: drop the ".sess" suffix and, when present, a trailing ".stageN".
std::string stem_string(const std::filesystem::path& path) {
    std::string s = path.string();
    const std::string ext = ".sess";
    if (s.size() > ext.size() && s.compare(s.size() - ext.size(), ext.size(), ext) == 0)
        s.resize(s.size() - ext.size());
    const size_t at = s.rfind(".stage");
    if (at != std::string::npos) {
        const std::string tail = s.substr(at + 6);
        if (!tail.empty() && tail.find_first_not_of("0123456789") == std::string::npos) s.resize(at);
    }
    return s;
}

bool is_spill_name(const std::string& name) {
    return name.rfind(kFilePrefix, 0) == 0;
}

} // namespace

std::string ConversationSpillCache::stage_path(const std::string& session_path, size_t stage) {
    std::string stem = session_path;
    const std::string ext = ".sess";
    if (stem.size() > ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
        stem.resize(stem.size() - ext.size());
    return stem + ".stage" + std::to_string(stage) + ".sess";
}

const ConversationSpillCache::Entry* ConversationSpillCache::find(const std::string& path) const {
    for (const Entry& entry : entries_) if (entry.session_path() == path) return &entry;
    return nullptr;
}

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
    if (!r.raw(magic, 4) || std::memcmp(magic, kMetaMagic, 4) != 0 || !r.u64(version) ||
        (version != 1 && version != kMetaVersion) ||
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
    entry.stages = 0;
    if (version >= 2) {
        if (!r.u64(n) || n > kMaxSidecarEntries) { error = "invalid stage count in sidecar"; return false; }
        entry.stages = (size_t) n;
    }
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
        if (!w.u64(entry.stages)) { error = "cannot write spill sidecar stages"; return false; }
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
                                  uint64_t budget_bytes, std::string& error, SpillWhenFull when_full,
                                  int64_t max_age_days) {
    enabled_ = false;
    stale_files_kept_ = 0;
    foreign_files_kept_ = 0;
    orphan_files_kept_ = 0;
    oversized_files_kept_ = 0;
    disk_evictions_ = 0;
    age_evictions_ = 0;
    entries_.clear();
    bytes_ = 0;
    pinned_path_.clear();
    directory_ = directory;
    identity_ = identity;
    budget_ = budget_bytes;
    when_full_ = when_full;
    max_age_days_ = max_age_days < 0 ? 0 : max_age_days;
    if (directory.empty() || !identity.model || !identity.config || !budget_) return true;
    try {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        if (ec || !std::filesystem::is_directory(directory_, ec) || ec) {
            error = "cannot create or inspect spill directory " + directory_.string(); return false;
        }
        // The scan only reads. A leftover temporary, a broken sidecar, a foreign identity, an orphan session file or
        // a conversation over the budget are all counted and left in place; removal is the GC's, below.
        std::vector<std::filesystem::path> metas;
        std::vector<std::filesystem::path> session_files;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) break;
            const std::string name = it->path().filename().string();
            if (!is_spill_name(name)) continue;
            if (!std::filesystem::is_regular_file(status)) continue;
            const std::string ext = it->path().extension().string();
            if (ext == ".meta") metas.push_back(it->path());
            else if (ext == ".sess") session_files.push_back(it->path());
            else ++stale_files_kept_;     // a leftover temporary or an unknown kind: reported, never removed
        }
        if (ec) { error = "cannot scan spill directory " + directory_.string(); return false; }
        std::sort(metas.begin(), metas.end());
        for (const auto& meta : metas) {
            Entry entry;
            entry.stem = meta;
            entry.stem.replace_extension();   // drop ".meta" -> the shared stem
            bool other_identity = false;
            std::string parse_error;
            if (!read_sidecar(meta, entry, other_identity, parse_error)) {
                if (other_identity) ++foreign_files_kept_;
                else ++stale_files_kept_;
                continue;   // ignored, not removed: a variant or a newer format beside this one is left alone
            }
            std::error_code sec;
            const uint64_t session_bytes = std::filesystem::file_size(entry.session_path(), sec);
            if (sec) { ++stale_files_kept_; continue; }   // session file gone: the sidecar is useless, kept anyway
            uint64_t total = session_bytes;
            bool complete = true;
            for (size_t k = 1; k <= entry.stages; ++k) {
                std::error_code kec;
                const uint64_t part = std::filesystem::file_size(entry.stage_path(k), kec);
                if (kec) { complete = false; break; }
                total += part;
            }
            if (!complete) { ++stale_files_kept_; continue; }
            entry.file_bytes = total;
            entry.stamp = std::filesystem::last_write_time(entry.session_path(), sec);   // for the age lever
            if (sec) entry.stamp = {};
            if (total > budget_) ++oversized_files_kept_;   // KEPT: the GC alone decides, and leaves it alone too
            entries_.push_back(std::move(entry));
            bytes_ += total;
        }
        // A session file (of the first stage or of a later one) with no sidecar left behind (a crash between the two
        // writes) would never be matched: report it, do not delete it.
        for (const auto& path : session_files) {
            std::filesystem::path meta = stem_string(path) + ".meta";
            if (!std::filesystem::exists(meta, ec) && !ec) ++orphan_files_kept_;
            ec.clear();
        }
        enabled_ = true;
        enforce_age();      // the age lever is explicit (off unless --conversation-cache-spill-max-age-days > 0)
        enforce_budget();   // the budget lever, oldest first, honoring --conversation-cache-spill-when-full
        return true;
    } catch (const std::bad_alloc&) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = "not enough RAM to index spill directory"; return false;
    } catch (const std::exception& e) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = std::string("cannot index spill directory: ") + e.what(); return false;
    }
}

bool ConversationSpillCache::load(const std::string& path, SavedConversation& image,
                                  const std::vector<SessionReadLimits>& limits, std::string& error) const {
    if (!enabled_) { error = "spill cache is disabled"; return false; }
    const Entry* entry = find(path);
    const size_t stages = entry ? entry->stages : 0;
    const SessionReadLimits open_limits;
    auto for_stage = [&](size_t i) -> const SessionReadLimits& {
        if (limits.empty()) return open_limits;
        return limits[i < limits.size() ? i : limits.size() - 1];
    };
    SavedConversation main;
    size_t bytes = 0;
    if (!session_file_read(path, identity_, main, bytes, error, for_stage(0))) return false;
    std::vector<SavedConversation> stage_images;
    stage_images.reserve(stages);
    for (size_t k = 1; k <= stages; ++k) {
        SavedConversation part;
        size_t part_bytes = 0;
        const std::string file = stage_path(path, k);
        if (!session_file_read(file, identity_, part, part_bytes, error, for_stage(k))) return false;
        stage_images.push_back(std::move(part));
    }
    main.stage_images = std::move(stage_images);
    image = std::move(main);
    return true;
}

bool ConversationSpillCache::spill(const SavedConversation& image, std::string& error) {
    if (!enabled_) return false;
    if (image.live.ids.empty()) { error = "spill needs a conversation with tokens"; return false; }
    // --conversation-cache-spill-when-full reject: a full directory refuses a new spill instead of evicting. The
    // estimate is the image's in-RAM size plus 1 MiB (the file is close); over-estimating only refuses slightly
    // earlier. Nothing already stored is ever removed in this mode.
    if (when_full_ == SpillWhenFull::reject) {
        const uint64_t estimate = (uint64_t) image.bytes() + (1ull << 20);
        if (entries_.size() >= kMaxSidecarEntries || bytes_ >= budget_ || estimate > budget_ - bytes_) {
            error = "spill directory is full and --conversation-cache-spill-when-full is reject (nothing evicted)";
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::path stem;
    do {
        stem = directory_ / (std::string(kFilePrefix) + std::to_string(++serial_));
    } while (std::filesystem::exists(stem.string() + ".sess", ec) || std::filesystem::exists(stem.string() + ".meta", ec));
    const std::string session = stem.string() + ".sess";
    std::vector<std::string> written;
    auto write_one = [&](const std::string& file, const SavedConversation& part, size_t& bytes) -> bool {
        SessionWriteOptions wo;
        wo.durable = true;
        SessionStatus st;
        if (!session_file_write(file, part, identity_, bytes, error, wo, &st)) return false;
        written.push_back(file);
        return true;
    };
    auto discard_written = [&] {
        for (const std::string& file : written) std::filesystem::remove(file, ec);
        written.clear();
    };
    size_t total = 0;
    size_t first_bytes = 0;
    if (!write_one(session, image, first_bytes)) { discard_written(); return false; }
    total = first_bytes;
    for (size_t k = 0; k < image.stage_images.size(); ++k) {
        size_t part_bytes = 0;
        if (!write_one(stage_path(session, k + 1), image.stage_images[k], part_bytes)) { discard_written(); return false; }
        total += part_bytes;
    }
    // Over the budget is written and indexed anyway: removing it on the spot would be a decision of the scan, and
    // this one cannot bring the directory under the budget by itself. The GC decides, and it leaves it alone too.
    Entry entry;
    entry.stem = stem;
    entry.file_bytes = total;
    entry.stages = image.stage_images.size();
    entry.cvec = image.cvec;
    entry.live_meta.ids = image.live.ids;
    entry.live_meta.imgs = image.live.imgs;
    entry.checkpoint_lengths.reserve(image.checkpoints.size());
    for (const auto& c : image.checkpoints) entry.checkpoint_lengths.push_back(c.ids.size());
    if (!write_sidecar(stem.string() + ".meta", entry, error)) { discard_written(); return false; }
    entry.stamp = std::filesystem::last_write_time(session, ec);
    ec.clear();
    if (total > budget_) ++oversized_files_kept_;
    try {
        entries_.push_back(std::move(entry));
    } catch (const std::bad_alloc&) {
        discard_written();
        std::filesystem::remove(stem.string() + ".meta", ec);
        error = "not enough RAM to index conversation spill file"; return false;
    }
    bytes_ += total;
    enforce_budget();
    return true;
}

// Removes every file of the entry at `index` and drops it from the index. False when the first (session) file
// could not be removed: the entry stays, the way a failed eviction leaves it in the RAM cache.
bool ConversationSpillCache::remove_entry(size_t index) {
    const std::string session = entries_[index].session_path();
    const std::string meta = entries_[index].meta_path();
    const size_t stages = entries_[index].stages;
    const uint64_t file_bytes = entries_[index].file_bytes;
    std::error_code ec;
    const bool removed = std::filesystem::remove(session, ec);
    ec.clear();
    if (!removed && std::filesystem::exists(session)) return false;
    for (size_t k = 1; k <= stages; ++k) { std::filesystem::remove(stage_path(session, k), ec); ec.clear(); }
    std::filesystem::remove(meta, ec);
    if (session == pinned_path_) pinned_path_.clear();
    bytes_ -= file_bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    return true;
}

bool ConversationSpillCache::erase(const std::string& path, std::string& error) {
    auto found = std::find_if(entries_.begin(), entries_.end(),
                              [&](const Entry& entry) { return entry.session_path() == path; });
    if (found == entries_.end()) return true;
    std::error_code ec;
    std::filesystem::remove(found->session_path(), ec);
    if (ec) { error = "cannot remove spill file " + found->session_path(); return false; }
    for (size_t k = 1; k <= found->stages; ++k) std::filesystem::remove(found->stage_path(k), ec);
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
            if (!remove_entry(i)) { ++i; continue; }   // a failed removal: leave this entry in the index
            ++dropped;
            continue;
        }
        ++i;
    }
    return dropped;
}

void ConversationSpillCache::enforce_budget() {
    // The GC's budget lever: oldest first, with a counter. A conversation larger than the whole budget is left
    // alone - removing it could never bring `bytes_` under `budget_` - and counted as oversized instead. With
    // --conversation-cache-spill-when-full reject this lever removes nothing at all (the spill itself refuses).
    if (when_full_ == SpillWhenFull::reject) return;
    for (size_t i = 0; (bytes_ > budget_ || entries_.size() > kMaxSidecarEntries) && i < entries_.size();) {
        if (entries_[i].session_path() == pinned_path_ || entries_[i].file_bytes > budget_) { ++i; continue; }
        if (!remove_entry(i)) { ++i; continue; }   // a failed removal: skip it
        ++disk_evictions_;
    }
}

void ConversationSpillCache::enforce_age() {
    // The GC's age lever: OFF unless --conversation-cache-spill-max-age-days is positive, so with 0 (the default)
    // there is no deletion by time at all. The oldest entries go first; a pinned conversation (mid-restore) is kept.
    if (max_age_days_ <= 0) return;
    const std::filesystem::file_time_type now = std::filesystem::file_time_type::clock::now();
    const std::chrono::seconds limit((int64_t) max_age_days_ * 86400);
    for (size_t i = 0; i < entries_.size();) {
        if (entries_[i].session_path() == pinned_path_) { ++i; continue; }
        if (now - entries_[i].stamp < limit) { ++i; continue; }
        if (!remove_entry(i)) { ++i; continue; }   // a failed removal: skip it
        ++age_evictions_;
    }
}

} // namespace strata::core
