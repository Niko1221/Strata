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
#include <map>
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

// Delta 5a: move one file to another name, preferring the cheap same-volume rename (the archive is a subdirectory
// of the spill directory by default). A cross-volume destination falls back to copy-then-remove, so an archive on
// another disk still works, just not as cheaply. A FAILURE of the final removal is a failure of the move (the file
// still exists under its old name), and the copy left behind is cleaned up so the caller sees an all-or-nothing move.
bool relocate_file(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::rename(from, to, ec);
    if (!ec) return true;
    ec.clear();
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) return false;
    std::filesystem::remove(from, ec);
    if (ec || std::filesystem::exists(from, ec)) {
        std::error_code rec;
        std::filesystem::remove(to, rec);   // the copy must not survive a failed move
        return false;
    }
    return true;
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
    std::lock_guard<std::mutex> lock(mu_);
    enabled_ = false;
    stale_files_kept_ = 0;
    foreign_files_kept_ = 0;
    orphan_files_kept_ = 0;
    oversized_files_kept_ = 0;
    disk_evictions_ = 0;
    age_evictions_ = 0;
    compacted_ = 0;
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
            // Delta 5a, gate refinement (b): the archive is a subdirectory of the spill directory by default, so the
            // scan skips directories EXPLICITLY - the archive never counts as a stale spill file.
            if (std::filesystem::is_directory(status)) continue;
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
    std::lock_guard<std::mutex> lock(mu_);
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

bool ConversationSpillCache::spill(const SavedConversation& image, std::string& error, std::string* stored_path) {
    std::lock_guard<std::mutex> lock(mu_);
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
    if (stored_path != nullptr) *stored_path = session;
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
    std::lock_guard<std::mutex> lock(mu_);
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

// ---- Delta 5a: the archive (gate refinements a, b, c) -------------------------------------------------------------
// The tier's three discard paths (the compaction, the supersede pass and the cancelled request's provisional park)
// no longer destroy the state when the archive is on: the copy is MOVED into <spill-dir>/archive (a cheap rename)
// instead of being removed. The archive has its OWN budget and its OWN GC, and its entries never enter entries_, so
// best() can never return one: it is evidence and recovery, not a reuse candidate (the attention is causal).

bool ConversationSpillCache::open_archive(const SpillArchiveConfig& config, std::string& error) {
    std::lock_guard<std::mutex> lock(mu_);
    archive_enabled_ = false;
    archive_mode_ = config.mode;
    archive_dir_ = config.directory;
    archive_budget_ = config.budget_bytes;
    archive_keep_ = config.keep;
    archive_max_age_days_ = config.max_age_days < 0 ? 0 : config.max_age_days;
    archive_turn_token_ = config.turn_token;
    archive_serial_ = 0;
    archive_bytes_ = 0;
    archive_evictions_ = 0;
    archive_age_evictions_ = 0;
    archive_entries_.clear();
    if (archive_mode_ == SpillArchiveMode::off || archive_dir_.empty()) return true;
    // The archive must not BE the tier directory: its entries would then be scanned back as reuse candidates,
    // breaking the "the archive is never a reuse candidate" invariant.
    if (!directory_.empty() && archive_dir_.lexically_normal() == directory_.lexically_normal()) {
        error = "archive directory must differ from the spill directory"; return false;
    }
    try {
        std::error_code ec;
        std::filesystem::create_directories(archive_dir_, ec);
        if (ec || !std::filesystem::is_directory(archive_dir_, ec) || ec) {
            error = "cannot create or inspect archive directory " + archive_dir_.string(); return false;
        }
        // The archive scan only indexes the sidecars it understands; a foreign identity or a broken sidecar is left
        // in place (never removed). The session file is optional: an ids-mode (or provisional) entry is just .meta.
        std::vector<std::filesystem::path> metas;
        for (std::filesystem::directory_iterator it(archive_dir_, ec), end; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) break;
            const std::string name = it->path().filename().string();
            if (!is_spill_name(name)) continue;
            if (std::filesystem::is_directory(status)) continue;
            if (!std::filesystem::is_regular_file(status)) continue;
            if (it->path().extension().string() == ".meta") metas.push_back(it->path());
        }
        if (ec) { error = "cannot scan archive directory " + archive_dir_.string(); return false; }
        std::sort(metas.begin(), metas.end());
        for (const auto& meta : metas) {
            Entry entry;
            entry.stem = meta;
            entry.stem.replace_extension();
            bool other_identity = false;
            std::string parse_error;
            if (!read_sidecar(meta, entry, other_identity, parse_error)) continue;   // left in place
            std::error_code sec;
            uint64_t total = std::filesystem::file_size(meta, sec);
            if (sec) total = 0;
            std::error_code kec;
            const uint64_t sess = std::filesystem::file_size(entry.session_path(), kec);
            if (!kec) {
                total += sess;
                for (size_t k = 1; k <= entry.stages; ++k) {
                    std::error_code pec;
                    const uint64_t part = std::filesystem::file_size(entry.stage_path(k), pec);
                    if (!pec) total += part;
                }
            }
            entry.file_bytes = total;
            entry.stamp = std::filesystem::last_write_time(meta, sec);
            if (sec) entry.stamp = {};
            archive_bytes_ += total;
            // seed the serial from the file's own number so a fresh archive stem never collides with an old one
            const std::string name = meta.filename().string();
            const std::string ext = ".meta";
            const std::string number = name.substr(std::strlen(kFilePrefix), name.size() - std::strlen(kFilePrefix) - ext.size());
            try {
                const uint64_t parsed = std::stoull(number);
                if (parsed > archive_serial_) archive_serial_ = parsed;
            } catch (...) {}
            archive_entries_.push_back(std::move(entry));
        }
        std::stable_sort(archive_entries_.begin(), archive_entries_.end(),
                         [](const Entry& a, const Entry& b) { return a.stamp < b.stamp; });
        archive_enabled_ = true;
        enforce_archive();
        return true;
    } catch (const std::exception& e) {
        archive_entries_.clear(); archive_bytes_ = 0; archive_enabled_ = false;
        error = std::string("cannot index archive directory: ") + e.what(); return false;
    }
}

std::filesystem::path ConversationSpillCache::next_archive_stem() {
    std::error_code ec;
    for (;;) {
        std::filesystem::path stem = archive_dir_ / (std::string(kFilePrefix) + std::to_string(++archive_serial_));
        if (!std::filesystem::exists(stem.string() + ".meta", ec) &&
            !std::filesystem::exists(stem.string() + ".sess", ec)) return stem;
        ec.clear();
    }
}

// How many bytes a tier entry would occupy in the archive: the sidecar (ids as int64 + images + checkpoints), plus
// the session files in state mode. Used to refuse archiving a copy that alone could not fit the archive budget.
uint64_t ConversationSpillCache::archive_entry_bytes(const Entry& entry) const {
    const uint64_t sidecar = 64 + (uint64_t) entry.live_meta.ids.size() * 8 +
                             (uint64_t) entry.live_meta.imgs.size() * 16 +
                             (uint64_t) entry.checkpoint_lengths.size() * 8;
    return archive_mode_ == SpillArchiveMode::state ? entry.file_bytes + sidecar : sidecar;
}

// The discard point shared by the compaction and the supersede pass (gate: "archivar en vez de borrar").
bool ConversationSpillCache::discard_entry(size_t index) {
    if (!archive_enabled_) return remove_entry(index);
    // A copy that ALONE exceeds the archive's own budget cannot be kept under it: archiving it would only write and
    // then delete it. Discard it as before instead, so the archive never holds more than its budget (the top can not
    // be exceeded by a single copy). The report documents this: an oversized conversation is not archived.
    if (archive_budget_ > 0 && archive_entry_bytes(entries_[index]) > archive_budget_) return remove_entry(index);
    return archive_entry(index);
}

// Moves the tier entry at `index` into the archive. The sidecar is written first (so a meta-only entry is already a
// valid archive entry); then, in state mode, the session and stage files are MOVED and, if any move fails, what
// already reached the archive is rolled back and the tier entry is left INTACT - the discard does not happen, so the
// tier is never left worse than before. In ids mode the session files are removed (the archive keeps the ids, not
// the K/V). A failed sidecar write leaves the tier entry untouched too.
bool ConversationSpillCache::archive_entry(size_t index) {
    const Entry entry = entries_[index];   // copy: the reference dies with entries_.erase below
    const std::string session = entry.session_path();
    const std::string meta = entry.meta_path();
    const size_t stages = entry.stages;
    const uint64_t file_bytes = entry.file_bytes;
    const std::filesystem::path stem = next_archive_stem();
    const std::string archive_meta = stem.string() + ".meta";
    const std::string archive_sess = stem.string() + ".sess";
    const bool keep_state = archive_mode_ == SpillArchiveMode::state;
    Entry archived = entry;
    archived.stem = stem;
    archived.stages = keep_state ? stages : 0;
    archived.file_bytes = 0;
    std::string write_error;
    if (!write_sidecar(archive_meta, archived, write_error)) return false;
    // O2 (audit): move every session file BEFORE the tier entry is touched. On any failure, put back what moved and
    // drop the archive sidecar, leaving the tier copy intact (a failed archive is not a discard).
    std::vector<std::pair<std::string, std::string>> moved;   // {archive path, original path}
    if (keep_state) {
        auto move_one = [&](const std::string& from, const std::string& to) -> bool {
            if (!relocate_file(from, to)) return false;
            moved.emplace_back(to, from);
            return true;
        };
        bool ok = move_one(session, archive_sess);
        for (size_t k = 1; ok && k <= stages; ++k) ok = move_one(stage_path(session, k), stage_path(archive_sess, k));
        if (!ok) {
            for (auto it = moved.rbegin(); it != moved.rend(); ++it) relocate_file(it->first, it->second);
            std::error_code rec;
            std::filesystem::remove(archive_meta, rec);
            std::filesystem::remove(archive_sess, rec);
            for (size_t k = 1; k <= stages; ++k) std::filesystem::remove(stage_path(archive_sess, k), rec);
            return false;
        }
    }
    std::error_code ec;
    // the tier's own files: the .meta was rewritten in the archive, so the original goes; in ids mode the session
    // and stage files go too (state mode moved them above)
    std::filesystem::remove(meta, ec);
    if (!keep_state) {
        std::filesystem::remove(session, ec);
        for (size_t k = 1; k <= stages; ++k) { ec.clear(); std::filesystem::remove(stage_path(session, k), ec); }
    }
    std::error_code mec;
    archived.file_bytes = std::filesystem::file_size(archive_meta, mec);
    if (mec) archived.file_bytes = 0;
    if (keep_state) {
        std::error_code sec;
        archived.file_bytes += std::filesystem::file_size(archive_sess, sec);
        for (size_t k = 1; k <= stages; ++k) {
            std::error_code kec;
            archived.file_bytes += std::filesystem::file_size(stage_path(archive_sess, k), kec);
        }
    }
    if (session == pinned_path_) pinned_path_.clear();
    bytes_ -= file_bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    archived.stamp = std::filesystem::last_write_time(archive_meta, ec);
    if (ec) archived.stamp = std::filesystem::file_time_type::clock::now();
    archive_bytes_ += archived.file_bytes;
    archive_entries_.push_back(std::move(archived));
    enforce_archive();
    return true;
}

// Gate refinement (a): the cancellation archives the state TAL COMO SE LEYÓ (pre-revert). Only the ids and images go
// (the K/V past the last turn boundary is the discarded answer's, not a clean prefix), so this is a sidecar-only
// entry. It is safe because the archive is never a reuse candidate.
bool ConversationSpillCache::archive_provisional(const std::vector<int32_t>& ids,
                                                 const std::vector<ConversationImageKey>& imgs,
                                                 const std::vector<size_t>& checkpoint_lengths, bool cvec,
                                                 std::string& error) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!archive_enabled_ || ids.empty()) return false;
    Entry entry;
    entry.stages = 0;
    entry.cvec = cvec;
    entry.live_meta.ids = ids;
    entry.live_meta.imgs = imgs;
    entry.checkpoint_lengths = checkpoint_lengths;
    if (!valid_meta_chain(entry.live_meta, entry.checkpoint_lengths)) {
        error = "cancelled request state is not a valid prefix chain"; return false;
    }
    entry.stem = next_archive_stem();
    const std::string meta = entry.stem.string() + ".meta";
    if (!write_sidecar(meta, entry, error)) return false;
    std::error_code ec;
    entry.file_bytes = std::filesystem::file_size(meta, ec);
    if (ec) entry.file_bytes = 0;
    entry.stamp = std::filesystem::last_write_time(meta, ec);
    if (ec) entry.stamp = std::filesystem::file_time_type::clock::now();
    archive_bytes_ += entry.file_bytes;
    archive_entries_.push_back(std::move(entry));
    enforce_archive();
    return true;
}

bool ConversationSpillCache::remove_archive_entry(size_t index) {
    const Entry& entry = archive_entries_[index];
    std::error_code ec;
    std::filesystem::remove(entry.meta_path(), ec);
    ec.clear();
    std::filesystem::remove(entry.session_path(), ec);
    for (size_t k = 1; k <= entry.stages; ++k) {
        ec.clear();
        std::filesystem::remove(stage_path(entry.session_path(), k), ec);
    }
    archive_bytes_ -= archive_bytes_ < entry.file_bytes ? archive_bytes_ : entry.file_bytes;
    archive_entries_.erase(archive_entries_.begin() + (std::ptrdiff_t) index);
    ++archive_evictions_;
    return true;
}

// Gate refinement (c): the archive's OWN GC. First the per-conversation keep (newest first), then the optional age
// retention, then the archive's own byte budget, oldest first. The budget bound is STRICT: if a single copy alone
// exceeds the whole budget it is removed too (an entry that could not fit is never archived in the first place;
// this covers a reopened archive whose budget shrank). It never touches the live tier's bytes_, disk_evictions_ or
// age_evictions_.
void ConversationSpillCache::enforce_archive() {
    if (archive_keep_ > 0) {
        std::map<uint64_t, size_t> seen;
        for (size_t i = archive_entries_.size(); i-- > 0;) {
            if (i >= archive_entries_.size()) continue;
            const uint64_t key = conversation_key(archive_entries_[i].live_meta.ids, archive_turn_token_);
            size_t& count = seen[key];
            if (++count > archive_keep_) remove_archive_entry(i);
        }
    }
    if (archive_max_age_days_ > 0) {
        const std::filesystem::file_time_type now = std::filesystem::file_time_type::clock::now();
        const std::chrono::seconds limit((int64_t) archive_max_age_days_ * 86400);
        for (size_t i = 0; i < archive_entries_.size();) {
            if (now - archive_entries_[i].stamp < limit) { ++i; continue; }   // the index does not advance on removal
            remove_archive_entry(i);
            ++archive_age_evictions_;
        }
    }
    while (archive_budget_ > 0 && archive_bytes_ > archive_budget_ && !archive_entries_.empty())
        remove_archive_entry(0);   // oldest first; the newest is kept only if it fits the budget
}

void ConversationSpillCache::pin(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);
    pinned_path_ = path;
}

void ConversationSpillCache::unpin(const std::string& path) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pinned_path_ == path) { pinned_path_.clear(); enforce_budget(); }
}

size_t ConversationSpillCache::drop_superseded(const std::vector<int32_t>& ids,
                                               const std::vector<ConversationImageKey>& images,
                                               const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                                               const std::string& keep) {
    std::lock_guard<std::mutex> lock(mu_);
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
        if (entry.cvec == cvec && deepest && held && entry.session_path() != pinned_path_ &&
            !(!keep.empty() && entry.session_path() == keep)) {
            if (!discard_entry(i)) { ++i; continue; }  // a failed move/removal: leave this entry in the index
            ++dropped;
            continue;
        }
        ++i;
    }
    return dropped;
}

size_t ConversationSpillCache::discard_diverged_impl(const std::vector<int32_t>& prompt, bool cvec,
                                                     size_t divergence_tokens, int64_t turn_token) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_ || prompt.empty() || turn_token < 0) return 0;
    size_t discarded = 0;
    for (size_t i = 0; i < entries_.size();) {
        const Entry& entry = entries_[i];
        if (entry.cvec != cvec) { ++i; continue; }                    // another steering mode: another conversation
        const std::vector<int32_t>& stored = entry.live_meta.ids;
        const size_t n = stored.size() < prompt.size() ? stored.size() : prompt.size();
        size_t common = 0;
        while (common < n && stored[common] == prompt[common]) ++common;
        if (common == stored.size()) { ++i; continue; }               // the prompt extends it: the normal turn
        const size_t header = conversation_header_length(stored, turn_token);
        if (header == 0 || common < header) { ++i; continue; }        // the headers differ: another conversation
        if (common >= divergence_tokens) { ++i; continue; }           // a small edit, not a rewritten tail
        if (entry.session_path() == pinned_path_) { ++i; continue; }
        if (!discard_entry(i)) { ++i; continue; }                     // a failed move/removal: leave it in the index
        ++discarded;
        ++compacted_;
    }
    return discarded;
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
