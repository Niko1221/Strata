// The system-prompt prefill cache: one ordinary session file per stage (a layer split) under a single key (a
// variant), and a small joint sidecar beside them holding the key, the prefix length, the stage count and the total
// file size, so a lookup never reads K/V. The session files are the same format the slot save/restore API writes, so
// a stage file and a hand-saved session are interchangeable and the same model/config identity rules refuse a
// foreign file.
//
// Causality: the key is the hash of the exact system-prompt token prefix (see the header). A variant is only ever
// attached to a prompt that begins with exactly those tokens; a changed system prompt is a different key, so its
// variant is reprocessed and rewritten, and the old one stays until the GC removes it (space or age, oldest first).
#include "strata/core/conversation_prompt_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <system_error>

namespace strata::core {
namespace {

constexpr char kMetaMagic[4] = {'S', 'P', 'C', '1'};   // Strata Prompt Cache
constexpr uint32_t kMetaVersion = 2;                   // v2 adds the stage count; v1 (single file) is still read
constexpr char kFilePrefix[] = "strata-prompt-";
// The fixed sidecar size without the digest, per version: v1 = model+config+key+tokens+file_bytes; v2 = the same
// plus the stage count. The digest is one more u64.
constexpr size_t kMetaBytesV1 = 4 + 4 + 8 + 8 + 8 + 8 + 8 + 8;   // magic+version+model+config+key+tokens+bytes+digest
constexpr size_t kMetaBytesV2 = kMetaBytesV1 + 8;
constexpr uint64_t kMaxStages = 64;                    // a layer split never has more GPUs than this

std::string hex16(uint64_t v) {
    char buffer[17];
    std::snprintf(buffer, sizeof buffer, "%016llx", (unsigned long long) v);
    return std::string(buffer, 16);
}

bool is_prompt_name(const std::string& name) { return name.rfind(kFilePrefix, 0) == 0; }

// The shared stem of every file of a variant: drop the ".sess" suffix and, when present, a trailing ".stageN".
std::string stem_string(const std::string& s) {
    std::string out = s;
    const std::string ext = ".sess";
    if (out.size() > ext.size() && out.compare(out.size() - ext.size(), ext.size(), ext) == 0)
        out.resize(out.size() - ext.size());
    const size_t at = out.rfind(".stage");
    if (at != std::string::npos) {
        const std::string tail = out.substr(at + 6);
        if (!tail.empty() && tail.find_first_not_of("0123456789") == std::string::npos) out.resize(at);
    }
    return out;
}

template<class T>
void put(std::string& out, T v) { out.append(reinterpret_cast<const char*>(&v), sizeof v); }

void put_digest(std::string& out) {
    SessionHasher h(0);
    h.update(out.data(), out.size());
    const uint64_t d = h.digest();
    put(out, d);
}

} // namespace

std::filesystem::path ConversationPromptCache::variant_file(uint64_t key) const {
    return directory_ / (std::string(kFilePrefix) + hex16(key) + ".sess");
}

std::filesystem::path ConversationPromptCache::stage_file(uint64_t key, size_t k) const {
    if (k == 0) return variant_file(key);
    return directory_ / (std::string(kFilePrefix) + hex16(key) + ".stage" + std::to_string(k) + ".sess");
}

std::filesystem::path ConversationPromptCache::meta_file(uint64_t key) const {
    return directory_ / (std::string(kFilePrefix) + hex16(key) + ".meta");
}

std::filesystem::path ConversationPromptCache::Entry::stage_path(size_t k) const {
    if (k == 0) return session;
    return std::filesystem::path(stem_string(session.string()) + ".stage" + std::to_string(k) + ".sess");
}

ConversationPromptCache::Entry* ConversationPromptCache::find(uint64_t key) {
    for (Entry& e : entries_) if (e.key == key) return &e;
    return nullptr;
}

const ConversationPromptCache::Entry* ConversationPromptCache::find(uint64_t key) const {
    for (const Entry& e : entries_) if (e.key == key) return &e;
    return nullptr;
}

const ConversationPromptCache::Entry* ConversationPromptCache::find_path(const std::string& path) const {
    for (const Entry& e : entries_) if (e.session.string() == path) return &e;
    return nullptr;
}

bool ConversationPromptCache::read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity,
                                           std::string& error) const {
    other_identity = false;
    std::error_code ec;
    const uint64_t file_bytes = std::filesystem::file_size(meta, ec);
    if (ec || file_bytes < kMetaBytesV1) { error = "prompt cache sidecar is missing or too short"; return false; }
    std::ifstream in(meta, std::ios::binary);
    if (!in) { error = "cannot open prompt cache sidecar"; return false; }
    std::string raw((size_t) std::min<uint64_t>(file_bytes, kMetaBytesV2), '\0');
    in.read(raw.data(), (std::streamsize) raw.size());
    if (!in || in.gcount() != (std::streamsize) raw.size()) { error = "truncated prompt cache sidecar"; return false; }
    size_t at = 0;
    auto take = [&](void* p, size_t n) { std::memcpy(p, raw.data() + at, n); at += n; };
    char magic[4];
    uint32_t version = 0;
    take(magic, 4); take(&version, 4);
    if (std::memcmp(magic, kMetaMagic, 4) != 0 || (version != 1 && version != kMetaVersion)) {
        error = "unknown prompt cache sidecar magic or version"; return false;
    }
    uint64_t model = 0, config = 0, key = 0, stored_bytes = 0, digest = 0, stages = 1;
    int64_t tokens = 0;
    take(&model, 8); take(&config, 8); take(&key, 8); take(&tokens, 8); take(&stored_bytes, 8);
    if (version >= 2) take(&stages, 8);
    take(&digest, 8);
    SessionHasher h(0);
    h.update(raw.data(), at - 8);
    if (h.digest() != digest) { error = "prompt cache sidecar checksum mismatch"; return false; }
    if (model != identity_.model || config != identity_.config) { other_identity = true; return false; }
    if (tokens <= 0) { error = "invalid prompt cache token count"; return false; }
    if (stages < 1 || stages > kMaxStages) { error = "invalid prompt cache stage count"; return false; }
    entry.key = key;
    entry.tokens = tokens;
    entry.stages = (size_t) stages;
    entry.file_bytes = stored_bytes;   // advisory: the scan re-sums the actual files
    return true;
}

bool ConversationPromptCache::write_sidecar(const Entry& entry, std::string& error) const {
    std::string raw;
    raw.reserve(kMetaBytesV2);
    raw.append(kMetaMagic, 4);
    put(raw, (uint32_t) kMetaVersion);
    put(raw, (uint64_t) identity_.model);
    put(raw, (uint64_t) identity_.config);
    put(raw, (uint64_t) entry.key);
    put(raw, (int64_t) entry.tokens);
    put(raw, (uint64_t) entry.file_bytes);
    put(raw, (uint64_t) entry.stages);
    put_digest(raw);
    const std::filesystem::path temporary(entry.meta.string() + ".tmp");
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { error = "cannot create prompt cache sidecar"; return false; }
        out.write(raw.data(), (std::streamsize) raw.size());
        out.flush();
        if (!out) { error = "cannot write prompt cache sidecar"; return false; }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, entry.meta, ec);
    if (ec) { std::filesystem::remove(temporary, ec); error = "cannot commit prompt cache sidecar"; return false; }
    return true;
}

bool ConversationPromptCache::open(const std::filesystem::path& directory, SessionFileIdentity identity,
                                   uint64_t budget_bytes, int64_t slots, int64_t max_age_days, std::string& error) {
    enabled_ = false;
    entries_.clear();
    bytes_ = 0;
    hits_ = misses_ = tokens_saved_ = hash_changes_ = 0;
    evicted_by_space_ = evicted_by_age_ = 0;
    foreign_files_kept_ = stale_files_kept_ = orphan_files_kept_ = oversized_files_kept_ = 0;
    directory_ = directory;
    identity_ = identity;
    budget_ = budget_bytes;
    slots_ = slots;
    max_age_days_ = max_age_days < 0 ? 0 : max_age_days;
    if (directory.empty() || !identity.model || !identity.config) return true;
    try {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        if (ec || !std::filesystem::is_directory(directory_, ec) || ec) {
            error = "cannot create or inspect prompt cache directory " + directory_.string(); return false;
        }
        // The scan only reads. A leftover temporary, a broken sidecar, a foreign identity and an orphan session file
        // are counted and left in place; removal is the GC's, below.
        std::vector<std::filesystem::path> metas, session_files;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) break;
            const std::string name = it->path().filename().string();
            if (!is_prompt_name(name)) continue;
            if (!std::filesystem::is_regular_file(status)) continue;
            const std::string ext = it->path().extension().string();
            if (ext == ".meta") metas.push_back(it->path());
            else if (ext == ".sess") session_files.push_back(it->path());
            else ++stale_files_kept_;   // a leftover temporary or an unknown kind: reported, never removed
        }
        if (ec) { error = "cannot scan prompt cache directory " + directory_.string(); return false; }
        std::sort(metas.begin(), metas.end());
        for (const auto& meta : metas) {
            Entry entry;
            entry.meta = meta;
            bool other_identity = false;
            std::string parse_error;
            if (!read_sidecar(meta, entry, other_identity, parse_error)) {
                if (other_identity) ++foreign_files_kept_;
                else ++stale_files_kept_;
                continue;   // ignored, not removed
            }
            entry.session = variant_file(entry.key);
            std::error_code sec;
            const uint64_t session_bytes = std::filesystem::file_size(entry.session, sec);
            if (sec || session_bytes == 0) { ++stale_files_kept_; continue; }   // first file gone: the sidecar is useless, kept
            uint64_t total = session_bytes;
            bool complete = true;
            for (size_t k = 1; k < entry.stages; ++k) {
                std::error_code kec;
                const uint64_t part = std::filesystem::file_size(stage_file(entry.key, k), kec);
                if (kec || part == 0) { complete = false; break; }   // a missing stage file: the variant is incomplete
                total += part;
            }
            if (!complete) { ++stale_files_kept_; continue; }   // kept, never removed: a later write may complete it
            entry.file_bytes = total;
            entry.stamp = std::filesystem::last_write_time(entry.session, sec);
            if (sec) entry.stamp = {};
            if (budget_ && total > budget_) ++oversized_files_kept_;   // KEPT: the GC leaves it alone too
            entries_.push_back(std::move(entry));
            bytes_ += total;
        }
        // A session file (of any stage) with no sidecar left behind (a crash between the writes) would never be
        // matched: report it, do not delete it.
        for (const auto& path : session_files) {
            const std::filesystem::path meta = stem_string(path.string()) + ".meta";
            if (!std::filesystem::exists(meta, ec) && !ec) ++orphan_files_kept_;
            ec.clear();
        }
        enabled_ = true;
        enforce();
        return true;
    } catch (const std::bad_alloc&) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = "not enough RAM to index prompt cache directory"; return false;
    } catch (const std::exception& e) {
        entries_.clear(); bytes_ = 0; enabled_ = false;
        error = std::string("cannot index prompt cache directory: ") + e.what(); return false;
    }
}

uint64_t ConversationPromptCache::key_for(const int64_t* tokens, size_t count, const std::string& declared) const {
    SessionIdentityBuilder b(0x5350524f4d505443ull);   // "SPROMPTC"
    b.str("kind", "system-prompt-prefix");
    b.u64("model", identity_.model);
    b.u64("config", identity_.config);
    b.str("declared", declared);
    b.bytes("tokens", tokens, count * sizeof(int64_t));
    return b.digest();
}

bool ConversationPromptCache::lookup(uint64_t key, ConversationPromptMatch& out) const {
    out = {};
    if (!enabled_) return false;
    const Entry* entry = find(key);
    if (entry == nullptr) return false;
    out.path = entry->session.string();
    out.file_bytes = entry->file_bytes;
    out.tokens = entry->tokens;
    out.stages = entry->stages;
    return true;
}

void ConversationPromptCache::note_hit(uint64_t key, int64_t tokens) {
    if (!enabled_) return;
    Entry* entry = find(key);
    if (entry != nullptr) ++entry->hits;
    ++hits_;
    if (tokens > 0) tokens_saved_ += (uint64_t) tokens;
}

void ConversationPromptCache::note_miss(uint64_t key) {
    (void) key;
    if (!enabled_) return;
    ++misses_;
}

bool ConversationPromptCache::erase(uint64_t key, std::string& error) {
    if (!enabled_) return false;
    Entry* entry = find(key);
    if (entry == nullptr) return true;
    const size_t index = (size_t) (entry - entries_.data());
    if (!remove_entry(index)) { error = "cannot remove prompt cache variant"; return false; }
    return true;
}

bool ConversationPromptCache::index_written(uint64_t key, int64_t tokens, size_t stages, uint64_t file_bytes,
                                            std::string& error) {
    std::error_code ec;
    const std::filesystem::path session = variant_file(key);
    Entry entry;
    entry.key = key;
    entry.tokens = tokens;
    entry.stages = stages;
    entry.file_bytes = file_bytes;
    entry.session = session;
    entry.meta = meta_file(key);
    entry.stamp = std::filesystem::last_write_time(session, ec);
    if (ec) entry.stamp = {};
    if (!write_sidecar(entry, error)) return false;
    Entry* existing = find(key);
    if (existing == nullptr) {
        // A new key beside a stored variant: the system prompt changed (or a second one is in use). Counted.
        if (!entries_.empty()) ++hash_changes_;
        try {
            entries_.push_back(std::move(entry));
        } catch (const std::bad_alloc&) {
            std::error_code rec;
            for (size_t k = 0; k < stages; ++k) std::filesystem::remove(stage_file(key, k), rec);
            std::filesystem::remove(meta_file(key), rec);
            error = "not enough RAM to index the prompt cache variant"; return false;
        }
        bytes_ += file_bytes;
    } else {
        // The same key rewritten (a crash between writes, or a re-store): drop the stage files it no longer has.
        for (size_t k = stages; k < existing->stages; ++k) {
            std::error_code rec;
            std::filesystem::remove(stage_file(key, k), rec);
        }
        bytes_ -= existing->file_bytes;
        *existing = std::move(entry);
        bytes_ += file_bytes;
    }
    if (budget_ && file_bytes > budget_) ++oversized_files_kept_;
    enforce();
    return true;
}

bool ConversationPromptCache::store(uint64_t key, const SavedConversation& image, std::string& error) {
    if (!enabled_) return false;
    std::vector<std::string> written;
    auto discard = [&] {
        std::error_code rec;
        for (const std::string& f : written) std::filesystem::remove(f, rec);
        written.clear();
    };
    auto write = [&](const std::filesystem::path& path, const SavedConversation& part, uint64_t& total) -> bool {
        size_t bytes = 0;
        SessionWriteOptions wo;
        wo.durable = true;
        SessionStatus st;
        if (!session_file_write(path.string(), part, identity_, bytes, error, wo, &st)) return false;
        total += bytes;
        written.push_back(path.string());
        return true;
    };
    uint64_t total = 0;
    if (!write(stage_file(key, 0), image, total)) { discard(); return false; }
    for (size_t k = 0; k < image.stage_images.size(); ++k)
        if (!write(stage_file(key, k + 1), image.stage_images[k], total)) { discard(); return false; }
    if (!index_written(key, (int64_t) image.live.ids.size(), 1 + image.stage_images.size(), total, error)) {
        discard();
        return false;
    }
    return true;
}

bool ConversationPromptCache::store_streamed(uint64_t key, const SavedConversation& meta,
                                             const std::vector<SessionKvSource>& sources, std::string& error) {
    std::vector<SavedConversation> metas;
    std::vector<std::vector<SessionKvSource>> stage_sources;
    metas.push_back(meta);
    stage_sources.push_back(sources);
    return store_streamed(key, metas, stage_sources, error);
}

bool ConversationPromptCache::store_streamed(uint64_t key, const std::vector<SavedConversation>& stage_metas,
                                             const std::vector<std::vector<SessionKvSource>>& stage_sources,
                                             std::string& error) {
    if (!enabled_) return false;
    if (stage_metas.empty() || stage_metas.size() != stage_sources.size()) {
        error = "prompt cache store needs one meta and its sources per stage"; return false;
    }
    std::vector<std::string> written;
    auto discard = [&] {
        std::error_code rec;
        for (const std::string& f : written) std::filesystem::remove(f, rec);
        written.clear();
    };
    uint64_t total = 0;
    for (size_t k = 0; k < stage_metas.size(); ++k) {
        const std::filesystem::path path = stage_file(key, k);
        size_t bytes = 0;
        SessionWriteOptions wo;
        wo.durable = true;
        SessionStatus st;
        if (!session_file_write(path.string(), stage_metas[k], stage_sources[k], identity_, bytes, error, wo, &st)) {
            discard();
            return false;
        }
        total += bytes;
        written.push_back(path.string());
    }
    // The sidecar is committed LAST: until it is there the variant does not exist, so a crash between the writes
    // leaves orphan session files (reported, never matched) rather than a half variant.
    if (!index_written(key, (int64_t) stage_metas[0].live.ids.size(), stage_metas.size(), total, error)) {
        discard();
        return false;
    }
    return true;
}

bool ConversationPromptCache::load(const std::string& path, SavedConversation& image,
                                   const std::vector<SessionReadLimits>& limits, std::string& error) const {
    if (!enabled_) { error = "prompt cache is disabled"; return false; }
    const Entry* entry = find_path(path);
    const size_t stages = entry != nullptr ? entry->stages : 1;
    const SessionReadLimits open_limits;
    auto for_stage = [&](size_t i) -> const SessionReadLimits& {
        if (limits.empty()) return open_limits;
        return limits[i < limits.size() ? i : limits.size() - 1];
    };
    // Every stage file is read and validated into a LOCAL image first; `image` is replaced only once all of them
    // read. A missing, foreign or corrupt stage therefore refuses the whole variant with no partial state applied.
    SavedConversation main;
    size_t bytes = 0;
    if (!session_file_read(path, identity_, main, bytes, error, for_stage(0))) return false;
    std::vector<SavedConversation> stage_images;
    stage_images.reserve(stages > 1 ? stages - 1 : 0);
    for (size_t k = 1; k < stages; ++k) {
        SavedConversation part;
        size_t part_bytes = 0;
        const std::string file = entry->stage_path(k).string();
        if (!session_file_read(file, identity_, part, part_bytes, error, for_stage(k))) return false;
        stage_images.push_back(std::move(part));
    }
    main.stage_images = std::move(stage_images);
    image = std::move(main);
    return true;
}

bool ConversationPromptCache::remove_entry(size_t index) {
    const Entry& entry = entries_[index];
    std::error_code ec;
    const bool removed = std::filesystem::remove(entry.session, ec);
    ec.clear();
    if (!removed && std::filesystem::exists(entry.session)) return false;
    for (size_t k = 1; k < entry.stages; ++k) std::filesystem::remove(entry.stage_path(k), ec);
    ec.clear();
    std::filesystem::remove(entry.meta, ec);
    bytes_ -= entry.file_bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    return true;
}

void ConversationPromptCache::enforce() {
    // oldest first by write time; a few entries, so sorting on each pass is cheap and keeps the order honest.
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.stamp < b.stamp; });
    if (max_age_days_ > 0) {
        const std::filesystem::file_time_type now = std::filesystem::file_time_type::clock::now();
        const std::chrono::seconds limit((int64_t) max_age_days_ * 86400);
        for (size_t i = 0; i < entries_.size();) {
            if (now - entries_[i].stamp < limit) { ++i; continue; }
            if (!remove_entry(i)) { ++i; continue; }
            ++evicted_by_age_;
        }
    }
    auto space_evict = [&](bool by_slots) -> bool {
        // oldest variant that can actually be removed (one larger than the whole budget cannot bring bytes_ under it)
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (budget_ && entries_[i].file_bytes > budget_ && !by_slots) continue;
            if (!remove_entry(i)) continue;
            ++evicted_by_space_;
            return true;
        }
        return false;
    };
    if (slots_ > 0)
        while ((int64_t) entries_.size() > slots_) if (!space_evict(true)) break;
    while (budget_ > 0 && bytes_ > budget_) if (!space_evict(false)) break;
}

} // namespace strata::core
