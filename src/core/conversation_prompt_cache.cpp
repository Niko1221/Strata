// The system-prompt prefill cache: one ordinary session file per system-prompt prefix (a variant), a small sidecar
// beside it holding the key, the prefix length and the file size, so a lookup never reads K/V. The session file is
// the same format the slot save/restore API writes, so a variant and a hand-saved session are interchangeable and
// the same model/config identity rules refuse a foreign file.
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
constexpr uint32_t kMetaVersion = 1;
constexpr char kFilePrefix[] = "strata-prompt-";
constexpr size_t kMetaBytes = 4 + 4 + 8 + 8 + 8 + 8 + 8 + 8;   // magic + version + model + config + key + tokens + file_bytes + digest

std::string hex16(uint64_t v) {
    char buffer[17];
    std::snprintf(buffer, sizeof buffer, "%016llx", (unsigned long long) v);
    return std::string(buffer, 16);
}

bool is_prompt_name(const std::string& name) { return name.rfind(kFilePrefix, 0) == 0; }

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

std::filesystem::path ConversationPromptCache::meta_file(uint64_t key) const {
    return directory_ / (std::string(kFilePrefix) + hex16(key) + ".meta");
}

ConversationPromptCache::Entry* ConversationPromptCache::find(uint64_t key) {
    for (Entry& e : entries_) if (e.key == key) return &e;
    return nullptr;
}

const ConversationPromptCache::Entry* ConversationPromptCache::find(uint64_t key) const {
    for (const Entry& e : entries_) if (e.key == key) return &e;
    return nullptr;
}

bool ConversationPromptCache::read_sidecar(const std::filesystem::path& meta, Entry& entry, bool& other_identity,
                                           std::string& error) const {
    other_identity = false;
    std::error_code ec;
    const uint64_t file_bytes = std::filesystem::file_size(meta, ec);
    if (ec || file_bytes < kMetaBytes) { error = "prompt cache sidecar is missing or too short"; return false; }
    std::ifstream in(meta, std::ios::binary);
    if (!in) { error = "cannot open prompt cache sidecar"; return false; }
    std::string raw(kMetaBytes, '\0');
    in.read(raw.data(), (std::streamsize) raw.size());
    if (!in || in.gcount() != (std::streamsize) raw.size()) { error = "truncated prompt cache sidecar"; return false; }
    size_t at = 0;
    auto take = [&](void* p, size_t n) { std::memcpy(p, raw.data() + at, n); at += n; };
    char magic[4];
    uint32_t version = 0;
    uint64_t model = 0, config = 0, key = 0, stored_bytes = 0, digest = 0;
    int64_t tokens = 0;
    take(magic, 4); take(&version, 4); take(&model, 8); take(&config, 8); take(&key, 8); take(&tokens, 8);
    take(&stored_bytes, 8); take(&digest, 8);
    if (std::memcmp(magic, kMetaMagic, 4) != 0 || version != kMetaVersion) {
        error = "unknown prompt cache sidecar magic or version"; return false;
    }
    SessionHasher h(0);
    h.update(raw.data(), at - 8);
    if (h.digest() != digest) { error = "prompt cache sidecar checksum mismatch"; return false; }
    if (model != identity_.model || config != identity_.config) { other_identity = true; return false; }
    if (tokens <= 0) { error = "invalid prompt cache token count"; return false; }
    entry.key = key;
    entry.tokens = tokens;
    entry.file_bytes = stored_bytes;
    return true;
}

bool ConversationPromptCache::write_sidecar(const Entry& entry, std::string& error) const {
    std::string raw;
    raw.reserve(kMetaBytes);
    raw.append(kMetaMagic, 4);
    put(raw, (uint32_t) kMetaVersion);
    put(raw, (uint64_t) identity_.model);
    put(raw, (uint64_t) identity_.config);
    put(raw, (uint64_t) entry.key);
    put(raw, (int64_t) entry.tokens);
    put(raw, (uint64_t) entry.file_bytes);
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
            if (sec || session_bytes == 0) { ++stale_files_kept_; continue; }   // session gone: the sidecar is useless, kept
            entry.file_bytes = session_bytes;
            entry.stamp = std::filesystem::last_write_time(entry.session, sec);
            if (sec) entry.stamp = {};
            if (budget_ && session_bytes > budget_) ++oversized_files_kept_;   // KEPT: the GC leaves it alone too
            entries_.push_back(std::move(entry));
            bytes_ += session_bytes;
        }
        // A session file with no sidecar left behind (a crash between the two writes) would never be matched.
        for (const auto& path : session_files) {
            std::filesystem::path meta = path;
            meta.replace_extension(".meta");
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

bool ConversationPromptCache::index_written(uint64_t key, int64_t tokens, uint64_t file_bytes, std::string& error) {
    std::error_code ec;
    const std::filesystem::path session = variant_file(key);
    Entry entry;
    entry.key = key;
    entry.tokens = tokens;
    entry.file_bytes = file_bytes;
    entry.session = session;
    entry.meta = meta_file(key);
    entry.stamp = std::filesystem::last_write_time(session, ec);
    if (ec) entry.stamp = {};
    if (!write_sidecar(entry, error)) return false;
    const bool known = find(key) != nullptr;
    if (!known) {
        // A new key beside a stored variant: the system prompt changed (or a second one is in use). Counted.
        if (!entries_.empty()) ++hash_changes_;
        try {
            entries_.push_back(std::move(entry));
        } catch (const std::bad_alloc&) {
            std::error_code rec;
            std::filesystem::remove(session, rec);
            std::filesystem::remove(meta_file(key), rec);
            error = "not enough RAM to index the prompt cache variant"; return false;
        }
        bytes_ += file_bytes;
    } else {
        Entry* existing = find(key);
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
    const std::filesystem::path session = variant_file(key);
    size_t bytes = 0;
    SessionWriteOptions wo;
    wo.durable = true;
    SessionStatus st;
    if (!session_file_write(session.string(), image, identity_, bytes, error, wo, &st)) return false;
    return index_written(key, (int64_t) image.live.ids.size(), bytes, error);
}

bool ConversationPromptCache::store_streamed(uint64_t key, const SavedConversation& meta,
                                             const std::vector<SessionKvSource>& sources, std::string& error) {
    if (!enabled_) return false;
    const std::filesystem::path session = variant_file(key);
    size_t bytes = 0;
    SessionWriteOptions wo;
    wo.durable = true;
    SessionStatus st;
    if (!session_file_write(session.string(), meta, sources, identity_, bytes, error, wo, &st)) return false;
    return index_written(key, (int64_t) meta.live.ids.size(), bytes, error);
}

bool ConversationPromptCache::load(const std::string& path, SavedConversation& image,
                                   const std::vector<SessionReadLimits>& limits, std::string& error) const {
    if (!enabled_) { error = "prompt cache is disabled"; return false; }
    size_t bytes = 0;
    if (limits.empty()) return session_file_read(path, identity_, image, bytes, error);
    return session_file_read(path, identity_, image, bytes, error, limits[0]);
}

bool ConversationPromptCache::remove_entry(size_t index) {
    const Entry& entry = entries_[index];
    std::error_code ec;
    const bool removed = std::filesystem::remove(entry.session, ec);
    ec.clear();
    if (!removed && std::filesystem::exists(entry.session)) return false;
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
