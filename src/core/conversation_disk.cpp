// src/core/conversation_disk.cpp - the L3 disk tier. CPU only: this file includes no CUDA header.
//
// File layout, from offset zero:
//   header   a fixed 96-byte block: magic, version, sizes, checksums, identity, cvec
//   metadata the record's shape: every scalar, every id and image key, and every section length
//   payload  the variable-length arrays back to back, in the metadata's section order
//
// The write is one pass over the payload. The header is patched with the payload checksum after
// the payload lands, then the temporary file is renamed over the target.
//
// The read validates the header and the metadata before it allocates any payload buffer. It then
// streams the payload into the destination buffers and verifies the checksum on the way.
#include "strata/core/conversation_disk.hpp"
#include <new>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <system_error>

namespace strata::core {
namespace {

// "STRCDISK", little-endian.
constexpr uint64_t kMagic = 0x4B53494443525453ull;
constexpr uint32_t kHeaderBytes = 96;
constexpr uint64_t kMaxMetadataBytes = 64ull << 20;
constexpr uint32_t kMaxStages = 1024;
constexpr uint32_t kMaxCheckpoints = 1u << 20;
constexpr uint32_t kMaxParts = 64;
constexpr uint32_t kMaxKv = 1u << 16;
constexpr uint32_t kMaxPartDepth = 4;
constexpr size_t kMaxNameBytes = 96;

struct Header {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint64_t record_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint64_t metadata_checksum = 0;
    uint64_t payload_bytes = 0;
    uint64_t payload_checksum = 0;
    uint8_t identity[ConversationDiskIdentity::size] = {};
    uint8_t cvec = 0;
    uint8_t reserved[7] = {};
};
static_assert(sizeof(Header) == kHeaderBytes, "the disk header is a fixed 96 bytes");

bool add_overflow(uint64_t& total, uint64_t extra) {
    if (extra > UINT64_MAX - total) return false;
    total += extra;
    return true;
}

bool read_exact(std::FILE* file, void* destination, uint64_t bytes) {
    uint8_t* out = static_cast<uint8_t*>(destination);
    while (bytes != 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes, 1u << 20));
        if (std::fread(out, 1, chunk, file) != chunk) return false;
        out += chunk;
        bytes -= chunk;
    }
    return true;
}

// One variable-length payload section: either a plain byte vector or a segmented buffer.
struct Section {
    std::vector<uint8_t>* bytes = nullptr;
    ConversationBuffer* buffer = nullptr;
};

template<class Fn>
void visit_checkpoint(ConversationCheckpoint& checkpoint, Fn& fn) {
    fn(Section{&checkpoint.gdn, nullptr});
    fn(Section{&checkpoint.ple, nullptr});
    fn(Section{&checkpoint.tails, nullptr});
    fn(Section{&checkpoint.dead, nullptr});
    fn(Section{&checkpoint.block_pos, nullptr});
    for (auto& part : checkpoint.stage_parts) visit_checkpoint(part, fn);
}

// Visits every payload section in file order. The metadata writer and reader use the same order.
template<class Fn>
void visit_payload(ConversationDiskRecord& record, Fn& fn) {
    for (auto& stage : record.stages) {
        visit_checkpoint(stage.live, fn);
        for (auto& checkpoint : stage.checkpoints) visit_checkpoint(checkpoint, fn);
        for (auto& kv : stage.kv) {
            fn(Section{nullptr, &kv.k});
            fn(Section{nullptr, &kv.v});
            fn(Section{nullptr, &kv.k_scale});
            fn(Section{nullptr, &kv.v_scale});
            fn(Section{nullptr, &kv.pooled});
        }
    }
}

uint64_t count_sections(ConversationDiskRecord& record) {
    uint64_t count = 0;
    auto fn = [&](Section) { ++count; };
    visit_payload(record, fn);
    return count;
}

// The match points of a record: the ids and image keys of the deepest stage's live checkpoint and
// of every checkpoint below it. A stage part repeats its parent's ids and images, so it adds none.
void points_of(const ConversationDiskRecord& record, std::vector<ConversationCheckpoint>& out) {
    const SavedConversation* deepest = nullptr;
    for (const auto& stage : record.stages)
        if (!deepest || stage.live.ids.size() > deepest->live.ids.size()) deepest = &stage;
    if (deepest == nullptr) return;
    auto add = [&](const ConversationCheckpoint& checkpoint) {
        ConversationCheckpoint point;
        point.ids = checkpoint.ids;
        point.imgs = checkpoint.imgs;
        out.push_back(std::move(point));
    };
    add(deepest->live);
    for (const auto& checkpoint : deepest->checkpoints) add(checkpoint);
}

// Writes the record's shape and collects the section lengths in payload order.
struct MetaWriter {
    std::vector<uint8_t> data;
    std::vector<uint64_t> lengths;

    void raw(const void* source, size_t bytes) {
        const uint8_t* first = static_cast<const uint8_t*>(source);
        data.insert(data.end(), first, first + bytes);
    }
    void u8(uint8_t value) { data.push_back(value); }
    void u32(uint32_t value) { raw(&value, sizeof(value)); }
    void u64(uint64_t value) { raw(&value, sizeof(value)); }
    void i32(int32_t value) { raw(&value, sizeof(value)); }
    void i64(int64_t value) { raw(&value, sizeof(value)); }
    void section(uint64_t bytes) { u64(bytes); lengths.push_back(bytes); }

    void write_checkpoint(const ConversationCheckpoint& value) {
        u32(static_cast<uint32_t>(value.ids.size()));
        if (!value.ids.empty()) raw(value.ids.data(), value.ids.size() * sizeof(int32_t));
        u32(static_cast<uint32_t>(value.imgs.size()));
        for (const auto& key : value.imgs) {
            i64(key.start);
            u64(key.hash);
        }
        u64(value.used);
        section(value.gdn.size());
        section(value.ple.size());
        section(value.tails.size());
        section(value.dead.size());
        section(value.block_pos.size());
        u32(static_cast<uint32_t>(value.stage_parts.size()));
        for (const auto& part : value.stage_parts) write_checkpoint(part);
    }

    void write_kv(const ConversationKv& layer) {
        i32(layer.format);
        i64(layer.cells);
        i64(layer.heads);
        i64(layer.head_dim);
        i64(layer.page_size);
        i64(layer.pooled_rows);
        i64(layer.idx_dim);
        section(layer.k.size());
        section(layer.v.size());
        section(layer.k_scale.size());
        section(layer.v_scale.size());
        section(layer.pooled.size());
    }

    void write_stage(const SavedConversation& image) {
        i64(image.layer_lo);
        i64(image.layer_hi);
        for (int64_t value : image.geometry) i64(value);
        u8(image.cvec ? 1 : 0);
        write_checkpoint(image.live);
        u32(static_cast<uint32_t>(image.checkpoints.size()));
        for (const auto& checkpoint : image.checkpoints) write_checkpoint(checkpoint);
        u32(static_cast<uint32_t>(image.kv.size()));
        for (const auto& layer : image.kv) write_kv(layer);
    }
};

// Reads the record's shape and collects the section lengths in payload order. It allocates only
// ids and image keys, never a payload buffer.
struct MetaReader {
    const uint8_t* cursor = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;
    std::vector<uint64_t> lengths;

    size_t remaining() const { return ok ? static_cast<size_t>(end - cursor) : 0; }
    bool need(size_t bytes) {
        if (!ok || remaining() < bytes) {
            ok = false;
            return false;
        }
        return true;
    }
    uint8_t u8() { return need(1) ? *cursor++ : 0; }
    uint32_t u32() {
        uint32_t value = 0;
        if (need(4)) { std::memcpy(&value, cursor, 4); cursor += 4; }
        return value;
    }
    uint64_t u64() {
        uint64_t value = 0;
        if (need(8)) { std::memcpy(&value, cursor, 8); cursor += 8; }
        return value;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    int64_t i64() { return static_cast<int64_t>(u64()); }
    bool section(uint64_t& bytes) {
        bytes = u64();
        if (!ok) return false;
        lengths.push_back(bytes);
        return true;
    }

    bool read_checkpoint(ConversationCheckpoint& out, uint32_t depth) {
        if (depth > kMaxPartDepth) { ok = false; return false; }
        const uint32_t ids = u32();
        if (!ok || ids > remaining() / sizeof(int32_t)) { ok = false; return false; }
        out.ids.resize(ids);
        if (ids != 0) {
            std::memcpy(out.ids.data(), cursor, ids * sizeof(int32_t));
            cursor += ids * sizeof(int32_t);
        }
        const uint32_t imgs = u32();
        if (!ok || imgs > remaining() / 16) { ok = false; return false; }
        out.imgs.resize(imgs);
        for (uint32_t i = 0; i < imgs; ++i) {
            out.imgs[i].start = i64();
            out.imgs[i].hash = u64();
        }
        out.used = u64();
        uint64_t bytes = 0;
        for (int i = 0; i < 5; ++i)
            if (!section(bytes)) return false;
        const uint32_t parts = u32();
        if (!ok || parts > kMaxParts || parts > remaining() / 8) { ok = false; return false; }
        out.stage_parts.resize(parts);
        for (uint32_t i = 0; i < parts; ++i)
            if (!read_checkpoint(out.stage_parts[i], depth + 1)) return false;
        return ok;
    }

    bool read_kv(ConversationKv& out) {
        out.format = i32();
        out.cells = i64();
        out.heads = i64();
        out.head_dim = i64();
        out.page_size = i64();
        out.pooled_rows = i64();
        out.idx_dim = i64();
        if (!ok) return false;
        uint64_t bytes = 0;
        for (int i = 0; i < 5; ++i)
            if (!section(bytes)) return false;
        return ok;
    }

    bool read_stage(SavedConversation& out) {
        out.layer_lo = i64();
        out.layer_hi = i64();
        for (int i = 0; i < 18; ++i) out.geometry[i] = i64();
        out.cvec = u8() != 0;
        if (!read_checkpoint(out.live, 0)) return false;
        const uint32_t checkpoints = u32();
        if (!ok || checkpoints > kMaxCheckpoints || checkpoints > remaining() / 24) { ok = false; return false; }
        out.checkpoints.resize(checkpoints);
        for (uint32_t i = 0; i < checkpoints; ++i)
            if (!read_checkpoint(out.checkpoints[i], 0)) return false;
        const uint32_t kv = u32();
        if (!ok || kv > kMaxKv || kv > remaining() / 40) { ok = false; return false; }
        out.kv.resize(kv);
        for (uint32_t i = 0; i < kv; ++i)
            if (!read_kv(out.kv[i])) return false;
        return ok;
    }

    bool parse(ConversationDiskRecord& out) {
        const uint32_t stages = u32();
        if (!ok || stages == 0 || stages > kMaxStages) { ok = false; return false; }
        out.stages.resize(stages);
        for (uint32_t i = 0; i < stages; ++i)
            if (!read_stage(out.stages[i])) return false;
        return ok;
    }
};

std::string hex64(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

// Validates the parsed metadata against the payload size. No allocation.
bool valid_lengths(ConversationDiskRecord& record, const std::vector<uint64_t>& lengths,
                   uint64_t payload_bytes) {
    uint64_t total = 0;
    for (const uint64_t bytes : lengths) {
        if (total > payload_bytes || bytes > payload_bytes - total) return false;
        total += bytes;
    }
    if (total != payload_bytes) return false;
    return count_sections(record) == lengths.size();
}

// Publishes one record: header, metadata, payload, then the rename. One pass over the payload.
ConversationDiskStatus write_record(const ConversationDiskIdentity& identity,
                                    const std::filesystem::path& target, ConversationDiskRecord& record,
                                    const MetaWriter& metadata, uint64_t record_bytes, uint64_t payload_bytes,
                                    std::string& error) {
    Header header;
    header.magic = kMagic;
    header.version = ConversationDiskStore::format_version;
    header.header_bytes = kHeaderBytes;
    header.record_bytes = record_bytes;
    header.metadata_bytes = metadata.data.size();
    header.metadata_checksum = conversation_disk_checksum(metadata.data.data(), metadata.data.size());
    header.payload_bytes = payload_bytes;
    header.payload_checksum = conversation_disk_checksum_seed;
    std::memcpy(header.identity, identity.bytes.data(), ConversationDiskIdentity::size);
    header.cvec = record.stages.front().cvec ? 1 : 0;
    const std::filesystem::path temporary = target.string() + ".tmp";
    std::error_code error_code;
    std::filesystem::remove(temporary, error_code);
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) {
        error = "cannot create " + temporary.string();
        return ConversationDiskStatus::failed;
    }
    bool ok = std::fwrite(&header, 1, sizeof(header), file) == sizeof(header) &&
              (metadata.data.empty() ||
               std::fwrite(metadata.data.data(), 1, metadata.data.size(), file) == metadata.data.size());
    uint64_t checksum = conversation_disk_checksum_seed;
    if (ok) {
        auto write_section = [&](Section section) {
            if (!ok) return;
            if (section.bytes != nullptr) {
                if (!section.bytes->empty() &&
                    std::fwrite(section.bytes->data(), 1, section.bytes->size(), file) != section.bytes->size())
                    ok = false;
                checksum = conversation_disk_checksum(section.bytes->data(), section.bytes->size(), checksum);
            } else {
                section.buffer->visit(0, section.buffer->size(), [&](const uint8_t* data, size_t bytes, size_t) {
                    if (ok && std::fwrite(data, 1, bytes, file) != bytes) ok = false;
                    checksum = conversation_disk_checksum(data, bytes, checksum);
                    return ok;
                });
            }
        };
        visit_payload(record, write_section);
    }
    if (ok) {
        header.payload_checksum = checksum;
        ok = std::fseek(file, 0, SEEK_SET) == 0 && std::fwrite(&header, 1, sizeof(header), file) == sizeof(header);
    }
    if (std::fclose(file) != 0) ok = false;
    if (ok) {
        std::filesystem::rename(temporary, target, error_code);
        if (error_code) ok = false;
    }
    if (!ok) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        error = "cannot write " + target.string();
        return ConversationDiskStatus::failed;
    }
    return ConversationDiskStatus::ok;
}

// Loads one record. Validates the metadata before it allocates the payload buffers.
ConversationDiskStatus read_record(const ConversationDiskIdentity& identity, uint64_t budget_bytes,
                                   const std::filesystem::path& target, ConversationDiskRecord& out,
                                   std::string& error) {
    std::FILE* file = std::fopen(target.string().c_str(), "rb");
    if (file == nullptr) {
        if (errno == ENOENT) return ConversationDiskStatus::miss;
        error = "cannot open " + target.string();
        return ConversationDiskStatus::failed;
    }
    auto bad = [&](const char* why) {
        std::fclose(file);
        error = std::string("corrupt conversation record: ") + why;
        return ConversationDiskStatus::corrupt;
    };
    Header header;
    bool ok = std::fread(&header, 1, sizeof(header), file) == sizeof(header);
    std::error_code size_error;
    const uint64_t file_bytes = std::filesystem::file_size(target, size_error);
    if (!ok || header.magic != kMagic || header.version != ConversationDiskStore::format_version ||
        header.header_bytes != kHeaderBytes) return bad("header");
    if (std::memcmp(header.identity, identity.bytes.data(), ConversationDiskIdentity::size) != 0)
        return bad("identity");
    if (header.metadata_bytes > kMaxMetadataBytes) return bad("metadata size");
    if (size_error || file_bytes != header.record_bytes ||
        header.record_bytes != kHeaderBytes + header.metadata_bytes + header.payload_bytes)
        return bad("lengths");
    if (header.payload_bytes > budget_bytes) return bad("payload size");
    std::vector<uint8_t> metadata(static_cast<size_t>(header.metadata_bytes));
    ok = read_exact(file, metadata.data(), header.metadata_bytes);
    if (!ok) return bad("metadata read");
    if (conversation_disk_checksum(metadata.data(), metadata.size()) != header.metadata_checksum)
        return bad("metadata checksum");
    ConversationDiskRecord record;
    try {
        MetaReader reader{metadata.data(), metadata.data() + metadata.size(), true, {}};
        if (!reader.parse(record) || !reader.ok) return bad("metadata layout");
        if (record.stages.front().cvec != (header.cvec != 0)) return bad("cvec");
        if (!valid_lengths(record, reader.lengths, header.payload_bytes)) return bad("section sizes");
        uint64_t checksum = conversation_disk_checksum_seed;
        size_t index = 0;
        auto read_section = [&](Section section) {
            if (!ok) return;
            const uint64_t bytes = reader.lengths[index++];
            if (section.bytes != nullptr) {
                section.bytes->resize(static_cast<size_t>(bytes));
                ok = read_exact(file, section.bytes->data(), bytes);
                checksum = conversation_disk_checksum(section.bytes->data(), static_cast<size_t>(bytes), checksum);
            } else {
                section.buffer->resize(static_cast<size_t>(bytes));
                section.buffer->visit(0, static_cast<size_t>(bytes), [&](uint8_t* data, size_t chunk, size_t) {
                    if (ok) ok = read_exact(file, data, chunk);
                    checksum = conversation_disk_checksum(data, chunk, checksum);
                    return ok;
                });
            }
        };
        visit_payload(record, read_section);
        if (!ok) return bad("payload read");
        if (checksum != header.payload_checksum) return bad("payload checksum");
        std::fclose(file);
        out = std::move(record);
        return ConversationDiskStatus::ok;
    } catch (const std::bad_alloc&) {
        std::fclose(file);
        error = "cannot allocate the conversation record";
        return ConversationDiskStatus::failed;
    }
}

} // namespace

ConversationDiskIdentity ConversationDiskIdentity::from_string(std::string_view text) {
    ConversationDiskIdentity identity;
    uint64_t lanes[4] = {};
    for (uint64_t lane = 0; lane < 4; ++lane) {
        const uint64_t seed = conversation_disk_checksum_seed + lane * 0x9e3779b97f4a7c15ull;
        lanes[lane] = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(text.data()), text.size(), seed);
    }
    std::memcpy(identity.bytes.data(), lanes, sizeof(lanes));
    return identity;
}

std::string conversation_disk_name(const std::vector<int32_t>& ids,
                                   const std::vector<ConversationImageKey>& images, bool cvec) {
    uint64_t hash = conversation_disk_checksum_seed;
    const uint64_t count = ids.size();
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&count), sizeof(count), hash);
    if (!ids.empty())
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(ids.data()),
                                          ids.size() * sizeof(int32_t), hash);
    const uint64_t image_count = images.size();
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&image_count), sizeof(image_count), hash);
    for (const auto& key : images) {
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.start), sizeof(key.start), hash);
        hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&key.hash), sizeof(key.hash), hash);
    }
    hash = conversation_disk_checksum(reinterpret_cast<const uint8_t*>(&cvec), sizeof(cvec), hash);
    return "c" + hex64(hash) + "-" + hex64(count);
}

bool ConversationDiskStore::valid_name(const std::string& name) {
    if (name.empty() || name.size() > kMaxNameBytes || name == "." || name == "..") return false;
    for (const char c : name)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-')) return false;
    return true;
}

std::filesystem::path ConversationDiskStore::path_for(const std::string& name) const {
    return options_.directory / (name + file_suffix);
}

const ConversationDiskStore::Entry* ConversationDiskStore::find(const std::string& name) const {
    for (const auto& entry : entries_)
        if (entry.name == name) return &entry;
    return nullptr;
}

ConversationDiskStore::Entry* ConversationDiskStore::find(const std::string& name) {
    for (auto& entry : entries_)
        if (entry.name == name) return &entry;
    return nullptr;
}

size_t ConversationDiskStore::lru_victim(const std::string& skip) const {
    size_t victim = SIZE_MAX;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (!skip.empty() && entries_[i].name == skip) continue;
        if (entries_[i].stamp < oldest) {
            oldest = entries_[i].stamp;
            victim = i;
        }
    }
    return victim;
}

void ConversationDiskStore::evict(size_t index) {
    std::error_code error;
    std::filesystem::remove(path_for(entries_[index].name), error);
    bytes_ -= entries_[index].bytes;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    ++evictions_;
}

ConversationDiskStatus ConversationDiskStore::make_room(uint64_t incoming, uint64_t replaced,
                                                        const std::string& skip, std::string& error) {
    auto over_budget = [&]() {
        const uint64_t held = bytes_ >= replaced ? bytes_ - replaced : 0;
        return held + incoming > options_.budget_bytes ||
               (options_.max_records != 0 && entries_.size() + (replaced != 0 ? 0 : 1) > options_.max_records);
    };
    while (over_budget()) {
        const size_t victim = lru_victim(skip);
        if (victim == SIZE_MAX) {
            error = "the disk cache cannot hold the record within its budget";
            return ConversationDiskStatus::invalid;
        }
        evict(victim);
    }
    std::error_code space_error;
    const std::filesystem::space_info space = std::filesystem::space(options_.directory, space_error);
    if (!space_error && space.available < incoming + options_.min_free_bytes) {
        uint64_t free_bytes = space.available;
        while (free_bytes < incoming + options_.min_free_bytes) {
            const size_t victim = lru_victim(skip);
            if (victim == SIZE_MAX) {
                error = "the disk cache has no free space for the record";
                return ConversationDiskStatus::failed;
            }
            free_bytes += entries_[victim].bytes;
            evict(victim);
        }
    }
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::index_file(const std::filesystem::path& path,
                                                         const std::string& name, Entry& entry,
                                                         std::string& error) const {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) {
        error = "cannot open " + path.string();
        return ConversationDiskStatus::failed;
    }
    auto bad = [&](const char* why) {
        std::fclose(file);
        error = std::string("invalid conversation record: ") + why;
        return ConversationDiskStatus::corrupt;
    };
    Header header;
    bool ok = std::fread(&header, 1, sizeof(header), file) == sizeof(header);
    std::error_code size_error;
    const uint64_t file_bytes = std::filesystem::file_size(path, size_error);
    if (!ok || header.magic != kMagic || header.version != format_version ||
        header.header_bytes != kHeaderBytes) return bad("header");
    if (std::memcmp(header.identity, options_.identity.bytes.data(), ConversationDiskIdentity::size) != 0)
        return bad("identity");
    if (header.metadata_bytes > kMaxMetadataBytes || size_error || file_bytes != header.record_bytes ||
        header.record_bytes != kHeaderBytes + header.metadata_bytes + header.payload_bytes)
        return bad("lengths");
    std::vector<uint8_t> metadata(static_cast<size_t>(header.metadata_bytes));
    ok = read_exact(file, metadata.data(), header.metadata_bytes);
    if (!ok) return bad("metadata read");
    if (conversation_disk_checksum(metadata.data(), metadata.size()) != header.metadata_checksum)
        return bad("metadata checksum");
    ConversationDiskRecord record;
    try {
        MetaReader reader{metadata.data(), metadata.data() + metadata.size(), true, {}};
        if (!reader.parse(record) || !reader.ok) return bad("metadata layout");
        if (record.stages.front().cvec != (header.cvec != 0)) return bad("cvec");
        if (!valid_lengths(record, reader.lengths, header.payload_bytes)) return bad("section sizes");
        entry.name = name;
        entry.bytes = file_bytes;
        entry.cvec = record.stages.front().cvec;
        points_of(record, entry.points);
    } catch (const std::bad_alloc&) {
        std::fclose(file);
        error = "cannot index " + path.string();
        return ConversationDiskStatus::failed;
    }
    std::fclose(file);
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::open(const ConversationDiskOptions& options, std::string& error) {
    options_ = options;
    entries_.clear();
    bytes_ = 0;
    stamp_ = 0;
    recoveries_ = 0;
    if (options_.budget_bytes == 0) return ConversationDiskStatus::disabled;
    if (options_.directory.empty()) {
        error = "the disk cache has no directory";
        return ConversationDiskStatus::invalid;
    }
    std::error_code error_code;
    std::filesystem::create_directories(options_.directory, error_code);
    if (error_code) {
        error = "cannot create " + options_.directory.string();
        return ConversationDiskStatus::failed;
    }
    const size_t suffix = std::strlen(file_suffix);
    const std::string temporary_suffix = std::string(file_suffix) + ".tmp";
    std::vector<Entry> found;
    std::vector<std::filesystem::file_time_type> written;
    for (std::filesystem::directory_iterator it(options_.directory, error_code), end; !error_code && it != end;
         it.increment(error_code)) {
        std::error_code type_error;
        if (!it->is_regular_file(type_error)) continue;
        const std::string filename = it->path().filename().string();
        if (filename.size() > temporary_suffix.size() &&
            filename.compare(filename.size() - temporary_suffix.size(), temporary_suffix.size(),
                             temporary_suffix) == 0) {
            std::filesystem::remove(it->path(), type_error);
            continue;
        }
        if (filename.size() <= suffix || filename.compare(filename.size() - suffix, suffix, file_suffix) != 0)
            continue;
        const std::string name = filename.substr(0, filename.size() - suffix);
        if (!valid_name(name)) {
            std::filesystem::remove(it->path(), type_error);
            continue;
        }
        Entry entry;
        const ConversationDiskStatus status = index_file(it->path(), name, entry, error);
        if (status == ConversationDiskStatus::ok) {
            found.push_back(std::move(entry));
            written.push_back(it->last_write_time(type_error));
            ++recoveries_;
        } else if (status == ConversationDiskStatus::corrupt) {
            std::filesystem::remove(it->path(), type_error);
            ++corruptions_;
        }
        // A read failure keeps the file and skips it, so a transient error loses no data.
    }
    std::vector<size_t> order(found.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (written[a] != written[b]) return written[a] < written[b];
        return found[a].name < found[b].name;
    });
    entries_.reserve(found.size());
    for (const size_t i : order) {
        Entry entry = std::move(found[i]);
        entry.stamp = ++stamp_;
        bytes_ += entry.bytes;
        entries_.push_back(std::move(entry));
    }
    while (bytes_ > options_.budget_bytes ||
           (options_.max_records != 0 && entries_.size() > options_.max_records)) {
        const size_t victim = lru_victim(std::string());
        if (victim == SIZE_MAX) break;
        evict(victim);
    }
    error.clear();
    return ConversationDiskStatus::ok;
}

bool ConversationDiskStore::best(const std::vector<int32_t>& ids,
                                 const std::vector<ConversationImageKey>& images, bool cvec,
                                 ConversationDiskMatch& match) const {
    const Entry* chosen = nullptr;
    int64_t chosen_tokens = 0;
    for (const auto& entry : entries_) {
        if (entry.cvec != cvec) continue;
        for (const auto& point : entry.points) {
            const int64_t tokens = conversation_prefix(point, ids, images);
            if (tokens > chosen_tokens ||
                (tokens == chosen_tokens && tokens > 0 && chosen != nullptr && entry.stamp > chosen->stamp)) {
                chosen_tokens = tokens;
                chosen = &entry;
            }
        }
    }
    if (chosen == nullptr || chosen_tokens <= 0) return false;
    match.name = chosen->name;
    match.tokens = chosen_tokens;
    return true;
}

ConversationDiskStatus ConversationDiskStore::put(ConversationDiskRecord&& record, std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (record.stages.empty()) {
        error = "the record has no stage";
        return ConversationDiskStatus::invalid;
    }
    const bool cvec = record.stages.front().cvec;
    for (const auto& stage : record.stages) {
        if (stage.cvec != cvec) {
            error = "the stages disagree on the cvec flag";
            return ConversationDiskStatus::invalid;
        }
    }
    const ConversationCheckpoint* deepest = nullptr;
    for (const auto& stage : record.stages)
        if (!deepest || stage.live.ids.size() > deepest->ids.size()) deepest = &stage.live;
    if (deepest == nullptr || deepest->ids.empty()) {
        error = "the record has no live token prefix";
        return ConversationDiskStatus::invalid;
    }
    const std::string name = conversation_disk_name(deepest->ids, deepest->imgs, cvec);
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    MetaWriter metadata;
    metadata.u32(static_cast<uint32_t>(record.stages.size()));
    for (const auto& stage : record.stages) metadata.write_stage(stage);
    uint64_t payload_bytes = 0;
    for (const uint64_t bytes : metadata.lengths)
        if (!add_overflow(payload_bytes, bytes)) {
            error = "the record size overflows";
            return ConversationDiskStatus::invalid;
        }
    uint64_t record_bytes = kHeaderBytes;
    if (!add_overflow(record_bytes, metadata.data.size()) || !add_overflow(record_bytes, payload_bytes)) {
        error = "the record size overflows";
        return ConversationDiskStatus::invalid;
    }
    if (record_bytes > options_.budget_bytes) {
        error = "the record exceeds the byte budget";
        return ConversationDiskStatus::invalid;
    }
    const bool had_old = find(name) != nullptr;
    const uint64_t replaced = had_old ? find(name)->bytes : 0;
    const ConversationDiskStatus room = make_room(record_bytes, replaced, name, error);
    if (room != ConversationDiskStatus::ok) return room;
    const ConversationDiskStatus written =
        write_record(options_.identity, path_for(name), record, metadata, record_bytes, payload_bytes, error);
    if (written != ConversationDiskStatus::ok) return written;
    if (had_old) {
        if (Entry* old = find(name)) {
            bytes_ -= old->bytes;
            entries_.erase(entries_.begin() + (old - entries_.data()));
        }
    }
    Entry entry;
    entry.name = name;
    entry.bytes = record_bytes;
    entry.stamp = ++stamp_;
    entry.cvec = cvec;
    points_of(record, entry.points);
    bytes_ += entry.bytes;
    entries_.push_back(std::move(entry));
    return ConversationDiskStatus::ok;
}

ConversationDiskStatus ConversationDiskStore::get(const std::string& name, ConversationDiskRecord& record,
                                                  std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    const Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    ConversationDiskRecord loaded;
    const ConversationDiskStatus status =
        read_record(options_.identity, options_.budget_bytes, path_for(name), loaded, error);
    if (status == ConversationDiskStatus::corrupt) {
        std::error_code error_code;
        std::filesystem::remove(path_for(name), error_code);
        if (Entry* stale = find(name)) {
            bytes_ -= stale->bytes;
            entries_.erase(entries_.begin() + (stale - entries_.data()));
        }
        ++corruptions_;
        return status;
    }
    if (status == ConversationDiskStatus::ok) {
        if (Entry* used = find(name)) used->stamp = ++stamp_;
        record = std::move(loaded);
    }
    return status;
}

ConversationDiskStatus ConversationDiskStore::remove(const std::string& name, std::string& error) {
    if (!enabled()) return ConversationDiskStatus::disabled;
    if (!valid_name(name)) {
        error = "the record name is not safe";
        return ConversationDiskStatus::invalid;
    }
    Entry* entry = find(name);
    if (entry == nullptr) return ConversationDiskStatus::miss;
    std::error_code error_code;
    std::filesystem::remove(path_for(name), error_code);
    if (error_code) {
        error = "cannot remove " + path_for(name).string();
        return ConversationDiskStatus::failed;
    }
    bytes_ -= entry->bytes;
    entries_.erase(entries_.begin() + (entry - entries_.data()));
    return ConversationDiskStatus::ok;
}

} // namespace strata::core
