// include/strata/core/conversation_disk.hpp - the L3 disk tier for parked conversations.
//
// The RAM cache (conversation_cache.hpp) holds whole conversations in host memory. It forgets
// them when the process exits. This store keeps the same images in files. A restarted server
// can then resume a conversation without a full prompt read.
//
// The store is CPU only. It makes no CUDA call. The caller captures and restores the images
// with conversation_snapshot.hpp.
//
// One record holds one conversation. The record holds one SavedConversation per --layer-split
// stage. A two-card run therefore parks and resumes as one unit. All stages must hold the same
// token and image chain. Each stage keeps its own layer carve in layer_lo and layer_hi.
//
// The record name is derived from the deepest token prefix. The same conversation always maps
// to the same file name, so a re-park replaces the old file.
//
// Ownership and lifetime:
//  * The store owns its index and its files.
//  * put() consumes the record. The store writes it and the caller must not reuse it.
//  * get() fills a record the caller owns. The record owns its buffers.
//  * A ConversationDiskMatch holds a record name. That name stays valid until the next put()
//    or remove(). The caller must call get() before then.
//  * One thread uses one store. The store is synchronous and makes no lock.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace strata::core {

// The FNV-1a 64 checksum seed and mix. The store uses one checksum for the metadata block and
// one for the payload block.
inline constexpr uint64_t conversation_disk_checksum_seed = 14695981039346656037ull;

inline uint64_t conversation_disk_checksum(const uint8_t* data, size_t size,
                                           uint64_t seed = conversation_disk_checksum_seed) {
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// A caller-supplied compatibility key. Two runs share a store only if the keys match.
// The key must cover the model weights and every runtime flag that changes the payload layout
// (quantization, rope scaling, layer split, cvec mode). The store compares the 32 bytes exactly.
struct ConversationDiskIdentity {
    static constexpr size_t size = 32;
    std::array<uint8_t, size> bytes{};
    bool operator==(const ConversationDiskIdentity&) const = default;
    // A stable label from a text key. This is a compatibility label, not a secure hash.
    static ConversationDiskIdentity from_string(std::string_view text);
};

// One conversation as a set of layer-split stage images.
struct ConversationDiskRecord {
    // One entry per stage, ordered by layer_lo. A single-GPU run uses one entry.
    std::vector<SavedConversation> stages;
    // The RAM size of the images. The caller uses it for admission.
    size_t bytes() const {
        size_t n = stages.capacity() * sizeof(SavedConversation);
        for (const auto& stage : stages) n += stage.bytes();
        return n;
    }
};

// A lookup result. name is the record to load. tokens is the matched prefix length.
struct ConversationDiskMatch {
    std::string name;
    int64_t tokens = 0;
};

enum class ConversationDiskStatus {
    ok,        // the call succeeded
    miss,      // no record has that name or prefix
    corrupt,   // the file failed validation and the store removed it
    invalid,   // the caller's record, name, or options are not usable
    disabled,  // the store has no byte budget
    failed,    // an I/O or allocation error; the caller may retry
};

struct ConversationDiskOptions {
    std::filesystem::path directory;   // the store root; open() creates it
    ConversationDiskIdentity identity; // the current run's compatibility key
    uint64_t budget_bytes = 0;         // the file budget; 0 disables the store
    size_t max_records = 0;            // the record cap; 0 means no cap
    uint64_t min_free_bytes = 0;       // the free-space floor the store keeps
};

// A safe record name from a token prefix. The store uses the same name in put().
std::string conversation_disk_name(const std::vector<int32_t>& ids,
                                   const std::vector<ConversationImageKey>& images, bool cvec);

class ConversationDiskStore {
public:
    static constexpr uint32_t format_version = 1;
    static constexpr const char* file_suffix = ".conversation";

    ConversationDiskStore() = default;
    ConversationDiskStore(const ConversationDiskStore&) = delete;
    ConversationDiskStore& operator=(const ConversationDiskStore&) = delete;
    ConversationDiskStore(ConversationDiskStore&&) = default;
    ConversationDiskStore& operator=(ConversationDiskStore&&) = default;

    // Opens the directory and rebuilds the index. Removes every file that fails validation.
    // Returns disabled when budget_bytes is zero.
    ConversationDiskStatus open(const ConversationDiskOptions& options, std::string& error);

    bool enabled() const { return options_.budget_bytes != 0; }

    // Finds the longest stored token prefix. It considers only records with the same cvec flag.
    // A tie prefers the most recently used record. This call does no I/O and no allocation.
    bool best(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
              bool cvec, ConversationDiskMatch& match) const;

    // Publishes the record. Replaces a record with the same name. Consumes the record in all cases.
    ConversationDiskStatus put(ConversationDiskRecord&& record, std::string& error);

    // Loads a record into RAM. Validates the metadata before it allocates. Verifies the payload
    // checksum while it streams. Removes the file on corruption.
    ConversationDiskStatus get(const std::string& name, ConversationDiskRecord& record, std::string& error);

    ConversationDiskStatus remove(const std::string& name, std::string& error);

    size_t records() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    uint64_t evictions() const { return evictions_; }
    uint64_t corruptions() const { return corruptions_; }
    uint64_t recoveries() const { return recoveries_; }
    const std::filesystem::path& directory() const { return options_.directory; }

private:
    // The in-RAM index of one record. The points hold ids and imgs only, never payload bytes.
    struct Entry {
        std::string name;
        uint64_t bytes = 0;
        uint64_t stamp = 0;
        bool cvec = true;
        std::vector<ConversationCheckpoint> points;
    };

    std::filesystem::path path_for(const std::string& name) const;
    static bool valid_name(const std::string& name);
    const Entry* find(const std::string& name) const;
    Entry* find(const std::string& name);
    size_t lru_victim(const std::string& skip) const;
    void evict(size_t index);
    ConversationDiskStatus make_room(uint64_t incoming, uint64_t replaced, const std::string& skip,
                                     std::string& error);
    ConversationDiskStatus index_file(const std::filesystem::path& path, const std::string& name,
                                      Entry& entry, std::string& error) const;

    ConversationDiskOptions options_{};
    std::vector<Entry> entries_;
    uint64_t bytes_ = 0;
    uint64_t stamp_ = 0;
    uint64_t evictions_ = 0;
    uint64_t corruptions_ = 0;
    uint64_t recoveries_ = 0;
};

} // namespace strata::core
