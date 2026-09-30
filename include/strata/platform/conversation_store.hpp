#pragma once

#include "strata/platform/conversation_file.hpp"

#include <memory>

namespace strata::platform {
// Single-writer, synchronous disk tier. Only put() writes a snapshot; the caller
// invokes it on RAM eviction. No GPU state or automatic per-turn persistence.
class ConversationStore {
public:
    struct Candidate {
        std::filesystem::path path;
        ConversationFileMatch match;
    };
    ConversationStore();
    ~ConversationStore();
    ConversationStore(const ConversationStore&) = delete;
    ConversationStore& operator=(const ConversationStore&) = delete;

    // Owns root/strata-conversations-v1, with an exclusive process-lifetime lock.
    // Quotas include all identities and any in-progress snapshot in that directory.
    // Failed open leaves the store closed. A zero quota disables it without I/O.
    bool open(const std::filesystem::path& root, const ConversationIdentity& identity,
              uint64_t bytes, size_t slots, std::string& error, ConversationIoProgress progress = nullptr);
    bool is_open() const;
    void close();
    // Protect a selected disk hit while RAM admission evicts other images. If
    // retaining it prevents quota admission, decline the spill instead.
    bool put(const core::SavedConversation& image, std::string& error, const Candidate* protected_entry = nullptr);
    bool best(const std::vector<int64_t>& prompt, const std::vector<core::ConversationImageKey>& images,
              bool cvec, uint64_t staging_limit, const std::vector<std::filesystem::path>& excluded,
              Candidate& candidate, std::string& error) const;
    bool read(const Candidate& candidate, uint64_t staging_limit, std::optional<uint64_t> available,
              uint64_t floor, core::SavedConversation& image, std::string& error) const;
    // Call only after shared-core validation and successful promotion.
    bool touch(const Candidate& candidate, std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::platform
