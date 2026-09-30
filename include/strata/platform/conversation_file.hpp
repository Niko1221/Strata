#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace strata::platform {
using ConversationIdentity = std::array<uint8_t, 32>;
// Optional heartbeat for callers with request watchdogs. Reports completed I/O
// progress, never a timer that could hide an operation stuck in the kernel.
using ConversationIoProgress = void (*)() noexcept;
struct ConversationAsset {
    std::string role;
    std::filesystem::path path;
};

// Hash complete asset contents and normalized runtime settings, not paths or samples.
// Assets must remain immutable from model loading through snapshot use.
bool conversation_identity(const std::vector<ConversationAsset>& assets, const std::string& settings,
                           ConversationIdentity& identity, std::string& error);

// Little-endian envelope of the shared image. No CUDA, model-state sizing or restore
// logic belongs here. Call conversation_snapshot_validate before applying a read image.
// This format deliberately refuses the experimental #52 v3/native-struct files.
bool conversation_file_write(std::ostream& stream, const core::SavedConversation& image,
                             const ConversationIdentity& identity, std::string& error, ConversationIoProgress progress = nullptr);
bool conversation_file_size(const core::SavedConversation& image, uint64_t& bytes, std::string& error);
// Reads directly into one staged image; no second whole-file buffer. Budget covers
// vector payload/capacity and a fixed codec allowance, not allocator/RSS overhead.
// Unknown RAM telemetry declines admission; failure leaves output unchanged.
bool conversation_file_read(std::istream& stream, const ConversationIdentity& identity,
                            uint64_t staging_limit, std::optional<uint64_t> available, uint64_t floor,
                            core::SavedConversation& output, std::string& error, ConversationIoProgress progress = nullptr);

struct ConversationFileMatch {
    int64_t tokens = 0;
    bool live = false;
    uint64_t staging_bytes = 0;
};
// Seek over state payloads to select candidates without allocating images. A
// candidate is untrusted until file_read and the shared core validate it.
// Requires a seekable stream positioned at its beginning.
bool conversation_file_match(std::istream& stream, const ConversationIdentity& identity,
                             uint64_t staging_limit, const std::vector<int64_t>& prompt,
                             const std::vector<core::ConversationImageKey>& images, bool cvec,
                             ConversationFileMatch& match, std::string& error, ConversationIoProgress progress = nullptr);

inline constexpr uint64_t kConversationFileWorkspace = 65536;
} // namespace strata::platform
