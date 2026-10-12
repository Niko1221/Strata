#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::core {
struct BranchDigest { uint8_t bytes[32]{}; };
inline bool operator==(const BranchDigest& a, const BranchDigest& b) {
    return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}
inline constexpr uint64_t branch_digest_tile_bytes = 4096;
struct BranchDigestCounters {
    uint64_t device_hashed_bytes = 0, digest_copy_bytes = 0, kernel_calls = 0;
    uint64_t workspace_allocations = 0, workspace_releases = 0, workspace_live_bytes = 0;
};
BranchDigestCounters branch_digest_counters();

// Private scratch allocated before entering a memory-pressure operation. Each
// 4096-byte tile is independently SHA-256 hashed on device; only the ordered
// digests cross to the host. This is integrity metadata, never model state.
class BranchDigestWorkspace {
public:
    BranchDigestWorkspace() = default;
    ~BranchDigestWorkspace();
    BranchDigestWorkspace(const BranchDigestWorkspace&) = delete;
    BranchDigestWorkspace& operator=(const BranchDigestWorkspace&) = delete;
    bool reserve(uint64_t maximum_input_bytes);
    bool digest(const uint8_t* device, uint64_t bytes, std::vector<BranchDigest>& result);
private:
    BranchDigest* device_ = nullptr;
    uint64_t capacity_ = 0;
};
} // namespace strata::core
