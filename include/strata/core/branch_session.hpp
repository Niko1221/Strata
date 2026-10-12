#pragma once
#include "strata/core/branch_pages.hpp"
#include "strata/core/session.hpp"
#include <memory>
#include <string>
#include <vector>

namespace strata::core {
struct BranchSessionSnapshot;
struct SavedConversation;
using BranchSessionImage = std::shared_ptr<const BranchSessionSnapshot>;
struct BranchSessionPool {
    void* address = nullptr;
    uint64_t bytes = 0, bytes_per_cell = 0;
    bool pooled = false;
};
struct BranchSessionCounters {
    uint64_t binding_copy_bytes = 0, saved_running_bytes = 0, restored_running_bytes = 0;
    uint64_t captures = 0, restores = 0;
    uint64_t written_file_bytes = 0, loaded_file_bytes = 0;
};
struct BranchSnapshotCounters {
    uint64_t allocations = 0, releases = 0, live_images = 0;
    // Actual vector capacities of published running/history/frontier payloads.
    // Excludes allocator rounding, control blocks, layout strings and page tables.
    uint64_t payload_capacity_bytes = 0, peak_payload_capacity_bytes = 0;
};
struct BranchAuditRegion {
    std::string kind;
    uint64_t offset = 0, bytes = 0;
    int64_t pool = -1, qsa_ordinal = -1;
    uint64_t pool_offset = 0, bytes_per_cell = 0;
    // Contiguous, completely consumed QSA blocks only. Excludes a partial
    // logical block, the indexer's moving spare row and unused capacity.
    uint64_t filled_prefix_bytes = 0;
    BranchPageInfo page;
};
struct BranchSessionAudit {
    uint64_t bytes = 0, consumed = 0, qsa_page_cells = 0, index_block_cells = 0;
    std::vector<BranchAuditRegion> regions;
};

// One synchronized, private inference lease. Bind once to an empty, initialized
// session BEFORE any graph capture. The original arena remains allocated and is
// reported as binding_copy_bytes, not hidden from resource accounting. A logical
// fork is a copy of BranchSessionImage; it does not call restore or copy pages.
// This is an optional derived checkpoint, never the authoritative history store.
class BranchSession {
public:
    BranchSession(SessionState& session, const ModelGeometry& geometry, std::string contract);
    ~BranchSession();
    BranchSession(const BranchSession&) = delete;
    BranchSession& operator=(const BranchSession&) = delete;
    bool bind(std::string& error);
    bool reset(std::string& error);
    bool begin(int64_t position, int64_t count, std::string& error);
    // Call only after the real forward AND recurrent commit have succeeded.
    bool finish(const std::vector<int32_t>& tokens, const std::vector<float>& logits, std::string& error);
    BranchSessionImage freeze(std::string& error);
    bool restore(const BranchSessionImage& image, std::string& error);
    // Optional cold state uses the existing session file. That file excludes
    // logits: the caller must SHA-256 bind its frontier and exact history in the
    // outer checkpoint manifest before calling load. Neither call publishes an
    // authoritative history snapshot or includes decoder/RNG/grammar state.
    bool write(const std::string& path, size_t& bytes, std::string& error);
    bool load(const std::string& path, const std::vector<int32_t>& expected_ids,
              const std::vector<float>& frontier, std::string& error);
    bool host_capture_bytes(size_t& bytes, std::string& error) const;
    bool capture_host(SavedConversation& image, std::string& error);
    bool restore_host(const SavedConversation& image, const std::vector<int32_t>& expected_ids,
                      const std::vector<float>& frontier, std::string& error);
    const std::vector<int32_t>& ids() const { return ids_; }
    const std::vector<float>& logits() const { return logits_; }
    std::vector<BranchSessionPool> pools() const;
    std::vector<BranchPageInfo> ownership() const;
    // Diagnostic byte export only; cannot be imported as a checkpoint. Reads
    // the retained image itself without restoring it or changing the frontier.
    bool audit(const BranchSessionImage& image, const std::string& path,
               BranchSessionAudit& report, std::string& error) const;
    const BranchSessionCounters& counters() const { return counters_; }
    static BranchSnapshotCounters snapshot_counters();
private:
    struct Binding;
    enum class State { unbound, ready, writing, failed };
    SessionState& session_;
    ModelGeometry geometry_;
    std::string contract_;
    std::vector<int64_t> layout_;
    std::vector<std::unique_ptr<Binding>> bindings_;
    std::vector<int32_t> ids_;
    std::vector<float> logits_;
    BranchSessionCounters counters_;
    State state_ = State::unbound;
    int64_t pending_ = 0;
};
} // namespace strata::core
