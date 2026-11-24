#pragma once
// The DFly drafter reads hidden states and sparse selections from five target layers.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4_flash {

struct Geometry;
struct KvCache;
struct DenseResidency;
struct DflyCapture;

enum class DflyResult {
    ok,              ///< k ids in the vocab; the stream drained; writes only in [a, a+w) of the five layers
    safe_fallback,   ///< drained, prefix and counters untouched, nothing pending: run a one-token step instead
    abort_request,   ///< the state cannot be trusted (or a precondition failed): dfly_abort_reset
    restart_worker   ///< the CUDA context cannot be trusted: exit the worker
};
const char* dfly_result_name(DflyResult r);

/// A read-only view of one row of a COMPLETED capture. Borrows the capture's device buffers; becomes stale as
/// soon as the capture is filled again, which `propose` detects through `generation`.
struct DflyContext {
    int source_pos = -1;                  ///< the position the row came from; must be the anchor's minus one
    const float* hidden = nullptr;        ///< device [n_embd], after the target's output_norm
    std::array<const int32_t*, 5> sel{};  ///< device [n_sel] per drafter layer; null when not engaged
    int n_sel = 0;
    bool engaged = false;                 ///< the selection was in force for the source row
    const DflyCapture* owner = nullptr;
    uint64_t generation = 0;
};

/// Row `row` of `cap`, with the selection of each of `layers` found BY BLOCK ID in `cap.sel_blocks` (never by
/// its position in that list). False when the capture is not valid or does not hold one of the layers.
bool dfly_context_from_capture(const DflyCapture& cap, int row, const std::array<int, 5>& layers,
                               DflyContext& out, std::string& err);

class DflyDrafter {
public:
    static constexpr int kLayers = 5;
    static constexpr int kMaxK = 5;   ///< the scratch is sized for the anchor and these candidates

    DflyDrafter();
    ~DflyDrafter();
    DflyDrafter(const DflyDrafter&) = delete;
    DflyDrafter& operator=(const DflyDrafter&) = delete;

    /// Read and check the file (schema v1: keys, 92 tensor names, shapes and types) against the target's
    /// geometry, and upload the weights. No target state is touched.
    bool load(const std::string& path, const Geometry& target, std::string& err);

    struct BindOptions {
        /// Reverse target layers only for the drafter self-test's negative control.
        bool reverse_layers = false;
    };
    /// Borrow E/O from `res` (after a whole pass with logits) and the KV of `cache`; compute E(MASK); allocate
    /// the scratch for `cache.max_tokens`. `res` and `cache` must outlive the binding.
    bool bind(const DenseResidency& res, KvCache& cache, std::string& err);
    bool bind(const DenseResidency& res, KvCache& cache, std::string& err, const BindOptions& o);
    /// Drop the borrowed pointers and the scratch. Call before the residency or the cache goes away.
    void unbind();

    /// Propose `k` (0..kMaxK) tokens after `anchor`, which sits at position `a` == `cache.len`. `draft` and
    /// `confidence` take k entries. `confidence` is read only by `--draft-window confidence` (dfly_policy.hpp),
    /// which decides from it how many of the drafts the verify takes.
    DflyResult propose(int32_t anchor, int a, const DflyContext& ctx, int k, int32_t* draft, float* confidence,
                       std::string& err);

    /// A successful proposal (a, w) that the same-window verify has not consumed yet. `w` is the rows the
    /// proposal wrote, [a, a+w) of the five layers.
    struct Pending {
        bool active = false;
        int a = 0, w = 0;
        uint64_t generation = 0;
    };
    const Pending& pending() const;
    void drop_pending();
    /// Verify a prefix of the pending proposal from the same anchor and cache state.
    bool consume_pending(int a, int verify_w, uint64_t generation, std::string& err);

    bool loaded() const;
    bool bound() const;
    const std::array<int, 5>& target_layers() const;   ///< as bound (reversed under the negative control)
    int max_drafts() const;
    int mask_id() const;
    int vocab() const;
    std::size_t weight_bytes() const;
    std::size_t scratch_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::hy4_flash
