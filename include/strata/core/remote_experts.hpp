#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// One helper tier's worth of pairs, chosen in rank order.  Skips any pair the `primary` (CUDA0) cache holds,
/// any pair `claimed` has already given to a stage's cache or an earlier tier, and any pair this walk already
/// picked - so tiers are disjoint by construction.  Stops at `slots` entries (`<= 0`: no limit) or when the
/// next pair would push the aligned total past `byte_budget` (`<= 0`: no limit).
///
/// A budget SKIPS a pair that does not fit instead of ending the walk: blob sizes vary by layer, and one hot
/// oversized blob must not hide the colder pairs behind it that would still fit.  CUDA-free, so the sizing
/// rule can be tested without a GPU.
void select_remote_pairs(const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                         const std::vector<uint8_t>& claimed, int64_t layers, int64_t experts, int64_t slots,
                         int64_t byte_budget, std::vector<std::pair<int32_t, int32_t>>& selected,
                         std::vector<int64_t>& sizes, uint64_t& needed);

/// A static, profile-filled expert tier on another CUDA device. CUDA0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
///
/// It used to be required to run on a GPU no stage ran on.  It does not any more: a layer split gives each
/// card's LEFTOVER VRAM to a tier of its own, CUDA0 included, because a stage's cache can only hold its own
/// layers' pairs and would otherwise leave that room with nothing to do.
class RemoteExperts {
public:
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    /// A tier of exactly `slots` experts (`--expert-cache-deviceN`).
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err);
    /// A tier that fills `byte_budget` of leftover VRAM, however many experts that is (the automatic spill
    /// tier a layer split opens on every card it already uses).
    bool open_budgeted(int device, int64_t byte_budget, int64_t layers, int64_t experts,
                       const std::vector<std::pair<int32_t, int32_t>>& ranked,
                       const ExpertCache& primary, ExpertSource& source,
                       std::vector<uint8_t>& claimed, std::string& err);
    void close();

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    /// True when activations and results cross as mapped host pages instead of two copies per layer.  MEASURED
    /// at `open` from `cudaHostGetDevicePointer`, never assumed: it needs `cudaHostAllocMapped` memory and a
    /// device that can address it, which every UVA device can whether or not `preflight` got there first to set
    /// `cudaDeviceMapHost`.  So a tier on a card that already runs a stage can - and on this rig does - still
    /// report true; a false is a real fallback to the copy path, not a guess about one.
    bool zero_copy() const { return zero_copy_; }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }

private:
    /// `slots > 0` sizes by expert count, `byte_budget > 0` by bytes; exactly one is set.
    bool open_tier(int device, int64_t slots, int64_t byte_budget, int64_t layers, int64_t experts,
                   const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                   ExpertSource& source, std::vector<uint8_t>& claimed, std::string& err);

    int device_ = -1;
    int64_t n_expert_ = 0;
    int32_t groups_ = 0;
    int64_t computed_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    ExpertCache cache_;
    cudaStream_t stream_ = nullptr;
    float* h_x_ = nullptr;
    float* h_out_ = nullptr;
    void* h_meta_ = nullptr;
    float* d_x_ = nullptr;
    float* z_x_ = nullptr;     ///< h_x_ as the helper GPU sees it (zero-copy: no input copy per layer)
    float* z_out_ = nullptr;   ///< h_out_ as the helper GPU sees it (zero-copy: no result copy)
    bool zero_copy_ = false;
    float* d_out_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    void* d_meta_ = nullptr;  ///< one contiguous upload of grouped indices, instead of five small copies
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, original_row_;
    std::vector<unsigned long long> ptr_;
};

} // namespace strata::core
