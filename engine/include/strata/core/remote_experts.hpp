#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <functional>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExpertOpt;

/// A static, profile-filled expert tier on another CUDA device. CUDA0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
class RemoteExperts {
public:
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err, bool auto_size = false);
    void close();

    /// strata-mi50 (2026-10-23): the tier on a GPU that only another HIP runtime sees, through a plugin DLL
    /// (include/strata/core/helper_plugin_abi.h, --expert-cache-plugin).  Set before preflight(); every RemoteExperts then
    /// forwards to the plugin, which runs this same class in its own runtime.  `plugin_device`: the device in that runtime.
    static void set_plugin(const std::string& dll, const std::string& pack_dir, int plugin_device = 0);
    static bool plugin_configured();
    /// open() with the primary cache and the expert source as callbacks (the plugin has neither object).  `standalone`:
    /// this runtime sees no primary device to keep clear of.
    bool open_with(int device, int slots, int64_t layers, int64_t experts,
                   const std::vector<std::pair<int32_t, int32_t>>& ranked,
                   const std::function<bool(int32_t, int32_t)>& on_primary,
                   const std::function<const uint8_t*(int64_t, int64_t)>& blob,
                   std::vector<uint8_t>& claimed, std::string& err, bool auto_size, bool standalone);

    /// strata-mi50 (2026-10-26): RemoteExpertOpt::adapt's swap rule for one helper, as a call that also crosses the plugin ABI
    /// (version 3).  `usage`, `primary` and `resident` have n = layers * experts entries; `resident` is updated for the swaps
    /// made.  Each new blob is copied and waited for before the next `blob` call.
    bool adapt_swaps(const float* usage, const int32_t* primary, uint8_t* resident, int64_t n, int max_swaps,
                     const std::function<const uint8_t*(int64_t, int64_t)>& blob, int& swapped, std::string& err);

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    bool owns(int64_t index) const { return plugin_ ? plugin_owns(index) : owned_[(size_t) index] != 0; }
    /// This helper's cache holds (layer, expert): begin() will take its rows unless the plan gave them away.
    bool holds(int64_t layer, int32_t expert) const { return plugin_ ? plugin_holds(layer, expert) : cache_.slot_of(layer, expert) >= 0; }
    bool optimized_decode() const { return remote_opt_ != nullptr; }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return plugin_ ? plugin_i(0) : cache_.resident(); }
    int64_t computed() const { return plugin_ ? plugin_i(1) : computed_; }
    int64_t launched_layers() const { return plugin_ ? plugin_i(2) : launched_layers_; }
    double gib() const { return plugin_ ? plugin_d(0) : cache_.gib(); }
    uint64_t returned_bytes() const { return plugin_ ? (uint64_t) plugin_i(3) : returned_bytes_; }
    uint64_t full_row_bytes() const { return plugin_ ? (uint64_t) plugin_i(4) : full_row_bytes_; }
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return plugin_ ? plugin_d(1) : ms_begin_; }
    double ms_wait() const { return plugin_ ? plugin_d(2) : ms_wait_; }

    /// STRATA_PREFILL_HELPER_SOURCE (opt-in): copy up to `n` cached blobs into the host buffers `dst[i]`
    /// on this helper's own stream and wait.  `hit[i]` is set where this helper held (layer, expert); the
    /// bytes are the ones open() filled from the same source, so the prompt path gets identical bytes.
    /// Misses are left to the caller.  False with `err` on a CUDA failure; the hits copied so far are
    /// complete (the stream is drained), so the caller falls back for the rest.  Safe from several threads
    /// (one stream, one lock): the prompt path's stager is threaded.
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err);

    /// 2026-10-27 STRATA_PREFILL_HELPER (ABI 5): the helper computes a share of the prompt's experts.  prefill_supported():
    /// the plugin has the entry points.  prefill_reserve / begin / finish: see helper_plugin_abi.h.
    bool prefill_supported() const;
    bool prefill_reserve(int64_t t_max, std::vector<uint8_t>& layer_ok, uint16_t** x16, uint16_t** sum16, std::string& err);
    bool prefill_begin(int64_t layer, int64_t t, int64_t k, const uint16_t* x16, const int32_t* ids, const float* w,
                       const uint8_t* take, std::string& err);
    bool prefill_finish(uint16_t* sum16, double* ms_wait, std::string& err);
    /// Plugin side: the device address of (layer, expert)'s blob in this tier's cache, nullptr when not held (or in a plugin forwarder).
    const uint8_t* slot_ptr(int64_t layer, int32_t expert) const {
        if (plugin_ != nullptr || device_ < 0) return nullptr;
        const int32_t s = cache_.slot_of(layer, expert);
        return s < 0 ? nullptr : cache_.device_slot(s);
    }

private:
    friend class RemoteExpertOpt;
    void* plugin_ = nullptr;   ///< the plugin's handle when this tier runs in the helper plugin (set_plugin)
    // 2026-10-26: --remote-expert-opt through the plugin: the window's reduced rows and owned flags (finish_opt), and whether
    // begin() asked the plugin to reduce
    std::vector<float> plugin_sum_;
    std::vector<uint8_t> plugin_owned_;
    bool plugin_reduced_ = false;
    bool plugin_owns(int64_t index) const;
    bool plugin_holds(int64_t layer, int32_t expert) const;
    int64_t plugin_i(int k) const;
    double plugin_d(int k) const;
    RemoteExpertOpt* remote_opt_ = nullptr;
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
    cudaStream_t read_stream_ = nullptr;   ///< STRATA_PREFILL_HELPER_SOURCE: cache -> host copies
    std::mutex read_mu_;                   ///< one cache reader at a time (the stager is threaded)
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
    float* d_xr_ = nullptr;                 // APR: the Hadamard-rotated expert input (CAP x H)
    const float* d_sx_ = nullptr;           // APR: device sign vectors (x: H, h: FF); nullptr = identity
    const float* d_sh_ = nullptr;
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

/// STRATA_PREFILL_HELPER_SOURCE (opt-in): the prompt path's view of the expert tiers, where a blob a helper
/// GPU already holds in its VRAM cache is copied from that cache instead of being read from the files again.
/// Everything else is forwarded to `fallback` unchanged, so its batched unbuffered reads, its pinned RAM copy
/// and the prompt path's own decisions keep their meaning.  Nothing is cached here: every call asks the
/// helpers, so an entry an adaptive tier moves between requests is a miss (a file read), never a wrong blob.
///
/// The serve loop builds this only for a serial request path (no batch slots, no layer split), while the
/// helpers are idle: during a prompt read the helper GPUs compute nothing, so their caches can be read
/// without ordering the copies against their kernel streams.
class RemotePrefillSource final : public ExpertSource {
public:
    RemotePrefillSource(ExpertSource& fallback, const std::vector<RemoteExperts*>& helpers)
        : fallback_(fallback), helpers_(helpers) {}

    /// Whether any helper's VRAM cache holds (layer, expert) right now.
    bool from_helper(int64_t layer, int64_t expert) const {
        for (const RemoteExperts* h : helpers_) if (h->holds(layer, (int32_t) expert)) return true;
        return false;
    }
    /// Blobs / bytes this source has taken from the helpers so far (the per-request evidence line).
    uint64_t helper_blobs() const { return helper_blobs_.load(std::memory_order_relaxed); }
    uint64_t helper_bytes() const { return helper_bytes_.load(std::memory_order_relaxed); }

    const uint8_t* blob(int64_t layer, int64_t expert) override { return fallback_.blob(layer, expert); }
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override {
        return fallback_.blob_stable(layer, expert);
    }
    int blob_parts(int64_t layer, int64_t expert, BlobPart* out) override {
        return from_helper(layer, expert) ? 0 : fallback_.blob_parts(layer, expert, out);
    }
    int64_t reads() const override { return fallback_.reads(); }
    bool pinned(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? false : fallback_.pinned(layer, expert);
    }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override {
        fallback_.begin_layer(layer, ids, k);
    }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? nullptr : fallback_.device_alias(layer, expert);
    }
    void prefetch(int64_t layer, int64_t expert) override { fallback_.prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override {
        return from_helper(layer, expert) ? 0 : fallback_.release(layer, expert);
    }
    bool pcie_layer(int64_t layer) const override { return fallback_.pcie_layer(layer); }
    /// True where a helper holds the blob: the prompt path's stager must then take the copy_blob(s) route,
    /// which is where the cache-to-host copy happens.  Everywhere else the fallback answers as it always did.
    bool transient(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? true : fallback_.transient(layer, expert);
    }
    double cached_share(int64_t samples) const override { return fallback_.cached_share(samples); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.prefetch(layer, experts, n);
    }
    void warm(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.warm(layer, experts, n);
    }
    bool warms() const override { return fallback_.warms(); }
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override {
        return fallback_.advise_pairs(pairs, n);
    }

private:
    void warn(const std::string& err);
    ExpertSource& fallback_;
    std::vector<RemoteExperts*> helpers_;
    std::atomic<uint64_t> helper_blobs_{0}, helper_bytes_{0};
    std::atomic<bool> warned_{false};
};
} // namespace strata::core
