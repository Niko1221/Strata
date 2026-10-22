#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <atomic>
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
    /// `n_embd` / `n_ff` are the model's expert widths (2560 / 640 for Flash-Next, 2048 / 512 for Qwen3.6) and `k`
    /// the experts routed per token (10 vs 8).  The window (MAXT * k), the staging rows and the scratch follow
    /// them.  The defaults are Flash-Next's numbers, so a call that omits them allocates and rejects exactly as
    /// before.
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err, bool auto_size = false,
              int64_t n_embd = 2560, int64_t n_ff = 640, int64_t k = 10,
              const std::vector<strata::kernels::NativeExpertLayout>* layer_layouts = nullptr);
    void close();

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    /// Legacy single-window path: always fills slot 0 (Flash-Next callers unchanged).
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    /// STRATA_Q36_HELPER_RING: launch into a free ring slot. Returns slot >= 0, -1 if the
    /// window is empty (nothing this helper owns), or -2 on error (`err` set).
    int begin_win(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                  int64_t k, const int32_t* kind, const int32_t* primary_res,
                  std::string& err);
    bool owns(int64_t index) const { return owns(0, index); }
    bool owns(int slot, int64_t index) const;
    /// This helper's cache holds (layer, expert): begin() will take its rows unless the plan gave them away.
    bool holds(int64_t layer, int32_t expert) const { return cache_.slot_of(layer, expert) >= 0; }
    /// The DEVICE address of (layer, expert)'s blob on this helper, or null when it is not resident here.  The
    /// prompt path (STRATA_Q36_PREFILL_PEERS, src/qwen35/q36_peer_prefill.cu) reads a helper's resident weights in
    /// place instead of stream-copying them, so it needs the address `holds()` alone does not give.  Read-only and
    /// side-effect free: nothing about begin()/finish()/the cache changes, so every existing caller is unaffected.
    /// PeerExperts has the same accessor (include/strata/core/peer_experts.hpp).
    const uint8_t* slot_ptr(int64_t layer, int32_t expert) const {
        const int32_t slot = cache_.slot_of(layer, expert);
        return slot >= 0 ? cache_.device_slot(slot) : nullptr;
    }
    bool optimized_decode() const { return remote_opt_ != nullptr; }
    /// This tier's CUDA device ordinal (the one open() was given, -1 before open).  The accessor PeerExperts has
    /// (origin/main include/strata/core/peer_experts.hpp: `int device() const { return device_; }`), copied here so a
    /// caller running work on the helper's card asks the tier instead of reverse-engineering the device from the
    /// pointers the tier hands out.  Read-only: no state of this class changes.
    int device() const { return device_; }
    /// Legacy: drain(0) then host-scatter into `out` (rings==1 statement order unchanged).
    bool finish(float* out, std::string& err);
    /// Wait for a slot's helper work (event sync when rings>1, stream sync when rings==1).
    /// No host scatter — device-side merge owns that on the ring path.
    bool drain(int slot, std::string& err);
    /// Drain every in-flight slot. Call at forward end and before any ExpertCache mutation.
    void drain_all();
    cudaEvent_t ev_done(int slot) const;
    cudaStream_t stream() const { return stream_; }
    int rings() const { return rings_; }
    bool zero_copy() const { return zero_copy_; }
    /// Pinned compact result rows for slot (primary maps this for the merge upload).
    float* h_out(int slot) const;
    /// Host entry→compact-row map for slot (length = last window's n_tok*k; -1 = not owned).
    const int32_t* src_row(int slot) const;
    int64_t compact_n(int slot) const;
    int64_t window_n(int slot) const;
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }
    /// STRATA_Q36_HELPER_TIMING=1 only: host time drain() spends AROUND its blocking wait (DeviceScope's runtime
    /// calls, the branches, the clock bookkeeping), cumulative.  ms_wait() is the blocking interval alone, so the
    /// two partition a drain into driver overhead and genuine GPU wait.  Zero (and never read) with the flag off.
    double ms_sync_overhead() const { return ms_sync_overhead_; }
    uint64_t drains_timed() const { return drains_timed_; }

    /// STRATA_PREFILL_HELPER_SOURCE (opt-in): copy up to `n` cached blobs into the host buffers `dst[i]`
    /// on this helper's own stream and wait.  `hit[i]` is set where this helper held (layer, expert); the
    /// bytes are the ones open() filled from the same source, so the prompt path gets identical bytes.
    /// Misses are left to the caller.  False with `err` on a CUDA failure; the hits copied so far are
    /// complete (the stream is drained), so the caller falls back for the rest.  Safe from several threads
    /// (one stream, one lock): the prompt path's stager is threaded.
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err);

private:
    friend class RemoteExpertOpt;
    /// One ring slot's begin/finish staging. rings==1 keeps a single Win; rings>1 is
    /// STRATA_Q36_HELPER_RING (device-side merge, no host scatter on drain).
    struct Win {
        float* h_x_ = nullptr;
        float* h_out_ = nullptr;
        void* h_meta_ = nullptr;
        float* d_x_ = nullptr;
        float* z_x_ = nullptr;
        float* z_out_ = nullptr;
        float* d_out_ = nullptr;
        uint8_t* d_q8_ = nullptr;
        float* d_scales_ = nullptr;
        void* d_scratch_ = nullptr;
        void* d_meta_ = nullptr;
        int32_t* d_start_ = nullptr;
        int32_t* d_dst_ = nullptr;
        int32_t* d_tok_ = nullptr;
        int32_t* d_count_ = nullptr;
        unsigned long long* d_ptr_ = nullptr;
        int32_t* d_src_row_ = nullptr;   ///< helper-side copy; engine uploads host src_row_ to primary
        cudaEvent_t ev_done_ = nullptr; ///< created only when rings_ > 1
        bool busy_ = false;
        int32_t groups_ = 0;
        int64_t n_ = 0;                 ///< last window's n_tok * k
        std::vector<uint8_t> owned_;
        std::vector<int32_t> group_of_, group_id_;
        std::vector<int32_t> start_, dst_, tok_, original_row_;
        std::vector<int32_t> src_row_;  ///< entry -> compact row (-1 if not owned)
        std::vector<unsigned long long> ptr_;
    };

    /// Fill `slot` (draining it first if busy). Shared by begin (slot 0) and begin_win.
    int begin_slot(int slot, int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                   int64_t k, const int32_t* kind, const int32_t* primary_res, std::string& err);
    bool alloc_win(Win& w, size_t meta_bytes, size_t scratch, std::string& err);
    void free_win(Win& w);

    RemoteExpertOpt* remote_opt_ = nullptr;
    int device_ = -1;
    int rings_ = 1;            ///< 1 = legacy (RING unset/0); >=2 = HELPER_RING depth
    int rr_ = 0;               ///< ping-pong index for begin_win
    int64_t n_expert_ = 0;
    int64_t n_embd_ = 2560;   ///< the model's row width, set by open(); Flash-Next's default
    int64_t n_ff_ = 640;      ///< the model's expert intermediate width, for the scratch sizing
    int64_t k_ = 10;          ///< the experts routed per token; begin() checks a window against it
    int64_t cap_ = 80;        ///< the window's entry count, MAXT * k_ (Flash-Next: 8 * 10)
    std::vector<strata::kernels::NativeExpertLayout> layer_layouts_;   ///< per-layer native layout override; empty = use the global expert_layout() table
    bool step_known_ = false;   ///< layer_layouts_ gives a step for EVERY layer (see open: known = verbatim, else 256)
    int64_t computed_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    double ms_sync_overhead_ = 0;
    uint64_t drains_timed_ = 0;
    ExpertCache cache_;
    cudaStream_t stream_ = nullptr;
    cudaStream_t read_stream_ = nullptr;   ///< STRATA_PREFILL_HELPER_SOURCE: cache -> host copies
    std::mutex read_mu_;                   ///< one cache reader at a time (the stager is threaded)
    bool zero_copy_ = false;
    float* d_xr_ = nullptr;                 // APR: the Hadamard-rotated expert input (CAP x H), shared by the rings
    const float* d_sx_ = nullptr;           // APR: device sign vectors (x: H, h: FF); nullptr = identity
    const float* d_sh_ = nullptr;           // process-cached (hadamard_device_signs), so never freed here
    std::vector<Win> win_;
    std::vector<uint8_t> layers_present_;
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
