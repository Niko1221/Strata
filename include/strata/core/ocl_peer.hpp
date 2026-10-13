// include/strata/core/ocl_peer.hpp - the OpenCL peer expert tier.
//
// PeerExperts (peer_experts.hpp) needs CUDA on the second card.  The MI50 in this box is a gfx906
// Vega 20 that no CUDA runtime will drive on Windows, but AMD's OpenCL 2.0 runtime does - so this is
// the same tier with the kernels written in OpenCL C, transcribed from src/kernels/cuda/iq_kernels.cu
// and verified bit-exact against it format by format.
//
// The contract is PeerExperts': for each layer the pool thread decides which routed experts the
// primary computes, and every remaining expert is a "CPU row" the graph reads from mapped host memory
// after the pool raises its flag.  This tier takes the rows whose experts it holds; the pool launches
// them right after publishing the primary's plan, the CPU computes the rest at the same time, and the
// rows are written into the same host block before the pool returns.
//
// `p2p()` is false - an OpenCL buffer has no CUDA address - so the prompt path's peer P2P streaming
// (Prefill::set_peer) declines and stays on the primary.  The decode path, which is where the tier
// earns its keep, is unaffected.
//
// Deliberately free of every other engine header: the tier needs a per-layer format table, a way to
// read a blob, and the set of pairs the primary already holds, and that is all.  ocl_peer_probe.cpp
// drives it from Python-generated blobs, peer_experts.cpp drives it from the pack.
#pragma once

#include "strata/core/peer_tier.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// One layer's expert tensor shapes, in the engine's own terms.
struct OclLayerInfo {
    int gu_type = -1;
    int d_type = -1;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    /// The engine's blob_bytes(layer).  The tier computes the same number from the format table and
    /// refuses to open if they disagree - that is the check that its byte arithmetic matches the pack.
    uint64_t blob_bytes = 0;
};

class OclPeer : public PeerTier {
public:
    OclPeer();
    ~OclPeer();
    OclPeer(const OclPeer&) = delete;
    OclPeer& operator=(const OclPeer&) = delete;

    /// The blob of (layer, expert) in host memory (its `blob_bytes` bytes), or nullptr.
    using BlobFn = const uint8_t* (*)(void* ctx, int64_t layer, int64_t expert);

    /// Whether an OpenCL device is there at all, and what it is (`index` = the ordinal in the AMD
    /// platform).  Used before anything is allocated, so a missing runtime is a message and not a crash.
    static bool probe(int index, std::string& name, double& gib, double& max_alloc_gib);
    /// How many devices the platform reports (0 when there is no platform).
    static int count();
    /// Whether the kernels below cover this layer's format pair.
    static bool layer_supported(const OclLayerInfo& L, std::string& why);

    /// Opens the tier on `index` and fills it with the first pairs of `ranked` that `primary_slot` does
    /// not hold (that array is n_layers*n_expert, >= 0 where the primary has the expert), as many as fit
    /// in the device's largest possible allocation minus `reserve_mib` (`max_slots` > 0 caps the count).
    bool open(int index, const std::vector<OclLayerInfo>& layers, int64_t n_expert, BlobFn blob, void* ctx,
              const int8_t* primary_slot, const std::vector<std::pair<int32_t, int32_t>>& ranked,
              int reserve_mib, int64_t max_slots, std::string& err);
    void close();

    bool valid() const override { return open_; }
    /// Whether the tier holds (layer, expert) right now.
    bool has(int64_t layer, int64_t expert) const override;
    /// No CUDA device is involved at all, so the CUDA ordinal is -1 and the prompt path stays on the primary.
    int device() const override { return -1; }
    /// No CUDA-visible address exists for an OpenCL buffer.
    const uint8_t* slot_ptr(int64_t, int64_t) override { return nullptr; }
    bool p2p() const override { return false; }

    /// Starts the tier's share of one layer: the entries i with kind[i] == 2 (x: n_tok rows of n_embd
    /// floats, the caller's pinned block).  `finish` copies the rows into `out`.
    bool launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                const int32_t* kind, std::string& err, float* out = nullptr) override;
    /// Waits for the launched share and writes its rows into `out` (row i = entry i, n_embd floats).
    bool finish(float* out, std::string& err) override;
    bool launched_direct() const override { return false; }

    /// Adaptive tier: the same policy as PeerExperts::adapt - the most-routed experts neither card holds
    /// into this tier's least-used slots.
    bool adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) override;
    void apply_pending(bool wait) override;

    int64_t resident() const override { return resident_; }
    double gib() const override { return gib_; }
    int64_t entries() const override { return entries_; }
    int64_t experts() const { return experts_; }
    int64_t swaps() const { return swaps_; }
    const std::string& device_name() const { return name_; }
    std::string describe() const override { return "OpenCL " + std::to_string(index_) + " (" + name_ + ")"; }
    double ms_wait = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool open_ = false;
    int index_ = -1;
    int64_t n_expert_ = 0;
    int64_t resident_ = 0, entries_ = 0, experts_ = 0, swaps_ = 0;
    double gib_ = 0;
    std::string name_;
};

}  // namespace strata::core
