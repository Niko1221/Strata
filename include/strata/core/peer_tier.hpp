// include/strata/core/peer_tier.hpp - what the engine needs from a second expert tier, and nothing more.
//
// There are two of them and they have nothing in common below this line.  `PeerExperts` (peer_experts.hpp)
// puts the tier on a second CUDA card and reaches it over P2P.  `OclPeer` (ocl_peer.hpp) puts it on a card
// no CUDA runtime will drive - the MI50 / gfx906 - through AMD's OpenCL 2.0 runtime, with the kernel set
// transcribed and verified bit-exact against the CUDA one format by format.
//
// The engine only ever asks five things: whether the tier is usable at all, which (layer, expert) pairs it
// holds, where those blobs are if the primary can read them directly, how to start this layer's share, and
// how to collect it.  Everything a caller does with a tier goes through `ExpertDispatch::peer`, so this is
// the whole contract; the two implementations carry their own options beyond it.
//
// `p2p()` being false is the important case: an OpenCL buffer has no CUDA address, so the prompt path's
// peer streaming declines and stays on the primary.  The decode path - where the tier earns its keep - does
// not care, because its rows come back through ordinary mapped host memory.
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

struct PeerTier {
    virtual ~PeerTier() = default;

    /// Whether open() succeeded.
    virtual bool valid() const = 0;
    /// Whether the primary can read the tier's blobs directly (NVLink / PCIe P2P).  False makes the prompt
    /// path's peer streaming decline - see Prefill::set_peer.
    virtual bool p2p() const = 0;
    /// The CUDA device ordinal, or -1 for a tier with no CUDA device at all.
    virtual int device() const = 0;
    /// The last launch wrote its rows into the mapped host rows itself, so the caller must not touch them.
    virtual bool launched_direct() const = 0;

    /// Whether the tier holds (layer, expert) right now.
    virtual bool has(int64_t layer, int64_t expert) const = 0;
    /// The device address of (layer, expert)'s blob, or null when it is not resident OR the tier has no
    /// CUDA-visible address at all.  Only meaningful together with p2p().
    virtual const uint8_t* slot_ptr(int64_t layer, int64_t expert) = 0;

    /// Starts the tier's share of one layer: the entries i with kind[i] == 2 (x: n_tok rows of n_embd
    /// floats).  With `out` (pinned, portable host rows) the tier may write its rows straight there.
    virtual bool launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                        const int32_t* kind, std::string& err, float* out = nullptr) = 0;
    /// Waits for the launched share and writes its rows into `out` (row i = entry i, n_embd floats).
    virtual bool finish(float* out, std::string& err) = 0;

    // ---- the adaptive tier (the decode loop's swap policy).  A tier that never swaps - one that is
    // filled once and left alone - is a perfectly good tier, so these all have answers that mean
    // "nothing to do" and only the two real tiers bother to override them.
    virtual int64_t resident() const { return 0; }          ///< experts the tier holds
    virtual double gib() const { return 0.0; }              ///< bytes they occupy
    virtual int64_t entries() const { return 0; }           ///< routed entries the tier computed
    /// Swap up to `max_swaps` experts routed at least twice that neither the primary (`res0`, slot or -1)
    /// nor the tier holds into the tier's least-used slots.  Call only between verify windows.
    virtual bool adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
        (void) usage; (void) res0; (void) max_swaps; (void) err;
        return true;
    }
    /// Publishes the swaps `adapt` queued that have landed.  `wait` also waits for the ones still in flight.
    virtual void apply_pending(bool wait) { (void) wait; }
    /// What to call this tier in the engine's own messages ("peer GPU 1", "OpenCL 0").
    virtual std::string describe() const { return "peer"; }
};

}  // namespace strata::core
