// src/qwen35/q36_peer_prefill.cuh - STRATA_Q36_PREFILL_PEERS: a prompt chunk's routed experts on the helper GPUs.
//
// THE PROBLEM: a prompt chunk (T > kGemmT) takes the MMQ branch of q36_engine.cu's layer(), which streams every
// not-yet-resident expert's blob from host memory over PCIe on the primary's `cst` stream (stream_kick), and the
// helper GPUs sit idle while it does.  That upload, not the arithmetic, is the bulk of a prompt chunk's expert time.
//
// THIS DOES the same arithmetic the primary's MMQ branch does (int8 MMQ, q8_1 activations, gate+up as one [2 FF]
// product, SwiGLU, down), but for the entries whose expert the helper holds in its VRAM cache, on the helper's own
// device, reading the weights IN PLACE from that cache - no H2D of the blob at all.  Its result rows go straight
// into the primary's `eparts` across the P2P link (cudaMemcpyPeerAsync), never through host memory.
//
// THIS IS DELIBERATELY NOT PeerExperts (src/core/peer_experts.cpp) NOR RemoteExperts::begin: both are fixed to
// their own window geometry (PeerExperts: H = 2560 / CAP = MAXT*10 / O(n^2) de-duplication; RemoteExperts::begin:
// n_tok <= MAXT <= 8).  A prompt chunk is thousands of entries wide and q36's rows are 2048, so neither launch
// fits.  What IS reused is the residency those classes already built (open / holds / slot_ptr), which is why this
// class takes a RemoteExperts& rather than opening a cache of its own.
//
// OPT-IN, and the default path is untouched: q36_engine.cu calls none of this unless STRATA_Q36_PREFILL_PEERS is
// set AND the branch conditions hold AND a two-way P2P probe passed.  With the flag off, no object of this type is
// constructed and no statement added here executes.
#pragma once

#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {
class RemoteExperts;
}

namespace strata::qwen35 {

/// One helper GPU's prompt-chunk expert executor.  Owns a device-side stream, the staged activations and MMQ
/// scratch, all on the helper's device; the only memory it touches on the primary is `eparts` (through P2P writes)
/// plus the two `const` inputs prepare() copies from.
///
/// ## The row contract (the one thing the engine and this class must agree on)
///
/// A window is `n_ent = n_tok * k` entries, entry e = token (e / k), routing slot (e % k).  Entry e BELONGS to this
/// helper iff `owns(layer, ids[e])`.
///
/// **Order = entry index, ascending.**  This helper's owned entries, taken in increasing e, fill the contiguous
/// `eparts` run `[eparts_row_base, eparts_row_base + rows())` in exactly that order.  So the entry at rank r (0-based
/// among this helper's owned entries, e ascending) lands in primary row `eparts_row_base + r`, and `launch()` writes
/// one contiguous run.  No expert ordering is implied or allowed: it is a plain prefix count over e, which is why
/// the engine can reproduce it on the device without seeing this class's internal grouping.
///
/// The engine must therefore give helper k the base `rows_local + sum_{j < k} rows_j` (its own local entries first,
/// then each helper's run in helper order) and build its per-entry row map (`prow2`) with the same rule: entry e's
/// row is `rinv[e]` when the primary computes it, else `eparts_row_base(owner) + rank_owner(e)`.
///
/// Internally this class may group entries by expert in any order it likes; it maps its output back onto the run
/// above through MMQ::Product's `ids` (output row = ids[activation row]), never by reordering the run.
///
/// Rows of the entries this helper does NOT own are never read and never written here: they stay the primary's.
class PeerPrefill36 {
public:
    PeerPrefill36();
    ~PeerPrefill36();
    PeerPrefill36(const PeerPrefill36&) = delete;
    PeerPrefill36& operator=(const PeerPrefill36&) = delete;

    /// Bind to `helper` (whose VRAM cache must already be open and must outlive this object) and allocate the
    /// helper-side buffers for windows of up to `cap_entries` entries (`n_tok * k`).
    ///
    /// `n_embd` / `n_ff` are q36's expert widths (2048 / 512).  `layouts` is the SAME per-layer native expert
    /// layout table the engine handed to RemoteExperts::open() (length `n_layers`), so the offsets and types used
    /// here are the ones the cache really holds.  Fails (false + `err`) rather than guessing: a null layout table,
    /// a zero size, or a device that cannot be set.
    bool open(const strata::core::RemoteExperts& helper, int64_t n_layers, int64_t n_expert, int64_t k,
              int64_t n_embd, int64_t n_ff, int64_t cap_entries,
              const strata::kernels::NativeExpertLayout* layouts, std::string& err);
    void close();
    bool valid() const;

    /// Whether the helper's cache holds (layer, expert), i.e. whether launch() can compute it.  The engine asks
    /// this to build its per-entry ownership plan and to decide which experts stream_kick may skip.
    /// False for a closed executor, a layer past n_layers, or an out-of-range expert.
    bool owns(int64_t layer, int32_t expert) const;

    /// The largest window open() allocated for.
    int64_t cap_entries() const;

    // ---- per layer ----
    /// Stage the primary's activations `h_primary` ([n_tok][n_embd] floats, the MoE input `h`) and routing
    /// `ids_primary` ([n_tok * k] ints) into the helper.  Both live on the PRIMARY device; the copies are P2P.
    /// Rejects (false + `err`) a window larger than cap_entries().  Enqueued on the helper's stream.
    bool prepare(int64_t layer, const float* h_primary, const int32_t* ids_primary, int64_t n_tok, std::string& err);

    /// Compute this helper's owned entries of the window prepare() staged, on the helper, and write their rows
    /// into the PRIMARY's `eparts` starting at `eparts_row_base` (the run order documented above).
    /// `cap_entries` and the staged window bound it; nothing here reads host memory.
    bool launch(int64_t layer, float* eparts_primary, int64_t eparts_row_base, std::string& err);

    /// Record the helper's completion and make the PRIMARY stream `st_primary` wait for it, so a kernel the caller
    /// issues on `st_primary` afterwards reads this helper's rows.  Nothing is copied to the host.
    bool wait(cudaStream_t st_primary, std::string& err);

    /// Owned entries the last launch() computed (0 before any), and launches so far (the evidence line's numbers).
    int64_t rows() const;
    int64_t launches() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::qwen35
