// include/strata/spec/dflash_drafter.hpp - DFlash2 block-diffusion drafter (--dflash), in place of the MTP layer.
//
// The draft model (z-lab DFlash2, a 5-layer non-causal transformer) runs in fused Metal kernels
// (metal/src/engine/dflash.cpp) on its own Metal context. It keeps a positional K/V of the target's features: per
// committed token, the outputs of target layers 5/15/25/35/45 averaged over the four hyper-connection streams
// (5 x 2560 floats). A draft decodes [last token, mask x (block-1)] at the next positions and walks the selector
// lattice (DFlash2's candidate path).
// The verify window checks every proposal, so the drafter changes the speed only, never the output.
// A build with STRATA_DFLASH_LLAMA can run the same graph through llama.cpp's DFLASH instead (the parity reference).
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::spec {

struct DflashBackend;

class DflashDrafter {
public:
    DflashDrafter();
    ~DflashDrafter();
    DflashDrafter(const DflashDrafter&) = delete;
    DflashDrafter& operator=(const DflashDrafter&) = delete;

    /// Load the draft GGUF (one that carries token_embd + output, see tools/dflash_head.py). False + err on failure.
    /// Backend: the native fused-Metal drafter (default), or llama.cpp's DFLASH graph when the build has it
    /// (STRATA_DFLASH_LLAMA; STRATA_DFLASH_NATIVE=0, or kLlama).
    enum class Backend { kAuto, kLlama, kNative };
    bool load(const std::string& path, std::string& err, Backend backend = Backend::kAuto);
    bool loaded() const;

    /// Target layers the features come from (completed outputs, zero-based) and the feature width per token.
    static constexpr int kLayers[5] = {5, 15, 25, 35, 45};
    static constexpr int kNumLayers = 5;
    int target_hidden() const;   // 2560
    int feat_width() const { return kNumLayers * target_hidden(); }
    int block() const;           // trained block size (8): at most block()-1 drafts

    /// The features of `n` committed tokens at positions pos0.. (rows of feat_width() floats). Positions at or past
    /// pos0 that the drafter already holds are dropped first; a gap (positions it never saw) clears it.
    bool inject(const float* feats, int n, int pos0, std::string& err);
    /// Up to max_n drafts that follow `id_last` (the token at position pos0, not yet verified). Stops where the
    /// selector's confidence falls below p_min (0 = never). Returns the count; the drafter's noise rows are removed.
    /// `conf` (optional, max_n floats): the selector's confidence in each draft (its softmax at the chosen candidate).
    int draft(int32_t id_last, int pos0, int max_n, float p_min, int32_t* out, float* conf = nullptr);
    /// Forget every position >= pos (0 = all).
    void trim(int pos);
    /// The last position held (-1 = none).
    int pos_max() const;
    /// Debug: the last draft's selector lattice, one row of feat_width() floats per block row (llama.cpp's nextn rows:
    /// 16 candidate ids, then the 16 x 16 scores [predecessor][successor]); valid until the next draft.
    const float* lattice() const;
    /// Host time in inject() and draft() (ms) and their call counts; the engine reads and zeroes them per request.
    double ms_inject = 0, ms_draft = 0;
    int64_t n_inject = 0, n_draft = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
    std::unique_ptr<DflashBackend> nat_;   // set: the native backend answers instead of d_
};

}  // namespace strata::spec
