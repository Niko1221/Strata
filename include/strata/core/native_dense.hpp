#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// `[layer_lo, layer_hi)` restricts the load to one stage's layers, the way `WeightTable::load`'s range
    /// does: the projection is uploaded only for layers this stage runs, and `hi < 0` loads every layer.
    ///
    /// THE RANGE MUST BE THE SAME ONE THE CANONICAL ARENA USED.  `served_names` is what puts these tensors
    /// into the canonical loader's `skip` set, so a range applied in one place and not the other leaves a
    /// tensor with neither canonical bytes nor a native override - two null pointers where a weight should
    /// be, which is not an error return but a kernel reading address 0.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D, in range), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err,
                             int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// What each layer's native projections cost, from the GGUF headers only - the split search's price
    /// model, the counterpart of `WeightTable::layer_bytes`, and the same eligibility `load` applies.
    /// `per_layer` comes back sized to `n_layers`; summing a range equals what `load` allocates for it.
    static bool served_layer_bytes(const std::vector<std::string>& shards, bool include_ple_key, int64_t n_layers,
                                   std::vector<uint64_t>& per_layer, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
};
} // namespace strata::core
