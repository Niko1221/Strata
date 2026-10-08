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
    /// With layer_hi >= 0, only the `blk.<l>.` matrices with layer_lo <= l < layer_hi are uploaded (a layer
    /// split's stage holds its own layers' projections, not the whole model's); the others keep data == nullptr.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// What `load` would upload for EACH LAYER, in bytes, from the GGUF headers only - the same
    /// eligible/type/shape rules and the same `native_mmvq_weight_bytes` the upload uses, so it prices the
    /// native half of a layer split's stage without allocating anything.  `out` is `n_layers` long and indexed
    /// by the GLOBAL layer ordinal (a layer the model has none of, such as a trunk's MTP block, is 0).  The
    /// whole curve in one pass, because a placement search wants every range and one pass over the shard
    /// headers is what keeps it cheap enough to run at every start.
    ///
    /// It is the ONLY way to price that half: these matrices are not in the canonical arena at all, so
    /// `WeightTable::pool_bytes` cannot see them, and on glm5-next they are the whole of a stage's weights.
    static bool served_bytes_per_layer(const std::vector<std::string>& shards, bool include_ple_key,
                                       int64_t n_layers, std::vector<uint64_t>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Whether the model's own block PAST the trunk (glm5-next's `--mtp` draft block, index `n_layers`) is to be
    /// uploaded too.  Process-wide, read by the next `load`; off by default, which is the behaviour every run had
    /// before the block was packable.  See `native_dense.cpp`'s `eligible`/`in_range` for why it is needed at all:
    /// that block's quantized tensors are written into `index.txt` as "served from the GGUF", so nothing else
    /// gives them bytes.
    static void set_draft_block(bool on);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size() - packed_keys_.size(); }

private:
    std::vector<void*> weights_;
    std::vector<const void*> packed_keys_;   // GGUF-layout pointers registered with STRATA_Q8_PACKED=1
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
};
} // namespace strata::core
