// include/strata/core/weights.hpp - the dense weights, loaded into VRAM in ENGINE form.
//
// The pack is the ARTIFACT's business; this is the ENGINE's.  The differences are deliberate, and they run in
// BOTH directions:
//
//   * SHRINKING.  A tensor whose source type is BF16 is stored in the pack promoted to 32 bits, because that is
//     lossless and it keeps the canonical decoder simple, but a running engine re-rounds it to 16 - which costs
//     nothing at all, because the value CAME from 16 bits.  Per `tools/pack_budget.py` that saving is the
//     difference between the dense weights fitting in VRAM beside the expert cache and not fitting at all.
//
//   * GROWING.  90 of the 303 quantized tensors carry fp16 SCALES in the pack (all 58 Q2_0 expert tensors, plus
//     Q4_0, Q5_0, Q8_0 and IQ4_NL) because that is what the source block holds.  Every kernel here takes
//     `const float* scales`, and fp16 is a subset of f32, so this loader WIDENS them - exactly, changing no
//     value - for 11,407,360 B.  See `docs/pack-format.md` Â§2.4; assuming 4 bytes per scale instead of reading
//     `scales_fp16` is what made the plane arithmetic disagree with the pack for a whole round.
//
//     pack      5.305 GiB dense + 0.444 GiB embd
//     engine    4.067 GiB dense + 0.444 GiB embd + 10.88 MiB widened scales = 4.522 GiB
//
// WHAT THE LOADER DOES NOT DO: it does not know what any tensor MEANS.  It reads a flat index
// (`<pack>/index.txt`, written by `tools/pack_index.py`), copies each tensor into one arena at a recorded
// offset - as a list of PLANE SEGMENTS, because a scale plane can change width and the rest cannot - and hands
// back a name -> WeightRef map.  Geometry, forms and semantics belong to the caller.
//
// WHY AN INDEX FILE AND NOT A JSON PARSER: a JSON parser in the engine would be a new, unaudited component
// whose failure mode is a wrong byte offset - which decodes to a plausible weight and produces plausible
// logits.  Python wrote the manifest and already parses it correctly.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// What the loader had to DO for a tensor, decided by `tools/pack_index.py` and written into the index
/// rather than inferred here.  The four cases are distinguishable in the manifest only by combining
/// `source_type` with `values_fp16`, and getting that combination wrong yields a plausible tensor.
enum class WeightKind : int {
    Verbatim = 0,   ///< quantized planes, or F16 already stored as 2 bytes: copy straight through
    Bf16InF32 = 1,  ///< the pack holds the bf16 value promoted to f32: take the HIGH 16 BITS (exact)
    F32 = 2,        ///< copy 4 bytes per element
    F16InF32 = 3,   ///< an f16 value promoted to f32: a real f32->f16 conversion, not a truncation
};

struct WeightRef {
    const void* data = nullptr;   ///< device pointer inside the arena
    uint64_t bytes = 0;
    int64_t ne0 = 0;              ///< the manifest's own shape, ne0 the CONTIGUOUS axis
    int64_t ne1 = 0;
    int64_t elements = 0;
    WeightKind kind = WeightKind::Verbatim;
    /// The canonical-form attributes, valid only for a quantized VERBATIM tensor (`code_bits != 0`).
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;

    /// THE THREE PLANE SIZES, IN ENGINE FORM, in the order they sit inside `data`.  Recorded by the loader
    /// from what the index TOLD it rather than re-derived by each caller: the one time this layout was
    /// re-derived independently it assumed 4-byte scales for tensors whose pack scales are fp16, and the
    /// only reason that was not a silent wrong answer is that the re-derivation checked itself and refused.
    /// Two places computing one layout is two places to disagree.
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;    ///< always 4 B per scale in the arena, whatever the pack held
    uint64_t offset_bytes = 0;    ///< 0 when the form has no offset plane

    /// WHERE THIS TENSOR CAME FROM, so a caller can check the loader's work against the pack's own bytes
    /// instead of against the loader's opinion of them.  Without `src_off` a test can only compare two
    /// derived numbers; with it, the widening of an fp16 scale plane can be verified value by value against
    /// `<file>.bin` - and the pack files are named, so `file_id` says which one to open.
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    int file_id = 0;
    bool scales_fp16 = false;     ///< the PACK's width, before the loader widened it

    /// WHICH ACTIVATION THIS WEIGHT WANTS - 0 for Q8_0, 1 for Q8_K.  Decided by 	ools/pack_index.py from
    /// source_type and carried here, because **the S-form CANNOT express it**: Q5_0 and Q5_K are both 8-bit
    /// with bias -16, and IQ4_NL and IQ4_XS differ in nothing the S-form holds.  docs/activation-contract.md.
    int act_kind = 0;
    bool wants_q8k() const { return act_kind == 1; }

    bool quantized() const { return code_bits != 0; }

    // Optional native GGUF projection, owned by NativeDense. All references in
    // one table share its scratch and must execute on one ordered session stream.
    const void* native_data = nullptr;
    void* native_q8_1 = nullptr;
    int native_type = -1;
    /// Plan v0.3 P1: false when the loader SKIPPED this tensor's canonical bytes because another form serves it
    /// (native GGUF projections, the native head).  The metadata above stays valid; `data` is null.
    bool resident = true;
};

struct LoadReport {
    size_t tensors = 0;
    uint64_t arena_bytes = 0;
    size_t re_rounded = 0;        ///< tensors written as 16 bits out of a 32-bit container
    uint64_t bytes_saved = 0;     ///< what those conversions saved against the pack
    double read_ms = 0.0;
    double upload_ms = 0.0;
};

/// The layer ordinal of a per-layer tensor name (`blk.<n>.<suffix>`), or -1 for a model-global name
/// (`token_embd`, `output`, `output_hc_*`) or anything that is not a per-layer name at all.
///
/// ONE PARSER.  The loader's range carve and NativeDense both have to answer "which layer is this", and
/// two answers to that question is a stage quietly running another stage's layer - which is not a crash,
/// it is a wrong number.  The suffix list lives in `layout.cpp` and is deliberately NOT duplicated here:
/// this function only needs the `blk.<n>.` prefix, which is the whole of what a range carve cares about.
inline int64_t tensor_layer(const std::string& name) {
    if (name.size() < 6 || name.compare(0, 4, "blk.") != 0) return -1;
    int64_t v = 0;
    size_t i = 4;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) {
        v = v * 10 + (name[i] - '0');
        if (v > 1000000) return -1;   // not a layer ordinal - refuse rather than overflow into one
    }
    return (i > 4 && i < name.size() && name[i] == '.') ? v : -1;
}

/// True when the layer range [lo, hi) DROPS this name: a per-layer tensor outside the range.
///
/// `hi < 0` disables the range entirely, which is the whole-model carve and is byte-identical to no
/// filter at all (the kept rows land at the same offsets in the same order).  A name in `keep` is never
/// dropped whatever the range says - that is for tensors a stage needs even though it does not run their
/// layer.  Model-global tensors have no layer ordinal, so the range never drops them: they are needed by
/// the stage the range would not have given them to (the head, the embedding).
inline bool layer_out_of_range(const std::string& name, int64_t lo, int64_t hi,
                               const std::set<std::string>* keep = nullptr) {
    if (hi < 0) return false;
    if (keep != nullptr && keep->count(name) != 0) return false;
    const int64_t l = tensor_layer(name);
    return l >= 0 && (l < lo || l >= hi);
}

class WeightTable {
public:
    /// The arena size the index asks for, readable WITHOUT loading anything - so a caller can size its
    /// `DeviceArena` before committing to the load, and a plan that cannot fit is refused at startup rather
    /// than halfway through a 5 GB upload.
    ///
    /// With `skip`, the size of the compacted arena that holds every tensor EXCEPT the named ones.  With a
    /// layer range `[layer_lo, layer_hi)`, the size of the arena that holds only that range's per-layer
    /// tensors plus every model-global one - what one stage of a layer split actually runs on.
    static bool pool_bytes(const std::string& pack_dir, uint64_t& out, std::string& err,
                           const std::set<std::string>* skip = nullptr,
                           int64_t layer_lo = 0, int64_t layer_hi = -1,
                           const std::set<std::string>* keep = nullptr);

    /// Load every tensor in `<pack_dir>/index.txt` into `arena_base`.
    ///
    /// `arena_bytes` MUST be at least `pool_bytes`; the loader checks rather than trusting the caller,
    /// because the failure mode otherwise is a device write past the end of the arena.
    /// With `skip`, or with a layer range, the dropped tensors are not read: their rows keep their metadata
    /// with `data == nullptr` and `resident == false`, and the others are packed into the compacted arena
    /// `pool_bytes` sized.  THE CALLER MUST NOT RUN A DROPPED TENSOR: `find` returns non-null for it, so a
    /// consumer that only checks for null hands a null device pointer to a kernel.
    bool load(const std::string& pack_dir, void* arena_base, uint64_t arena_bytes, std::string& err,
              const std::set<std::string>* skip = nullptr,
              int64_t layer_lo = 0, int64_t layer_hi = -1,
              const std::set<std::string>* keep = nullptr);

    /// What one stage's arena costs for a layer range, priced WITHOUT loading anything: per-layer and
    /// model-global byte totals, aligned exactly as `load` places them.  Pure arithmetic, which is what
    /// lets the split search price a candidate placement before committing VRAM to it - the same reason
    /// `session_bytes` is pure (see the session carve).
    ///
    /// `per_layer` comes back sized to `n_layers`; `keep_bytes` totals the names in `keep`, which every
    /// stage holds whatever its range.  For any range, `globals + keep_bytes + sum(per_layer[lo,hi))`
    /// equals `pool_bytes(..., lo, hi)` - the per-row alignment is order-independent, so summing a range
    /// agrees with the loader's own compaction exactly.
    static bool layer_bytes(const std::string& pack_dir, int64_t n_layers, std::vector<uint64_t>& per_layer,
                            uint64_t& globals, uint64_t& keep_bytes, std::string& err,
                            const std::set<std::string>* skip = nullptr,
                            const std::set<std::string>* keep = nullptr);

    const WeightRef* find(const std::string& name) const;
    const std::map<std::string, WeightRef>& all() const { return table_; }
    const LoadReport& report() const { return report_; }

private:
    friend class NativeDense;
    std::map<std::string, WeightRef> table_;
    LoadReport report_;
};

}  // namespace strata::core
