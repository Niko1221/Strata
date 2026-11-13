// include/strata/core/shard.hpp - what tensor parallelism does to each tensor, as rules that need no GPU.
//
// WHERE THIS COMES FROM.  llama.cpp's tensor mode decides the same thing per operator, in
// `ggml/src/ggml-backend-meta.cpp`'s `ggml_backend_meta_get_split_state` and its `handle_*` helpers
// (`handle_mul_mat:577`, `handle_per_row:542`, `handle_flash_attn_ext:773`, `handle_ssm_conv:801`,
// `handle_gated_delta_net:813`, `handle_lightning_indexer:1036`, `MIRRORED:868`, `UNKNOWN:1063`).  What can be
// taken from it is the table, not the machinery: Strata has no `ggml_cgraph` to dispatch, so a split state
// becomes an explicit argument to the layer functions instead of a per-node callback.  The expert rule is
// llama.cpp's too - `handle_mul_mat:600-603` splits the EXPERT AXIS with a replicated activation, which is
// "each rank's cache holds its own experts, whole", and NOT a split of one expert's weights.
//
// THE AXES, AND WHY THEY COST DIFFERENTLY.
//
//   `Output` splits ne1, the output/feature axis.  Each rank computes DIFFERENT outputs from the SAME input,
//   so nothing has to be combined: the result is bit-identical to an unsplit run.  In this engine's storage
//   (ne0 is the contiguous axis, `layout.cpp:9`) a range of ne1 is a contiguous run of rows, and the quant
//   blocks run along ne0 inside a row, so an Output split is legal for EVERY format at any row boundary,
//   with no alignment to respect.
//
//   `Input` splits ne0, the contraction axis.  Each rank computes a PARTIAL dot product, so the two partials
//   must be added across the cards (llama.cpp's `GGML_BACKEND_SPLIT_AXIS_PARTIAL`, :591-593) and the
//   arithmetic is then not bit-identical to a single-card run: the summation order changed.  It is still
//   deterministic - same build, same config, same output.  The boundaries have constraints: a row is stored
//   as groups of `group_elems`, and the activation the kernel quantizes into is coarser still (Q8_K is 128
//   elements, `src/core/layer.cpp:66`), which is what `kInputBlock` stands for.
//
//   `Replicated` is every tensor whose cost is small, or whose value the two ranks must AGREE on.  The router
//   is the sharp case: two ranks that pick different expert ids are running two different models and nothing
//   downstream can notice.  The hyper-connection block is replicated for a measured reason: splitting it
//   needs two more reduces per sublayer (the 320-wide down projection and the 4-wide injection), which at 48
//   layers is roughly 1.5 ms per window of added round trips against roughly 1.1 ms of bandwidth saved.  The
//   shared expert is replicated because 640/2 = 320 is not a multiple of 128, and it is ~6% of a layer's
//   dense bytes.
//
//   `Rank0` is held by one rank alone: the head and the embedding are read once per token, their input is
//   replicated anyway, and splitting them would mean gathering the logits before the sampler can see them.
//
// WHAT THE TABLE BUYS.  `shard_table` builds it per rank from the geometry and REFUSES a geometry it cannot
// serve, naming the tensor and the number.  The failure it exists to prevent is a two-way split of a two-head
// KV cache on an asymmetric pair, where a rank quietly ends up owning no head at all - kvmem's documented
// trap (`D:\AI\tools\kvmem\kvmem-src\docs\tensor-parallel-design.md:74`).
#pragma once

#include "strata/core/geometry.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

/// The rank count this table is written for.  `n_head_kv` is 2, so a third rank has no KV head to own and
/// the table refuses rather than inventing one.
inline constexpr int kShardRanks = 2;

/// The coarsest thing an `Input` boundary has to respect: Q8_K's 128-element block, what the K-quantised dot
/// products quantize their activation into.  The 32-element groups (Q4_0/Q5_1/Q8_0/IQ4_NL) divide 128, so
/// aligning to 128 is the conservative choice for every format in the tree (`docs/QUANTS.md:16-22`).
inline constexpr int64_t kInputBlock = 128;

/// A half-open range of indices along the split axis.
struct ShardRange {
    int64_t begin = 0;
    int64_t end = 0;
    int64_t count() const { return end - begin; }
};

enum class ShardAxis : int {
    Replicated = 0,   ///< every rank holds the whole tensor
    Output = 1,       ///< ne1 is split; no reduce, bit-identical to a single card
    Input = 2,        ///< ne0 is split; a reduce follows, and the result is reproducible but not identical
    Rank0 = 3,        ///< rank 0 holds it and rank 1 does not read it
};

/// One tensor's footing.  `ranges` is 1 for every split tensor except the GDN projections, whose input is
/// `q|k|v` in one tensor and has to be cut into three pieces per rank to keep whole heads.
inline constexpr int kShardMaxRanges = 6;
struct ShardSpec {
    ShardAxis axis = ShardAxis::Replicated;
    int ranges = 0;
    ShardRange range[kShardMaxRanges] = {};
};

/// What the LOADER should keep of one named tensor - the same decision as `ShardSpec`, plus the two things the
/// loader needs and the layer code does not: whether this rank holds any bytes at all, and the byte budget.
struct TensorShard {
    ShardAxis axis = ShardAxis::Replicated;
    /// false: this rank keeps NOTHING of it (a `Rank0` tensor on the rank that does not hold it).  Distinct
    /// from a range that happens to be empty, which is a bug, and distinct from `Replicated`.
    bool keep = true;
    int ranges = 0;
    ShardRange range[kShardMaxRanges] = {};
    /// How many units of the split axis this rank keeps.  Zero for `Replicated`, where the axis is not split.
    int64_t kept() const {
        int64_t n = 0;
        for (int i = 0; i < ranges; ++i) n += range[i].count();
        return n;
    }
};

/// Tensor name -> what to keep.  Names are the pack's, i.e. `blk.<layer>.<suffix>` for the layer tensors.
using ShardPlan = std::map<std::string, TensorShard>;

/// The bytes a shard keeps of one plane.  Every plane is uniform per unit of the split axis - a row of a
/// quantized tensor is a whole number of quant blocks, and a column of one is one value per row - so the
/// shard is `full / total * kept`, and the ONLY interesting case is when `full` is not divisible by `total`:
/// that means the plane is not uniform and the arena this would size is not the arena the pack describes.
/// Refused loudly, naming the tensor, rather than sizing an arena that is off by a row.
inline bool shard_bytes(uint64_t full, int64_t total, int64_t kept, uint64_t& out, const std::string& name,
                        std::string& err) {
    if (total <= 0 || kept < 0) {
        err = name + ": a shard of " + std::to_string(kept) + " of " + std::to_string(total) + " units";
        return false;
    }
    if (full % (uint64_t) total != 0) {
        err = name + ": " + std::to_string(full) + " B over " + std::to_string(total) +
              " units does not divide evenly, so a plane of this tensor is not uniform per unit and a shard of "
              "it cannot be sized";
        return false;
    }
    out = full / (uint64_t) total * (uint64_t) kept;
    return true;
}

/// One layer's table, keyed by the tensor name's suffix ("attn_q.weight"), so one table serves all 48 layers.
/// The model's tensor names and shapes are `family_layout.hpp`'s (`add_layer_tensors:160-215`,
/// `add_global_tensors:220-231`); EVERY name it lists has an entry here, because a lookup that misses has to
/// be an error at the call site rather than a silent "hold the whole thing on both cards".
struct ShardTable {
    static constexpr int kMax = 64;
    struct Entry {
        const char* suffix;
        ShardSpec spec;
    };
    int n = 0;
    Entry entries[kMax] = {};
    /// `std::strcmp` and not `std::string`: an uncaptured token path asks this once per tensor per layer, and
    /// a heap allocation per lookup on that path is exactly the kind of cost this tree measures.
    const ShardSpec* find(const char* suffix) const {
        for (int i = 0; i < n; ++i)
            if (std::strcmp(entries[i].suffix, suffix) == 0) return &entries[i].spec;
        return nullptr;
    }
    bool add(const char* suffix, const ShardSpec& s, std::string& err) {
        if (n >= kMax) {
            err = std::string("shard table: more than ") + std::to_string(kMax) + " tensors";
            return false;
        }
        entries[n].suffix = suffix;
        entries[n].spec = s;
        ++n;
        return true;
    }
};

// ---- the rules, all inline: the table is data, and it is useful to tests and to the loader without
// linking anything (the same shape as `program/split_rules.hpp`) --------------------------------

/// The shape a sharded tensor should have on THIS rank, from its full shape and the spec.  This is the rule
/// the loader obeys and `check_layer` asserts, in one place so the two cannot drift: a tensor the loader cut
/// in half and the shape check expects whole is a mismatch that would only show up as wrong arithmetic.
///
/// `Replicated` and `Rank0` keep the full shape (rank 0's tensors are whole or absent, never partial); an
/// `Output` split halves ne1 and leaves ne0; an `Input` split halves ne0 and leaves ne1.
inline bool shard_shape(const ShardSpec& s, int64_t ne0, int64_t ne1, int64_t& out_ne0, int64_t& out_ne1,
                        std::string& err) {
    out_ne0 = ne0;
    out_ne1 = ne1;
    if (s.axis == ShardAxis::Replicated || s.axis == ShardAxis::Rank0) return true;
    const bool output = s.axis == ShardAxis::Output;
    int64_t kept = 0;
    for (int i = 0; i < s.ranges; ++i) kept += s.range[i].count();
    const int64_t total = output ? ne1 : ne0;
    if (kept <= 0 || kept > total) {
        err = "a shard of " + std::to_string(kept) + " of " + std::to_string(total) + " units on the " +
              (output ? "output" : "contraction") + " axis";
        return false;
    }
    if (output) out_ne1 = kept;
    else out_ne0 = kept;
    return true;
}

/// THIS RANK'S SLICE OF ONE AXIS, for a caller that writes into (or reads from) a full-width buffer: `begin`
/// is where its part starts and `count` how wide it is.  `full` is the UNSPLIT extent of that axis.
///
/// Returns false when the split is not ONE contiguous slice - the GDN `q|k|v` takes three ranges, and a caller
/// that needs a single offset and a single length cannot use it.  Saying so is the point: the alternative is
/// returning the first range and quietly computing two thirds of an attention projection.
///
/// A tensor with no entry, or one this rank holds whole (replicated, rank 0's own), is the WHOLE axis - so the
/// layer code can ask unconditionally and the unsharded path gets `(0, full)` and stays byte-identical.
/// A template so the same rule serves the TABLE's entry (`ShardSpec`) and the PLAN's (`TensorShard`): both
/// carry an axis and up to `kShardMaxRanges` ranges, and the only difference between them is what the loader
/// needs, not what a caller reading one slice needs.
template <typename T>
inline bool shard_slice(const T* ts, ShardAxis axis, int64_t full, int64_t& begin, int64_t& count) {
    const bool split_this_axis = ts != nullptr && ts->axis == axis;
    if (!split_this_axis || ts->ranges == 0) {
        begin = 0;
        count = full;
        return true;
    }
    if (ts->ranges != 1) return false;
    begin = ts->range[0].begin;
    count = ts->range[0].count();
    return begin >= 0 && count > 0 && begin + count <= full;
}

/// WHERE THIS RANK'S PART OF THE CONTRACTION AXIS STARTS, in each of the activation forms a layer holds at
/// once.  A contraction split means every rank computes a PARTIAL SUM over its own columns, so each kernel
/// must be handed ITS slice of the activation and the length of that slice - and the forms are indexed four
/// different ways:
///
///   * f32 / bf16: `k_begin` scalars in (the bf16 form is a different pointer type, not a different offset);
///   * Q8_0: `k_begin / 32` blocks of 34 bytes;
///   * Q8_K: `k_begin / 256` blocks of however many bytes this build's Q8_K block is.
///
/// A `k_begin` that is not a whole number of blocks is REFUSED rather than rounded: the kernel reads from the
/// block it is given, and half a super-block earlier is a plausible-looking activation made of the wrong
/// numbers - a wrong answer, not a crash.  The 256-element bound is the real one for a Q8_K activation, which
/// is stricter than the 128 the static table check uses.
struct KActivation {
    int64_t scalars = 0;   ///< elements into a `float*` or a `uint16_t*`
    int64_t q8_0 = 0;      ///< bytes into a Q8_0 array (32 elements a block, 34 bytes)
    int64_t q8k = 0;       ///< bytes into a Q8_K array (256 elements a block)
};

inline bool k_activation_slice(int64_t k_begin, int64_t q8k_block_bytes, KActivation& out, std::string& err) {
    if (k_begin < 0) {
        err = "a negative contraction offset (" + std::to_string(k_begin) + ")";
        return false;
    }
    if (k_begin % 32 != 0) {
        err = "a contraction offset of " + std::to_string(k_begin) +
              " elements is not a whole number of 32-element Q8_0 blocks";
        return false;
    }
    if (k_begin % 256 != 0) {
        err = "a contraction offset of " + std::to_string(k_begin) +
              " elements is not a whole number of 256-element Q8_K blocks";
        return false;
    }
    if (q8k_block_bytes <= 0) {
        err = "no Q8_K block size";
        return false;
    }
    out.scalars = k_begin;
    out.q8_0 = (k_begin / 32) * 34;
    out.q8k = (k_begin / 256) * q8k_block_bytes;
    return true;
}

/// THE GDN BLOCK'S BUFFERS FOR ONE RANK of a tensor-parallel pair.
///
/// The split is by HEAD: `attn_qkv` is three ranges (q, k, v) cut in half along the heads, so rank `r` of
/// `ranks` works on `ssm_conv_channels / ranks` head-channels and `ssm_v_heads / ranks` v-heads.  That is what
/// makes this a clean split rather than a reshuffle - the conv is per channel, the recurrence is per v-head, and
/// `ssm_out` sums over the v-heads, which is exactly why the contraction split owes a reduction.
///
/// The three ACTIVATION images are FULL width (`n_embd`) on both ranks: the projection reads the whole input
/// row, it is only the weight's columns that are cut.  So the two ranks' totals add up to more than one card's -
/// by exactly those three parts plus the alignment - and a test asserts that rather than the other way round.
///
/// `q8k_block_bytes` is this build's Q8_K block size, passed in rather than hard-coded for the same reason
/// `k_activation_slice` takes it: this header has no kernel dependencies.
inline uint64_t tp_gdn_buffer_bytes(const ModelGeometry& g, int rank, int ranks, int64_t q8k_block_bytes) {
    if (ranks <= 0 || rank < 0 || rank >= ranks) return 0;
    const int64_t C = g.ssm_conv_channels / ranks;      // this rank's head-channels (conv width)
    const int64_t V = g.ssm_value_dim / ranks;          // this rank's v-head outputs
    const int64_t HV = g.ssm_v_heads / ranks;           // ... and its v-heads
    const int64_t ss = g.ssm_state_size;
    const uint64_t parts[] = {
        (uint64_t) (g.n_embd / 256) * (uint64_t) q8k_block_bytes,   // x_q8k  (full width)
        (uint64_t) (g.n_embd / 32) * 34,                           // x_q8_0 (full width)
        (uint64_t) g.n_embd * 2,                                   // x_bf16 (full width)
        (uint64_t) C * 4, (uint64_t) C * 4, (uint64_t) C * 4,      // qkv, conv_out, h
        (uint64_t) HV * 4, (uint64_t) HV * 4, (uint64_t) HV * 4,   // alpha, beta, gate
        (uint64_t) HV * (uint64_t) ss * 4,                         // o
        (uint64_t) V * 4, (uint64_t) V * 4,                        // z, y
        (uint64_t) (V / 256) * (uint64_t) q8k_block_bytes,         // y_q8k
        (uint64_t) (V / 32) * 34,                                  // y_q8_0
        (uint64_t) ss * (uint64_t) HV * (uint64_t) ss * 4,         // the recurrence state
        (uint64_t) C * (uint64_t) (g.ssm_d_conv - 1) * 4,          // the conv state
    };
    uint64_t total = 0;
    for (uint64_t v : parts) total += (v + 15) & ~(uint64_t) 15;   // the same 16-byte alignment the layer uses
    return total;
}

/// THE HEAD RANGES ONE RANK OWNS, derived from cutting `attn_qkv`'s three ranges in half.
///
/// The model's qkv width is `2*128*16 + 128*48` - q, k, then v (`geometry.hpp` derives `ssm_conv_channels`
/// exactly that way, which is what says q and k each have `ssm_k_heads` heads) - and the split puts the same
/// FRACTION of each on a rank.  That is what makes the block runnable at all: for every v-head a rank owns, the
/// recurrence needs the q and k of the SAME index, and half of a fraction still lines up only because both are
/// cut by the same fraction.
///
/// The rank's own `qkv` buffer is `[q | k | v]` of ITS widths - the loader packs the shard's ranges in order -
/// so the kernels are handed counts and offsets and the buffer needs no remapping: the conv is per channel and
/// does not care about the order, and the recurrence indexes q, k and v inside their own sections.
struct TpHeads {
    int64_t q_begin = 0, q_count = 0;   // in heads
    int64_t k_begin = 0, k_count = 0;
    int64_t v_begin = 0, v_count = 0;
    int64_t head = 0;                   // the head width, so callers do not re-derive it
    int64_t qkv_count() const { return (q_count + k_count + v_count) * head; }   // this rank's conv width
    int64_t k_offset() const { return q_count * head; }
    int64_t v_offset() const { return (q_count + k_count) * head; }
};

inline bool tp_heads(const ModelGeometry& g, int rank, int ranks, TpHeads& out, std::string& err) {
    if (ranks <= 0 || rank < 0 || rank >= ranks) {
        err = "rank " + std::to_string(rank) + " of " + std::to_string(ranks) + " is not a rank";
        return false;
    }
    if (g.ssm_k_heads % ranks != 0 || g.ssm_v_heads % ranks != 0) {
        err = "the head counts (" + std::to_string(g.ssm_k_heads) + " q/k, " + std::to_string(g.ssm_v_heads) +
              " v) do not divide across " + std::to_string(ranks) + " ranks - and half a head is not a thing";
        return false;
    }
    out.head = g.ssm_state_size;
    out.q_count = out.k_count = g.ssm_k_heads / ranks;
    out.v_count = g.ssm_v_heads / ranks;
    out.q_begin = (int64_t) rank * out.q_count;
    out.k_begin = (int64_t) rank * out.k_count;
    out.v_begin = (int64_t) rank * out.v_count;
    // the derived width has to be the geometry's own, or the conv would run over the wrong number of channels
    if (ranks > 0 && out.qkv_count() * ranks != g.ssm_conv_channels) {
        err = "the split's channels (" + std::to_string(out.qkv_count()) + " x " + std::to_string(ranks) +
              ") are not the geometry's " + std::to_string(g.ssm_conv_channels);
        return false;
    }
    if (ranks == 1 && out.qkv_count() != g.ssm_conv_channels) {
        err = "the unsplit width is not the geometry's";
        return false;
    }
    return true;
}

/// WHICH RANK A RUN IS, and how many there are.  PER THREAD, because a tensor-parallel pair runs its two ranks
/// as two THREADS in one process (`generate.cpp`), and every per-rank dimension - the head ranges `Verifier::init`
/// and `record_window` compute, the GDN/QSA widths the layers take - is read on each rank's own thread.  A
/// process-wide value hands both ranks the same geometry, which is a deterministic wrong answer rather than a
/// crash: measured, the whole TP decode was garbage for exactly this reason.  A single-card run never sets it, so
/// both accessors answer the "one rank holds the whole model" case and the default path is byte-identical.
inline int& tp_rank_ref() { thread_local int r = 0; return r; }
inline int& tp_ranks_ref() { thread_local int r = 1; return r; }
inline int tp_rank_current() { return tp_rank_ref(); }
inline int tp_ranks_current() { return tp_ranks_ref(); }
/// THE EXPERT GROUP (opt-in, STRATA_TP_DENSE=replicated).  In that mode every rank holds the WHOLE dense model
/// (`tp_ranks_current()` stays 1, so every geometry is the single card's) and only the experts are split between
/// the ranks.  These two say which member of the expert group a thread is; 0 of 1 when the mode is off, where the
/// expert group is the tensor-parallel group itself.
inline int& tp_xrank_ref() { thread_local int r = 0; return r; }
inline int& tp_xranks_ref() { thread_local int r = 0; return r; }
inline int tp_expert_rank() { return tp_xranks_ref() > 0 ? tp_xrank_ref() : tp_rank_ref(); }
inline int tp_expert_ranks() { return tp_xranks_ref() > 0 ? tp_xranks_ref() : tp_ranks_ref(); }
inline void tp_set_expert_ranks(int rank, int ranks) { tp_xrank_ref() = rank; tp_xranks_ref() = ranks; }

/// The engine calls this ONCE, before anything is sized, when it assembles a tensor-parallel pair.  A pair that
/// is not a pair (rank outside [0, ranks)) is refused rather than clamped: it would silently size a rank's
/// buffers from another rank's geometry.
inline bool tp_set_ranks(int rank, int ranks, std::string& err) {
    if (ranks < 1 || rank < 0 || rank >= ranks) {
        err = "rank " + std::to_string(rank) + " of " + std::to_string(ranks) + " is not a rank";
        return false;
    }
    tp_rank_ref() = rank;
    tp_ranks_ref() = ranks;
    return true;
}

/// Turn the per-layer suffix table into this rank's plan over the pack's full tensor names.
///
/// It emits BOTH spellings for every entry - the bare suffix and `blk.<l>.<suffix>` for every layer - instead
/// of deciding which entries are per-layer and which are global.  A name the pack does not have costs one map
/// entry and can never match, and the alternative is a second list of names to keep in step with
/// `family_layout.hpp`, which is exactly the kind of duplicate that drifts.
inline bool shard_plan_layers(const ShardTable& t, const ModelGeometry& g, int rank, ShardPlan& out,
                              std::string& err) {
    out.clear();
    for (int i = 0; i < t.n; ++i) {
        const ShardTable::Entry& e = t.entries[i];
        TensorShard ts;
        ts.axis = e.spec.axis;
        ts.keep = !(e.spec.axis == ShardAxis::Rank0 && rank != 0);
        ts.ranges = e.spec.ranges;
        for (int s = 0; s < e.spec.ranges; ++s) ts.range[s] = e.spec.range[s];
        out[e.suffix] = ts;
        for (int64_t l = 0; l < g.n_layers; ++l)
            out["blk." + std::to_string(l) + "." + e.suffix] = ts;
    }
    if (out.empty()) { err = "shard plan: the table is empty"; return false; }
    return true;
}

/// Whether `n` items divide into two whole, non-empty pieces.  False for an odd `n` or `n < 2`; the caller
/// names the tensor.  This is the check that a two-head KV cache on a two-rank pair still gives each rank one.
inline bool shard_split_even(int64_t n) { return n >= 2 && n % 2 == 0; }

/// The checks that need no model: every range of a width-`ne` axis lies inside it and is non-empty, and the
/// two ranks' ranges TILE the axis exactly once (no gap, no overlap).  An `Input` range must begin and end on
/// a `kInputBlock` boundary WHEN THE AXIS IS A CONTRACTION one (`contraction`, the default): that is where the
/// activation block contract lives.  A one-dimensional tensor splits on its only axis - `ssm_a` is 48 v-head
/// scalars - and there is no dot product to quantise, so the alignment does not apply to it and demanding it
/// would reject a legal split.  `Output` ranges need no alignment either way, which is why they are cheaper.
inline bool shard_check_pair(const ShardSpec& a, const ShardSpec& b, int64_t ne, const char* what,
                             std::string& err, bool contraction = true) {
    const std::string name = what == nullptr ? std::string("?") : std::string(what);
    auto fail = [&](const std::string& why) {
        err = name + ": " + why;
        return false;
    };
    if (a.axis != b.axis) return fail("the two ranks disagree about the axis");
    if (a.axis == ShardAxis::Replicated || a.axis == ShardAxis::Rank0) return true;
    const bool input = a.axis == ShardAxis::Input;
    if (a.ranges < 1 || b.ranges < 1) return fail("a split with no ranges");
    if (a.ranges != b.ranges) return fail("the two ranks have different range counts");

    // Every range inside the axis, aligned when the axis feeds a dot product.
    for (int s = 0; s < a.ranges; ++s) {
        const ShardRange r[2] = {a.range[s], b.range[s]};
        for (int k = 0; k < 2; ++k) {
            if (r[k].count() <= 0) return fail("range " + std::to_string(s) + " of rank " + std::to_string(k) +
                                               " is empty");
            if (r[k].begin < 0 || r[k].end > ne)
                return fail("range " + std::to_string(s) + " of rank " + std::to_string(k) + " is [" +
                            std::to_string(r[k].begin) + ", " + std::to_string(r[k].end) + "), past the axis's " +
                            std::to_string(ne));
            if (input && contraction && (r[k].begin % kInputBlock != 0 || r[k].end % kInputBlock != 0))
                return fail("range " + std::to_string(s) + " of rank " + std::to_string(k) + " is [" +
                            std::to_string(r[k].begin) + ", " + std::to_string(r[k].end) +
                            "), which is not a whole number of " + std::to_string(kInputBlock) +
                            "-element activation blocks");
        }
        // the two ranks' ranges for this segment must not overlap
        if (a.range[s].begin < b.range[s].end && b.range[s].begin < a.range[s].end)
            return fail("range " + std::to_string(s) + " is claimed by both ranks");
    }
    // ... and together they must cover the axis exactly, once.  Only meaningful for the single-range splits;
    // the multi-range ones (the GDN `q|k|v`) are checked segment by segment above and by the total below.
    int64_t total = 0;
    for (int s = 0; s < a.ranges; ++s) total += a.range[s].count() + b.range[s].count();
    if (total != ne)
        return fail("the two ranks cover " + std::to_string(total) + " of the axis's " + std::to_string(ne));
    return true;
}

/// Validates that `specs` (one per rank) tile `ne` exactly once: every range inside the axis, no overlaps,
/// and the counts sum to `ne`.  `Input` ranges must begin and end on a `kInputBlock` boundary when the axis is a
/// contraction one.  A `Replicated`/`Rank0` set is accepted as-is.
inline bool shard_check_tile(const std::vector<ShardSpec>& specs, int64_t ne, const char* what, std::string& err,
                             bool contraction = true) {
    const std::string name = what == nullptr ? std::string("?") : std::string(what);
    auto fail = [&](const std::string& why) {
        err = name + ": " + why;
        return false;
    };
    if (specs.empty()) return fail("no ranks");
    const ShardAxis axis = specs[0].axis;
    for (const ShardSpec& s : specs)
        if (s.axis != axis) return fail("the ranks disagree about the axis");
    if (axis == ShardAxis::Replicated || axis == ShardAxis::Rank0) return true;
    const bool input = axis == ShardAxis::Input;
    int64_t total = 0;
    for (size_t r = 0; r < specs.size(); ++r) {
        const ShardSpec& s = specs[r];
        if (s.ranges < 1) return fail("rank " + std::to_string(r) + " has no ranges");
        for (int k = 0; k < s.ranges; ++k) {
            const ShardRange& rg = s.range[k];
            if (rg.count() <= 0) return fail("range " + std::to_string(k) + " of rank " + std::to_string(r) + " is empty");
            if (rg.begin < 0 || rg.end > ne)
                return fail("range " + std::to_string(k) + " of rank " + std::to_string(r) + " is [" +
                            std::to_string(rg.begin) + ", " + std::to_string(rg.end) + "), past the axis's " +
                            std::to_string(ne));
            if (input && contraction && (rg.begin % kInputBlock != 0 || rg.end % kInputBlock != 0))
                return fail("range " + std::to_string(k) + " of rank " + std::to_string(r) + " is [" +
                            std::to_string(rg.begin) + ", " + std::to_string(rg.end) +
                            "), which is not a whole number of " + std::to_string(kInputBlock) +
                            "-element activation blocks");
            total += rg.count();
        }
    }
    // pairwise overlap check, segment by segment
    for (size_t a = 0; a < specs.size(); ++a)
        for (size_t b = a + 1; b < specs.size(); ++b) {
            if (specs[a].ranges != specs[b].ranges) return fail("the ranks have different range counts");
            for (int k = 0; k < specs[a].ranges; ++k) {
                const ShardRange& x = specs[a].range[k];
                const ShardRange& y = specs[b].range[k];
                if (x.begin < y.end && y.begin < x.end)
                    return fail("range " + std::to_string(k) + " is claimed by ranks " + std::to_string(a) +
                                " and " + std::to_string(b));
            }
        }
    if (total != ne)
        return fail("the ranks cover " + std::to_string(total) + " of the axis's " + std::to_string(ne));
    return true;
}

/// Builds the tensor-parallel table of one layer for `ranks` ranks (2..4).  `rank` is this rank's index.
/// Builds EVERY rank's table internally and validates the tiling as it goes, so a table that does not tile its
/// axis cannot be produced at all - the check is not a separate step a caller can forget.
///
/// TP=4 AND THE KV HEADS.  `n_head_kv` is 2 on this model, so four ranks cannot each own a KV head.  When the
/// KV head count does not divide by `ranks`, the KV projection and cache are REPLICATED on every rank (each
/// rank computes all KV heads and keeps the full KV cache) while the QUERY heads are split.  That is correct:
/// attention is per query head against the shared KV, so a rank that owns 6 query heads can read the whole KV.
/// The alternative - inventing a fractional KV head - is the asymmetric-pair trap `kvmem` documents.
inline bool shard_table_n(const ModelGeometry& g, int rank, int ranks, ShardTable& out, std::string& err) {
    if (ranks < 1 || ranks > 4) {
        err = "tensor parallelism: " + std::to_string(ranks) + " ranks is not supported (1..4)";
        return false;
    }
    if (rank < 0 || rank >= ranks) {
        err = "shard table: rank " + std::to_string(rank) + " is outside 0.." + std::to_string(ranks - 1);
        return false;
    }
    auto need_div = [&](int64_t n, const char* what) {
        if (n >= ranks && n % ranks == 0) return true;
        err = std::string("tensor parallelism: ") + what + " is " + std::to_string(n) +
              ", which does not divide across " + std::to_string(ranks) + " ranks; each rank needs at least one";
        return false;
    };
    if (!need_div(g.n_head, "attention.head_count")) return false;
    const bool kv_split = g.n_head_kv >= ranks && g.n_head_kv % ranks == 0;
    if (!kv_split && ranks > g.n_head_kv) {
        // replicate the KV: correct, and the only option when there are fewer KV heads than ranks
    } else if (!kv_split) {
        err = "tensor parallelism: attention.head_count_kv is " + std::to_string(g.n_head_kv) +
              ", which does not divide across " + std::to_string(ranks) + " ranks";
        return false;
    }
    if (!need_div(g.ssm_v_heads, "ssm.time_step_rank")) return false;
    if (!need_div(g.ssm_k_heads, "ssm.group_count")) return false;
    if (g.has_indexer() && !need_div(g.idx_q_heads, "attention.indexer.head_count")) return false;

    std::vector<ShardTable> t((size_t) ranks);
    auto add = [&](const char* suffix, int64_t ne, std::vector<ShardSpec> specs, bool contraction = true) {
        if (!shard_check_tile(specs, ne, suffix, err, contraction)) return false;
        for (int r = 0; r < ranks; ++r)
            if (!t[(size_t) r].add(suffix, specs[(size_t) r], err)) return false;
        return true;
    };
    auto replicated = [] { return ShardSpec{}; };
    auto rank0_only = [] {
        ShardSpec s;
        s.axis = ShardAxis::Rank0;
        return s;
    };
    auto slice = [&](ShardAxis axis, int64_t ne, int r) {
        const int64_t h = ne / ranks;
        ShardSpec s;
        s.axis = axis;
        s.ranges = 1;
        s.range[0] = {h * r, h * (r + 1)};
        return s;
    };
    auto all = [&](auto fn) {
        std::vector<ShardSpec> v((size_t) ranks);
        for (int r = 0; r < ranks; ++r) v[(size_t) r] = fn(r);
        return v;
    };

    // ---- the gated residual: replicated (see the header for the arithmetic) ----
    for (const char* suf : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                            "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                            "hc_attn_norm.weight", "hc_ffn_norm.weight",
                            "attn_norm.weight", "post_attention_norm.weight"})
        if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;

    // ---- MoE ----
    // The router must agree on every rank or they run different models; the shared expert is too small to split
    // (640/ranks is not a multiple of 128 for ranks=4) and is ~6% of a layer's dense bytes.  The three EXPERT
    // tensors are replicated as TENSORS and partitioned as CACHES: each rank's expert cache holds whole experts
    // of its own, and the per-layer combine carries the partial sums.
    for (const char* suf : {"ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
                            "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
        if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;

    // ---- GDN: q|k|v in one tensor, so three ranges per rank, each a whole number of heads ----
    const int64_t q_dim = g.ssm_state_size * g.ssm_k_heads;   // 128 * 16 = 2048
    const int64_t k_dim = q_dim;
    const int64_t v_dim = g.ssm_value_dim;                    // 128 * 48 = 6144
    const int64_t conv = q_dim + k_dim + v_dim;               // 10240
    const int64_t bands = g.ssm_v_heads / g.ssm_k_heads;      // 3
    const int64_t per_k = g.ssm_k_heads / ranks;              // this rank's q/k heads
    const int64_t vhw = g.ssm_value_dim / g.ssm_v_heads;      // one v-head's width (128)
    auto vband = [&](ShardAxis axis, int r, int64_t unit) {
        ShardSpec s;
        s.axis = axis;
        s.ranges = (int) bands;
        for (int64_t b = 0; b < bands; ++b) {
            const int64_t lo = b * g.ssm_k_heads + (int64_t) r * per_k;
            s.range[b] = {lo * unit, (lo + per_k) * unit};
        }
        return s;
    };
    for (const char* suf : {"ssm_alpha.weight", "ssm_beta.weight"})
        if (!add(suf, g.ssm_v_heads, all([&](int r) { return vband(ShardAxis::Output, r, 1); }))) return false;
    for (const char* suf : {"ssm_a", "ssm_dt.bias"})
        if (!add(suf, g.ssm_v_heads, all([&](int r) { return vband(ShardAxis::Input, r, 1); }),
                 /*contraction=*/false)) return false;
    if (!add("ssm_norm.weight", 0, all([&](int) { return replicated(); }))) return false;
    auto qkv = [&](int r) {
        ShardSpec s;
        s.axis = ShardAxis::Output;
        s.ranges = 2 + (int) bands;   // q | k | v(bands)
        s.range[0] = {q_dim / ranks * r, q_dim / ranks * (r + 1)};
        s.range[1] = {q_dim + k_dim / ranks * r, q_dim + k_dim / ranks * (r + 1)};
        for (int64_t b = 0; b < bands; ++b) {
            const int64_t lo = b * g.ssm_k_heads + (int64_t) r * per_k;
            s.range[2 + b] = {q_dim + k_dim + lo * vhw, q_dim + k_dim + (lo + per_k) * vhw};
        }
        return s;
    };
    if (!add("attn_qkv.weight", conv, all([&](int r) { return qkv(r); }))) return false;
    if (!add("ssm_conv1d.weight", conv, all([&](int r) { return qkv(r); }))) return false;
    if (!add("attn_gate.weight", v_dim, all([&](int r) { return vband(ShardAxis::Output, r, vhw); }))) return false;
    // the value projection contracts over the v heads' slice, so the ranges are 128-aligned
    if (!add("ssm_out.weight", v_dim, all([&](int r) { return vband(ShardAxis::Input, r, vhw); }))) return false;

    // ---- QSA ----
    const int64_t q_out = 2 * g.n_head * g.head_dim;          // the per-head row is 2 * head_dim wide
    const int64_t kv_out = g.n_head_kv * g.head_dim;
    const int64_t o_in = g.n_head * g.head_dim;
    if (ranks == 4) {
        // THE QSA LAYERS ARE REPLICATED FOR TP=4.  The compiled QSA attention kernel requires
        // n_head == G * n_head_kv with G in {4, 8, 12, 16}; TP=4's 6 query heads over 2 replicated KV heads
        // give G=3, which has no kernel.  Replicating the QSA projections keeps the model's G=12 on every rank
        // and costs 4x the QSA dense compute - 12 of 48 layers, and the decode is expert-bound.
        for (const char* suf : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"})
            if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;
    } else {
        if (!add("attn_q.weight", q_out, all([&](int r) { return slice(ShardAxis::Output, q_out, r); }))) return false;
        for (const char* suf : {"attn_k.weight", "attn_v.weight"}) {
            if (kv_split) {
                if (!add(suf, kv_out, all([&](int r) { return slice(ShardAxis::Output, kv_out, r); }))) return false;
            } else {
                // REPLICATED: every rank computes all KV heads and keeps the full KV cache.
                if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;
            }
        }
        if (!add("attn_output.weight", o_in, all([&](int r) { return slice(ShardAxis::Input, o_in, r); }))) return false;
    }
    // the per-head norms are over head_dim, which every rank needs whole for the heads it owns
    for (const char* suf : {"attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight",
                            "indexer.k_norm.weight"})
        if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;
    if (g.has_indexer()) {
        // THE WHOLE INDEXER IS REPLICATED, projection and selection both.  It picks which cells the attention
        // reads, from a score over ALL of its query heads against the shared keys: cut those queries and each
        // rank scores a PARTIAL sum, so the ranks select DIFFERENT cells and the attention reads different
        // history - a deterministic wrong answer, not a tolerance.
        if (!add("indexer.q_proj.weight", 0, all([&](int) { return replicated(); }))) return false;
        if (!add("indexer.k_proj.weight", 0, all([&](int) { return replicated(); }))) return false;
    }

    // ---- the PLE block, which only layer 1 carries ----
    for (const char* suf : {"ple_key.weight", "ple_value.weight", "ple_conv1d.weight", "ple_norm_query.weight",
                            "ple_norm_key.weight", "ple_norm_conv.weight"})
        if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;

    // ---- the head, the embedding and the draft layer stay on rank 0 ----
    for (const char* suf : {"token_embd.weight", "output.weight", "output_norm.weight"})
        if (!add(suf, 0, all([&](int) { return rank0_only(); }))) return false;
    for (const char* suf : {"output_hc_down.weight", "output_hc_up.weight", "output_hc_norm.weight"})
        if (!add(suf, 0, all([&](int) { return replicated(); }))) return false;
    for (const char* suf : {"nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
                            "nextn.shared_head_norm.weight"})
        if (!add(suf, 0, all([&](int) { return rank0_only(); }))) return false;

    out = t[(size_t) rank];
    return true;
}

/// The TP=2 entry point kept for the existing two-rank path: it is `shard_table_n` with two ranks.
inline bool shard_table(const ModelGeometry& g, int rank, ShardTable& out, std::string& err) {
    return shard_table_n(g, rank, 2, out, err);
}

}  // namespace strata::core
