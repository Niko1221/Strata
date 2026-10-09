// include/strata/core/glm_experts.hpp - glm5-next's routed experts, on the CPU, one token at a time.
//
// **WHY THIS IS NOT `expert_pool_dispatch`.**  The first family's single-token pool is built around two things
// that are properties of ITS pack and not of the job: a compile-time 2560-wide activation (`H`), and the
// canonical Q2_0 blob layout, whose internal offsets are constants.  glm5-next is 4096 wide and its experts
// keep their GGUF formats - IQ3_S gate/up over IQ4_XS down on the pack this was written against - so both of
// those refusals fire, correctly, and there is nothing to relax: the arithmetic underneath is different.
//
// What it DOES reuse is the part that is geometry-parameterized already: `native_fmt` +
// `native_gu_rows`/`native_down_rows` (ggml-cpu's own vec_dot, so a routed expert computes what llama.cpp's CPU
// backend computes for it) and `ExpertPool::run_split_multi_native`, which spreads the rows of `k` experts
// across every worker and reads each expert's bytes once for all of them.
//
// THE THREE THINGS THAT ARE EASY TO GET WRONG, all of them silent:
//
//   * **`ExpertSource::blob` POINTERS DIE ON THE NEXT CALL.**  Its contract says the pointer only has to stay
//     valid until the next `blob()`, and the whole point of a pool is to hold `k` of them at once.  This copies
//     each expert into a slot of its own with `copy_blob` - the explicit "I am keeping this" entry point -
//     rather than relying on `FileExpertSource`'s staging pool happening to keep them distinct.
//   * **THE ACTIVATION IS QUANTIZED ONCE, NOT ONCE PER EXPERT.**  It depends on the layer's gate/up `vec_dot`
//     type, so every expert of the layer shares it; `ExpertJobMulti` carries a POINTER per (expert, token) for
//     exactly that reason.
//   * **`out` IS UNWEIGHTED.**  `moe_combine_parts` multiplies by the router weights on the device, which is
//     where `weights` is already sitting.  Applying them here as well squares them, which is finite, plausible
//     and wrong by a factor of `w`.
#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/native_expert.hpp"   // NativeFmt, for the private `grow_to`
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

/// The shape `session_token` calls: `x` is the layer's normed FFN input (HOST, `nt * n_embd` floats, token `t`
/// at `x + t * n_embd`), `ids` the routed experts (HOST, `nt * k` int32, token `t`'s slot `i` at
/// `ids[t * k + i]`), and `out` receives `nt * k` unweighted expert outputs (HOST, same layout, row `i` of
/// token `t` belonging to `ids[t * k + i]`).  `nt == 1` is the decode step and the only shape this had before
/// chunked prefill; `nt > 1` is a prefill chunk.  Returns false with a reason rather than latching a flag -
/// unlike the first family's `PoolFn`, whose caller is a stream-ordered loop that has nowhere to put an error,
/// this one is called from straight-line code that can stop the token.
using GlmPoolFn = bool (*)(void* user, int64_t layer, const float* x, const int32_t* ids, int64_t nt, int64_t k,
                           float* out, std::string& err);

/// `GlmPoolFn` bound to a `GlmExpertPool`.
bool glm_expert_pool_call(void* user, int64_t layer, const float* x, const int32_t* ids, int64_t nt, int64_t k,
                          float* out, std::string& err);

class GlmExpertPool {
public:
    GlmExpertPool() = default;
    ~GlmExpertPool() = default;
    GlmExpertPool(const GlmExpertPool&) = delete;
    GlmExpertPool& operator=(const GlmExpertPool&) = delete;

    /// `k` is the routing width the session was built with.  It sizes one staging slot per routed expert, so it
    /// has to be the real one - a pool sized for 8 that is handed 10 writes 2 slots past the end of the vector.
    bool init(ExpertSource* src, strata::kernels::cpu::ExpertPool* pool, int64_t n_layers, int64_t n_expert,
              int64_t k, std::string& err);

    /// One layer's routed experts for `nt` tokens into `out` (`nt * k * n_embd` floats, unweighted).  `x` and
    /// `ids` are HOST, laid out as `GlmPoolFn` documents.  `nt == 1` is `run`'s only pre-chunk shape.
    bool run(int64_t layer, const float* x, const int32_t* ids, int64_t nt, int64_t k, float* out, std::string& err);

    /// The largest chunk `run` will take.  Each token costs one activation image and `k` slots of `jobs_`, so
    /// this bounds both - and a chunk larger than the geometry can route is refused rather than truncated.
    ///
    /// Every byte of that is DYNAMIC (`grow_to` sizes `jobs_`/`next_`/`act_` from the request), so this number
    /// is a policy, not a buffer size: raising it costs nothing until someone actually asks for the chunk.
    /// 4096 is the ceiling now rather than 1024 because the sweep below had to reach past 1024 to find out
    /// where the curve turns - and it turns EARLIER than the byte count suggests, because `MAXT` is 8.  An
    /// expert that the chunk routes to more than 8 times becomes more than one job, and `run`'s whole win is
    /// decoding an expert's rows once per job; past ~288 tokens (288 experts / 8 a token = 36 tokens an
    /// expert) every expert is over `MAXT` and the job count grows linearly with the chunk while the byte
    /// saving it buys is already flat.  See the note on `run`'s chunk branch.
    static constexpr int64_t kMaxChunk = 4096;

    int64_t k() const { return k_; }

    /// `--expert-profile-save`: start counting which (layer, expert) pairs get routed.  Off by default, and off
    /// costs one empty-vector test a layer.  The first family's counts live in `ExpertDispatch::usage`, which the
    /// verify window's dispatch fills in - a path glm5-next never takes, so without this its profile would be
    /// empty and `rank_learned_profile` would have nothing to rank.
    void count_routing(bool on);
    /// How often each (layer, expert) pair has been routed: `n_layers * n_expert` floats indexed
    /// `layer * n_expert + expert`, in the same units as the first family's `usage`.  EMPTY unless
    /// `count_routing(true)` was called, and empty is the one thing every caller tests.
    const std::vector<float>& routing() const { return routing_; }

private:
    /// Size `jobs_`/`head_`/`next_`/`act_` for a chunk of `nt` tokens of a layer whose fmt is `f`.  A no-op
    /// once the largest chunk seen covers `nt`, which is every call after the first of a given size.
    bool grow_to(const strata::kernels::cpu::NativeFmt& f, int64_t nt);

    /// **THE POOL TAKES ONE CALLER, AND A PIPELINE HAS SEVERAL.**  `run` is a host-thread protocol over
    /// `ExpertPool`'s own members - the batch, the job array, the per-job scratch, the epoch - and none of it
    /// is re-entrant.  It never had to be: every caller was the token loop, one layer at a time.  The chunk
    /// pipeline runs a stage a thread, so two stages can reach a MoE layer at the same instant and both would
    /// be writing `mjobs_`.  This makes them wait instead, which costs the pipeline nothing it was going to get
    /// anyway - the pool is one shared object with one set of cores, so it is the floor under the whole thing
    /// and the pipeline's win is exactly the GPU time it hides behind it.  Taken for the DURATION of a call,
    /// not per layer, so the failure paths (`err` set, `return false`) release it too.
    std::mutex serial_;

    ExpertSource* src_ = nullptr;
    strata::kernels::cpu::ExpertPool* pool_ = nullptr;
    int64_t n_layers_ = 0, n_expert_ = 0, k_ = 0;
    /// One assembled blob per routed expert of the current layer, so all `k` are alive at once (see the note on
    /// `blob()` above).  Sized in `init`, reused every layer - the bytes are the same size every time only
    /// because a native pack's per-layer blob size is fixed; a layer whose blob is larger re-sizes the slot once
    /// and never again.
    std::vector<std::vector<uint8_t>> slot_;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_;
    /// The layer's quantized gate/up activations, one image per token of the chunk.  At `nt == 1` this is the
    /// single image all `k` experts share; at `nt > 1` it is `nt` of them back to back, because two tokens of a
    /// chunk have different activations and `ExpertJobMulti::nact` carries a pointer per (expert, token).
    std::vector<uint8_t> act_;
    /// The grouping scratch, sized once for the largest chunk seen so it never allocates in the token path.
    /// `head_[e]` is the first `(t, i)` pair routed to expert `e` (or -1) and `next_[p]` chains the rest, so the
    /// grouping is a counting sort over the expert ids and not a comparison sort of `nt * k` pairs a layer.
    std::vector<int32_t> head_, next_;
    int64_t job_cap_ = 0;   ///< the `jobs_` size the current `head_`/`next_`/`act_` were sized for
    uint64_t bytes_ = 0;
    int64_t calls_ = 0;
    /// See `count_routing`.  Empty means nothing is counted; it is sized to the whole (layer, expert) table the
    /// moment counting is asked for, so `run` never has to grow it and never tests anything but emptiness.
    std::vector<float> routing_;
};

}  // namespace strata::core
