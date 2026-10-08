// src/core/glm_experts.cpp - see the header.
#include "strata/core/glm_experts.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::core {

using strata::kernels::cpu::ExpertJobMulti;
using strata::kernels::cpu::MAXT;
using strata::kernels::cpu::NativeFmt;

bool GlmExpertPool::init(ExpertSource* src, strata::kernels::cpu::ExpertPool* pool, int64_t n_layers,
                         int64_t n_expert, int64_t k, std::string& err) {
    if (src == nullptr || pool == nullptr) { err = "glm5-next expert pool: no source or no worker pool"; return false; }
    if (k < 1 || k > strata::kernels::cpu::ExpertPool::kMaxSplitMulti) {
        err = "glm5-next expert pool: the routing width is " + std::to_string(k) + ", outside 1.." +
              std::to_string(strata::kernels::cpu::ExpertPool::kMaxSplitMulti);
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) {
        // A canonical Q2_0 pack's experts are in Strata's own form, which this pool has no reader for - and a
        // glm5-next model would never be packed that way.  Refusing by name beats reading Q2_0 bytes as IQ.
        err = "glm5-next expert pool: this pack is not native (its experts are in the Q2_0 pack form)";
        return false;
    }
    src_ = src;
    pool_ = pool;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    k_ = k;
    slot_.assign((size_t) k, std::vector<uint8_t>());
    jobs_.assign((size_t) k, ExpertJobMulti());
    head_.assign((size_t) n_expert, -1);
    return true;
}

bool GlmExpertPool::grow_to(const NativeFmt& f, int64_t nt) {
    if (nt <= job_cap_) return true;
    // ONE JOB PER (expert, token) AT WORST, and a job holds at most `MAXT` tokens of one expert, so `nt * k` is
    // a bound on both the pair count and the job count.  This runs on the FIRST chunk of a new size and never
    // again: a token path that allocated would fail P2.T10, and `job_cap_` is what makes that true rather than
    // hopeful - `run` grows here and every later call of the same size is a no-op.
    const size_t np = (size_t) nt * (size_t) k_;
    jobs_.resize(np + 1);
    head_.resize((size_t) n_expert_);
    next_.resize(np);
    act_.resize((size_t) nt * (size_t) f.act_bytes);
    job_cap_ = nt;
    return true;
}

void GlmExpertPool::count_routing(const bool on) {
    if (on) {
        if (routing_.size() != (size_t) (n_layers_ * n_expert_))
            routing_.assign((size_t) (n_layers_ * n_expert_), 0.0f);
    } else {
        routing_.clear();
    }
}

bool GlmExpertPool::run(int64_t layer, const float* x, const int32_t* ids, int64_t nt, int64_t k, float* out,
                        std::string& err) {
    if (src_ == nullptr) { err = "glm5-next expert pool was never initialised"; return false; }
    // k < k_ is the VRAM tier's misses (`glm_gpu_experts`): the pool computes only the experts the card does
    // not hold.  More than the pool was sized for would write past its `slot_`/`jobs_`, which is why the upper
    // end is still refused.
    if (k < 1 || k > k_) {
        err = "glm5-next expert pool: this layer routes " + std::to_string(k) + " experts but the pool was sized for " +
              std::to_string(k_);
        return false;
    }
    if (layer < 0 || layer >= n_layers_) { err = "glm5-next expert pool: layer out of range"; return false; }
    if (nt < 1 || nt > kMaxChunk) {
        err = "glm5-next expert pool: a chunk of " + std::to_string(nt) + " tokens is outside 1.." +
              std::to_string(kMaxChunk);
        return false;
    }

    const auto& lay = strata::kernels::cpu::expert_layout();
    const NativeFmt& f = lay.fmt[(size_t) layer];
    if (f.bytes == 0) {
        // The dense-lead blocks carry no routed experts at all.  Reaching here means the layer dispatch sent a
        // dense FFN to the pool, which is a bug above this function - and a zero-size blob would otherwise be
        // projected against as if it were one.
        err = "glm5-next expert pool: layer " + std::to_string(layer) + " has no routed experts (dense-lead)";
        return false;
    }

    // `begin_layer` ages the source's assembled-blob buffers.  It is not optional here for the same reason it is
    // not optional in the first family's dispatch: without it the epoch never advances, every blob is "fresh",
    // and the staging pool has no buffer it is allowed to reclaim.
    src_->begin_layer(layer, ids, k);

    for (int64_t p = 0; p < nt * k; ++p) {
        if (ids[p] < 0 || ids[p] >= n_expert_) {
            err = "glm5-next expert pool: a routed expert id is out of range";
            return false;
        }
    }

    // `--expert-profile-save`: the routing counted here.  This is the ONE place a glm5-next (layer, expert) pair
    // is known on the host - `session_token` hands these ids over to be computed, and the first family's own
    // counter (`ExpertDispatch::usage`, filled by `expert_pool_dispatch_multi`) is on a verify window this arch
    // does not have.  The ids were range-checked immediately above, so the index needs no second test.
    if (!routing_.empty()) {
        for (int64_t p = 0; p < nt * k; ++p) routing_[(size_t) (layer * n_expert_ + ids[p])] += 1.0f;
    }

    if (!grow_to(f, nt)) return false;

    // ================================ A PREFILL CHUNK: `nt` TOKENS AT ONCE ================================
    //
    // **WHAT A CHUNK BUYS IS THE EXPERT'S BYTES READ ONCE FOR EVERY TOKEN THAT ROUTED TO IT.**  One token reads
    // its 8 experts a layer: 3.82 GiB over the trunk, and the pool is bound by that read (measured 101.9 ms of
    // a 229.5 ms token on four cards, against 3.82 GiB of host memory at ~37 GB/s).  A chunk of C tokens reads
    // only the DISTINCT experts the chunk routed to - measured off a 516-token router trace: 5.6x fewer bytes a
    // token at C=128.  The FLOPs do not change; every token still dots against its own experts.
    //
    // The grouping is what makes that true, and it is a counting sort, not a sort of the pairs: `head_[e]` holds
    // the first pair routed to expert `e` and `next_` chains the rest, so `nt * k` pairs are bucketed in one
    // pass over the ids.  Each bucket is then cut into jobs of at most `MAXT` tokens - `ExpertJobMulti` carries
    // one activation and one output row per token, and `MAXT` is where that array ends.  A hot expert (one the
    // chunk routed to more than `MAXT` times) becomes several jobs over the SAME slices, and
    // `run_split_multi_native` batches whatever comes out at its own `kMaxSplitMulti`.
    //
    // **THE SLICES DO NOT COST THE GROUP ITS KERNEL - AND THEY MUST NOT.**  `native_rows_sliceable` answers "no"
    // at `nt > 1` for this pack's IQ3_S gate/up, which reads as "assemble a blob or take the per-token dot".
    // Neither is necessary: the multi-token kernel addresses the up row as `blob + up_off + r * gu_row`, so two
    // slices out of one mapping ARE a blob with `up_off = up - gate`, and `native_gu_rows_slice` hands them over
    // as one.  That matters because the per-token dot re-decodes every row for EVERY token of the group, and the
    // decode - not the read - is what the pool spends its time on: measured here, `--prefill 128` read 4.5x
    // fewer expert bytes than `--prefill 1` (1245.3 -> 276.6 GiB) and the pool took the same 30.4 s until the
    // group reached the multi-token kernel, which took it to 12.9 s.  The ~3e-8 the kernel rounds differently
    // from ggml's dot is the price; see the note on `native_gu_rows_slice`, and `STRATA_NO_SLICE_MT`.
    if (nt > 1) {
        const int64_t np = nt * k;
        std::fill(head_.begin(), head_.end(), -1);
        for (int64_t p = 0; p < np; ++p) {
            const int32_t e = ids[p];
            next_[(size_t) p] = head_[(size_t) e];
            head_[(size_t) e] = (int32_t) p;
        }

        // One activation image per token, produced once for the whole layer: it is the gate/up `vec_dot`
        // type's own quantization of that token's `x`, and it is the same bytes for every expert that reads it.
        for (int64_t t = 0; t < nt; ++t)
            strata::kernels::cpu::native_quant_act(f, x + (size_t) t * (size_t) f.n_embd,
                                                   act_.data() + (size_t) t * (size_t) f.act_bytes);

        ExpertJobMulti* jobs = jobs_.data();
        int32_t n_jobs = 0;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            int32_t p = head_[(size_t) e];
            if (p < 0) continue;
            // THE EXPERT'S THREE RUNS ARE RESOLVED ONCE, then shared by every job of this expert.  Those
            // pointers stay valid for the process (`ExpertSource::slices`), so a second job over the same
            // expert is free rather than a second lookup.
            const uint8_t* gate = nullptr;
            const uint8_t* up = nullptr;
            const uint8_t* down = nullptr;
            if (!src_->slices(layer, e, &gate, &up, &down)) {
                err = "glm5-next expert pool: layer " + std::to_string(layer) + " expert " + std::to_string(e) +
                      " cannot be read as slices, and a chunk needs them: the assembled-blob form would round "
                      "its rows differently from a single token";
                return false;
            }
            while (p >= 0) {
                if (n_jobs >= (int32_t) jobs_.size()) {
                    err = "glm5-next expert pool: the chunk's job list overflowed";
                    return false;
                }
                ExpertJobMulti& j = jobs[n_jobs++];
                j.blob = nullptr;
                j.gate = gate;
                j.up = up;
                j.down = down;
                int n = 0;
                for (; p >= 0 && n < MAXT; p = next_[(size_t) p], ++n) {
                    const int64_t t = p / k, i = p % k;
                    j.act[n] = nullptr;   // only the Q2_0-native arm reads this, and it is not this pack
                    j.nact[n] = act_.data() + (size_t) t * (size_t) f.act_bytes;
                    j.out[n] = out + (size_t) (t * k + i) * (size_t) f.n_embd;
                }
                j.nt = n;
                // The tail is cleared rather than left as the previous layer's: `nt` is what the kernels read,
                // but a stale pointer in an array a future kernel scans without checking `nt` is a trap.
                for (int u = n; u < MAXT; ++u) { j.act[u] = nullptr; j.nact[u] = nullptr; j.out[u] = nullptr; }
            }
            bytes_ += (uint64_t) f.bytes;   // the same bytes read once for every token of the chunk that wanted it
        }
        pool_->run_split_multi_native(f, jobs, n_jobs);
        ++calls_;
        return true;
    }

    // ================================ ONE TOKEN: THE DECODE STEP, UNCHANGED ================================
    //
    // Kept as its own body rather than folded into the chunk above, because it is the shape every measured
    // number in this engine was taken at and the blob fallback below is only reachable from here: with `nt == 1`
    // the blob form reaches the SAME ggml per-token dot as the slice form, so the choice is free of rounding
    // and can be made on availability.

    // ---- THE k EXPERTS OF THIS LAYER, READ WHERE THEY LIE.
    //
    // A GGUF-in-place pack holds gate, up and down as three tensors, and within each one expert's rows are one
    // contiguous run (`ExpertSource::slices`); the kernels read exactly those three runs.  So hand them over as
    // they are.  Building the assembled blob they were originally written for costs a memcpy of the whole
    // expert, 8 times a layer over 42 layers - 3.9 GB per token, on THIS thread, before a worker is asked for
    // anything.  Measured on the 2x Xeon E5-2680 v4 rig: 0.82 tok/s copied against llama.cpp's 3.49 on the same
    // file, and 0.56 tok/s with the pool cut to 2 workers, which is what said the copy and not the pool was the
    // cost.  See `native_gu_rows_ptrs`.
    //
    // Slices are only usable while the blob path would stay on ggml-cpu's per-token dot: the multi-token
    // kernels address the assembled offsets.  This pool decodes one token at a time, so that is a property of
    // the two types and the CPU rather than of the call, and it is the same kernel and the same bits.
    ExpertJobMulti* jobs = jobs_.data();
    bool sliced = strata::kernels::cpu::native_rows_sliceable(f.gu_type, f.d_type, 1);
    for (int64_t i = 0; i < k && sliced; ++i) {
        ExpertJobMulti& j = jobs[i];
        j.blob = nullptr;
        sliced = src_->slices(layer, ids[i], &j.gate, &j.up, &j.down);
    }
    if (!sliced) {
        // The assembled form, all k alive at once: `blob()` would hand back a pointer the next call is allowed
        // to invalidate, and this needs eight of them to survive until the pool has read them.
        for (int64_t i = 0; i < k; ++i) {
            ExpertJobMulti& j = jobs[i];
            j.gate = j.up = j.down = nullptr;
            std::vector<uint8_t>& s = slot_[(size_t) i];
            if (s.size() < f.bytes) s.resize((size_t) f.bytes);
            if (!src_->copy_blob(layer, ids[i], s.data())) {
                err = "glm5-next expert pool: the expert source could not produce the blob of layer " +
                      std::to_string(layer) + " expert " + std::to_string(ids[i]);
                return false;
            }
            j.blob = s.data();
        }
    }

    // ---- ONE activation for the layer.  It is the gate/up `vec_dot` type's own quantization of `x`, so every
    // expert of this layer reads the same bytes; producing it per expert would be the same work eight times.
    if (act_.size() < f.act_bytes) act_.resize(f.act_bytes);
    strata::kernels::cpu::native_quant_act(f, x, act_.data());

    for (int64_t i = 0; i < k; ++i) {
        ExpertJobMulti& j = jobs[i];
        j.nt = 1;
        j.act[0] = nullptr;                  // only the Q2_0-native arm reads this, and it is not this pack
        j.nact[0] = act_.data();
        j.out[0] = out + (size_t) i * (size_t) f.n_embd;
        // The same bytes are read either way - only the copy is gone - so the report still counts them.
        bytes_ += (uint64_t) f.bytes;
    }
    // The row split: every worker takes a share of the `k * n_ff` gate/up rows, then of the `k * n_embd` down
    // rows, so one expert is not one core.  `out` comes back UNWEIGHTED - the device's `moe_combine` applies the
    // router weights, and the first family's pool has the same rule for the same reason.
    pool_->run_split_multi_native(f, jobs, (int) k);
    ++calls_;
    return true;
}

bool glm_expert_pool_call(void* user, int64_t layer, const float* x, const int32_t* ids, int64_t nt, int64_t k,
                          float* out, std::string& err) {
    if (user == nullptr) { err = "glm5-next expert pool: the callback has no pool"; return false; }
    return ((GlmExpertPool*) user)->run(layer, x, ids, nt, k, out, err);
}

}  // namespace strata::core
