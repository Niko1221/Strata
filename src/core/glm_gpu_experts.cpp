// src/core/glm_gpu_experts.cpp - see include/strata/core/glm_gpu_experts.hpp.
#include "strata/core/glm_gpu_experts.hpp"

#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata::core {

namespace {
// A q8_1 block is 32 int8 plus a half `d` and a half `s`.
constexpr size_t kQ8_1Block = 36;
double since(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}
}  // namespace

GlmGpuExperts::~GlmGpuExperts() {
    // The frees are device-specific and the tier may outlive the caller's device choice, so they run under the
    // device the buffers were allocated on.
    if (dev_ < 0) return;
    const OnDevice on(dev_);
    if (xq_) cudaFree(xq_);
    if (scratch_) cudaFree(scratch_);
    if (d_ptr_) cudaFree(d_ptr_);
    if (d_idx_) cudaFree(d_idx_);
    if (h_ptr_) cudaFreeHost(h_ptr_);
    if (h_idx_) cudaFreeHost(h_idx_);
    if (cxq_) cudaFree(cxq_);
    if (cscratch_) cudaFree(cscratch_);
    if (c_ptr_) cudaFree(c_ptr_);
    if (c_idx_) cudaFree(c_idx_);
    if (hc_ptr_) cudaFreeHost(hc_ptr_);
    if (hc_idx_) cudaFreeHost(hc_idx_);
    for (uint8_t* p : stage_) cudaFreeHost(p);
    for (uint8_t* p : stage_chunk_) cudaFreeHost(p);
    for (void* e : stage_ev_)
        if (e != nullptr) cudaEventDestroy((cudaEvent_t) e);
}

bool GlmGpuExperts::init(ExpertSource* src, const std::vector<int>& gu_type, const std::vector<int>& d_type,
                         const std::vector<uint64_t>& blob_bytes, int64_t layer_lo, int64_t layer_hi,
                         int64_t n_expert, int64_t k, int64_t n_embd, int64_t n_ff, int64_t budget_bytes,
                         int64_t chunk_tokens, std::string& err) {
    if (src == nullptr) { err = "glm gpu experts: no expert source"; return false; }
    if (k < 1 || n_expert < 1 || n_embd < 1 || n_ff < 1) {
        err = "glm gpu experts: bad geometry";
        return false;
    }
    if (cudaGetDevice(&dev_) != cudaSuccess) { err = "glm gpu experts: no current device"; return false; }
    src_ = src;
    n_layers_ = (int64_t) blob_bytes.size();
    n_expert_ = n_expert;
    k_ = k;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    if (layer_lo < 0) layer_lo = 0;
    if (layer_hi < 0 || layer_hi > n_layers_) layer_hi = n_layers_;
    lay_.assign((size_t) n_layers_, {});
    lo_.assign((size_t) n_layers_, 0);
    hi_.assign((size_t) n_layers_, 0);
    // THE STAGE'S OWN LAYERS ONLY (see the header): both the blob bytes summed into the quota and the slots
    // handed out are this range's, so a card keeps four times as many slots a layer as a whole-model sizing
    // would give it - and pays for none of the layers another card computes.
    int64_t moe_layers = 0, moe_bytes = 0;
    for (int64_t l = layer_lo; l < layer_hi; ++l) {
        if (blob_bytes[(size_t) l] == 0) continue;      // a dense lead layer, or one the pack has no row for
        if (!strata::kernels::native_expert_supported(gu_type[(size_t) l], d_type[(size_t) l], n_embd, n_ff)) {
            err = "glm gpu experts: layer " + std::to_string(l) + " has no grouped kernel for its formats";
            return false;
        }
        lay_[(size_t) l] = strata::kernels::native_expert_layout(gu_type[(size_t) l], d_type[(size_t) l], n_embd, n_ff);
        if (lay_[(size_t) l].bytes != blob_bytes[(size_t) l]) {
            err = "glm gpu experts: layer " + std::to_string(l) + ": the pack's blob is " +
                  std::to_string(blob_bytes[(size_t) l]) + " B, the kernels' layout " +
                  std::to_string(lay_[(size_t) l].bytes) + " B";
            return false;
        }
        ++moe_layers;
        moe_bytes += (int64_t) blob_bytes[(size_t) l];
    }
    if (moe_layers == 0) { err = "glm gpu experts: no MoE layer in this stage's range"; return false; }

    // ---- THE CHUNK PATH'S OWN BILL, PAID BEFORE THE SLOTS ARE DIVIDED.  `run_chunk` needs a q8_1 image per
    // token, a plan and a grouped-kernel scratch sized for a wave's entries; all of it is allocated once here
    // and all of it comes out of the same budget the slots do, so a tier asked to serve prefill holds a few
    // experts a layer fewer than the same tier asked only for decode.  That is the honest trade and it is why
    // this is a startup decision rather than a lazy allocation.
    //
    // `cap_entry_` is the entries one wave may hold.  A wave is a run of consecutive experts whose entries all
    // fit, so this only decides HOW MANY waves a layer's experts take - never whether the chunk can be served
    // at all: a single expert is routed by at most `T` tokens (`top-k` picks distinct experts), so any cap of
    // `T` or more can always take the next group and the loop below cannot stall.  `min(T*k)` is the whole
    // chunk in one wave, which for T=2048, k=8 is 16,384 entries and 405 MiB of scratch - too much to take
    // from a cache, so it is capped and then shrunk further if the card cannot afford even that.
    if (chunk_tokens > 0) {
        chunk_tokens_ = chunk_tokens;
        cap_entry_ = std::min<int64_t>(chunk_tokens_ * k, 8192);
        const size_t xq_need = (size_t) chunk_tokens_ * ((size_t) (n_embd / 32) * kQ8_1Block);
        // A tenth of the budget for the whole chunk path, and at least one group's worth of scratch so the
        // stream always makes progress.
        const int64_t chunk_allow = std::max<int64_t>(1, budget_bytes / 10);
        while (cap_entry_ > chunk_tokens_ &&
               (int64_t) strata::kernels::native_expert_scratch_bytes(cap_entry_, n_ff) + (int64_t) xq_need >
                   chunk_allow)
            cap_entry_ /= 2;
        const size_t sc = strata::kernels::native_expert_scratch_bytes(cap_entry_, n_ff);
        const int64_t spent = (int64_t) (sc + xq_need);
        if (spent >= budget_bytes) {
            err = "glm gpu experts: a chunk of " + std::to_string(chunk_tokens_) + " tokens needs " +
                  std::to_string(spent) + " B of activations and scratch, and only " + std::to_string(budget_bytes) +
                  " B of the card is free (raise --glm-gpu-experts, lower STRATA_GLM_GPU_RESERVE_MIB, or lower "
                  "--prefill)";
            return false;
        }
        budget_bytes -= spent;
        if (cudaMalloc(&cxq_, xq_need) != cudaSuccess || cudaMalloc(&cscratch_, sc) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk path's buffers: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        chunk_calls_ = 0;
    }
    // An even quota a MoE layer.  The first family's R4.2g lesson is why it is even and not arrival-ordered:
    // a position looks at k experts in every layer, so a shared counter fills the whole cache inside the first
    // few layers it ever sees and never changes again.
    const int64_t per_layer = std::min<int64_t>(n_expert, budget_bytes / moe_bytes);
    if (per_layer < 1) {
        err = "glm gpu experts: " + std::to_string(budget_bytes) + " B of budget does not hold one expert of " +
              std::to_string(moe_bytes / std::max<int64_t>(1, moe_layers)) + " B a layer";
        return false;
    }
    std::vector<int64_t> sizes;
    for (int64_t l = layer_lo; l < layer_hi; ++l) {
        lo_[(size_t) l] = (int64_t) sizes.size();
        if (blob_bytes[(size_t) l] != 0) sizes.insert(sizes.end(), (size_t) per_layer, (int64_t) blob_bytes[(size_t) l]);
        hi_[(size_t) l] = (int64_t) sizes.size();
    }
    next_ = lo_;
    if (!cache_.open_sized(sizes, n_layers_, n_expert, err)) return false;
    slot_.assign((size_t) (n_layers_ * n_expert), kNotResident);
    owner_.assign((size_t) cache_.full_slots(), -1);
    count_.assign((size_t) (n_layers_ * n_expert), 0);
    moe_layers_ = moe_layers;
    per_layer_ = per_layer;

    const size_t xq_bytes = (size_t) (n_embd / 32) * kQ8_1Block;
    const size_t scratch = strata::kernels::native_expert_scratch_bytes(k, n_ff);
    if (cudaMalloc(&xq_, xq_bytes) != cudaSuccess || cudaMalloc(&scratch_, scratch) != cudaSuccess ||
        cudaMalloc((void**) &d_ptr_, (size_t) k * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc((void**) &d_idx_, (size_t) (3 * k + 2) * sizeof(int32_t)) != cudaSuccess ||
        cudaMallocHost((void**) &h_ptr_, (size_t) k * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMallocHost((void**) &h_idx_, (size_t) (3 * k + 2) * sizeof(int32_t)) != cudaSuccess) {
        err = std::string("glm gpu experts: buffers: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // The chunk plan's device/host pair, sized on the WAVE and not on the chunk: a wave never holds more than
    // `per_layer_` groups (one slot each) or `cap_entry_` entries, and every wave reuses these.
    if (chunk_tokens_ > 0) {
        cap_group_ = per_layer_;
        const size_t idx = (size_t) (cap_group_ + 2 + 2 * cap_entry_);
        if (cudaMalloc((void**) &c_ptr_, (size_t) cap_group_ * sizeof(unsigned long long)) != cudaSuccess ||
            cudaMalloc((void**) &c_idx_, idx * sizeof(int32_t)) != cudaSuccess ||
            cudaMallocHost((void**) &hc_ptr_, (size_t) cap_group_ * sizeof(unsigned long long)) != cudaSuccess ||
            cudaMallocHost((void**) &hc_idx_, idx * sizeof(int32_t)) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk plan: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        plan_off_.resize((size_t) n_expert + 1);
        plan_dst_.resize((size_t) (chunk_tokens_ * k));
        plan_tok_.resize((size_t) (chunk_tokens_ * k));
        plan_e_.reserve((size_t) n_expert);
    }
    uint64_t max_blob = 0;
    for (int64_t l = layer_lo; l < layer_hi; ++l) max_blob = std::max(max_blob, blob_bytes[(size_t) l]);
    // The chunk's staging ring, deep enough that the host's assembly of wave n+1 runs while the card computes
    // wave n.  `nslots` bounds a wave, so nothing deeper is ever useful; 24 is what fits comfortably in pinned
    // memory at GLM's 11.13 MiB a blob (267 MiB a stage) and is deep enough that an event is normally already
    // satisfied.  If the box will not give that, take less rather than refusing the tier - a shallower ring
    // stalls, it does not corrupt.
    if (chunk_tokens_ > 0) {
        int64_t depth = 0;
        for (int64_t l = layer_lo; l < layer_hi; ++l) depth = std::max(depth, hi_[(size_t) l] - lo_[(size_t) l]);
        depth = std::min<int64_t>(depth, 64);
        // How many threads assemble a wave.  The assembly is a memcpy into PINNED memory and measured 3.4 GB/s
        // on one thread, which is the single largest cost in the whole chunk path (see run_chunk), so this is a
        // tuning knob with an environment knob of its own rather than a constant.
        const char* pt = std::getenv("STRATA_GLM_GPU_CHUNK_THREADS");
        const unsigned hw = std::thread::hardware_concurrency();
        par_threads_ = pt != nullptr ? std::atoll(pt) : (int64_t) std::min<unsigned>(hw == 0 ? 8u : hw, 8u);
        if (par_threads_ < 1) par_threads_ = 1;
        while (depth > 0) {
            bool ok = true;
            for (int64_t i = 0; i < depth && ok; ++i) {
                uint8_t* p = nullptr;
                ok = cudaMallocHost((void**) &p, (size_t) max_blob) == cudaSuccess;
                if (ok) stage_chunk_.push_back(p);
            }
            if (ok) break;
            for (uint8_t* p : stage_chunk_) cudaFreeHost(p);
            stage_chunk_.clear();
            cudaGetLastError();                         // clear the failure before the next attempt
            depth = depth >= 8 ? depth / 2 : depth - 1;
        }
        if (stage_chunk_.empty()) {
            err = "glm gpu experts: no pinned host memory for the chunk staging ring (" +
                  std::to_string((unsigned long long) max_blob) + " B a blob)";
            return false;
        }
        stage_ev_.assign(stage_chunk_.size(), nullptr);
    }
    stage_.assign((size_t) k, nullptr);
    for (auto& p : stage_) {
        if (cudaMallocHost((void**) &p, (size_t) max_blob) != cudaSuccess) {
            err = std::string("glm gpu experts: staging: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    std::fprintf(stderr, "strata generate: glm5-next experts in VRAM: %lld slots a MoE layer over layers "
                         "[%lld, %lld), %lld in all, %.2f GiB (the rest on the CPU; slots fill as experts are "
                         "routed)\n",
                 (long long) per_layer, (long long) layer_lo, (long long) layer_hi,
                 (long long) cache_.full_slots(), cache_.gib());
    if (chunk_tokens_ > 0)
        std::fprintf(stderr, "strata generate: glm5-next experts serve prefill too: chunks up to %lld tokens, "
                             "%lld entries a wave, %lld blobs of staging (%.0f MiB pinned)\n",
                     (long long) chunk_tokens_, (long long) cap_entry_, (long long) stage_chunk_.size(),
                     (double) stage_chunk_.size() * (double) max_blob / (1024.0 * 1024.0));
    return true;
}

bool GlmGpuExperts::run_hits(int64_t layer, const float* cur_dev, const int32_t* ids, int64_t k, float* parts_dev,
                             void* stream, std::vector<int32_t>& miss, std::string& err) {
    miss.clear();
    hit_pos_.clear();
    if (k < 1 || k > k_ || layer < 0 || layer >= n_layers_) {
        err = "glm gpu experts: layer " + std::to_string(layer) + " routes " + std::to_string(k) + " experts";
        return false;
    }
    if (hi_[(size_t) layer] == lo_[(size_t) layer]) {       // a layer this stage has no slots for
        for (int64_t i = 0; i < k; ++i) miss.push_back((int32_t) i);
        return true;
    }
    // Every 4096 tokens a layer, the counts are halved, so an expert that was hot an hour ago does not keep its
    // slot for ever.  Counted in tokens, not calls, so the period does not depend on the layer split.
    if (++calls_ % (4096 * moe_layers_) == 0)
        for (uint32_t& c : count_) c >>= 1;
    for (int64_t i = 0; i < k; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < n_expert_) ++count_[(size_t) (layer * n_expert_ + e)];
        const int32_t s = (e >= 0 && e < n_expert_) ? slot_[(size_t) (layer * n_expert_ + e)] : kNotResident;
        if (s == kNotResident) {
            miss.push_back((int32_t) i);
            continue;
        }
        h_ptr_[hit_pos_.size()] = (unsigned long long) (uintptr_t) cache_.device_slot(s);
        hit_pos_.push_back((int32_t) i);
    }
    hits_ += (int64_t) hit_pos_.size();
    misses_ += (int64_t) miss.size();
    rep_hits_ += (int64_t) hit_pos_.size();
    rep_miss_ += (int64_t) miss.size();
    if (calls_ % (256 * moe_layers_) == 0) {
        std::fprintf(stderr, "strata glm gpu experts: last 256 tokens %.1f%% hits, %lld resident, %lld swaps so far\n",
                     100.0 * (double) rep_hits_ / (double) std::max<int64_t>(1, rep_hits_ + rep_miss_),
                     (long long) admitted_, (long long) swaps_);
        rep_hits_ = rep_miss_ = 0;
    }
    const int32_t ng = (int32_t) hit_pos_.size();
    if (ng == 0) return true;
    // One group per hit, one entry per group: start[g] = g, dst = the hit's position in `parts`, tok = 0.
    int32_t* start = h_idx_;
    int32_t* n_groups = h_idx_ + k_ + 1;
    int32_t* dst = h_idx_ + k_ + 2;
    int32_t* tok = h_idx_ + 2 * k_ + 2;
    for (int32_t g = 0; g <= ng; ++g) start[g] = g;
    *n_groups = ng;
    for (int32_t g = 0; g < ng; ++g) { dst[g] = hit_pos_[(size_t) g]; tok[g] = 0; }
    cudaStream_t cs = (cudaStream_t) stream;
    if (cudaMemcpyAsync(d_ptr_, h_ptr_, (size_t) ng * sizeof(unsigned long long), cudaMemcpyHostToDevice, cs) !=
            cudaSuccess ||
        cudaMemcpyAsync(d_idx_, h_idx_, (size_t) (3 * k_ + 2) * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
            cudaSuccess) {
        err = std::string("glm gpu experts: index upload: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    strata::kernels::quantize_q8_1_rows(cur_dev, 1, n_embd_, xq_, stream);
    strata::kernels::native_expert_grouped(lay_[(size_t) layer], d_ptr_, d_idx_, d_idx_ + k_ + 1, d_idx_ + k_ + 2,
                                           d_idx_ + 2 * k_ + 2, k_, k_, xq_, scratch_, parts_dev, stream, ng);
    return true;
}

bool GlmGpuExperts::admit(int64_t layer, const int32_t* ids, const std::vector<int32_t>& miss, void* stream,
                          std::string& err) {
    if (miss.empty() || hi_[(size_t) layer] == lo_[(size_t) layer]) return true;
    size_t used = 0;
    const uint32_t* cnt = count_.data() + (size_t) (layer * n_expert_);
    bool swapped = false;
    for (int32_t i : miss) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_ || slot_[(size_t) (layer * n_expert_ + e)] != kNotResident) continue;
        int32_t s = kNotResident;
        if (next_[(size_t) layer] < hi_[(size_t) layer]) {
            s = (int32_t) next_[(size_t) layer]++;
            ++admitted_;
        } else if (!swapped) {
            // LFU: the least routed expert this layer holds gives its slot up, if the newcomer is clearly
            // hotter (+2, so two experts of equal heat do not trade places every token).  One swap a layer a
            // token, which is also the most the staging buffers can hold.
            int32_t victim = kNotResident;
            for (int64_t v = lo_[(size_t) layer]; v < hi_[(size_t) layer]; ++v)
                if (victim == kNotResident || cnt[owner_[(size_t) v]] < cnt[owner_[(size_t) victim]])
                    victim = (int32_t) v;
            if (victim == kNotResident || cnt[e] < cnt[owner_[(size_t) victim]] + 2) continue;
            slot_[(size_t) (layer * n_expert_ + owner_[(size_t) victim])] = kNotResident;
            s = victim;
            swapped = true;
            ++swaps_;
        } else {
            continue;
        }
        uint8_t* host = stage_[used++];
        if (!src_->copy_blob(layer, e, host)) {
            err = "glm gpu experts: reading layer " + std::to_string(layer) + " expert " + std::to_string(e);
            return false;
        }
        // The copy is ordered on `stream` behind this layer's hit kernel - which may still be reading the slot
        // being replaced - and ahead of the next token's kernel, which is what reads it.  The staging buffer is
        // reused on a later layer, and every MoE layer is preceded by a stream sync (the pool's hand-off), so
        // the previous copy has landed by then.
        if (!cache_.fill_slot(s, host, stream, err, (int64_t) lay_[(size_t) layer].bytes)) return false;
        slot_[(size_t) (layer * n_expert_ + e)] = s;
        owner_[(size_t) s] = e;
    }
    return true;
}

// ================================ THE PREFILL CHUNK ================================
//
// `run_hits` above answers "which of this token's k experts are already here"; this answers "get me all 263".
// The two are different enough that they share only the cache underneath them (glm_gpu_experts.hpp says why),
// and the shape of the difference is the wave loop: a token's k experts fit the layer's slots, a chunk's do not,
// so the chunk is walked `per_layer` experts at a time and each wave overwrites the one before it.
//
// **WHAT A WAVE COSTS AND WHERE IT GOES.**  The bytes are the same whatever the chunk size - 88 tokens already
// touch 263 of 288 experts, so a chunk cannot read fewer of them than a bigger chunk would - and the work is:
//   host:  copy_blob assembles each blob out of the mapping's three slices     ~11 MB, ~2 ms an expert
//   PCIe:  fill_slot DMAs the staged blob into the slot                        ~11 MB, ~1 ms an expert at 12 GB/s
//   card:  native_expert_grouped, one weight pass over every token that chose it
// Only the third scales with the chunk, which is the whole reason the route is worth building.  The first two
// do not, which is why they are a ring and an event rather than a sync: the host copy for wave n+1 has to be
// running while the card computes wave n, or the PCIe and the memcpy become the wall instead of the tensor
// cores.  Nothing here synchronizes; the stream orders the DMA before the kernel that reads the slot, and
// `stage_ev_` is what stops a staging buffer being refilled while its DMA is still in flight.
bool GlmGpuExperts::run_chunk(int64_t layer, const float* cur_dev, const int32_t* ids, int64_t T, int64_t k,
                              float* parts_dev, void* stream, bool& served, std::string& err) {
    served = false;
    if (chunk_tokens_ <= 0 || cap_entry_ <= 0) return true;            // a decode-only tier: the pool has it
    if (T < 1 || T > chunk_tokens_ || k < 1 || k > k_ || layer < 0 || layer >= n_layers_) return true;
    if (!serves_layer(layer)) return true;
    // STRATA_GLM_GPU_CHUNK_TIME: the breakdown below, and the wave-end sync that measures it (which is why the
    // instrument changes the schedule and its numbers are SHARES, not the arm's throughput).
    static const bool timing = std::getenv("STRATA_GLM_GPU_CHUNK_TIME") != nullptr;
    auto t_all = std::chrono::steady_clock::now(), t_mark = t_all;
    chunk_ = ChunkTimes();
    const size_t M = (size_t) (T * k);
    // **ALL OR NOTHING, SO THE IDS ARE CHECKED BEFORE ANYTHING MOVES.**  One out-of-range id would otherwise
    // leave that pair's row of `parts` holding whatever the previous layer wrote there - a wrong expert
    // contributing to a token, silently.  Handing the layer back to the pool is always correct, so a bad id
    // costs the layer's speed-up and nothing else.
    for (size_t j = 0; j < M; ++j)
        if (ids[j] < 0 || ids[j] >= n_expert_) return true;

    // ---- the plan: which experts the chunk routes to, and which entries chose each.  A counting sort over
    // `n_expert` (288) buckets and one pass over the entries, so it is linear and it comes out ascending -
    // which is what makes a wave's slots contiguous in the cache.
    std::fill(plan_off_.begin(), plan_off_.begin() + (size_t) n_expert_ + 1, 0);
    for (size_t j = 0; j < M; ++j) ++plan_off_[(size_t) ids[j] + 1];
    for (int64_t e = 0; e < n_expert_; ++e) plan_off_[(size_t) e + 1] += plan_off_[(size_t) e];
    plan_e_.clear();
    for (int64_t e = 0; e < n_expert_; ++e)
        if (plan_off_[(size_t) e + 1] > plan_off_[(size_t) e]) plan_e_.push_back((int32_t) e);
    if (plan_e_.empty()) { served = true; last_chunk_entries_ = 0; return true; }   // a chunk that routes nowhere
    plan_cur_.assign(plan_off_.begin(), plan_off_.begin() + (size_t) n_expert_);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < k; ++i) {
            const int32_t e = ids[(size_t) (t * k + i)];
            const int32_t p = plan_cur_[(size_t) e]++;
            plan_dst_[(size_t) p] = (int32_t) (t * k + i);   // the row of `parts` this pair owns
            plan_tok_[(size_t) p] = (int32_t) t;             // ...and the token whose activation it needs
        }
    }

    cudaStream_t cs = (cudaStream_t) stream;
    chunk_.plan = since(t_mark);
    t_mark = std::chrono::steady_clock::now();
    // One q8_1 image a token, the same quantization the single-token path does, done once for the whole chunk
    // because `native_expert_grouped` reads its activations from here by token index.
    strata::kernels::quantize_q8_1_rows(cur_dev, T, n_embd_, cxq_, stream);

    int32_t* start = hc_idx_;                                   // start[nw + 1], the group's first entry
    int32_t* n_groups = hc_idx_ + cap_group_ + 1;               // ...and the layout `run_hits` already builds
    int32_t* dst = hc_idx_ + cap_group_ + 2;
    int32_t* tok = hc_idx_ + cap_group_ + 2 + cap_entry_;
    const int64_t nslots = hi_[(size_t) layer] - lo_[(size_t) layer];
    const int64_t G = (int64_t) plan_e_.size();
    int64_t entries = 0;
    // A wave's blobs, assembled on the host before they are DMA'd.  Which blobs a wave needs is known from the
    // plan, so they are independent of each other and of the card - which is what makes the assembly something
    // that can run on several threads at once instead of on the one this function was called from.
    struct Job {
        int32_t e = 0, s = 0;
        uint8_t* host = nullptr;
    };
    std::vector<Job> jobs;

    for (int64_t g0 = 0; g0 < G;) {
        // A wave is as many consecutive experts as the slots, the entry cap and the staging ring all allow.
        // The first group is always taken whatever its size, because one expert is routed by at most `T`
        // tokens and `cap_entry_ >= T` (init), so this loop cannot fail to advance.
        // The wave is also bounded by the staging ring, since one buffer an expert is what lets the assembly
        // run at once: `stage_chunk_[(size_t) jobs.size()]` below indexes it, so a wave longer than the ring
        // would run off the end of it.
        int64_t g1 = g0, wave_entries = 0;
        while (g1 < G && g1 - g0 < nslots && g1 - g0 < cap_group_ && g1 - g0 < (int64_t) stage_chunk_.size()) {
            const int64_t e = plan_e_[(size_t) g1];
            const int64_t ge = plan_off_[(size_t) e + 1] - plan_off_[(size_t) e];
            if (g1 > g0 && wave_entries + ge > cap_entry_) break;
            wave_entries += ge;
            ++g1;
        }
        const int64_t nw = g1 - g0;
        const int64_t e0 = plan_off_[(size_t) plan_e_[(size_t) g0]];

        // ---- which of the wave's experts are not here yet, and where each of them goes.  One staging buffer
        // an expert - the ring is as deep as a wave - so a buffer is only ever refilled by a LATER wave, and
        // the event recorded when its blob was DMA'd is the whole of what has to be waited on.
        jobs.clear();
        for (int64_t g = g0; g < g1; ++g) {
            const int32_t e = plan_e_[(size_t) g];
            const size_t se = (size_t) (layer * n_expert_ + e);
            int32_t s = slot_[se];
            if (s == kNotResident) {
                // The expert is not here, so it takes this wave's slot for its position in the plan.  The
                // mapping is injective over a window this short (`nslots` groups, `% nslots`), so no two
                // experts of one wave can pick the same slot, and an expert that IS resident has already
                // claimed the slot it names - which is why the eviction below can never throw away an expert
                // this same wave is about to ask for.
                s = (int32_t) (lo_[(size_t) layer] + (g % nslots));
                const int32_t old = owner_[(size_t) s];
                if (old >= 0) slot_[(size_t) (layer * n_expert_ + old)] = kNotResident;
                jobs.push_back({e, s, stage_chunk_[(size_t) jobs.size()]});
            }
            start[g - g0] = (int32_t) (plan_off_[(size_t) e] - e0);
            hc_ptr_[g - g0] = (unsigned long long) (uintptr_t) cache_.device_slot(s);
        }

        // ---- the assembly.  Measured at 3.4 GB/s on one thread into pinned memory, which is 0.8 s a layer of
        // a 512-token chunk - an order of magnitude more than the card spends on the same layer, so it is the
        // one part of this that has to be spread over more than the one thread that called it.  `copy_blob` is
        // safe from several threads only for a source whose `transient` can be true (its contract), which is
        // exactly the GGUF-in-place case this path was built for; anything else is read one at a time.
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (stage_ev_[j] == nullptr) continue;
            const auto tw = std::chrono::steady_clock::now();
            if (cudaEventSynchronize((cudaEvent_t) stage_ev_[j]) != cudaSuccess) {
                err = std::string("glm gpu experts: the chunk staging ring: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
            chunk_.wait += since(tw);
        }
        const auto th = std::chrono::steady_clock::now();
        int32_t failed = -1;
        const int64_t nthreads = src_->transient(layer, 0)
                                     ? std::min<int64_t>(par_threads_, (int64_t) jobs.size())
                                     : 1;
        last_chunk_threads_ = nthreads;
        if (nthreads > 1) {
            std::atomic<size_t> at{0};
            std::vector<std::thread> pool;
            pool.reserve((size_t) nthreads);
            for (int64_t t = 0; t < nthreads; ++t)
                pool.emplace_back([&] {
                    for (size_t j = at++; j < jobs.size(); j = at++) {
                        if (!src_->copy_blob(layer, jobs[j].e, jobs[j].host)) {
                            failed = jobs[j].e;
                            return;
                        }
                    }
                });
            for (std::thread& t : pool) t.join();
        } else {
            for (const Job& jb : jobs)
                if (!src_->copy_blob(layer, jb.e, jb.host)) { failed = jb.e; break; }
        }
        chunk_.host += since(th);
        if (failed >= 0) {
            err = "glm gpu experts: reading layer " + std::to_string(layer) + " expert " + std::to_string(failed);
            return false;
        }
        chunk_.blobs += (int64_t) jobs.size();

        // ---- then the DMA, in plan order so the kernel that follows finds every slot it was promised.
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (!cache_.fill_slot(jobs[j].s, jobs[j].host, stream, err, (int64_t) lay_[(size_t) layer].bytes))
                return false;
            slot_[(size_t) (layer * n_expert_ + jobs[j].e)] = jobs[j].s;
            owner_[(size_t) jobs[j].s] = jobs[j].e;
            if (stage_ev_[j] == nullptr &&
                cudaEventCreateWithFlags((cudaEvent_t*) &stage_ev_[j], cudaEventDisableTiming) != cudaSuccess) {
                err = std::string("glm gpu experts: the chunk staging ring's events: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (cudaEventRecord((cudaEvent_t) stage_ev_[j], cs) != cudaSuccess) {
                err = std::string("glm gpu experts: marking the chunk staging ring: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        start[nw] = (int32_t) wave_entries;
        *n_groups = (int32_t) nw;
        for (int64_t j = 0; j < wave_entries; ++j) {
            dst[j] = plan_dst_[(size_t) (e0 + j)];
            tok[j] = plan_tok_[(size_t) (e0 + j)];
        }
        // Three uploads, one an array the kernel reads, rather than the whole cap-sized block: the layout is
        // strided (`start | n_groups | dst | tok`), and at k=8, T=2048 the caps make that block 66 KB an upload
        // where the wave actually uses 30.
        if (cudaMemcpyAsync(c_ptr_, hc_ptr_, (size_t) nw * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                            cs) != cudaSuccess ||
            cudaMemcpyAsync(c_idx_, start, (size_t) (nw + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(c_idx_ + cap_group_ + 1, n_groups, sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(c_idx_ + cap_group_ + 2, dst, (size_t) wave_entries * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(c_idx_ + cap_group_ + 2 + cap_entry_, tok, (size_t) wave_entries * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk plan upload: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        strata::kernels::native_expert_grouped(lay_[(size_t) layer], c_ptr_, c_idx_, c_idx_ + cap_group_ + 1,
                                               c_idx_ + cap_group_ + 2, c_idx_ + cap_group_ + 2 + cap_entry_,
                                               cap_group_, cap_entry_, cxq_, cscratch_, parts_dev, stream, 0);
        entries += wave_entries;
        ++chunk_.waves;
        // Only the instrument waits here: it is the one thing that turns the card's queue back into a
        // per-wave wall figure.  A run without it never synchronizes inside a layer.
        if (timing) {
            const auto tw = std::chrono::steady_clock::now();
            if (cudaStreamSynchronize(cs) != cudaSuccess) {
                err = "glm gpu experts: waiting for a chunk wave";
                return false;
            }
            chunk_.wave += since(tw);
        }
        g0 = g1;
    }
    chunk_.entries = entries;
    chunk_.total = since(t_all);
    if (timing) {
        const double gib = (double) chunk_.blobs * (double) lay_[(size_t) layer].bytes / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr, "strata glm gpu chunk: layer %lld, %lld entries in %lld waves, %lld blobs (%.2f GiB): "
                             "%.3f s host copy at %.2f GB/s, %.3f s ring wait, %.3f s plan, %.3f s card wall "
                             "(%.1f GiB/s into slots), %.3f s all [par %lld transient %d]\n",
                     (long long) layer, (long long) entries, (long long) chunk_.waves, (long long) chunk_.blobs,
                     gib, chunk_.host, gib * 1.073741824 / std::max(0.001, chunk_.host), chunk_.wait, chunk_.plan, chunk_.wave,
                     gib / std::max(0.001, chunk_.wave), chunk_.total, (long long) last_chunk_threads_,
                     src_ != nullptr && src_->transient(layer, 0) ? 1 : 0);
    }
    ++chunk_calls_;
    chunk_entries_ += entries;
    last_chunk_entries_ = entries;
    served = true;
    return true;
}

}  // namespace strata::core
