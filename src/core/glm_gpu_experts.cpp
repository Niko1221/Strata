// src/core/glm_gpu_experts.cpp - see include/strata/core/glm_gpu_experts.hpp.
#include "strata/core/glm_gpu_experts.hpp"

#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace strata::core {

namespace {
// A q8_1 block is 32 int8 plus a half `d` and a half `s`.
constexpr size_t kQ8_1Block = 36;
double since(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}
/// **THE ARENA'S BUDGET IS THE MACHINE'S, NOT A STAGE'S.**  A 4-way split is four `GlmGpuExperts` in ONE
/// process, and each of them serves a different quarter of the layers - so a per-stage cap would let a run pin
/// four times what the knob says the box may lose.  `STRATA_GLM_GPU_PIN_GIB` is therefore claimed down here, as
/// the bytes are actually taken, and a stage that arrives after the budget is gone keeps the pageable path.
///
/// **UNSET IS OFF, AND THAT IS A MEASUREMENT.**  The shape this began as - unset means `MemAvailable` less the
/// floor, so the arena sizes itself - was measured on the 5,304-token prompt, chunk 4096, 4-way split, against
/// the same binary with the knob at 0: **245.7 s / 21.6 tok/s against 103.3 s / 51.4 tok/s, a 2.4x regression.**
/// The registration is not the cost (the probe above measures pinned 2.2x faster than pageable); what the box
/// loses is the page cache, because every registered byte is a byte the kernel can no longer hand to the pool's
/// own reads - and this model's experts are read from a file mapping over and over.  So the arena is opt-in:
/// the knob names the GiB it may take, and 0 or unset is off, one `if` per layer, with the default path byte for
/// byte what it was.  Whoever picks this up owes the regression an explanation before it earns a default.
std::atomic<int64_t> g_pin_left{-1};
std::once_flag g_pin_once;
std::atomic<bool> g_pin_warned{false};   ///< one line, once, if a layer cannot be pinned

constexpr int64_t kGiB = 1024LL * 1024LL * 1024LL;
/// **WHAT THE ARENA MAY NOT TAKE: A FLOOR, NOT A FRACTION.**  `MemAvailable` less this is the budget when
/// `STRATA_GLM_GPU_PIN_GIB` is unset, and a reservation that would leave less than this is refused (checked
/// again per layer, because the arena is not the only thing on the box that grows).  It has to cover the
/// process's own buffers and the page cache the chunks that are NOT served by an arena still read through.
constexpr int64_t kArenaFloor = 16 * kGiB;

int64_t host_available_bytes() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return 0;
    char line[256];
    int64_t kb = -1;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        long long v = 0;
        if (std::sscanf(line, "MemAvailable: %lld kB", &v) == 1) {
            kb = (int64_t) v;
            break;
        }
    }
    std::fclose(f);
    return kb > 0 ? kb * 1024 : 0;   // 0 = "could not read it", which the caller treats as "no budget"
}

int64_t pin_left_bytes() {
    std::call_once(g_pin_once, [] {
        const char* v = std::getenv("STRATA_GLM_GPU_PIN_GIB");
        if (v != nullptr) {
            const long long gib = std::atoll(v);
            g_pin_left.store(gib > 0 ? gib * kGiB : 0);
            return;
        }
        // **UNSET IS OFF** - see the note on the budget above: an auto-sized arena measured a 2.4x regression on
        // the prompt this was built for (245.7 s against 103.3 s), which is a price the pool's page cache pays
        // and not one the knob's own bytes save.  A number is the opt-in.
        g_pin_left.store(0);
    });
    return g_pin_left.load();
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
    for (void* e : plan_ev_)
        if (e != nullptr) cudaEventDestroy((cudaEvent_t) e);
    if (wave_ev_ != nullptr) cudaEventDestroy((cudaEvent_t) wave_ev_);
    if (dma_ev_ != nullptr) cudaEventDestroy((cudaEvent_t) dma_ev_);
    if (dma_ev2_ != nullptr) cudaEventDestroy((cudaEvent_t) dma_ev2_);
    if (copy_stream_ != nullptr) cudaStreamDestroy((cudaStream_t) copy_stream_);
    // Off the card first, then off the CPU: a registered mapping cannot be unmapped while it is still registered.
    for (size_t l = 0; l < pin_map_.size(); ++l) {
        if (pin_map_[l] == nullptr) continue;
        cudaHostUnregister(pin_map_[l]);
        munmap(pin_map_[l], (size_t) ((uint64_t) n_expert_ * (uint64_t) lay_[l].bytes));
    }
}

bool GlmGpuExperts::init(ExpertSource* src, const std::vector<int>& gu_type, const std::vector<int>& d_type,
                         const std::vector<uint64_t>& blob_bytes, const std::vector<float>& swiglu_limit,
                         int64_t layer_lo, int64_t layer_hi, int64_t n_expert, int64_t k, int64_t n_embd,
                         int64_t n_ff, int64_t budget_bytes, int64_t chunk_tokens, std::string& err) {
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
        // The layer's own `swiglu_clamp_exp` travels with the layout: without it the card computes
        // `silu(gate) * up` where the pool computes `min(silu(gate), 10) * clamp(up, +-10)`, and the two
        // disagree on every entry whose gates or ups clear the limit (see the field's note in iq_kernels.hpp).
        lay_[(size_t) l] = strata::kernels::native_expert_layout(
            gu_type[(size_t) l], d_type[(size_t) l], n_embd, n_ff,
            (size_t) l < swiglu_limit.size() ? swiglu_limit[(size_t) l] : 0.0f);
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

    // ---- STRATA_GLM_GPU_CHUNK_DIRECT: DMA the blob rather than copy it.  `ExpertSource::slices` hands over an
    // expert's three role slices as pointers into the source's own mapping - the same three ranges
    // `copy_from_files` memcpys into a blob, minus the memcpy (native_expert_parity.cpp:727 assembles the same
    // blob the same way, so this is that layout, read in place).  A slot is already laid out [gate | up | down]
    // at exactly those three offsets, so each slice can be DMA'd straight to where it belongs and the wave
    // never makes a copy on the host at all.  At chunk 4096 that copy measured 0.84 s of a layer's 1.83 s, all
    // of it the card waiting on the CPU, and it is the larger half of the chunk path's non-card time.
    //
    // **THE OFFSETS ARE THE WHOLE RISK, SO EVERY LAYER IS CHECKED AGAINST `copy_blob` HERE, ONCE.**  A slice
    // boundary that is off by one row produces a plausible token, which is the failure this project pays for
    // most (see the header); comparing the assembled blob byte for byte at startup costs ~0.1 s and removes it.
    // The switch: on by default for a chunk, `STRATA_GLM_GPU_CHUNK_DIRECT=0` puts the ring back.  It is the
    // default because it was measured faster on BOTH source shapes - the pageable mapping (80.7 s against
    // 101.8 s for the same prompt at chunk 4096) and the resident arena - and because it is the only arm in
    // which the host copies nothing at all.
    const char* const direct_env = std::getenv("STRATA_GLM_GPU_CHUNK_DIRECT");
    bool direct = chunk_tokens > 0 &&
                  (direct_env == nullptr || (direct_env[0] != '\0' && direct_env[0] != '0'));
    bool direct_slices = false;   // three ranges to issue; false: the source's own blob, in registered memory
    if (direct) {
        // **A SOURCE THE DMA CAN READ FROM, IN ONE OF TWO SHAPES.**  `slices` is a GGUF-in-place native pack:
        // gate, up and down are three tensors, so a blob is three ranges and there is nothing to assemble.  A
        // source with no slices can still be read in place - `ArenaExpertSource` holds every expert assembled
        // in one registered mapping (`pinned`) - and then a blob is one range.  Neither, and the ring stays.
        int64_t probe = -1;
        for (int64_t l = layer_lo; l < layer_hi && probe < 0; ++l)
            if (blob_bytes[(size_t) l] != 0) probe = l;
        const uint8_t *pg = nullptr, *pu = nullptr, *pd = nullptr;
        if (probe < 0) {
            direct = false;
        } else if (src_->slices(probe, 0, &pg, &pu, &pd)) {
            direct_slices = true;
        } else {
            // The blob shape.  The registration is a contiguous prefix of the arena, asked for on the layer's
            // LAST expert (one call a layer, nothing to walk), and the blob is then compared with what
            // `copy_blob` assembles - the same promise the slice branch makes, that what the DMA reads is what
            // the staged path would have written.  For the arena that comparison is what it says it is (its
            // copy IS a memcpy of this range), which is exactly why it costs nothing and is still worth asking.
            for (int64_t l = layer_lo; l < layer_hi && direct; ++l) {
                if (blob_bytes[(size_t) l] == 0) continue;
                const strata::kernels::NativeExpertLayout& L = lay_[(size_t) l];
                const uint8_t* const bs = src_->blob(l, 0);
                if (!src_->pinned(l, n_expert_ - 1) || bs == nullptr) {
                    direct = false;
                    std::fprintf(stderr, "strata generate: glm5-next experts: layer %lld is not in registered "
                                         "memory, and has no slices to DMA; the chunk path keeps staging "
                                         "through RAM\n", (long long) l);
                    break;
                }
                std::vector<uint8_t> blob((size_t) L.bytes);
                if (!src_->copy_blob(l, 0, blob.data()) || std::memcmp(blob.data(), bs, (size_t) L.bytes) != 0) {
                    direct = false;
                    std::fprintf(stderr, "strata generate: glm5-next experts: layer %lld's blob is not the blob "
                                         "copy_blob assembles; the chunk path keeps staging through RAM\n",
                                 (long long) l);
                }
            }
        }
    }
    if (direct && direct_slices) {
        for (int64_t l = layer_lo; l < layer_hi && direct; ++l) {
            if (blob_bytes[(size_t) l] == 0) continue;
            const uint8_t *g = nullptr, *u = nullptr, *d = nullptr;
            const strata::kernels::NativeExpertLayout& L = lay_[(size_t) l];
            if (!src_->slices(l, 0, &g, &u, &d)) {
                direct = false;
                std::fprintf(stderr, "strata generate: glm5-next experts: layer %lld has no slices to DMA; the "
                                     "chunk path keeps staging through RAM\n", (long long) l);
                break;
            }
            std::vector<uint8_t> blob((size_t) L.bytes);
            if (!src_->copy_blob(l, 0, blob.data()) || std::memcmp(blob.data(), g, L.up_off) != 0 ||
                std::memcmp(blob.data() + L.up_off, u, L.down_off - L.up_off) != 0 ||
                std::memcmp(blob.data() + L.down_off, d, L.bytes - L.down_off) != 0) {
                direct = false;
                std::fprintf(stderr, "strata generate: glm5-next experts: layer %lld's slices are not the blob "
                                     "copy_blob assembles; the chunk path keeps staging through RAM\n", (long long) l);
            }
        }
    }
    direct_ = direct;
    direct_slices_ = direct && direct_slices;

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
    // `per_layer_` groups (one slot each) or `cap_entry_` entries.  `kPlanRing` copies of it, used round-robin -
    // see the header for why one copy is a correctness bug and not a saving (a pinned source makes the upload
    // asynchronous, so a single copy is refilled under the DMA that is still reading it).
    if (chunk_tokens_ > 0) {
        cap_group_ = per_layer_;
        plan_ring_ = kPlanRing;
        const size_t idx = (size_t) (cap_group_ + 2 + 2 * cap_entry_);
        const size_t ring = (size_t) plan_ring_;
        if (cudaMalloc((void**) &c_ptr_, ring * (size_t) cap_group_ * sizeof(unsigned long long)) != cudaSuccess ||
            cudaMalloc((void**) &c_idx_, ring * idx * sizeof(int32_t)) != cudaSuccess ||
            cudaMallocHost((void**) &hc_ptr_,
                           ring * (size_t) cap_group_ * sizeof(unsigned long long)) != cudaSuccess ||
            cudaMallocHost((void**) &hc_idx_, ring * idx * sizeof(int32_t)) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk plan: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        plan_ev_.assign(ring, nullptr);
        plan_off_.resize((size_t) n_expert + 1);
        plan_dst_.resize((size_t) (chunk_tokens_ * k));
        plan_tok_.resize((size_t) (chunk_tokens_ * k));
        plan_e_.reserve((size_t) n_expert);
        // The diagnostic map the check reads: one wave index an entry, and one start index a wave (a wave holds
        // at least one entry, so no chunk can have more waves than it has entries).
        entry_wave_.assign((size_t) (chunk_tokens_ * k), 0);
        wave_off_.assign((size_t) (chunk_tokens_ * k) + 2, 0);
    }
    uint64_t max_blob = 0;
    for (int64_t l = layer_lo; l < layer_hi; ++l) max_blob = std::max(max_blob, blob_bytes[(size_t) l]);
    // The chunk's staging ring, deep enough that the host's assembly of wave n+1 runs while the card computes
    // wave n.  `nslots` bounds a wave, so nothing deeper is ever useful; 24 is what fits comfortably in pinned
    // memory at GLM's 11.13 MiB a blob (267 MiB a stage) and is deep enough that an event is normally already
    // satisfied.  If the box will not give that, take less rather than refusing the tier - a shallower ring
    // stalls, it does not corrupt.
    // In direct mode there is nothing to stage: the source's own mapping is what the DMA reads, so the ring is
    // not allocated and a wave is bounded by the slots alone (which is what it has to be anyway - the slots are
    // where the DMA writes, and a wave longer than they are would overwrite its own).
    if (direct_) stage_wave_ = per_layer_;
    // How many threads assemble a wave.  The assembly is a memcpy into PINNED memory and measured 3.4 GB/s on
    // one thread, which is the single largest cost in the whole chunk path (see run_chunk), so this is a tuning
    // knob with an environment knob of its own rather than a constant.  The default was the machine's cores
    // capped at 8; the cap is now 16, because a standalone memcpy into pinned memory of this size measures
    // 16.7 GB/s over 16 threads against 6.9-11.1 GB/s for the H2D that consumes it - the link, not the copy, is
    // the limit from there on, so more threads than this buy nothing.
    //
    // Outside the ring's `if` because the arena wants it too: a layer's first chunk stages into the arena with
    // the same pool, and a layer whose arena is already settled stages nothing at all.
    {
        const char* pt = std::getenv("STRATA_GLM_GPU_CHUNK_THREADS");
        const unsigned hw = std::thread::hardware_concurrency();
        par_threads_ = pt != nullptr ? std::atoll(pt) : (int64_t) std::min<unsigned>(hw == 0 ? 16u : hw, 16u);
        if (par_threads_ < 1) par_threads_ = 1;
    }
    // ---- the pinned arena's table, one entry a layer.  Sized here and filled on first use (`pin_layer`), so a
    // run that never serves a chunk, or a box whose budget is already spent, pays one null pointer a layer.
    if (chunk_tokens_ > 0) {
        pin_map_.assign((size_t) n_layers_, nullptr);
        pin_have_.assign((size_t) (n_layers_ * n_expert_), 0);
    }
    if (chunk_tokens_ > 0 && !direct_) {
        int64_t depth = 0;
        for (int64_t l = layer_lo; l < layer_hi; ++l) depth = std::max(depth, hi_[(size_t) l] - lo_[(size_t) l]);
        depth = std::min<int64_t>(depth, 64);
        // THE RING IS TWICE A WAVE DEEP.  One wave deep - which is what this was - makes the host's assembly of
        // wave g+1 wait on wave g's DMA before it may write into its buffer, and that wait is the whole of why a
        // chunk's wall is `host + card` and not `max(host, card)`.  Measured on layer 44 of a 4096-token chunk:
        // 2.080 s all = 0.871 s host copy at 4.40 GB/s + 1.205 s card.  Two waves' worth of buffers lets the
        // assembly of wave g+1 write into a buffer no DMA has looked at since wave g-1, so it waits on nothing
        // and the memcpy runs while the card streams.  A wave is still capped at one wave's worth
        // (`stage_wave_`), so it can never wrap onto a buffer it is itself still filling.
        depth *= 2;
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
        stage_wave_ = (int64_t) (stage_chunk_.size() / 2);
        if (stage_wave_ < 1) stage_wave_ = 1;
        stage_ev_.assign(stage_chunk_.size(), nullptr);
        ring_head_ = 0;
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
    if (chunk_tokens_ > 0) {
        std::fprintf(stderr, "strata generate: glm5-next experts serve prefill too: chunks up to %lld tokens, "
                             "%lld entries a wave, ", (long long) chunk_tokens_, (long long) cap_entry_);
        if (direct_) std::fprintf(stderr, "no staging ring at all (a wave DMAs the source's own bytes)\n");
        else std::fprintf(stderr, "%lld blobs of staging (%.0f MiB pinned)\n", (long long) stage_chunk_.size(),
                          (double) stage_chunk_.size() * (double) max_blob / (1024.0 * 1024.0));
    }
    if (direct_)
        std::fprintf(stderr, direct_slices_
                                 ? "strata generate: glm5-next experts: the chunk path DMAs each expert's gate, "
                                   "up and down slices straight from the model mapping into its slot - no "
                                   "staging copy\n"
                                 : "strata generate: glm5-next experts: the chunk path DMAs each expert's blob "
                                   "straight out of the source's registered memory into its slot - the host "
                                   "copies nothing at all\n");
    return true;
}

// ---- **THE SLOT TABLE'S OWN INVARIANT.**  `slot_[layer * n_expert + e] = s` and `owner_[s] = e` are two halves
// of one mapping, and every path that fills a slot - the decode's `admit`, the chunk's waves - writes both.  A
// path that writes one and not the other produces exactly what a wrong-expert check reports: a plausible token
// from another expert's bytes, found one token later and in another function.  This is the check that names the
// layer, the slot and both experts at the moment the two disagree.  `STRATA_GLM_GPU_SLOT_CHECK=1`; the cost is a
// walk of one layer's `n_expert_` entries and its slots, a few hundred comparisons, once a layer a call.
static bool slot_table_ok(const std::vector<int32_t>& slot, const std::vector<int32_t>& owner, int64_t layer,
                          int64_t n_expert, int64_t lo, int64_t hi, std::string& err) {
    const int32_t* const row = slot.data() + (size_t) (layer * n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        const int32_t s = row[e];
        if (s == kNotResident) continue;
        if (s < lo || s >= hi) {
            err = "layer " + std::to_string(layer) + " expert " + std::to_string(e) + " claims slot " +
                  std::to_string(s) + ", outside the layer's own [" + std::to_string(lo) + ", " + std::to_string(hi) +
                  ") - the table says where an expert lives, and it lives where no slot is";
            return false;
        }
        if (owner[(size_t) s] != e) {
            err = "layer " + std::to_string(layer) + " expert " + std::to_string(e) + " claims slot " +
                  std::to_string(s) + ", and that slot says it holds expert " + std::to_string(owner[(size_t) s]) +
                  " - the expert that really is there is unclaimed, or the bytes under it were replaced without "
                  "clearing this entry";
            return false;
        }
    }
    for (int64_t s = lo; s < hi; ++s) {
        const int32_t o = owner[(size_t) s];
        if (o < 0) continue;
        if (o >= n_expert || row[o] != s) {
            err = "layer " + std::to_string(layer) + " slot " + std::to_string(s) + " says it holds expert " +
                  std::to_string(o) + ", and that expert's own entry " +
                  (o >= n_expert ? std::string("does not exist") : "points at slot " + std::to_string(row[o]));
            return false;
        }
    }
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
    static const bool slot_check = [] {
        const char* v = std::getenv("STRATA_GLM_GPU_SLOT_CHECK");
        return v != nullptr && v[0] == '1';
    }();
    if (slot_check &&
        !slot_table_ok(slot_, owner_, layer, n_expert_, lo_[(size_t) layer], hi_[(size_t) layer], err)) {
        err = "glm gpu experts: " + err;
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
            // **A SLOT NOTHING CLAIMS IS TAKEN BEFORE ANYTHING IS EVICTED, AND IT HAS TO BE SKIPPED ANYWAY.**
            // The chunk path leaves such slots behind: a wave that finds an expert resident in the OTHER half
            // re-homes it to its own half and clears `owner_` on the slot it left - those bytes are two waves
            // old and no `slot_` entry points at them any more.  Filling one costs no expert its slot, so it
            // goes first.  When every slot is claimed, the LFU comparison runs; either way the unclaimed ones
            // must not be compared, because `owner_[v]` is -1 there and `cnt[-1]` is a count read out of the
            // PREVIOUS layer's table, while the clear below would write `slot_[layer * n_expert_ - 1]` - one
            // expert of that same wrong layer.  Measured, not theorized: with the chunk path on, this arm read
            // the wrong expert at layer 6 of the first decode token after the prefill (max/rms 5.08 of the
            // row's rms, `STRATA_GLM_GPU_CHECK=1`, 5,304-token prompt, chunk 4096, arena source, direct DMA).
            int32_t vacant = kNotResident, victim = kNotResident;
            for (int64_t v = lo_[(size_t) layer]; v < hi_[(size_t) layer]; ++v) {
                const int32_t o = owner_[(size_t) v];
                if (o < 0) {
                    if (vacant == kNotResident) vacant = (int32_t) v;
                } else if (victim == kNotResident || cnt[o] < cnt[owner_[(size_t) victim]]) {
                    victim = (int32_t) v;
                }
            }
            if (vacant != kNotResident) {
                s = vacant;
                ++admitted_;
            } else {
                // LFU: the least routed expert this layer holds gives its slot up, if the newcomer is clearly
                // hotter (+2, so two experts of equal heat do not trade places every token).  One swap a layer a
                // token, which is also the most the staging buffers can hold.
                if (victim == kNotResident || cnt[e] < cnt[owner_[(size_t) victim]] + 2) continue;
                slot_[(size_t) (layer * n_expert_ + owner_[(size_t) victim])] = kNotResident;
                s = victim;
                swapped = true;
                ++swaps_;
            }
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
// ---- **THE ONE COPY THE WHOLE CHUNK PATH RESTS ON.**  `run_chunk` feeds a wave by handing `cudaMemcpyAsync` a
// pointer into the model - three slices of a `MAP_SHARED` file mapping - and a file mapping is PAGEABLE, so the
// driver cannot DMA from it.  It stages every byte through an internal pinned buffer, on the calling thread, and
// blocks that thread for the whole transfer; there is no flag that changes this, and registering the mapping is
// refused outright (`cudaHostRegister` on a `MAP_SHARED` mapping: `cudaHostRegisterReadOnly` -> "operation not
// supported", every other flag combination -> "invalid argument", measured on the real shard).
//
// The engine then measures something that looks impossible and is the tell: in direct mode a layer's "card wall"
// reads 22.4 GiB/s into the slots, well past any PCIe link on this box.  It reads that because the enqueue has
// ALREADY moved the bytes by the time the event is recorded - the copy was synchronous - and the time it really
// cost is sitting in the host, 0.53 s a layer at 5.5 GiB/s, where nothing overlaps it: `all` = enqueue + kernel,
// exactly additive, and the card idles while the host copies.
//
// So the bytes are moved ONCE into memory the driver accepts: an anonymous mapping, registered, holding the
// layer's experts in `copy_blob`'s own layout (gate, up, down back to back - byte for byte what a slot wants).
// Filling it is the host's assembly, the same memcpy the staged path already does, done once a layer for the
// process rather than once a layer per chunk.  After that a wave is a single `cudaMemcpyAsync` an expert out of
// RAM at the pinned rate (measured 10.07 GiB/s against the pageable 4.55, with the host blocked for none of it,
// `bw/pin.cu`), the host runs ahead of the card instead of beside it, and the transfer happens UNDER the
// kernel rather than before it.  This is the shape glm53-flash-offload runs - the whole model in a registered
// anonymous mapping - and it is why its prefill can sit at its PCIe limit with an idle host.
bool GlmGpuExperts::pin_layer(int64_t layer) {
    if (pin_map_.size() <= (size_t) layer) return false;
    if (pin_map_[(size_t) layer] != nullptr) return true;
    if (src_ == nullptr) return false;
    const size_t bytes = (size_t) lay_[(size_t) layer].bytes;
    const size_t len = (size_t) n_expert_ * bytes;
    if (len == 0) return false;
    // Claim the budget before taking the memory, so four stages arriving together cannot each be told there is
    // room for all of them.
    int64_t left = pin_left_bytes();
    bool claimed = false;
    while (left >= (int64_t) len) {
        if (g_pin_left.compare_exchange_weak(left, left - (int64_t) len)) {
            claimed = true;
            break;
        }
    }
    if (!claimed) return false;
    // ...and the floor, checked against what the box has RIGHT NOW rather than against the budget the first
    // chunk computed: the arena stays for the life of the process, and so does everything else that grew since.
    const int64_t avail = host_available_bytes();
    if (avail > 0 && avail < (int64_t) len + kArenaFloor) {
        g_pin_left.fetch_add((int64_t) len);
        return false;
    }
    void* const p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        g_pin_left.fetch_add((int64_t) len);
        return false;
    }
    // **THE FAULTS ARE THE FILL, AND 4 KB AT A TIME THEY COST MORE THAN THE COPY.**  A fresh anonymous mapping
    // has no pages: the chunk's own copy faults every one in, and the kernel zeroes it before the memcpy writes
    // it, so 3.13 GiB a layer is 800k faults, 800k zeroing passes and then the copy - measured on this box as
    // 1.9 s a layer, which turned the first chunk's 18.25 s into 35.40 s and made the whole change a wash on a
    // two-chunk prompt.  MADV_HUGEPAGE makes those faults 2 MiB at a time (1,536 a layer) and the zeroing
    // proportional, and it still lets the copy fault its own pages on its own sixteen threads, in parallel,
    // which a `MAP_POPULATE` on the calling thread would not.  It is a hint; a box with THP off keeps the small
    // pages and merely stays as slow as it measured before.
    (void) madvise(p, len, MADV_HUGEPAGE);
    if (cudaHostRegister(p, len, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();                         // the registration failed; nothing else is wrong
        munmap(p, len);
        g_pin_left.fetch_add((int64_t) len);
        return false;
    }
    // **NOTHING IS COPIED HERE.**  The bytes arrive with the first chunk that reads them: its own staging copy
    // - the `copy_blob` the staged path already runs, over the same experts, in the same waves - writes them
    // into this mapping instead of into a ring buffer, and `pin_have_` records which experts are therefore in
    // it.  So the chunk that pays for the arena pays what it always paid, and every chunk after it reads
    // registered memory the driver can DMA out of.  See the header.
    pin_map_[(size_t) layer] = (uint8_t*) p;
    pin_bytes_ += (int64_t) len;
    ++pin_layers_;
    return true;
}

bool GlmGpuExperts::run_chunk(int64_t layer, const float* cur_dev, const int32_t* ids, int64_t T, int64_t k,
                              float* parts_dev, void* stream, bool& served, std::string& err) {
    served = false;
    if (chunk_tokens_ <= 0 || cap_entry_ <= 0) return true;            // a decode-only tier: the pool has it
    if (T < 1 || T > chunk_tokens_ || k < 1 || k > k_ || layer < 0 || layer >= n_layers_) return true;
    if (!serves_layer(layer)) return true;
    // STRATA_GLM_GPU_CHUNK_TIME: the breakdown below, and the wave-end sync that measures it (which is why the
    // instrument changes the schedule and its numbers are SHARES, not the arm's throughput).
    // 1 is the breakdown; 2 is the breakdown AND a second launch of the same wave kernel, which splits `wave`
    // into its DMA half and its kernel half (see the kernel launch below).
    static const int timing = [] { const char* v = std::getenv("STRATA_GLM_GPU_CHUNK_TIME"); return v ? std::atoi(v) : 0; }();
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
    // The chunk's own copy stream and the event that hands a wave back to `cs`.  Made once and kept: this
    // function runs once a layer, and a layer is 21 to 32 waves of copies.
    if (copy_stream_ == nullptr) {
        if (cudaStreamCreateWithFlags((cudaStream_t*) &copy_stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &wave_ev_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &dma_ev_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags((cudaEvent_t*) &dma_ev2_, cudaEventDisableTiming) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk's copy stream: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    cudaStream_t copy = (cudaStream_t) copy_stream_;
    // **THE PAGES, ASKED FOR BEFORE THEY ARE WANTED.**  A chunk reads ~3 GiB of experts a layer out of a 157 GB
    // model, and this box's page cache does not hold that model whole - measured residency is 91-98% of a
    // shard, so a few per cent of every wave is a minor fault, and sixteen threads faulting 4 KB at a time
    // defeat the kernel's readahead, which turns a handful of missing pages into a stall of the whole wave.
    // `warm` is the source's own MADV_WILLNEED over exactly the slices this layer's plan needs, so the
    // readahead runs under the wave loop instead of inside it.  It is a hint, and a source with nothing to
    // advise - or no source at all - ignores it.
    // **AND NEVER FOR THE EXPERTS THE ARENA HOLDS.**  A hint is a READ: once a layer's experts sit in registered
    // RAM, the file pages behind them are dead - and on this box they are evicted, because the arena is what
    // fills the memory they used to live in.  Asking for them again turns every chunk after the first into
    // 3 GiB a layer of block-device traffic at 0.55-0.71 GiB/s, which is the one way this path could be slower
    // than the copy it replaces.  So the hint covers exactly the experts still to be assembled, and a layer
    // whose plan is wholly settled asks for nothing at all.
    if (src_ != nullptr) {
        plan_warm_.clear();
        const bool any_arena = pin_map_.size() > (size_t) layer && pin_map_[(size_t) layer] != nullptr;
        for (const int32_t e : plan_e_)
            if (!any_arena || pin_have_[(size_t) layer * (size_t) n_expert_ + (size_t) e] == 0) plan_warm_.push_back(e);
        if (!plan_warm_.empty()) src_->warm(layer, plan_warm_.data(), (int64_t) plan_warm_.size());
        chunk_.warm += since(t_mark);
        t_mark = std::chrono::steady_clock::now();
    }
    // ---- **THE LAYER'S EXPERTS, MOVED WHERE THE DRIVER WILL READ THEM FROM.**  Once a layer, ever: the first
    // chunk a stage serves of this layer RESERVES the arena - anonymous memory the driver will DMA out of - and
    // its own staging copy is what fills it, expert by expert, at the rate and in the waves the ring path was
    // assembling at anyway.  So the chunk that pays for the arena pays what it always paid, and every chunk
    // after it reads registered memory instead of the page cache.  A layer that cannot be reserved keeps the
    // pageable path it is on today, so this can only ever cost the speed it was meant to buy - never the answer.
    if (!pin_map_.empty() && pin_map_[(size_t) layer] == nullptr) {
        const auto tf = std::chrono::steady_clock::now();
        const bool got_arena = pin_layer(layer);
        chunk_.fill += since(tf);                   // the refusal is timed too: a floor check is not free either
        if (got_arena) {
            std::fprintf(stderr,
                         "strata glm gpu experts: CUD%d layer %lld reserved %.2f GiB of RAM for its experts "
                         "(%.2f GiB over %lld layers here); this chunk's own copy fills it, and the chunks after "
                         "this one DMA out of it\n",
                         dev_, (long long) layer, (double) (n_expert_ * (int64_t) lay_[(size_t) layer].bytes) /
                                                        (1024.0 * 1024.0 * 1024.0),
                         (double) pin_bytes_ / (1024.0 * 1024.0 * 1024.0), (long long) pin_layers_);
        } else if (!g_pin_warned.exchange(true)) {
            std::fprintf(stderr, "strata glm gpu experts: CUD%d layer %lld could not be given an arena (the "
                                 "budget is spent, or there is not enough free RAM for it); those layers keep "
                                 "the staged path\n",
                         dev_, (long long) layer);
        }
        t_mark = std::chrono::steady_clock::now();
    }
    // One q8_1 image a token, the same quantization the single-token path does, done once for the whole chunk
    // because `native_expert_grouped` reads its activations from here by token index.
    strata::kernels::quantize_q8_1_rows(cur_dev, T, n_embd_, cxq_, stream);

    // One wave's slice of the plan ring.  `plan_h_`/`plan_d_` are this wave's pinned source and its device copy,
    // and `plan_r_` indexes `plan_ev_` - the event that says the upload from this slice has landed, which is
    // what makes refilling it safe (see the header).
    const int64_t plan_idx = cap_group_ + 2 + 2 * cap_entry_;
    const int64_t nslots = hi_[(size_t) layer] - lo_[(size_t) layer];
    // ---- THE SLOTS ARE CUT IN TWO HALVES, AND CONSECUTIVE WAVES TAKE TURNS BETWEEN THEM.  This is the whole
    // of the overlap: wave n+1's copies write the half wave n's kernel is not reading, so they may be issued
    // as soon as they are known instead of after that kernel retires - and the copies of a wave are a few
    // GiB, which on this box is several times what the wave's kernel costs.  Measured on the 5,294-token
    // prompt, a stage of 12 layers at chunk 4096: 9.40 s on the card against 27.44 s of layer wall, i.e. the
    // card was idle for two thirds of every chunk waiting for its own experts.
    //
    // A wave is then `halves`-th of the layer's slots, and the reverse edge is the wave that last used THIS
    // half - two waves back, not one (`dma_ev_`/`dma_ev2_`).  A layer with a single slot gets one half and the
    // old one-wave-deep alternation, because two halves do not fit in one slot.
    const int64_t halves = nslots >= 2 ? 2 : 1;
    const int64_t wave_cap = nslots / halves;
    const int64_t G = (int64_t) plan_e_.size();
    int64_t entries = 0;
    // A wave's blobs, assembled on the host before they are DMA'd.  Which blobs a wave needs is known from the
    // plan, so they are independent of each other and of the card - which is what makes the assembly something
    // that can run on several threads at once instead of on the one this function was called from.
    struct Job {
        int32_t e = 0, s = 0;
        uint8_t* host = nullptr;
        size_t ring = 0;             // which staging buffer, and so which event guards its previous DMA
        bool settled = false;        // the arena already holds this expert: nothing to assemble, DMA it
    };
    const size_t kNoRing = (size_t) -1;   // an arena job stages nothing, so no ring buffer guards it
    const size_t ring_size = stage_chunk_.size();
    std::vector<Job> jobs;
    int64_t wave_i = 0;   // counts waves, so each takes the next slice of the plan ring
    // ---- STRATA_GLM_GPU_VERIFY_SLOT=<layer>: WHAT THE KERNEL IS ABOUT TO READ, CHECKED AGAINST ITS SOURCE.
    // A wrong row has exactly two possible parents - the bytes the kernel read, or the activations it applied
    // them to - and the host cannot tell them apart, because the check compares the card's row against a CPU row
    // computed from the file.  This is the arm that separates them: before a wave's kernel is launched, every
    // slot that wave filled is copied back and compared byte for byte against the very bytes `issue` copied into
    // it, and the plan the kernel will index is read back against the host mirror it was uploaded from.  It is a
    // diagnostic, off unless the variable names a layer, and it costs a read-back and a sync a wave - so the
    // layers it does not name run exactly as they always do.  A layer is verified from `verify_from` up, because
    // a slot filled while a lower layer ran is still resident when this one reads it.
    static const int64_t verify_from = [] {
        const char* v = std::getenv("STRATA_GLM_GPU_VERIFY_SLOT");
        return (v != nullptr) ? (int64_t) std::strtoll(v, nullptr, 10) : (int64_t) -1;
    }();
    const bool verify = verify_from >= 0 && layer >= verify_from;
    int64_t vfy_slots = 0, vfy_sec_bad = 0, vfy_plan_bad = 0;

    for (int64_t g0 = 0; g0 < G;) {
        // A wave is as many consecutive experts as the slots, the entry cap and the staging ring all allow.
        // The first group is always taken whatever its size, because one expert is routed by at most `T`
        // tokens and `cap_entry_ >= T` (init), so this loop cannot fail to advance.
        // The wave is also bounded by one wave's worth of the staging ring, so a wave can never wrap round the
        // ring onto a buffer it is itself still filling.
        int64_t g1 = g0, wave_entries = 0;
        while (g1 < G && g1 - g0 < wave_cap && g1 - g0 < cap_group_ && g1 - g0 < stage_wave_) {
            const int64_t e = plan_e_[(size_t) g1];
            const int64_t ge = plan_off_[(size_t) e + 1] - plan_off_[(size_t) e];
            if (g1 > g0 && wave_entries + ge > cap_entry_) break;
            wave_entries += ge;
            ++g1;
        }
        const int64_t nw = g1 - g0;
        const int64_t e0 = plan_off_[(size_t) plan_e_[(size_t) g0]];
        // Diagnostic only - see `last_chunk_entry_wave`: which wave every plan entry rides in, and where each
        // wave starts.  Waves take consecutive experts of an ascending plan, so the wave's first entry is
        // simply the count of entries the waves before it carried, and its own are `[e0, e0 + wave_entries)`.
        if (wave_i < (int64_t) wave_off_.size()) wave_off_[(size_t) wave_i] = (int32_t) entries;
        for (int64_t j = 0; j < wave_entries && e0 + j < (int64_t) entry_wave_.size(); ++j)
            entry_wave_[(size_t) (e0 + j)] = (int32_t) wave_i;

        // ---- this wave's plan ring slot, and the wait that makes refilling it legal.  A pinned source makes
        // the upload a real DMA the host does not wait for, so the slot is only reusable once the previous
        // upload FROM IT has landed - which is what `plan_ev_[r]` says.  `plan_ring_` is 8 deep and the host
        // runs two or three waves ahead (it is bounded by the staging ring, and in sliced mode by the pageable
        // copies it has to issue itself), so this wait is satisfied on arrival rather than a stall.
        int64_t r = 0;
        if (plan_ring_ > 0) {
            r = wave_i % plan_ring_;
            if (plan_ev_[(size_t) r] != nullptr &&
                cudaEventSynchronize((cudaEvent_t) plan_ev_[(size_t) r]) != cudaSuccess) {
                err = std::string("glm gpu experts: waiting for a chunk wave's plan upload: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        unsigned long long* h_ptr_w = hc_ptr_ + (size_t) (r * cap_group_);
        int32_t* h_idx_w = hc_idx_ + (size_t) (r * plan_idx);
        int32_t* start = h_idx_w;                               // start[nw + 1], the group's first entry
        int32_t* n_groups = h_idx_w + cap_group_ + 1;           // ...and the layout `run_hits` already builds
        int32_t* dst = h_idx_w + cap_group_ + 2;
        int32_t* tok = h_idx_w + cap_group_ + 2 + cap_entry_;

        // ---- which of the wave's experts are not here yet, and where each of them goes.  One staging buffer
        // an expert - the ring is as deep as a wave - so a buffer is only ever refilled by a LATER wave, and
        // the event recorded when its blob was DMA'd is the whole of what has to be waited on.
        // In direct mode a wave stages nothing, so there is no ring buffer to protect and no event to wait on.
        // `slices` was checked for every layer at init, so this cannot fail per wave.
        //
        // **A PINNED LAYER IS ALWAYS "SLICED".**  What `sliced` really picks is where the DMA reads from - a
        // pointer into memory the driver may DMA out of, rather than a staging buffer the host filled first -
        // and the arena is exactly that.  So a pinned layer takes this branch whether or not the source hands
        // over slices of its own, and the ring it would otherwise have staged through is not touched.
        //
        // **AND THE ARENA IS DECIDED ONE EXPERT AT A TIME, WHICH IS THE WHOLE POINT OF IT.**  A layer that has
        // one does not wait for it to be filled: the experts it already holds are DMA'd out of it, and the ones
        // it does not are assembled into it by this very wave - the copy the ring path was doing anyway, into a
        // buffer that is still there next time.  `pin_have_` tells the two apart and is only ever written on
        // this thread, after the wave's pool has joined, never by the pool itself.
        uint8_t* const arena = pin_map_.size() > (size_t) layer ? pin_map_[(size_t) layer] : nullptr;
        const bool has_arena = arena != nullptr;
        const size_t arena_bytes = (size_t) lay_[(size_t) layer].bytes;
        // ---- THIS WAVE'S HALF, AND THE ONE RULE THAT MAKES THE OVERLAP LEGAL: A KERNEL READS ONLY ITS OWN
        // HALF.  Wave n's kernel must be running while wave n+1's copies land, and the copies overwrite whole
        // slots - so no slot wave n reads may be one of them.  The half takes care of the misses; the rule
        // takes care of the residents, which is why an expert whose slot sits in the other half is NOT a hit
        // here.  It is dropped and fetched into this half instead.  (At the chunk sizes that matter every
        // expert of the layer is routed anyway, so a resident from the last chunk is a few percent of the
        // wave and this costs a few percent of the traffic - against a wave time that halves.)
        const int64_t half = (halves == 2) ? (wave_i & 1) : 0;
        const int64_t base = lo_[(size_t) layer] + half * wave_cap;
        if ((int64_t) slot_busy_.size() < wave_cap) slot_busy_.assign((size_t) wave_cap, 0);
        // The residents first, so the misses can step over them: a slot one of this wave's own experts is
        // sitting in is read by the kernel about to run and may not be overwritten.
        for (int64_t g = g0; g < g1; ++g) {
            const int32_t s = slot_[(size_t) (layer * n_expert_ + plan_e_[(size_t) g])];
            if (s != kNotResident && s >= base && s < base + wave_cap) slot_busy_[(size_t) (s - base)] = 1;
        }
        int64_t take = 0;
        size_t n_ring = 0;                    // this wave's ring jobs, for the head advance below
        jobs.clear();
        for (int64_t g = g0; g < g1; ++g) {
            const int32_t e = plan_e_[(size_t) g];
            const size_t se = (size_t) (layer * n_expert_ + e);
            int32_t s = slot_[se];
            if (s != kNotResident && (s < base || s >= base + wave_cap)) {
                owner_[(size_t) s] = -1;              // it is the other half's; this wave may not read it
                s = kNotResident;
            }
            if (s == kNotResident) {
                // The expert is not here, so it takes the next free slot of this wave's half.  There is always
                // one: the wave holds `wave_cap` groups at most and the half is `wave_cap` slots, of which the
                // residents above take some - so `misses <= wave_cap - residents`, which is the free count.
                while (take < wave_cap && slot_busy_[(size_t) take]) ++take;
                s = (int32_t) (base + take);
                slot_busy_[(size_t) take] = 1;
                ++take;
                const int32_t old = owner_[(size_t) s];
                if (old >= 0) slot_[(size_t) (layer * n_expert_ + old)] = kNotResident;
                // **THE JOB, AND NOTHING ELSE MAY BE SKIPPED HERE.**  The `slot_` write and the plan's two rows
                // below are what the kernel is handed this wave; an arena job skips only the RING's own
                // bookkeeping (there is no staging buffer to guard), which is why the two are branches of one
                // job push rather than an early `continue` - and a `continue` here is exactly what handed the
                // kernel an unwritten `h_ptr_w` and killed the first arena run with an illegal address.
                if (has_arena) {
                    // This expert's window of the arena, and whether it is already there.  A settled one is a
                    // DMA and nothing else; an unsettled one is assembled below and settles for good.
                    const size_t have = (size_t) layer * (size_t) n_expert_ + (size_t) e;
                    jobs.push_back({e, s, arena + (size_t) e * arena_bytes, kNoRing, pin_have_[have] != 0});
                } else if (direct_) {
                    // ---- **DIRECT MODE STAGES NOTHING, SO NOTHING GUARDS THIS JOB.**  The bytes are the
                    // source's own - the three slices of a GGUF-in-place pack in the mmap shape, the registered
                    // blob in the resident-arena one - so there is no ring buffer to name and none to advance.
                    // It must be `kNoRing`: `init` does not allocate the ring at all in this mode, so
                    // `ring_size` is 0 (the head advance below would divide by it, which is the SIGFPE this
                    // branch used to reach the DMA with) and `stage_ev_` is empty (an index into it is not a
                    // missing event but a read off the end of a vector).  What orders the copies against the
                    // kernel is the wave's own `wave_ev_` below, which every copy of the wave is recorded on.
                    jobs.push_back({e, s, nullptr, kNoRing, false});
                } else {
                    const size_t rj = (size_t) ((ring_head_ + (int64_t) n_ring) % (int64_t) ring_size);
                    jobs.push_back({e, s, stage_chunk_[rj], rj, false});
                    ++n_ring;
                }
            }
            slot_[se] = s;
            start[g - g0] = (int32_t) (plan_off_[(size_t) e] - e0);
            h_ptr_w[g - g0] = (unsigned long long) (uintptr_t) cache_.device_slot(s);
        }
        for (int64_t i = 0; i < wave_cap; ++i) slot_busy_[(size_t) i] = 0;
        if (n_ring > 0) ring_head_ = (ring_head_ + (int64_t) n_ring) % (int64_t) ring_size;

        // ---- THE ASSEMBLY, AND IT RUNS FOR THE ARENA TOO: an expert the arena does not hold yet is copied
        // into it here, which is what settles it.  A settled one is skipped - there is nothing to assemble.
        if (!direct_ || has_arena) {
            // ---- the assembly.  Measured at 3.4 GB/s on one thread into pinned memory, which is 0.8 s a layer
            // of a 512-token chunk - an order of magnitude more than the card spends on the same layer, so it is
            // the one part of this that has to be spread over more than the one thread that called it.
            // `copy_blob` is safe from several threads only for a source whose `transient` can be true (its
            // contract), which is exactly the GGUF-in-place case this path was built for; anything else is read
            // one at a time.
            for (size_t j = 0; j < jobs.size(); ++j) {
                if (jobs[j].ring == kNoRing) continue;              // an arena job stages nothing
                if (stage_ev_[jobs[j].ring] == nullptr) continue;   // this buffer has never been DMA'd from
                const auto tw = std::chrono::steady_clock::now();
                if (cudaEventSynchronize((cudaEvent_t) stage_ev_[jobs[j].ring]) != cudaSuccess) {
                    err = std::string("glm gpu experts: the chunk staging ring: ") +
                          cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                chunk_.wait += since(tw);
            }
            const auto th = std::chrono::steady_clock::now();
            int32_t failed = -1;
            int64_t todo = 0;
            for (const Job& jb : jobs) todo += jb.settled ? 0 : 1;
            const int64_t nthreads = src_->transient(layer, 0) ? std::min<int64_t>(par_threads_, todo) : 1;
            last_chunk_threads_ = nthreads;
            if (nthreads > 1) {
                std::atomic<size_t> at{0};
                std::vector<std::thread> pool;
                pool.reserve((size_t) nthreads);
                for (int64_t t = 0; t < nthreads; ++t)
                    pool.emplace_back([&] {
                        for (size_t j = at++; j < jobs.size(); j = at++) {
                            if (jobs[j].settled) continue;
                            if (!src_->copy_blob(layer, jobs[j].e, jobs[j].host)) {
                                failed = jobs[j].e;
                                return;
                            }
                        }
                    });
                for (std::thread& t : pool) t.join();
            } else {
                for (const Job& jb : jobs) {
                    if (jb.settled) continue;
                    if (!src_->copy_blob(layer, jb.e, jb.host)) { failed = jb.e; break; }
                }
            }
            chunk_.host += since(th);
            // The wave's assembly has joined, so every expert it copied is in the arena for good - and this is
            // the only place that bit is ever set, on the thread that will read it.
            if (has_arena && failed < 0) {
                for (const Job& jb : jobs) {
                    if (jb.settled) continue;
                    pin_have_[(size_t) layer * (size_t) n_expert_ + (size_t) jb.e] = 1;
                    ++chunk_.to_arena;
                }
            }
            if (failed >= 0) {
                err = "glm gpu experts: reading layer " + std::to_string(layer) + " expert " + std::to_string(failed);
                return false;
            }
        }
        chunk_.blobs += (int64_t) jobs.size();

        // **THE REVERSE EDGE, AND WHY IT IS NOW TWO WAVES DEEP.**  `copy` writes the slots and `cs` reads
        // them, so a wave's copies must land before its kernel - that is `wave_ev_` at the end of this block.
        // The OTHER direction is what makes the second stream legal: a copy may not overwrite a slot a running
        // kernel is reading.  It used to be the wave immediately before, which is exactly what made the two
        // streams take turns - the card idle through every copy, the bus idle through every kernel.  With the
        // slots cut in halves, this wave's copies can only touch this wave's half, and the only kernel that
        // ever reads that half is the one from two waves ago: the wave between them ran on the other half.
        // So the edge is two waves deep and wave n's kernel runs UNDER wave n+1's copies.  Without it the two
        // streams would race and a slot could be overwritten under a running kernel.
        //
        // A single-slot layer has one half and no overlap to buy, so it keeps the one-wave alternation.
        if (wave_i >= halves &&
            cudaStreamWaitEvent(copy, (cudaEvent_t) (half == 0 ? dma_ev_ : dma_ev2_), 0) != cudaSuccess) {
            err = std::string("glm gpu experts: gating the chunk's copy stream: ") +
                  cudaGetErrorString(cudaGetLastError());
            return false;
        }
        // ---- then the DMA, in plan order so the kernel that follows finds every slot it was promised.
        const strata::kernels::NativeExpertLayout& ly = lay_[(size_t) layer];
        // ---- WHAT THE ARENA DOES INSTEAD, AND WHY IT IS THE WHOLE BALL GAME.  The arena holds each expert
        // as `copy_blob` assembles it - gate, up, down back to back - which is byte for byte the layout a
        // slot wants (`up_off`/`down_off` are the same section ends, and `init` checks that against
        // `copy_blob` for every layer).  An expert already in it is ONE `cudaMemcpyAsync` out of registered
        // memory, which the driver hands to the DMA engine and returns from - no host buffer, no memcpy, no
        // bounce buffer, and the transfer runs under the wave's kernel.  That copy is `fill_slot` below,
        // the same call the ring path uses; only the source differs, and the ring's events are skipped
        // because an arena window is written once ever rather than refilled by a later wave.
        //
        // The pageable path below is what a layer with neither an arena nor `direct_` does NOT take: it is
        // the three-slice issue, kept for `STRATA_GLM_GPU_CHUNK_DIRECT` and for a box whose budget is spent.
        // Three copies, one a slice of the source's mapping - which is the whole point: no host buffer,
        // no memcpy, and the page cache is the only thing the bytes are read from.  `up_off` and
        // `down_off` are the section ends inside the assembled blob, so `[0, up_off)` is gate,
        // `[up_off, down_off)` is up and `[down_off, bytes)` is down.  (A PAGEABLE mapping, so the
        // driver bounce-buffers this and the host blocks for it - see `pin_layer`.)
        //
        // On the COPY stream, like `fill_slot` below: the caller's stream is where the kernel and the
        // plan uploads live, and a wave's copies issued there would take turns with the kernel
        // instead of running under it.  The slot is read by a kernel that `cs` only reaches after the
        // `cudaStreamWaitEvent` at the end of this wave, which is what orders the two.
        //
        // One job's copies.  0 no device slot, 1 no slices to read, 2 a CUDA failure.
        auto issue = [&](size_t j) -> int {
            uint8_t* dv = cache_.device_slot(jobs[j].s);
            if (dv == nullptr) return 0;
            const uint8_t *gs = nullptr, *us = nullptr, *ds = nullptr;
            if (direct_slices_ && src_->slices(layer, jobs[j].e, &gs, &us, &ds)) {
                const size_t gb = ly.up_off, ub = ly.down_off - ly.up_off, db = ly.bytes - ly.down_off;
                if (cudaMemcpyAsync(dv, gs, gb, cudaMemcpyHostToDevice, copy) != cudaSuccess ||
                    cudaMemcpyAsync(dv + gb, us, ub, cudaMemcpyHostToDevice, copy) != cudaSuccess ||
                    cudaMemcpyAsync(dv + gb + ub, ds, db, cudaMemcpyHostToDevice, copy) != cudaSuccess) return 2;
                return -1;
            }
            // ---- **THE RESIDENT ARENA, AND THE ONE COPY THAT WAS STILL LEFT.**  A source with no slices is
            // not a source that cannot be DMA'd: `ArenaExpertSource` holds every expert assembled - gate, up,
            // down back to back, `copy_blob`'s own layout and the slot's - in one anonymous mapping, and that
            // mapping is REGISTERED (`cudaHostRegister`, one slice a layer, `PinnedArena`; the engine's own
            // default since R2.1, and what `--mmap-experts` opts out of).  So the blob is ONE range and the
            // DMA is ONE `cudaMemcpyAsync` from page-locked RAM: the driver hands it to the copy engine and
            // returns, the host copies NOTHING, and nothing pages in under it - the arena is not page cache
            // and the OS never takes it back.  That is the `glm53-flash-offload` shape the file's header
            // describes, and it is what the staged ring could not be: there the host assembled every expert
            // into a ring and the driver read it back out, three passes over the memory controllers a byte.
            // `pinned` is asked again here rather than trusted from `init`: it is one predicate, and the cost
            // of being wrong about it is a DMA out of memory the driver may not have pinned - a wrong answer
            // that reads as a plausible token.  A blob that is not registered is not a slow DMA, it is no DMA,
            // so this does not fall back to the ring mid-request: `init` is where the tier decides, and it arms
            // `direct_` only after every layer's whole blob range has answered `pinned` (and `blob` compares
            // against `copy_blob`).  Here the honest answer is to fail the request - the reachable case is a
            // source whose registration ends mid-layer, and a tier that answers that with a token is worse than
            // one that stops.  `blob_stable` because the copy is asynchronous: the pointer must outlive this
            // call, which a transient source's `blob` does not.
            if (!src_->pinned(layer, jobs[j].e)) return 1;
            const uint8_t* bs = src_->blob_stable(layer, jobs[j].e);
            if (bs == nullptr) return 1;
            if (cudaMemcpyAsync(dv, bs, (size_t) ly.bytes, cudaMemcpyHostToDevice, copy) != cudaSuccess) return 2;
            return -1;
        };
        // **THE ISSUE IS SPREAD OVER THE POOL, AND THAT IS WORTH 1.7x ON THE FEED.**  A pageable source means
        // the driver stages every byte through its own pinned buffer ON THE CALLING THREAD, so one thread
        // issuing a wave's copies is one thread doing the whole wave's memcpy - and a layer's experts are not
        // contiguous, they sit wherever the quantizer put them across a 148 GB file.  Measured standalone
        // (bw/issue.cu, the same three slices a blob, waves of 25, 300 blobs): one thread 3.68 GB/s over the
        // scattered offsets the engine really reads, eight threads 6.10 GB/s over the same bytes - where a
        // CONTIGUOUS window gets 6.90 GB/s from one thread, which is why this went unnoticed.  The engine's
        // own figure was 3.2 GB/s a stage.  A job writes a slot no other job names and both `slices` and
        // `device_slot` are pure lookups, so the only thing shared is the stream - and a stream queues from
        // any thread.
        // The threads below ONLY issue; every shared write (the ring, the events, `owner_`) stays on this
        // thread, after they have all joined.  `first_bad` keeps the lowest failing job so the message names
        // the same expert a serial run would have stopped at.
        int bad = -1;
        if (direct_ && !has_arena) {
            std::atomic<size_t> at{0};
            std::atomic<int> first{-1};
            std::atomic<int> code{0};
            const int64_t nth = std::min<int64_t>(par_threads_, (int64_t) jobs.size());
            if (nth > 1) {
                std::vector<std::thread> pool;
                pool.reserve((size_t) nth);
                for (int64_t t = 0; t < nth; ++t)
                    pool.emplace_back([&] {
                        for (size_t j = at++; j < jobs.size(); j = at++) {
                            const int r = issue(j);
                            if (r < 0) continue;
                            int cur = first.load();
                            while (cur < 0 || (int) j < cur) {
                                if (first.compare_exchange_weak(cur, (int) j)) { code.store(r); break; }
                            }
                        }
                    });
                for (std::thread& th : pool) th.join();
            } else {
                for (size_t j = 0; j < jobs.size(); ++j) {
                    const int r = issue(j);
                    if (r >= 0) { first.store((int) j); code.store(r); break; }
                }
            }
            bad = first.load();
            if (bad >= 0) {
                const int r = code.load();
                err = r == 1 ? "glm gpu experts: layer " + std::to_string(layer) + " expert " +
                                   std::to_string(jobs[(size_t) bad].e) +
                                   ": no bytes to DMA into its slot (no slices, and its blob is not in "
                                   "registered memory)"
                           : r == 0 ? "glm gpu experts: layer " + std::to_string(layer) + " expert " +
                                          std::to_string(jobs[(size_t) bad].e) +
                                          ": its slot is not the cache's"
                                    : std::string("glm gpu experts: the chunk's slice DMA: ") +
                                          cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        for (size_t j = 0; j < jobs.size(); ++j) {
            // The DMA: from the ring buffer the assembly just wrote, from the window of the arena it wrote
            // (or found already written), or - in direct mode - issued above, from the source's own slices.
            if (!direct_ || has_arena) {
                if (!cache_.fill_slot(jobs[j].s, jobs[j].host, copy, err, (int64_t) ly.bytes)) return false;
                if (jobs[j].settled) ++chunk_.from_arena;
            }
            owner_[(size_t) jobs[j].s] = jobs[j].e;
            if (jobs[j].ring == kNoRing) continue;   // nothing staged, so no buffer to guard
            if (stage_ev_[jobs[j].ring] == nullptr &&
                cudaEventCreateWithFlags((cudaEvent_t*) &stage_ev_[jobs[j].ring], cudaEventDisableTiming) !=
                    cudaSuccess) {
                err = std::string("glm gpu experts: the chunk staging ring's events: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (cudaEventRecord((cudaEvent_t) stage_ev_[jobs[j].ring], copy) != cudaSuccess) {
                err = std::string("glm gpu experts: marking the chunk staging ring: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        // ONE EVENT A WAVE, and `cs` waits on it.  This is the whole of the ordering the two streams need: every
        // copy above is on `copy` - the ring's blobs, the arena's windows, the direct mode slices - so a kernel
        // on `cs` may read a slot only after the wave's last copy has landed.  A pinned or direct layer records
        // nothing on the ring, so this event is the only guard there - which is why it is issued unconditionally
        // rather than per job.
        if (cudaEventRecord((cudaEvent_t) wave_ev_, copy) != cudaSuccess ||
            cudaStreamWaitEvent(cs, (cudaEvent_t) wave_ev_, 0) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk wave's copy event: ") +
                  cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (verify) {
            // `cs` has just waited on `wave_ev_`, so a copy queued on `cs` now runs after every copy of this
            // wave has landed: the slots are frozen from here to the kernel, and this is what they hold.  A job
            // is a slot this wave FILLED; a resident it inherited was checked when it was filled.
            static thread_local std::vector<uint8_t> dv;
            dv.resize((size_t) ly.bytes);
            for (const Job& jb : jobs) {
                uint8_t* dvp = cache_.device_slot(jb.s);
                if (dvp == nullptr) continue;
                if (cudaMemcpyAsync(dv.data(), dvp, (size_t) ly.bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                    cudaStreamSynchronize(cs) != cudaSuccess) {
                    err = std::string("glm gpu experts: the slot verification's read-back: ") +
                          cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                ++vfy_slots;
                // Compared against exactly what this wave put in the slot's source: the arena's window on this
                // expert when the layer has one, the three source slices otherwise.  For a settled arena window
                // this reads back the DMA and nothing else - the window itself was written by `copy_blob`, the
                // same call and the same bytes the ring path stages, and STRATA_GLM_GPU_CHECK is what checks
                // those against the pool's own arithmetic.
                const uint8_t* sec[3] = {nullptr, nullptr, nullptr};
                const size_t off[3] = {0, (size_t) ly.up_off, (size_t) ly.down_off};
                size_t len[3] = {(size_t) ly.up_off, (size_t) (ly.down_off - ly.up_off),
                                 (size_t) (ly.bytes - ly.down_off)};
                const char* const names[3] = {"gate", "up", "down"};
                if (arena != nullptr) {
                    sec[0] = arena + (size_t) jb.e * (size_t) ly.bytes;
                    len[0] = (size_t) ly.bytes;
                    len[1] = 0;
                    len[2] = 0;
                } else if (!src_->slices(layer, jb.e, &sec[0], &sec[1], &sec[2])) {
                    std::fprintf(stderr, "strata glm gpu verify: layer %lld expert %d: no slices to compare\n",
                                 (long long) layer, (int) jb.e);
                    continue;
                }
                for (int si = 0; si < 3; ++si) {
                    if (sec[si] == nullptr || len[si] == 0) continue;
                    const uint8_t* got = dv.data() + off[si];
                    size_t i = 0;
                    while (i < len[si] && got[i] == sec[si][i]) ++i;
                    if (i >= len[si]) continue;
                    size_t nd = 0;
                    for (size_t q = i; q < len[si]; ++q) nd += (got[q] != sec[si][q]);
                    ++vfy_sec_bad;
                    std::fprintf(stderr, "strata glm gpu verify: layer %lld wave %lld expert %d slot %d: the %s "
                                         "section differs - first at byte %zu of %zu (card 0x%02x, file 0x%02x), "
                                         "%zu bytes differ in this section\n",
                                 (long long) layer, (long long) wave_i, (int) jb.e, (int) jb.s, names[si], i,
                                 len[si], (unsigned) got[i], (unsigned) sec[si][i], nd);
                }
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
        unsigned long long* d_ptr_w = c_ptr_ + (size_t) (r * cap_group_);
        int32_t* d_idx_w = c_idx_ + (size_t) (r * plan_idx);
        if (cudaMemcpyAsync(d_ptr_w, h_ptr_w, (size_t) nw * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                            cs) != cudaSuccess ||
            cudaMemcpyAsync(d_idx_w, start, (size_t) (nw + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(d_idx_w + cap_group_ + 1, n_groups, sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
                cudaSuccess ||
            cudaMemcpyAsync(d_idx_w + cap_group_ + 2, dst, (size_t) wave_entries * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cs) != cudaSuccess ||
            cudaMemcpyAsync(d_idx_w + cap_group_ + 2 + cap_entry_, tok, (size_t) wave_entries * sizeof(int32_t),
                            cudaMemcpyHostToDevice, cs) != cudaSuccess) {
            err = std::string("glm gpu experts: the chunk plan upload: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (verify) {
            // The plan as the DEVICE holds it, read back against the host mirror it was uploaded from - the
            // slot pointers, the group starts, the group count, and the dst/tok arrays the kernel indexes an
            // entry by.  A torn or stale plan is the other way a row can come out wrong and the only other way
            // it is invisible from here: the check on the host assumes the device plan says what the host plan
            // says.  ~30 KB a wave.
            static thread_local std::vector<unsigned long long> dp;
            static thread_local std::vector<int32_t> di, dd, dt;
            dp.resize((size_t) nw);
            di.resize((size_t) nw + 1);
            dd.resize((size_t) wave_entries);
            dt.resize((size_t) wave_entries);
            int32_t dg = 0;
            if (cudaMemcpyAsync(dp.data(), d_ptr_w, (size_t) nw * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                cudaMemcpyAsync(di.data(), d_idx_w, (size_t) (nw + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                cs) != cudaSuccess ||
                cudaMemcpyAsync(&dg, d_idx_w + cap_group_ + 1, sizeof(int32_t), cudaMemcpyDeviceToHost, cs) !=
                    cudaSuccess ||
                cudaMemcpyAsync(dd.data(), d_idx_w + cap_group_ + 2, (size_t) wave_entries * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                cudaMemcpyAsync(dt.data(), d_idx_w + cap_group_ + 2 + cap_entry_,
                                (size_t) wave_entries * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
                cudaStreamSynchronize(cs) != cudaSuccess) {
                err = std::string("glm gpu experts: the plan verification's read-back: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
            int64_t bad = 0;
            for (int64_t g = 0; g < nw; ++g)
                bad += (dp[(size_t) g] != h_ptr_w[g]) + (di[(size_t) g] != start[g]);
            bad += (di[(size_t) nw] != start[nw]) + (dg != *n_groups);
            for (int64_t j = 0; j < wave_entries; ++j) bad += (dd[(size_t) j] != dst[j]) + (dt[(size_t) j] != tok[j]);
            if (bad > 0) {
                vfy_plan_bad += bad;
                std::fprintf(stderr, "strata glm gpu verify: layer %lld wave %lld: the DEVICE plan differs from "
                                     "the host's in %lld of %lld values\n",
                             (long long) layer, (long long) wave_i, (long long) bad, (long long) (3 * nw + 3));
            }
        }
        // This slot's upload is now queued.  The event is what lets the host refill the slot `plan_ring_` waves
        // from now; without it the refill races the DMA, which is the bug the ring exists to close.
        if (plan_ring_ > 0) {
            if (plan_ev_[(size_t) r] == nullptr &&
                cudaEventCreateWithFlags((cudaEvent_t*) &plan_ev_[(size_t) r], cudaEventDisableTiming) !=
                    cudaSuccess) {
                err = std::string("glm gpu experts: the chunk plan ring's events: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (cudaEventRecord((cudaEvent_t) plan_ev_[(size_t) r], cs) != cudaSuccess) {
                err = std::string("glm gpu experts: marking the chunk plan ring: ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        strata::kernels::native_expert_grouped(lay_[(size_t) layer], d_ptr_w, d_idx_w, d_idx_w + cap_group_ + 1,
                                               d_idx_w + cap_group_ + 2, d_idx_w + cap_group_ + 2 + cap_entry_,
                                               cap_group_, cap_entry_, cxq_, cscratch_, parts_dev, stream, 0);
        // STRATA_GLM_GPU_CHUNK_TIME=2 adds one more launch of the SAME kernel over the SAME slots.  The kernel
        // writes each row of `parts` outright, so a second pass rewrites the identical values - the chunk's
        // answer is unchanged, and the two wave figures differ by exactly one kernel.  `wave` is the sum of the
        // DMA and the kernel and the print cannot tell them apart; this is what separates them:
        //   K = wave(twice) - wave(once)          D = 2 * wave(once) - wave(twice)
        if (timing == 2)
            strata::kernels::native_expert_grouped(lay_[(size_t) layer], d_ptr_w, d_idx_w, d_idx_w + cap_group_ + 1,
                                                   d_idx_w + cap_group_ + 2, d_idx_w + cap_group_ + 2 + cap_entry_,
                                                   cap_group_, cap_entry_, cxq_, cscratch_, parts_dev, stream, 0);
        // ...and the other half of the pair the comment above the DMA describes: `cs` says when this wave's
        // half is free to be overwritten, which is what the wave that takes this half NEXT - two waves on -
        // waits for.  One event a half, so the two halves gate independently.  After BOTH kernel launches, so
        // the timing double-launch is inside the gate rather than racing it.
        if (cudaEventRecord((cudaEvent_t) (half == 0 ? dma_ev_ : dma_ev2_), cs) != cudaSuccess) {
            err = std::string("glm gpu experts: marking the chunk wave's kernel: ") +
                  cudaGetErrorString(cudaGetLastError());
            return false;
        }
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
        ++wave_i;
    }
    // ---- **THE CHUNK HANDS THE LAYER'S SLOTS OVER WHOLE.**  Every one of them has been filled (a chunk streams
    // all 288 experts through ~50 slots), so the decode that follows may not take one of them as a "free" slot -
    // which is what `admit` does while `next_ < hi_`, and it fills that slot WITHOUT clearing the `slot_` entry
    // of the expert the chunk left there, so the next token that routes to that expert reads another expert's
    // bytes.  `next_` at `hi_` puts every later admission on the eviction path above, which clears what it
    // replaces.  The alternative - leaving `next_` alone - is how this arm read the wrong expert at layer 6.
    if (hi_[(size_t) layer] > lo_[(size_t) layer]) next_[(size_t) layer] = hi_[(size_t) layer];
    // The table this chunk just rewrote, against itself - `STRATA_GLM_GPU_SLOT_CHECK=1` (see `slot_table_ok`).
    static const bool slot_check = [] {
        const char* v = std::getenv("STRATA_GLM_GPU_SLOT_CHECK");
        return v != nullptr && v[0] == '1';
    }();
    if (slot_check &&
        !slot_table_ok(slot_, owner_, layer, n_expert_, lo_[(size_t) layer], hi_[(size_t) layer], err)) {
        err = "glm gpu experts: " + err;
        return false;
    }
    chunk_.entries = entries;
    chunk_.total = since(t_all);
    if (verify)
        std::fprintf(stderr, "strata glm gpu verify: layer %lld: %lld slots read back and %lld section(s) differ "
                             "from the file (%lld plan values stale) - a clean layer means the kernel read the "
                             "file's own bytes through the plan the host built\n",
                     (long long) layer, (long long) vfy_slots, (long long) vfy_sec_bad, (long long) vfy_plan_bad);
    if (timing) {
        const double gib = (double) chunk_.blobs * (double) lay_[(size_t) layer].bytes / (1024.0 * 1024.0 * 1024.0);
        // from_arena is the number that says whether the arena is doing its job: every one of those blobs the
        // host did not touch at all this chunk - the driver read registered RAM.  to_arena counts the ones this
        // chunk DID assemble, which is what settles a layer the first time it is served.
        std::fprintf(stderr, "strata glm gpu chunk: layer %lld, %lld entries in %lld waves, %lld blobs (%.2f GiB): "
                             "%.3f s host copy at %.2f GB/s, %.3f s ring wait, %.3f s plan, %.3f s arena fill, "
                             "%.3f s warm, %.3f s card wall (%.1f GiB/s into slots), %.3f s all "
                             "[par %lld transient %d%s, %lld from arena %lld into it]\n",
                     (long long) layer, (long long) entries, (long long) chunk_.waves, (long long) chunk_.blobs,
                     gib, chunk_.host, gib * 1.073741824 / std::max(0.001, chunk_.host), chunk_.wait, chunk_.plan,
                     chunk_.fill, chunk_.warm, chunk_.wave, gib / std::max(0.001, chunk_.wave), chunk_.total,
                     (long long) last_chunk_threads_,
                     src_ != nullptr && src_->transient(layer, 0) ? 1 : 0,
                     pin_map_.size() > (size_t) layer && pin_map_[(size_t) layer] != nullptr ? " arena" : "",
                     (long long) chunk_.from_arena, (long long) chunk_.to_arena);
    }
    ++chunk_calls_;
    chunk_entries_ += entries;
    last_chunk_entries_ = entries;
    served = true;
    return true;
}

}  // namespace strata::core
