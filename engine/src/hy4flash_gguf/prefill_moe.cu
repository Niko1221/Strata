// src/hy4flash_gguf/prefill_moe.cu - see include/strata/hy4flash_gguf/prefill_moe.hpp.
#include "strata/hy4flash_gguf/prefill_moe.hpp"

#include "mmq/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <numeric>

namespace MMQ = strata::hy4_flash::mmq;

namespace strata::hy4_flash {
namespace {

constexpr int kBlock = 256;

}  // namespace

/// The chunk at which batching starts to pay. Below it an expert sees 16 * n_tok / n_expert tokens on
/// average, so the batched path reads the same weights the same number of times and pays the sort and the
/// staging on top; the scratch is also zero below it, so the two cannot disagree.
const int kMoeBatchMin = 8;

namespace {

unsigned grid_for(long long n) {
    const long long g = (n + kBlock - 1) / kBlock;
    return (unsigned) (g < 1 ? 1 : (g > 65535 ? 65535 : g));
}

// Sort routed rows by expert before grouped MMQ.

/// Count assigned rows by slot because a partial tier can remap expert IDs.
__device__ inline int moe_slot_of(const int32_t* slot_of, int expert, int n_expert) {
    if (slot_of == nullptr) return expert;
    if (expert < 0 || expert >= n_expert) return -1;
    return slot_of[expert];
}

__global__ void route_count_k(const int32_t* __restrict__ ids, int n_ent, int n_expert, int n_slot,
                              const int32_t* __restrict__ slot_of, int32_t* __restrict__ cnt) {
    const int e = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_ent) return;
    const int s = moe_slot_of(slot_of, ids[e], n_expert);
    if (s >= 0 && s < n_slot) atomicAdd(&cnt[s], 1);
}

/// The prefix sum, the write cursors and the maximum, in one thread. `n_expert` is 512 here, so a parallel
/// scan would cost more in launch than it saves - and it must agree with the cursors exactly.
__global__ void route_scan_k(const int32_t* __restrict__ cnt, int n_slot, int32_t* __restrict__ bounds,
                             int32_t* __restrict__ cur, int32_t* __restrict__ max_rows) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_slot; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = cnt[x] > mx ? cnt[x] : mx;
    }
    bounds[n_slot] = pos;
    max_rows[0] = mx;
    // Count only rows assigned to this tier.
    max_rows[1] = pos;
}

/// Entry e (= token e/k, pick e%k) takes the next row of its expert's range. `src[r]` is row r's TOKEN,
/// which `mmq::quantize` reads as its gather map; `inv[e]` is entry e's ROW, which the combine reads to
/// find that entry's product.
__global__ void route_scatter_k(const int32_t* __restrict__ ids, int n_ent, int k, int n_expert, int n_slot,
                                const int32_t* __restrict__ slot_of, int32_t* __restrict__ cur,
                                int32_t* __restrict__ src, int32_t* __restrict__ inv,
                                int32_t* __restrict__ row_entry) {
    const int e = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_ent) return;
    // Skip invalid expert IDs and experts absent from this tier.
    const int s = moe_slot_of(slot_of, ids[e], n_expert);
    if (s < 0 || s >= n_slot) {
        inv[e] = -1;
        return;
    }
    const int r = atomicAdd(&cur[s], 1);
    src[r] = e / k;
    inv[e] = r;
    if (row_entry != nullptr) row_entry[r] = e;
}

/// Apply clamped SwiGLU to the separate gate and up projection buffers.
__global__ __launch_bounds__(kBlock) void swiglu_clamp_rows_k(const float* g,
                                                              const float* __restrict__ u,
                                                              float* h, long long n,
                                                              float limit) {
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        float gi = g[i], ui = u[i];
        gi = gi < limit ? gi : limit;
        ui = ui < -limit ? -limit : (ui > limit ? limit : ui);
        h[i] = __fdividef(gi, 1.f + __expf(-gi)) * ui;
    }
}

/// Combine the expert outputs in routing-rank order.
__global__ __launch_bounds__(kBlock) void moe_scatter_parts_k(const float* __restrict__ d,
                                                              const int32_t* __restrict__ inv, int width,
                                                              int n_ent, float* __restrict__ parts) {
    const long long n = (long long) width * n_ent;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int c = (int) (i % width);
        const int e = (int) (i / width);
        const int32_t r = inv[e];
        if (r < 0) continue;
        parts[(long long) e * width + c] = d[(long long) r * width + c];
    }
}

// Scatter only rows produced by this batch. The destination remains the original (token, pick),
// so all batches are combined once in exactly the existing top-k order.
__global__ void moe_scatter_active_k(const float* d, const int32_t* row_entry,
                                      int width, int n_rows, float* parts) {
    const long long n = (long long) width * n_rows;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int row = (int) (i / width), col = (int) (i % width);
        parts[(long long) row_entry[row] * width + col] = d[i];
    }
}

__global__ __launch_bounds__(kBlock) void moe_combine_rows_k(const float* __restrict__ d,
                                                             const int32_t* __restrict__ inv,
                                                             const float* __restrict__ rw, int k, int width,
                                                             int n_tok, float* __restrict__ latent,
                                                             int accumulate) {
    const long long n = (long long) width * n_tok;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x) {
        const int c = (int) (i % width);
        const int t = (int) (i / width);
        const int32_t* iv = inv + (long long) t * k;
        const float* w = rw + (long long) t * k;
        // Negative inverse-map entries have no product in the current batch.
        float sum = 0.f;
        bool first = true;
        for (int j = 0; j < k; ++j) {
            if (iv[j] < 0) continue;
            const float v = d[(long long) iv[j] * width + c];
            if (first) {
                sum = v * w[j];
                first = false;
            } else {
                sum = __fadd_rn(sum, __fmul_rn(v, w[j]));
            }
        }
        // Accumulate partial-tier batches by their original token and rank.
        if (accumulate) latent[i] += sum;
        else latent[i] = sum;
    }
}

/// One per thread. `MMQ::Context` holds llama.cpp's stream-k fixup pool, and the pinned word is the only
/// thing the host reads back.
struct PlanStorage {
    void* dev = nullptr;
    void* host = nullptr;
    void* xq = nullptr;
    std::size_t bytes = 0, qbytes = 0;
    cudaEvent_t consumed = nullptr;
    bool live = false;
    ~PlanStorage() {
        if (live) cudaEventSynchronize(consumed);
        if (dev) cudaFree(dev);
        if (host) cudaFreeHost(host);
        if (xq) cudaFree(xq);
        if (consumed) cudaEventDestroy(consumed);
    }
};

struct MoeCtx {
    PlanStorage plan;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t* pinned = nullptr;
    ~MoeCtx() {
        if (pinned != nullptr) cudaFreeHost(pinned);
    }
};

std::unique_ptr<MoeCtx>& ctx_slot() {
    static thread_local std::unique_ptr<MoeCtx> c;
    return c;
}

MoeCtx* moe_ctx() {
    std::unique_ptr<MoeCtx>& s = ctx_slot();
    if (s == nullptr) {
        s.reset(new MoeCtx());
        s->mctx = std::make_unique<MMQ::Context>();
        if (cudaHostAlloc((void**) &s->pinned, 64, cudaHostAllocDefault) != cudaSuccess) {
            s.reset();
            return nullptr;
        }
    }
    return s.get();
}

bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_MOE_MMQ");
        return v == nullptr || *v != '0';
    }();
    return on;
}

/// Said once: `mmq::supported` is false for the K-quants without STRATA_MMQ_KQ=1 and the DSO beside the
/// executable, and a silent fall back to the per-token loop is an 8x slowdown with no visible cause.
void say_once(const char* why) {
    static bool said = false;
    if (said) return;
    said = true;
    std::fprintf(stderr, "hy4_flash: the prompt's routed experts stay on the per-token matvecs - %s\n", why);
}

std::size_t align16(std::size_t n) { return (n + 15) & ~(std::size_t) 15; }

/// Choose the MMQ tile width from the typical expert's row count.
std::int64_t tile_opt_rows(std::int64_t total_rows, int n_expert, std::int64_t max_rows) {
    // 0 = off (the tile follows `max_rows`, the old behaviour); 1 = the average expert's rows; a value
    // above 1 is used as the row count directly, which is what the sweep below was run with.
    static const int mode = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_MOE_TILE_OPT");
        return v != nullptr ? std::atoi(v) : -1;   // -1: the measured rule below
    }();
    if (mode == 0 || n_expert <= 0 || total_rows <= 0) return 0;
    std::int64_t want;
    if (mode > 1) {
        want = mode;
    } else if (mode == 1) {
        want = (total_rows + n_expert - 1) / n_expert;
    } else {
        // Round the tile estimate to a supported size in the range 32–128.
        want = (8 * total_rows + 3 * n_expert - 1) / (3 * n_expert);
        want = (want + 7) & ~(std::int64_t) 7;
        if (want < 32) want = 32;
        if (want > 128) want = 128;
    }
    if (want < 8) want = 8;
    return want < max_rows ? want : 0;
}

}  // namespace

bool moe_mmq_usable(int gate_type, int up_type, int down_type, int expert_dim, int expert_ff, int n_tok) {
    if (!enabled()) return false;
    // Below this there is nothing to batch: an expert sees 16 * n_tok / n_expert tokens on average, so at a
    // small chunk the batched path reads the same weights the same number of times and pays the sort and
    // the staging on top.
    if (n_tok <= kMoeBatchMin) return false;
    if (!MMQ::built()) { say_once("this build has no MMQ kernels"); return false; }
    if (!MMQ::supported(gate_type) || !MMQ::supported(up_type) || !MMQ::supported(down_type)) {
        say_once("MMQ does not cover these expert types here (K-quants need STRATA_MMQ_KQ=1 and "
                 "libstrata_kq.so beside the executable)");
        return false;
    }
    if (!MMQ::fits(gate_type, expert_ff) || !MMQ::fits(up_type, expert_ff) ||
        !MMQ::fits(down_type, expert_dim)) {
        say_once("no MMQ tile for these shapes fits this card's shared memory");
        return false;
    }
    return true;
}

std::size_t latent_moe_mmq_scratch_bytes(int expert_dim, int expert_ff, int n_expert, int k, int n_tok, int row_cap) {
    if (expert_dim <= 0 || expert_ff <= 0 || n_expert <= 0 || k <= 0 || n_tok <= 0) return 0;
    // Match the execution threshold when reserving scratch.
    if (n_tok <= kMoeBatchMin) return 0;
    const std::size_t ent = (std::size_t) n_tok * (std::size_t) k;
    const std::size_t work = row_cap > 0 ? std::min(ent, (std::size_t) row_cap) : ent;
    std::size_t b = 0;
    b += align16((std::size_t) n_expert * 4);        // cnt
    b += align16((std::size_t) n_expert * 4);        // cur
    b += align16((std::size_t) (n_expert + 1) * 4);  // bounds
    b += align16(16);                                // max_rows
    b += align16(ent * 4);                           // src: row -> token
    b += align16(ent * 4);                           // inv: entry -> row
    b += align16(ent * 4);                           // the active row -> (token, pick) map
    b += align16(ent * 4);                           // the identity row map MMQ writes through
    b += align16(MMQ::q8_bytes((std::int64_t) work, expert_dim));   // the activation rows
    b += align16(MMQ::q8_bytes((std::int64_t) work, expert_ff));    // the SwiGLU rows
    b += align16(work * (std::size_t) expert_ff * 4);   // the gate product, which the SwiGLU overwrites
    b += align16(work * (std::size_t) expert_ff * 4);   // the up product
    b += align16(work * (std::size_t) expert_dim * 4);  // the down product
    return b + 256;
}

int moe_group_rows(int k, int n_tok) {
    static const int cap = [] {
        const char* p = std::getenv("STRATA_HY4_FLASH_PREFILL_ROW_PLAN");
        const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_GROUP_ROWS");
        return p != nullptr && std::atoi(p) != 0 && v != nullptr ? std::max(0, std::atoi(v)) : 0;
    }();
    // At least a complete expert's rows fit; the router chooses each expert at most once per token.
    return cap > 0 && n_tok > kMoeBatchMin ? std::min(k * n_tok, std::max(cap, n_tok)) : 0;
}

std::vector<MoeBatchPlan> moe_prepare_batches(const std::vector<int32_t>& ids,
    const std::vector<MoeBatchSpec>& specs, int n_expert, int k, int n_tok,
    const float* routed_in, int gate_type, int expert_dim, void* stream) {
    const auto st = static_cast<cudaStream_t>(stream);
    const int ent = n_tok * k;
    if (n_tok <= 0 || k <= 0 || n_expert <= 0 || (int) ids.size() != ent)
        throw std::runtime_error("MoE plan: invalid routing shape");
    MoeCtx* mc = moe_ctx();
    if (!mc) throw std::runtime_error("MoE plan: no context");
    PlanStorage& store = mc->plan;
    if (store.live && cudaEventSynchronize(store.consumed) != cudaSuccess)
        throw std::runtime_error("MoE plan: previous consumer failed");
    store.live = false;
    if (!store.consumed && cudaEventCreateWithFlags(&store.consumed, cudaEventDisableTiming) != cudaSuccess)
        throw std::runtime_error("MoE plan: no completion event");
    std::vector<std::vector<int32_t>> by_expert((std::size_t) n_expert);
    for (int i = 0; i < ent; ++i) {
        if (ids[i] < 0 || ids[i] >= n_expert) throw std::runtime_error("MoE plan: expert ID out of range");
        by_expert[(std::size_t) ids[i]].push_back(i);
    }
    std::vector<int32_t> metadata, row_entry, row_src, inverse((std::size_t) ent, -1);
    std::vector<std::size_t> offsets, absolute_offsets, starts;
    std::vector<int32_t> global_to_local;
    std::vector<MoeBatchPlan> plans;
    for (const auto& spec : specs) {
        if (spec.n_slots <= 0 || (int) spec.slot_of.size() != n_expert)
            throw std::runtime_error("MoE plan: invalid slot map");
        std::vector<int32_t> expert_at((std::size_t) spec.n_slots, -1);
        for (int e = 0; e < n_expert; ++e) if (spec.slot_of[e] >= 0) {
            const int slot = spec.slot_of[e];
            if (slot >= spec.n_slots || expert_at[(std::size_t) slot] >= 0)
                throw std::runtime_error("MoE plan: invalid or duplicate slot");
            expert_at[(std::size_t) slot] = e;
        }
        while (metadata.size() % 4) metadata.push_back(0);
        offsets.push_back(metadata.size()); starts.push_back(row_entry.size());
        MoeBatchPlan plan; plan.n_slots = spec.n_slots;
        for (int slot = 0; slot < spec.n_slots; ++slot) {
            metadata.push_back(plan.n_rows);
            const int e = expert_at[(std::size_t) slot];
            if (e < 0) continue;
            const auto& entries = by_expert[(std::size_t) e];
            plan.max_rows = std::max(plan.max_rows, (int) entries.size());
            for (int entry : entries) {
                if (inverse[(std::size_t) entry] >= 0) throw std::runtime_error("MoE plan: duplicate route entry");
                inverse[(std::size_t) entry] = (int32_t) row_entry.size();
                row_entry.push_back(entry); row_src.push_back(entry / k);
                global_to_local.push_back(plan.n_rows); ++plan.n_rows;
            }
        }
        metadata.push_back(plan.n_rows);
        while (metadata.size() % 4) metadata.push_back(0);
        absolute_offsets.push_back(metadata.size());
        for (int slot = 0; slot <= spec.n_slots; ++slot)
            metadata.push_back(metadata[offsets.back() + (std::size_t) slot] + (int32_t) starts.back());
        plan.total_rows = ent; plans.push_back(plan);
    }
    struct ChildView { std::size_t bounds, absolute, row; };
    std::vector<std::vector<ChildView>> child_views(plans.size());
    const int capacity = moe_group_rows(k, n_tok);
    for (std::size_t pi = 0; pi < plans.size(); ++pi) {
        auto& plan = plans[pi]; plan.workspace_rows = capacity;
        if (capacity <= 0 || plan.n_rows <= capacity) continue;
        const std::size_t bo = offsets[pi];
        for (int first = 0; first < plan.n_slots;) {
            while (first < plan.n_slots && metadata[bo + first] == metadata[bo + first + 1]) ++first;
            if (first == plan.n_slots) break;
            const int row0 = metadata[bo + first];
            int end = first + 1;
            if (metadata[bo + end] - row0 > capacity)
                throw std::runtime_error("MoE group: a single expert exceeds row capacity");
            while (end < plan.n_slots && metadata[bo + end + 1] - row0 <= capacity) ++end;
            int last = end;
            while (last > first + 1 && metadata[bo + last] == metadata[bo + last - 1]) --last;
            MoeBatchPlan child;
            child.n_slots = plan.n_slots; child.n_rows = metadata[bo + last] - row0;
            child.weight_base = 0; child.workspace_rows = capacity; child.total_rows = ent;
            // Preserve the parent's MMQ launch geometry. Shrinking n_slots/max_rows changes stream-K
            // partitions and therefore FP32 accumulation order even when each expert is computed whole.
            child.max_rows = plan.max_rows;
            child.tile_rows_hint = tile_opt_rows(plan.n_rows, plan.n_slots, plan.max_rows);
            while (metadata.size() % 4) metadata.push_back(0);
            ChildView view{metadata.size(), 0, starts[pi] + (std::size_t) row0};
            for (int slot = 0; slot <= plan.n_slots; ++slot)
                metadata.push_back(std::clamp(metadata[bo + slot] - row0, 0, child.n_rows));
            while (metadata.size() % 4) metadata.push_back(0);
            view.absolute = metadata.size();
            for (int slot = 0; slot <= plan.n_slots; ++slot)
                metadata.push_back(std::clamp(metadata[bo + slot] - row0, 0, child.n_rows) +
                                   row0 + (int32_t) starts[pi]);
            for (int row = 0; row < child.n_rows; ++row) global_to_local[view.row + (std::size_t) row] = row;
            plan.children.push_back(child); child_views[pi].push_back(view);
            first = end;
        }
    }
    if ((int) row_entry.size() != ent || std::find(inverse.begin(), inverse.end(), -1) != inverse.end())
        throw std::runtime_error("MoE plan: routed entries not fully covered");
    while (metadata.size() % 4) metadata.push_back(0);
    const auto entry_offset = metadata.size(); metadata.insert(metadata.end(), row_entry.begin(), row_entry.end());
    const auto src_offset = metadata.size(); metadata.insert(metadata.end(), row_src.begin(), row_src.end());
    const auto inv_offset = metadata.size(); metadata.insert(metadata.end(), inverse.begin(), inverse.end());
    const auto quant_ids_offset = metadata.size();
    metadata.insert(metadata.end(), global_to_local.begin(), global_to_local.end());
    const auto ident_offset = metadata.size();
    for (int i = 0; i < ent; ++i) metadata.push_back(i);
    const std::size_t need = metadata.size() * sizeof(int32_t);
    if (need > store.bytes) {
        if (store.dev) cudaFree(store.dev);
        if (store.host) cudaFreeHost(store.host);
        store.dev = store.host = nullptr; store.bytes = 0;
        if (cudaMalloc(&store.dev, need) != cudaSuccess ||
            cudaHostAlloc(&store.host, need, cudaHostAllocDefault) != cudaSuccess)
            throw std::runtime_error("MoE plan: metadata allocation failed");
        store.bytes = need;
    }
    std::memcpy(store.host, metadata.data(), need);
    if (cudaMemcpyAsync(store.dev, store.host, need, cudaMemcpyHostToDevice, st) != cudaSuccess)
        throw std::runtime_error("MoE plan: metadata upload failed");
    // Always record a retirement point, including the exceptional path before MMQ is launched.
    if (cudaEventRecord(store.consumed, st) != cudaSuccess)
        throw std::runtime_error("MoE plan: metadata retirement failed");
    store.live = true;
    auto* data = static_cast<int32_t*>(store.dev);
    static const bool reuse = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_QUANT_REUSE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (reuse) {
        const std::size_t qb = MMQ::q8_bytes(ent, expert_dim);
        if (qb > store.qbytes) {
            if (store.xq) cudaFree(store.xq);
            store.xq = nullptr; store.qbytes = 0;
            if (cudaMalloc(&store.xq, qb) != cudaSuccess) throw std::runtime_error("MoE plan: quant allocation failed");
            store.qbytes = qb;
        }
        MMQ::quantize_scatter(routed_in, data + inv_offset, data + src_offset, store.xq,
                             gate_type, expert_dim, expert_dim, n_tok, k, st);
    }
    for (std::size_t i = 0; i < plans.size(); ++i) {
        plans[i].bounds = data + offsets[i]; plans[i].entries = data + entry_offset + starts[i];
        plans[i].src = data + src_offset + starts[i]; plans[i].identity = data + ident_offset;
        plans[i].absolute_bounds = data + absolute_offsets[i];
        plans[i].quant_ids = data + quant_ids_offset;
        if (reuse) plans[i].xq = store.xq; // MMQ Q8 rows are K-block-major, not independently sliceable rows.
        for (std::size_t j = 0; j < plans[i].children.size(); ++j) {
            auto& child = plans[i].children[j]; const auto& view = child_views[i][j];
            child.bounds = data + view.bounds; child.absolute_bounds = data + view.absolute;
            child.entries = data + entry_offset + view.row; child.src = data + src_offset + view.row;
            child.identity = data + ident_offset; child.quant_ids = data + quant_ids_offset;
            if (reuse) child.xq = store.xq;
        }
    }
    return plans;
}

void moe_finish_batches(void* stream) {
    MoeCtx* mc = moe_ctx();
    if (mc == nullptr || cudaEventRecord(mc->plan.consumed, static_cast<cudaStream_t>(stream)) != cudaSuccess)
        throw std::runtime_error("MoE plan: cannot retire batch consumers");
    mc->plan.live = true;
}

bool latent_moe_experts_mmq(const float* routed_in, const ExpertGroup& grp, int resident_base,
                            int gate_type, int up_type, int down_type, int expert_dim, int expert_ff,
                            int n_expert, int k, int n_tok, const int32_t* ids, const float* rw,
                            float swiglu_clamp, float* latent, void* scratch, std::size_t scratch_bytes,
                            void* stream, const int32_t* slot_of, int n_slot, bool accumulate,
                            float* parts_out, const MoeBatchPlan* plan) {
    if (!moe_mmq_usable(gate_type, up_type, down_type, expert_dim, expert_ff, n_tok)) return false;
    // Use contiguous layer weights for full residency and indexed slots for a partial tier.
    if (plan != nullptr && !plan->children.empty()) {
        for (const auto& child : plan->children) {
            ExpertGroup group = grp;
            group.gate = static_cast<const char*>(group.gate) + (std::size_t) child.weight_base * group.gate_stride;
            group.up = static_cast<const char*>(group.up) + (std::size_t) child.weight_base * group.up_stride;
            group.down = static_cast<const char*>(group.down) + (std::size_t) child.weight_base * group.down_stride;
            if (!latent_moe_experts_mmq(routed_in, group, 0, gate_type, up_type, down_type,
                expert_dim, expert_ff, n_expert, k, n_tok, ids, rw, swiglu_clamp, latent,
                scratch, scratch_bytes, stream, slot_of, child.n_slots, accumulate, parts_out, &child)) return false;
        }
        return true;
    }
    const bool by_slot = slot_of != nullptr;
    if (routed_in == nullptr || ids == nullptr || rw == nullptr || latent == nullptr || scratch == nullptr ||
        stream == nullptr || resident_base < 0 || grp.gate == nullptr || grp.up == nullptr ||
        grp.down == nullptr)
        return false;
    const int n_idx = by_slot ? n_slot : n_expert;
    // The ring may hold fewer slots than there are experts. Every mapped slot in
    // the current batch must be within the launch bounds.
    if (n_idx <= 0) return false;
    const int row_cap = plan != nullptr ? plan->workspace_rows : 0;
    if (scratch_bytes < latent_moe_mmq_scratch_bytes(expert_dim, expert_ff, n_idx, k, n_tok, row_cap))
        return false;
    MoeCtx* mc = moe_ctx();
    if (mc == nullptr) return false;

    const auto st = static_cast<cudaStream_t>(stream);
    const std::size_t ent = (std::size_t) n_tok * (std::size_t) k;
    const std::size_t work = row_cap > 0 ? std::min(ent, (std::size_t) row_cap) : ent;
    char* p = static_cast<char*>(scratch);
    const auto take = [&p](std::size_t bytes) {
        void* at = p;
        p += align16(bytes);
        return at;
    };
    auto* cnt = (int32_t*) take((std::size_t) n_idx * 4);
    auto* cur = (int32_t*) take((std::size_t) n_idx * 4);
    auto* bounds = (int32_t*) take((std::size_t) (n_idx + 1) * 4);
    auto* d_max = (int32_t*) take(16);
    auto* src = (int32_t*) take(ent * 4);
    auto* inv = (int32_t*) take(ent * 4);
    auto* row_entry = (int32_t*) take(ent * 4);
    auto* ident = (int32_t*) take(ent * 4);
    void* xq = take(MMQ::q8_bytes((std::int64_t) work, expert_dim));
    void* hq = take(MMQ::q8_bytes((std::int64_t) work, expert_ff));
    auto* g_out = (float*) take(work * (std::size_t) expert_ff * 4);
    auto* u_out = (float*) take(work * (std::size_t) expert_ff * 4);
    // SwiGLU can overwrite gate because each row is consumed at the same index.
    float* h_out = g_out;
    auto* d_out = (float*) take(work * (std::size_t) expert_dim * 4);

    static const bool active_scatter = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_PREFILL_ROW_SCATTER");
        return v != nullptr && std::atoi(v) != 0;
    }();
    std::int64_t max_rows = 0, n_row = 0;
    if (plan != nullptr) {
        if (plan->n_slots != n_idx || parts_out == nullptr)
            throw std::runtime_error("MoE plan: batch extent mismatch");
        bounds = plan->bounds; src = plan->src; ident = plan->identity; row_entry = plan->entries;
        max_rows = plan->max_rows; n_row = plan->n_rows;
        if (plan->xq) xq = plan->xq;
    } else {
    // ---- sort the (token, pick) entries by expert, on the device
    if (cudaMemsetAsync(cnt, 0, (std::size_t) n_idx * 4, st) != cudaSuccess) return false;
    route_count_k<<<grid_for((long long) ent), kBlock, 0, st>>>(ids, (int) ent, n_expert, n_idx, slot_of, cnt);
    route_scan_k<<<1, 32, 0, st>>>(cnt, n_idx, bounds, cur, d_max);
    route_scatter_k<<<grid_for((long long) ent), kBlock, 0, st>>>(ids, (int) ent, k, n_expert, n_idx, slot_of,
                                                                 cur, src, inv, active_scatter ? row_entry : nullptr);
    MMQ::iota(ident, (std::int64_t) ent, st);

    // Read back the maximum rows per expert to size the MMQ grid.
    if (cudaMemcpyAsync(mc->pinned, d_max, 8, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;
    max_rows = (std::int64_t) mc->pinned[0];
    n_row = (std::int64_t) mc->pinned[1];
    if (max_rows <= 0 || n_row <= 0) return false;  // nothing landed: let the caller take the known-good path

    // ---- the activation rows, gathered by `src` as they are quantised


    }
    if (n_row <= 0 || max_rows <= 0 || (std::size_t) n_row > work) return false;
    if (plan == nullptr || plan->xq == nullptr)
        MMQ::quantize(routed_in, src, xq, gate_type, expert_dim, expert_dim, n_row, st);

    // ---- gate and up: two Products over the tier IN PLACE, one per plane. `resident_base` makes the
    // router's own expert id the plane offset (prefill_moe.hpp).
    const auto product = [&](const void* w, int type, std::int64_t rows, std::int64_t cols,
                             std::size_t stride, const void* x, float* dst) {
        MMQ::Product pr;
        pr.w = w;
        pr.type = type;
        pr.w_rows = rows;
        pr.w_cols = cols;
        pr.expert_bytes = stride;
        pr.n = n_idx;
        pr.xq = x;
        pr.bounds = bounds;
        pr.ids = ident;
        pr.total_rows = n_row;
        if (plan != nullptr && plan->xq != nullptr && x == plan->xq) {
            // Shared quantization uses the complete chunk's K-block-major row stride. Absolute bounds
            // select this group's rows, while ids write its gate/up outputs into local scratch rows.
            pr.bounds = plan->absolute_bounds;
            pr.ids = plan->quant_ids;
            pr.total_rows = plan->total_rows;
        }
        pr.max_rows = max_rows;
        // Size the grid for the busiest expert but choose tile width for typical occupancy.
        pr.opt_rows = plan != nullptr && plan->tile_rows_hint >= 0 ? plan->tile_rows_hint
                      : tile_opt_rows(n_row, n_idx, max_rows);
        pr.dst = dst;
        pr.ld_dst = rows;
        mc->mctx->run(pr, st);
    };
    // BY SLOT: the plane base is the pool's, not the block's - `slot_of` already returns a pool-local slot,
    // so adding `resident_base` again would shift every address by a block.
    const std::size_t base_g = by_slot ? 0 : (std::size_t) resident_base * grp.gate_stride;
    const std::size_t base_u = by_slot ? 0 : (std::size_t) resident_base * grp.up_stride;
    const std::size_t base_d = by_slot ? 0 : (std::size_t) resident_base * grp.down_stride;
    product(static_cast<const char*>(grp.gate) + base_g, gate_type,
            expert_ff, expert_dim, grp.gate_stride, xq, g_out);
    product(static_cast<const char*>(grp.up) + base_u, up_type,
            expert_ff, expert_dim, grp.up_stride, xq, u_out);

    // Run SwiGLU and the down projection only for rows present in this batch.
    swiglu_clamp_rows_k<<<grid_for((long long) (n_row * (std::size_t) expert_ff)), kBlock, 0, st>>>(
        g_out, u_out, h_out, (long long) (n_row * (std::size_t) expert_ff), swiglu_clamp);
    MMQ::quantize(h_out, nullptr, hq, down_type, expert_ff, expert_ff, n_row, st);
    product(static_cast<const char*>(grp.down) + base_d, down_type,
            expert_dim, expert_ff, grp.down_stride, hq, d_out);

    // ---- back to token order. `parts_out` means the caller will run the original combine itself once
    // every batch has scattered; see `moe_scatter_parts_k` on why that is not optional.
    if (parts_out != nullptr) {
        if (active_scatter || plan != nullptr)
            moe_scatter_active_k<<<grid_for((long long) expert_dim * n_row), kBlock, 0, st>>>(
                d_out, row_entry, expert_dim, (int) n_row, parts_out);
        else
            moe_scatter_parts_k<<<grid_for((long long) expert_dim * (std::size_t) ent), kBlock, 0, st>>>(
                d_out, inv, expert_dim, (int) ent, parts_out);
    } else {
        moe_combine_rows_k<<<grid_for((long long) expert_dim * n_tok), kBlock, 0, st>>>(
            d_out, inv, rw, k, expert_dim, n_tok, latent, accumulate ? 1 : 0);
    }

    if (plan != nullptr) moe_finish_batches(stream);
    return cudaGetLastError() == cudaSuccess;
}

// ---- the attention's per-head projections

namespace {

/// `bounds[g] = g * n_tok`, the group ranges of a batch where every group has the same row count.
__global__ void fill_bounds_k(int32_t* __restrict__ bounds, int n, int n_tok) {
    const int g = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (g < n) bounds[g] = g * n_tok;
}

}  // namespace

namespace {

/// STRATA_HY4_FLASH_GROUPED_MMQ: the per-head batched product, separately from the experts', so the two can be
/// A/B'd apart. They are different shapes and nothing says they have to win or lose together.
bool grouped_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_GROUPED_MMQ");
        return v == nullptr || *v != '0';
    }();
    return on;
}

}  // namespace

bool grouped_mmq_usable(int ggml_type, int n_out, int n_tok) {
    if (!grouped_enabled() || n_tok <= kMoeBatchMin) return false;
    if (!MMQ::built()) { say_once("the per-head products: this build has no MMQ"); return false; }
    if (!MMQ::supported(ggml_type)) {
        static int said = 0;
        if (said++ == 0)
            std::fprintf(stderr, "hy4_flash: the per-head products stay on the matvecs - MMQ does not cover "
                                 "ggml type %d here\n", ggml_type);
        return false;
    }
    if (!MMQ::fits(ggml_type, n_out)) {
        static int said2 = 0;
        if (said2++ == 0)
            std::fprintf(stderr, "hy4_flash: the per-head products stay on the matvecs - no MMQ tile for "
                                 "type %d x %d rows fits this card\n", ggml_type, n_out);
        return false;
    }
    return true;
}

std::size_t grouped_mmq_scratch_bytes(int ggml_type, int n_in, int n_group, int n_tok) {
    if (n_in <= 0 || n_group <= 0 || n_tok <= 0) return 0;
    // `n_out` is not known here, so the sizing checks what IS - the switch, the token count, the build and
    // the type - and a shape whose tile does not fit is refused at the launch, which then costs only the
    // scratch. Sizing on `grouped_mmq_usable` would need n_out and would make the two disagree.
    if (!grouped_enabled() || n_tok <= kMoeBatchMin || !MMQ::built()) return 0;
    if (!MMQ::supported(ggml_type)) {
        // Report an unsupported projection when routing diagnostics are enabled.
        static int said = 0;
        if (said++ == 0)
            std::fprintf(stderr, "hy4_flash: a per-head product stays on the matvecs - MMQ has no instance "
                                 "for ggml type %d in this build (Q5_0 needs STRATA_ORCA_Q4KS_MMQ=ON)\n",
                         ggml_type);
        return 0;
    }
    const std::size_t rows = (std::size_t) n_group * (std::size_t) n_tok;
    std::size_t b = 0;
    b += align16((std::size_t) (n_group + 1) * 4);   // bounds
    b += align16(rows * 4);                          // the identity row map
    b += align16(MMQ::q8_bytes((std::int64_t) rows, n_in));
    return b + 128;
}

bool grouped_product_mmq(const void* w, int ggml_type, std::size_t head_stride, const float* x, int n_in,
                         int n_out, int n_group, int n_tok, float* y, void* scratch,
                         std::size_t scratch_bytes, void* stream) {
    if (!grouped_mmq_usable(ggml_type, n_out, n_tok)) return false;
    if (w == nullptr || x == nullptr || y == nullptr || scratch == nullptr || stream == nullptr) return false;
    const std::size_t need = grouped_mmq_scratch_bytes(ggml_type, n_in, n_group, n_tok);
    if (need == 0 || scratch_bytes < need) return false;
    MoeCtx* mc = moe_ctx();
    if (mc == nullptr) return false;

    const auto st = static_cast<cudaStream_t>(stream);
    const std::size_t rows = (std::size_t) n_group * (std::size_t) n_tok;
    char* p = static_cast<char*>(scratch);
    const auto take = [&p](std::size_t bytes) {
        void* at = p;
        p += align16(bytes);
        return at;
    };
    auto* bounds = (int32_t*) take((std::size_t) (n_group + 1) * 4);
    auto* ident = (int32_t*) take(rows * 4);
    void* xq = take(MMQ::q8_bytes((std::int64_t) rows, n_in));

    fill_bounds_k<<<grid_for(n_group + 1), kBlock, 0, st>>>(bounds, n_group + 1, n_tok);
    MMQ::iota(ident, (std::int64_t) rows, st);
    // The activation is ALREADY the rows MMQ wants: `[n_in, n_tok, n_group]` head-major means row
    // (g, t) sits at `x + (g * n_tok + t) * n_in`, so no gather map and no reordering.
    MMQ::quantize(x, nullptr, xq, ggml_type, n_in, n_in, (std::int64_t) rows, st);

    MMQ::Product pr;
    pr.w = w;
    pr.type = ggml_type;
    pr.w_rows = n_out;
    pr.w_cols = n_in;
    pr.expert_bytes = head_stride;
    pr.n = n_group;
    pr.xq = xq;
    pr.bounds = bounds;
    pr.ids = ident;
    pr.total_rows = (std::int64_t) rows;
    pr.max_rows = n_tok;   // every group has the same count, so this is exact and needs no readback
    pr.dst = y;
    pr.ld_dst = n_out;
    mc->mctx->run(pr, st);
    return cudaGetLastError() == cudaSuccess;
}

void prefill_moe_release() { ctx_slot().reset(); }

}  // namespace strata::hy4_flash
