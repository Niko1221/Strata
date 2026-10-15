#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstring>

namespace strata::core {
namespace {
// The reduce metadata follows the window: row[cap] int32s then weight[cap] floats, one entry per
// window row (cap = MAXT * k).  The kernels get the geometry as launch arguments because a
// __global__ cannot read instance members.
static_assert(strata::kernels::cpu::MAXT == 8 && strata::kernels::cpu::H == 2560,
              "the defaults below are Flash-Next's geometry (8 token window, 2560-wide rows)");
// Flash-Next's arithmetic, so a helper opened with the defaults changes no number: cap = 8 * 10 = 80,
// metadata = 80 * 4 + 80 * 4 = 640 B (the old sizeof(ReduceMeta)), the pinned buffer =
// (2560 * 8 + 80) * 4 = 82240 B (the old MAXT * (H + K) * 4), one peer's sum = 2560 * 8 * 4 = 81920 B,
// the reduce grid ((2560 + 255) / 256, tokens) and copy_cpu_rows' row stride 2560 / 4 float4s.
static_assert(strata::kernels::cpu::MAXT * 10 * sizeof(int32_t) + strata::kernels::cpu::MAXT * 10 * sizeof(float)
                  == 2 * 80 * 4,
              "Flash-Next's reduce metadata must stay the old sizeof(ReduceMeta) = 640 B");
static_assert((strata::kernels::cpu::H * strata::kernels::cpu::MAXT + strata::kernels::cpu::MAXT * 10)
                      * sizeof(float) == strata::kernels::cpu::MAXT
                          * (strata::kernels::cpu::H + 10) * sizeof(float),
              "Flash-Next's pinned sum+mask buffer must stay the old MAXT * (H + K) floats");

bool check(cudaError_t status, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("remote expert decode optimization: ") + cudaGetErrorString(status);
    return false;
}

__global__ void reduce_experts(const float* parts, const int32_t* rows, const float* weights, float* sum,
                               int H, int K) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int i = t * K + j, r = rows[i];
        if (r >= 0) v += parts[(int64_t) r * H + c] * weights[i];
    }
    sum[(int64_t) t * H + c] = v;
}

// Like the upstream mapped-row copy, also skip rows returned in the remote sum.
__global__ void copy_cpu_rows(float4* dst, const volatile float4* src, const int32_t* mask,
                              const int32_t* hit_rows, const int32_t* count, int H) {
    const int row = blockIdx.x;
    __shared__ int skip;
    if (threadIdx.x == 0) {
        int s = mask[row];
        for (int i = 0; i < *count; ++i) s |= hit_rows[i] == row;
        skip = s;
    }
    __syncthreads();
    const int64_t start = (int64_t) row * (H / 4);
    for (int i = threadIdx.x; i < H / 4; i += blockDim.x)
        dst[start + i] = skip ? make_float4(0, 0, 0, 0) : const_cast<const float4*>(src)[start + i];
}
} // namespace

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

bool RemoteExpertOpt::init(std::string& err) {
    // The helpers' open() ran before this; every peer serves one model, so their geometry agrees.
    // (attach() is too early: open() has not set n_embd_ / k_ / cap_ yet.)
    if (!peers_.empty()) {
        n_embd_ = peers_[0].remote->n_embd_;
        k_ = peers_[0].remote->k_;
        cap_ = peers_[0].remote->cap_;
        for (auto& p : peers_)
            if (p.remote->n_embd_ != n_embd_ || p.remote->k_ != k_ || p.remote->cap_ != cap_) {
                err = "remote expert decode optimization: the helper caches disagree on the model's geometry";
                return false;
            }
    }
    void* mapped = nullptr;
    if (!check(cudaHostAlloc((void**) &h_sum_, (n_embd_ * strata::kernels::cpu::MAXT + cap_) * sizeof(float),
                             cudaHostAllocPortable | cudaHostAllocMapped), err) ||
        !check(cudaHostGetDevicePointer(&mapped, h_sum_, 0), err)) return false;
    m_sum_ = (float*) mapped;
    h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + n_embd_ * strata::kernels::cpu::MAXT);
    m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + n_embd_ * strata::kernels::cpu::MAXT);
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (!check(cudaMalloc((void**) &p.sum, n_embd_ * strata::kernels::cpu::MAXT * sizeof(float)), err)) return false;
    }
    return true;
}

RemoteExpertOpt::~RemoteExpertOpt() {
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (p.remote->stream_) cudaStreamSynchronize(p.remote->stream_);
        if (p.sum) cudaFree(p.sum);
        p.remote->remote_opt_ = nullptr;
    }
    if (h_sum_) cudaFreeHost(h_sum_);
}

size_t RemoteExpertOpt::metadata_bytes() const {
    // row[cap] int32s then weight[cap] floats: for Flash-Next's cap = 80 this is 640 B, the old
    // sizeof(ReduceMeta); for Qwen3.6's cap = 64 it is 512 B.  remote_experts.cpp allocates this
    // plus sizeof(RemoteMeta) at open() and copies it (plus the fixed part) at begin().
    return cap_ * sizeof(int32_t) + cap_ * sizeof(float);
}

void RemoteExpertOpt::begin(const float* weights, int token_begin, int tokens) {
    weights_ = weights;
    token_begin_ = token_begin;
    tokens_ = tokens;
    std::memset(h_sum_ + (size_t) token_begin * n_embd_, 0, (size_t) tokens * n_embd_ * sizeof(float));
    std::memset(h_mask_ + (size_t) token_begin * k_, 0, (size_t) tokens * k_ * sizeof(int32_t));
}

void RemoteExpertOpt::prepare(const RemoteExperts& remote, void* metadata) const {
    auto* row = (int32_t*) metadata;
    auto* weight = row + cap_;
    std::fill(row, row + tokens_ * k_, -1);
    for (size_t i = 0; i < remote.original_row_.size(); ++i)
        row[remote.original_row_[i]] = (int32_t) i;
    std::memcpy(weight, weights_, (size_t) tokens_ * k_ * sizeof(float));
}

bool RemoteExpertOpt::reduce(RemoteExperts& remote, const void* metadata, std::string& err) {
    const auto& p = *std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    // finish() synchronizes this stream before the host reads h_out_. When mapped output is available,
    // write the reduced rows there directly, avoiding a separate device-to-host copy for every layer.
    // The row index, the routing width and the row count come from the helper this peer was opened for
    // (n_embd_ / k_ / cap_), not from Flash-Next's constants: Qwen3.6's helper tier is 2048 wide with
    // top-8, and the metadata block prepare() filled is sized for that cap.
    float* sum = remote.zero_copy_ ? remote.z_out_ : p.sum;
    reduce_experts<<<dim3((n_embd_ + 255) / 256, tokens_), 256, 0, remote.stream_>>>(
        remote.d_out_, (const int32_t*) metadata, (const float*) metadata + cap_, sum,
        (int) n_embd_, (int) k_);
    if (remote.zero_copy_) return check(cudaGetLastError(), err);
    return check(cudaMemcpyAsync(remote.h_out_, p.sum, (size_t) tokens_ * n_embd_ * sizeof(float),
                                  cudaMemcpyDeviceToHost, remote.stream_), err);
}

void RemoteExpertOpt::accumulate(const RemoteExperts& remote) {
    for (int64_t i = 0; i < tokens_ * n_embd_; ++i)
        h_sum_[(size_t) token_begin_ * n_embd_ + i] += remote.h_out_[i];
    for (int i = 0; i < (int) (tokens_ * k_); ++i)
        h_mask_[(size_t) token_begin_ * k_ + i] |= remote.owned_[(size_t) i] != 0;
}

void RemoteExpertOpt::copy_rows(float* dst, const float* src, int token_begin, int tokens,
                          const int32_t* hit_rows, const int32_t* count, void* stream) const {
    copy_cpu_rows<<<tokens * (int) k_, 128, 0, (cudaStream_t) stream>>>(
        (float4*) dst, (const volatile float4*) src, m_mask_ + (size_t) token_begin * (int) k_, hit_rows, count,
        (int) n_embd_);
}

void RemoteExpertOpt::combine(float* dst, float* scratch, int token_begin, int tokens,
                       const uint32_t* skip, uint32_t ring, void* stream) const {
    const float* sum = m_sum_ + (size_t) token_begin * n_embd_;
    if (skip) {
        // An all-primary device-planned group bypasses the host: ignore its stale sum.
        strata::kernels::copy_or_zero_from_mapped(scratch, sum, (size_t) tokens * n_embd_, skip, ring, stream);
        sum = scratch;
    }
    strata::kernels::add_inplace(dst, sum, (size_t) tokens * n_embd_, stream);
}

bool RemoteExpertOpt::owns(int64_t layer, int32_t expert) const {
    for (const auto& p : peers_)
        if (p.remote->cache_.slot_of(layer, expert) >= 0) return true;
    return false;
}

bool RemoteExpertOpt::adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                     const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
                     ExpertSource& source) {
    if (max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> resident(primary.size(), 0);
    for (size_t i = 0; i < primary.size(); ++i) resident[i] = primary[i] >= 0;
    for (const auto& [i, slot] : pending) resident[(size_t) i] = 1;
    for (const auto& p : peers_) {
        const auto& r = *p.remote;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l)
            for (int32_t e = 0; e < r.n_expert_; ++e)
                if (r.cache_.slot_of(l, e) >= 0) resident[(size_t) l * r.n_expert_ + e] = 1;
    }
    struct Swap { float gain; int32_t layer, in, out; };
    std::string err;
    for (auto& p : peers_) {
        auto& r = *p.remote;
        std::vector<Swap> swaps;
        std::vector<std::pair<float, int32_t>> candidates, victims;
        for (int32_t l = 0; l < (int32_t) r.layers_present_.size(); ++l) {
            candidates.clear(); victims.clear();
            const size_t base = (size_t) l * r.n_expert_;
            float primary_cold = 1.0e30f;
            for (int32_t e = 0; e < r.n_expert_; ++e) {
                const float u = usage[base + e];
                if (primary[base + e] >= 0) primary_cold = std::min(primary_cold, u);
                if (r.cache_.slot_of(l, e) >= 0) victims.emplace_back(u, e);
                else if (!resident[base + e] && u >= 2.0f) candidates.emplace_back(u, e);
            }
            const size_t n = std::min(candidates.size(), victims.size());
            if (!n) continue;
            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::partial_sort(victims.begin(), victims.begin() + n, victims.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < n; ++i) {
                if (victims[i].first > primary_cold) break;
                if (candidates[i].first < victims[i].first + 1.5f) break;
                swaps.push_back({candidates[i].first - victims[i].first, l, candidates[i].second, victims[i].second});
            }
        }
        std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
        if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
        if (swaps.empty()) continue;
        const OnDevice on(r.device_);
        for (const auto& s : swaps) {
            const uint8_t* in_blob = source.blob(s.layer, s.in);
            if (!r.cache_.fill_slot(r.cache_.slot_of(s.layer, s.out), in_blob,
                                    r.stream_, err, (int64_t) lay.blob_bytes(s.layer))) return false;
            source.note_async_read(in_blob, r.stream_);   // #1237
        }
        if (!check(cudaStreamSynchronize(r.stream_), err)) return false;
        for (const auto& s : swaps) {
            r.cache_.replace(s.layer, s.out, s.in);
            const size_t base = (size_t) s.layer * r.n_expert_;
            resident[base + s.out] = 0;
            resident[base + s.in] = 1;
        }
    }
    return true;
}
} // namespace strata::core
