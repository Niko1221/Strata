#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int K = 10, MAXT = strata::kernels::cpu::MAXT, CAP = MAXT * K;
struct ReduceMeta { int32_t row[CAP]; float weight[CAP]; };

bool check(dpct::err0 status, std::string &err) {
    if (status == 0) return true;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    err = std::string("remote expert decode optimization: ") +
          dpct::get_error_string_dummy(status);
    return false;
}

__dpct_inline__ void reduce_experts(const float *parts, const ReduceMeta *meta,
                                    float *sum) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int i = t * K + j, r = meta->row[i];
        if (r >= 0) v += parts[(int64_t) r * H + c] * meta->weight[i];
    }
    sum[(int64_t) t * H + c] = v;
}

// Like the upstream mapped-row copy, also skip rows returned in the remote sum.
/*
DPCT1052: SYCL does not support the member access for a volatile qualified
vector type. The volatile qualifier was removed. You may need to rewrite the
code.
*/
__dpct_inline__ void copy_cpu_rows(sycl::float4 *dst, const sycl::float4 *src,
                                   const int32_t *mask, const int32_t *hit_rows,
                                   const int32_t *count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2);
    auto &skip = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        int s = mask[row];
#pragma unroll
        for (int i = 0; i < *count; ++i) s |= hit_rows[i] == row;
        skip = s;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int64_t start = (int64_t) row * (H / 4);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < H / 4;
         i += item_ct1.get_local_range(2))
        dst[start + i] = skip
                             ? sycl::float4(0, 0, 0, 0)
                             : const_cast<const sycl::float4 *>(src)[start + i];
}
} // namespace

void RemoteExpertOpt::attach(RemoteExperts& remote) {
    remote.remote_opt_ = this;
    peers_.push_back({&remote});
}

bool RemoteExpertOpt::init(std::string &err) try {
    void* mapped = nullptr;
    if (!check(
            DPCT_CHECK_ERROR(h_sum_ = sycl::malloc_host<float>(
                                 MAXT * (H + K), dpct::get_in_order_queue())),
            err) ||
        !check(DPCT_CHECK_ERROR(mapped = (void *)h_sum_), err)) return false;
    m_sum_ = (float*) mapped;
    h_mask_ = reinterpret_cast<int32_t*>(h_sum_ + MAXT * H);
    m_mask_ = reinterpret_cast<int32_t*>(m_sum_ + MAXT * H);
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (!check(DPCT_CHECK_ERROR(p.sum = sycl::malloc_device<float>(
                                        MAXT * H, dpct::get_in_order_queue())),
                   err)) return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

RemoteExpertOpt::~RemoteExpertOpt() {
    for (auto& p : peers_) {
        const OnDevice on(p.remote->device_);
        if (p.remote->stream_) p.remote->stream_->wait();
        if (p.sum) sycl::free(p.sum, dpct::get_in_order_queue());
        p.remote->remote_opt_ = nullptr;
    }
    if (h_sum_) sycl::free(h_sum_, dpct::get_in_order_queue());
}

size_t RemoteExpertOpt::metadata_bytes() { return sizeof(ReduceMeta); }

void RemoteExpertOpt::begin(const float* weights, int token_begin, int tokens) {
    weights_ = weights;
    token_begin_ = token_begin;
    tokens_ = tokens;
    std::memset(h_sum_ + token_begin * H, 0, tokens * H * sizeof(float));
    std::memset(h_mask_ + token_begin * K, 0, tokens * K * sizeof(int32_t));
}

void RemoteExpertOpt::prepare(const RemoteExperts& remote, void* metadata) const {
    auto* m = (ReduceMeta*) metadata;
    std::fill(m->row, m->row + tokens_ * K, -1);
    for (size_t i = 0; i < remote.original_row_.size(); ++i)
        m->row[remote.original_row_[i]] = (int32_t) i;
    std::memcpy(m->weight, weights_, tokens_ * K * sizeof(float));
}

bool RemoteExpertOpt::reduce(RemoteExperts& remote, const void* metadata, std::string& err) {
    const auto& p = *std::find_if(peers_.begin(), peers_.end(), [&](const Peer& p) { return p.remote == &remote; });
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(remote.stream_))->submit([&](sycl::handler &cgh) {
            const float *remote_d_out__ct0 = remote.d_out_;

            cgh.parallel_for<dpct_kernel_name<class reduce_experts_36dfcf>>(
                sycl::nd_range<3>(sycl::range(1, tokens_, (H + 255) / 256) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    reduce_experts(remote_d_out__ct0,
                                   (const ReduceMeta *)metadata, p.sum);
                });
        });
    }
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    return check(DPCT_CHECK_ERROR(remote.stream_->memcpy(
                     remote.h_out_, p.sum, tokens_ * H * sizeof(float))),
                 err);
}

void RemoteExpertOpt::accumulate(const RemoteExperts& remote) {
    for (int64_t i = 0; i < tokens_ * H; ++i) h_sum_[token_begin_ * H + i] += remote.h_out_[i];
    for (int i = 0; i < tokens_ * K; ++i)
        h_mask_[token_begin_ * K + i] |= remote.owned_[(size_t) i] != 0;
}

void RemoteExpertOpt::copy_rows(float* dst, const float* src, int token_begin, int tokens,
                         const int32_t* hit_rows, const int32_t* count, void* stream) const {
    auto exp_props = sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};

    strata::q_of(stream)->submit([&](sycl::handler &cgh) {
        const int32_t *m_mask__token_begin_K_ct2 = m_mask_ + token_begin * K;

        cgh.parallel_for<dpct_kernel_name<class copy_cpu_rows_7f2e51>>(
            sycl::nd_range<3>(sycl::range(1, 1, tokens * K) *
                                  sycl::range(1, 1, 128),
                              sycl::range(1, 1, 128)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                copy_cpu_rows((sycl::float4 *)dst,
                              (const sycl::float4 *)src,
                              m_mask__token_begin_K_ct2, hit_rows, count);
            });
    });
}

void RemoteExpertOpt::combine(float* dst, float* scratch, int token_begin, int tokens,
                       const uint32_t* skip, uint32_t ring, void* stream) const {
    const float* sum = m_sum_ + token_begin * H;
    if (skip) {
        // An all-primary device-planned group bypasses the host: ignore its stale sum.
        strata::kernels::copy_or_zero_from_mapped(scratch, sum, tokens * H, skip, ring, stream);
        sum = scratch;
    }
    strata::kernels::add_inplace(dst, sum, tokens * H, stream);
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
        for (const auto& s : swaps)
            if (!r.cache_.fill_slot(r.cache_.slot_of(s.layer, s.out), source.blob(s.layer, s.in),
                                    r.stream_, err, (int64_t) lay.blob_bytes(s.layer))) return false;
        if (!check(DPCT_CHECK_ERROR(r.stream_->wait()), err)) return false;
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
