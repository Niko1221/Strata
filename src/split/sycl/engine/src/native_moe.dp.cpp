#include "native_moe.hpp"
#include "sycl_port.hpp"
#include "strata/kprof.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/xmx_moe.hpp"
#include "strata/sycl_queue.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace fnx {
namespace i8 = strata::prefill::i8;

NativeMoe::NativeMoe(dpct::queue_ptr queue, int max_rows, int n_embd, int n_ff)
    : queue_(queue), max_rows_(max_rows), n_embd_(n_embd), n_ff_(n_ff) {
    const char* device = std::getenv("FNX_PF_MOE_DEVICE");
    device_layout_ = device && std::atoi(device) != 0;
    const char* direct = std::getenv("FNX_PF_MOE_DIRECT");
    direct_input_ = direct && std::atoi(direct) != 0;
    const char* grouped = std::getenv("FNX_PF_MOE_GROUPED");
    grouped_f16_ = grouped && std::atoi(grouped) != 0;
    const char* trace = std::getenv("FNX_PF_MOE_FALLBACK_TRACE");
    fallback_trace_ = trace && std::atoi(trace) != 0;
    const char* host_trace = std::getenv("FNX_PF_HOST_TRACE");
    host_trace_ = host_trace && std::atoi(host_trace) != 0;
    if (grouped_f16_ &&
        (strata::intel_gpu_gen(queue_->get_device()) != strata::IntelGpuGen::Battlemage ||
         n_embd % 128 || n_ff % 64))
        throw std::invalid_argument("Grouped FP16 MoE requires Xe2 and aligned projection shapes");
    static_assert(i8::kMaxGroup == strata::prefill::xmx::kMaxGroup);
    try {
        x_ = alloc<uint16_t>((size_t)max_rows * n_embd);
        h_ = alloc<uint16_t>((size_t)max_rows * n_ff);
        gu_ = alloc<float>((size_t)max_rows * 2 * n_ff);
        down_ = alloc<float>((size_t)max_rows * n_embd);
        dst_ = alloc<int>(max_rows);
        if (direct_input_) source_rows_ = alloc<int>(max_rows);
        job_rows_ = alloc<int>(max_rows + (device_layout_ ? NEXP : 0));
        const int slots = grouped_f16_ ? i8::kMaxGroup : 1;
        wgu_ = alloc<uint16_t>((size_t)2 * n_ff * n_embd);
        wd_ = alloc<uint16_t>((size_t)slots * n_embd * n_ff);
        const size_t padded = max_rows + 7 * (device_layout_ ? NEXP : i8::kMaxGroup);
        if (device_layout_) {
            device_meta_ = alloc<int>((NEXP + 1) * 4 +
                ((max_rows + 7) / 8 + NEXP - 1) * 4 +
                ((max_rows + 15) / 16 + NEXP - 1) * 8);
            device_jobs_ = alloc<int>(1);
        }
        ws_.xq = alloc<int8_t>(padded * n_embd);
        ws_.dx = alloc<float>(padded * n_embd / 32);
        ws_.hq = alloc<int8_t>(padded * n_ff);
        ws_.dh = alloc<float>(padded * n_ff / 32);
        std::string error;
        if (!gemm_.init_external(queue, wgu_, (int64_t)2 * n_ff * n_embd,
                                 nullptr, 0, error))
            throw std::runtime_error("Native MoE GEMM initialization: " + error);
    } catch (...) {
        for (void* p : buffers_) sycl::free(p, *queue_);
        throw;
    }
}

NativeMoe::~NativeMoe() {
    for (void* p : buffers_) sycl::free(p, *queue_);
}

void NativeMoe::run(const DW& gu, size_t gu_stride, const DW& down,
                    size_t down_stride, const int* counts, const BigJob* jobs,
                    const uint16_t* x, float* part, bool use_i8, int rows,
                    const sycl::event* weights_ready, int input_rows) {
    const bool profile = strata::kprof::enabled() || host_trace_;
    const auto entered = profile ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    if (gu.n_in != n_embd_ || gu.n_out != 2 * n_ff_ ||
        down.n_in != n_ff_ || down.n_out != n_embd_)
        throw std::invalid_argument("Native MoE slice shape mismatch");
    if (rows > max_rows_)
        throw std::invalid_argument("Native MoE rows exceed allocated capacity");
    auto wait_for_weights = [&] {
        if (weights_ready) {
            if (host_trace_) {
                ++profile_weights_checks_;
                if (weights_ready->get_info<sycl::info::event::command_execution_status>() !=
                    sycl::info::event_command_status::complete)
                    ++profile_weights_pending_;
            }
            const auto begin = profile ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
            strata::kprof::Section section(queue_, "pipeline cold weights wait");
            sycl_port::stream_barrier(queue_, {*weights_ready});
            if (profile)
                profile_weights_dep_ms_ += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
        }
    };
    const bool device_format = (gu.type == T_IQ3_S || gu.type == T_IQ3_XXS) &&
        (down.type == T_IQ4_NL || down.type == 42);
    if (device_layout_ && rows >= 0 && use_i8 && device_format &&
        gu.ep && down.ep && i8::available(queue_)) {
        if (!rows) return;
        const int ne = n_embd_;
        auto* offsets = job_rows_;
        auto* njobs = device_jobs_;
        strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->single_task([=] {
            int row = 0, job = 0;
            for (int e = 0; e < NEXP; ++e) {
                for (int r = 0; r < counts[e]; r += PF_JOB)
                    offsets[job++] = row + r;
                row += counts[e];
            }
            *njobs = job;
        });
        auto* gathered = x_;
        auto* dst = dst_;
        const bool direct = direct_input_ && rows >= 1024 && input_rows > 0 &&
            input_rows <= rows && (reinterpret_cast<uintptr_t>(x) & 15) == 0;
        auto* source_rows = source_rows_;
        const int bound = (rows + PF_JOB - 1) / PF_JOB + NEXP - 1;
        i8::DeviceInput input{};
        if (direct) {
            strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
                sycl::range<2>(bound, PF_JOB), [=](sycl::id<2> id) {
                    if (id[0] >= size_t(*njobs)) return;
                    const BigJob& job = jobs[id[0]];
                    const int r = int(id[1]);
                    if (r >= job.n) return;
                    const int out = offsets[id[0]] + r;
                    source_rows[out] = job.y[r];
                    dst[out] = job.out[r];
                });
            // x_ is unused as FP16 on this path; borrow it for the smaller token-level Q8 image.
            i8::Layout dense;
            i8::make_layout(dense, &input_rows, 1);
            auto* xq = reinterpret_cast<int8_t*>(x_);
            auto* dx = reinterpret_cast<float*>(xq + i8::xq_bytes(dense, ne));
            i8::quantize_rows(x, dense, ne, xq, dx, queue_);
            input = {source_rows, xq, dx};
        } else {
            const bool vector_gather = ne % 8 == 0 &&
                ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(gathered)) & 15) == 0;
            const int width = vector_gather ? ne / 8 : ne;
            strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
                sycl::range<2>(bound, (size_t)PF_JOB * width), [=](sycl::id<2> id) {
                    if (id[0] >= size_t(*njobs)) return;
                    const BigJob& job = jobs[id[0]];
                    const int r = id[1] / width, k = id[1] % width;
                    if (r >= job.n) return;
                    const int out = offsets[id[0]] + r;
                    if (vector_gather)
                        reinterpret_cast<sycl::uint4*>(gathered)[(size_t)out * width + k] =
                            reinterpret_cast<const sycl::uint4*>(x)[(size_t)job.y[r] * width + k];
                    else gathered[(size_t)out * ne + k] = x[(size_t)job.y[r] * ne + k];
                    if (k == 0) dst[out] = job.out[r];
                });
        }
        wait_for_weights();
        if (i8::device_chain(gu.type, down.type,
                reinterpret_cast<const uint8_t* const*>(gu.ep),
                reinterpret_cast<const uint8_t* const*>(down.ep),
                (size_t)n_ff_ * gu.row_bytes, NEXP, rows, ne, n_ff_,
                counts, device_meta_, x_, gu_, down_, ws_, queue_, direct ? &input : nullptr)) {
            const auto* result = down_;
            const bool vector_scatter = ne % 4 == 0 &&
                ((reinterpret_cast<uintptr_t>(part) | reinterpret_cast<uintptr_t>(result)) & 15) == 0;
            const int scatter_width = vector_scatter ? ne / 4 : ne;
            strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
                sycl::range<1>((size_t)rows * scatter_width), [=](sycl::id<1> id) {
                    const int r = id[0] / scatter_width, k = id[0] % scatter_width;
                    if (vector_scatter)
                        reinterpret_cast<sycl::float4*>(part)[(size_t)dst[r] * scatter_width + k] =
                            reinterpret_cast<const sycl::float4*>(result)[id[0]];
                    else part[(size_t)dst[r] * ne + k] = result[id];
                });
            if (profile) {
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - entered).count();
                profile_submit_ms_ += elapsed;
                profile_device_submit_ms_ += elapsed;
                ++profile_calls_;
                ++profile_device_calls_;
                if (direct) ++profile_direct_calls_;
            }
            return;
        }
    }
    std::array<int, NEXP> cnt{};
    std::array<const char*, NEXP> gp{}, dp{};
    queue_->memcpy(cnt.data(), counts, sizeof(cnt));
    if (gu.ep) queue_->memcpy(gp.data(), gu.ep, sizeof(gp));
    else for (int e = 0; e < NEXP; ++e) gp[e] = (const char*)gu.p + e * gu_stride;
    if (down.ep) queue_->memcpy(dp.data(), down.ep, sizeof(dp));
    else for (int e = 0; e < NEXP; ++e) dp[e] = (const char*)down.p + e * down_stride;
    // The official launch ABI takes host row counts and embeds pointers by value.
    queue_->wait_and_throw();
    const auto metadata_done = profile ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
    const int ne = n_embd_, nf = n_ff_;
    int total_rows = 0;
    host_job_rows_.clear();
    for (int e = 0; e < NEXP; ++e) {
        if (cnt[e] < 0 || cnt[e] > max_rows_ - total_rows)
            throw std::invalid_argument("Native MoE counts exceed allocated row capacity");
        for (int r = 0; r < cnt[e]; r += PF_JOB)
            host_job_rows_.push_back(total_rows + r);
        total_rows += cnt[e];
    }
    if (!total_rows) return;
    queue_->memcpy(job_rows_, host_job_rows_.data(), host_job_rows_.size() * sizeof(int));
    auto* gathered = x_;
    auto* dst = dst_;
    const auto* job_rows = job_rows_;
    const bool vector_gather = ne % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(gathered)) & 15) == 0;
    const int gather_width = vector_gather ? ne / 8 : ne;
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
        sycl::range<2>(host_job_rows_.size(), (size_t)PF_JOB * gather_width), [=](sycl::id<2> id) {
            const BigJob& job = jobs[id[0]];
            const int r = id[1] / gather_width, k = id[1] % gather_width;
            if (r >= job.n) return;
            const int out = job_rows[id[0]] + r;
            if (vector_gather)
                reinterpret_cast<sycl::uint4*>(gathered)[(size_t)out * gather_width + k] =
                    reinterpret_cast<const sycl::uint4*>(x)[(size_t)job.y[r] * gather_width + k];
            else gathered[(size_t)out * ne + k] = x[(size_t)job.y[r] * ne + k];
            if (k == 0) dst[out] = job.out[r];
        });
    wait_for_weights();
    int row_base = 0;
    bool used_grouped = false;
    for (int first = 0; first < NEXP; first += i8::kMaxGroup) {
        i8::Layout lo;
        std::array<const uint8_t*, i8::kMaxGroup> gates{}, downs{};
        const int n = std::min(i8::kMaxGroup, NEXP - first);
        for (int e = 0; e < n; ++e) {
            gates[e] = (const uint8_t*)gp[first + e];
            downs[e] = (const uint8_t*)dp[first + e];
        }
        i8::make_layout(lo, cnt.data() + first, n);
        const int nr = lo.row0[n];
        if (!nr) continue;
        const auto* group_x = x_ + (size_t)row_base * ne;
        auto* group_down = down_ + (size_t)row_base * ne;
        bool int8_done = false;
        const bool int8_format = (gu.type == T_IQ3_S || gu.type == T_IQ3_XXS ||
                                  gu.type == T_IQ2_S) &&
                                 (down.type == T_IQ4_NL || down.type == 42);
        if (use_i8 && int8_format && i8::available(queue_)) {
            i8::quantize_rows(group_x, lo, ne, ws_.xq, ws_.dx, queue_);
            if (!i8::gate_up(gu.type, gates.data(), (size_t)nf * gu.row_bytes,
                             nf, lo, ne, ws_.xq, ws_.dx, gu_, queue_))
                throw std::runtime_error("Native MoE DPAS gate/up rejected slice");
            i8::swiglu_quantize(gu_, lo, nf, ws_.hq, ws_.dh, queue_);
            if (!i8::down(down.type, downs.data(), 0, ne, lo, nf, ws_.hq,
                          ws_.dh, group_down, queue_))
                throw std::runtime_error("Native MoE DPAS down rejected slice");
            int8_done = true;
        }
        if (!int8_done) {
            if (fallback_trace_)
                for (int e = 0; e < n; ++e)
                    if (lo.cnt[e])
                        ++fallback_rows_[{int(gu.type), int(down.type), lo.cnt[e]}];
            if (grouped_f16_ && strata::kernels::native_expert_supported(
                    gu.type, down.type, ne, nf)) {
                std::array<const uint16_t*, i8::kMaxGroup> wd{};
                for (int e = 0; e < n; ++e) {
                    if (!lo.cnt[e]) continue;
                    auto* down_weight = wd_ + (size_t)e * ne * nf;
                    wd[e] = down_weight;
                    strata::kernels::iq_dequant_expert_f16(
                        gu.type, down.type, gates[e],
                        gates[e] + (size_t)nf * gu.row_bytes, downs[e],
                        nf, ne, wgu_, down_weight, queue_);
                    // Keep GU's accumulation order before the FP16 SwiGLU rounding boundary.
                    const int r0 = lo.row0[e];
                    gemm_.f16(group_x + (size_t)r0 * ne, wgu_,
                        gu_ + (size_t)r0 * 2 * nf, lo.cnt[e], 2 * nf, ne);
                }
                const auto* g = gu_;
                auto* h = h_;
                strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
                    sycl::range<1>((size_t)nr * nf), [=](sycl::id<1> id) {
                        h[id] = native_swiglu_f16(g[2 * id[0]], g[2 * id[0] + 1]);
                    });
                strata::prefill::xmx::grouped_f16(
                    h_, wd.data(), lo.cnt, n, group_down, ne, nf, queue_);
                used_grouped = true;
                row_base += nr;
                continue;
            }
            for (int e = 0; e < n; ++e) {
                const int rows = lo.cnt[e], r0 = lo.row0[e];
                if (!rows) continue;
                const bool fused = strata::kernels::native_expert_supported(
                    gu.type, down.type, ne, nf);
                if (fused) {
                    strata::kernels::iq_dequant_expert_f16(
                        gu.type, down.type, gates[e],
                        gates[e] + (size_t)nf * gu.row_bytes, downs[e],
                        nf, ne, wgu_, wd_, queue_);
                    gemm_.f16(group_x + (size_t)r0 * ne, wgu_,
                               gu_ + (size_t)r0 * 2 * nf, rows, 2 * nf, ne);
                } else {
                    gemm_.native(group_x + (size_t)r0 * ne, gu.type, gates[e],
                                  gu_ + (size_t)r0 * 2 * nf, rows, 2 * nf, ne);
                }
                const auto* g = gu_ + (size_t)r0 * 2 * nf;
                auto* h = h_ + (size_t)r0 * nf;
                strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>((size_t)rows * nf),
                    [=](sycl::id<1> id) {
                        const int r = id[0] / nf, k = id[0] % nf;
                        const float a = g[(size_t)r * 2 * nf + (fused ? 2 * k : k)];
                        const float b = g[(size_t)r * 2 * nf +
                                          (fused ? 2 * k + 1 : nf + k)];
                        h[id] = native_swiglu_f16(a, b);
                    });
                if (fused) gemm_.f16(h, wd_, group_down + (size_t)r0 * ne, rows, ne, nf);
                else gemm_.native(h, down.type, downs[e],
                                   group_down + (size_t)r0 * ne, rows, ne, nf);
            }
        }
        row_base += nr;
    }
    const auto* result = down_;
    const bool vector_scatter = ne % 4 == 0 &&
        ((reinterpret_cast<uintptr_t>(part) | reinterpret_cast<uintptr_t>(result)) & 15) == 0;
    const int scatter_width = vector_scatter ? ne / 4 : ne;
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
        sycl::range<1>((size_t)total_rows * scatter_width), [=](sycl::id<1> id) {
            const int r = id[0] / scatter_width, k = id[0] % scatter_width;
            if (vector_scatter)
                reinterpret_cast<sycl::float4*>(part)[(size_t)dst[r] * scatter_width + k] =
                    reinterpret_cast<const sycl::float4*>(result)[id[0]];
            else part[(size_t)dst[r] * ne + k] = result[id];
        });
    if (profile) {
        profile_metadata_ms_ += std::chrono::duration<double, std::milli>(metadata_done - entered).count();
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - metadata_done).count();
        profile_submit_ms_ += elapsed;
        profile_fallback_submit_ms_ += elapsed;
        ++profile_calls_;
        if (used_grouped) ++profile_grouped_calls_;
    }
}

void NativeMoe::report_profile(int card) {
    if (strata::kprof::enabled() || host_trace_)
        std::fprintf(stderr, "hostprof native_moe card=%d calls=%zu metadata_wait_ms=%.3f submit_ms=%.3f device_calls=%zu grouped_calls=%zu direct_calls=%zu device_submit_ms=%.3f fallback_submit_ms=%.3f weights_dep_submit_ms=%.3f weights_pending=%zu weights_checks=%zu (host wall time; includes existing GPU waits; weights dependency time is a subset of submit time)\n",
            card, profile_calls_, profile_metadata_ms_, profile_submit_ms_, profile_device_calls_,
            profile_grouped_calls_, profile_direct_calls_, profile_device_submit_ms_,
            profile_fallback_submit_ms_, profile_weights_dep_ms_,
            profile_weights_pending_, profile_weights_checks_);
    for (const auto& entry : fallback_rows_)
        std::fprintf(stderr, "moe_fallback_rows card=%d gu_type=%d down_type=%d M=%d expert_calls=%zu grouped=%d\n",
                     card, entry.first[0], entry.first[1], entry.first[2], entry.second,
                     int(grouped_f16_));
    fallback_rows_.clear();
    profile_metadata_ms_ = profile_submit_ms_ = 0;
    profile_device_submit_ms_ = profile_fallback_submit_ms_ = profile_weights_dep_ms_ = 0;
    profile_weights_checks_ = profile_weights_pending_ = 0;
    profile_calls_ = 0;
    profile_device_calls_ = 0;
    profile_grouped_calls_ = 0;
    profile_direct_calls_ = 0;
}

} // namespace fnx
