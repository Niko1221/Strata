#pragma once

#include "kernels.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/esimd_moe.hpp"

#include <array>
#include <map>
#include <vector>

namespace fnx {

inline uint16_t native_swiglu_f16(float gate, float up) {
    const float v = gate / (1.f + sycl::native::exp(-gate)) * up;
    const float saturated = v > 65504.f ? 65504.f : (v < -65504.f ? -65504.f : v);
    return sycl::bit_cast<uint16_t>(sycl::half(saturated));
}

class NativeMoe {
public:
    NativeMoe(dpct::queue_ptr queue, int max_rows, int n_embd, int n_ff);
    ~NativeMoe();
    NativeMoe(const NativeMoe&) = delete;
    NativeMoe& operator=(const NativeMoe&) = delete;

    size_t bytes() const { return bytes_; }
    // A nonnegative rows enables device routing: counts sum to rows and jobs are expert-ordered PF_JOB pieces.
    // Weight readiness does not block routing metadata readback or input gathering.
    void run(const DW& gu, size_t gu_stride, const DW& down, size_t down_stride,
             const int* counts, const BigJob* jobs, const uint16_t* x,
             float* part, bool use_i8, int rows = -1,
             const sycl::event* weights_ready = nullptr, int input_rows = -1);
    void report_profile(int card);

private:
    template<class T> T* alloc(size_t n) {
        auto* p = sycl::malloc_device<T>(n, *queue_);
        if (!p) throw std::bad_alloc();
        buffers_.push_back(p);
        bytes_ += n * sizeof(T);
        return p;
    }
    dpct::queue_ptr queue_;
    int max_rows_, n_embd_, n_ff_;
    size_t bytes_ = 0;
    std::vector<void*> buffers_;
    strata::prefill::Gemm gemm_;
    strata::prefill::i8::Workspace ws_;
    uint16_t *x_, *h_, *wgu_, *wd_;
    float *gu_, *down_;
    int *dst_, *job_rows_;
    std::vector<int> host_job_rows_;
    bool device_layout_ = false;
    bool grouped_f16_ = false;
    bool direct_input_ = false;
    bool fallback_trace_ = false;
    bool host_trace_ = false;
    std::map<std::array<int, 3>, size_t> fallback_rows_;
    int* source_rows_ = nullptr;
    int *device_meta_ = nullptr, *device_jobs_ = nullptr;
    double profile_metadata_ms_ = 0, profile_submit_ms_ = 0;
    double profile_device_submit_ms_ = 0, profile_fallback_submit_ms_ = 0;
    double profile_weights_dep_ms_ = 0;
    size_t profile_weights_checks_ = 0, profile_weights_pending_ = 0;
    size_t profile_calls_ = 0, profile_device_calls_ = 0;
    size_t profile_grouped_calls_ = 0;
    size_t profile_direct_calls_ = 0;
};

} // namespace fnx
