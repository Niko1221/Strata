#include "native_qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kprof.hpp"
#include <oneapi/mkl.hpp>
#include <algorithm>
#include <stdexcept>

namespace fnx {

NativeQsa::NativeQsa(dpct::queue_ptr queue, int max_rows, int max_pools)
    : queue_(queue), max_rows_(max_rows), max_pools_(max_pools),
      tile_(std::min(8192, max_pools)) {
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, *queue_);
        if (!p) throw std::bad_alloc();
        buffers_.push_back(p);
        bytes_ += bytes;
        return p;
    };
    try {
        keys_ = static_cast<float*>(alloc((size_t)max_pools * 128 * sizeof(float)));
        dots_ = static_cast<float*>(alloc((size_t)tile_ * max_rows * 4 * sizeof(float)));
        steps_ = static_cast<int*>(alloc((size_t)max_rows * strata::kernels::kStepCount * sizeof(int)));
        dead_ = static_cast<float*>(alloc(128 * sizeof(float)));
        queue_->memset(dead_, 0, 128 * sizeof(float));
    } catch (...) {
        queue_->wait_and_throw();
        for (void* p : buffers_) sycl::free(p, *queue_);
        throw;
    }
}

NativeQsa::~NativeQsa() {
    for (void* p : buffers_) sycl::free(p, *queue_);
}

void NativeQsa::scores(const float* query, const uint16_t* pooled, const KvPages& pages,
                       const int* position, int rows, int active_pools, float* output) {
    if (rows <= 0 || rows > max_rows_ || active_pools < 0 || active_pools > max_pools_)
        throw std::invalid_argument("Native QSA score capacity mismatch");
    if (active_pools <= QSA_TOP_K / 4) return;
    const auto* half_keys = reinterpret_cast<const sycl::half*>(pooled);
    auto* keys = keys_;
    auto* steps = steps_;
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
        sycl::range<1>((size_t)active_pools * 128), [=](sycl::id<1> id) {
            const int block = id[0] / 128, dim = id[0] % 128;
            keys[id] = float(half_keys[kvp_block(pages, block) + dim]);
        });
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(
        sycl::range<1>(rows), [=](sycl::id<1> id) {
            const int p = *position + int(id[0]);
            auto* step = steps + id[0] * strata::kernels::kStepCount;
            step[strata::kernels::kStepPos] = p;
            step[strata::kernels::kStepNKv] = p + 1;
            step[strata::kernels::kStepNBid] = (p + 1) / 4;
            step[strata::kernels::kStepWidth] = 0;
        });
    for (int b0 = 0; b0 < active_pools; b0 += tile_) {
        const int nb = std::min(tile_, active_pools - b0);
        {
            strata::kprof::Section section(queue_, "native_qsa.oneMKL.sgemm");
            oneapi::mkl::blas::column_major::gemm(
                *queue_, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                int64_t(nb), int64_t(rows * 4), int64_t(128),
                1.f, keys_ + (size_t)b0 * 128, int64_t(128), query, int64_t(128),
                0.f, dots_, int64_t(tile_));
        }
        // Same FP32 GEMM and ordered per-head ReLU reduction as PP prefill.
        strata::kernels::qsa_block_scores_reduce(
            dots_, tile_, b0, nb, dead_, query, steps_, rows, max_pools_, output, queue_);
    }
}

} // namespace fnx
