#include "native_gdn.hpp"
#include "strata/kprof.hpp"
#include <stdexcept>

namespace fnx::native_prefill {
void gdn_gates(const float*, const float*, const float*, float*, float*, int64_t, void*);
void gdn_conv(float*, const float*, const float*, float*, int64_t, float, void*);
void gdn_recurrence_variant(int, float*, const float*, const float*, const float*, const float*,
                            const float*, float, float*, uint16_t*, int64_t, void*, int64_t);
}

namespace fnx {
NativeGdn::NativeGdn(dpct::queue_ptr queue, int rows) : queue_(queue), rows_(rows) {
    constexpr int ch = (2 * nk + nv) * hs;
    try {
        qkv_ = alloc<float>((size_t)rows * ch);
        h_ = alloc<float>((size_t)rows * ch);
        z_ = alloc<float>((size_t)rows * nv * hs);
        gate_ = alloc<float>((size_t)rows * nv);
        beta_ = alloc<float>((size_t)rows * nv);
        history_ = alloc<float>(3 * ch);
        state_ = alloc<float>(nv * hs * hs);
        output16_ = alloc<uint16_t>((size_t)rows * nv * hs);
    } catch (...) {
        for (void* p : buffers_) sycl::free(p, *queue_);
        throw;
    }
}
NativeGdn::~NativeGdn() {
    for (void* p : buffers_) sycl::free(p, *queue_);
}
void NativeGdn::run(const float* qkvz, const float* ab, float* history,
                    const float* conv, const float* dt, const float* a,
                    const float* norm, float* state, const int* ctl, int groups,
                    int rows, float eps, float* out) {
    if (rows <= 0 || rows > rows_ || groups != (rows + MAXT - 1) / MAXT)
        throw std::invalid_argument("Native GDN row/group count mismatch");
    constexpr int ch = (2 * nk + nv) * hs, stride = ch + nv * hs;
    auto* x = qkv_;
    auto* z = z_;
    auto* hist = history_;
    auto* st = state_;
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>((size_t)rows * stride), [=](sycl::id<1> id) {
        const int r = id[0] / stride, c = id[0] % stride;
        if (c < ch) x[(size_t)r * ch + c] = qkvz[id];
        else z[(size_t)r * nv * hs + c - ch] = qkvz[id];
    });
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>(3 * ch), [=](sycl::id<1> id) {
        const int c = id[0] / 3, t = id[0] % 3;
        const unsigned parity = ((const unsigned*)ctl)[3] & 1u;
        hist[id] = history[(size_t)parity * 3 * ch + t * ch + c];
    });
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>(nv * hs * hs), [=](sycl::id<1> id) {
        const int j = id[0] % hs, h = id[0] / hs % nv, i = id[0] / (nv * hs);
        st[id] = state[((size_t)h * hs + i) * hs + j];
    });
    native_prefill::gdn_gates(ab, dt, a, gate_, beta_, rows, queue_);
    native_prefill::gdn_conv(history_, qkv_, conv, h_, rows, eps, queue_);
    native_prefill::gdn_recurrence_variant(0, state_, h_, gate_, beta_, z_, norm,
                                          eps, out, output16_, rows, queue_, 0);
    const auto* output16 = output16_;
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>((size_t)rows * nv * hs), [=](sycl::id<1> id) {
        out[id] = float(sycl::bit_cast<sycl::half>(output16[id]));
    });
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>(nv * hs * hs), [=](sycl::id<1> id) {
        const int j = id[0] % hs, h = id[0] / hs % nv, i = id[0] / (nv * hs);
        state[((size_t)h * hs + i) * hs + j] = st[id];
    });
    strata::kprof::kp(queue_, __FILE__, __func__, __LINE__)->parallel_for(sycl::range<1>(3 * ch), [=](sycl::id<1> id) {
        const int c = id[0] / 3, t = id[0] % 3;
        const unsigned parity = (((const unsigned*)ctl)[3 + 2 * (groups - 1)] & 1u) ^ 1u;
        history[(size_t)parity * 3 * ch + t * ch + c] = hist[id];
    });
}
}
