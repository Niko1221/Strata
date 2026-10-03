#pragma once
// Device intrinsics the CUDA kernels use, implemented with SYCL subgroups
// (Intel B60 supports subgroup size 32, the same width these kernels assume).
#include <cstdint>
#include <sycl/sycl.hpp>

inline int __dp4a(int a, int b, int c) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    uint32_t sum = static_cast<uint32_t>(c);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const int sa = static_cast<int>((ua >> (lane * 8)) & 0xffu);
        const int sb = static_cast<int>((ub >> (lane * 8)) & 0xffu);
        const int xa = sa < 0x80 ? sa : sa - 0x100;
        const int xb = sb < 0x80 ? sb : sb - 0x100;
        sum += static_cast<uint32_t>(xa * xb);
    }
    return static_cast<int>(sum);
}

inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    const uint64_t pair = (static_cast<uint64_t>(y) << 32) | x;
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned sel = (s >> (i * 4)) & 7u;
        out |= ((pair >> (sel * 8)) & 0xffu) << (i * 8);
    }
    return out;
}

inline int __vsub4(int a, int b) {
    constexpr uint32_t kHigh = 0x80808080u;
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    return static_cast<int>(((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh));
}
inline int __vsubss4(int a, int b) {
    constexpr uint32_t kHigh = 0x80808080u;
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    const uint32_t d = static_cast<uint32_t>(__vsub4(a, b));
    const uint32_t overflow = (ua ^ ub) & (ua ^ d) & kHigh;
    const uint32_t mask = (overflow >> 7) * 0xffu;
    const uint32_t bound = 0x7f7f7f7fu + ((ua & kHigh) >> 7);
    return static_cast<int>((d & ~mask) | (bound & mask));
}
inline int __vcmpne4(int a, int b) {
    constexpr uint32_t kHigh = 0x80808080u;
    const uint32_t t = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b);
    const uint32_t nonzero = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
    return static_cast<int>((nonzero >> 7) * 0xffu);
}

inline int __popc(unsigned int x) { return static_cast<int>(sycl::popcount(x)); }
inline int __popcll(unsigned long long x) { return static_cast<int>(sycl::popcount(x)); }
inline unsigned int __umulhi(unsigned int a, unsigned int b) {
    return static_cast<unsigned int>((static_cast<unsigned long long>(a) * b) >> 32);
}
inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int n) {
    n &= 31u;
    if (n == 0) return lo;
    return (hi << (32u - n)) | (lo >> n);
}
inline void __nanosleep(unsigned int) {
    // A pure volatile READ loop over host USM (the doorbell flags) can spin forever on a stale
    // GPU-cache line on this stack: neither volatile nor system-scope atomic loads snoop the
    // CPU's stores (poll-test2 modes 0/1/3 FAIL). A system-scope fence each iteration refreshes
    // the line (mode 5 OK, 5 ms end-to-end), so the pause doubles as the coherence poke.
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
}
inline long long clock64() { return 0; }
inline unsigned long long wall_clock64() { return 0; }
inline void __threadfence() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}
inline void __threadfence_block() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
}
inline void __threadfence_system() {
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __dsub_rn(double a, double b) { return a - b; }
inline double __ddiv_rn(double a, double b) { return a / b; }
inline double __dsqrt_rn(double a) { return sycl::sqrt(a); }
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fdividef(float a, float b) { return a / b; }
inline float __fmaf_rn(float a, float b, float c) { return sycl::fma(a, b, c); }
inline float __frcp_rn(float a) { return 1.0f / a; }
// Math helpers are substituted in tools/xpu/rewrite_cuda.py. Function-like
// macros named expf/sqrtf/fabsf break <math.h> token pasting, so they stay out.
template <typename T>
inline T min(T a, T b) { return a < b ? a : b; }
template <typename T>
inline T max(T a, T b) { return a > b ? a : b; }

#define __ldg(p) (*(p))
#define __trap() __builtin_trap()

namespace strata::xpu {
inline sycl::sub_group sg() { return sycl::ext::oneapi::this_work_item::get_sub_group(); }
}

template <typename T>
inline T __shfl_sync(unsigned, T val, int src) {
    return sycl::select_from_group(strata::xpu::sg(), val, src);
}
template <typename T>
inline T __shfl_down_sync(unsigned, T val, unsigned delta, int width = 32) {
    auto sg = strata::xpu::sg();
    const int lane = static_cast<int>(sg.get_local_id()[0]);
    const int base = lane & ~(width - 1);
    const int src = lane + static_cast<int>(delta);
    if (src >= base + width) return val;
    return sycl::select_from_group(sg, val, src);
}
template <typename T>
inline T __shfl_up_sync(unsigned, T val, unsigned delta) {
    return sycl::shift_group_left(strata::xpu::sg(), val, delta);
}
template <typename T>
inline T __shfl_xor_sync(unsigned, T val, unsigned mask, int width = 32) {
    (void) width;
    return sycl::permute_group_by_xor(strata::xpu::sg(), val, mask);
}
inline unsigned __ballot_sync(unsigned, int pred) {
    auto m = sycl::ext::oneapi::group_ballot(strata::xpu::sg(), pred != 0);
    unsigned bits = 0;
    for (unsigned i = 0; i < 32 && i < m.size(); ++i)
        if (m[i]) bits |= 1u << i;
    return bits;
}
inline unsigned __activemask() { return 0xffffffffu; }
inline int __all_sync(unsigned mask, int pred) {
    (void) mask;
    return sycl::all_of_group(strata::xpu::sg(), pred != 0) ? 1 : 0;
}
inline int __any_sync(unsigned mask, int pred) {
    (void) mask;
    return sycl::any_of_group(strata::xpu::sg(), pred != 0) ? 1 : 0;
}
inline unsigned __match_any_sync(unsigned, unsigned value) {
    unsigned bits = 0;
    auto g = strata::xpu::sg();
    const unsigned lane = static_cast<unsigned>(g.get_local_id());
    const unsigned lanes = static_cast<unsigned>(g.get_local_range()[0]);
    for (unsigned i = 0; i < lanes; ++i) {
        const unsigned other = sycl::select_from_group(g, value, static_cast<int>(i));
        if (other == value) bits |= 1u << i;
    }
    (void) lane;
    return bits;
}

template <typename T>
inline T atomicOr(T* addr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    return ref.fetch_or(val);
}
template <typename T>
inline T atomicAnd(T* addr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    return ref.fetch_and(val);
}
template <typename T>
inline T atomicXor(T* addr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    return ref.fetch_xor(val);
}
template <typename T>
inline T atomicExch(T* addr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    return ref.exchange(val);
}
template <typename T>
inline T atomicAdd(T* addr, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    return ref.fetch_add(val);
}
inline int atomicCAS(int* addr, int expected, int desired) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    ref.compare_exchange_strong(expected, desired);
    return expected;
}
inline unsigned int atomicCAS(unsigned int* addr, unsigned int expected, unsigned int desired) {
    sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>
        ref(*addr);
    ref.compare_exchange_strong(expected, desired);
    return expected;
}

inline void __syncthreads() {
    sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier(sycl::access::fence_space::local_space);
}
inline void __syncwarp(unsigned = 0xffffffffu) {
    strata::xpu::sg().barrier();
}
