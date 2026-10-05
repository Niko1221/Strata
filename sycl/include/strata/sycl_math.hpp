// include/strata/sycl_math.hpp - the SYCL port's device math that dpct emulates but the hardware has.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

namespace strata {
// CUDA's __dp4a(int, int, int): four signed byte products summed into c. One place to change it: the byte
// unpack + multiply-add form is what IGC pattern-matches into the DP4A instruction (sycl::ext::oneapi::dot_acc is
// the same emulation and its header defines non-inline functions, which breaks the link across TUs).
template <typename A, typename B, typename C>
inline int32_t dp4a(A a, B b, C c) {
    return dpct::dp4a((int32_t) a, (int32_t) b, (int32_t) c);
}
}  // namespace strata
