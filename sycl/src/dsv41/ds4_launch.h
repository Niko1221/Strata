#pragma once
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
inline void ds4_metal_flush() {}
namespace strata::gpu { inline void* host_view(const void* p) { return const_cast<void*>(p); } }
constexpr bool ds4_dbg(const char*) { return false; }
