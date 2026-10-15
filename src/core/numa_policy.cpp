// src/core/numa_policy.cpp - S1's range binder, moved out of expert_source.cpp's anonymous namespace.  The header
// says what it does and why the call goes through syscall(2); this file is only where it is defined.  The binding
// and the node accessors S2/S3 use are inline in the header, because conversation_buffer.hpp (a header-only class)
// reads them and its own tests link nothing.
#include "strata/core/numa_policy.hpp"

#if defined(__linux__)
#include <cerrno>
#include <unistd.h>   // syscall(2)
#endif

namespace strata::core::detail {

#if defined(__linux__) && defined(SYS_mbind) && defined(SYS_set_mempolicy) && defined(SYS_get_mempolicy)

bool numa_bind_range(void* addr, uint64_t bytes, int node, int& error_number) {
    unsigned long mask[kNumaMaxNode / 64];
    numa_node_mask(node, mask);
    errno = 0;
    if (syscall(SYS_mbind, addr, (unsigned long) bytes, kMpolBind, mask, kNumaMaxNode, 0) == 0) return true;
    error_number = errno;
    return false;
}

#endif

}  // namespace strata::core::detail
