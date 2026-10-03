#pragma once

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdio>

namespace strata::kernels::cpu::detail {

// `core` encodes group * 64 + processor-in-group. Shared by the pool workers, host pinning, and its test so
// the tested processor mapping is exactly the one used when a worker starts.
inline bool set_thread_group_affinity(int core, int worker = -1, GROUP_AFFINITY* previous = nullptr) {
    if (core < 0) return false;
    GROUP_AFFINITY target{};
    target.Group = (WORD) (core / 64);
    target.Mask = KAFFINITY(1) << (core & 63);
    if (SetThreadGroupAffinity(GetCurrentThread(), &target, previous)) return true;
    if (worker >= 0) {
        std::fprintf(stderr, "strata cpu pool: SetThreadGroupAffinity for worker %d (group %u, mask 0x%llx) failed: %lu; previous affinity kept\n",
                     worker, (unsigned) target.Group, (unsigned long long) target.Mask, (unsigned long) GetLastError());
    } else {
        std::fprintf(stderr, "strata cpu pool: SetThreadGroupAffinity for host (group %u, mask 0x%llx) failed: %lu\n",
                     (unsigned) target.Group, (unsigned long long) target.Mask, (unsigned long) GetLastError());
    }
    return false;
}

}  // namespace strata::kernels::cpu::detail
#endif
