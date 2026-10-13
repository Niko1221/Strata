// Does ggml-common.h give up its codebook tables to a plain C++ translation unit?  The peer kernels
// need iq2xxs_grid / iq2s_grid / iq3s_grid / kvalues_iq4nl, and the CUDA kernels get them by including
// this header, so the OpenCL side should too rather than carrying a copy of 2800 literals.
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <cstdio>

int main() {
    std::printf("iq2xxs_grid[0]  = 0x%08x   n=%llu\n", (unsigned) iq2xxs_grid[0],
                (unsigned long long) (sizeof(iq2xxs_grid) / sizeof(iq2xxs_grid[0])));
    std::printf("iq3s_grid[0]    = 0x%08x   n=%llu\n", (unsigned) iq3s_grid[0],
                (unsigned long long) (sizeof(iq3s_grid) / sizeof(iq3s_grid[0])));
    std::printf("iq2s_grid[0]    = 0x%016llx n=%llu\n", (unsigned long long) iq2s_grid[0],
                (unsigned long long) (sizeof(iq2s_grid) / sizeof(iq2s_grid[0])));
    std::printf("kvalues_iq4nl[] = %d %d %d %d  n=%llu\n", (int) kvalues_iq4nl[0], (int) kvalues_iq4nl[1],
                (int) kvalues_iq4nl[14], (int) kvalues_iq4nl[15],
                (unsigned long long) (sizeof(kvalues_iq4nl) / sizeof(kvalues_iq4nl[0])));
    return 0;
}
