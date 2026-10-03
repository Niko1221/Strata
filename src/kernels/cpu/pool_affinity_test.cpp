#include "strata/kernels/cpu/pool.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>

int main() {
    const auto cores = strata::kernels::cpu::physical_cores(false);
    if (cores.empty()) {
        std::fprintf(stderr, "Windows affinity test: topology returned no processors\n");
        return 1;
    }

    GROUP_AFFINITY original{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &original)) {
        std::fprintf(stderr, "Windows affinity test: GetThreadGroupAffinity failed: %lu\n",
                     (unsigned long) GetLastError());
        return 1;
    }

    bool tested_other_group = false;
    bool tested_current_group = false;
    for (int core : cores) {
        const WORD expected_group = (WORD) (core / 64);
        if ((expected_group == original.Group && tested_current_group) ||
            (expected_group != original.Group && tested_other_group)) continue;
        const auto previous = strata::kernels::cpu::pin_current_thread(core);
        if (!previous.valid) return 1;

        GROUP_AFFINITY current{};
        const bool queried = GetThreadGroupAffinity(GetCurrentThread(), &current) != 0;
        const KAFFINITY expected_mask = KAFFINITY(1) << (core & 63);
        const bool pinned = queried && current.Group == expected_group && current.Mask == expected_mask;
        strata::kernels::cpu::restore_thread_affinity(previous);

        GROUP_AFFINITY restored{};
        const bool restored_ok = GetThreadGroupAffinity(GetCurrentThread(), &restored) != 0 &&
                                 restored.Group == original.Group && restored.Mask == original.Mask;
        if (!pinned || !restored_ok) {
            std::fprintf(stderr, "Windows affinity test: pin/restore failed for group %u processor %d\n",
                         (unsigned) expected_group, core & 63);
            return 1;
        }
        if (expected_group != original.Group) tested_other_group = true;
        else tested_current_group = true;
        if (tested_current_group && (GetActiveProcessorGroupCount() <= 1 || tested_other_group)) break;
    }

    if (GetActiveProcessorGroupCount() > 1 && !tested_other_group) {
        std::fprintf(stderr, "Windows affinity test: topology did not expose a processor outside group %u\n",
                     (unsigned) original.Group);
        return 1;
    }
    std::printf("Windows affinity test: pin and restore passed%s\n",
                tested_other_group ? " across processor groups" : " within the current processor group");
    return 0;
}
