#pragma once

#include <string_view>

namespace strata::program {

// A measured default, not a topology/affinity fix. See docs/benchmarks/windows-zen2-gfx1030-workers.md.
// Zero still means the pool's ordinary automatic sizing; positive user values win.
inline int select_pool_workers(int requested, int available, std::string_view cpu,
                               std::string_view arch, bool windows_hip, bool single_gpu) {
    if (requested != 0 || !windows_hip || !single_gpu || available <= 31)
        return requested;
    constexpr std::string_view model = "AMD Ryzen Threadripper 3990X";
    const auto pos = cpu.find(model);
    if (pos == std::string_view::npos || (pos != 0 && cpu[pos - 1] != ' '))
        return requested;
    const auto end = pos + model.size();
    if (end != cpu.size() && cpu[end] != ' ')
        return requested;
    // HIP may append feature flags, e.g. gfx1030:xnack-.
    if (arch != "gfx1030" && !arch.starts_with("gfx1030:"))
        return requested;
    return 31;
}

}  // namespace strata::program
