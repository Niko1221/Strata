#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::core::detail {

// Reserve a job for this routed entry. The caller fills one token slot and
// increments nt; jobs and the per-expert last index are reused across layers.
// A repeated ID can appear more than once per token. Keep every routed output,
// but start another job before writing beyond the fixed MAXT token arrays.
inline int claim_host_expert_job(std::vector<kernels::cpu::ExpertJobMulti>& jobs,
                                 int& count, int16_t& last) {
    if (last < 0 || jobs[(size_t) last].nt >= kernels::cpu::MAXT) {
        if (count < 0 || (size_t) count >= jobs.size() || count > (std::numeric_limits<int16_t>::max)())
            throw std::length_error("SYCL host expert job capacity exceeded");
        last = (int16_t) count++;
        jobs[(size_t) last] = kernels::cpu::ExpertJobMulti{};
    }
    return last;
}

}  // namespace strata::core::detail
