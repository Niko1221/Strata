#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace strata::kernels::cpu {
// A physical core per participant; two distinct, equally sized LLC domains.
// Logical CPU numbering is not assumed (SMT/cpusets may renumber it).
inline bool ccx_shard_plan(const std::vector<int> &cpus,
                           const std::vector<int> &l3,
                           std::array<int, 6> &slice) {
  if (cpus.size() != 6 || l3.size() != 6 || l3[0] < 0)
    return false;
  int other = -1, count[2] = {};
  for (std::size_t i = 0; i < 6; ++i) {
    if (cpus[i] < 0)
      return false;
    for (std::size_t j = 0; j < i; ++j)
      if (cpus[i] == cpus[j])
        return false;
    if (l3[i] != l3[0]) {
      if (other < 0)
        other = l3[i];
      if (l3[i] != other)
        return false;
      slice[i] = 1;
    } else
      slice[i] = 0;
    ++count[slice[i]];
  }
  return count[0] == 3 && count[1] == 3;
}
// Initial affinity preference based on expert ID parity; pool dynamically
// rebalances to prevent skew.
inline int ccx_owner(int expert_id) { return expert_id & 1; }
} // namespace strata::kernels::cpu