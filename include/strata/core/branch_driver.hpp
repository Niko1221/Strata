#pragma once
#include "strata/core/verify.hpp"
namespace strata::core {
struct ExpertDispatch;
// Private local pipe protocol used by the immutable-history adapter. No network
// listener and no native branch-reference database. Handles name derived state.
int branch_driver(const WeightTable&, const ModelGeometry&, SessionState&, const VerifyHits&,
                  const NativeHead*, PoolMultiFn, void*, ExpertDispatch&, int pcie_mode);
}
