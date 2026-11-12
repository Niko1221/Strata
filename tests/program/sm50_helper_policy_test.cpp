#ifdef NDEBUG
#undef NDEBUG
#endif
#include "strata/core/sm50_helper_policy.hpp"

#include <cassert>

int main() {
    using strata::core::sm50_primary_or_stage_supported;
    assert(!sm50_primary_or_stage_supported(50));
    assert(!sm50_primary_or_stage_supported(51));
    assert(sm50_primary_or_stage_supported(52));
    assert(sm50_primary_or_stage_supported(60));
    assert(sm50_primary_or_stage_supported(70));
    assert(sm50_primary_or_stage_supported(86));
}
