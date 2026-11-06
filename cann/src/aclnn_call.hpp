// cann/src/aclnn_call.hpp - the one place that knows how to run a two-phase aclnn op.
//
// Every operator in this project shares a single workspace buffer and synchronizes once after the
// launch. ops.cpp owns that machinery; operators implemented in other files (the fused delta net)
// include this instead of duplicating it.
#pragma once

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <cstdint>
#include <functional>

#include "strata/cann/device.hpp"

namespace strata::cann {

using GetWs = std::function<aclnnStatus(uint64_t *, aclOpExecutor **)>;
using Launch = std::function<aclnnStatus(void *, uint64_t, aclOpExecutor *, aclrtStream)>;

void execute(Stream & stream, GetWs get_ws, Launch launch, const char * what);

} // namespace strata::cann
