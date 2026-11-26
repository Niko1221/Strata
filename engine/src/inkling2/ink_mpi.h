#pragma once

#include "ink_backend.h"
#include <functional>

namespace ink2 {
int run_mpi_experts(int& argc, char**& argv, const std::string& path, const BackendOpts& opts,
                    const std::function<int(InkModelFile&, InkBackend&)>& serve);
} // namespace ink2
