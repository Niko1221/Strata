#pragma once

#include <string>

namespace strata::core {

// ROCm reports feature suffixes such as "gfx90a:sramecc+:xnack-".
inline std::string hip_base_arch(const char* raw_arch) {
    const std::string raw = raw_arch ? raw_arch : "";
    return raw.substr(0, raw.find(':'));
}

// Pure host policy: an empty result means the device matches a compiled base
// architecture and the backend's physical wave width. Feature loading remains
// the runtime's responsibility; a matching prefix is never enough.
inline std::string hip_arch_problem(const char* raw_arch, int actual_wave,
                                    const char* compiled_archs, int expected_wave) {
    const std::string raw = raw_arch ? raw_arch : "";
    const std::string arch = hip_base_arch(raw_arch);
    if (arch.empty() || raw.find_first_of(" \t\r\n") != std::string::npos)
        return "HIP device returned an empty or malformed architecture";
    const std::string compiled = compiled_archs ? compiled_archs : "";
    bool found = false;
    for (size_t begin = 0; begin < compiled.size();) {
        const size_t end = compiled.find(',', begin);
        const std::string target = compiled.substr(begin, end == std::string::npos ? end : end - begin);
        if (hip_base_arch(target.c_str()) == arch) found = true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    if (!found)
        return "architecture " + arch + " is not in the compiled HIP targets (" + compiled + ")";
    if (actual_wave != expected_wave)
        return "this HIP backend requires wave" + std::to_string(expected_wave) +
               "; the device reports wave" + std::to_string(actual_wave);
    return {};
}

}  // namespace strata::core
