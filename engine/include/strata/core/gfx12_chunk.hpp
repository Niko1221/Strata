// include/strata/core/gfx12_chunk.hpp - the prompt-chunk ceiling a bare `--prefill auto` takes on gfx12 (Linux), as pure functions so a test can pin them.
//
// 0.2.10 took 16384.  0.2.10.1 takes 32768 (owner 11-03): a 20K prompt is one chunk instead of 16384 + a 3.6K tail, and every chunk of 1,024 tokens or more
// streams all the experts the card does not hold.  R9700, IQ3_S: 20K +12.1%, 64K +2.6%, 32K +1.3%, 8K +0.9%; UD-Q4_K_XL 20K +26%; first-token KL against 16384 <= 5e-5.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::core {

/// STRATA_GFX12_CHUNK: unset -> 32768; "16" -> 16384 (the 0.2.10 ceiling); "0" -> 0 (no gfx12 ceiling: the engine's 8192); anything else (also "1") -> 32768.
inline int64_t gfx12_chunk_ceiling(const char* env) noexcept {
    if (env == nullptr) return 32768;
    if (std::strcmp(env, "16") == 0) return 16384;
    if (env[0] == '0') return 0;
    return 32768;
}

/// The RAM guard (#2090) steps the ceiling down when the extra loan does not fit: 32768 -> 16384 -> 8192.
inline int64_t gfx12_chunk_step_down(int64_t ceiling) noexcept { return ceiling > 16384 ? 16384 : 8192; }

/// Host RAM reserve (MiB) kept free by the #2090 prompt-chunk RAM guard (#6687).
/// - STRATA_PREFILL_RAM_RESERVE_MIB: explicit override in MiB (>= 0) when set.
/// - Default: 6144 MiB when --resident-experts will pin experts after this check (#2090),
///   or 3072 MiB when experts are not pinned in RAM so 64 GB desktops keep 16K/32K chunks safely (#6687).
inline uint64_t prefill_ram_reserve_mib(const char* env, bool resident_cpu_experts) noexcept {
    if (env != nullptr && env[0] != '\0') {
        char* end = nullptr;
        const long long v = std::strtoll(env, &end, 10);
        if (end != env && *end == '\0' && v >= 0) return static_cast<uint64_t>(v);
    }
    return resident_cpu_experts ? 6144ull : 3072ull;
}

}  // namespace strata::core
