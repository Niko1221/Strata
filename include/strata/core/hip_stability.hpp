#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

// Experimental #872 mitigation. Call before any runtime/device query: ROCr
// caches allocation and copy policy at initialization. Unset means no change.
inline bool hip_stability_requested() {
    const char* value = std::getenv("STRATA_HIP_STABLE");
    return value && (std::strcmp(value, "1") == 0 || std::strcmp(value, "2") == 0);
}

inline bool hip_stability_full_kv() {
    const char* value = std::getenv("STRATA_HIP_STABLE");
    return value && std::strcmp(value, "2") == 0;
}

inline bool configure_hip_stability() {
    if (!hip_stability_requested()) return true;
    struct Setting { const char* name; const char* value; };
    const Setting settings[] = {
        {"HSA_USERPTR_FOR_PAGED_MEM", "0"},
        {"HSA_ENABLE_SDMA", "0"},
        // ROCclr's threshold is in MiB. Route even large pageable expert
        // transfers through runtime staging instead of retaining file USERPTRs.
        {"GPU_PINNED_MIN_XFER_SIZE", "1048576"},
        {"STRATA_RESIDENT_PIN", "0"},
        {"STRATA_STAGE_PIN", "0"},
        {"STRATA_SH_STREAM", "0"},
        {"STRATA_PREFILL_MMQ", "0"},
        {"STRATA_PF_FUSED", "0"},
        {"STRATA_PF_STEP_SYNC", "1"},
        {"STRATA_KV_HOST_DMA", "1"},
        {"STRATA_DOORBELL_STORE", "1"},
        {"STRATA_VERIFY_COHERENT", "1"},
        {"STRATA_REFILL_BLOCKING", "1"},
        {"STRATA_TRACE", "1"},
        {"STRATA_COMMIT_SYNC", "1"},
    };
    for (const auto& setting : settings) {
        // Explicit settings allow one-variable diagnosis without another build.
        // Print effective values so an overridden profile is never mistaken for
        // the complete conservative mode.
        const char* previous = std::getenv(setting.name);
        const bool overridden = previous != nullptr;
        if (!overridden) {
#if defined(_WIN32)
            const int result = _putenv_s(setting.name, setting.value);
#else
            const int result = setenv(setting.name, setting.value, 0);
#endif
            if (result != 0) {
                std::fprintf(stderr, "strata hip-stable: cannot set %s\n", setting.name);
                return false;
            }
        }
        std::fprintf(stderr, "strata hip-stable: %s=%s%s\n", setting.name,
                     std::getenv(setting.name), overridden ? " (explicit override)" : "");
    }
    std::fflush(stderr);
    return true;
}

// No cache loan/refill at the prompt-to-reply boundary. Own buffers fit a small
// chunk; retain token-at-a-time prefill if the caller disabled batched prefill.
inline constexpr long long hip_stability_chunk(long long chunk, bool automatic) {
    return automatic || chunk > 2048 ? 2048 : chunk;
}

}  // namespace strata::core
