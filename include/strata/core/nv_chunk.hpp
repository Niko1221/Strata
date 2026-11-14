// include/strata/core/nv_chunk.hpp - the prompt-chunk ceiling a bare `--prefill auto` takes on NVIDIA cards, as pure functions so a test can pin them.
//
// Until 0.2.10 a bare auto stopped at 8192 on NVIDIA and 16384 / 32768 were opt-ins (#282).  Owner 11-03 ("after the full test suite passes on it"): the
// ceiling is 32768 for cards of compute capability 8.0 and up, the gfx12 rule (generate.cpp): a 20K prompt is one chunk instead of 8192 + 8192 + 3.6K, and
// every chunk of 1,024 tokens or more streams all the experts the card does not hold.  The fit scan still picks a smaller chunk when the cache cannot lend
// the buffers, and the #2090 RAM guard steps 32768 -> 16384 -> 8192 when the RAM available does not cover the extra loan.
// Round 0212, boston A6000 UD-Q4_K_XL (13.1K of 24.6K experts in VRAM): 20K +43.4%, 32K +22.5%, 64K +21.3%, 8K 0.0%; cards that hold every expert: flat.
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::core {

/// STRATA_NV_CHUNK: unset -> 32768; "16" -> 16384; "0" -> 0 (no NVIDIA ceiling: the engine's 8192); anything else (also "1") -> 32768.
/// `defaults026`: STRATA_DEFAULTS=026 gives back the 0.2.6 behaviour (8192).
inline int64_t nv_chunk_ceiling(const char* env, bool defaults026) noexcept {
    if (defaults026) return 0;
    if (env == nullptr) return 32768;
    if (std::strcmp(env, "16") == 0) return 16384;
    if (env[0] == '0') return 0;
    return 32768;
}

/// Volta / Turing / Pascal were not measured: they keep 8192. `min_major` is the lowest compute capability major among the visible cards.
inline bool nv_chunk_card_ok(int min_major) noexcept { return min_major >= 8; }

/// The RAM guard steps the ceiling down when the extra loan does not fit: 32768 -> 16384 -> 8192.
inline int64_t nv_chunk_step_down(int64_t ceiling) noexcept { return ceiling > 16384 ? 16384 : 8192; }

}  // namespace strata::core
