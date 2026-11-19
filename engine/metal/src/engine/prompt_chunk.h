// metal/src/engine/prompt_chunk.h - the --prefill auto prompt call size on Metal.
//
// Window::prompt reads a prompt in row chunks of its own size (STRATA_METAL_PROMPT_CHUNK, else
// kAutoPromptChunkRows under --prefill auto, see prompt_chain.inc).  The auto call is four of those row chunks, so
// the prompt workspace stays the window's row size; the auto:N ceiling still caps it.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::metal_engine {

inline constexpr int kAutoPromptChunkRows = 1024;
inline constexpr int kAutoPromptChunksPerCall = 4;

inline int64_t auto_prompt_call_rows(int64_t inner_rows, int64_t auto_max) {
    return std::min<int64_t>(kAutoPromptChunksPerCall * inner_rows, auto_max);
}

}  // namespace strata::metal_engine
