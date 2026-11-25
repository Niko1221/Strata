// engine/metal/src/hy4flash_gguf/vis_pos.h - the trunk-side positions a picture takes: 3-axis M-RoPE.
//
// WHY THIS IS ITS OWN FILE AND PURE INTEGER ARITHMETIC. It has no reference in the engine we validate
// against: llama.cpp reports LLAMA_ROPE_TYPE_NORM for hy_v4_flash and the trunk GGUF carries no
// `rope.dimension_sections`, so on positions that engine is a different model. The authority is
// HuggingFace's own get_rope_index, which is integer arithmetic over the token layout - no weights, no
// GPU, no forward pass. Keeping this file free of both means the whole thing is gated by comparing
// integers against that authority, which is the only honest way to validate it.
//
// WHAT GOES WRONG IF IT IS WRONG. A position error does not fail, does not produce NaNs and does not look
// like a bug: it produces fluent, plausible, wrong text. For one 256x256 picture the correct advance is 9
// positions while the picture occupies 72 token slots, so the most natural implementation - one position a
// token - is off by 63 for everything that follows.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// One picture's placeholder run in the prompt. `gh`, `gw` are the PATCH grid sides (pixels / patch), not
/// the merged ones; `count` is what the run must be and is derived, never supplied.
struct ImageSpan {
    int64_t start = 0;   ///< index of the run's first token
    int64_t count = 0;   ///< (gh / merge) * (gw / merge + 1)
    int32_t gh = 0, gw = 0;
};

/// Pairs each run of `image_token` in `ids` with the next picture in `pictures` (as {gh, gw} patch grids),
/// in order. Refuses a run whose length disagrees with its picture, and a count mismatch either way: a
/// prompt with one run too few would otherwise silently drop a picture's embeddings.
/// Returns "" or the reason.
std::string build_image_spans(std::vector<ImageSpan>& out, const int32_t* ids, int64_t n, int32_t image_token,
                              const std::vector<std::pair<int32_t, int32_t>>& pictures, int32_t merge);

/// The 3 x n position array, axis-major: axis `a` of token `t` is `pos[a * n + t]`.
///
/// Text advances by one on all three axes. A picture takes a block whose temporal axis is constant, whose
/// height axis is the merged row and whose width axis is the merged column - the newline column included,
/// as column `gw / merge` - and the token after the picture resumes at `max(block) + 1`.
/// Returns "" or the reason.
std::string build_mrope_positions(std::vector<int32_t>& pos, int64_t n, const std::vector<ImageSpan>& spans,
                                  int32_t merge);

}  // namespace strata::hy4_flash
