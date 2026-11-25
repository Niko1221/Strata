// engine/metal/src/hy4flash_gguf/vis_bundle.h - the DSI1 picture bundle a GENI line names.
//
// FORMAT (terminal/serve/ds_pictures.py build()): "DSI1", int32 count, then per picture int64 start, int32 w,
// int32 h and w * h * 3 RGB bytes, little-endian, pictures in prompt order.
//
// THE GEOMETRY COMES FROM THE HEADER AND IS NEVER RECOMPUTED. The server resizes the picture and writes
// the result here; `src/hy4flash_gguf/vision.cu` reads `pic.w` / `pic.h` and derives the placeholder run from
// those same values. A reader that recomputed the size from an original would refuse a picture the server
// produced - the server rounds the max-pixels clamp through a float reciprocal and lands 32 pixels low on
// a known set of sizes - so a validator correct by the authority, placed here, would manufacture a
// rejection that does not exist today. The checks below are therefore all on the header's own numbers.
//
// AND THEY MATCH THE SHIPPING CONTRACT'S LOOSENESS, deliberately. vision.cu requires sides at least 32 and
// multiples of 32 and `w * h` within the cap - and NOT min_pixels, so a 32x32 header passes even though
// 1024 pixels is under the shard's 4096 floor. The server never produces one, because it clamps upward. A
// stricter check here would reject headers the CUDA engine accepts, which is the same mistake in a smaller
// costume.
#pragma once

#include "vision.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// One picture, as the bundle gives it.
struct BundlePicture {
    int64_t start = 0;    ///< index of its first placeholder token in the prompt
    int64_t count = 0;    ///< image_token_count of THIS picture's header dimensions
    int32_t w = 0, h = 0;
    std::vector<uint8_t> rgb;
};

/// Reads and validates `path` against the prompt `ids`.
///
/// Refuses: a bad magic or count; a side below `patch * merge` or not a multiple of it; `w * h` over the
/// cap (with `>`, so an exact 4194304 is accepted as every other bound in the stack accepts it); a run
/// that is not `image_token_count` of the header's own dimensions; any id in the run that is not
/// `image_token`; runs that overlap or go backwards; a truncated payload; trailing bytes.
/// Returns "" or the reason.
std::string read_bundle(std::vector<BundlePicture>& out, const std::string& path, const int32_t* ids,
                        int64_t n_ids, const VisionGeometry& g);

}  // namespace strata::hy4_flash
