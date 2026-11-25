// engine/metal/src/hy4flash_gguf/vision.h - the hy4-flash vision shard: a second GGUF beside the trunk.
//
// WHAT THIS OWNS. The mmproj file's metadata and tensor descriptors, validated, plus the image geometry the
// prompt side and the engine must agree on. It owns no kernels and no scratch; the tower that consumes it
// lives beside this file.
//
// WHY IT IS A SEPARATE FILE AND NOT PART OF Model. The trunk's `Model` rewrites each dense tensor's offset
// to be relative to one resident buffer; the vision weights must stay independently mappable and
// releasable, because whether they are resident at all is a per-request decision on a unified-memory Mac.
#pragma once

#include "gguf_lite.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::hy4_flash {

/// The geometry the mmproj declares, after validation. Every field is read from the file; none is a
/// constant here, so a shard that disagrees is refused rather than reinterpreted.
struct VisionGeometry {
    int32_t hidden = 0;          ///< clip.vision.embedding_length
    int32_t heads = 0;           ///< clip.vision.attention.head_count
    int32_t head_dim = 0;        ///< hidden / heads
    int32_t ffn = 0;             ///< clip.vision.feed_forward_length
    int32_t layers = 0;          ///< clip.vision.block_count
    int32_t patch = 0;           ///< clip.vision.patch_size
    int32_t merge = 0;           ///< clip.vision.spatial_merge_size
    int32_t out_hidden = 0;      ///< clip.vision.projection_dim, which must equal the trunk's hidden size
    int32_t pos_edge = 0;        ///< clip.vision.image_size / patch: the learned position grid's side
    float eps = 0;               ///< clip.vision.attention.layer_norm_epsilon, used by both norms
    int64_t min_pixels = 0;
    int64_t max_pixels = 0;      ///< the file's limit; an engine cap may lower it
    int32_t image_token = 0;     ///< the id a picture's positions hold in the prompt
};

/// `cell` is the stride between patches of a merged cell in the flattened column index: the 2x2 window is
/// laid out channel-major, which is mm.0's own memory order.
struct VisionShard {
    VisionGeometry geo;
    gguf::File tensors;
    mtl::Buffer weights;         ///< one no-copy wrap of the mapped file; every tensor is an offset into it
    std::string path;

    /// The descriptor of `name`, or nullptr. Offsets are absolute in the mapped file.
    const gguf::Tensor* find(const std::string& name) const { return tensors.find(name); }
    /// An argument bound to `name`'s bytes.
    mtl::Arg arg(const std::string& name) const;
    bool loaded() const { return weights.handle != nullptr; }
};

/// Positions one picture of `h` x `w` pixels takes in the prompt: (h / cell) * (w / cell + 1), where the
/// extra column per row is the learned `v.image_newline` embedding. `cell` is patch * merge.
int64_t image_token_count(const VisionGeometry& g, int32_t h, int32_t w);

/// The authority's `smart_resize` target size: both sides a multiple of `factor`, the pixel count inside
/// [min_pixels, max_pixels], and the aspect ratio kept as closely as that allows.
///
/// WHY THIS IS HERE AND NOT TAKEN FROM THE REFERENCE ENGINE. llama.cpp implements smart_resize too, and
/// for this projector it differs from the authority twice: `image_round_half_even` defaults to false and
/// the HY_V4_FLASH block never sets it, so it rounds half AWAY from zero where HuggingFace's Python
/// `round` goes half to EVEN; and its resize algorithm is LANCZOS where the processor's is BICUBIC. At a
/// side whose `h / factor` lands on exactly .5 the two therefore choose different target sizes, hence
/// different patch grids and different token counts - so the geometry is gated against HuggingFace, as
/// integers, and not against the engine we compare embeddings with.
///
/// The rounding is done on the RATIONAL h/factor rather than on a float, so it is exact for every input;
/// only the two clamping branches need a square root, as the authority's own does.
/// Returns "" or the reason (only the aspect-ratio bound refuses).
std::string smart_resize_target(int32_t h, int32_t w, int32_t factor, int64_t min_pixels, int64_t max_pixels,
                                int32_t& out_h, int32_t& out_w);

/// The patch grid a picture of `h` x `w` pixels ends up with: smart_resize at `patch * merge`, then
/// divided by `patch`.
///
/// NOT FOR THE REQUEST PATH, and this is the important part. The server resizes the picture and writes the
/// RESULT into the DSI1 header; `src/hy4flash_gguf/vision.cu` reads `pic.w` / `pic.h` from that header and
/// derives the placeholder run from them, never from an original size. A splice that recomputed the
/// geometry here instead would disagree with a server that is 32 pixels off (see the surfaced divergence
/// in the report's §0.8e) and refuse a picture that works today - a validator correct by the authority,
/// placed downstream of a producer that is not, manufacturing a rejection out of nothing.
///
/// So this pair exists to be GATED, not called: it proves our arithmetic is upstream's should the engine
/// ever compute geometry itself. What the request path needs instead is a consistency check on the header
/// alone - sides at least `patch * merge` and multiples of it, the pixel cap, and a run length equal to
/// `image_token_count` of the header's own dimensions - which is what vision.cu already does.
std::string patch_grid_for(const VisionGeometry& g, int32_t h, int32_t w, int64_t max_pixels, int32_t& gh,
                           int32_t& gw);

/// Both sides a multiple of patch * merge, at least one cell, and within the shard's pixel budget.
/// Returns "" or the reason, phrased for a caller to print.
std::string check_picture_size(const VisionGeometry& g, int32_t h, int32_t w, int64_t max_pixels);

/// Opens the mmproj, validates it against the trunk, and wraps it as one no-copy buffer.
///
/// `trunk_shard1` is read for two checks that cannot be made from the mmproj alone: that the projector's
/// output width equals the trunk's hidden size, and that the image token id names a token whose text is
/// the image-pad placeholder. The second one matters because a wrong id does not fail - it produces
/// plausible text with the picture silently ignored, and nothing points at the loader.
///
/// Returns "" or the first problem, named.
std::string open_vision(VisionShard& out, mtl::Context& cx, const std::string& mmproj_path,
                        const std::string& trunk_shard1, int32_t trunk_hidden);

/// Unmaps and releases the wrap. Safe on an unopened shard.
void close_vision(VisionShard& s, mtl::Context& cx);

/// One line for the startup block.
std::string describe_vision(const VisionShard& s);

}  // namespace strata::hy4_flash
