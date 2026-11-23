// src/inkling2/ink2_image.hpp - Inkling-2's pictures on the host (plain C++, no GPU): decode, the processor's
// preprocessing, and the ViT's input patches.  The ViT itself is ink2_vision.cuh.
//
// transformers' Inkling2ImageProcessor (huggingface/transformers#51050) never resizes: the picture is padded at the
// right and the bottom to whole 40 x 40 tiles with -1 on the 0-255 scale, scaled to [-1, 1] ((p - 127.5) / 127.5),
// and each tile is one image token (rows x cols tokens, raster order).  The ViT cuts a tile into 2 x 2 native patches
// of 20 x 20 pixels, each a 1200-vector in (y, x, channel) order.  A picture with an alpha channel is composited onto
// the processor's checkerboard (8 px squares of 204 / 153) first.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ink2img {

constexpr int TILE = 40, PATCH = 20, PIN = 3 * PATCH * PATCH;   // 1200 inputs per native patch

struct Picture {
    int w = 0, h = 0;             // the size the ViT reads (after the token cap)
    int src_w = 0, src_h = 0;     // as decoded
    int rows = 0, cols = 0;       // image tokens: ceil(h / 40) x ceil(w / 40)
    uint64_t key = 0;             // FNV-1a of the pixels the ViT reads and their size (what prefix reuse compares)
    std::vector<float> patches;   // [rows * cols * 4][1200]: per tile (raster order) its 2 x 2 patches, row-major
};

// Decodes a JPEG / PNG / BMP / GIF (first frame) file and prepares it.  A picture over `max_tokens` image tokens is
// scaled down to fit first (the reference never resizes: this is the engine's own cap).  "" or the error.
std::string load_picture(const std::string& path, int max_tokens, Picture& p);

// The processor's own pixel tensor (terminal/tools/inkling2/vision_oracle.py .pix: int32 rows, int32 cols, then
// float32 [rows * cols][40][40][3]), cut into the ViT's patches.  "" or the error.
std::string load_pixels(const std::string& path, Picture& p);

// %20 -> space, %25 -> % (the protocol's image=<path> keys, as serve/hy4_kind.py encode_path writes them)
std::string pct_decode(const std::string& s);

}  // namespace ink2img
