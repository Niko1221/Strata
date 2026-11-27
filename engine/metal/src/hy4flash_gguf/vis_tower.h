// engine/metal/src/hy4flash_gguf/vis_tower.h - the hy4-flash vision tower: pixels in, language-model rows out.
//
// WHAT THIS OWNS. The pipelines, the per-shape dense GEMMs and the scratch for one picture, plus the order
// the stages run in. It owns no weights: those belong to the VisionShard it is opened against, and a tower
// outlives any single picture.
//
// THE STAGE SINK IS PART OF THE INTERFACE, not a debug aid bolted on. The end-to-end bound for this tower
// is calibrated against an fp64 reference, and an fp32-accumulate tower legitimately sits near it, so a
// near-threshold result has to be localised before it is called a bug. That is only possible if the
// candidate can be asked for the same intermediate points the reference dumps, named identically.
#pragma once

#include "dispatch.h"
#include "vision.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace strata::hy4_flash {

/// Called once per stage, in computation order, with that stage's rows after the GPU has produced them.
/// The names are the reference's: emb0, L0..L<layers-1>, pre_norm, mm0, mm2.
using StageSink = std::function<void(const std::string& name, const float* data, int64_t rows, int64_t cols)>;

/// The patch and cell grid one picture takes. Derived, never guessed: every field follows from the shard's
/// geometry and the picture's size, and `check_picture_size` has already refused the sizes that would make
/// any of these inexact.
struct PictureGrid {
    int32_t gh = 0, gw = 0;      ///< patches down and across
    int64_t n = 0;               ///< gh * gw: patches, each one a token of the encoder
    int32_t oh = 0, ow = 0;      ///< cells down and across, i.e. gh / merge, gw / merge
    int64_t cells = 0;           ///< oh * ow
    int64_t rows = 0;            ///< oh * (ow + 1): the rows the language model reads
};

PictureGrid picture_grid(const VisionGeometry& g, int32_t h, int32_t w);

class VisionTower {
public:
    /// Builds every pipeline and dense shape the shard's geometry implies. `lib` is the hy4-flash library, which
    /// already holds hy4_flash_vision.metal (hy4_flash_kernel_files lists it unconditionally).
    /// Returns "" or the reason.
    std::string open(mtl::Context& cx, const VisionShard& s, int lib);

    /// Encodes one picture. `rgb` is h * w * 3 bytes, row-major, 8 bits a channel; `out` is resized to
    /// grid.rows * out_hidden floats. With `sink`, each stage is flushed and handed over before the next
    /// one starts, which serialises the tower - that is the price of exact correspondence, and it is only
    /// paid when a caller asks.
    /// Returns "" or the reason.
    std::string encode(mtl::Context& cx, const VisionShard& s, const uint8_t* rgb, int32_t h, int32_t w,
                       std::vector<float>& out, const StageSink& sink = {});

    void close(mtl::Context& cx);

private:
    /// A dense shape, keyed by the tensor name it was built for. Every one is built with `allow_mm`, which
    /// means hy4_flash_mm's MMA for a wide chunk and the dense GEMV otherwise - and a shape whose K is not a
    /// multiple of 32 (ffn_down's 4304) keeps the GEMV for every chunk width, which is where it is exact.
    const DenseGemvF32* shape(const std::string& tensor) const;

    /// Grows `b` to at least `floats` floats, keeping it when it is already large enough.
    std::string need(mtl::Context& cx, mtl::Buffer& b, int64_t floats, const char* what);

    VisionGeometry geo_;
    int lib_ = -1;
    std::map<std::string, DenseGemvF32> dense_;
    mtl::Pipeline patchify_, posemb_, ln_, score_, softmax_, pv_, merge_, rows_;
    mtl::Pipeline act_tanh_, act_erf_, rmsnorm_, add_row_, add_f32_;
    mtl::Buffer rgb_, px_, x_, t_, q_, k_, v_, f_, sc_, cell_, mid_, co_, out_;
    int64_t sc_floats_ = 0;
};

}  // namespace strata::hy4_flash
