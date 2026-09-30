// strata/kernels/mrope.hpp - multimodal rotary positions (the vision path).
//
// qwen4exp rotates with interleaved M-RoPE (llama.cpp LLAMA_ROPE_TYPE_IMROPE, rope.dimension_sections 11/11/10/0):
// each of the 32 rotated pairs takes its position from one of three streams, time t, height h and width w.  Text
// has t = h = w, which is why every rope kernel here takes one position per row.  An image does not: its tokens
// share t and differ in h and w (mtmd: t = p, h = p + y, w = p + x), and the text after it continues at
// p + max(nx, ny), not at its cell index.
//
// The table: nullptr (the default) keeps every kernel exactly as before - the position a caller passes is the
// rotary position.  Set, it is a DEVICE int32 [cells][3] (t, h, w) and the position a caller passes is read as a
// CELL index; pair i rotates by tab[cell * 3 + mrope_sector(i)].  Every caller in the engine passes cell indices
// (pos_base is 0 everywhere), so the table covers the prompt path, the verify window and the MTP drafter at once.
// The pointer must be set before any CUDA graph is captured (kernels take it as an argument); the contents may
// change between requests.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// The table of the CURRENT device (a layer split sets one per device; null = the identity).
void mrope_table_set(const int32_t* device_table);
const int32_t* mrope_table();

/// The rotation angles, 32 pairs of a 64-wide rotary slice: the session's float64 table (build_rope_table,
/// [max_pos][32] cos and sin in VRAM) of the CURRENT device.  Every rope kernel reads it, so the prompt path's keys
/// and the decode queries rotate by the same exact angles; fast-math cosf/sinf of pos * powf(...) was 0.0014 rad off
/// at 32K and ~0.02 at 262K, and the prompt path (precise libm) and decode (fast-math) disagreed with each other.
struct RopeTab {
    const float* cos = nullptr;
    const float* sin = nullptr;
    int max_pos = 0;
    double base = 0.0;
    float theta_scale = 0.0f;   ///< STRATA_ROPE_LEGACY=1 (an A/B arm): the old float angle, each file's cosf/sinf
};
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, double base);
/// The registered table when it was built with `freq_base`; otherwise none, and the kernels compute in float64.
RopeTab rope_table_for(double freq_base);

#if defined(__CUDACC__) || defined(__HIPCC__)
/// ggml rope_multi, is_imrope, sections {11, 11, 10, 0}: sector = pair % 32; sector % 3 == 1 -> h (sector < 33),
/// == 2 -> w (sector < 30), == 0 -> t (sector < 33).  For pairs 0..31 all three bounds hold, so it is pair % 3.
__device__ __forceinline__ int mrope_pos(const int32_t* tab, int pos, int pair) {
    return tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos;
}
/// cos and sin of rotary position p, pair `pair` (0..31): the table's float64 values, or float64 on the device
/// past its end (build_rope_table's formula)
__device__ __forceinline__ void rope_cs(const RopeTab& t, int p, int pair, float& c, float& s) {
    if (t.theta_scale != 0.0f) {
        const float th = (float) p * powf(t.theta_scale, (float) pair);
        c = cosf(th);
        s = sinf(th);
        return;
    }
    if (t.cos != nullptr && p >= 0 && p < t.max_pos) {
        c = __ldg(t.cos + (size_t) p * 32 + pair);
        s = __ldg(t.sin + (size_t) p * 32 + pair);
        return;
    }
    const double ang = (double) p * pow(t.base, -2.0 * (double) pair / 64.0);
    double sd, cd;
    sincos(ang, &sd, &cd);
    c = (float) cd;
    s = (float) sd;
}
#endif

}  // namespace strata::kernels
