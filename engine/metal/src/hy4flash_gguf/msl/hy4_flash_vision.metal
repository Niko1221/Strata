// engine/metal/src/hy4flash_gguf/msl/hy4_flash_vision.metal - the hy4-flash vision tower's own kernels.
//
// NUMERICS: fp32 activations, fp32 accumulation. The tower is neither decode nor the text prefill, so no
// bitwise constraint applies to it; what does apply is that every matrix in the mmproj is BF16 or F32 and
// hy4_flash_mm.metal already widens both exactly, so the tower's error is accumulation order alone.
//
// THE TWO GELUs ARE DIFFERENT AND THAT IS NOT A DETAIL. The encoder FFN uses the TANH approximation (the
// reference's bias_activation_fusion) and the projector uses the EXACT erf form (nn.GELU()). The GGUF's
// `clip.use_gelu` is a single flag and cannot express the split, so it is spelled out here. They are
// exported under two names rather than selected by a function constant, so a call site NAMES the variant
// it wants and a test can assert which one it asked for - an output-side check cannot do that job, because
// the projector attenuates the difference on its way to the embeddings.
//
// MSL has no erf. The erfc fit below follows the upstream hy4-flash Metal tower
// (github.com/Niko1221/Strata, engine/src/hy4flash/metal/hy4_vision.metal, MIT), which cites Numerical
// Recipes for it and states a fractional error under 1.2e-7 - four orders inside the ~1e-3 difference
// between the two GELU forms, which is what makes the per-variant check decisive at any precision.

/// tanh overflows to NaN for a large argument in MSL, so the argument is clamped; tanh(10) is 1 - 4e-9,
/// i.e. the clamp is below the representable difference from 1.
static inline float hy4_flash_vis_gelu_tanh(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(clamp(inner, -10.0f, 10.0f)));
}

static inline float hy4_flash_vis_erf(float x) {
    const float z = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f +
        t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
        t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f +
        t * 0.17087277f))))))));
    const float e = 1.0f - t * exp(poly);           // 1 - erfc(z)
    return x >= 0.0f ? e : -e;
}

static inline float hy4_flash_vis_gelu_erf(float x) {
    return 0.5f * x * (1.0f + hy4_flash_vis_erf(x * 0.7071067811865476f));
}

struct Hy4FlashVisActArgs {
    uint n;        ///< elements, not rows: the activation is elementwise and the caller flattens
};

/// In place: y[i] = gelu(y[i]). Elementwise, so in-place is safe and no second buffer is sized for it.
/// The bias a call site needs first is hy4_flash_add_row's job; keeping them apart keeps each one testable.
template <bool ERF>
kernel void hy4_flash_vis_act(device float* y [[buffer(0)]], constant Hy4FlashVisActArgs& a [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n) return;
    const float v = y[gid];
    y[gid] = ERF ? hy4_flash_vis_gelu_erf(v) : hy4_flash_vis_gelu_tanh(v);
}

template [[host_name("hy4_flash_vis_act_tanh")]] kernel decltype(hy4_flash_vis_act<false>) hy4_flash_vis_act<false>;
template [[host_name("hy4_flash_vis_act_erf")]] kernel decltype(hy4_flash_vis_act<true>) hy4_flash_vis_act<true>;

// ====================================================================================================
// The tower's own kernels.  Every linear layer is hy4_flash_mm's (BF16 or F32 weights, fp32 activations, fp32
// accumulation); every bias is hy4_flash_add_row's; the residual adds are hy4_flash_add_f32's and the projector's
// pre-norm is hy4_flash_rmsnorm's.  What is left is what those cannot express.
//
// SHAPES, in the names used throughout: N patches = gh * gw; D = hidden; HD = D / heads; FF = ffn;
// oh, ow = gh / 2, gw / 2; M2 = oh * ow; OD = out_hidden.  Attention is FULL and bidirectional - no mask,
// no window, no rope - so the score matrix is blocked over query tokens by the host, never materialised
// whole.

struct Hy4FlashVisPatchArgs {
    uint width;      ///< the image's pixel width, i.e. the RGB row stride in pixels
    uint gw;         ///< patches across
    uint patch;      ///< pixels a patch side
};

/// rgb [h][width][3] u8 -> x [N][3 * patch * patch] fp32, column c * patch^2 + dy * patch + dx, patches in
/// raster order.  The normalisation is `(v / 255 - 0.5) / 0.5` and not the algebraically equal
/// `v / 127.5 - 1`: the two are different fp32 expressions and the reference uses the first.
/// grid N thread groups, HY4_FLASH_TG threads.
kernel void hy4_flash_vis_patchify(device const uchar* rgb [[buffer(0)]], device float* x [[buffer(1)]],
                             constant Hy4FlashVisPatchArgs& a [[buffer(2)]],
                             uint p [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]]) {
    const uint P = a.patch, per = 3u * P * P;
    const uint py = p / a.gw, px = p % a.gw;
    for (uint e = tid; e < per; e += HY4_FLASH_TG) {
        const uint c = e / (P * P), r = e % (P * P), dy = r / P, dx = r % P;
        const ulong src = (ulong) ((py * P + dy) * a.width + px * P + dx) * 3u + c;
        const float v = float(rgb[src]) * (1.0f / 255.0f);
        x[(ulong) p * per + e] = (v - 0.5f) / 0.5f;
    }
}

struct Hy4FlashVisPosArgs {
    uint gh, gw;     ///< the patch grid this image needs
    uint edge;       ///< the learned grid's side (image_size / patch)
    uint dim;        ///< channels
    float inv_sy;    ///< edge / (gh + 0.1), computed in double on the host
    float inv_sx;    ///< edge / (gw + 0.1)
};

/// The learned `edge x edge` position grid, bilinearly resampled to gh x gw and ADDED to x in place.
///
/// align_corners=False, i.e. source coordinate `(out + 0.5) * inv_s - 0.5`, indices floored then clamped
/// into the grid and the fractional weight clamped to [0, 1] - which is what makes the edges behave.  The
/// scale is `(n + 0.1) / edge` on purpose: the 0.1 is what makes `floor(edge * s)` land on n exactly.
/// The grid's rows are [row = y * edge + x][channel] and there is NO leading CLS row; the loader refuses a
/// shard whose row count says otherwise.
/// grid N thread groups, HY4_FLASH_TG threads.
kernel void hy4_flash_vis_posemb(device const float* grid [[buffer(0)]], device float* x [[buffer(1)]],
                           constant Hy4FlashVisPosArgs& a [[buffer(2)]],
                           uint p [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    const uint y = p / a.gw, xx = p % a.gw;
    const int edge = int(a.edge);
    float fy = (float(y) + 0.5f) * a.inv_sy - 0.5f;
    float fx = (float(xx) + 0.5f) * a.inv_sx - 0.5f;
    int y0 = int(floor(fy)), x0 = int(floor(fx));
    int y1 = clamp(y0 + 1, 0, edge - 1), x1 = clamp(x0 + 1, 0, edge - 1);
    y0 = clamp(y0, 0, edge - 1);
    x0 = clamp(x0, 0, edge - 1);
    const float wy = clamp(fy - float(y0), 0.0f, 1.0f), wx = clamp(fx - float(x0), 0.0f, 1.0f);
    device const float* r00 = grid + (ulong) (y0 * edge + x0) * a.dim;
    device const float* r01 = grid + (ulong) (y0 * edge + x1) * a.dim;
    device const float* r10 = grid + (ulong) (y1 * edge + x0) * a.dim;
    device const float* r11 = grid + (ulong) (y1 * edge + x1) * a.dim;
    device float* dst = x + (ulong) p * a.dim;
    for (uint c = tid; c < a.dim; c += HY4_FLASH_TG)
        dst[c] += (1.0f - wy) * ((1.0f - wx) * r00[c] + wx * r01[c]) +
                  wy * ((1.0f - wx) * r10[c] + wx * r11[c]);
}

struct Hy4FlashVisLnArgs {
    uint n;          ///< elements in a row
    uint x_stride, y_stride;
    float eps;
};

/// LayerNorm with a BIASED variance and an affine weight AND bias - the encoder's norm, which is not the
/// projector's: `mm.pre_norm` is an RMSNorm (weight only) and uses hy4_flash_rmsnorm instead.  Subtracting a mean
/// there, or failing to subtract one here, is wrong by an amount that still looks like an embedding.
/// grid rows, HY4_FLASH_TG threads.
kernel void hy4_flash_vis_ln(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                       device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                       constant Hy4FlashVisLnArgs& a [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HY4_FLASH_TG_SG];
    device const float* xr = x + (ulong) row * a.x_stride;
    device float* yr = y + (ulong) row * a.y_stride;
    float s = 0;
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) s += xr[i];
    const float mean = hy4_flash_tg_sum(s, part, sg, lane) / float(a.n);
    float v = 0;
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    // hy4_flash_tg_sum barriers AFTER its write to `part` and not before, so it is safe to call once a thread
    // group and no more: a second call's write would race its predecessor's read loop. (hy4_flash_tg_sum_u, the
    // integer twin the indexer calls in a loop, carries the second barrier for exactly this reason.) This
    // is the one kernel in the tower that needs two sums of the same row, so the barrier is here.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = precise::rsqrt(hy4_flash_tg_sum(v, part, sg, lane) / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

struct Hy4FlashVisAttnArgs {
    uint n_keys;     ///< N: every patch is a key, and every query sees all of them
    uint n_query;    ///< this block's query count (the last block is short)
    uint q0;         ///< the block's first query
    uint heads, head_dim;
    uint qkv_stride; ///< floats between tokens of q / k / v, i.e. D
    uint s_stride;   ///< floats between score rows, i.e. n_keys rounded up by the host
    float scale;     ///< 1 / sqrt(head_dim)
};

/// S[h][t][j] = sum_d q[q0 + t][h * HD + d] * k[j][h * HD + d], UNSCALED - the scale is folded into the
/// softmax so the scores keep their own magnitude here.
/// grid (n_query, heads) thread groups, HY4_FLASH_TG threads; the threads split the keys.
kernel void hy4_flash_vis_score(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                          device float* s [[buffer(2)]], constant Hy4FlashVisAttnArgs& a [[buffer(3)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.x, h = tgp.y, HD = a.head_dim;
    device const float* qr = q + (ulong) (a.q0 + t) * a.qkv_stride + h * HD;
    device float* sr = s + ((ulong) h * a.n_query + t) * a.s_stride;
    for (uint j = tid; j < a.n_keys; j += HY4_FLASH_TG) {
        device const float* kr = k + (ulong) j * a.qkv_stride + h * HD;
        float acc = 0;
        for (uint d = 0; d < HD; ++d) acc += qr[d] * kr[d];
        sr[j] = acc;
    }
}

/// softmax over the first n_keys of each score row, with `scale` applied before the exponent; the columns
/// past n_keys are set to zero so the P V product can read a padded row.
/// grid (n_query, heads) thread groups, HY4_FLASH_TG threads.
kernel void hy4_flash_vis_softmax(device float* s [[buffer(0)]], constant Hy4FlashVisAttnArgs& a [[buffer(1)]],
                            uint2 tgp [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HY4_FLASH_TG_SG];
    threadgroup float mx[HY4_FLASH_TG_SG];
    device float* sr = s + ((ulong) tgp.y * a.n_query + tgp.x) * a.s_stride;
    float m = -INFINITY;
    for (uint j = tid; j < a.n_keys; j += HY4_FLASH_TG) m = max(m, sr[j] * a.scale);
    // max is exact and order-independent in fp32, so a plain simd_max and a fold over the simdgroups is
    // not a reduction-order choice the way a sum would be.
    m = simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) mx[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort g = 0; g < HY4_FLASH_TG_SG; ++g) m = max(m, mx[g]);
    float sum = 0;
    for (uint j = tid; j < a.n_keys; j += HY4_FLASH_TG) {
        const float e = precise::exp(sr[j] * a.scale - m);
        sr[j] = e;
        sum += e;
    }
    const float inv = 1.0f / hy4_flash_tg_sum(sum, part, sg, lane);
    for (uint j = tid; j < a.s_stride; j += HY4_FLASH_TG) sr[j] = j < a.n_keys ? sr[j] * inv : 0.0f;
}

/// O[q0 + t][h * HD + d] = sum_j P[h][t][j] * v[j][h * HD + d].
/// grid (n_query, heads) thread groups, HY4_FLASH_TG threads; the threads split the head's dimensions.
kernel void hy4_flash_vis_pv(device const float* p [[buffer(0)]], device const float* v [[buffer(1)]],
                       device float* o [[buffer(2)]], constant Hy4FlashVisAttnArgs& a [[buffer(3)]],
                       uint2 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.x, h = tgp.y, HD = a.head_dim;
    device const float* pr = p + ((ulong) h * a.n_query + t) * a.s_stride;
    device float* orow = o + (ulong) (a.q0 + t) * a.qkv_stride + h * HD;
    for (uint d = tid; d < HD; d += HY4_FLASH_TG) {
        float acc = 0;
        for (uint j = 0; j < a.n_keys; ++j) acc += pr[j] * v[(ulong) j * a.qkv_stride + h * HD + d];
        orow[d] = acc;
    }
}

struct Hy4FlashVisMergeArgs {
    uint gw;         ///< patches across, so a cell's four patches are (2y + dy) * gw + (2x + dx)
    uint ow;         ///< cells across
    uint dim;        ///< channels
    uint merge;      ///< 2
};

/// The 2x2 spatial merge: cell (y, x) of the oh x ow grid becomes one row of dim * merge^2, at column
/// `c * merge^2 + dy * merge + dx`.  That order is mm.0's own memory order - the projector's first layer is
/// a stride-2 conv2d whose weight is [out][in][kh][kw] - so transposing dy and dx here is wrong by an
/// amount that still produces rows.  Exact, never padded: both image sides are multiples of patch * merge,
/// so gh and gw are even.
/// grid M2 thread groups, HY4_FLASH_TG threads.
kernel void hy4_flash_vis_merge(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                          constant Hy4FlashVisMergeArgs& a [[buffer(2)]],
                          uint cell [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    const uint m = a.merge, cy = cell / a.ow, cx = cell % a.ow;
    const uint per = a.dim * m * m;
    device float* dst = out + (ulong) cell * per;
    for (uint e = tid; e < per; e += HY4_FLASH_TG) {
        const uint c = e / (m * m), r = e % (m * m), dy = r / m, dx = r % m;
        dst[e] = x[(ulong) ((cy * m + dy) * a.gw + cx * m + dx) * a.dim + c];
    }
}

struct Hy4FlashVisRowsArgs {
    uint oh, ow;     ///< cells down and across
    uint dim;        ///< out_hidden
};

/// Merged cells -> the rows the language model reads: cell m goes to row `m + m / ow`, and the learned
/// `v.image_newline` fills column ow of each of the oh rows.  So one picture is oh * (ow + 1) rows, which
/// is what the prompt's run of image-pad tokens has to be.
/// grid oh * (ow + 1) thread groups, HY4_FLASH_TG threads.
kernel void hy4_flash_vis_rows(device const float* cells [[buffer(0)]], device const float* newline [[buffer(1)]],
                         device float* rows [[buffer(2)]], constant Hy4FlashVisRowsArgs& a [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    const uint r = row / (a.ow + 1), c = row % (a.ow + 1);
    device float* dst = rows + (ulong) row * a.dim;
    if (c == a.ow) {
        for (uint i = tid; i < a.dim; i += HY4_FLASH_TG) dst[i] = newline[i];
        return;
    }
    device const float* src = cells + (ulong) (r * a.ow + c) * a.dim;
    for (uint i = tid; i < a.dim; i += HY4_FLASH_TG) dst[i] = src[i];
}
