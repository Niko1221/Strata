// include/strata/kernels/kv_q4.hpp - Q4_0 KV storage with Walsh-Hadamard rotation for the QSA layers (`--kv q4_0`).
//
// From PR #21 (code-martin). K and V are rotated by the orthonormal 256-point Hadamard matrix H before they are
// quantized to ggml's q4_0 (32 values per block: an fp16 scale + 16 bytes of 4-bit codes, 144 B per head per cell,
// 576 B per cell for K and V of both heads, vs 1,056 B in INT8). The rotation spreads a head's outlier channels
// over all 256 dimensions, which is what makes 4 bits usable. The query is rotated the same way, so
// <Hq, Hk> = <q, k> and the scores are unchanged, and the attention output (a mix of rotated values) is rotated
// back with H (self-inverse). What is lossy is the 4-bit rounding itself: measured, not assumed, in
// bench/results/2026-09-27-kv-q4.
//
// With KV streaming (--kv-resident) the codes live in the host copy and the resident slots like INT8's (KvHostPools
// k_q4/v_q4, the same block granule), so both options combine.
//
// `--kv q4lm32` and `--kv q4lm32s` (the `Q4Mode` argument of every function here) keep all of that byte for byte -
// the 18-byte block, 144 B per head, the rotation, the page/streaming granule, the snapshot layout - and change only
// the rule that turns a nibble into a value. See the level table below.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int QK4_0 = 32;

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;             // scale (fp16 bits)
    uint8_t qs[QK4_0 / 2];  // 32 4-bit codes: element j in the low nibble of qs[j], j + 16 in the high one
};
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// The block's size travels with the mode, not with a process-wide variable: 18 bytes for every
// shipped mode, 20 for the shear (its block carries the constant).  The sizing helpers below take
// the mode, and every call site passes the one its state or its arm is in (kv_q4.hpp,
// `kv_q4_bytes_per_head`); a mode-less helper would let a shear state report 18-byte rows.

// ---- `--kv q4lm32` / `--kv q4lm32s`: the same blocks, Max-Lloyd's levels instead of ggml's uniform grid ---------
//
// q4_0's grid is uniform - value = (code - 8) * d, one d per 32 values - so its 4-bit error is the error of a
// *uniform* quantizer on a source the Hadamard rotation has made roughly Gaussian (each rotated coordinate is a sum
// of many independent terms). A uniform grid is not the best 16-level quantizer for a Gaussian.
//
// The levels below are Max-Lloyd's 16 reconstruction points for N(0,1) - the table quant.cpp uses, from TurboQuant's
// tq_codebook.c - multiplied by 32 and rounded, so they are integers. That is deliberate: q4_0's attention path
// enters `code - 8` as an int8 MMA operand, and int8 carries these unchanged, which is why neither format needs a new
// layout, block size or byte count. They differ only in how a block's scale is chosen:
//
//   Q4Mode::kLloyd    `--kv q4lm32`   d = max|x| / 87, so the widest level sits on the block maximum
//   Q4Mode::kLloydLs  `--kv q4lm32s`  d = sum(l * x) / sum(l * l) over the block's own 32 values - one least-squares
//                                     pass, the scale that best fits this block to the very same levels
//
// Both exist because they are two levers, and the scale turns out to be the larger one. Measured (kv_q4_parity,
// gfx1200: 487 drawn cells x 2 KV heads x 256 values x K and V = 498,688 rotated values, 15,584 blocks; MSE, and
// dB under q4_0's 2.938e-02):
//
//   q4_0    kUniform            2.938e-02     0 dB
//   q4lm32  kLloyd   max|x|/87  2.740e-02   -0.30 dB
//   q4lm32s kLloydLs one pass   2.551e-02   -0.61 dB
//   (ceiling) a per-block scale fitted to convergence  2.456e-02   -0.78 dB
//
// So the level table alone is worth 0.30 dB and this block's own fit another 0.31 dB. One least-squares pass takes
// 79% of the way to a converged fit, and the scale it lands on is only 0.4% from max|x| / 87 on average over 15,584
// blocks - what the pass buys is the *fit*, not a wider scale. Neither gain is a fixed rescale of the levels: with
// the block's own RMS / 32 as the scale the error is 3.547e-02, WORSE than q4_0 (-0.82 dB), because the block
// maximum of 32 samples sits below the 2.72 sigma the widest level is drawn for (measured: mean max|x| / RMS =
// 2.363, and max|x| / 87 is on average 0.869 of rms / 32). The adaptation has to be per block, and both formats
// are.
//
// Note for a shipping decision: a divergent index into these 16 bytes is the access pattern `__constant__` is worst
// at - s_gemv.cu measured 2.12x for IQ4_NL's codebook while it sat in `__constant__` instead of shared (its probe:
// attn_qkv, IQ4_XS, 10240 x 2560, 0.2032 ms against 0.0960 ms). kLloydLs reads a level per lane twice (the fit's
// `l`, then the map the reader applies), and qsa_decode_attn.cu is hotter than this writer, so if either format
// ships, this table belongs in shared memory.
enum class Q4Mode : int {
    kUniform = 0,  ///< `--kv q4_0`: ggml's grid, value = (code - 8) * d
    kLloyd = 1,    ///< `--kv q4lm32`: the levels below, d = max|x| / 87
    kLloydLs = 2,  ///< `--kv q4lm32s`: the levels below, d = sum(l * x) / sum(l * l)
    // WP-2 probe (`--kv q4lm8`, `q4lm8k`, `q4lm8v`): the same 18-byte block, rotation and readers, with an 8-level
    // table for N(0,1) - a 3-bit *codebook* in 4-bit *bytes*. It is a quality probe, not a byte layout: the point is
    // to see whether the fixed Hadamard survives 3 bits on this model before any 3-bit packing is built. Three modes
    // so the probe can also ask which tensor of the pair the bits belong on at an equal average width (3.5 bits).
    kLloyd8 = 3,   ///< `--kv q4lm8`:  8 levels on K and V
    kLloyd8K = 4,  ///< `--kv q4lm8k`: 8 levels on K, 16 on V
    kLloyd8V = 5,  ///< `--kv q4lm8v`: 16 levels on K, 8 on V
    // Window overlay: the same blocks as the base mode, plus a side pool of exact fp16 cells. The first R
    // (sink) and the most recent S cells read exact instead of q4. S/R come from STRATA_KV_WINDOW_S/_R
    // (defaults 128/8); S + R = 0 is exactly the base mode. The side pool is always resident - streaming moves
    // only the quantized blocks - and snapshots refuse windowed states. `STRATA_KV_WINDOW=1` turns the same
    // window on for any other q4 mode (`--kv q4lm8v` + the env is the production point); this value is the
    // explicit overlay on kLloyd (`--kv q4lm32w`).
    // The pool is a ring: R cell-addressed sink rows plus S + B recent rows, B = the largest append batch (a
    // prompt chunk: every query has its own horizon, so the batch's whole window span is live at once). Recent
    // rows carry the cell they hold, so a row a rewind or resume left stale is read quantized, never as the
    // wrong cell - this is what makes the pool independent of the context length.
    kLloydWin = 6,  ///< `--kv q4lm32w`: kLloyd's blocks with an exact fp16 window (window arm)
    // The shear (bench/results/2026-11-11-shear-codec/): a UNIFORM grid with a per-block offset.
    // The reconstruction is v = s * (code - 8) + c, with c = o + 8*s and the grid [o, o + 15*s] in
    // the code's own units (code 0 is the grid's left end).  The offset is a per-block zero point:
    // it corrects the position of the grid against the block's own distribution, not a vector.
    // It is still affine, so unlike the level tables it can ride the prompt fast path (that is the
    // point of the arm: the level table is what the fast path declines).  The writer fits (o, s)
    // by alternating least squares -- kShearPasses passes -- and the constant c travels with the
    // block.  The blocks are 20 bytes (18 + the constant), so this mode has its own byte size and
    // the shipped formats keep theirs.
    kShear = 7,  ///< `--kv q4u32`: a uniform grid with a per-block offset (the zero point)
    // The same grid WITHOUT the offset: the step fitted per block, the grid centred on zero.  The
    // pair (kUniformLs, kShear) is the arm's design: the two arms differ in the offset and nothing
    // else, so its end-to-end effect is isolated.  kUniformLs is affine, so it rides the prompt
    // fast path; kShear carries a constant and therefore does not.
    kUniformLs = 8,  ///< `--kv q4u32` without STRATA_KV_ZERO=1: a uniform grid, the step fitted per block
};

/// The shear's block: the codes, the scale, and the constant (the offset in the reader's units,
/// c = o + 8*s).  20 bytes -- this mode has its own byte size, so the shipped formats keep theirs.
struct block_q4_0z {
    uint8_t qs[16];
    uint16_t d;
    uint16_t c;
};

/// Bytes per head for a mode: the shear's blocks carry the constant as well (20 bytes), so the
/// mode decides the row stride.  The shipped modes keep the 18-byte block.
inline uint64_t kv_q4_bytes_per_head(int head_dim, Q4Mode m) {
    return (uint64_t) (head_dim / QK4_0) *
           (m == Q4Mode::kShear ? sizeof(block_q4_0z) : sizeof(block_q4_0));
}

/// Bytes per cell for a mode (the shear's row is wider).
inline uint64_t kv_q4_bytes_per_cell(const QsaShapes& s, Q4Mode m) {
    return (uint64_t) s.n_head_kv * kv_q4_bytes_per_head((int) s.head_dim, m) * 2;
}

/// How many least-squares passes the shear's writer runs.  This is a DIAL, not a constant: the
/// price of each pass is measured (bench/results/2026-11-11-shear-codec/README.md, "der Preis des
/// Fits").  Six passes are the reference (they define the effectiveness gate G1: the engine must
/// reproduce the harness's block MSE exactly); two passes are the default, because the price of
/// giving up four passes (0.136 dB of block error, about 0.0006 nats end to end) sits below the
/// corpus's resolution (0.00254 nats at 128K) while the writer's price does not (35 shuffles per
/// block against 15 today, against 95 at six passes).
inline constexpr int kShearPasses = 2;

/// True for the level-table modes and for the shear: every mode whose reader needs more than
/// `code - 8`.
inline constexpr bool q4_lloyd(Q4Mode m) {
    return m == Q4Mode::kLloyd || m == Q4Mode::kLloydLs || m == Q4Mode::kLloydWin || m == Q4Mode::kLloyd8 ||
           m == Q4Mode::kLloyd8K || m == Q4Mode::kLloyd8V;
}

/// The shear: a uniform grid with a per-block offset, so its reader needs the constant too.
inline constexpr bool q4_shear(Q4Mode m) { return m == Q4Mode::kShear; }

/// The uniform grid with the step fitted per block: its reader is q4_0's (code - 8).
inline constexpr bool q4_uniform_ls(Q4Mode m) { return m == Q4Mode::kUniformLs; }

/// Every mode whose reader needs more than `code - 8`: the level tables and the shear.
inline constexpr bool q4_needs_table(Q4Mode m) { return q4_lloyd(m) || q4_shear(m); }

inline constexpr int8_t kQ4lMax = 87;   // == max |level|, the 16-level table
inline constexpr int8_t kQ4l8Max = 69;  // == max |level|, the 8-level table (2.1519 sigma)

/// The 8-level ladder is Max-Lloyd's optimum for N(0,1) (0.2451, 0.7560, 1.3439, 2.1519 sigma) rounded to int8 by the
/// same x32 rule as the 16-level table above, so the writer divides the block maximum by 69 instead of 87.
inline constexpr int8_t kQ4l8Levels[8] = {-69, -43, -24, -8, 8, 24, 43, 69};

/// Which tensor of the pair carries the 8-level table - the probe's only knob (q4l_group<FIT, N> turns it into the
/// scale divisor 69 instead of 87).
inline constexpr bool q4_l8_k(Q4Mode m) { return m == Q4Mode::kLloyd8 || m == Q4Mode::kLloyd8K; }
inline constexpr bool q4_l8_v(Q4Mode m) { return m == Q4Mode::kLloyd8 || m == Q4Mode::kLloyd8V; }

#if defined(__CUDACC__) || defined(__HIPCC__)
/// The dequant map: the nibble a q4lm32 block stores, to the integer level it stands for.
static __constant__ int8_t kQ4lMap[16] = {-87, -66, -52, -40, -30, -21, -12, -4, 4, 12, 21, 30, 40, 52, 66, 87};

/// The 8-level map. Codes 0..7 are the ladder; 8..15 are never written by an 8-level block (it stores the level
/// index 0..7) and are clamped to the widest level so a stray code cannot run away.
static __constant__ int8_t kQ4l8Map[16] = {-69, -43, -24, -8, 8, 24, 43, 69, 69, 69, 69, 69, 69, 69, 69, 69};

/// The map a block written under `m` needs on the K side / the V side. Never called for kUniform (that uses code-8).
__device__ __forceinline__ const int8_t* q4l_map_k(Q4Mode m) { return q4_l8_k(m) ? kQ4l8Map : kQ4lMap; }
__device__ __forceinline__ const int8_t* q4l_map_v(Q4Mode m) { return q4_l8_v(m) ? kQ4l8Map : kQ4lMap; }
#endif

/// Orthonormal Fast Walsh-Hadamard Transform of rows of 256 floats (scale 1/16): its own inverse.
void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
inline void fwht256_inplace_cuda(float* data, int64_t n_rows, void* stream) { fwht256_cuda(data, data, n_rows, stream); }

/// The rotation of the KV rows **with the centering** (bench/results/2026-11-10-centering/): the
/// per-(KV head, channel) offset is subtracted before the transform, in the channel space, so the
/// quantizer's scale (a maximum statistic) sees the signal without its fixed part.  The rows of the
/// buffer are (position, KV head, channel), so the head of a row is (row / 256) % n_heads.
/// `offsets` null means the plain rotation - every writer site calls this one function, so none of
/// them can forget the centering.
void fwht256_kv_inplace_cuda(float* data, int64_t n_rows, int n_heads, const float* offsets, void* stream);

/// The V add-back on the attention output: the stored V rows are (V - m_V) rotated and the attention
/// weights sum to one, so the output carries -m_V.  The output rows are (position, QUERY head,
/// channel) and the query heads are grouped per KV head (G query heads per KV head), so the row's KV
/// head is (query head) / G.  `offsets` null does nothing.
void add_center_offsets_cuda(float* data, int64_t n_rows, int n_q_heads, int g, const float* offsets,
                             void* stream);

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident.
void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host = nullptr, Q4Mode mode = Q4Mode::kUniform);

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host = nullptr, Q4Mode mode = Q4Mode::kUniform);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host = nullptr,
                  const KvHostPools* stage = nullptr, Q4Mode mode = Q4Mode::kUniform);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated). Both
/// level-table modes read alike (see q4_lloyd), so a mode here only decides table or uniform grid.
/// The trailing window params (default off) serve WP-2a's exact fp16 side pool instead of 4-bit for sink and
/// recent cells, with the horizon from step[kStepNKv] in-kernel; existing callers compile unchanged. The row is
/// `qsa_win_slot` (sinks cell-addressed, recent rows a ring guarded by `win_tag`), never the cell itself.
void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream, Q4Mode mode = Q4Mode::kUniform,
                       const uint16_t* k_win = nullptr, const uint16_t* v_win = nullptr,
                       const int32_t* win_tag = nullptr, int win_s = 0, int win_r = 0, int win_b = 0);

/// WP-2a window side pool (kLloydWin): the exact fp16 twin of the cells a windowed reader takes exact, rows
/// [R sinks][S + B recent ring] per tensor (see qsa_win_slot). Steps variant (positions from the step array, like
/// kv_append_q4_steps) and contiguous batch variant (pos0/T, like kv_append_q4) - called with the same rotated
/// K/V as the q4 append, next to it. `win_tag` gets the cell id of every recent row.
/// The INVARIANT is that one call's cells land on distinct ring rows: a row written by two cells of the same
/// launch would race its tag against its values, so n_steps (or T) must be <= S + B. A larger batch (a longer
/// prefill chunk than STRATA_KV_WINDOW_BATCH) fails loudly here, as does a geometry the grid cannot cover.
void kv_append_win_steps(uint16_t* k_win, uint16_t* v_win, int32_t* win_tag, const int32_t* step, int step_stride,
                         int n_steps, const float* kcur, const float* vcur, int n_head_kv, int head_dim,
                         void* stream, int win_s, int win_r, int win_b);
void kv_append_win(uint16_t* k_win, uint16_t* v_win, int32_t* win_tag, int64_t pos0, int64_t T, const float* K,
                   const float* V, int n_head_kv, int head_dim, void* stream, int win_s, int win_r, int win_b);

}  // namespace strata::kernels
