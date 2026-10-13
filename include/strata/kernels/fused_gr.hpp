// include/strata/kernels/fused_gr.hpp - plan v0.3 P3: the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row: 20.6 us for 6.5 MB.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
    /// S23 experiment (STRATA_HC_Q8=1): the GGUF's Q8_0 projections (null: the BF16 ones above); fused_gr_read_multi
    /// only
    const uint8_t* q8_down = nullptr;  ///< Q8_0 [hc_lr][hc*n_embd]
    const uint8_t* q8_up = nullptr;    ///< Q8_0 [hc*n_embd][hc_lr]
    const uint8_t* q8_inject = nullptr;///< Q8_0 [hc][hc*n_embd]
    /// S26 STRATA_QFUSE=1 (fused_gr_read_multi, the default and Q8_0 reads): also write `mixed`'s q8_1 image here (the
    /// bytes native_quantize_q8_1 would write); q8_cnt = n_embd / 32 zeroed counters owned by the caller (token 0's)
    uint8_t* q8_mixed = nullptr;
    unsigned* q8_cnt = nullptr;
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
void fused_gr_read(const FusedGrArgs& a, void* stream);
/// STRATA_HC_Q8 on a bf16 pack: `out` (n_blocks * 34 bytes) = the Q8_0 blocks (fp16 d = amax / 127, round(x / d), ggml's
/// reference rounding) of `n_blocks * 32` bf16 values on the device.
/// The inverse for a path with no Q8 kernel: bf16 (round to nearest even) of d * q for `n_blocks` Q8_0 blocks.
void fused_gr_dequant_q8_0(const uint8_t* q8, uint16_t* bf16, int64_t n_blocks, void* stream);
void fused_gr_quantize_q8_0(const uint16_t* bf16, uint8_t* out, int64_t n_blocks, void* stream);

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` is n_tok * hc * n_embd floats.  Every
/// token's outputs are bitwise `fused_gr_read(a[t])`.
constexpr int kFusedGrMaxT = 8;
/// Returns true when it also wrote the q8_1 images (every token's q8_mixed set and this read supports it).
bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

/// The bench only: the AMD latency-hidden kernels on (1) or off (0); -1 = STRATA_GR_FAST.
void fused_gr_set_fast(int on);
/// The multi read's variants (#315; not main's opt-in STRATA_GR_V3 read, which sums in another order), all computing
/// every output with exactly the staged read's operations in its order.  Unset `STRATA_HC_SPLIT` runs the staged
/// upstream path on every architecture: it never runs the P4 self-test and never selects P4.  `=2` requests staged
/// (split/plain fallback); `=1` requests split (plain fallback); `=0` selects plain.  Only `=7` opts into the P4
/// register-half down kernel, and only on runtime gfx1200/gfx1201: it runs the self-test and selects P4 iff its
/// bitwise parity passes against staged AND staged passes against plain, and falls back staged -> split -> plain
/// otherwise.  The check covers T=1..8, apply on/off, inject weights present/null, and (when compiled) q8_1 /
/// QFUSE_HC up-kernel outputs.  Context-fatal statuses from queries, copies/memsets, shared-memory setup,
/// launch/sync, read-back, allocation, stream setup/teardown or cleanup end the process before further GPU work;
/// uncorrectable ECC and unknown/unclassified runtime statuses fail closed, with the fatal exit following #224's
/// generate-path precedent.  Recoverable P4-only launch, sync, read-back or candidate-stream cleanup errors disable
/// P4 while preserving staged/split checks; recoverable shared self-test resource errors fail the self-test and
/// prevent P4 selection.  `fused_gr_check` runs once per card and latches its choice,
/// `fused_gr_variant` reports the requested path capped at staged until it has.
void fused_gr_check();
int fused_gr_variant();

}  // namespace strata::kernels
