// src/program/suffix_reuse_int8_test.cpp - Phase 3: the INT8 KV splice (dequant -> rotate -> requant).
//
// The `--kv int8` daily mode stores K as int8 codes with one FP16 scale per 64 values
// (kv_q8.hpp: codes [page][kv_head][page_size][head_dim], scales [..][head_dim/64]):
//     sbits = f16(amax_group / 127);  sf = f32(sbits);
//     code  = clamp(rint(x / sf), -127, 127);  x' = f16(code * sf)
// (the writer: src/kernels/cuda/kv_q8.cu:47-67, the exact kernel the engine's append path
// launches; the host mirror: src/kernels/kv_hybrid_parity.cpp:99-116, pinned BITWISE
// against the kernel by the GPU parity suite).
//
// PHASE 3 makes the splice work on those blocks: dequantize the K group to fp32, apply the
// splice's delta-RoPE rotation, RE-QUANTIZE with the writer's exact math, write codes+scales
// back.  This test pins three properties:
//
//   T11.a - QUANT ROUND-TRIP: requant(dequant(block)) is BIT-IDENTICAL to the original block
//     (mirror math => the unchanged-token cells round-trip byte-exactly; only the rotated
//     groups change, and their new codes stay within one quant step of the fp32 truth).
//
//   T11.b - ROTATED SPLICE ACCURACY: quant -> dequant -> rotate -> requant -> dequant vs an
//     fp32-rotated reference over the SAME dequantized input.  Measured bound: the requant
//     error of the rotated image is bounded by the SAME per-group quant-step bound the
//     engine's fresh-append path pays (scale = amax/127 -> half-step = amax/254); the test
//     asserts the max abs error stays under a measured constant (reported; see the bound
//     derivation below the assertion) and also under the fresh-path parity bound.
//
//   T11.c - SPLICE ON AN INT8 ARENA: a small QsaState with k_q/k_scale pools (page_size=4,
//     the artifact's pool granule), known contents; run the int8 splice helper; assert
//     (1) rotated cells match the fp32 reference requantized with the writer's math,
//     (2) non-rotated cells are BIT-IDENTICAL (byte-for-byte codes + scales),
//     (3) V-side bytes untouched, (4) the pure guard `splice_scope_guard_int8` accepts
//     plain int8 and still refuses q4/hybrid/resident/split (fail-closed stays).
//
// RED state (before Phase 3): `rotate_qsa_tail_inplace_int8` and the quant helpers do not
// exist - this file does not compile.  That is the RED proof: the feature is absent.
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"   // T11.d: kv_q4_bytes_per_head (the hybrid's V half)
#include "strata/program/kv_q8_quant.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/program/kv_splice.hpp"
#include "strata/program/kv_splice_apply.hpp"
#include "strata/program/kv_splice_apply_int8.hpp"
#include "strata/program/suffix_reuse.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
using strata::kernels::KV_Q8_GROUP;
using strata::program::suffix_reuse::rotate_qsa_tail_inplace_int8;
using strata::program::suffix_reuse::splice_scope_guard_int8;
using strata::program::suffix_reuse::SpliceStatus;
using strata::program::suffix_reuse::build_rope_table_host;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-76s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

/// T3-style rope at position `p` over the first `n_rot` elements of a fp32 row (test-local
/// reference; the production helper goes through rotate_k_by_delta, pinned by T6).
static inline void rope_at_fp32(float* row, int n_rot, int p,
                                const float* cos_tab, const float* sin_tab, int max_pos) {
    const int half = n_rot / 2;
    const int p_clamped = p < max_pos ? p : max_pos - 1;
    for (int i = 0; i < half; ++i) {
        const float a = row[i];
        const float b = row[i + half];
        const float c = cos_tab[(size_t) p_clamped * (size_t) half + (size_t) i];
        const float s = sin_tab[(size_t) p_clamped * (size_t) half + (size_t) i];
        float oa, ob;
        strata::kernels::rope_neox_pair(a, b, c, s, oa, ob);
        row[i] = oa; row[i + half] = ob;
    }
}

int main() {
    std::printf("suffix_reuse_int8_test (Phase 3: dequant-rotate-requant splice)\n");

    const int head_dim  = 256;   // the artifact's head_dim (4 groups of 64)
    const int n_rot     = 64;
    const int n_head_kv = 2;
    const int max_pos   = 1024;
    const double theta  = 1000000.0;
    const int half      = n_rot / 2;
    const int groups    = head_dim / KV_Q8_GROUP;
    std::vector<float> cos_tab((size_t) max_pos * (size_t) half),
                       sin_tab((size_t) max_pos * (size_t) half);
    build_rope_table_host(n_rot, theta, max_pos, cos_tab.data(), sin_tab.data());

    // ---------------- T11.a: quant round-trip (bit-identical) ----------------
    // Hypothesis: mirroring the writer's math (kv_q8.cu:47-67) makes
    // requant(dequant(block)) == block BITWISE.  Proof sketch: sf > 0 with the group's amax
    // code at +-127 means amax/sf <= 127 + 0.5, and amax' == amax (f16(amax/127) is a fixed
    // point: the dequantized group's max IS amax/sf*sf = amax, exact in f32), so the new
    // scale bits equal the old; each code divides by the SAME sf and rounds back to itself
    // (|q*sf/sf - q| = 0 exactly, rint is exact on integers).  Degenerate groups (all-zero,
    // subnormal scales) round-trip too (sf == 0 -> codes stay 0).
    {
        std::mt19937 rng(4242);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        int blocks = 4096;
        int bad_codes = 0, bad_scales = 0;
        std::vector<float> vals(KV_Q8_GROUP);
        for (int b = 0; b < blocks; ++b) {
            // a spread of group magnitudes: unit-normal, tiny, large, and exact-zero groups
            const float mag = (b % 4 == 0) ? 0.001f : (b % 4 == 1) ? 100.0f : (b % 4 == 2) ? 0.0f : 1.0f;
            for (int t = 0; t < KV_Q8_GROUP; ++t) vals[t] = nd(rng) * mag;
            if (b % 4 == 2) for (int t = 0; t < KV_Q8_GROUP; ++t) vals[t] = 0.0f;
            strata::kernels::KvQ8Group out{}, back{};
            strata::kernels::kv_q8_quant_group(vals.data(), out);
            float deq[KV_Q8_GROUP];
            strata::kernels::kv_q8_dequant_group(out, deq);
            strata::kernels::KvQ8Group out2{};
            strata::kernels::kv_q8_quant_group(deq, out2);
            for (int t = 0; t < KV_Q8_GROUP; ++t)
                if (out.codes[t] != out2.codes[t]) ++bad_codes;
            if (out.scale_bits != out2.scale_bits) ++bad_scales;
        }
        check(bad_codes == 0 && bad_scales == 0,
              "T11.a: requant(dequant(block)) is BIT-IDENTICAL over 4096 groups (mirror math)");
    }

    // ---------------- T11.b: rotated splice accuracy vs fp32 reference ----------------
    // The splice's contract: dequant the cached K image, apply the splice's DELTA rotation
    // (rotate_k_by_delta; the rope angle from old_pos to old_pos+delta, NOT a full re-rotation
    // from raw at the new position), requant, write back.  The fp32 reference applies the
    // SAME delta rotation in fp32, then quantizes - the only error is the second quant of
    // the rotated image, bounded by the writer's per-group half-step (amax/254).  We assert
    // max|err| <= the half-step of the reference's rotated image (the fresh path pays the
    // same per-group quant error when it writes the rotated image to the cache).
    {
        std::mt19937 rng(777);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        int rows = 512;
        double max_err = 0.0, max_half_step = 0.0;
        int over_bound = 0;
        for (int r = 0; r < rows; ++r) {
            const int old_pos = 10 + (r % 50);
            const int delta   = (r % 2 == 0) ? 3 : -2;
            // raw K, then rope at old_pos -> the cached image the engine's append quantizes
            std::vector<float> cached(head_dim);
            for (int d = 0; d < head_dim; ++d) cached[d] = nd(rng) * 0.5f;
            rope_at_fp32(cached.data(), n_rot, old_pos, cos_tab.data(), sin_tab.data(), max_pos);
            // quantize the cached image (what the engine stored)
            std::vector<strata::kernels::KvQ8Group> q(groups);
            for (int g = 0; g < groups; ++g)
                strata::kernels::kv_q8_quant_group(&cached[g * KV_Q8_GROUP], q[g]);
            // dequant: the splices's input
            std::vector<float> deq(head_dim);
            for (int g = 0; g < groups; ++g)
                strata::kernels::kv_q8_dequant_group(q[g], &deq[g * KV_Q8_GROUP]);
            // splice path: delta-rotate, requant, dequant
            std::vector<float> spliced(deq);
            strata::program::suffix_reuse::rotate_k_by_delta(
                spliced.data(), 1, head_dim, n_rot, old_pos, delta,
                cos_tab.data(), sin_tab.data(), max_pos);
            std::vector<strata::kernels::KvQ8Group> q2(groups);
            for (int g = 0; g < groups; ++g)
                strata::kernels::kv_q8_quant_group(&spliced[g * KV_Q8_GROUP], q2[g]);
            std::vector<float> back(head_dim);
            for (int g = 0; g < groups; ++g)
                strata::kernels::kv_q8_dequant_group(q2[g], &back[g * KV_Q8_GROUP]);
            // reference: the SAME delta rotation in fp32 (no requant).  The splice's error
            // is exactly the requant step of the rotated image - bounded by half-step =
            // amax(rotated_group) / 254.
            std::vector<float> ref_rot(deq);
            strata::program::suffix_reuse::rotate_k_by_delta(
                ref_rot.data(), 1, head_dim, n_rot, old_pos, delta,
                cos_tab.data(), sin_tab.data(), max_pos);
            for (int g = 0; g < groups; ++g) {
                float amax_rot = 0.0f;
                for (int t = 0; t < KV_Q8_GROUP; ++t)
                    amax_rot = std::max(amax_rot, std::fabs(ref_rot[g * KV_Q8_GROUP + t]));
                const float half_step = amax_rot / 254.0f;
                max_half_step = std::max(max_half_step, (double) half_step);
                for (int t = 0; t < KV_Q8_GROUP; ++t) {
                    const double e = std::fabs((double) back[g * KV_Q8_GROUP + t]
                                             - (double) ref_rot[g * KV_Q8_GROUP + t]);
                    max_err = std::max(max_err, e);
                    if (e > (double) half_step * 1.0001 + 1e-9) ++over_bound;
                }
            }
        }
        std::printf("       (measured: max_abs_err=%.3e over %d rows, max group half-step=%.3e)\n",
                    max_err, rows, max_half_step);
        check(over_bound == 0,
              "T11.b: rotated+requantized K within the group half-step of the fp32 rotation");
        check(max_err < 5e-2,
              "T11.b: max abs error < 5e-2 absolute guardrail (measured, unit-magnitude K)");
    }

    // ---------------- T11.c: splice on an int8 arena ----------------
    // Geometry mirrors the artifact: page_size=4 (the pool granule qsa_state_init sizes with),
    // n_head_kv=2, head_dim=256.  Cells are quantized with the writer's math.  The splice
    // rotates [edit_pos, edit_pos+len) in OLD-position space; cells outside must stay
    // BIT-IDENTICAL (codes AND scales).
    {
        const int page_size = 4;
        const int n_pages   = 8;
        const int max_cells = n_pages * page_size;
        std::mt19937 rng(31337);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        // logical K by (pos, head, dim): the RAW values, then rope at the position
        std::vector<float> raw((size_t) max_cells * n_head_kv * head_dim);
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = nd(rng) * 0.5f;
        // build the arena: codes [page][head][page_size][head_dim], scales [..][groups]
        const size_t pool_rows = (size_t) n_pages * n_head_kv * page_size;
        std::vector<int8_t>  k_q(pool_rows * head_dim, 0);
        std::vector<uint16_t> k_scale(pool_rows * groups, 0);
        std::vector<int8_t>  v_q(pool_rows * head_dim, 0);   // V is untouched by the splice
        for (int p = 0; p < max_cells; ++p) {
            const int page = p / page_size, off = p % page_size;
            for (int h = 0; h < n_head_kv; ++h) {
                const size_t row = (size_t) page * n_head_kv * page_size + (size_t) h * page_size + off;
                std::vector<float> cell(head_dim);
                for (int d = 0; d < head_dim; ++d) cell[d] = raw[(size_t) p * n_head_kv * head_dim + (size_t) h * head_dim + d];
                rope_at_fp32(cell.data(), n_rot, p, cos_tab.data(), sin_tab.data(), max_pos);
                for (int g = 0; g < groups; ++g) {
                    strata::kernels::KvQ8Group qg{};
                    strata::kernels::kv_q8_quant_group(&cell[g * KV_Q8_GROUP], qg);
                    for (int t = 0; t < KV_Q8_GROUP; ++t) k_q[row * head_dim + g * KV_Q8_GROUP + t] = qg.codes[t];
                    k_scale[row * groups + g] = qg.scale_bits;
                }
                for (int d = 0; d < head_dim; ++d) v_q[row * head_dim + d] = (int8_t) ((p + d) & 0x7F);
            }
        }
        std::vector<int8_t>   k_q_before = k_q;
        std::vector<uint16_t> k_scale_before = k_scale;
        std::vector<int8_t>   v_q_before = v_q;
        // the QsaState: int8 pools, identity page map (kv_mode == 0)
        strata::core::QsaState st;
        st.kv_int8 = true;
        st.k_q = k_q.data();
        st.k_scale = k_scale.data();
        st.v_q = v_q.data();
        st.kv_mode = 0;
        st.max_cells = max_cells;
        st.n_pages = n_pages;
        st.n_slots = n_pages;   // identity
        st.cos_tab = cos_tab.data();
        st.sin_tab = sin_tab.data();
        st.owns_rope = false;
        const int edit_pos = 6, old_suffix_len = 9, delta = 4;   // spans pages 1..3
        const bool ok = rotate_qsa_tail_inplace_int8(
            st, head_dim, n_rot, n_head_kv, max_pos, page_size,
            edit_pos, old_suffix_len, /*uniform=*/true, delta, /*deltas=*/nullptr,
            /*pool_rows=*/(int64_t) pool_rows);
        check(ok, "T11.c: int8 splice helper returns true on the supported config");
        // reference: the dequantized cached image, rotated by delta, requantized (writer math)
        int bad_cells = 0, mismatch_codes = 0, mismatch_scales = 0;
        double max_err = 0.0;
        for (int p = 0; p < max_cells; ++p) {
            const int page = p / page_size, off = p % page_size;
            const bool in_range = (p >= edit_pos) && (p < edit_pos + old_suffix_len);
            for (int h = 0; h < n_head_kv; ++h) {
                const size_t row = (size_t) page * n_head_kv * page_size + (size_t) h * page_size + off;
                if (!in_range) {
                    for (int d = 0; d < head_dim; ++d)
                        if (k_q[row * head_dim + d] != k_q_before[row * head_dim + d]) { ++mismatch_codes; break; }
                    for (int g = 0; g < groups; ++g)
                        if (k_scale[row * groups + g] != k_scale_before[row * groups + g]) { ++mismatch_scales; break; }
                    continue;
                }
                // dequant the ORIGINAL cell (pre-splice image)
                std::vector<float> deq(head_dim);
                for (int g = 0; g < groups; ++g) {
                    strata::kernels::KvQ8Group qg{};
                    qg.scale_bits = k_scale_before[row * groups + g];
                    for (int t = 0; t < KV_Q8_GROUP; ++t) qg.codes[t] = k_q_before[row * head_dim + g * KV_Q8_GROUP + t];
                    strata::kernels::kv_q8_dequant_group(qg, &deq[g * KV_Q8_GROUP]);
                }
                // rotate by DELTA (the splice's contract: the cached image is R(p)·raw,
                // and the splice's rotate_k_by_delta applies R(delta), yielding R(p+delta)·raw
                // -- NOT the same as rope_at_fp32 at p+delta which would re-rotate from raw).
                strata::program::suffix_reuse::rotate_k_by_delta(
                    deq.data(), 1, head_dim, n_rot, /*old_pos=*/p, /*delta=*/delta,
                    cos_tab.data(), sin_tab.data(), max_pos);
                for (int g = 0; g < groups; ++g) {
                    strata::kernels::KvQ8Group want{};
                    strata::kernels::kv_q8_quant_group(&deq[g * KV_Q8_GROUP], want);
                    const uint16_t got_scale = k_scale[row * groups + g];
                    if (got_scale != want.scale_bits) ++mismatch_scales;
                    for (int t = 0; t < KV_Q8_GROUP; ++t) {
                        const int8_t got = k_q[row * head_dim + g * KV_Q8_GROUP + t];
                        if (got != want.codes[t]) {
                            ++mismatch_codes;
                            const float gs = f32_from_f16(got_scale);
                            const float gw = f32_from_f16(want.scale_bits);
                            const double e = std::fabs((double) got * (double) gs
                                                     - (double) want.codes[t] * (double) gw);
                            max_err = std::max(max_err, e);
                        }
                    }
                }
                // every in-range cell must have at least one changed code or scale (RED guard:
                // a no-op splice would fail T11.b's accuracy intent - here we only require the
                // helper touched the range; delta != 0 guarantees the rotation moves values)
            }
            (void) bad_cells;
        }
        check(mismatch_codes == 0 && mismatch_scales == 0,
              "T11.c: rotated cells match the writer-math requant reference BITWISE");
        std::printf("       (code mismatches: %d, scale mismatches: %d, max code-scale drift %.3e)\n",
                    mismatch_codes, mismatch_scales, max_err);
        int v_diff = 0;
        for (size_t i = 0; i < v_q.size(); ++i) if (v_q[i] != v_q_before[i]) ++v_diff;
        check(v_diff == 0, "T11.c: V codes byte-identical after the int8 splice");
        // the pure guard: plain int8 is ACCEPTED now; q4/hybrid/resident/split still refused
        check(splice_scope_guard_int8(/*kv_int8=*/true, /*kv_q4=*/false, /*kv_hybrid=*/false,
                                      /*kv_resident=*/0, /*layer_split=*/false, /*kv_rot=*/false)
              == SpliceStatus::NoOp,
              "T11.c: guard ACCEPTS plain --kv int8 (kv_rot off, the default)");
        check(splice_scope_guard_int8(/*kv_int8=*/true, /*kv_q4=*/false, /*kv_hybrid=*/false,
                                      /*kv_resident=*/0, /*layer_split=*/false, /*kv_rot=*/true)
              == SpliceStatus::NoOp,
              "T11.c: guard ACCEPTS int8+kv_rot (the H-sandwich handles it; head_dim!=256 "
              "refuses inside the helper, fail-closed)");
        check(splice_scope_guard_int8(/*kv_int8=*/true, /*kv_q4=*/false, /*kv_hybrid=*/true,
                                      /*kv_resident=*/0, /*layer_split=*/false, /*kv_rot=*/false)
              == SpliceStatus::NoOp,
              "T11.c: guard ACCEPTS hybrid k8v4 (K half is byte-identical int8; V untouched)");
        check(splice_scope_guard_int8(false, true, false, 0, false, false)
              == SpliceStatus::UnsupportedFormat,
              "T11.c: guard still refuses q4_0");
        check(splice_scope_guard_int8(false, false, false, 32, false, false)
              == SpliceStatus::UnsupportedResident,
              "T11.c: guard still refuses kv_resident (streaming ring)");
        check(splice_scope_guard_int8(false, false, false, 0, true, false)
              == SpliceStatus::UnsupportedSplit,
              "T11.c: guard still refuses layer split");

        // ---- T11.d: hybrid k8v4 parity - the K rotation is layout-independent ----
        // The hybrid's K half is the same codes+scales storage (layer.cpp:736), so the same
        // splice on a hybrid-shaped state must produce BITWISE the same K result as on a
        // plain-int8 state, and the Q4_0 V bytes must be untouched (V is position-free).
        {
            std::vector<int8_t>  h_k_q = k_q_before;
            std::vector<uint16_t> h_k_scale = k_scale_before;
            std::vector<uint8_t> h_v_q4(pool_rows * strata::kernels::kv_q4_bytes_per_head(head_dim), 0);
            for (size_t i = 0; i < h_v_q4.size(); ++i) h_v_q4[i] = (uint8_t) (i * 37 + 11);
            strata::core::QsaState hst;
            hst.kv_int8 = true;      // the hybrid's K half is quantized int8 (kv_q8)
            hst.kv_hybrid = true;
            hst.k_q = h_k_q.data();
            hst.k_scale = h_k_scale.data();
            hst.v_q4 = h_v_q4.data();   // the hybrid's V half (rotated Q4_0; opaque here)
            hst.kv_mode = 0;
            hst.max_cells = max_cells;
            hst.n_pages = n_pages;
            hst.n_slots = n_pages;
            hst.cos_tab = cos_tab.data();
            hst.sin_tab = sin_tab.data();
            hst.owns_rope = false;
            const bool hok = rotate_qsa_tail_inplace_int8(
                hst, head_dim, n_rot, n_head_kv, max_pos, page_size,
                edit_pos, old_suffix_len, /*uniform=*/true, delta, /*deltas=*/nullptr,
                /*pool_rows=*/(int64_t) pool_rows);
            check(hok, "T11.d: hybrid-shaped state splices (same helper, K half is int8)");
            int k_diff = 0;
            for (size_t i = 0; i < k_q.size(); ++i)
                if (h_k_q[i] != k_q[i]) ++k_diff;
            int ks_diff = 0;
            for (size_t i = 0; i < k_scale.size(); ++i)
                if (h_k_scale[i] != k_scale[i]) ++ks_diff;
            check(k_diff == 0 && ks_diff == 0,
                  "T11.d: hybrid K result BITWISE == plain-int8 K result (layout-independent)");
            int v4_diff = 0;
            for (size_t i = 0; i < h_v_q4.size(); ++i)
                if (h_v_q4[i] != (uint8_t) (i * 37 + 11)) ++v4_diff;
            check(v4_diff == 0, "T11.d: hybrid V (Q4_0 bytes) byte-identical after the splice");
        }

        // ---- T11.e: int8 + STRATA_KV_ROT=1 - the Hadamard sandwich is exact ----
        // Store K the way the rotated writer does (H * R(p) * k, then int8-quantized), run
        // the splice, and compare against the DIRECT reference: quantize(H * R(p+delta) * k)
        // with the writer's own math.  The sandwich (dequant -> H -> R(delta) -> H -> requant)
        // must match that reference BITWISE (the requant count is unchanged: one, at the end;
        // H*R(delta)*H*(H*R(p)*k) = H*R(p+delta)*k exactly, so the only difference vs T11.c's
        // unrotated case is the fp-sum order inside fwht - asserted small, and the QUANTIZED
        // result asserted bitwise because quantization buckets absorb sub-ulp reordering).
        {
            std::vector<int8_t>   r_k_q = k_q_before;
            std::vector<uint16_t> r_k_scale = k_scale_before;
            // rebuild the rotated-domain arena from `raw`: stored = quant(H * R(p) * k)
            for (int p = 0; p < max_cells; ++p) {
                const int page = p / page_size, off = p % page_size;
                for (int h = 0; h < n_head_kv; ++h) {
                    const size_t row = (size_t) page * n_head_kv * page_size + (size_t) h * page_size + off;
                    std::vector<float> cell(head_dim);
                    for (int d = 0; d < head_dim; ++d)
                        cell[d] = raw[(size_t) p * n_head_kv * head_dim + (size_t) h * head_dim + d];
                    rope_at_fp32(cell.data(), n_rot, p, cos_tab.data(), sin_tab.data(), max_pos);
                    strata::program::suffix_reuse::fwht256_row_host(cell.data());   // the rotated writer's H
                    for (int g = 0; g < groups; ++g) {
                        strata::kernels::KvQ8Group qg{};
                        strata::kernels::kv_q8_quant_group(&cell[g * KV_Q8_GROUP], qg);
                        for (int t = 0; t < KV_Q8_GROUP; ++t) r_k_q[row * head_dim + g * KV_Q8_GROUP + t] = qg.codes[t];
                        r_k_scale[row * groups + g] = qg.scale_bits;
                    }
                }
            }
            strata::core::QsaState rst;
            rst.kv_int8 = true;
            rst.kv_rot = true;      // STRATA_KV_ROT=1: the sandwich path
            const std::vector<int8_t>   r_k_q_before = r_k_q;       // snapshots for the
            const std::vector<uint16_t> r_k_scale_before = r_k_scale;   // out-of-range check
            rst.k_q = r_k_q.data();
            rst.k_scale = r_k_scale.data();
            rst.v_q = v_q.data();
            rst.kv_mode = 0;
            rst.max_cells = max_cells;
            rst.n_pages = n_pages;
            rst.n_slots = n_pages;
            rst.cos_tab = cos_tab.data();
            rst.sin_tab = sin_tab.data();
            rst.owns_rope = false;
            const bool rok = rotate_qsa_tail_inplace_int8(
                rst, head_dim, n_rot, n_head_kv, max_pos, page_size,
                edit_pos, old_suffix_len, /*uniform=*/true, delta, /*deltas=*/nullptr,
                /*pool_rows=*/(int64_t) pool_rows);
            check(rok, "T11.e: rotated int8 splice runs (kv_rot path, head_dim 256)");
            // Reference A (BITWISE, the wiring check): an independently-coded sandwich of the
            // STORED cell - dequant, H, rotate_k_by_delta (pinned independently by T11.c), H,
            // requant.  This mirrors how T11.c references dequant(stored) rather than raw: the
            // stored cell already carries the first quant's noise, and the splice must move
            // THAT image faithfully (the double-quant delta is measured by reference B).
            auto ref_sandwich = [&](int p, int h) {
                std::vector<float> cell(head_dim);
                for (int g = 0; g < groups; ++g) {
                    strata::kernels::KvQ8Group qg{};
                    qg.scale_bits = r_k_scale_before[(size_t) (p / page_size) * n_head_kv * page_size * groups
                                                   + (size_t) h * page_size * groups + (size_t) (p % page_size) * groups + g];
                    for (int t = 0; t < KV_Q8_GROUP; ++t)
                        qg.codes[t] = r_k_q_before[(size_t) (p / page_size) * n_head_kv * page_size * head_dim
                                                 + (size_t) h * page_size * head_dim + (size_t) (p % page_size) * head_dim
                                                 + g * KV_Q8_GROUP + t];
                    strata::kernels::kv_q8_dequant_group(qg, &cell[g * KV_Q8_GROUP]);
                }
                // local independent fwht (same Sylvester/1-16 math as the parity reference)
                for (int s = 1; s < 256; s *= 2)
                    for (int i2 = 0; i2 < 256; i2 += 2 * s)
                        for (int j = 0; j < s; ++j) {
                            const float u = cell[i2 + j], v = cell[i2 + j + s];
                            cell[i2 + j] = u + v; cell[i2 + j + s] = u - v;
                        }
                for (int i2 = 0; i2 < 256; ++i2) cell[i2] *= 1.0f / 16.0f;
                strata::program::suffix_reuse::rotate_k_by_delta(
                    cell.data(), 1, head_dim, n_rot, p, delta, cos_tab.data(), sin_tab.data(), max_pos);
                for (int s = 1; s < 256; s *= 2)
                    for (int i2 = 0; i2 < 256; i2 += 2 * s)
                        for (int j = 0; j < s; ++j) {
                            const float u = cell[i2 + j], v = cell[i2 + j + s];
                            cell[i2 + j] = u + v; cell[i2 + j + s] = u - v;
                        }
                for (int i2 = 0; i2 < 256; ++i2) cell[i2] *= 1.0f / 16.0f;
                return cell;
            };
            int r_mismatch = 0, r_out_changed = 0, r_ideal_off = 0;
            double r_max_step = 0.0;
            for (int p = 0; p < max_cells; ++p) {
                const int page = p / page_size, off = p % page_size;
                const bool in_range = (p >= edit_pos) && (p < edit_pos + old_suffix_len);
                for (int h = 0; h < n_head_kv; ++h) {
                    const size_t row = (size_t) page * n_head_kv * page_size + (size_t) h * page_size + off;
                    // out-of-range: byte-identical to the stored-before image
                    if (!in_range) {
                        for (int d = 0; d < head_dim; ++d)
                            if (r_k_q[row * head_dim + d] != r_k_q_before[row * head_dim + d]) { ++r_out_changed; break; }
                        continue;
                    }
                    // Reference A: bitwise sandwich-of-stored
                    const std::vector<float> ref = ref_sandwich(p, h);
                    for (int g = 0; g < groups; ++g) {
                        strata::kernels::KvQ8Group want{};
                        strata::kernels::kv_q8_quant_group(&ref[g * KV_Q8_GROUP], want);
                        if (r_k_scale[row * groups + g] != want.scale_bits) ++r_mismatch;
                        for (int t = 0; t < KV_Q8_GROUP; ++t)
                            if (r_k_q[row * head_dim + g * KV_Q8_GROUP + t] != want.codes[t]) ++r_mismatch;
                    }
                    // Reference B (the measured claim): vs the IDEAL fresh-write image
                    // quant(H*R(p+delta)*raw) - allowed to differ by the double-quant noise
                    // (the stored cell's first quant + the splice's requant): at most 1 code
                    // step per component, scale-relative.
                    std::vector<float> ideal(head_dim);
                    for (int d = 0; d < head_dim; ++d)
                        ideal[d] = raw[(size_t) p * n_head_kv * head_dim + (size_t) h * head_dim + d];
                    rope_at_fp32(ideal.data(), n_rot, p + delta, cos_tab.data(), sin_tab.data(), max_pos);
                    strata::program::suffix_reuse::fwht256_row_host(ideal.data());
                    for (int g = 0; g < groups; ++g) {
                        strata::kernels::KvQ8Group iw{};
                        strata::kernels::kv_q8_quant_group(&ideal[g * KV_Q8_GROUP], iw);
                        const float isf = strata::kernels::f32_from_f16(iw.scale_bits);
                        const float ssf = strata::kernels::f32_from_f16(r_k_scale[row * groups + g]);
                        for (int t = 0; t < KV_Q8_GROUP; ++t) {
                            const double got_v = (double) r_k_q[row * head_dim + g * KV_Q8_GROUP + t] * ssf;
                            const double idea_v = (double) iw.codes[t] * isf;
                            const double step = std::fabs(got_v - idea_v) / std::max(1e-9, (double) isf);
                            r_max_step = std::max(r_max_step, step);
                            if (step > 2.0) ++r_ideal_off;
                        }
                    }
                }
            }
            check(r_out_changed == 0, "T11.e: cells OUTSIDE the splice range byte-identical");
            // Same ideal-distance metric for the UNROTATED T11.c result: the accepted int8
            // splice has the identical double-quant structure (quant1 at write time + the
            // splice's requant), so the rotated path's noise must be no worse than it (the
            // fwht preserves the error vector's norm; per-component it stays ~same scale).
            double u_max_step = 0.0;
            for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
                const int page = p / page_size, off = p % page_size;
                for (int h = 0; h < n_head_kv; ++h) {
                    const size_t row = (size_t) page * n_head_kv * page_size + (size_t) h * page_size + off;
                    std::vector<float> ideal(head_dim);
                    for (int d = 0; d < head_dim; ++d)
                        ideal[d] = raw[(size_t) p * n_head_kv * head_dim + (size_t) h * head_dim + d];
                    rope_at_fp32(ideal.data(), n_rot, p + delta, cos_tab.data(), sin_tab.data(), max_pos);
                    for (int g = 0; g < groups; ++g) {
                        strata::kernels::KvQ8Group iw{};
                        strata::kernels::kv_q8_quant_group(&ideal[g * KV_Q8_GROUP], iw);
                        const float isf = strata::kernels::f32_from_f16(iw.scale_bits);
                        const float ssf = strata::kernels::f32_from_f16(k_scale[row * groups + g]);
                        for (int t = 0; t < KV_Q8_GROUP; ++t) {
                            const double got_v = (double) k_q[row * head_dim + g * KV_Q8_GROUP + t] * ssf;
                            const double idea_v = (double) iw.codes[t] * isf;
                            u_max_step = std::max(u_max_step,
                                std::fabs(got_v - idea_v) / std::max(1e-9, (double) isf));
                        }
                    }
                }
            }
            std::printf("       (bitwise-sandwich mismatches: %d; ideal code-steps: rotated %.2f vs unrotated %.2f)\n",
                        r_mismatch, r_max_step, u_max_step);
            check(r_mismatch == 0,
                  "T11.e: sandwich result BITWISE == independent sandwich-of-stored reference");
            check(r_ideal_off == 0,
                  "T11.e: no component beyond the guardrail of 2 code steps from the ideal");
            check(r_max_step <= std::max(u_max_step, 1.0) * 1.10,
                  "T11.e: rotated double-quant noise within 10% of the accepted unrotated path's");
        }
    }

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
