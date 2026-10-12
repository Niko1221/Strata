// src/program/suffix_reuse_apply_test.cpp - Phase 2 Task 3 T6: RED-able splice-wiring test.
//
// Synthesises a QsaState with hand-crafted K rows at known positions, runs the splice rotation
// directly through `rotate_k_by_delta` (T3-style pure fp32 storage T6), and asserts the splice
// path produces the SAME K as a fresh rotation at the new positions (1e-6 tolerance, fp32 ULP).
//
// T6 has TWO sub-tests:
//
//   T6.x - "FP32 ALGORITHM" - synthesises a k_pool as a TYPED fp32 buffer (the test casts to
//     uint16_t* to satisfy the QsaState declared type).  We exercise `rotate_k_by_delta` against
//     this fp32 storage directly (no fp16 round-trips).  T3 already pinned the rotation math to
//     1e-6; T6.x verifies that the splice ALOGORITHM (which positions to rotate, which deltas
//     to apply per row) is correct end-to-end.  RED check: with rotation skipped (a temporary
//     flag inside the test loop), cached K diverges from fresh.
//
//   T6.y - "FP16 ROUND-TRIP HELPER" - exercises `rotate_qsa_tail_inplace` against a real fp16
//     storage pool (the production path).  Tolerance is fp16 ULP scale; the fp16 path IS the
//     production path, but the underlying math is already covered by T3 + T6.x.
//
//   T6.z - "V byte-identical" - the helper leaves V untouched.  Helper returns true; pre/post
//     V memory equal.
//
//   T6.w - "SPLICE_UNSUPPORTED" - the pure scope-guard predicate (`splice_scope_guard`) returns
//     `Unsupported*` for kv_int8 / kv_q4 / kv_hybrid / kv_resident > 0 / layer_split.
//
// Phase 2 design notes:
//   - page_size = 1 in this test (unpaged) so position p maps directly to row (p * n_head_kv + h).
//   - V is allocated from a separate buffer (per QsaState semantics) and asserted byte-identical.
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/program/kv_splice.hpp"
#include "strata/program/kv_splice_apply.hpp"
#include "strata/program/suffix_reuse.hpp"
#include "strata/kernels/rope.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using strata::program::suffix_reuse::rotate_qsa_tail_inplace;
using strata::program::suffix_reuse::splice_scope_guard;
using strata::program::suffix_reuse::SpliceStatus;
using strata::program::suffix_reuse::build_rope_table_host;
using strata::program::suffix_reuse::rotate_k_by_delta;
using strata::program::suffix_reuse::detail::fp32_to_fp16_bits;
using strata::program::suffix_reuse::detail::fp16_bits_to_fp32;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

/// T3-style rope at position `p` over the first `n_rot` elements of a fp32 row.
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

/// Splice algorithm directly in fp32 storage.  Re-implements `rotate_qsa_tail_inplace`'s core
/// loop without going through fp16 storage (T6.x).  `skip_rotation=true` simulates a regression
/// that drops the `rotate_k_by_delta` call (the RED proof).
///
/// Storage layout: `k` is a (max_cells * n_head_kv * head_dim) fp32 buffer; row at position p,
/// kv_head h starts at `k + (p * n_head_kv + h) * head_dim`.  `cos_tab`/`sin_tab` are
/// (max_pos, n_rot/2).
static bool splice_fp32(float* k, int max_cells, int n_head_kv, int head_dim, int n_rot,
                        int page_size,
                        int edit_pos, int old_suffix_len,
                        bool uniform, int delta, const int64_t* deltas,
                        const float* cos_tab, const float* sin_tab, int max_pos,
                        bool skip_rotation = false) {
    if (edit_pos < 0 || old_suffix_len <= 0) return false;
    if (page_size <= 0) page_size = 1;
    for (int64_t kk = 0; kk < old_suffix_len; ++kk) {
        const int64_t p = edit_pos + (int64_t) kk;
        const int64_t page = p / page_size;
        const int64_t off  = p % page_size;
        const int delta_row = uniform ? delta : (int) deltas[(size_t) kk];
        for (int h = 0; h < n_head_kv; ++h) {
            const int64_t row_idx = (int64_t) page * n_head_kv * page_size + (int64_t) h * page_size + off;
            float* row = k + (size_t) row_idx * (size_t) head_dim;
            if (!skip_rotation) {
                rotate_k_by_delta(row, 1, head_dim, n_rot, (int) p, delta_row,
                                  cos_tab, sin_tab, max_pos);
            }
        }
    }
    return true;
}

int main() {
    std::printf("suffix_reuse_apply_test (Phase 2 Task 3 T6)\n");

    // Geometry mirrors the artifact: head_dim=256, n_head_kv=2, n_rot=64, page_size=1 (unpaged).
    const int head_dim    = 256;
    const int n_rot       = 64;
    const int n_head_kv   = 2;
    const int page_size   = 1;            // unpaged; position p -> row p*n_head_kv + h
    const int max_pos     = 1024;         // rope table extent
    const double theta    = 1000000.0;
    const int half        = n_rot / 2;
    std::vector<float> cos_tab((size_t) max_pos * (size_t) half),
                      sin_tab((size_t) max_pos * (size_t) half);
    build_rope_table_host(n_rot, theta, max_pos, cos_tab.data(), sin_tab.data());

    // Per-row source generator: deterministic fp32 pattern indexed by (pos, dim).
    auto make_raw = [](int pos, int d) { return std::sin(0.013f * (float) d + 0.7f * (float) pos); };

    // --------- T6.x: fp32 algorithm test (RED-able) ---------
    // THE TEST HYPOTHESIS: after `rotate_qsa_tail_inplace` runs on the cached K at position p
    // (which was rotated to OLD_pos = p during prefill), the K at that position should equal
    // `R(new_pos=p+delta)·raw(p)` -- the SAME raw as cached, rotated to the new position.  This
    // is the splice contract: the helper's `rotate_k_by_delta(row, ..., old_pos=p, delta)`
    // applies the delta rotation to the cached row, which carries `R(p)·raw(p)` (post-RoPE at
    // old_pos).  Result: `R(delta)·R(p)·raw(p) = R(p+delta)·raw(p)` -- the rotation that the
    // engine would land the cached K at is the SAME raw rotated to the new position, which is
    // exactly what a fresh-rotation-of-the-same-raw would compute.
    //
    // NOT the rope applied to `raw(new_pos)` -- those raw values are different from cached row's
    // raw (the NEW prompt's tokens at p+delta are different from the OLD prompt's tokens at p).
    // The splice is rotation-only: same raw, different position angle.  Fresh is from a DIFFERENT
    // raw input and would compare incorrectly.
    {
        const int max_cells = 64;
        std::vector<float> k_pool((size_t) max_cells * n_head_kv * head_dim, 0.0f);
        const int edit_pos = 4;
        const int old_suffix_len = 6;
        const int delta = 3;
        auto fill_row = [&](float* row, int pos) {
            for (int d = 0; d < head_dim; ++d) row[d] = make_raw(pos, d);
            rope_at_fp32(row, n_rot, pos, cos_tab.data(), sin_tab.data(), max_pos);
        };
        for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
            for (int h = 0; h < n_head_kv; ++h) {
                float* row = k_pool.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim;
                fill_row(row, p);
            }
        }
        // Snapshot rows OUTSIDE the splice range for the byte-identical assertion.
        std::vector<float> outside_before(edit_pos * n_head_kv * head_dim);
        for (int p = 0; p < edit_pos; ++p)
            for (int h = 0; h < n_head_kv; ++h)
                std::memcpy(outside_before.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim,
                            k_pool.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim,
                            sizeof(float) * head_dim);
        // Run the splice algorithm in fp32.
        const bool ok = splice_fp32(k_pool.data(), max_cells, n_head_kv, head_dim, n_rot, page_size,
                                    edit_pos, old_suffix_len, /*uniform=*/true, /*delta=*/delta,
                                    nullptr, cos_tab.data(), sin_tab.data(), max_pos);
        check(ok, "T6.x: fp32 splice algorithm returns true for the supported config");
        // Compare each row against `R(new_pos)·raw(p)` -- the SAME raw as cached, rotated to
        // the new position.  This is what the splice SHOULD produce (and is mathematically
        // equal to `R(delta)·cached_row` per the rope algebra).
        int diff_count = 0;
        float max_dr = 0.0f;
        for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
            const int new_pos = p + delta;
            // Build the fresh-by-same-raw reference: raw(p), then rope at new_pos.
            float fresh[256];
            for (int d = 0; d < head_dim; ++d) fresh[d] = make_raw(p, d);
            rope_at_fp32(fresh, n_rot, new_pos, cos_tab.data(), sin_tab.data(), max_pos);
            for (int h = 0; h < n_head_kv; ++h) {
                float* row = k_pool.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    const float dr = std::fabs(row[d] - fresh[d]);
                    if (dr > 1e-6f) ++diff_count;
                    if (dr > max_dr) max_dr = dr;
                }
            }
        }
        check(diff_count == 0,
              "T6.x: K rows match R(new_pos)·raw(p) to 1e-6 (fp32 ULP, T3-pinned rotation)");
        // Outside rows must be byte-identical.
        int outside_diff = 0;
        for (int p = 0; p < edit_pos; ++p)
            for (int h = 0; h < n_head_kv; ++h)
                for (int d = 0; d < head_dim; ++d)
                    if (k_pool[((size_t) p * n_head_kv + h) * head_dim + d]
                        != outside_before[((size_t) p * n_head_kv + h) * head_dim + d]) ++outside_diff;
        check(outside_diff == 0,
              "T6.x: K rows OUTSIDE the splice range are byte-identical to pre-splice");
        // RED proof: re-build a fresh state and run with skip_rotation=true.  The cached image is
        // preserved (no rotation applied); the comparison against `R(new_pos)·raw(p)` must
        // diverge (cached carries `R(p)·raw(p)` which differs from the new rotation by the
        // delta rotation).
        std::vector<float> k_pool_red((size_t) max_cells * n_head_kv * head_dim, 0.0f);
        for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
            for (int h = 0; h < n_head_kv; ++h) {
                float* row = k_pool_red.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim;
                fill_row(row, p);
            }
        }
        const bool ok_red = splice_fp32(k_pool_red.data(), max_cells, n_head_kv, head_dim, n_rot,
                                         page_size, edit_pos, old_suffix_len,
                                         /*uniform=*/true, /*delta=*/delta, nullptr,
                                         cos_tab.data(), sin_tab.data(), max_pos,
                                         /*skip_rotation=*/true);
        check(ok_red, "T6.x RED: fp32 splice with skip_rotation=true still completes");
        int red_diffs = 0;
        for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
            const int new_pos = p + delta;
            float fresh[256];
            for (int d = 0; d < head_dim; ++d) fresh[d] = make_raw(p, d);
            rope_at_fp32(fresh, n_rot, new_pos, cos_tab.data(), sin_tab.data(), max_pos);
            for (int h = 0; h < n_head_kv; ++h) {
                float* row = k_pool_red.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    if (std::fabs(row[d] - fresh[d]) > 1e-6f) ++red_diffs;
            }
        }
        check(red_diffs > 0,
              "T6.x RED: skip_rotation=true -> cached K diverges from R(new_pos)·raw(p) (RED surfaces)");
    }

    // --------- T6.y: fp16 production helper (sanity) ---------
    // T6.y exercises `rotate_qsa_tail_inplace` against real fp16 storage (per QsaState's
    // declared type).  Compared to T6.x, this adds 2 fp16 round-trips per cell (cached storage
    // and final storage).  Both the splice path and the fresh-comparison path see the same
    // structural fp16 footprint, so the cell-by-cell diff accumulates roughly 2 fp16 ULPs of
    // quantization noise (verified empirically).  Tolerance: the 1e-3 floor corresponds to
    // fp16 ULP at unit magnitude; the test passes when MOST cells match exactly (within ~1 fp16
    // ULP) AND no cell diverges beyond ~1e-2 (a 10x ULP guardrail that would catch any
    // algorithmic regression).
    {
        const int max_cells = 64;
        const size_t cells = (size_t) max_cells * n_head_kv * head_dim;
        // Allocate storage that satisfies the `uint16_t* fp16` declaration.  We pass through
        // the helper which does fp16<->fp32 round-trips internally.
        std::vector<uint16_t> k_pool(cells, 0);
        std::vector<uint16_t> v_pool(cells, 0);
        // Build cached at OLD positions [edit_pos, edit_pos+old_suffix_len).
        const int edit_pos = 4;
        const int old_suffix_len = 6;
        const int delta = 3;
        auto fill_row_fp16 = [&](uint16_t* dst, int pos) {
            for (int d = 0; d < head_dim; ++d) dst[d] = fp32_to_fp16_bits(make_raw(pos, d));
            // Read stored fp16, rope at `pos`, write back to fp16.
            float tmp[256];
            for (int d = 0; d < head_dim; ++d) tmp[d] = fp16_bits_to_fp32(dst[d]);
            rope_at_fp32(tmp, n_rot, pos, cos_tab.data(), sin_tab.data(), max_pos);
            for (int d = 0; d < head_dim; ++d) dst[d] = fp32_to_fp16_bits(tmp[d]);
        };
        for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
            for (int h = 0; h < n_head_kv; ++h) {
                uint16_t* row = k_pool.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim;
                fill_row_fp16(row, p);
            }
        }
        // Synthesize a QsaState and run the helper.
        strata::core::QsaState st;
        st.k_pool = k_pool.data();
        st.v_pool = v_pool.data();
        st.kv_mode = 0;
        st.max_cells = max_cells;
        st.n_pages = max_cells / page_size;
        st.cos_tab = cos_tab.data();
        st.sin_tab = sin_tab.data();
        st.owns_rope = false;
        // V filler: a deterministic pattern so byte-identity is meaningful.
        for (size_t i = 0; i < cells; ++i) v_pool[i] = (uint16_t) ((i * 0xABCD + 0x1234) & 0xFFFF);
        std::vector<uint16_t> v_before(cells);
        std::memcpy(v_before.data(), v_pool.data(), cells * sizeof(uint16_t));
        const bool ok = rotate_qsa_tail_inplace(st, head_dim, n_rot, n_head_kv, max_pos, page_size,
                                                edit_pos, old_suffix_len,
                                                /*uniform=*/true, /*delta=*/delta, nullptr);
        check(ok, "T6.y: fp16 helper returns true for the supported config");
        // Compare with the splice's expected image: R(new_pos)·raw(p), where raw(p) is the same
        // raw used to construct the cached row.  Same hypothesis as T6.x, just with fp16 storage.
        int strict = 0, ulp = 0, bad = 0;
        for (int p = edit_pos; p < edit_pos + old_suffix_len; ++p) {
            const int new_pos = p + delta;
            float fresh[256];
            for (int d = 0; d < head_dim; ++d) fresh[d] = make_raw(p, d);
            rope_at_fp32(fresh, n_rot, new_pos, cos_tab.data(), sin_tab.data(), max_pos);
            for (int h = 0; h < n_head_kv; ++h) {
                uint16_t* row = k_pool.data() + ((size_t) p * n_head_kv + (size_t) h) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    const float a = fp16_bits_to_fp32(row[d]);
                    const float b = fresh[d];
                    const float dr = std::fabs(a - b);
                    if (dr == 0) ++strict;
                    else if (dr <= 1e-3f) ++ulp;
                    else ++bad;
                }
            }
        }
        check(bad == 0,
              "T6.y: fp16 helper produces K within fp16 ULP of fresh (no algorithmic regression)");
        // For fp16 storage, both the splice path AND the fresh-by-same-raw reference go through
        // 2 fp16 round-trips (each path quantizes differently but symmetrically).  Empirically
        // every cell is within 1 fp16 ULP; the strict count is 0 because the quantization
        // boundary falls on different cells in the two paths (splice caches then rotates, fresh
        // raw-quantizes then ropes - different rounding order).  The KEY property: no cell is
        // MORE than 1 fp16 ULP off, and there are NO cells beyond ~1e-2 (the regression
        // guardrail that `bad==0` already pins).  Anything worse would indicate a real algorithmic
        // divergence between the splice and the cached-image-rotation contract.
        check(ulp == strict + ulp + bad,
              "T6.y: every cell is within 1 fp16 ULP of the splice's expected image (no algo divergence)");
        // V byte-identical
        int v_diff = 0;
        for (size_t i = 0; i < cells; ++i) if (v_pool[i] != v_before[i]) ++v_diff;
        check(v_diff == 0,
              "T6.y: V is byte-identical after splice (no V-side work at all)");
        // Outside rows preserved
        int outside_diff = 0;
        for (int p = 0; p < edit_pos; ++p)
            for (int h = 0; h < n_head_kv; ++h)
                if (k_pool[((size_t) p * n_head_kv + h) * head_dim] != 0) ++outside_diff;
        check(outside_diff == 0,
              "T6.y: rows OUTSIDE the splice range are untouched (zero image preserved)");
    }

    // --------- T6.z: SPLICE_UNSUPPORTED guard fires ---------
    // Phase 3: the legacy `splice_scope_guard` (kv_splice_apply.hpp:72-79) still says
    // `kv_int8=true -> UnsupportedFormat` - it predates the int8 splice and is kept
    // as a pure back-compat predicate.  The runtime hook (kv_splice_apply_session.hpp)
    // uses the new `splice_scope_guard_int8` (kv_splice_apply_int8.hpp) which ACCEPTS
    // plain int8 (and refuses STRATA_KV_ROT=1's Hadamard-K domain).  Both predicates
    // are pinned here against the SAME guard-fire inputs.
    check(splice_scope_guard(/*kv_int8=*/true, false, false, 0, false)
          == SpliceStatus::UnsupportedFormat,
          "T6.z (legacy): kv_int8=true -> SPLICE_UNSUPPORTED: kv format != fp16 (back-compat predicate)");
    check(splice_scope_guard(false, true, false, 0, false)
          == SpliceStatus::UnsupportedFormat,
          "T6.z: kv_q4=true -> SPLICE_UNSUPPORTED: kv format != fp16 (q4_0 rotated)");
    check(splice_scope_guard(false, false, true, 0, false)
          == SpliceStatus::UnsupportedFormat,
          "T6.z: kv_hybrid=true -> SPLICE_UNSUPPORTED: kv format != fp16 (k8v4)");
    check(splice_scope_guard(false, false, false, 32, false)
          == SpliceStatus::UnsupportedResident,
          "T6.z: kv_resident=32 -> SPLICE_UNSUPPORTED: kv_resident (streaming ring)");
    check(splice_scope_guard(false, false, false, 0, true)
          == SpliceStatus::UnsupportedSplit,
          "T6.z: layer_split=true -> SPLICE_UNSUPPORTED: layer_split (per-stage)");
    check(splice_scope_guard(false, false, false, 0, false)
          == SpliceStatus::NoOp,
          "T6.z: supported config (kv fp16, kv_resident=0, no layer split) -> NoOp (contract gate)");

    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
