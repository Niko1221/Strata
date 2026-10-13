// src/kernels/kv_window_parity.cpp - the FP16 sliding window + attention sinks (kv_window.hpp) against a
// full-precision reference and against the quantized path it overlays (GPU, no model).
//
// The pools are built by hand: a Q4_0 pool and the FP16 overlay on one side, an FP16 pool holding the same cells on
// the other, and the real fused decode attention (qsa_decode_attn_step) runs on both. Then:
//   1. a selection of sinks and window cells only: the overlay's output is BITWISE equal to the FP16 pool's - the
//      sinks and the window are read exactly, which is the whole point of the overlay;
//   2. that same selection with the overlay off: NOT equal - the 4-bit rounding is what the overlay removes;
//   3. a selection of cells outside the window and the sinks: overlay on and off agree BITWISE, so the bulk read
//      is untouched;
//   4. kv_window_restore reproduces the bulk's own dequantization bitwise for the sinks and for the window.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_window.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) d = std::fmax(d, std::fabs(a[i] - b[i]));
    return d;
}
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 4;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size;
    const int N = 1024;                        // cells in the cache
    const int WINDOW = 128, SINKS = 8;
    const int pages = N / P;
    const k::KvWindowShape shape{WINDOW, SINKS};
    const int rows = N * H;                    // one row per (cell, kv head) in the identity layout
    const size_t q4_bytes = (size_t) rows * k::kv_q4_bytes_per_head(D);

    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K((size_t) N * H * D), V((size_t) N * H * D);
    for (int cell = 0; cell < N; ++cell) {
        const float sc = (cell % 17 == 0) ? 25.0f : (cell % 23 == 0 ? 1e-2f : 1.0f);
        for (int i = 0; i < H * D; ++i) {
            K[(size_t) cell * H * D + i] = nd(rng) * sc;
            V[(size_t) cell * H * D + i] = nd(rng) * sc;
        }
    }

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;   // identity: the resident mode
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    uint8_t *kq4 = dalloc<uint8_t>(q4_bytes), *vq4 = dalloc<uint8_t>(q4_bytes);
    uint16_t *kp = dalloc<uint16_t>((size_t) rows * D), *vp = dalloc<uint16_t>((size_t) rows * D);
    const int64_t win_rows = k::kv_window_slots(shape) * H * D;
    uint16_t* kw = dalloc<uint16_t>((size_t) win_rows);
    uint16_t* vw = dalloc<uint16_t>((size_t) win_rows);
    float* kcur = dalloc<float>((size_t) H * D);
    float* vcur = dalloc<float>((size_t) H * D);
    int32_t* step = dalloc<int32_t>(k::kStepCount);
    ck(cudaMemset(step, 0, k::kStepCount * 4), "step");

    // append every cell through both stores, and mirror the last `WINDOW` cells plus the sinks into the overlay
    for (int cell = 0; cell < N; ++cell) {
        const float* kc = K.data() + (size_t) cell * H * D;
        const float* vc = V.data() + (size_t) cell * H * D;
        ck(cudaMemcpy(kcur, kc, (size_t) H * D * 4, cudaMemcpyHostToDevice), "k");
        ck(cudaMemcpy(vcur, vc, (size_t) H * D * 4, cudaMemcpyHostToDevice), "v");
        int32_t hstep[k::kStepCount] = {cell, cell + 1, 0, 0};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        k::kv_append_q4_step(kq4, vq4, d_table, step, kcur, vcur, s, nullptr, nullptr);
        k::kv_append_step(kp, vp, d_table, step, kcur, vcur, s, nullptr, nullptr);
    }
    ck(cudaDeviceSynchronize(), "appends");
    float* dK = dalloc<float>(K.size());
    float* dV = dalloc<float>(V.size());
    ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
    ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
    k::kv_window_write_batch({kw, vw}, shape, 0, N, dK, dV, s, nullptr);   // prompt form, pos0 0
    ck(cudaDeviceSynchronize(), "window write");

    // the decode attention over three choices of pools
    const int64_t cap = N;
    float* q = dalloc<float>((size_t) s.n_head * D);
    {
        std::vector<float> hq((size_t) s.n_head * D);
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
    }
    int32_t* ids = dalloc<int32_t>(cap);
    float* scratch = dalloc<float>(k::qsa_decode_attn_scratch_floats(cap, s));
    float* attn = dalloc<float>((size_t) s.n_head * D);

    auto run = [&](const std::vector<int32_t>& sel, bool overlay, bool fp16_pool) {
        ck(cudaMemcpy(ids, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice), "ids");
        int32_t hstep[k::kStepCount] = {N - 1, N, 0, (int32_t) sel.size()};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        k::QsaAttnPools p;
        p.page_table = d_table;
        if (fp16_pool) {
            p.k_pool = kp;
            p.v_pool = vp;
        } else {
            p.k_q4 = kq4;
            p.v_q4 = vq4;
        }
        if (overlay) {
            p.k_win = kw;
            p.v_win = vw;
            p.win_cells = WINDOW;
            p.win_sinks = SINKS;
            p.bulk_rows = rows;
        }
        k::qsa_decode_attn_step(q, p, ids, step, cap, s, scratch, attn, nullptr);
        ck(cudaDeviceSynchronize(), "attention");
        std::vector<float> out((size_t) s.n_head * D);
        ck(cudaMemcpy(out.data(), attn, out.size() * 4, cudaMemcpyDeviceToHost), "attn out");
        return out;
    };

    // 1/2. sinks and window only: exact with the overlay, quantized without it
    std::vector<int32_t> hot;
    for (int c = 0; c < SINKS; ++c) hot.push_back(c);
    for (int c = N - WINDOW; c < N; ++c) hot.push_back(c);
    const std::vector<float> a_ref = run(hot, false, true);
    const std::vector<float> a_win = run(hot, true, false);
    const std::vector<float> a_q4 = run(hot, false, false);
    check(max_diff(a_win, a_ref) == 0.0f, "sinks + window read exactly (bitwise equal to the FP16 pool)");
    check(max_diff(a_q4, a_ref) > 0.0f, "q4_0 alone is not exact (the overlay is what removes the rounding)");

    // 3. cells outside the window and the sinks: the overlay changes nothing
    std::vector<int32_t> cold;
    for (int c = SINKS; c < N - WINDOW; c += 7) cold.push_back(c);
    const std::vector<float> c_win = run(cold, true, false);
    const std::vector<float> c_q4 = run(cold, false, false);
    check(max_diff(c_win, c_q4) == 0.0f, "a cell outside the window and the sinks reads the pool (bitwise)");
    check(max_diff(run(cold, false, true), c_q4) > 0.0f, "the bulk is still quantized (the overlay is not a copy)");

    // 4. the restore path reproduces the bulk dequantization bitwise
    ck(cudaMemset(kw, 0, (size_t) win_rows * 2), "clear k overlay");
    ck(cudaMemset(vw, 0, (size_t) win_rows * 2), "clear v overlay");
    k::QsaAttnPools bulk;
    bulk.page_table = d_table;
    bulk.k_q4 = kq4;
    bulk.v_q4 = vq4;
    k::kv_window_restore({kw, vw}, shape, bulk, 2, N, s, nullptr);
    ck(cudaDeviceSynchronize(), "restore");
    std::vector<uint8_t> hq4(q4_bytes), hv4(q4_bytes);
    ck(cudaMemcpy(hq4.data(), kq4, q4_bytes, cudaMemcpyDeviceToHost), "read k q4");
    ck(cudaMemcpy(hv4.data(), vq4, q4_bytes, cudaMemcpyDeviceToHost), "read v q4");
    std::vector<uint16_t> hkwin((size_t) win_rows), hvwin((size_t) win_rows);
    ck(cudaMemcpy(hkwin.data(), kw, (size_t) win_rows * 2, cudaMemcpyDeviceToHost), "read k overlay");
    ck(cudaMemcpy(hvwin.data(), vw, (size_t) win_rows * 2, cudaMemcpyDeviceToHost), "read v overlay");
    int mismatch = 0;
    const int row_bytes = (int) k::kv_q4_bytes_per_head(D);
    for (int cell = 0; cell < N; ++cell) {
        const int32_t slot = k::kv_window_slot(cell, N, WINDOW, SINKS);
        if (slot < 0) continue;
        for (int h = 0; h < H; ++h) {
            // the pools are page-major: [page][kv_head][page_size][head_dim / 32 * 18]
            const int row = (cell / P * H + h) * P + (cell % P);
            const int64_t dst = ((int64_t) slot * H + h) * D;
            for (int d = 0; d < D; ++d) {
                const int rem = d % 32, blk = d / 32;
                const uint8_t* kb = hq4.data() + (size_t) row * row_bytes + (size_t) blk * sizeof(k::block_q4_0);
                const uint8_t* kb_d = kb;
                const uint16_t kd = (uint16_t) (kb_d[0] | (kb_d[1] << 8));
                const uint8_t* knib = kb_d + 2;
                const int kcode = rem < 16 ? (knib[rem] & 0x0F) : (knib[rem - 16] >> 4);
                const uint16_t want_k = k::f16_from_f32((float) (kcode - 8) * k::f32_from_f16(kd));
                const uint8_t* vb = hv4.data() + (size_t) row * row_bytes + (size_t) blk * sizeof(k::block_q4_0);
                const uint16_t vd = (uint16_t) (vb[0] | (vb[1] << 8));
                const uint8_t* vnib = vb + 2;
                const int vcode = rem < 16 ? (vnib[rem] & 0x0F) : (vnib[rem - 16] >> 4);
                const uint16_t want_v = k::f16_from_f32((float) (vcode - 8) * k::f32_from_f16(vd));
                if (hkwin[dst + d] != want_k || hvwin[dst + d] != want_v) ++mismatch;
            }
        }
    }
    check(mismatch == 0, "kv_window_restore reproduces the bulk dequantization (bitwise)");
    // Membership, as the properties the readers rely on: a cell is in the overlay exactly when it is a sink or
    // inside the window; every slot is in range; and the sinks and the window together own DISTINCT slots - a
    // collision would have two live cells reading each other's K/V.
    {
        const int32_t slots = (int32_t) k::kv_window_slots(shape);
        bool ok = true, distinct = true, sink_slots = (k::kv_window_slot(SINKS - 1, N, WINDOW, SINKS) == SINKS - 1);
        std::vector<int> seen(slots, 0);
        for (int cell = 0; cell < N; ++cell) {
            const int32_t slot = k::kv_window_slot(cell, N, WINDOW, SINKS);
            const bool wanted = cell < SINKS || cell >= N - WINDOW;
            ok = ok && (wanted ? (slot >= 0 && slot < slots) : slot < 0);
            if (slot >= 0) {
                if (seen[slot]++) distinct = false;
            }
        }
        check(ok && distinct && sink_slots,
              "membership: sinks + the last `window` cells, in range and pairwise distinct");
    }

    // 5. cost: the overlay's own price in the fused decode attention, timed where it is paid - the kernel reads a
    // few hundred KB more per token and resolves one extra compare per selected cell. Wall time over many runs of
    // the same selection, overlay off against on (this is the only speed figure that is not confounded by the
    // model's own kernels or by draft acceptance).
    {
        const int cap2 = 8192;
        std::vector<int32_t> big(cap2);
        for (int i = 0; i < cap2; ++i) big[i] = (i * 7919) % N;
        int32_t* ids2 = dalloc<int32_t>(cap2);
        float* scratch2 = dalloc<float>(k::qsa_decode_attn_scratch_floats(cap2, s));
        ck(cudaMemcpy(ids2, big.data(), big.size() * 4, cudaMemcpyHostToDevice), "ids2");
        int32_t hstep[k::kStepCount] = {N - 1, N, 0, cap2};
        ck(cudaMemcpy(step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
        k::QsaAttnPools p0, p1;
        p0.page_table = d_table;
        p0.k_q4 = kq4;
        p0.v_q4 = vq4;
        p1 = p0;
        p1.k_win = kw;
        p1.v_win = vw;
        p1.win_cells = WINDOW;
        p1.win_sinks = SINKS;
        p1.bulk_rows = rows;
        auto time_it = [&](const k::QsaAttnPools& pp, int iters) {
            for (int i = 0; i < 20; ++i) k::qsa_decode_attn_step(q, pp, ids2, step, cap2, s, scratch2, attn, nullptr);
            ck(cudaDeviceSynchronize(), "warm");
            cudaEvent_t a, b;
            ck(cudaEventCreate(&a), "ev");
            ck(cudaEventCreate(&b), "ev");
            ck(cudaEventRecord(a), "ev");
            for (int i = 0; i < iters; ++i) k::qsa_decode_attn_step(q, pp, ids2, step, cap2, s, scratch2, attn, nullptr);
            ck(cudaEventRecord(b), "ev");
            ck(cudaEventSynchronize(b), "ev");
            float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, a, b), "ev");
            ck(cudaEventDestroy(a), "ev");
            ck(cudaEventDestroy(b), "ev");
            return ms / iters;
        };
        const double off = time_it(p0, 300), on = time_it(p1, 300);
        std::printf("\ndecode attention over %d cells: %.3f ms without the overlay, %.3f ms with it (%.1f%%)\n",
                    cap2, off, on, 100.0 * (on - off) / off);
        check(on / off < 1.25, "the overlay costs less than 25% of the attention kernel");
    }

    // 6. quality in error terms, for the pull request: how far each stored form is from an FP16 cache, and what the
    // overlay changes about it - reported per position bucket, since the overlay only covers two of them.
    {
        std::vector<uint16_t> hk((size_t) rows * D), hv((size_t) rows * D), hw((size_t) win_rows), hvw((size_t) win_rows);
        std::vector<uint8_t> hq4(q4_bytes);
        // the restore above overwrote the overlay with dequantized bulk; put the appends' exact copies back, since
        // this section measures what a running cache holds
        k::kv_window_write_batch({kw, vw}, shape, 0, N, dK, dV, s, nullptr);
        ck(cudaDeviceSynchronize(), "window rewrite");
        ck(cudaMemcpy(hk.data(), kp, (size_t) rows * D * 2, cudaMemcpyDeviceToHost), "read fp16 pool");
        ck(cudaMemcpy(hv.data(), vp, (size_t) rows * D * 2, cudaMemcpyDeviceToHost), "read fp16 pool");
        ck(cudaMemcpy(hq4.data(), kq4, q4_bytes, cudaMemcpyDeviceToHost), "read q4 pool");
        ck(cudaMemcpy(hw.data(), kw, (size_t) win_rows * 2, cudaMemcpyDeviceToHost), "read overlay");
        ck(cudaMemcpy(hvw.data(), vw, (size_t) win_rows * 2, cudaMemcpyDeviceToHost), "read overlay");
        const int row_bytes = (int) k::kv_q4_bytes_per_head(D);
        struct Bucket { const char* name; int lo, hi; };
        const Bucket buckets[3] = {{"sinks (first 8)", 0, SINKS},
                                   {"window (last 128)", N - WINDOW, N},
                                   {"bulk (the rest)", SINKS, N - WINDOW}};
        std::printf("\nK error against an FP16 cache (V is stored and read the same way), relative RMS:\n");
        std::printf("  %-18s %14s %18s\n", "bucket", "q4_0", "q4_0 + overlay");
        for (const Bucket& b : buckets) {
            double eq = 0.0, er = 0.0, ref = 0.0;
            long long n = 0;
            for (int cell = b.lo; cell < b.hi; ++cell) {
                const int32_t slot = k::kv_window_slot(cell, N, WINDOW, SINKS);
                for (int h = 0; h < H; ++h) {
                    const int row = (cell / P * H + h) * P + (cell % P);
                    const int64_t dst = slot >= 0 ? ((int64_t) slot * H + h) * D : 0;
                    for (int d = 0; d < D; ++d) {
                        const float exact = k::f32_from_f16(hk[(size_t) row * D + d]);   // the FP16 pool = reference
                        const int blk = d / 32, rem = d % 32;
                        const uint8_t* qb = hq4.data() + (size_t) row * row_bytes + (size_t) blk * sizeof(k::block_q4_0);
                        const uint16_t qd = (uint16_t) (qb[0] | (qb[1] << 8));
                        const int code = rem < 16 ? (qb[2 + rem] & 0x0F) : (qb[2 + rem - 16] >> 4);
                        const double q4 = (double) (code - 8) * k::f32_from_f16(qd);
                        const double w4 = slot >= 0 ? k::f32_from_f16(hw[dst + d]) : q4;   // what the readers see
                        eq += (q4 - exact) * (q4 - exact);
                        er += (w4 - exact) * (w4 - exact);
                        ref += exact * exact;
                        ++n;
                    }
                }
            }
            const double rms_q4 = std::sqrt(eq / (double) n), rms_w = std::sqrt(er / (double) n);
            const double scale = std::sqrt(ref / (double) n);
            std::printf("  %-18s %13.3f%% %17.3f%%\n", b.name, 100.0 * rms_q4 / scale, 100.0 * rms_w / scale);
            if (b.lo != SINKS || b.hi != N - WINDOW)
                check(rms_w == 0.0, "     that bucket reads exactly FP16 with the overlay (0.000%)");
            else
                check(rms_w == rms_q4, "     the bulk is untouched by the overlay (same error as q4_0)");
        }
        std::printf("  footprint: overlay %lld KB per layer for %d cells (%d sinks + %d window + %d slack); "
                    "q4_0 bulk %zu KB for %d cells\n",
                    (long long) (k::kv_window_bytes(s, shape) / 1024), (int) k::kv_window_slots(shape), SINKS, WINDOW,
                    (int) k::kKvWindowSlack, q4_bytes / 1024, N);
    }

    std::printf(g_fail == 0 ? "\nkv_window_parity: PASS\n" : "\nkv_window_parity: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
