// src/core/dflash2_small_bench.cpp - time of each of the drafter's small kernels, back to back (graph-free), on dummy data.
//
//     dflash2_small_bench [iters]
#include "strata/kernels/dflash2.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace strata::kernels;

static float* dev(size_t n, float v = 0.5f) {
    float* p;
    cudaMalloc((void**) &p, n * 4);
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = v * (float) ((i * 2654435761u >> 12) % 1000) / 1000.0f + 0.01f;
    cudaMemcpy(p, h.data(), n * 4, cudaMemcpyHostToDevice);
    return p;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int N = 2560, B = 8, I = 7680;
    float *h = dev(B * N), *xn = dev(B * N), *o = dev(B * N), *dyn = dev(B * 640), *base = dev(4 * N), *gamma = dev(I), *gate = dev(B * I), *up = dev(B * I);
    float *q = dev(B * 4096), *k = dev(B * 1024), *qn = dev(B * 4096), *kb = dev(B * 1024), *tab = dev(B * 128), *xd = dev(5 * N / 32 * 8);
    float *kr = dev((size_t) 2064 * 1024), *vr = dev((size_t) 2064 * 1024), *lg = dev((size_t) 7 * 106299), *tv = dev(7 * 16), *scr = dev(7 * 64 * 32), *hp = dev(7 * 256);
    uint8_t* xq;
    cudaMalloc((void**) &xq, 5 * N * 8 + 4096);
    cudaMemset(xq, 1, 5 * N * 8);
    int32_t *pos, *tok, *outt;
    float* outp;
    cudaMalloc((void**) &pos, 64);
    cudaMalloc((void**) &tok, 7 * 16 * 4);
    cudaMalloc((void**) &outt, 64);
    cudaMalloc((void**) &outp, 64);
    int32_t hpos[16] = {100, 101, 102, 103, 104, 105, 106, 107};
    cudaMemcpy(pos, hpos, 64, cudaMemcpyHostToDevice);
    cudaMemset(tok, 0, 7 * 16 * 4);
    uint16_t* cb;
    cudaMalloc((void**) &cb, (size_t) 1000 * 256 * 2);
    cudaMemset(cb, 0, (size_t) 1000 * 256 * 2);
    float* part;
    cudaMalloc((void**) &part, (size_t) kDf2MaxChunks * 8 * 32 * 130 * 4);
    cudaMemset(part, 0, (size_t) kDf2MaxChunks * 8 * 32 * 130 * 4);
    uint8_t* w;
    cudaMalloc((void**) &w, (size_t) 640 * N * 34 / 32);
    cudaMemset(w, 1, (size_t) 640 * N * 34 / 32);
    float* y = dev(8 * 640);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    auto time = [&](const char* name, const std::function<void()>& f) {
        f();
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        std::printf("%-28s %7.2f us\n", name, 1000.0 * ms / iters);
    };
    time("rms_quant (8 x 2560)", [&] { df2_rms_quant(h, gamma, xn, B, N, 1e-6f, xq, xd, nullptr); });
    time("conv_quant", [&] { df2_conv_quant(xn, dyn, 640, base, B, N, 16, xq, xd, nullptr); });
    time("conv_rms_quant", [&] { df2_conv_rms_quant(o, dyn, 640, base, h, gamma, xn, B, N, 16, 1e-6f, xq, xd, nullptr); });
    time("swiglu_quant (8 x 7680)", [&] { df2_swiglu_quant(gate, up, B, I, xq, xd, nullptr); });
    time("quant_rows (8 x 4096)", [&] { df2_quant_rows(q, 4096, B, xq, xd, nullptr); });
    time("norm_rope_qk", [&] { df2_norm_rope_qk(q, k, gamma, gamma, tab, qn, kb, B, 32, 8, 1e-6f, nullptr); });
    time("kv_ring (3 rows)", [&] { df2_kv_ring(k, k, gamma, pos, tab, kr, vr, 2064, 3, 8, 1e-6f, nullptr); });
    time("rope_table (8)", [&] { df2_rope_table(pos, 8, 10000.0f, tab, nullptr); });
    time("topk (7 x 106299)", [&] { df2_topk(lg, 106299, nullptr, 7, reinterpret_cast<uint32_t*>(scr), tv, tok, nullptr); });
    time("select (7 rows)", [&] { df2_select(hp, tv, tok, cb, cb, 256, 7, tok, outt, outp, nullptr); });
    int32_t h_bmeta[16] = {0, 3, 100, 0,  0, 3, 100, 0,  0, 3, 100, 0,  0, 3, 100, 0};
    int32_t* d_bmeta;
    cudaMalloc((void**) &d_bmeta, sizeof(h_bmeta));
    cudaMemcpy(d_bmeta, h_bmeta, sizeof(h_bmeta), cudaMemcpyHostToDevice);
    time("select_multi (4 x 3 rows)", [&] { df2_select_multi(hp, tv, tok, cb, cb, 256, d_bmeta, 4, outt, outp, nullptr); });
    time("pack_head_rows (3 rows)", [&] { df2_pack_head_rows(h, hp, xn, hp, 3, N, 256, nullptr); });
    {
        // Bitwise parity check: df2_pack_head_rows + df2_select_multi across 4 slots (k = 7, 3, 1, 5; R = 16) vs 4 solo df2_select calls
        const int ks[4] = {7, 3, 1, 5};
        const int prevs[4] = {3, 7, 1, 11};
        const int R = 16;
        float *fn_s[4], *hs_s[4], *fn_b, *hs_b, *tv_b = dev(R * 16, 0.2f), *ref_p, *got_p;
        int32_t *tk_b, *ref_t, *got_t, *d_prev;
        cudaMalloc((void**) &fn_b, (size_t) R * N * 4);
        cudaMalloc((void**) &hs_b, (size_t) R * 256 * 4);
        cudaMalloc((void**) &tk_b, (size_t) R * 16 * 4);
        cudaMalloc((void**) &ref_t, 32 * 4);
        cudaMalloc((void**) &got_t, 32 * 4);
        cudaMalloc((void**) &ref_p, 32 * 4);
        cudaMalloc((void**) &got_p, 32 * 4);
        cudaMalloc((void**) &d_prev, 4);
        std::vector<int32_t> h_tk(R * 16);
        for (int i = 0; i < R * 16; ++i) h_tk[i] = (i * 7 + 3) % 64;
        cudaMemcpy(tk_b, h_tk.data(), h_tk.size() * 4, cudaMemcpyHostToDevice);
        int off = 0;
        int32_t meta4[16] = {};
        for (int s = 0; s < 4; ++s) {
            fn_s[s] = dev(7 * N, 0.1f * (s + 1));
            hs_s[s] = dev(7 * 256, 0.15f * (s + 1));
            df2_pack_head_rows(fn_s[s], hs_s[s], fn_b + (size_t) off * N, hs_b + (size_t) off * 256, ks[s], N, 256, nullptr);
            cudaMemcpy(d_prev, &prevs[s], 4, cudaMemcpyHostToDevice);
            df2_select(hs_s[s], tv_b + off * 16, tk_b + off * 16, cb, cb, 256, ks[s], d_prev, ref_t + s * 8, ref_p + s * 8, nullptr);
            meta4[s * 4 + 0] = off;
            meta4[s * 4 + 1] = ks[s];
            meta4[s * 4 + 2] = prevs[s];
            off += ks[s];
        }
        cudaMemcpy(d_bmeta, meta4, sizeof(meta4), cudaMemcpyHostToDevice);
        df2_select_multi(hs_b, tv_b, tk_b, cb, cb, 256, d_bmeta, 4, got_t, got_p, nullptr);
        cudaDeviceSynchronize();
        int32_t ht_ref[32] = {}, ht_got[32] = {};
        float hp_ref[32] = {}, hp_got[32] = {};
        cudaMemcpy(ht_ref, ref_t, sizeof(ht_ref), cudaMemcpyDeviceToHost);
        cudaMemcpy(ht_got, got_t, sizeof(ht_got), cudaMemcpyDeviceToHost);
        cudaMemcpy(hp_ref, ref_p, sizeof(hp_ref), cudaMemcpyDeviceToHost);
        cudaMemcpy(hp_got, got_p, sizeof(hp_got), cudaMemcpyDeviceToHost);
        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < ks[s]; ++i) {
                if (ht_ref[s * 8 + i] != ht_got[s * 8 + i] || hp_ref[s * 8 + i] != hp_got[s * 8 + i]) {
                    std::fprintf(stderr, "FAIL: select_multi slot %d row %d (%d, %g) != solo (%d, %g)\n",
                                 s, i, ht_got[s * 8 + i], hp_got[s * 8 + i], ht_ref[s * 8 + i], hp_ref[s * 8 + i]);
                    return 1;
                }
            }
        }
        std::printf("select_multi + pack_head_rows parity (4 slots, k={7,3,1,5}): PASS\n");
    }
    time("conv (resid)", [&] { df2_conv(o, dyn, 640, base, h, h, B, N, 16, nullptr); });
    time("attention merge+quant only", [&] { df2_attention(q, kr, vr, kr, vr, pos, 2064, 2048, part, xq, xd, nullptr); });
    Df2Mat m{w, y, 640};
    time("gemm 640 x 2560 (tiny)", [&] { df2_gemm(&m, 1, xq, xd, N, 8, nullptr); });
    return 0;
}
