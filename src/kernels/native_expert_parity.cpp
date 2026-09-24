// src/kernels/native_expert_parity.cpp - plan v0.3 P6: one native expert three ways, on real GGUF rows.
//
//     build/native_expert_parity <shard1.gguf> [layer ...]
//
// (a) float reference: ggml's own dequantizer (`to_float`) and a float SwiGLU expert, (b) the CPU path
// (ggml-cpu vec_dot with its quantized activations), (c) the GPU path (`native_expert_grouped`, q8_1
// activations).  (b) and (c) each differ from (a) by their activation rounding only (a few 1e-3 relative).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "ggml-cpu.h"
#include "strata/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: native_expert_parity <shard1.gguf> [layer ...]\n"); return 2; }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
    if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
    const int NT = 3, E = 7;
    const int64_t H = 2560, FF = 640;
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        std::vector<uint8_t> blob(f.bytes);
        std::memcpy(blob.data(), gguf.tensor_data(*t[0]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.up_off, gguf.tensor_data(*t[1]) + (size_t) E * f.up_off, f.up_off);
        std::memcpy(blob.data() + f.down_off, gguf.tensor_data(*t[2]) + (size_t) E * (f.bytes - f.down_off),
                    f.bytes - f.down_off);
        // (a) the float reference
        const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
        const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
        std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
        for (int64_t r = 0; r < FF; ++r) {
            tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
            tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
        }
        for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
        std::mt19937 rng(11 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NT * H);
        for (auto& v : x) v = nd(rng);
        std::vector<float> ref((size_t) NT * H), got_c((size_t) NT * H), got_g((size_t) NT * H);
        for (int k = 0; k < NT; ++k) {
            std::vector<float> h(FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * x[k * H + i]; u += (double) U[r * H + i] * x[k * H + i]; }
                h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
            }
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
                ref[k * H + r] = (float) o;
            }
        }
        // (b) the CPU
        {
            std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
            std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
            const void* a[NT];
            float* ffp[NT];
            const void* hp[NT];
            float* op[NT];
            for (int k = 0; k < NT; ++k) {
                cpu::native_quant_act(f, x.data() + k * H, act[k].data());
                a[k] = act[k].data();
                ffp[k] = ff[k].data();
            }
            cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
            if (cpu::iq512_supported(f.gu_type)) {
                // the AVX-512 rows against ggml's own vec_dot, same Q8_K activations: float-order differences only
                std::vector<float> g512((size_t) NT * FF), gref((size_t) NT * FF);
                float* gp[NT];
                for (int k = 0; k < NT; ++k) gp[k] = g512.data() + k * FF;
                cpu::iq512_rows(f.gu_type, blob.data(), f.gu_row, (int) H, a, NT, gp, 0, (int) FF);
                const auto* tc = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
                for (int k = 0; k < NT; ++k)
                    for (int64_t r = 0; r < FF; ++r)
                        tc->vec_dot((int) H, &gref[k * FF + r], 0, blob.data() + r * f.gu_row, 0, a[k], 0, 1);
                {   // single-thread throughput, the weights cache-resident (compute bound)
                    float* f1[1] = {ffp[0]};
                    const int it = 50;
                    auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < it; ++i) cpu::iq512_gu_rows(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, 1, f1, 0, (int) FF);
                    auto t1 = std::chrono::steady_clock::now();
                    for (int i = 0; i < it; ++i)
                        for (int64_t r = 0; r < FF; ++r) {
                            float gg, uu;
                            tc->vec_dot((int) H, &gg, 0, blob.data() + r * f.gu_row, 0, a[0], 0, 1);
                            tc->vec_dot((int) H, &uu, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[0], 0, 1);
                        }
                    auto t2 = std::chrono::steady_clock::now();
                    const double us512 = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
                    const double usg = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
                    std::printf("          gate+up one token one thread: avx512 %.0f us (%.2f GB/s), ggml %.0f us (%.2f GB/s)\n",
                                us512, 2.0 * f.up_off / us512 / 1e3, usg, 2.0 * f.up_off / usg / 1e3);
                    auto t3 = std::chrono::steady_clock::now();
                    for (int i = 0; i < it; ++i) cpu::iq512_gu_rows(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, NT, ffp, 0, (int) FF);
                    auto t4 = std::chrono::steady_clock::now();
                    std::printf("          gate+up %d tokens one thread: avx512 %.0f us vs ggml %d x %.0f us\n", NT,
                                std::chrono::duration<double, std::micro>(t4 - t3).count() / it, NT, usg);
                }
                std::printf("          %s AVX-512 gate rows vs ggml vec_dot: rel %.2e\n",
                            ggml_type_name((ggml_type) f.gu_type), rel(g512, gref));
            }
            for (int k = 0; k < NT; ++k) {
                cpu::native_quant_h(f, ff[k].data(), hq[k].data());
                hp[k] = hq[k].data();
                op[k] = got_c.data() + k * H;
            }
            cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
            if (f.d_type == 42) {
                // (b2) the AVX-512 GGUF-layout Q2_0 kernel the pool uses for Q2_0 down projections
                std::vector<cpu::ActQ> a2(NT);
                const cpu::ActQ* ap[NT];
                std::vector<float> alt((size_t) NT * H);
                float* altp[NT];
                for (int k = 0; k < NT; ++k) {
                    cpu::act_quant_q8_1(ff[k].data(), (int) FF, a2[k]);
                    ap[k] = &a2[k];
                    altp[k] = alt.data() + k * H;
                }
                cpu::q2_0_gguf_rows_multi(blob.data() + f.down_off, f.d_row, (int) (FF / 64), ap, NT, altp, 0, (int) H);
                std::printf("          q2_0 AVX-512 down vs ggml down: rel %.2e\n", rel(alt, got_c));
            }
        }
        // (c) the GPU: one group holding the NT entries
        {
            const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
            void *dblob, *dx, *dxq, *dscr;
            float* dout;
            unsigned long long* dptr;
            int32_t *dstart, *dn, *ddst, *dtok;
            cudaMalloc(&dblob, blob.size());
            cudaMalloc(&dx, x.size() * 4);
            cudaMalloc(&dxq, (size_t) NT * H / 32 * 36);
            cudaMalloc(&dscr, strata::kernels::native_expert_scratch_bytes(NT, FF));
            cudaMalloc((void**) &dout, (size_t) NT * H * 4);
            cudaMalloc((void**) &dptr, 8);
            cudaMalloc((void**) &dstart, 8);
            cudaMalloc((void**) &dn, 4);
            cudaMalloc((void**) &ddst, NT * 4);
            cudaMalloc((void**) &dtok, NT * 4);
            cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            const unsigned long long p = (unsigned long long) dblob;
            const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
            cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
            cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
            cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(ddst, idx, NT * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dtok, idx, NT * 4, cudaMemcpyHostToDevice);
            strata::kernels::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);
            strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(got_g.data(), dout, got_g.size() * 4, cudaMemcpyDeviceToHost);
            cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
            cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
        }
        const double ec = rel(got_c, ref), eg = rel(got_g, ref), ecg = rel(got_c, got_g);
        const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
        std::printf("layer %2d  %-8s/%-7s blob %8zu  cpu rel %.2e  gpu rel %.2e  cpu-gpu %.2e  %s\n", l,
                    ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), f.bytes, ec, eg, ecg,
                    ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    }
    std::printf("native_expert_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
