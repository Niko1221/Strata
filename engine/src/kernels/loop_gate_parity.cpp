// src/kernels/loop_gate_parity.cpp - #7295's test for the recurrent-depth loop's per-layer message +
// gate, a port of Inkling2Loop's attn_res_mix_f32/loop_control_f32
// (engine/src/inkling2/inkling2_loop.cpp). The reference below is a direct, independent double-precision
// transcription of that file's math (not of this file's kernels), adapted for 2 channels - see
// include/strata/kernels/loop_gate.hpp's file comment for the exact formulas and the adaptation.
#include "strata/kernels/loop_gate.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint16_t bf16_bits(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;   // round-to-nearest-even
    return (uint16_t) (i >> 16);
}

/// `attn_res_mix_f32`, transcribed in double, for 2 queries (channels) and a bank of `bank_n` states plus
/// `bank_out`.
void reference_msg_attn(const std::vector<float>& bank, const std::vector<float>& bank_out,
                        const std::vector<float>& query, int64_t n_embd, int64_t bank_n,
                        std::vector<float>& msgs) {
    msgs.assign((size_t) (2 * n_embd), 0.0f);
    const int64_t n_states = bank_n + 1;
    auto state = [&](int64_t s) -> const float* {
        return s < bank_n ? bank.data() + (size_t) s * n_embd : bank_out.data();
    };
    for (int64_t q = 0; q < 2; ++q) {
        const float* qv = query.data() + (size_t) q * n_embd;
        std::vector<double> rs(n_states), score(n_states);
        for (int64_t s = 0; s < n_states; ++s) {
            const float* sv = state(s);
            double dot = 0.0, ss = 0.0;
            for (int64_t d = 0; d < n_embd; ++d) { dot += (double) sv[d] * qv[d]; ss += (double) sv[d] * sv[d]; }
            rs[s] = 1.0 / std::sqrt(ss / (double) n_embd + 1e-6);
            score[s] = dot * rs[s] / std::sqrt((double) n_embd);
        }
        const double mx = *std::max_element(score.begin(), score.end());
        std::vector<double> w(n_states);
        double sum_w = 0.0;
        for (int64_t s = 0; s < n_states; ++s) { w[s] = std::exp(score[s] - mx); sum_w += w[s]; }
        for (int64_t s = 0; s < n_states; ++s) w[s] /= sum_w;
        for (int64_t d = 0; d < n_embd; ++d) {
            double acc = 0.0;
            for (int64_t s = 0; s < n_states; ++s) acc += w[s] * state(s)[d] * rs[s];
            msgs[(size_t) (q * n_embd + d)] = (float) acc;
        }
    }
}

/// `loop_control_f32`, transcribed in double.
void reference_ctrl_gate(const std::vector<float>& cur, const std::vector<float>& msg,
                         const std::vector<float>& down, const std::vector<float>& gain,
                         const std::vector<float>& forget, int64_t n_embd, int64_t rank,
                         std::vector<float>& out) {
    out.resize((size_t) n_embd);
    double ss_c = 0.0, ss_m = 0.0;
    for (int64_t d = 0; d < n_embd; ++d) { ss_c += (double) cur[d] * cur[d]; ss_m += (double) msg[d] * msg[d]; }
    const double rs_c = 1.0 / std::sqrt(ss_c / (double) n_embd + 1e-6);
    const double rs_m = 1.0 / std::sqrt(ss_m / (double) n_embd + 1e-6);
    std::vector<double> cat((size_t) (2 * n_embd));
    for (int64_t d = 0; d < n_embd; ++d) { cat[(size_t) d] = cur[d] * rs_c; cat[(size_t) (n_embd + d)] = msg[d] * rs_m; }
    std::vector<double> gate((size_t) rank);
    for (int64_t r = 0; r < rank; ++r) {
        double acc = 0.0;
        for (int64_t i = 0; i < 2 * n_embd; ++i) acc += down[(size_t) (r * 2 * n_embd + i)] * cat[(size_t) i];
        gate[(size_t) r] = acc;
    }
    for (int64_t d = 0; d < n_embd; ++d) {
        double ga = 0.0, fa = 0.0;
        for (int64_t r = 0; r < rank; ++r) {
            ga += gain[(size_t) (d * rank + r)] * gate[(size_t) r];
            fa += forget[(size_t) (d * rank + r)] * gate[(size_t) r];
        }
        const double g = std::tanh(ga), f = std::tanh(fa);
        out[(size_t) d] = (float) ((1.0 - f) * cur[d] + g * msg[d]);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(7295);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ---- loop_msg_attn ----
    {
        const int64_t n_embd = 96, bank_n = 3;
        std::vector<float> bank((size_t) (bank_n * n_embd)), bank_out((size_t) n_embd), query((size_t) (2 * n_embd));
        for (auto& v : bank) v = dist(rng);
        for (auto& v : bank_out) v = dist(rng);
        for (auto& v : query) v = dist(rng) * 0.3f;

        std::vector<float> want;
        reference_msg_attn(bank, bank_out, query, n_embd, bank_n, want);

        float *d_bank, *d_bank_out, *d_query, *d_msgs;
        check(cudaMalloc(&d_bank, bank.size() * 4), "malloc bank");
        check(cudaMalloc(&d_bank_out, bank_out.size() * 4), "malloc bank_out");
        check(cudaMalloc(&d_query, query.size() * 4), "malloc query");
        check(cudaMalloc(&d_msgs, want.size() * 4), "malloc msgs");
        check(cudaMemcpy(d_bank, bank.data(), bank.size() * 4, cudaMemcpyHostToDevice), "copy bank");
        check(cudaMemcpy(d_bank_out, bank_out.data(), bank_out.size() * 4, cudaMemcpyHostToDevice), "copy bank_out");
        check(cudaMemcpy(d_query, query.data(), query.size() * 4, cudaMemcpyHostToDevice), "copy query");

        strata::kernels::loop_msg_attn(d_bank, d_bank_out, d_query, strata::kernels::LoopMsgShapes{n_embd, bank_n},
                                       d_msgs, nullptr);
        std::vector<float> got(want.size());
        check(cudaMemcpy(got.data(), d_msgs, got.size() * 4, cudaMemcpyDeviceToHost), "copy msgs");

        double max_err = 0.0;
        for (size_t i = 0; i < got.size(); ++i) max_err = std::max(max_err, (double) std::fabs(got[i] - want[i]));
        if (max_err > 1.0e-4) {
            std::fprintf(stderr, "loop_msg_attn: mixed vs reference max abs err %.6g\n", max_err);
            return 1;
        }
        std::fprintf(stderr, "loop_msg_attn: vs reference OK, max abs err %.3g\n", max_err);

        // Property: bank_n == 0 (only `bank_out` as a state) - softmax of one state is weight 1, so the
        // message is exactly `bank_out` normalized (not raw `bank_out`: the state IS normalized before
        // the weighted sum, with weight 1).
        {
            float* d_msgs0;
            check(cudaMalloc(&d_msgs0, (size_t) (2 * n_embd) * 4), "malloc msgs0");
            strata::kernels::loop_msg_attn(nullptr, d_bank_out, d_query, strata::kernels::LoopMsgShapes{n_embd, 0},
                                           d_msgs0, nullptr);
            std::vector<float> got0((size_t) (2 * n_embd));
            check(cudaMemcpy(got0.data(), d_msgs0, got0.size() * 4, cudaMemcpyDeviceToHost), "copy msgs0");
            double ss = 0.0;
            for (float v : bank_out) ss += (double) v * v;
            const double rs = 1.0 / std::sqrt(ss / (double) n_embd + 1e-6);
            double max_err0 = 0.0;
            for (int64_t q = 0; q < 2; ++q)
                for (int64_t d = 0; d < n_embd; ++d)
                    max_err0 = std::max(max_err0, (double) std::fabs(got0[(size_t) (q * n_embd + d)] -
                                                                     (float) (bank_out[(size_t) d] * rs)));
            if (max_err0 > 1.0e-4) {
                std::fprintf(stderr, "loop_msg_attn: bank_n=0 property failed, max abs err %.6g\n", max_err0);
                return 1;
            }
            std::fprintf(stderr, "loop_msg_attn: bank_n=0 property OK\n");
            cudaFree(d_msgs0);
        }
        cudaFree(d_bank); cudaFree(d_bank_out); cudaFree(d_query); cudaFree(d_msgs);
    }

    // ---- loop_ctrl_gate ----
    {
        const int64_t n_embd = 96, rank = 8;
        std::vector<float> cur((size_t) n_embd), msg((size_t) n_embd);
        std::vector<float> down_f((size_t) (rank * 2 * n_embd)), gain_f((size_t) (n_embd * rank)), forget_f((size_t) (n_embd * rank));
        for (auto& v : cur) v = dist(rng);
        for (auto& v : msg) v = dist(rng);
        for (auto& v : down_f) v = dist(rng) * 0.1f;
        for (auto& v : gain_f) v = dist(rng) * 0.1f;
        for (auto& v : forget_f) v = dist(rng) * 0.1f;

        std::vector<uint16_t> down_bf(down_f.size()), gain_bf(gain_f.size()), forget_bf(forget_f.size());
        for (size_t i = 0; i < down_f.size(); ++i) down_bf[i] = bf16_bits(down_f[i]);
        for (size_t i = 0; i < gain_f.size(); ++i) gain_bf[i] = bf16_bits(gain_f[i]);
        for (size_t i = 0; i < forget_f.size(); ++i) forget_bf[i] = bf16_bits(forget_f[i]);
        for (size_t i = 0; i < down_f.size(); ++i) down_f[i] = strata::kernels::f32_from_bf16(down_bf[i]);
        for (size_t i = 0; i < gain_f.size(); ++i) gain_f[i] = strata::kernels::f32_from_bf16(gain_bf[i]);
        for (size_t i = 0; i < forget_f.size(); ++i) forget_f[i] = strata::kernels::f32_from_bf16(forget_bf[i]);

        std::vector<float> want;
        reference_ctrl_gate(cur, msg, down_f, gain_f, forget_f, n_embd, rank, want);

        float *d_cur, *d_msg, *d_out, *d_ws;
        uint16_t *d_down, *d_gain, *d_forget;
        const strata::kernels::LoopCtrlShapes shapes{n_embd, rank};
        const size_t ws_bytes = strata::kernels::loop_ctrl_workspace_bytes(shapes);
        check(cudaMalloc(&d_cur, cur.size() * 4), "malloc cur");
        check(cudaMalloc(&d_msg, msg.size() * 4), "malloc msg");
        check(cudaMalloc(&d_out, want.size() * 4), "malloc out");
        check(cudaMalloc(&d_ws, ws_bytes), "malloc ws");
        check(cudaMalloc(&d_down, down_bf.size() * 2), "malloc down");
        check(cudaMalloc(&d_gain, gain_bf.size() * 2), "malloc gain");
        check(cudaMalloc(&d_forget, forget_bf.size() * 2), "malloc forget");
        check(cudaMemcpy(d_cur, cur.data(), cur.size() * 4, cudaMemcpyHostToDevice), "copy cur");
        check(cudaMemcpy(d_msg, msg.data(), msg.size() * 4, cudaMemcpyHostToDevice), "copy msg");
        check(cudaMemcpy(d_down, down_bf.data(), down_bf.size() * 2, cudaMemcpyHostToDevice), "copy down");
        check(cudaMemcpy(d_gain, gain_bf.data(), gain_bf.size() * 2, cudaMemcpyHostToDevice), "copy gain");
        check(cudaMemcpy(d_forget, forget_bf.data(), forget_bf.size() * 2, cudaMemcpyHostToDevice), "copy forget");

        strata::kernels::loop_ctrl_gate(d_cur, d_msg, d_down, d_gain, d_forget, shapes, d_ws, d_out, nullptr);
        std::vector<float> got(want.size());
        check(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "copy out");

        double max_err = 0.0;
        for (size_t i = 0; i < got.size(); ++i) max_err = std::max(max_err, (double) std::fabs(got[i] - want[i]));
        if (max_err > 2.0e-4) {
            std::fprintf(stderr, "loop_ctrl_gate: mixed vs reference max abs err %.6g\n", max_err);
            return 1;
        }
        std::fprintf(stderr, "loop_ctrl_gate: vs reference OK, max abs err %.3g\n", max_err);

        // Property: every output lies within [cur, msg]'s span is NOT guaranteed here (g, f are
        // independent tanh gates, not a single convex blend like loop_gate.hpp's predecessor) - so no
        // structural property is asserted beyond the reference match, honestly: this IS the ported
        // formula's actual shape.
        cudaFree(d_cur); cudaFree(d_msg); cudaFree(d_out); cudaFree(d_ws);
        cudaFree(d_down); cudaFree(d_gain); cudaFree(d_forget);
    }

    std::fprintf(stderr, "loop_gate_parity: PASS\n");
    return 0;
}
