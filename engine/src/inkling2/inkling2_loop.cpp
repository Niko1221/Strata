// src/inkling2/inkling2_loop.cpp - Native ModelFamily::Inkling2Loop + DSpark Block-8 Markov ops
// MIT License - Copyright (c) 2026 Strata contributors
#include "strata/inkling2/inkling2_loop.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/isa.hpp"
#include "strata/quant/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::inkling2 {

using namespace strata::quant::sk;

namespace {

inline float fp16_bits_to_f32(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127 - 14;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03FFu;
            f = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

inline float dot_f32(const float* a, const float* b, size_t n) noexcept {
    return strata::quant::sk::dot_f32(a, b, n);   // the run-time ISA tier (AVX-512 / AVX2 / NEON; plain C++ on a CPU with none)
}

}  // namespace

float situ_scalar(float g, float u, float beta, float lb) noexcept {
    const float sig = 1.0f / (1.0f + std::exp(-g));
    const float g_act = beta * std::tanh(g / beta) * sig;
    const float u_act = lb * std::tanh(u / lb);
    return g_act * u_act;
}

void rms_norm_f32(const float* x, const float* w, float* out, size_t n, float eps) noexcept {
    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    if (w) {
        for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
    } else {
        for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv;
    }
}

void l2_clip_f32(const float* x, const float* scale, float* out, size_t n, float threshold) noexcept {
    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = scale ? (x[i] * scale[i]) : x[i];
        out[i] = v;
        ss += v * v;
    }
    const float norm = std::sqrt(ss) + 1e-6f;
    const float factor = std::min(1.0f, threshold / norm);
    for (size_t i = 0; i < n; ++i) out[i] *= factor;
}

float log_pos_tau(int pos, int n_floor, float alpha) noexcept {
    const float r = static_cast<float>(pos + 1) / static_cast<float>(n_floor);
    return 1.0f + alpha * std::log(std::max(1.0f, r));
}

void attn_res_mix_f32(
    const std::vector<const float*>& blocks,
    const float* cur,
    const float* queries,
    size_t n_queries,
    size_t D,
    float* out) noexcept {
    const size_t n_states = blocks.size() + 1;
    std::vector<float> normed(n_states * D);
    for (size_t b = 0; b < blocks.size(); ++b) {
        rms_norm_f32(blocks[b], nullptr, normed.data() + b * D, D, kInklingRmsEps);
    }
    rms_norm_f32(cur, nullptr, normed.data() + blocks.size() * D, D, kInklingRmsEps);

    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> scores(n_states);

    for (size_t q = 0; q < n_queries; ++q) {
        const float* qv = queries + q * D;
        float mx = -1e30f;
        for (size_t s = 0; s < n_states; ++s) {
            scores[s] = dot_f32(normed.data() + s * D, qv, D) * inv_sqrt_d;
            if (scores[s] > mx) mx = scores[s];
        }
        float sum = 0.0f;
        for (size_t s = 0; s < n_states; ++s) {
            scores[s] = std::exp(scores[s] - mx);
            sum += scores[s];
        }
        const float inv_sum = 1.0f / sum;
        float* dst = out + q * D;
        std::memset(dst, 0, D * sizeof(float));
        for (size_t s = 0; s < n_states; ++s) {
            const float w = scores[s] * inv_sum;
            const float* sv = normed.data() + s * D;
            for (size_t d = 0; d < D; ++d) dst[d] += w * sv[d];
        }
    }
}

void loop_control_f32(
    const float* cur,
    const float* msg,
    const float* w_down,
    const float* w_gain,
    const float* w_forget,
    size_t D,
    size_t rank,
    float* out) noexcept {
    std::vector<float> cat(2 * D);
    rms_norm_f32(cur, nullptr, cat.data(), D, kInklingRmsEps);
    rms_norm_f32(msg, nullptr, cat.data() + D, D, kInklingRmsEps);

    std::vector<float> gate(rank);
    for (size_t r = 0; r < rank; ++r) {
        gate[r] = dot_f32(w_down + r * (2 * D), cat.data(), 2 * D);
    }
    for (size_t d = 0; d < D; ++d) {
        const float g = std::tanh(dot_f32(w_gain + d * rank, gate.data(), rank));
        const float f = std::tanh(dot_f32(w_forget + d * rank, gate.data(), rank));
        out[d] = (1.0f - f) * cur[d] + g * msg[d];
    }
}

void shortconv4_step_f32(
    const float* cur,
    const float* static_w,
    const float* dynamic_taps,
    size_t C,
    size_t n_heads,
    float* history,
    float* out) noexcept {
    // history layout: [3, C] where row 0 = x[t-3], row 1 = x[t-2], row 2 = x[t-1]
    // static_w layout: [C, 4] where index 0 = lag 3 (t-3), ..., index 3 = lag 0 (t)
    const float* h0 = history;          // t-3
    const float* h1 = history + C;      // t-2
    const float* h2 = history + 2 * C;  // t-1
    const size_t head_dim = (dynamic_taps && n_heads > 0) ? (C / n_heads) : C;
    const float scale = dynamic_taps ? (1.0f / 3.0f) : (1.0f / std::sqrt(5.0f));

    for (size_t c = 0; c < C; ++c) {
        const float* sw = static_w + c * 4;
        float acc = h0[c] * sw[0] + h1[c] * sw[1] + h2[c] * sw[2] + cur[c] * sw[3];
        if (dynamic_taps) {
            const size_t hd = c / head_dim;
            const float* dt = dynamic_taps + hd * 4;  // lag 0=t, 1=t-1, 2=t-2, 3=t-3
            acc += cur[c] * dt[0] + h2[c] * dt[1] + h1[c] * dt[2] + h0[c] * dt[3];
        }
        out[c] = (cur[c] + acc) * scale;
    }
    std::memmove(history, history + C, 2 * C * sizeof(float));
    std::memcpy(history + 2 * C, cur, C * sizeof(float));
}

void eval_inkling2_expert(
    const uint8_t* body,
    SkCode gu_code,
    SkCode dn_code,
    uint32_t dn_offset,
    const float* x_in,
    size_t hidden,
    size_t inter,
    float* y_out) noexcept {
    eval_inkling2_expert_batch(body, gu_code, dn_code, dn_offset, x_in, 1, hidden, inter, y_out);
}

void eval_inkling2_expert_batch(
    const uint8_t* body,
    SkCode gu_code,
    SkCode dn_code,
    uint32_t dn_offset,
    const float* x_in,
    size_t m,
    size_t hidden,
    size_t inter,
    float* y_out,
    void (*wait_down)(void*),
    void* wait_ctx) noexcept {
    if (m == 0) return;
    std::vector<float> gu(m * 2 * inter);
    std::vector<float> act(m * inter);

    // 1. w13 projection [2*inter, hidden] (interleaved: even rows = gate, odd rows = up)
    if (gu_code == SkCode::GGUF_RAW) {
        GgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        int be = 0, bb = 0;
        strata::block_geometry(hdr.gu_type, be, bb);
        const size_t row_b = (be > 0) ? (hidden / static_cast<size_t>(be)) * static_cast<size_t>(bb) : 0;
        const uint8_t* w_base = body + sizeof(GgufExpertHeader);
        const bool interleaved = (hdr.flags & 1u) != 0;

        const auto qt = static_cast<strata::quant::Type>(hdr.gu_type);
        const auto* ti = strata::quant::type_info(qt);
        const auto arith = strata::quant::default_arith();
        const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
        const auto* dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
        const auto* ak = (dk && ti && ti->act != qt) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
        const size_t act_rb = ak ? strata::quant::row_bytes(static_cast<int>(ti->act), static_cast<int64_t>(hidden)) : 0;

        if (dk && ak && act_rb > 0 && row_b > 0) {
            std::vector<uint8_t> qa(m * act_rb);
            for (size_t t = 0; t < m; ++t) {
                ak->fn(x_in + t * hidden, qa.data() + t * act_rb, static_cast<int64_t>(hidden));
            }
            if (interleaved) {
                size_t r = 0;
                if (dk4) {
                    for (; r + 4 <= 2 * inter; r += 4) {
                        const uint8_t* wr = w_base + r * row_b;
                        for (size_t t = 0; t < m; ++t) {
                            float s4[4];
                            dk4->fn(static_cast<int>(hidden), s4, wr, row_b, qa.data() + t * act_rb);
                            float* dst = gu.data() + t * (2 * inter) + r;
                            dst[0] = s4[0] * hdr.gate_scale;
                            dst[1] = s4[1] * hdr.up_scale;
                            dst[2] = s4[2] * hdr.gate_scale;
                            dst[3] = s4[3] * hdr.up_scale;
                        }
                    }
                }
                for (; r < 2 * inter; ++r) {
                    const uint8_t* wr = w_base + r * row_b;
                    const float sc = ((r & 1u) == 0) ? hdr.gate_scale : hdr.up_scale;
                    for (size_t t = 0; t < m; ++t) {
                        float s = 0.0f;
                        dk->fn(static_cast<int>(hidden), &s, wr, qa.data() + t * act_rb);
                        gu[t * (2 * inter) + r] = s * sc;
                    }
                }
            } else {
                // Split layout: rows [0..inter-1] = gate, rows [inter..2*inter-1] = up
                size_t j = 0;
                if (dk4) {
                    for (; j + 4 <= inter; j += 4) {
                        const uint8_t* wg = w_base + j * row_b;
                        const uint8_t* wu = w_base + (inter + j) * row_b;
                        for (size_t t = 0; t < m; ++t) {
                            float g4[4], u4[4];
                            const void* ap = qa.data() + t * act_rb;
                            dk4->fn(static_cast<int>(hidden), g4, wg, row_b, ap);
                            dk4->fn(static_cast<int>(hidden), u4, wu, row_b, ap);
                            float* dst = gu.data() + t * (2 * inter) + 2 * j;
                            for (int k = 0; k < 4; ++k) {
                                dst[2 * k + 0] = g4[k] * hdr.gate_scale;
                                dst[2 * k + 1] = u4[k] * hdr.up_scale;
                            }
                        }
                    }
                }
                for (; j < inter; ++j) {
                    const uint8_t* wg = w_base + j * row_b;
                    const uint8_t* wu = w_base + (inter + j) * row_b;
                    for (size_t t = 0; t < m; ++t) {
                        float gs = 0.0f, us = 0.0f;
                        const void* ap = qa.data() + t * act_rb;
                        dk->fn(static_cast<int>(hidden), &gs, wg, ap);
                        dk->fn(static_cast<int>(hidden), &us, wu, ap);
                        gu[t * (2 * inter) + 2 * j + 0] = gs * hdr.gate_scale;
                        gu[t * (2 * inter) + 2 * j + 1] = us * hdr.up_scale;
                    }
                }
            }
        } else {
            std::vector<float> wrow(hidden);
            auto deq_row = [&](const uint8_t* src) {
                if (hdr.gu_type == 0) {
                    std::memcpy(wrow.data(), src, hidden * 4);
                } else if (hdr.gu_type == 1) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t c = 0; c < hidden; ++c) wrow[c] = fp16_bits_to_f32(p[c]);
                } else if (hdr.gu_type == 30) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t c = 0; c < hidden; ++c) {
                        uint32_t u = static_cast<uint32_t>(p[c]) << 16;
                        std::memcpy(&wrow[c], &u, 4);
                    }
                } else {
                    strata::quant::dequant_row(qt, src, wrow.data(), static_cast<int64_t>(hidden));
                }
            };
            if (interleaved) {
                for (size_t r = 0; r < 2 * inter; ++r) {
                    deq_row(w_base + r * row_b);
                    const float sc = ((r & 1u) == 0) ? hdr.gate_scale : hdr.up_scale;
                    for (size_t t = 0; t < m; ++t) {
                        gu[t * (2 * inter) + r] = dot_f32(wrow.data(), x_in + t * hidden, hidden) * sc;
                    }
                }
            } else {
                for (size_t j = 0; j < inter; ++j) {
                    deq_row(w_base + j * row_b);
                    for (size_t t = 0; t < m; ++t) {
                        gu[t * (2 * inter) + 2 * j + 0] = dot_f32(wrow.data(), x_in + t * hidden, hidden) * hdr.gate_scale;
                    }
                }
                for (size_t j = 0; j < inter; ++j) {
                    deq_row(w_base + (inter + j) * row_b);
                    for (size_t t = 0; t < m; ++t) {
                        gu[t * (2 * inter) + 2 * j + 1] = dot_f32(wrow.data(), x_in + t * hidden, hidden) * hdr.up_scale;
                    }
                }
            }
        }
    } else if (gu_code == SkCode::SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w13_outer = hdr[0] / hdr[1];
        const size_t w13_p_bytes = (2 * inter) * (hidden / 2);
        const uint8_t* w13_p = body + 16;
        const uint8_t* w13_s = w13_p + w13_p_bytes;
        if (m == 1) {
            for (size_t r = 0; r < 2 * inter; ++r) {
                gu[r] = dot_sf4_f32_row(w13_p + r * (hidden / 2), w13_s + r * (hidden / 16), w13_outer, x_in, hidden);
            }
        } else {
            // Decode each row once into L1 and dot across all m tokens!
            std::vector<float> wrow(hidden);
            for (size_t r = 0; r < 2 * inter; ++r) {
                dequant_sf4_row(w13_p + r * (hidden / 2), w13_s + r * (hidden / 16), w13_outer, wrow.data(), hidden);
                for (size_t t = 0; t < m; ++t) {
                    gu[t * (2 * inter) + r] = dot_f32(wrow.data(), x_in + t * hidden, hidden);
                }
            }
        }
    } else if (gu_code == SkCode::MX8) {
        const size_t w13_w_bytes = (2 * inter) * hidden;
        const uint8_t* w13_w = body;
        const uint8_t* w13_s = w13_w + w13_w_bytes;
        if (m == 1) {
            for (size_t r = 0; r < 2 * inter; ++r) {
                gu[r] = dot_mx8_f32_row(w13_w + r * hidden, w13_s + r * (hidden / 32), x_in, hidden);
            }
        } else {
            std::vector<float> wrow(hidden);
            for (size_t r = 0; r < 2 * inter; ++r) {
                dequant_mx8_row(w13_w + r * hidden, w13_s + r * (hidden / 32), wrow.data(), hidden);
                for (size_t t = 0; t < m; ++t) {
                    gu[t * (2 * inter) + r] = dot_f32(wrow.data(), x_in + t * hidden, hidden);
                }
            }
        }
    } else {
        const size_t sq_xd_bytes = inter * 2;
        const uint8_t* gu_ptr = body + sq_xd_bytes;
        const size_t x_stride = hidden / SUB_G;
        std::vector<BlockQ8_1_G64> qx(m * x_stride);
        for (size_t t = 0; t < m; ++t) {
            quantize_q8_1_g64(x_in + t * hidden, qx.data() + t * x_stride, hidden);
        }
        gemm_sk_q8_1(gu_ptr, gu_code, qx.data(), gu.data(), 2 * inter, hidden, m);
    }

    // 2. SITU(beta=4.0, lb=25.0) on interleaved (even=gate, odd=up)
    for (size_t t = 0; t < m; ++t) {
        const float* gt = gu.data() + t * (2 * inter);
        float* at = act.data() + t * inter;
        for (size_t j = 0; j < inter; ++j) {
            at[j] = situ_scalar(gt[2 * j], gt[2 * j + 1], kSituBeta, kSituLb);
        }
    }

    // 3. w2 projection [hidden, inter] (wait for down projection bytes if overlapped with NVMe read!)
    if (wait_down) wait_down(wait_ctx);
    if (dn_code == SkCode::GGUF_RAW) {
        GgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        int be = 0, bb = 0;
        strata::block_geometry(hdr.dn_type, be, bb);
        const size_t row_b = (be > 0) ? (inter / static_cast<size_t>(be)) * static_cast<size_t>(bb) : 0;
        const uint8_t* w2_base = body + dn_offset;

        const auto qt = static_cast<strata::quant::Type>(hdr.dn_type);
        const auto* ti = strata::quant::type_info(qt);
        const auto arith = strata::quant::default_arith();
        const auto* dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
        const auto* dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
        const auto* ak = (dk && ti && ti->act != qt) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
        const size_t act_rb = ak ? strata::quant::row_bytes(static_cast<int>(ti->act), static_cast<int64_t>(inter)) : 0;

        if (dk && ak && act_rb > 0 && row_b > 0) {
            std::vector<uint8_t> qa(m * act_rb);
            for (size_t t = 0; t < m; ++t) {
                ak->fn(act.data() + t * inter, qa.data() + t * act_rb, static_cast<int64_t>(inter));
            }
            size_t r = 0;
            if (dk4) {
                for (; r + 4 <= hidden; r += 4) {
                    const uint8_t* wr = w2_base + r * row_b;
                    for (size_t t = 0; t < m; ++t) {
                        float s4[4];
                        dk4->fn(static_cast<int>(inter), s4, wr, row_b, qa.data() + t * act_rb);
                        float* dst = y_out + t * hidden + r;
                        dst[0] = s4[0] * hdr.dn_scale;
                        dst[1] = s4[1] * hdr.dn_scale;
                        dst[2] = s4[2] * hdr.dn_scale;
                        dst[3] = s4[3] * hdr.dn_scale;
                    }
                }
            }
            for (; r < hidden; ++r) {
                const uint8_t* wr = w2_base + r * row_b;
                for (size_t t = 0; t < m; ++t) {
                    float s = 0.0f;
                    dk->fn(static_cast<int>(inter), &s, wr, qa.data() + t * act_rb);
                    y_out[t * hidden + r] = s * hdr.dn_scale;
                }
            }
        } else {
            std::vector<float> wrow(inter);
            for (size_t r = 0; r < hidden; ++r) {
                const uint8_t* src = w2_base + r * row_b;
                if (hdr.dn_type == 0) {
                    std::memcpy(wrow.data(), src, inter * 4);
                } else if (hdr.dn_type == 1) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t c = 0; c < inter; ++c) wrow[c] = fp16_bits_to_f32(p[c]);
                } else if (hdr.dn_type == 30) {
                    const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
                    for (size_t c = 0; c < inter; ++c) {
                        uint32_t u = static_cast<uint32_t>(p[c]) << 16;
                        std::memcpy(&wrow[c], &u, 4);
                    }
                } else {
                    strata::quant::dequant_row(qt, src, wrow.data(), static_cast<int64_t>(inter));
                }
                for (size_t t = 0; t < m; ++t) {
                    y_out[t * hidden + r] = dot_f32(wrow.data(), act.data() + t * inter, inter) * hdr.dn_scale;
                }
            }
        }
    } else if (dn_code == SkCode::SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w2_outer = hdr[2] / hdr[3];
        const size_t w2_p_bytes = hidden * (inter / 2);
        const uint8_t* w2_p = body + dn_offset;
        const uint8_t* w2_s = w2_p + w2_p_bytes;
        if (m == 1) {
            for (size_t r = 0; r < hidden; ++r) {
                y_out[r] = dot_sf4_f32_row(w2_p + r * (inter / 2), w2_s + r * (inter / 16), w2_outer, act.data(), inter);
            }
        } else {
            std::vector<float> wrow(inter);
            for (size_t r = 0; r < hidden; ++r) {
                dequant_sf4_row(w2_p + r * (inter / 2), w2_s + r * (inter / 16), w2_outer, wrow.data(), inter);
                for (size_t t = 0; t < m; ++t) {
                    y_out[t * hidden + r] = dot_f32(wrow.data(), act.data() + t * inter, inter);
                }
            }
        }
    } else if (dn_code == SkCode::MX8) {
        const size_t w2_w_bytes = hidden * inter;
        const uint8_t* w2_w = body + dn_offset;
        const uint8_t* w2_s = w2_w + w2_w_bytes;
        if (m == 1) {
            for (size_t r = 0; r < hidden; ++r) {
                y_out[r] = dot_mx8_f32_row(w2_w + r * inter, w2_s + r * (inter / 32), act.data(), inter);
            }
        } else {
            std::vector<float> wrow(inter);
            for (size_t r = 0; r < hidden; ++r) {
                dequant_mx8_row(w2_w + r * inter, w2_s + r * (inter / 32), wrow.data(), inter);
                for (size_t t = 0; t < m; ++t) {
                    y_out[t * hidden + r] = dot_f32(wrow.data(), act.data() + t * inter, inter);
                }
            }
        }
    } else {
        // Apply inverse MACaQ scale sq_xd_dn (FP16[inter] at body[0..2*inter]) before paired 2-row S_K GEMM
        std::vector<float> inv_s(inter);
        for (size_t j = 0; j < inter; ++j) {
            uint16_t bits;
            std::memcpy(&bits, body + 2 * j, 2);
            inv_s[j] = 1.0f / fp16_bits_to_f32(bits);
        }
        const size_t a_stride = inter / SUB_G;
        std::vector<BlockQ8_1_G64> qact(m * a_stride);
        for (size_t t = 0; t < m; ++t) {
            float* at = act.data() + t * inter;
            for (size_t j = 0; j < inter; ++j) at[j] *= inv_s[j];
            quantize_q8_1_g64(at, qact.data() + t * a_stride, inter);
        }
        gemm_paired2_sk_q8_1(body + dn_offset, dn_code, qact.data(), y_out, hidden, inter, m);
    }
}

namespace {

// The GGUF_RAW kernels eval_inkling2_expert_batch picks for one tensor type (quantized dot with an activation quantizer).
struct GgufDot {
    const strata::quant::DotKernel* dk = nullptr;
    const strata::quant::DotX4Kernel* dk4 = nullptr;
    const strata::quant::ActKernel* ak = nullptr;
    size_t row_b = 0, act_rb = 0;
    bool ok = false;
};

GgufDot gguf_dot_for(uint32_t type, size_t cols) noexcept {
    GgufDot g;
    int be = 0, bb = 0;
    strata::block_geometry(type, be, bb);
    g.row_b = (be > 0) ? (cols / static_cast<size_t>(be)) * static_cast<size_t>(bb) : 0;
    const auto qt = static_cast<strata::quant::Type>(type);
    const auto* ti = strata::quant::type_info(qt);
    const auto arith = strata::quant::default_arith();
    g.dk = (ti && ti->has_dot) ? strata::quant::dot_kernel(qt, arith) : nullptr;
    g.dk4 = (ti && ti->has_dot) ? strata::quant::dot_x4_kernel(qt, arith) : nullptr;
    g.ak = (g.dk && ti && ti->act != qt) ? strata::quant::act_kernel(ti->act, arith) : nullptr;
    g.act_rb = g.ak ? strata::quant::row_bytes(static_cast<int>(ti->act), static_cast<int64_t>(cols)) : 0;
    g.ok = g.dk && g.ak && g.act_rb > 0 && g.row_b > 0;
    return g;
}

// One weight row -> floats, as the dequantize fallback of eval_inkling2_expert_batch does it.
void gguf_deq_row(uint32_t type, const uint8_t* src, float* wrow, size_t n) noexcept {
    if (type == 0) {
        std::memcpy(wrow, src, n * 4);
    } else if (type == 1) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t c = 0; c < n; ++c) wrow[c] = fp16_bits_to_f32(p[c]);
    } else if (type == 30) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
        for (size_t c = 0; c < n; ++c) {
            uint32_t u = static_cast<uint32_t>(p[c]) << 16;
            std::memcpy(&wrow[c], &u, 4);
        }
    } else {
        strata::quant::dequant_row(static_cast<strata::quant::Type>(type), src, wrow, static_cast<int64_t>(n));
    }
}

}  // namespace

bool expert_m1_splittable(SkCode gu_code, SkCode dn_code, const uint8_t* body, size_t hidden, size_t inter) noexcept {
    if (gu_code != SkCode::GGUF_RAW && dn_code != SkCode::GGUF_RAW) return true;
    if (gu_code != dn_code || !body) return false;
    // GGUF_RAW: the quantized-dot fast path and the dequantize fallback (Q8_0, F16, ...) are both split, as long as
    // the tensor type has a known block geometry
    GgufExpertHeader hdr{};
    std::memcpy(&hdr, body, sizeof(hdr));
    return gguf_dot_for(hdr.gu_type, hidden).row_b > 0 && gguf_dot_for(hdr.dn_type, inter).row_b > 0;
}

void expert_m1_gu_rows(const uint8_t* body, SkCode gu_code, const float* x_in, size_t hidden, size_t inter,
                       size_t r0, size_t r1, float* gu) noexcept {
    if (r1 <= r0) return;
    if (gu_code == SkCode::GGUF_RAW) {
        // same kernels and the same 4-row grouping as the whole-expert function (r0, r1 are multiples of 4)
        GgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        const GgufDot g = gguf_dot_for(hdr.gu_type, hidden);
        const uint8_t* w_base = body + sizeof(GgufExpertHeader);
        if (!g.ok) {   // dequantize fallback (Q8_0 ...): one decoded row at a time, dot in float
            static thread_local std::vector<float> wrow;
            wrow.resize(hidden);
            if ((hdr.flags & 1u) != 0) {
                for (size_t r = r0; r < r1; ++r) {
                    gguf_deq_row(hdr.gu_type, w_base + r * g.row_b, wrow.data(), hidden);
                    gu[r] = dot_f32(wrow.data(), x_in, hidden) * (((r & 1u) == 0) ? hdr.gate_scale : hdr.up_scale);
                }
            } else {
                for (size_t j = r0 / 2; j < r1 / 2; ++j) {
                    gguf_deq_row(hdr.gu_type, w_base + j * g.row_b, wrow.data(), hidden);
                    gu[2 * j + 0] = dot_f32(wrow.data(), x_in, hidden) * hdr.gate_scale;
                    gguf_deq_row(hdr.gu_type, w_base + (inter + j) * g.row_b, wrow.data(), hidden);
                    gu[2 * j + 1] = dot_f32(wrow.data(), x_in, hidden) * hdr.up_scale;
                }
            }
            return;
        }
        static thread_local std::vector<uint8_t> qa;
        qa.resize(g.act_rb);
        g.ak->fn(x_in, qa.data(), static_cast<int64_t>(hidden));
        if ((hdr.flags & 1u) != 0) {
            size_t r = r0;
            if (g.dk4) {
                for (; r + 4 <= r1; r += 4) {
                    float s4[4];
                    g.dk4->fn(static_cast<int>(hidden), s4, w_base + r * g.row_b, g.row_b, qa.data());
                    gu[r + 0] = s4[0] * hdr.gate_scale;
                    gu[r + 1] = s4[1] * hdr.up_scale;
                    gu[r + 2] = s4[2] * hdr.gate_scale;
                    gu[r + 3] = s4[3] * hdr.up_scale;
                }
            }
            for (; r < r1; ++r) {
                float s = 0.0f;
                g.dk->fn(static_cast<int>(hidden), &s, w_base + r * g.row_b, qa.data());
                gu[r] = s * (((r & 1u) == 0) ? hdr.gate_scale : hdr.up_scale);
            }
        } else {
            // split layout: gate rows [0, inter), up rows [inter, 2*inter); the output is interleaved (2j gate, 2j+1 up)
            size_t j = r0 / 2;
            const size_t j1 = r1 / 2;
            if (g.dk4) {
                for (; j + 4 <= j1; j += 4) {
                    float g4[4], u4[4];
                    g.dk4->fn(static_cast<int>(hidden), g4, w_base + j * g.row_b, g.row_b, qa.data());
                    g.dk4->fn(static_cast<int>(hidden), u4, w_base + (inter + j) * g.row_b, g.row_b, qa.data());
                    float* dst = gu + 2 * j;
                    for (int k = 0; k < 4; ++k) {
                        dst[2 * k + 0] = g4[k] * hdr.gate_scale;
                        dst[2 * k + 1] = u4[k] * hdr.up_scale;
                    }
                }
            }
            for (; j < j1; ++j) {
                float gs = 0.0f, us = 0.0f;
                g.dk->fn(static_cast<int>(hidden), &gs, w_base + j * g.row_b, qa.data());
                g.dk->fn(static_cast<int>(hidden), &us, w_base + (inter + j) * g.row_b, qa.data());
                gu[2 * j + 0] = gs * hdr.gate_scale;
                gu[2 * j + 1] = us * hdr.up_scale;
            }
        }
        return;
    }
    if (gu_code == SkCode::SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w13_outer = hdr[0] / hdr[1];
        const size_t w13_p_bytes = (2 * inter) * (hidden / 2);
        const uint8_t* w13_p = body + 16;
        const uint8_t* w13_s = w13_p + w13_p_bytes;
        for (size_t r = r0; r < r1; ++r) {
            gu[r] = dot_sf4_f32_row(w13_p + r * (hidden / 2), w13_s + r * (hidden / 16), w13_outer, x_in, hidden);
        }
    } else if (gu_code == SkCode::MX8) {
        const size_t w13_w_bytes = (2 * inter) * hidden;
        const uint8_t* w13_w = body;
        const uint8_t* w13_s = w13_w + w13_w_bytes;
        for (size_t r = r0; r < r1; ++r) {
            gu[r] = dot_mx8_f32_row(w13_w + r * hidden, w13_s + r * (hidden / 32), x_in, hidden);
        }
    } else {
        const uint8_t* gu_ptr = body + inter * 2;
        const size_t x_stride = hidden / SUB_G;
        static thread_local std::vector<BlockQ8_1_G64> qx;
        qx.resize(x_stride);
        quantize_q8_1_g64(x_in, qx.data(), hidden);
        const size_t row_bytes = (hidden / QK_S) * superblock_bytes(gu_code);
        gemm_sk_q8_1(gu_ptr + r0 * row_bytes, gu_code, qx.data(), gu + r0, r1 - r0, hidden, 1);
    }
}

void expert_m1_act(const float* gu, size_t j0, size_t j1, float* act) noexcept {
    for (size_t j = j0; j < j1; ++j) act[j] = situ_scalar(gu[2 * j], gu[2 * j + 1], kSituBeta, kSituLb);
}

void expert_m1_dn_rows(const uint8_t* body, SkCode dn_code, uint32_t dn_offset, const float* act,
                       size_t hidden, size_t inter, size_t r0, size_t r1, float* y) noexcept {
    if (r1 <= r0) return;
    if (dn_code == SkCode::GGUF_RAW) {
        GgufExpertHeader hdr{};
        std::memcpy(&hdr, body, sizeof(hdr));
        const GgufDot g = gguf_dot_for(hdr.dn_type, inter);
        const uint8_t* w2_base = body + dn_offset;
        if (!g.ok) {
            static thread_local std::vector<float> wrow;
            wrow.resize(inter);
            for (size_t r = r0; r < r1; ++r) {
                gguf_deq_row(hdr.dn_type, w2_base + r * g.row_b, wrow.data(), inter);
                y[r] = dot_f32(wrow.data(), act, inter) * hdr.dn_scale;
            }
            return;
        }
        static thread_local std::vector<uint8_t> qa;
        qa.resize(g.act_rb);
        g.ak->fn(act, qa.data(), static_cast<int64_t>(inter));
        size_t r = r0;
        if (g.dk4) {
            for (; r + 4 <= r1; r += 4) {
                float s4[4];
                g.dk4->fn(static_cast<int>(inter), s4, w2_base + r * g.row_b, g.row_b, qa.data());
                y[r + 0] = s4[0] * hdr.dn_scale;
                y[r + 1] = s4[1] * hdr.dn_scale;
                y[r + 2] = s4[2] * hdr.dn_scale;
                y[r + 3] = s4[3] * hdr.dn_scale;
            }
        }
        for (; r < r1; ++r) {
            float s = 0.0f;
            g.dk->fn(static_cast<int>(inter), &s, w2_base + r * g.row_b, qa.data());
            y[r] = s * hdr.dn_scale;
        }
        return;
    }
    if (dn_code == SkCode::SF4) {
        float hdr[4];
        std::memcpy(hdr, body, 16);
        const float w2_outer = hdr[2] / hdr[3];
        const size_t w2_p_bytes = hidden * (inter / 2);
        const uint8_t* w2_p = body + dn_offset;
        const uint8_t* w2_s = w2_p + w2_p_bytes;
        for (size_t r = r0; r < r1; ++r) {
            y[r] = dot_sf4_f32_row(w2_p + r * (inter / 2), w2_s + r * (inter / 16), w2_outer, act, inter);
        }
    } else if (dn_code == SkCode::MX8) {
        const size_t w2_w_bytes = hidden * inter;
        const uint8_t* w2_w = body + dn_offset;
        const uint8_t* w2_s = w2_w + w2_w_bytes;
        for (size_t r = r0; r < r1; ++r) {
            y[r] = dot_mx8_f32_row(w2_w + r * inter, w2_s + r * (inter / 32), act, inter);
        }
    } else {
        static thread_local std::vector<float> at;
        static thread_local std::vector<BlockQ8_1_G64> qact;
        at.resize(inter);
        qact.resize(inter / SUB_G);
        for (size_t j = 0; j < inter; ++j) {
            uint16_t bits;
            std::memcpy(&bits, body + 2 * j, 2);
            const float inv_s = 1.0f / fp16_bits_to_f32(bits);
            at[j] = act[j] * inv_s;
        }
        quantize_q8_1_g64(at.data(), qact.data(), inter);
        const size_t pair_bytes = ((2 * inter) / QK_S) * superblock_bytes(dn_code);
        gemv_paired2_sk_q8_1(body + dn_offset + (r0 / 2) * pair_bytes, dn_code, qact.data(), y + r0, r1 - r0, inter);
    }
}

int32_t dspark_markov_step_f32(
    int32_t prev_token,
    float* state,
    const float* hidden,
    const float* context,
    const float* base_logits,
    const float* markov_w1,
    const float* rnn_proj_w,
    const float* rnn_proj_b,
    const float* hidden_out_w,
    const float* hidden_out_b,
    const float* unembed_w,
    size_t D,
    size_t rank,
    size_t V_unpadded,
    float softcap,
    float* out_logits) noexcept {
    const size_t in_dim = 2 * rank + 2 * D;
    std::vector<float> joint_in(in_dim);
    std::memcpy(joint_in.data(), state, rank * sizeof(float));
    std::memcpy(joint_in.data() + rank, markov_w1 + static_cast<size_t>(prev_token) * rank, rank * sizeof(float));
    std::memcpy(joint_in.data() + 2 * rank, hidden, D * sizeof(float));
    std::memcpy(joint_in.data() + 2 * rank + D, context, D * sizeof(float));

    std::vector<float> proj(3 * rank);
    for (size_t r = 0; r < 3 * rank; ++r) {
        proj[r] = dot_f32(rnn_proj_w + r * in_dim, joint_in.data(), in_dim) + rnn_proj_b[r];
    }
    std::vector<float> latent(rank);
    for (size_t r = 0; r < rank; ++r) {
        const float gate = 1.0f / (1.0f + std::exp(-proj[r]));
        const float cand = std::tanh(proj[rank + r]);
        state[r] = cand + gate * (state[r] - cand);
        latent[r] = std::tanh(proj[2 * rank + r]);
    }

    std::vector<float> corr(D);
    for (size_t d = 0; d < D; ++d) {
        corr[d] = dot_f32(hidden_out_w + d * rank, latent.data(), rank) + hidden_out_b[d];
    }

    int32_t best_id = 0;
    float best_val = -1e30f;
    const float inv_cap = 1.0f / softcap;
    for (size_t v = 0; v < V_unpadded; ++v) {
        const float bias = dot_f32(unembed_w + v * D, corr.data(), D);
        const float val = softcap * std::tanh((base_logits[v] + bias) * inv_cap);
        out_logits[v] = val;
        if (val > best_val) {  // strictly greater -> deterministic first-max tie-break!
            best_val = val;
            best_id = static_cast<int32_t>(v);
        }
    }
    return best_id;
}

}  // namespace strata::inkling2
