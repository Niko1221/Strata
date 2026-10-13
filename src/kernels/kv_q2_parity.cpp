// src/kernels/kv_q2_parity.cpp - the 2-bit KV format (kv_q2.hpp) against a full-precision reference (GPU, no
// model). Design and the byte math: docs/kv-q2.md.
//
// The pools are built by the real append kernels from rotated data, then read back by the real gather kernel:
//   1. the device quantizer reproduces the host codebook bit for bit (the parity the parity test is named for -
//      the engine's writer and reader cannot drift apart);
//   2. the relative RMS of K and V against an FP16 cache, for q2 and for q4_0 side by side, which is the cost of
//      going from 4 bits to 2 stated in the same units as the window work measures it;
//   3. the attention-output error that follows from it: the same query, the same cells, softmax over the reference
//      K/V and over the quantized ones, so a number that reaches the model is reported, not just a per-element one;
//   4. both V codebooks (the 3-level {-1,0,+1} and the 4-level {-3,-1,+1,+3}) measured, so the codebook is chosen
//      from data. The 3-level form is the one SKVQ's 1.5-bit V needs; if it is too lossy, that is said here.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q2.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"   // kKvQ2K, kv_stream_unit_blocks, kv_block_bytes (the q2k byte math)
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
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
/// A directional finding is a property of the DATA as much as of the scheme. Running this gate on a real
/// cache dump (STRATA_KV_PROBE_CACHE) REVERSED six of the rankings pinned below, so each one is pinned in
/// BOTH directions with its data source named. Quoting any of these numbers without saying which cache it
/// was measured on is exactly how this table becomes a lie.
void check_data(bool real_cache, bool synthetic_holds, bool real_holds, const char* synthetic_claim,
                const char* real_claim) {
    if (real_cache) check(real_holds, real_claim);
    else check(synthetic_holds, synthetic_claim);
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

/// A REAL KV cache replaces the synthetic one when STRATA_KV_PROBE_CACHE names a file - the SDKV dump
/// `--dump-kv` writes (header: magic, version, n_layers, n_cells, n_head_kv, head_dim, layer ids, then per
/// layer K then V as fp16 in (cell, head, dim) order). That element order is the order these arrays use, so
/// a real cache drops straight in, and it is the STORED cache - rotated, post-RoPE, post-norm - which is
/// exactly what the quantizers are being asked about. STRATA_KV_PROBE_LAYER picks the layer.
///
/// This exists because the synthetic cache is structurally blind to per-channel scaling: i.i.d. Gaussian
/// noise gives every channel the same RMS, and KIVI's per-channel result (arXiv 2402.02750) exists to
/// exploit a handful of channels being far larger than the rest. Every per-channel number measured on the
/// synthetic cache is a floor, not the number.
bool load_real_cache(const char* path, int want_layer, int H, int D, int max_cells, int& n_out,
                     std::vector<float>& K, std::vector<float>& V, int& layer_out, std::string& err) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { err = std::string("probe cache: cannot open ") + path; return false; }
    int hdr[6] = {0, 0, 0, 0, 0, 0};
    if (std::fread(hdr, 4, 6, f) != 6 || hdr[0] != 0x564B4453 || hdr[1] != 1) {
        std::fclose(f);
        err = "probe cache: not an SDKV v1 dump (write one with --dump-kv)";
        return false;
    }
    const int n_layers = hdr[2], n_cells = hdr[3], n_head_kv = hdr[4], head_dim = hdr[5];
    if (n_head_kv != H || head_dim != D) {
        std::fclose(f);
        err = "probe cache: " + std::to_string(n_head_kv) + " heads x " + std::to_string(head_dim) +
              " dims does not match this model's " + std::to_string(H) + " x " + std::to_string(D);
        return false;
    }
    std::vector<int> ids(n_layers, -1);
    if (n_layers <= 0 || n_cells <= 0 || std::fread(ids.data(), 4, n_layers, f) != (size_t) n_layers) {
        std::fclose(f);
        err = "probe cache: short layer id table";
        return false;
    }
    int slot = -1;
    for (int i = 0; i < n_layers; ++i)
        if (ids[i] == want_layer) slot = i;
    if (slot < 0) {
        std::fclose(f);
        err = "probe cache: layer " + std::to_string(want_layer) + " is not in the dump";
        return false;
    }
    const size_t plane = (size_t) n_cells * (size_t) H * (size_t) D;
    std::vector<uint16_t> buf(plane);
    for (int i = 0; i < slot; ++i)                    // skip the layers before the one we want
        if (std::fread(buf.data(), 2, plane, f) != plane || std::fread(buf.data(), 2, plane, f) != plane) {
            std::fclose(f);
            err = "probe cache: file is shorter than its header claims";
            return false;
        }
    std::vector<float> k((size_t) n_cells * H * D), v((size_t) n_cells * H * D);
    if (std::fread(buf.data(), 2, plane, f) != plane) { std::fclose(f); err = "probe cache: short K plane"; return false; }
    for (size_t i = 0; i < plane; ++i) k[i] = k::f32_from_f16(buf[i]);
    if (std::fread(buf.data(), 2, plane, f) != plane) { std::fclose(f); err = "probe cache: short V plane"; return false; }
    std::fclose(f);
    for (size_t i = 0; i < plane; ++i) v[i] = k::f32_from_f16(buf[i]);
    for (size_t i = 0; i < plane; ++i)
        if (!std::isfinite(k[i]) || !std::isfinite(v[i])) {
            err = "probe cache: a non-finite value in the dump";
            return false;
        }
    // Whole 32-cell groups only: the group is the per-channel scale's unit, so a tail that does not fill one
    // would be quantized half a scale at a time.
    const int usable = (n_cells / 32) * 32;
    n_out = usable < max_cells ? usable : max_cells;
    K.assign(k.begin(), k.begin() + (size_t) n_out * H * D);
    V.assign(v.begin(), v.begin() + (size_t) n_out * H * D);
    layer_out = want_layer;
    return true;
}

/// The statistics the synthetic cache cannot have, printed so the reader of the table knows what the data is.
/// Per-channel scaling exists to exploit the first ratio (a channel's RMS varying across the cache) and the
/// second (a few channels carrying most of the variance); if both are flat, per-channel scaling has nothing
/// to exploit and the per-token scale is the honest choice.
void channel_stats(const std::vector<float>& X, int n_cells, int H, int D, const char* name) {
    const int nd = H * D;
    std::vector<double> chan((size_t) nd, 0.0), cell((size_t) n_cells, 0.0);
    for (int c = 0; c < n_cells; ++c)
        for (int i = 0; i < nd; ++i) {
            const double x = X[(size_t) c * nd + i];
            chan[i] += x * x;
            cell[c] += x * x;
        }
    std::vector<double> cv((size_t) nd);
    double total = 0.0;
    for (int i = 0; i < nd; ++i) { chan[i] = std::sqrt(chan[i] / n_cells); cv[i] = chan[i] * chan[i]; total += cv[i]; }
    for (int c = 0; c < n_cells; ++c) cell[c] = std::sqrt(cell[c] / nd);
    std::sort(cv.begin(), cv.end());
    std::sort(cell.begin(), cell.end());
    double top8 = 0.0;
    for (size_t i = cv.size() - 8; i < cv.size(); ++i) top8 += cv[i];
    const double cmed = cv[cv.size() / 2], cmax = cv.back(), cmin = cv.front();
    const double lmed = cell[cell.size() / 2], lmax = cell.back(), lmin = cell.front();
    std::printf("  %s: channel RMS min %.4f median %.4f max %.4f (max/median %.2fx), top-8 channels carry %.1f%% "
                "of the variance; cell RMS min %.4f median %.4f max %.4f (max/median %.2fx)\n",
                name, cmin, cmed, cmax, cmed > 0.0 ? cmax / cmed : 0.0, total > 0.0 ? 100.0 * top8 / total : 0.0,
                lmin, lmed, lmax, lmed > 0.0 ? lmax / lmed : 0.0);
}

/// The host reference quantizer: the same warp rule as the kernel, run serially. d is the block's scale from the
/// largest |x| in the block; ties resolve to the larger value, as they do in every lane on the device.
void quant_block(const float* x, int n, bool three_level, float* out) {
    float amax = 0.f, mval = 0.f;
    for (int i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax || (a == amax && x[i] > mval)) { amax = a; mval = x[i]; }
    }
    const float d = three_level ? amax : amax / 3.0f;
    // Codes are chosen with the float scale, exactly as the kernel does, but the value the kernel writes back is
    // built from the scale as STORED: d lives in the block as fp16, so the host has to round it the same way
    // before multiplying, or the two disagree on every block whose scale is not exactly representable.
    const float ds = k::f32_from_f16(k::f16_from_f32(d));
    for (int i = 0; i < n; ++i) {
        const int c = three_level ? k::q2_code_3level(x[i], d) : k::q2_code_4level(x[i], d);
        out[i] = three_level ? k::q2_value_3level(c, ds) : k::q2_value_4level(c, ds);
    }
}

/// Lloyd-Max levels for a Gaussian source: the codebook SKVQ's gain rests on. Scaled by the block's RMS rather
/// than its max, so the inner level sits at 0.46 sigma where the mass is, not at 0.9 sigma where amax/3 puts it.
void quant_lm(const float* x, int n, bool three_level, float* out) {
    // The warp sums x*x in a xor-butterfly (offsets 16,8,4,2,1), which every lane finishes with the same value
    // but in a fixed addition order. Summing in double, or left to right, gives an RMS one ulp away, and an ulp
    // in the scale is enough to flip a code at a threshold - so the host replays the butterfly.
    float ss[k::Q2_0], prev[k::Q2_0];
    for (int i = 0; i < n; ++i) ss[i] = x[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) {
        for (int i = 0; i < n; ++i) prev[i] = ss[i];
        for (int i = 0; i < n; ++i) ss[i] = prev[i] + prev[i ^ o];
    }
    const float s = sqrtf(ss[0] * (1.0f / (float) k::Q2_0));
    const float ds = k::f32_from_f16(k::f16_from_f32(s));               // the scale as the block stores it
    for (int i = 0; i < n; ++i)
        out[i] = three_level ? k::q2_value_lm3(k::q2_code_lm3(x[i], s), ds)
                             : k::q2_value_lm4(k::q2_code_lm4(x[i], s), ds);
}

/// The same 4-level Lloyd-Max codebook with a scale every 16 values instead of every 32: 12 B per 32 values
/// (two fp16 scales + the same 8 code bytes) against 10 B, bought for a scale that tracks the data twice as
/// closely.  Host-side only - this is the measurement that decides whether the layout changes.
void quant_lm16(const float* x, int n, float* out) {
    for (int b = 0; b < n; b += 16) {
        const float* p = x + b;
        float ss[16], prev[16];
        for (int i = 0; i < 16; ++i) ss[i] = p[i] * p[i];
        for (int o = 8; o > 0; o >>= 1) {                       // the same butterfly the device runs on 16 lanes
            for (int i = 0; i < 16; ++i) prev[i] = ss[i];
            for (int i = 0; i < 16; ++i) ss[i] = prev[i] + prev[i ^ o];
        }
        const float s = sqrtf(ss[0] * (1.0f / 16.0f));
        const float ds = k::f32_from_f16(k::f16_from_f32(s));   // the scale as the block would store it
        for (int i = 0; i < 16; ++i) out[b + i] = k::q2_value_lm4(k::q2_code_lm4(p[i], s), ds);
    }
}

/// KIVI-style per-channel scaling (arXiv 2402.02750): one scale per (dimension, 32-cell group) instead of one per
/// (cell, 32-value block).  It costs the same 80 B per cell and head - 256 dims x 2 bits = 64 B of codes, plus
/// 256 dims x 2 B of scale per 32 cells = 16 B - so this is a like-for-like swap of WHAT the scale measures, not
/// of how many bits are spent.
///
/// THIS IS THE REFERENCE THE DEVICE HAS TO MATCH BIT FOR BIT.  kv_q2.cu's group kernel sums `ss += v*v` over the
/// group's 32 cells in cell order and takes `sqrtf(ss / 32)`; this sums the same 32 numbers in the same order and
/// takes the same square root, then rounds the scale to fp16 and quantizes with the full-precision scale while
/// writing back with the stored one.  Every one of those choices is the kernel's, because an ulp in the scale
/// flips a code at a threshold.
///
/// Caveat the reader must keep: this gate's cache is i.i.d. Gaussian, so every channel's RMS is the same number;
/// per-channel scaling cannot show its gain here beyond the 2.8x on the attention output that it does show.  The
/// structure KIVI's gain rests on - outliers aligned to dimensions - has to come from a real cache.
void quant_kivi(const float* x, int n_cells, int n_heads, int D, bool three_level, float* out) {
    const int stride = n_heads * D;
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d)
            for (int g = 0; g < n_cells; g += 32) {
                float ss = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g + i) * stride + h * D + d];
                    ss += v * v;
                }
                const float s = sqrtf(ss * (1.0f / 32.0f));
                const float ds = k::f32_from_f16(k::f16_from_f32(s));
                for (int i = 0; i < 32; ++i) {
                    const size_t idx = (size_t) (g + i) * stride + h * D + d;
                    out[idx] = three_level ? k::q2_value_lm3(k::q2_code_lm3(x[idx], s), ds)
                                           : k::q2_value_lm4(k::q2_code_lm4(x[idx], s), ds);
                }
            }
}

/// 1-bit (two-level) quantization: the levels are the Lloyd-Max optimal pair for a Gaussian, +-sqrt(2/pi)*sigma,
/// and the code is the sign.  `per_channel` puts the scale on a (dimension, 32-cell) group like quant_kivi;
/// `per_token` puts one fp16 scale on the whole head_dim of a cell.  KIVI's V arm is the per-token one, and the
/// point of the arm is that V is quantized differently from K, not harder than K.
void quant_1bit(const float* x, int n_cells, int n_heads, int D, bool per_channel, float* out) {
    const float lm2 = 0.7978845608028654f;                       // sqrt(2/pi)
    const int stride = n_heads * D;
    if (per_channel) {
        for (int h = 0; h < n_heads; ++h)
            for (int d = 0; d < D; ++d)
                for (int g = 0; g < n_cells; g += 32) {
                    float ss = 0.0f;
                    for (int i = 0; i < 32; ++i) {
                        const float v = x[(size_t) (g + i) * stride + h * D + d];
                        ss += v * v;
                    }
                    const float ds = k::f32_from_f16(k::f16_from_f32(sqrtf(ss * (1.0f / 32.0f))));
                    for (int i = 0; i < 32; ++i) {
                        const size_t idx = (size_t) (g + i) * stride + h * D + d;
                        out[idx] = x[idx] >= 0.0f ? lm2 * ds : -lm2 * ds;
                    }
                }
    } else {
        for (int c = 0; c < n_cells; ++c)
            for (int h = 0; h < n_heads; ++h) {
                const size_t base = (size_t) c * stride + h * D;
                float ss = 0.0f;
                for (int i = 0; i < D; ++i) ss += x[base + i] * x[base + i];
                const float ds = k::f32_from_f16(k::f16_from_f32(sqrtf(ss * (1.0f / (float) D))));
                for (int i = 0; i < D; ++i) out[base + i] = x[base + i] >= 0.0f ? lm2 * ds : -lm2 * ds;
            }
    }
}

/// KIVI-style ASYMMETRIC quantization (arXiv 2402.02750): scale AND A ZERO-POINT from the group's extremes,
/// `z = min`, `s = (max - min) / 3`, levels z, z+s, z+2s, z+3s. SKVQ (arXiv 2405.06219) clips those extremes
/// first; `clip_sigma > 0` does the same (mean +- clip_sigma * sd). `per_channel` groups 32 cells of one
/// dimension, `per_token` groups one cell's head_dim.
///
/// This is the scheme both published 2-bit methods use and ours does not: we store a symmetric RMS scale with
/// no zero-point. It costs two fp16 per group (scale + zero-point) - 32 B per cell and head per-channel,
/// 4 B per cell and head per-token - so 96 B per cell and head for a per-channel 2-bit K, against 80 B for
/// the symmetric block.
void quant_asym(const float* x, int n_cells, int n_heads, int D, bool per_channel, float clip_sigma, float* out) {
    const int stride = n_heads * D;
    auto qz = [&](const float* v, int n, float* o) {
        float lo = v[0], hi = v[0], ss = 0.f, s1 = 0.f;
        for (int i = 0; i < n; ++i) {
            if (v[i] < lo) lo = v[i];
            if (v[i] > hi) hi = v[i];
            ss += v[i] * v[i];
            s1 += v[i];
        }
        const float mean = s1 / (float) n;
        const float sd = std::sqrt(std::max(0.0f, ss / (float) n - mean * mean));
        if (clip_sigma > 0.0f) {
            lo = std::max(lo, mean - clip_sigma * sd);
            hi = std::min(hi, mean + clip_sigma * sd);
        }
        float s = (hi - lo) / 3.0f;
        if (!(s > 0.0f)) s = 1e-8f;
        const float ds = k::f32_from_f16(k::f16_from_f32(s));       // stored as the device would store it
        const float dz = k::f32_from_f16(k::f16_from_f32(lo));
        for (int i = 0; i < n; ++i) {
            float q = std::nearbyintf((v[i] - dz) / ds);
            q = q < 0.f ? 0.f : (q > 3.f ? 3.f : q);
            o[i] = dz + q * ds;
        }
    };
    std::vector<float> tmp(512), tmpo(512);
    if (per_channel) {
        for (int h = 0; h < n_heads; ++h)
            for (int d = 0; d < D; ++d)
                for (int g = 0; g < n_cells; g += 32) {
                    for (int i = 0; i < 32; ++i) tmp[i] = x[(size_t) (g + i) * stride + h * D + d];
                    qz(tmp.data(), 32, tmpo.data());
                    for (int i = 0; i < 32; ++i) out[(size_t) (g + i) * stride + h * D + d] = tmpo[i];
                }
    } else {
        for (int c = 0; c < n_cells; ++c)
            for (int h = 0; h < n_heads; ++h) {
                const size_t base = (size_t) c * stride + h * D;
                qz(x + base, D, out + base);
            }
    }
}

/// JOINT per-channel x per-token scaling (the Coupled-Quantization / SKVQ direction): a value's scale is the
/// PRODUCT of a channel scale and a token scale, so one 2-bit code carries both structures at once instead of
/// choosing one axis. Costs the per-channel fp16 (16 B per cell and head) plus one fp16 per cell and head
/// (2 B): 82 B per cell and head at 2 bits, against 80 B for the single-axis per-channel block.
void quant_joint(const float* x, int n_cells, int n_heads, int D, float* out) {
    const int stride = n_heads * D;
    const int ngroups = n_cells / 32;
    std::vector<float> a((size_t) ngroups * stride, 1.0f), b((size_t) n_cells * n_heads, 1.0f);
    for (int h = 0; h < n_heads; ++h)
        for (int d = 0; d < D; ++d)
            for (int g = 0; g < ngroups; ++g) {
                float ss = 0.f;
                for (int i = 0; i < 32; ++i) {
                    const float v = x[(size_t) (g * 32 + i) * stride + h * D + d];
                    ss += v * v;
                }
                a[(size_t) g * stride + h * D + d] = std::sqrt(ss / 32.0f);
            }
    // The token scale is measured with the channel scale divided out, then normalized to mean 1 so the two
    // factors cannot both claim the same variance.
    std::vector<float> tr((size_t) n_cells * n_heads, 0.f);
    double acc = 0.0;
    for (int c = 0; c < n_cells; ++c)
        for (int h = 0; h < n_heads; ++h) {
            float ss = 0.f;
            for (int d = 0; d < D; ++d) {
                const size_t idx = (size_t) c * stride + h * D + d;
                const float g = a[(size_t) (c / 32) * stride + h * D + d];
                const float r = g > 0.f ? x[idx] / g : 0.f;
                ss += r * r;
            }
            tr[(size_t) c * n_heads + h] = std::sqrt(ss / (float) D);
            acc += tr[(size_t) c * n_heads + h];
        }
    const double mean = acc / ((double) n_cells * n_heads);
    for (size_t i = 0; i < tr.size(); ++i) b[i] = mean > 0.0 ? (float) (tr[i] / mean) : 1.0f;
    for (int c = 0; c < n_cells; ++c)
        for (int h = 0; h < n_heads; ++h) {
            const float bs = k::f32_from_f16(k::f16_from_f32(b[(size_t) c * n_heads + h]));
            for (int d = 0; d < D; ++d) {
                const size_t idx = (size_t) c * stride + h * D + d;
                const float as = k::f32_from_f16(k::f16_from_f32(a[(size_t) (c / 32) * stride + h * D + d]));
                const float s = as * bs;
                out[idx] = s > 0.f ? k::q2_value_lm4(k::q2_code_lm4(x[idx], s), s) : 0.f;
            }
        }
}

/// Relative RMS of `got` against `ref`, over the whole vector: the same units the window work reports in.
double rel_rms(const std::vector<float>& got, const std::vector<float>& ref) {
    double se = 0.0, ss = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = (double) got[i] - ref[i];
        se += e * e;
        ss += (double) ref[i] * ref[i];
    }
    return ss > 0.0 ? std::sqrt(se / ss) : 0.0;
}

/// The attention output's error measured against the cache's own value scale. Dividing by the norm of the
/// reference output is the obvious choice and is wrong: attention averages zero-mean values, so the reference
/// output is small, and any error reads as hundreds of percent. Against sqrt(mean(V^2)) the number says what it
/// looks like next to the data the model is mixing.
double out_err(const std::vector<float>& got, const std::vector<float>& ref, const std::vector<float>& scale_of) {
    double se = 0.0, ss = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = (double) got[i] - ref[i];
        se += e * e;
    }
    for (const float x : scale_of) ss += (double) x * x;
    return (ss > 0.0 && !got.empty()) ? std::sqrt(se / got.size()) / std::sqrt(ss / scale_of.size()) : 0.0;
}

/// Softmax attention with the given K/V, one head, over `cells` cell indices: the score a query would get, and the
/// output it would return. Reference and quantized run through this the same way, so the difference is the error
/// the model would see. The 1/sqrt(D) score scaling is not optional: without it the scores of a 256-dim head span
/// hundreds of logits, the softmax degenerates to an argmax, and a 4-bit cache measures worse than a 2-bit one
/// because flipping the single winning cell moves the whole output.
void attention(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V, int D,
               const std::vector<int>& cells, std::vector<float>& out) {
    out.assign(q.size(), 0.f);
    const double temp = 1.0 / std::sqrt((double) D);
    std::vector<double> score(cells.size());
    double mx = -1e300, sum = 0.0;
    for (size_t c = 0; c < cells.size(); ++c) {
        double s = 0.0;
        for (int d = 0; d < D; ++d) s += (double) q[d] * K[(size_t) cells[c] * D + d];
        score[c] = s * temp;
        if (s * temp > mx) mx = s * temp;
    }
    for (double& s : score) { s = std::exp(s - mx); sum += s; }
    for (double& s : score) s /= sum;
    for (size_t c = 0; c < cells.size(); ++c)
        for (int d = 0; d < D; ++d) out[d] += (float) (score[c] * V[(size_t) cells[c] * D + d]);
}
}  // namespace

int main() {
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 4;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size;
    int N = 1024;                                        // cells in the cache (the loader may set a smaller one)
    int pages = 0, rows = 0;                             // derived from N below, once N is final

    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K, V;
    bool real_cache = false;
    int probe_layer = -1;                                 // the layer the real cache came from (-1 = synthetic)
    if (const char* e = std::getenv("STRATA_KV_PROBE_CACHE")) {
        int want = 0, layer_out = 0;
        if (const char* l = std::getenv("STRATA_KV_PROBE_LAYER")) want = std::atoi(l);
        std::string cerr;
        if (!load_real_cache(e, want, H, D, 1024, N, K, V, layer_out, cerr)) {
            std::fprintf(stderr, "strata kv_q2_parity: %s\n", cerr.c_str());
            return 2;
        }
        real_cache = true;
        probe_layer = layer_out;
        std::printf("REAL CACHE (stored, rotated): %s layer %d - %d cells x %d kv heads x %d dims\n",
                    e, layer_out, N, H, D);
        channel_stats(K, N, H, D, "K");
        channel_stats(V, N, H, D, "V");
    }
    pages = N / P;
    rows = N * H;                                        // identity layout: one row per (cell, kv head)
    if (!real_cache) {
        K.assign((size_t) N * H * D, 0.f);
        V.assign((size_t) N * H * D, 0.f);
        // Mild per-cell magnitude spread on purpose: a cache where every cell has the same scale flatters every
        // quantizer, because the softmax weights barely move. 25x spikes were tried first and made the probe
        // useless - they hand the whole softmax to a handful of cells, so flipping which one wins (which 4 bits
        // can do) moves the entire output, and q4_0 measured worse than a 2-bit cache. 3x is the regime a real
        // rotated cache has.
        for (int cell = 0; cell < N; ++cell) {
            const float sc = (cell % 17 == 0) ? 3.0f : (cell % 23 == 0 ? 0.1f : 1.0f);
            for (int i = 0; i < H * D; ++i) {
                K[(size_t) cell * H * D + i] = nd(rng) * sc;
                V[(size_t) cell * H * D + i] = nd(rng) * sc;
            }
        }
        std::printf("SYNTHETIC CACHE: i.i.d. Gaussian, 3x per-cell spread - blind to channel structure, so "
                    "every per-channel number here is a floor\n");
    }

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    const size_t q2_bytes = (size_t) rows * k::kv_q2_bytes_per_head(D);
    const size_t q4_bytes = (size_t) rows * k::kv_q4_bytes_per_head(D);
    uint8_t *kq2 = dalloc<uint8_t>(q2_bytes), *vq2 = dalloc<uint8_t>(q2_bytes);
    uint8_t* vq2_4 = dalloc<uint8_t>(q2_bytes);                 // V written with the 4-level codebook, its own pool
    uint8_t *kq4 = dalloc<uint8_t>(q4_bytes), *vq4 = dalloc<uint8_t>(q4_bytes);
    uint16_t* k_scr = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr4 = dalloc<uint16_t>((size_t) rows * D);      // V read back with the 4-level codebook
    uint16_t* k_scr_q4 = dalloc<uint16_t>((size_t) rows * D);    // q4_0 gathers into their own scratch, so they
    uint16_t* v_scr_q4 = dalloc<uint16_t>((size_t) rows * D);    // cannot overwrite the q2 results already there
    float* dK = dalloc<float>(K.size());
    float* dV = dalloc<float>(V.size());
    ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
    ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");

    // Rotate, exactly as the engine does before it quantizes: the reference below is the ROTATED cache, because
    // that is what the codes represent, and the query is rotated the same way.
    k::fwht256_cuda(dK, dK, (int64_t) rows, nullptr);
    k::fwht256_cuda(dV, dV, (int64_t) rows, nullptr);
    ck(cudaDeviceSynchronize(), "rotate");
    std::vector<float> Kref(K.size()), Vref(V.size());
    ck(cudaMemcpy(Kref.data(), dK, K.size() * 4, cudaMemcpyDeviceToHost), "Kref");
    ck(cudaMemcpy(Vref.data(), dV, V.size() * 4, cudaMemcpyDeviceToHost), "Vref");

    // Append through the real kernels: the prompt path, T = N cells from pos 0. The pools are the destination;
    // dK/dV are the rotated source tensors.
    k::kv_append_q2(kq2, vq2, d_table, 0, N, dK, dV, s, nullptr, nullptr, nullptr, /*v3level*/ true);
    k::kv_append_q2(kq2, vq2_4, d_table, 0, N, dK, dV, s, nullptr, nullptr, nullptr, /*v3level*/ false);
    k::kv_append_q4(kq4, vq4, d_table, 0, N, dK, dV, s, nullptr, nullptr);
    ck(cudaDeviceSynchronize(), "append");

    // Read every cell back with the real gather kernel.
    std::vector<int32_t> ids(N);
    for (int i = 0; i < N; ++i) ids[i] = i;
    int32_t* d_ids = dalloc<int32_t>(N);
    ck(cudaMemcpy(d_ids, ids.data(), N * 4, cudaMemcpyHostToDevice), "ids");
    int32_t hstep[k::kStepCount] = {N - 1, N, 0, N};
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);
    ck(cudaMemcpy(d_step, hstep, sizeof hstep, cudaMemcpyHostToDevice), "step");
    k::kv_gather_q2_step(kq2, vq2, d_table, d_ids, d_step, N, s, k_scr, v_scr, nullptr, /*v3level*/ true);
    k::kv_gather_q2_step(kq2, vq2_4, d_table, d_ids, d_step, N, s, k_scr, v_scr4, nullptr, /*v3level*/ false);
    ck(cudaDeviceSynchronize(), "gather");

    std::vector<uint16_t> h_k(rows * D), h_v(rows * D), h_v4(rows * D);
    ck(cudaMemcpy(h_k.data(), k_scr, h_k.size() * 2, cudaMemcpyDeviceToHost), "k back");
    ck(cudaMemcpy(h_v.data(), v_scr, h_v.size() * 2, cudaMemcpyDeviceToHost), "v back");
    ck(cudaMemcpy(h_v4.data(), v_scr4, h_v4.size() * 2, cudaMemcpyDeviceToHost), "v4 back");

    // 1. the device quantizer equals the PER-CHANNEL host reference, bit for bit, over every (dimension, group).
    // quant_kivi is the reference; the per-token Lloyd-Max arm (quant_lm) is a comparison now, not the parity
    // target - the device no longer writes per-token blocks.
    std::vector<float> ref_k(Kref.size()), ref_v3(Vref.size()), ref_v4(Vref.size());
    quant_kivi(Kref.data(), N, H, D, false, ref_k.data());     // K: 4-level Lloyd-Max, per-channel scale
    quant_kivi(Vref.data(), N, H, D, true, ref_v3.data());     // V: the 3-level codebook, same scale
    quant_kivi(Vref.data(), N, H, D, false, ref_v4.data());    // V: the 4-level, same scale
    bool bit_exact = true;
    for (size_t i = 0; i < h_k.size() && bit_exact; ++i) {
        // The gathered values are fp16 bits straight off the device; the reference is quantized on the host and
        // rounded the same way, so an unequal bit pattern is a real writer/reader drift.
        if (h_k[i] != k::f16_from_f32(ref_k[i])) bit_exact = false;
        if (h_v[i] != k::f16_from_f32(ref_v3[i])) bit_exact = false;
        if (h_v4[i] != k::f16_from_f32(ref_v4[i])) bit_exact = false;
    }
    check(bit_exact, "the device quantizer reproduces the per-channel host reference bitwise (K 4-level, "
                     "V 3-level and 4-level)");

    // 2. the cost of 2 bits, in the same units as the window work
    std::vector<float> h_k2(h_k.size()), h_v2_3(h_v.size()), h_v2_4(h_v4.size()), h_k4(rows * D), h_vq4(rows * D);
    for (size_t i = 0; i < h_k.size(); ++i) h_k2[i] = k::f32_from_f16(h_k[i]);
    for (size_t i = 0; i < h_v.size(); ++i) h_v2_3[i] = k::f32_from_f16(h_v[i]);
    for (size_t i = 0; i < h_v4.size(); ++i) h_v2_4[i] = k::f32_from_f16(h_v4[i]);

    std::vector<uint16_t> w_k4(rows * D), w_v4(rows * D);
    k::kv_gather_q4_step(kq4, vq4, d_table, d_ids, d_step, N, s, k_scr_q4, v_scr_q4, nullptr);
    ck(cudaDeviceSynchronize(), "gather q4");
    ck(cudaMemcpy(w_k4.data(), k_scr_q4, w_k4.size() * 2, cudaMemcpyDeviceToHost), "k q4");
    ck(cudaMemcpy(w_v4.data(), v_scr_q4, w_v4.size() * 2, cudaMemcpyDeviceToHost), "v q4");
    for (size_t i = 0; i < w_k4.size(); ++i) h_k4[i] = k::f32_from_f16(w_k4[i]);
    for (size_t i = 0; i < w_v4.size(); ++i) h_vq4[i] = k::f32_from_f16(w_v4[i]);

    // ---- THE q2k DEVICE ARM (`--kv q2k`): K filled by the q2 group kernels, V by the q4 kernels, each half
    // read back by its own gather kernel. These are the engine's own calls, in the shape the engine makes them:
    // the q2 append called with its pool and tensor twice (its K plane only - the fold layer.cpp/verify.cpp use),
    // the q4 append called the same way for V. Nothing here is composed on the host: the bytes are the pools a
    // q2k state carves (K at 80 B per cell and head, V at 144) and the numbers are what its readers return.
    uint8_t* kq2_dev = dalloc<uint8_t>(q2_bytes);
    uint8_t* vq4_dev = dalloc<uint8_t>(q4_bytes);
    uint16_t* k_scr_2k = dalloc<uint16_t>((size_t) rows * D);
    uint16_t* v_scr_2k = dalloc<uint16_t>((size_t) rows * D);
    k::kv_append_q2(kq2_dev, kq2_dev, d_table, 0, N, dK, dK, s, nullptr, nullptr, nullptr, /*v3level*/ false);
    k::kv_append_q4(vq4_dev, vq4_dev, d_table, 0, N, dV, dV, s, nullptr, nullptr);
    ck(cudaDeviceSynchronize(), "q2k append");
    k::kv_gather_q2_step(kq2_dev, kq2_dev, d_table, d_ids, d_step, N, s, k_scr_2k, k_scr_2k, nullptr,
                         /*v3level*/ false);
    k::kv_gather_q4_step(vq4_dev, vq4_dev, d_table, d_ids, d_step, N, s, v_scr_2k, v_scr_2k, nullptr);
    ck(cudaDeviceSynchronize(), "q2k gather");
    std::vector<uint16_t> w_k2k(rows * D), w_v2k(rows * D);
    ck(cudaMemcpy(w_k2k.data(), k_scr_2k, w_k2k.size() * 2, cudaMemcpyDeviceToHost), "k q2k");
    ck(cudaMemcpy(w_v2k.data(), v_scr_2k, w_v2k.size() * 2, cudaMemcpyDeviceToHost), "v q2k");
    std::vector<float> h_k2k(rows * D), h_v2k(rows * D);
    for (size_t i = 0; i < w_k2k.size(); ++i) h_k2k[i] = k::f32_from_f16(w_k2k[i]);
    for (size_t i = 0; i < w_v2k.size(); ++i) h_v2k[i] = k::f32_from_f16(w_v2k[i]);
    // The mixed format must not move either half's bytes: K is bit for bit what the q2 cache holds, V bit for
    // bit what the q4_0 cache holds. A mixed dispatch that re-derived either side would fail here, and only here.
    bool k2k_is_q2 = true, v2k_is_q4 = true;
    for (size_t i = 0; i < w_k2k.size(); ++i) {
        if (w_k2k[i] != h_k[i]) k2k_is_q2 = false;
        if (w_v2k[i] != w_v4[i]) v2k_is_q4 = false;
    }
    check(k2k_is_q2, "q2k's K is bit for bit the q2 cache's K (the mixed dispatch re-quantizes nothing)");
    check(v2k_is_q4, "q2k's V is bit for bit the q4_0 cache's V (the mixed dispatch re-quantizes nothing)");
    const double r_k2k = rel_rms(h_k2k, Kref), r_v2k = rel_rms(h_v2k, Vref);

    const double r_k2 = rel_rms(h_k2, Kref), r_v2_3 = rel_rms(h_v2_3, Vref);
    const double r_v2_4 = rel_rms(h_v2_4, Vref), r_k4 = rel_rms(h_k4, Kref), r_v4 = rel_rms(h_vq4, Vref);

    // The uniform grid, quantized on the host: the device now ships Lloyd-Max, so the uniform rows are the
    // comparison, not a second read of the same pools.
    std::vector<float> h_k_uni(Kref.size()), h_v_uni4(Vref.size()), h_v_uni3(Vref.size());
    for (size_t row = 0; row < rows; ++row)
        for (int b = 0; b < D / k::Q2_0; ++b) {
            const size_t base = row * D + (size_t) b * k::Q2_0;
            quant_block(Kref.data() + base, k::Q2_0, false, h_k_uni.data() + base);
            quant_block(Vref.data() + base, k::Q2_0, false, h_v_uni4.data() + base);
            quant_block(Vref.data() + base, k::Q2_0, true, h_v_uni3.data() + base);
        }
    const double r_k_uni = rel_rms(h_k_uni, Kref), r_v_uni4 = rel_rms(h_v_uni4, Vref);
    const double r_v_uni3 = rel_rms(h_v_uni3, Vref);
    // Scale granularity: the shipping codebook with a scale every 16 values.  Host-side, so the number costs
    // nothing to obtain; the layout only changes if it pays for the extra 2 bytes per 32 values.
    std::vector<float> h_k_lm16(Kref.size()), h_v_lm16(Vref.size());
    for (size_t row = 0; row < rows; ++row) {
        quant_lm16(Kref.data() + row * D, D, h_k_lm16.data() + row * D);
        quant_lm16(Vref.data() + row * D, D, h_v_lm16.data() + row * D);
    }
    const double r_k_lm16 = rel_rms(h_k_lm16, Kref), r_v_lm16 = rel_rms(h_v_lm16, Vref);
    // KIVI per-channel scaling, same 2 bits and the same 80 B per cell and head: the host arm of the row the
    // device fills, so the table's per-channel row and the device's Lloyd-Max rows are the same measurement by
    // two paths (and check 1 says they agree bit for bit).
    std::vector<float> h_k_kivi(Kref.size()), h_v_kivi(Vref.size());
    quant_kivi(Kref.data(), N, H, D, false, h_k_kivi.data());
    quant_kivi(Vref.data(), N, H, D, false, h_v_kivi.data());
    const double r_k_kivi = rel_rms(h_k_kivi, Kref), r_v_kivi = rel_rms(h_v_kivi, Vref);
    // h_k2 / h_v2_4 / h_v2_3 are the device's PER-CHANNEL pools, read back through the real gather kernel.
    const double r_k_lm = r_k2, r_v_lm4 = r_v2_4, r_v_lm3 = r_v2_3;
    // The per-token Lloyd-Max layout this format had before: host-only now (the device writes per-channel
    // groups), kept because the scale-placement measurement is stated against it - 33.4% per element against
    // 44.4%, and 19.9% of the attention output against 7.15%.
    std::vector<float> h_k_lmt(Kref.size()), h_v_lmt4(Vref.size());
    for (size_t row = 0; row < rows; ++row)
        for (int b = 0; b < D / k::Q2_0; ++b) {
            const size_t base = row * D + (size_t) b * k::Q2_0;
            quant_lm(Kref.data() + base, k::Q2_0, false, h_k_lmt.data() + base);
            quant_lm(Vref.data() + base, k::Q2_0, false, h_v_lmt4.data() + base);
        }
    const double r_k_lmt = rel_rms(h_k_lmt, Kref), r_v_lmt4 = rel_rms(h_v_lmt4, Vref);
    // ---- Asymmetric K/V bit widths (the KIVI line of work: K and V are not equally sensitive, so they are
    // quantized DIFFERENTLY, not at the same bit width).  Host-side arms composed from the same dequantized
    // arrays the device pools produce, so each arm's byte cost is stated and the comparison is like for like.
    std::vector<float> h_v_1c(Vref.size()), h_v_1t(Vref.size()), h_k_1c(Kref.size());
    quant_1bit(Vref.data(), N, H, D, true, h_v_1c.data());    // V 1-bit, per-channel scale
    quant_1bit(Vref.data(), N, H, D, false, h_v_1t.data());   // V 1-bit, per-token scale (KIVI's V arm)
    quant_1bit(Kref.data(), N, H, D, true, h_k_1c.data());    // K 1-bit, per-channel scale
    const double r_v_1c = rel_rms(h_v_1c, Vref), r_v_1t = rel_rms(h_v_1t, Vref), r_k_1c = rel_rms(h_k_1c, Kref);
    // ---- The published schemes' QUANTIZER, not ours: asymmetric min/max with a zero-point (KIVI,
    // arXiv 2402.02750), with SKVQ's clipping (arXiv 2405.06219), and the joint per-channel x per-token scale.
    // Same 2 bits, a different scale geometry - this is the arm the "near lossless" claim rests on.
    std::vector<float> h_k_ac(Kref.size()), h_v_ac(Vref.size()), h_v_at(Vref.size()), h_k_ac3(Kref.size()),
                       h_k_j(Kref.size()), h_v_j(Vref.size());
    quant_asym(Kref.data(), N, H, D, true, 0.0f, h_k_ac.data());    // KIVI: per-channel, z = min
    quant_asym(Kref.data(), N, H, D, true, 3.0f, h_k_ac3.data());   // the same, SKVQ-clipped
    quant_asym(Vref.data(), N, H, D, true, 0.0f, h_v_ac.data());
    quant_asym(Vref.data(), N, H, D, false, 0.0f, h_v_at.data());   // KIVI's V axis: per-token
    quant_joint(Kref.data(), N, H, D, h_k_j.data());
    quant_joint(Vref.data(), N, H, D, h_v_j.data());
    const double r_k_ac = rel_rms(h_k_ac, Kref), r_v_ac = rel_rms(h_v_ac, Vref), r_v_at = rel_rms(h_v_at, Vref);
    const double r_k_ac3 = rel_rms(h_k_ac3, Kref), r_k_j = rel_rms(h_k_j, Kref), r_v_j = rel_rms(h_v_j, Vref);
    std::printf("\nError against an FP16 cache (rotated), relative RMS:\n");
    std::printf("  %-26s %8s %8s\n", "format", "K", "V");
    std::printf("  (per-element diagnostics - the decision metric is the attention output table below)\n");
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q4_0 (the format in use)", 100 * r_k4, 100 * r_v4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 uniform, V 4-level", 100 * r_k_uni, 100 * r_v_uni4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 uniform, V 3-level", 100 * r_k_uni, 100 * r_v_uni3);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, V 4-level", 100 * r_k_lm, 100 * r_v_lm4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, V 3-level", 100 * r_k_lm, 100 * r_v_lm3);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, 16-value scale", 100 * r_k_lm16, 100 * r_v_lm16);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, per-token (host)", 100 * r_k_lmt, 100 * r_v_lmt4);
    std::printf("  %-26s %7.3f%% %7.3f%%\n", "q2 Lloyd-Max, per-channel", 100 * r_k_kivi, 100 * r_v_kivi);
    std::printf("  (the two Lloyd-Max V rows are the DEVICE pools; the per-channel row is the same pools quantized\n");
    std::printf("   on the host, which check 1 says agree bit for bit. Per-channel is the WORSE per-element row\n");
    std::printf("   and the better attention row - see the note under the second table.)\n");

    check(r_k_lm > r_k4, "2-bit K is worse than 4-bit K (the table is not measuring the same thing twice)");
    check(r_v_lm3 > r_v_lm4, "the 3-level V is lossier than the 4-level V (so the codebook is a real choice)");
    std::printf("  ratio 2-bit K / 4-bit K: %.2fx\n", r_k_lm / r_k4);
    check(r_k_lm < 8.0 * r_k4, "2-bit K's error stays under 8x the 4-bit error (the 2-bit cache is not collapsing)");
    check(r_k_lm < r_k_uni && r_v_lm4 < r_v_uni4, "the Lloyd-Max codebook beats the uniform one at the same 2 bits");
    std::printf("  Lloyd-Max gain over uniform at 2 bits: K %.2fx, V %.2fx\n", r_k_uni / r_k_lm, r_v_uni4 / r_v_lm4);
    // The two metrics split, and the split is the finding: per-channel is lossier per element than the per-token
    // block scale and better on the attention output. Both halves are pinned so neither can be quoted alone.
    check_data(real_cache, r_k_kivi > r_k_lmt, r_k_kivi <= r_k_lmt,
               "per-element, SYNTHETIC: per-channel is the lossier row (no channel outliers for it to exploit)",
               "per-element, REAL CACHE: per-channel is the better row (the channel outliers are what it exploits)");
    std::printf("  per-channel vs per-token, per element: K %.2fx %s, V %.2fx %s\n",
                r_k_kivi > r_k_lmt ? r_k_kivi / r_k_lmt : r_k_lmt / r_k_kivi,
                r_k_kivi > r_k_lmt ? "worse" : "better",
                r_v_kivi > r_v_lmt4 ? r_v_kivi / r_v_lmt4 : r_v_lmt4 / r_v_kivi,
                r_v_kivi > r_v_lmt4 ? "worse" : "better");

    // 3. the attention-output error, one head, over the whole cache
    std::vector<float> q(D);
    for (auto& x : q) x = nd(rng);
    std::vector<int> cells(N);
    for (int i = 0; i < N; ++i) cells[i] = i;
    std::vector<float> out_ref, out_lm3, out_lm4, out_q4, out_uni4, out_uni3, out_lm16, out_kivi, out_lmt,
                       out_k2v4, out_k2v4_dev, out_k4v2, out_k2v1c, out_k2v1t, out_k2c2t, out_k2t2c, out_k1v1,
                       out_ac, out_acat, out_acatc, out_jj, out_jat;
    attention(q, Kref, Vref, D, cells, out_ref);
    attention(q, h_k2, h_v2_3, D, cells, out_lm3);       // device pools (per-channel), Lloyd-Max 3-level V
    attention(q, h_k2, h_v2_4, D, cells, out_lm4);       // device pools (per-channel), Lloyd-Max 4-level V
    attention(q, h_k4, h_vq4, D, cells, out_q4);
    attention(q, h_k_uni, h_v_uni4, D, cells, out_uni4);
    attention(q, h_k_uni, h_v_uni3, D, cells, out_uni3);
    attention(q, h_k_lm16, h_v_lm16, D, cells, out_lm16);
    attention(q, h_k_kivi, h_v_kivi, D, cells, out_kivi);   // the host per-channel arm: out_lm4's own number
    attention(q, h_k_lmt, h_v_lmt4, D, cells, out_lmt);     // the per-token layout, host-only
    // The asymmetric arms: same attention probe, K and V quantized at different widths / scale placements.
    attention(q, h_k_kivi, h_vq4, D, cells, out_k2v4);      // K 2-bit per-channel, V 4-bit (host K reference)
    attention(q, h_k2k, h_v2k, D, cells, out_k2v4_dev);     // the SAME arm off the DEVICE pools: `--kv q2k`
    attention(q, h_k4, h_v_kivi, D, cells, out_k4v2);       // K 4-bit, V 2-bit per-channel
    attention(q, h_k_kivi, h_v_1c, D, cells, out_k2v1c);    // K 2-bit, V 1-bit per-channel
    attention(q, h_k_kivi, h_v_1t, D, cells, out_k2v1t);    // K 2-bit, V 1-bit per-token (KIVI's split)
    attention(q, h_k_kivi, h_v_lmt4, D, cells, out_k2c2t);  // K per-channel 2-bit, V per-token 2-bit
    attention(q, h_k_lmt, h_v_kivi, D, cells, out_k2t2c);   // K per-token 2-bit, V per-channel 2-bit
    attention(q, h_k_1c, h_v_1c, D, cells, out_k1v1);       // 1-bit both, per-channel
    attention(q, h_k_ac, h_v_ac, D, cells, out_ac);         // KIVI's quantizer, per-channel on both
    attention(q, h_k_ac, h_v_at, D, cells, out_acat);       // KIVI's exact split: K per-channel, V per-token
    attention(q, h_k_ac3, h_v_at, D, cells, out_acatc);     // the same with SKVQ clipping on K
    attention(q, h_k_j, h_v_j, D, cells, out_jj);           // joint channel x token scales on both
    attention(q, h_k_j, h_v_at, D, cells, out_jat);         // joint K, per-token asymmetric V
    const double e_ac = out_err(out_ac, out_ref, Vref), e_acat = out_err(out_acat, out_ref, Vref);
    const double e_acatc = out_err(out_acatc, out_ref, Vref), e_jj = out_err(out_jj, out_ref, Vref);
    const double e_jat = out_err(out_jat, out_ref, Vref);
    const double e_k2v4 = out_err(out_k2v4, out_ref, Vref), e_k4v2 = out_err(out_k4v2, out_ref, Vref);
    const double e_k2v4_dev = out_err(out_k2v4_dev, out_ref, Vref);   // the device arm: `--kv q2k` as it runs
    const double e_k2v1c = out_err(out_k2v1c, out_ref, Vref), e_k2v1t = out_err(out_k2v1t, out_ref, Vref);
    const double e_k2c2t = out_err(out_k2c2t, out_ref, Vref), e_k2t2c = out_err(out_k2t2c, out_ref, Vref);
    const double e_k1v1 = out_err(out_k1v1, out_ref, Vref);
    const double e_lm3 = out_err(out_lm3, out_ref, Vref), e_lm4 = out_err(out_lm4, out_ref, Vref);
    const double e_q4 = out_err(out_q4, out_ref, Vref);
    const double e_uni4 = out_err(out_uni4, out_ref, Vref), e_uni3 = out_err(out_uni3, out_ref, Vref);
    const double e_lm16 = out_err(out_lm16, out_ref, Vref);
    const double e_kivi = out_err(out_kivi, out_ref, Vref);
    const double e_lmt = out_err(out_lmt, out_ref, Vref);
    std::printf("\nAttention output against the FP16 cache, relative RMS:\n");
    std::printf("  %-26s %7.3f%%\n", "q4_0 (the format in use)", 100 * e_q4);
    std::printf("  %-26s %7.3f%%\n", "q2 uniform, V 4-level", 100 * e_uni4);
    std::printf("  %-26s %7.3f%%\n", "q2 uniform, V 3-level", 100 * e_uni3);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, V 4-level", 100 * e_lm4);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, V 3-level", 100 * e_lm3);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, 16-value scale", 100 * e_lm16);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, per-token (host)", 100 * e_lmt);
    std::printf("  %-26s %7.3f%%\n", "q2 Lloyd-Max, per-channel", 100 * e_kivi);
    std::printf("  (the V 4-level / V 3-level rows are the DEVICE pools read through the real gather kernel; the\n");
    std::printf("   per-channel row is the host reference for the same pools. THIS IS THE DECISION METRIC.)\n");
    // ---- Asymmetric K/V.  The published 2-bit schemes do not spend the same bits on K and V, so the arms differ
    // in WHICH tensor is coarse, not only in how many bits.  Byte cost per cell and KV head at head_dim 256:
    // q4_0 288, q2 per-channel 80, 1-bit per-channel 48, 1-bit per-token 34.
    const int B4 = (int) k::kv_q4_bytes_per_head(D), B2 = (int) k::kv_q2_bytes_per_head(D);
    const int B1c = D / 8 + 2 * D / 32, B1t = D / 8 + 2;
    std::printf("\n  asymmetric K/V arms (attention output; bytes per cell and KV head):\n");
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 4-bit\n", "K2c + V4", 100 * e_k2v4, B2 + B4);
    std::printf("  %-34s %7.3f%%  %4d B   the SAME arm off the device pools: --kv q2k (q2 append for K, q4 for V)\n",
                "K2c + V4 (device)", 100 * e_k2v4_dev, B2 + B4);
    std::printf("  %-34s %7.3f%%  %4d B   K 4-bit, V 2b per-channel\n", "K4 + V2c", 100 * e_k4v2, B4 + B2);
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 1b per-channel\n", "K2c + V1c", 100 * e_k2v1c, B2 + B1c);
    std::printf("  %-34s %7.3f%%  %4d B   K 2b per-channel, V 1b per-token (KIVI's split)\n", "K2c + V1t", 100 * e_k2v1t, B2 + B1t);
    std::printf("  %-34s %7.3f%%  %4d B   K per-channel, V per-token, both 2b\n", "K2c + V2t", 100 * e_k2c2t, 2 * B2);
    std::printf("  %-34s %7.3f%%  %4d B   K per-token, V per-channel, both 2b\n", "K2t + V2c", 100 * e_k2t2c, 2 * B2);
    std::printf("  %-34s %7.3f%%  %4d B   both 1-bit per-channel\n", "K1c + V1c", 100 * e_k1v1, 2 * B1c);
    std::printf("  reference rows: symmetric q2c %7.3f%% (%4d B), q4_0 %7.3f%% (%4d B)\n",
                100 * e_lm4, 2 * B2, 100 * e_q4, 2 * B4);
    // The published quantizers, at their own byte costs (per-channel asymmetric: 64 B codes + 2 fp16 per
    // dimension per 32 cells = 96 B; per-token asymmetric: 64 B codes + 2 fp16 = 68 B; joint: 64 + 16 + 2 = 82).
    std::printf("\n  the published quantizers (asymmetric min/max + zero-point, joint scales):\n");
    std::printf("  %-34s %7.3f%%  %4d B   KIVI's quantizer, per-channel on K and V\n", "asym-c K+V", 100 * e_ac, 2 * 96);
    std::printf("  %-34s %7.3f%%  %4d B   KIVI's split: K per-channel, V per-token\n", "asym-c K + asym-t V", 100 * e_acat, 96 + 68);
    std::printf("  %-34s %7.3f%%  %4d B   the same, K clipped at 3 sd (SKVQ)\n", "asym-c3 K + asym-t V", 100 * e_acatc, 96 + 68);
    std::printf("  %-34s %7.3f%%  %4d B   joint channel x token scales on K and V\n", "joint K+V", 100 * e_jj, 2 * 82);
    std::printf("  %-34s %7.3f%%  %4d B   joint K, per-token asymmetric V\n", "joint K + asym-t V", 100 * e_jat, 82 + 68);
    std::printf("  per-element: asym-c K %.3f%% (symmetric %.3f%%), asym-t V %.3f%%, joint K %.3f%%, joint V %.3f%%\n",
                100 * r_k_ac, 100 * r_k_kivi, 100 * r_v_at, 100 * r_k_j, 100 * r_v_j);
    check(e_ac > 0.0 && e_jj > 0.0, "the published quantizers are live in this probe");
    // Measured, and it is the largest single lever found in this work: the ZERO-POINT, not the bit count.
    // Same 2 bits, same per-channel group; a min/max scale with an offset instead of a symmetric RMS scale
    // moves the attention output 7.150% -> 3.777% for 12% more bytes (96 B against 80 B per cell and head).
    check(e_ac < e_lm4, "the asymmetric zero-point quantizer beats the symmetric RMS block on the attention output");
    // Clipping is LAYER-dependent on real data: it helps on layers 11 and 47, not on 3 and 23. Pinned as a
    // synthetic-only check; printed with the layer named so the reversal is visible instead of fatal.
    if (!real_cache)
        check(e_acatc < e_acat, "SYNTHETIC: clipping the group extremes helps (SKVQ's clipping)");
    std::printf("  clipping (layer %d): clipped %.3f%% vs unclipped %.3f%% -> %s\n", probe_layer,
                100 * e_acatc, 100 * e_acat, e_acatc < e_acat ? "clipping helps" : "clipping does not help");
    // Joint scales: on i.i.d. Gaussian noise a channel factor is estimator noise from 32 samples, multiplied
    // into every code, so it loses. On a real cache it depends on the layer: it ties the best arm on layer 3
    // and loses on layer 47. Same treatment - synthetic-only check, measured print everywhere.
    if (!real_cache)
        check(e_jj > e_lm4, "SYNTHETIC: joint scales lose - with no channel structure the channel factor is estimator noise");
    std::printf("  joint scales (layer %d): joint %.3f%% vs per-channel %.3f%% -> %s\n", probe_layer,
                100 * e_jj, 100 * e_lm4, e_jj <= e_lm4 ? "joint does not lose" : "joint loses");
    std::printf("  zero-point lever: %.3f%% against the shipping symmetric block's %.3f%% (%.2fx, 96 B vs 80 B); "
                "clipping %.3f%%; joint scales %.3f%%\n",
                100 * e_ac, 100 * e_lm4, e_lm4 / e_ac, 100 * e_acatc, 100 * e_jj);
    check(e_k2v4 > 0.0 && e_k1v1 > 0.0, "the asymmetric arms are live (the probe responds to K and V separately)");
    // The device arm IS the host arm's format, run through the engine's own writer and reader. The two differ
    // only in which path quantized K - the host reference against the q2 group kernel - and check 1 says those
    // two agree bit for bit, so the rows have to land on top of each other. A gap here means the q2k dispatch
    // is not the pair of formats it claims to be (a V read out of the q2 pool, a K read out of the q4 one).
    check(e_k2v4_dev > 0.0, "the q2k device arm is live (the engine's appends filled both halves)");
    check(e_k2v4_dev < 1.02 * e_k2v4 && e_k2v4_dev > 0.98 * e_k2v4,
          "the q2k device arm lands on the host K2c + V4 row (within 2% - same bytes, same readers)");
    check(e_k2v4_dev < e_lm4, "q2k beats the shipping symmetric 2-bit cache on the attention output (its reason to exist)");
    check(e_k2v4_dev > e_q4, "q2k is lossier than q4_0 (it is the cheaper format, not the better one)");
    // The device-pool twin of the host-side claim above, and it is data-scoped for the same reason: on the
    // synthetic cache the K side is the cheap one to coarsen, on a real cache the V side is.
    check_data(real_cache, e_k4v2 < e_k2v4_dev, e_k2v4_dev < e_k4v2,
               "SYNTHETIC: spending the bits on K beats spending them on V, on the device pools too",
               "REAL CACHE: spending the bits on V beats spending them on K, on the device pools too");
    std::printf("  q2k (device): %.3f%% against q2's %.3f%% (%.2fx better) and q4_0's %.3f%% (%.2fx worse), at %d B "
                "per cell and head; per element K %.3f%% (q2's %.3f%%), V %.3f%% (q4_0's %.3f%%)\n",
                100 * e_k2v4_dev, 100 * e_lm4, e_lm4 / e_k2v4_dev, 100 * e_q4, e_k2v4_dev / e_q4, B2 + B4,
                100 * r_k2k, 100 * r_k2, 100 * r_v2k, 100 * r_v4);
    // Measured, and the two data sources disagree on three of these four - which is the finding. KIVI's
    // placement claims (arXiv 2402.02750) rest on channel outliers; the synthetic cache has none, the real
    // rotated cache has them (K max/median channel RMS 5.8x, V 13.6x), and the V-side claims flip.
    check_data(real_cache, e_k2c2t < e_lm4, e_k2c2t >= e_lm4,
               "SYNTHETIC: KIVI's split (K per-channel, V per-token) beats symmetric per-channel",
               "REAL CACHE: KIVI's split loses - rotation left V's channel structure in place, so V wants a per-channel scale too");
    // K's scale axis, measured with V held per-channel in BOTH arms (the earlier phrasing compared
    // K2t+V2c against K2c+V2t, which moves both tensors' scale axes at once and measures V's channel
    // structure, not K's). Per-channel K wins on the synthetic cache and on real layers 3/11/23; on layer
    // 47, where K's channel RMS spans 74.9x, per-token K ties it - K's error is largely common-mode under
    // the softmax, so a per-channel scale buys less as the channel spread grows.
    if (!real_cache)
        check(e_k2t2c > e_lm4, "SYNTHETIC: a per-TOKEN K scale is the lossy side - K is the tensor that needs per-channel scaling");
    std::printf("  K scale axis (V per-channel in both arms, layer %d): per-token K %.3f%% vs per-channel K %.3f%% -> %s\n",
                probe_layer, 100 * e_k2t2c, 100 * e_lm4,
                e_k2t2c > e_lm4 ? "per-channel K wins" : "per-token K ties or wins");
    check_data(real_cache, e_k2v1t < e_k2v1c, e_k2v1t >= e_k2v1c,
               "SYNTHETIC: 1-bit V wants a per-token scale (KIVI's V finding)",
               "REAL CACHE: 1-bit V wants the per-channel scale (V's channel outliers survive the rotation)");
    // The condition is "K4 + V2c beats K2c + V4" (synthetic) / "K2c + V4 beats K4 + V2c" (real): whichever
    // tensor's coarse arm costs MORE is the sensitive one. On the synthetic cache that is K; on the real
    // rotated cache it is V - the ranking --kv q2k is built on.
    check_data(real_cache, e_k4v2 < e_k2v4, e_k4v2 >= e_k2v4,
               "SYNTHETIC: the K side is the sensitive one (coarse K costs more than coarse V: K4+V2c beats K2c+V4)",
               "REAL CACHE: the V side is the sensitive one (coarse V costs more than coarse K: K2c+V4 beats K4+V2c) - the reverse of the synthetic ranking");
    std::printf("  asymmetry: KIVI's split %.3f%% vs symmetric per-channel %.3f%% (%.2fx, same 160 B); "
                "the best 224 B arm is %.3f%% against q4_0's %.3f%% - asymmetry narrows the gap, it does not close it\n",
                100 * e_k2c2t, 100 * e_lm4, e_lm4 / e_k2c2t, 100 * std::min(e_k4v2, e_k2v4), 100 * e_q4);
    // The probe's own limit, stated where it can be seen: the gap between 1-bit both and 2-bit both is small
    // on the synthetic cache and large on a real one, because a diffuse softmax averaging over N cells washes
    // coarse quantization away. This table RANKS schemes against each other; it is not an absolute quality
    // bar. The absolute bar is the logit-level harness (bench/results/2026-10-12-kv-precision).
    std::printf("  probe limit: 1-bit both (%d B) %.3f%% vs 2-bit both (%d B) at %.3f%% (%.2fx) over %d cells\n"
                "  - a diffuse softmax averages coarse quantization away, so this table RANKS schemes;\n"
                "  the absolute bar is the logit-level KL/top-1 table.\n",
                2 * B1c, 100 * e_k1v1, 2 * B2, 100 * e_lm4, e_k1v1 / e_lm4, N);
    // On i.i.d. Gaussian data every channel's RMS is the same number, so per-channel scaling can only show that
    // it does no harm here.  The gain KIVI claims comes from outliers aligned to dimensions, which this probe's
    // data does not have - the number that decides it has to come from a real cache.
    check(e_kivi < 1.10 * e_lm4, "per-channel scaling does no harm on this data (its gain needs real cache statistics)");
    // The two arms of the scale-placement measurement, on the metric that decides it. Per-channel is 33% worse
    // per element and 2.8x better here; the reason is in the header of kv_q2.hpp.
    check(e_lm4 < e_lmt, "per-channel beats the per-token block scale on the attention output (the shipping choice)");
    std::printf("  per-channel over per-token, attention output: %.2fx better (%.3f%% against %.3f%%), while being\n"
                "  %.2fx WORSE per element - the metrics disagree on purpose, see docs/kv-q2.md\n",
                e_lmt / e_lm4, 100 * e_lm4, 100 * e_lmt, r_k_kivi / r_k_lmt);
    std::printf("  16-value scale gain over 32-value: K %.2fx, V %.2fx, attention %.2fx "
                "(12 B per 32 values, not 10)\n", r_k_lmt / r_k_lm16, r_v_lmt4 / r_v_lm16, e_lmt / e_lm16);
    // Measured, not assumed: a 16-value scale moves K and V by 2% and makes the attention output 1.87x WORSE
    // (37.2% against 19.9%).  A shorter block estimates the scale from 16 samples instead of 32, and that
    // estimator noise lands on every code in the block; nothing here pays for it.  The gate pins the finding so a
    // future change cannot quietly adopt a finer block because finer sounds better.  Pinned against both the
    // per-token arm it was measured against and the shipping per-channel pools.
    check_data(real_cache, e_lm16 > e_lmt, e_lm16 <= e_lmt,
               "SYNTHETIC: the 16-value scale does not beat 32 on the attention output (estimator noise from 16 samples)",
               "REAL CACHE: the 16-value scale does beat 32 (a finer scale tracks the channel structure) - the block size is a data question, not a law");
    check(e_lm16 > e_lm4, "the 16-value scale does not beat the shipping per-channel pools either");
    check(e_lm4 > 0.0 && e_q4 > 0.0, "both quantized caches move the attention output (the probe is live)");
    check(e_lm4 < e_uni4 && e_lm3 < e_uni3, "the Lloyd-Max codebook moves the attention output less than the grid");

    // 4. the byte math the design doc quotes
    std::printf("\nBytes per cell and KV head: fp16 %d, q4_0 %llu, q2 %llu (head_dim %d)\n", 2 * D,
                (unsigned long long) k::kv_q4_bytes_per_head(D), (unsigned long long) k::kv_q2_bytes_per_head(D), D);
    check(k::kv_q2_bytes_per_head(D) == 80, "q2 is 80 B per cell and head (8 blocks x 10 B)");
    check(k::kv_q2_bytes_per_cell(s) == 2 * 2 * 80, "q2 is 320 B per cell for 2 KV heads (vs 576 for q4_0)");
    // q2k's byte math, pinned from the two widths the format is made of: 80 (K, q2 per-channel) + 144 (V, q4_0)
    // = 224 per cell and KV head, 448 per cell for 2 KV heads - 40% over q2's 320, 22% under q4_0's 576.
    const int B2K = B2 + B4;
    std::printf("Bytes per cell and KV head: q2k %d (q2 K %d + q4_0 V %d); per cell for %d KV heads: q2k %d, "
                "q2 %d, q4_0 %d\n",
                B2K, B2, B4, (int) s.n_head_kv, B2K * (int) s.n_head_kv, (int) k::kv_q2_bytes_per_cell(s),
                (int) k::kv_q4_bytes_per_cell(s));
    check(B2K == 224, "q2k is 224 B per cell and KV head (80 q2 + 144 q4_0)");
    check(B2K * (int) s.n_head_kv == 448, "q2k is 448 B per cell for 2 KV heads (q2: 320, q4_0: 576)");
    check(k::kv_stream_unit_blocks(s, k::kKvQ2K) == (int) (32 / s.page_size),
          "q2k's residency unit is the q2 group (32/page_size blocks): its K scale spans the group");
    check(k::kv_block_bytes(s, k::kKvQ2K) ==
              (uint64_t) (s.n_head_kv * s.page_size) * (k::kv_q2_bytes_per_head(D) + k::kv_q4_bytes_per_head(D)),
          "the q2k block mover moves K's q2 run and V's q4_0 run (224 B per cell and head)");

    // 5. A FRESH PARTIAL GROUP'S SCALE. A group whose first cells arrive in a batch shorter than 32 has no
    //    stored scale to keep, so the kernel takes the RMS of the cells the call has - of the 12 it has, not of
    //    32. Dividing a 12-cell sum by 32 under-scales the group by sqrt(32/12) = 1.63x, and every code in it
    //    is then quantized against a scale 1.63x too small. This is the real prompt-path append (use_prev = 0,
    //    a zeroed pool, so the group is fresh and the kernel writes its own scale), read back as stored bits;
    //    the expected bits are the host sum in cell order with the multiply and the add kept separate, for the
    //    reason quant_kivi states - an ulp in the scale is a different code at a threshold.
    if (N >= 12) {
        const int n12 = 12;
        const size_t pool12 = (size_t) H * D * sizeof(k::block_q2c_0);   // group 0's slots, every head and dim
        uint8_t* kq2_12 = dalloc<uint8_t>(pool12);                       // zeroed: a scale of 0 bits = no scale
        std::vector<int32_t> tab12(P);                                    // cells 0..11 = pages 0..2, identity
        for (int i = 0; i < P; ++i) tab12[i] = i;
        int32_t* d_tab12 = dalloc<int32_t>(P);
        ck(cudaMemcpy(d_tab12, tab12.data(), P * 4, cudaMemcpyHostToDevice), "12-cell table");
        k::kv_append_q2(kq2_12, kq2_12, d_tab12, 0, n12, dK, dK, s, nullptr, nullptr, nullptr, /*v3level*/ false);
        ck(cudaDeviceSynchronize(), "12-cell append");
        std::vector<uint8_t> h12(pool12);
        ck(cudaMemcpy(h12.data(), kq2_12, pool12, cudaMemcpyDeviceToHost), "12-cell pool");
        bool scale_ok = true;
        for (int h = 0; h < H && scale_ok; ++h)
            for (int d = 0; d < D && scale_ok; ++d) {
                float ss = 0.0f;
                for (int i = 0; i < n12; ++i) {
                    const float v = Kref[(size_t) i * H * D + (size_t) h * D + d];
                    ss += v * v;
                }
                if (k::q2c_slot(h12.data(), 0, h, d, H, D)->d !=
                    k::f16_from_f32(sqrtf(ss * (1.0f / (float) n12))))
                    scale_ok = false;
            }
        check(scale_ok, "fresh partial group scale: a 12-cell append into an empty group stores the RMS of the 12 "
                        "cells the call has, bit for bit (not of 32)");
    } else {
        std::printf("  fresh partial group scale: NOT RUN - the probe cache has %d cells, fewer than the 12 the\n"
                    "  check appends\n", N);
    }

    // 6. STAGING MOVES WHOLE GROUPS. A q2 group's bytes are group-major (one slot per (group, head, dim)), so a
    //    PAGE is not a byte range: 3 of a group's 8 pages are spread over every dim slot of that group. Staging a
    //    block count that stops mid-group must copy the whole group (else most of its dim slots stay stale in the
    //    staging pool the prompt attention reads), and nothing past it; unstaging [b0, b1) inside a group must
    //    write the whole group back. Pinned on q2k, the format whose unit is set by its q2 K half.
    {
        const int64_t unit = k::kv_stream_unit_blocks(s, k::kKvQ2K);       // pages per 32-cell group
        const size_t kb = (size_t) (2 * unit) * s.n_head_kv * s.page_size * k::kv_q2_bytes_per_head(D);
        const size_t vb = (size_t) (2 * unit) * s.n_head_kv * s.page_size * k::kv_q4_bytes_per_head(D);
        uint8_t* hk = dalloc<uint8_t>(kb);
        uint8_t* hv = dalloc<uint8_t>(vb);
        uint8_t* sk = dalloc<uint8_t>(kb);
        uint8_t* sv = dalloc<uint8_t>(vb);
        std::vector<uint8_t> pk(kb), pv(vb), gk(kb), gv(vb);
        for (size_t i = 0; i < kb; ++i) pk[i] = (uint8_t) (i * 131u + 7u);
        for (size_t i = 0; i < vb; ++i) pv[i] = (uint8_t) (i * 197u + 3u);
        const auto zero_from = [](const std::vector<uint8_t>& b, size_t at) {
            return std::all_of(b.begin() + (std::ptrdiff_t) at, b.end(), [](uint8_t x) { return x == 0; });
        };
        k::KvHostPools host{};
        host.k_q2 = hk;
        host.v_q4 = hv;
        k::QsaAttnPools stage{};
        stage.k_q2 = sk;
        stage.v_q4 = sv;
        // stage 3 pages (mid-group) from a host copy holding a pattern; the staging pool starts zeroed
        ck(cudaMemcpy(hk, pk.data(), kb, cudaMemcpyHostToDevice), "stage: host K");
        ck(cudaMemcpy(hv, pv.data(), vb, cudaMemcpyHostToDevice), "stage: host V");
        k::kv_stage_from_host(stage, host, k::kKvQ2K, 3, s, nullptr);
        ck(cudaDeviceSynchronize(), "stage");
        ck(cudaMemcpy(gk.data(), sk, kb, cudaMemcpyDeviceToHost), "stage: read K");
        ck(cudaMemcpy(gv.data(), sv, vb, cudaMemcpyDeviceToHost), "stage: read V");
        const bool staged = std::memcmp(gk.data(), pk.data(), kb / 2) == 0 &&
                            std::memcmp(gv.data(), pv.data(), vb / 2) == 0 && zero_from(gk, kb / 2) &&
                            zero_from(gv, vb / 2);
        check(staged, "staging 3 of a group's 8 pages copies the WHOLE q2k group (every dim slot, K and V) and "
                      "nothing past it");
        // unstage pages [3, 5) - inside group 0 - from a staging pool holding the pattern into a zeroed host copy
        ck(cudaMemset(hk, 0, kb), "unstage: zero host K");
        ck(cudaMemset(hv, 0, vb), "unstage: zero host V");
        ck(cudaMemcpy(sk, pk.data(), kb, cudaMemcpyHostToDevice), "unstage: stage K");
        ck(cudaMemcpy(sv, pv.data(), vb, cudaMemcpyHostToDevice), "unstage: stage V");
        k::kv_unstage_to_host(stage, host, k::kKvQ2K, 3, 5, s, nullptr);
        ck(cudaDeviceSynchronize(), "unstage");
        ck(cudaMemcpy(gk.data(), hk, kb, cudaMemcpyDeviceToHost), "unstage: read K");
        ck(cudaMemcpy(gv.data(), hv, vb, cudaMemcpyDeviceToHost), "unstage: read V");
        const bool unstaged = std::memcmp(gk.data(), pk.data(), kb / 2) == 0 &&
                              std::memcmp(gv.data(), pv.data(), vb / 2) == 0 && zero_from(gk, kb / 2) &&
                              zero_from(gv, vb / 2);
        check(unstaged, "unstaging pages [3, 5) writes the WHOLE q2k group back to the host copy and nothing past "
                        "it");
    }

    std::printf("\n%s\n", g_fail == 0 ? "kv_q2_parity: all checks passed" : "kv_q2_parity: FAILURES");
    return g_fail == 0 ? 0 : 1;
}
