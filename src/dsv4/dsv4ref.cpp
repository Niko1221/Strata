// src/dsv4/dsv4ref.cpp — DeepSeek V4-Flash host-side reference forward pass (MVP phase 1.3).
//
// Full 43-layer forward on CPU, translating tools/dsv4/oracle.py (authoritative
// semantics) section by section: HC residual stream (4 streams, sinkhorn 20 iters),
// dense MLA attention (hash layers L0/L1), compressor + indexer + sparse attention
// (L2 hash+DSA hybrid, r=4 even layers, r=128 odd layers), MoE with sqrtsoftplus
// routing / tid2eid hash selection / swiglu clamp, and the hc_head + final norm +
// LM head tail. FP32 dequant + naive GEMM; correctness over speed.
//
// Stage gates (golden: /data/research/runs/202610-dsv4-golden/, MANIFEST.md):
//   prefill: token_ids[13] -> logits[13, 129280]  (golden next token 9544)
//   decode:  next_id[0] -> logits[1, 129280]      (golden next tokens 260, 4654)
//
// Usage:
//   dsv4ref <model.gguf> <pack-dir> <out-dir> [--layers N] [--skip-decode] [--threads T]
//
// GGUF tensor layout (pinned by include/strata/dsv4/layout.hpp):
//   GGUF dim0 (fastest/contiguous) = torch dim-1 (input dim for [out, in] mats).
//   blk.L.attn_output_a.weight [4096, 8192] -> rows 8192 (8 groups x 1024), cols 4096.
//   experts.bin per-expert blob 14,155,792 B = gate|up|down roles, each 4,718,592 B
//   ([out, in/64] NVFP4 blocks of 36 B: 4 UE4M3 scale bytes + 32 E2M1 code bytes),
//   plus a 16 B FP32 tail {s_gate, s_up, s_down, 0}. Bit-exact vs official checkpoint.

#include "strata/dsv4/layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using strata::GgufFile;
using strata::TensorInfo;
using strata::dsv4::Dsv4Geometry;

namespace {

constexpr int64_t kMaxSeqLen = 4096;  // oracle cfg.max_seq_len (rope tables, caches)

constexpr int64_t kGoldenPrompt[] = {
    0, 128803, 21750, 260, 15255, 2019, 396, 16803, 1234, 3737, 16, 128804, 128822,
};
constexpr int64_t kPromptTokens = sizeof(kGoldenPrompt) / sizeof(kGoldenPrompt[0]);
constexpr int64_t kGoldenGenerated[3] = {734, 260, 4654};

// ---------------------------------------------------------------------------
// FP16 -> FP32
// ---------------------------------------------------------------------------
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    float v = 0.f;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(mant), -24);
    } else if (exp == 31) {
        v = mant ? std::numeric_limits<float>::quiet_NaN()
                 : std::numeric_limits<float>::infinity();
    } else {
        v = std::ldexp(static_cast<float>(1024 + mant), exp - 25);
    }
    return sign ? -v : v;
}

// ---------------------------------------------------------------------------
// Q8_0 dequantisation (row of n elements: n/32 blocks of fp16 scale + 32 int8)
// ---------------------------------------------------------------------------
void dequant_q8_0(const uint8_t* src, int64_t n, float* dst) {
    const int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; ++b) {
        uint16_t dh;
        std::memcpy(&dh, src + b * 34, 2);
        const float d = fp16_to_fp32(dh);
        const int8_t* qs = reinterpret_cast<const int8_t*>(src + b * 34 + 2);
        float* o = dst + b * 32;
        for (int j = 0; j < 32; ++j) o[j] = static_cast<float>(qs[j]) * d;
    }
}

// ---------------------------------------------------------------------------
// NVFP4 (block_nvfp4) dequantisation: [rows, cols] from 36-byte blocks.
// Block (r, b): 4 UE4M3 scale bytes (unsigned, ggml convention) + 32 E2M1 code
// bytes; sub-block s (16 elems): byte s*8+j low nibble -> elem s*16+j, high
// nibble -> s*16+j+8; value = FP4[nib] * UE4M3(d[s]) * s2.
// Bit-exact vs the official safetensors checkpoint (validated on w1 across
// layers 0/7/42).
// ---------------------------------------------------------------------------
const float kFP4Lut[16] = {0.f,  .5f, 1.f,  1.5f, 2.f,  3.f,  4.f,  6.f,
                           -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

inline float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;  // 0x7f is NaN; ggml yields 0
    const int e = (x >> 3) & 0xf;
    const int m = x & 7;
    if (e == 0) return std::ldexp(static_cast<float>(m), -9);
    return std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

void dequant_nvfp4(const uint8_t* src, float s2, int64_t rows, int64_t cols,
                   float* dst) {
    const int64_t nb = cols / 64;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* brow = src + r * nb * 36;
        float* orow = dst + r * cols;
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = brow + b * 36;
            const float d[4] = {ue4m3_to_fp32(blk[0]) * s2, ue4m3_to_fp32(blk[1]) * s2,
                                ue4m3_to_fp32(blk[2]) * s2, ue4m3_to_fp32(blk[3]) * s2};
            const uint8_t* qs = blk + 4;
            float* o = orow + b * 64;
            for (int s = 0; s < 4; ++s) {
                const float dv = d[s];
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = qs[s * 8 + j];
                    o[s * 16 + j] = kFP4Lut[byte & 0x0f] * dv;
                    o[s * 16 + 8 + j] = kFP4Lut[byte >> 4] * dv;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------
inline float dot_fp32(const float* a, const float* b, int64_t n) {
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        acc4 += a[i + 4] * b[i + 4];
        acc5 += a[i + 5] * b[i + 5];
        acc6 += a[i + 6] * b[i + 6];
        acc7 += a[i + 7] * b[i + 7];
    }
    float acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }

void rmsnorm_rows(float* x, int64_t rows, int64_t cols, const float* w, double eps) {
    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * cols;
        double sq = 0.0;
        for (int64_t i = 0; i < cols; ++i) sq += static_cast<double>(row[i]) * row[i];
        const float inv = static_cast<float>(1.0 / std::sqrt(sq / cols + eps));
        for (int64_t i = 0; i < cols; ++i) row[i] = w[i] * (row[i] * inv);
    }
}

// ---------------------------------------------------------------------------
// GEMM helpers: out[m][r] = dot(x[m], W_row(r)); W is GGUF row-major [rows, cols].
// ---------------------------------------------------------------------------
void gemm_f32(const float* w, int64_t rows, int64_t cols, const float* x, int64_t m,
              float* out, bool accumulate = false) {
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float* wr = w + r * cols;
        for (int64_t t = 0; t < m; ++t) {
            const float v = dot_fp32(x + t * cols, wr, cols);
            float& o = out[t * rows + r];
            o = accumulate ? o + v : v;
        }
    }
}

void gemm_q8(const uint8_t* wq, int64_t rows, int64_t cols, const float* x, int64_t m,
             float* out, bool accumulate = false) {
    const int64_t row_bytes = (cols / 32) * 34;
#pragma omp parallel
    {
        std::vector<float> buf((size_t)cols);
#pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            dequant_q8_0(wq + r * row_bytes, cols, buf.data());
            for (int64_t t = 0; t < m; ++t) {
                const float v = dot_fp32(x + t * cols, buf.data(), cols);
                float& o = out[t * rows + r];
                o = accumulate ? o + v : v;
            }
        }
    }
}

// Grouped GEMM for attn_output_a: row r consumes x[m] + (r / group_rows) * cols,
// where x rows are group_count * cols wide (o.view(s, 8, 4096)).
void gemm_q8_grouped(const uint8_t* wq, int64_t rows, int64_t cols, int64_t group_rows,
                     const float* x, int64_t m, float* out) {
    const int64_t row_bytes = (cols / 32) * 34;
    const int64_t x_row = cols * (rows / group_rows);
#pragma omp parallel
    {
        std::vector<float> buf((size_t)cols);
#pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            dequant_q8_0(wq + r * row_bytes, cols, buf.data());
            const int64_t g = r / group_rows;
            for (int64_t t = 0; t < m; ++t)
                out[t * rows + r] = dot_fp32(x + t * x_row + g * cols, buf.data(), cols);
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF tensor access helpers. GGUF dim0 contiguous -> row-major [out, in].
// ---------------------------------------------------------------------------
const TensorInfo* need_tensor(const GgufFile& g, const std::string& name) {
    const TensorInfo* t = g.find(name);
    if (!t) {
        std::fprintf(stderr, "missing GGUF tensor %s\n", name.c_str());
        std::exit(2);
    }
    return t;
}

const uint8_t* tensor_payload(const GgufFile& g, const TensorInfo& t) {
    const uint64_t bytes = strata::tensor_payload_bytes(t);
    const uint64_t region = g.file_size() - g.data_start();
    if (bytes == 0 || t.offset > region || bytes > region - t.offset) {
        std::fprintf(stderr, "tensor %s payload out of bounds\n", t.name.c_str());
        std::exit(2);
    }
    return g.tensor_data(t);
}

std::vector<float> load_f32_tensor(const GgufFile& g, const std::string& name) {
    const TensorInfo* t = need_tensor(g, name);
    const uint64_t n = t->elements();
    std::vector<float> out(n);
    if (t->type == strata::dsv4::kGgufF32) {
        const uint8_t* p = tensor_payload(g, *t);
        std::memcpy(out.data(), p, n * 4);
    } else if (t->type == strata::dsv4::kGgufBF16) {
        const uint8_t* p = tensor_payload(g, *t);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
        for (uint64_t i = 0; i < n; ++i) {
            const uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            out[i] = f;
        }
    } else {
        std::fprintf(stderr, "tensor %s not F32/BF16 (type %s)\n", name.c_str(),
                     t->type_name());
        std::exit(2);
    }
    return out;
}

const uint8_t* q8_ptr(const GgufFile& g, const std::string& name) {
    const TensorInfo* t = need_tensor(g, name);
    if (t->type != strata::dsv4::kGgufQ8_0) {
        std::fprintf(stderr, "tensor %s not Q8_0\n", name.c_str());
        std::exit(2);
    }
    return tensor_payload(g, *t);
}

std::vector<int32_t> load_i32_tensor(const GgufFile& g, const std::string& name) {
    const TensorInfo* t = need_tensor(g, name);
    const uint64_t n = t->elements();
    std::vector<int32_t> out(n);
    const uint8_t* p = tensor_payload(g, *t);
    std::memcpy(out.data(), p, n * 4);
    return out;
}

// ---------------------------------------------------------------------------
// Stats / dump helpers
// ---------------------------------------------------------------------------
struct Stats {
    int64_t n = 0;
    double mean = 0;
    double std = 0;
    double absmax = 0;
    int64_t nan = 0;
};

Stats stats_of(const float* p, int64_t n) {
    Stats s;
    s.n = n;
    double sum = 0, sumsq = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (std::isnan(v)) {
            ++s.nan;
            continue;
        }
        sum += v;
        sumsq += static_cast<double>(v) * v;
        const double a = std::fabs(static_cast<double>(v));
        if (a > s.absmax) s.absmax = a;
    }
    const int64_t ok = n - s.nan;
    if (ok > 0) {
        s.mean = sum / ok;
        s.std = std::sqrt(std::max(0.0, sumsq / ok - s.mean * s.mean));
    }
    return s;
}

void print_stats(const char* tag, const float* p, int64_t n) {
    const Stats s = stats_of(p, n);
    std::printf("%-24s n=%8" PRId64 " mean=%+.6e std=%.6e absmax=%.6e nan=%" PRId64 "\n",
                tag, s.n, s.mean, s.std, s.absmax, s.nan);
}

void gate_nan(const char* stage, const float* p, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(p[i])) {
            std::fprintf(stderr, "FATAL: NaN at stage %s (index %" PRId64 ")\n", stage, i);
            std::exit(3);
        }
    }
}

void dump_bin(const std::string& path, const float* p, int64_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    std::fwrite(p, sizeof(float), n, f);
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// RoPE tables (oracle precompute_freqs_cis).
// Plain (orig <= 0): theta = base^(-2i/dim), i in [0, dim/2).
// YaRN: correction range computed against original_seq_len (65536, NOT 4096);
// smooth = 1 - linear_ramp(low, high); freqs = f/factor*(1-smooth) + f*smooth.
// ---------------------------------------------------------------------------
struct Rope {
    int64_t half = 32;
    std::vector<float> cosv, sinv;  // [maxpos][half]
};

Rope rope_build(int64_t dim, int64_t maxpos, int64_t orig, double base, double factor,
                double beta_fast, double beta_slow) {
    Rope r;
    r.half = dim / 2;
    std::vector<double> f((size_t)r.half);
    for (int64_t i = 0; i < r.half; ++i)
        f[(size_t)i] = std::pow(base, -static_cast<double>(2 * i) / dim);
    if (orig > 0) {
        auto fcd = [&](double nr) {
            return static_cast<double>(dim) *
                   std::log(static_cast<double>(orig) / (nr * 2.0 * M_PI)) /
                   (2.0 * std::log(base));
        };
        const int64_t low = std::max<int64_t>(0, (int64_t)std::floor(fcd(beta_fast)));
        const int64_t high =
            std::min<int64_t>(dim - 1, (int64_t)std::ceil(fcd(beta_slow)));
        for (int64_t i = 0; i < r.half; ++i) {
            double ramp =
                (static_cast<double>(i) - low) / static_cast<double>(high - low);
            ramp = std::min(1.0, std::max(0.0, ramp));
            const double smooth = 1.0 - ramp;
            f[(size_t)i] = f[(size_t)i] / factor * (1.0 - smooth) + f[(size_t)i] * smooth;
        }
    }
    r.cosv.resize((size_t)maxpos * (size_t)r.half);
    r.sinv.resize((size_t)maxpos * (size_t)r.half);
    for (int64_t p = 0; p < maxpos; ++p)
        for (int64_t i = 0; i < r.half; ++i) {
            const double ang = static_cast<double>(p) * f[(size_t)i];
            r.cosv[(size_t)(p * r.half + i)] = static_cast<float>(std::cos(ang));
            r.sinv[(size_t)(p * r.half + i)] = static_cast<float>(std::sin(ang));
        }
    return r;
}

// Rotate the last rope.half*2 dims of a row of width D at position pos.
void rope_apply_at(float* t, int64_t d, const Rope& r, int64_t pos, bool inverse) {
    float* x = t + (d - 2 * r.half);
    const float* c = r.cosv.data() + pos * r.half;
    const float* s = r.sinv.data() + pos * r.half;
    for (int64_t i = 0; i < r.half; ++i) {
        const float re = x[2 * i], im = x[2 * i + 1];
        if (inverse) {
            x[2 * i] = re * c[i] + im * s[i];
            x[2 * i + 1] = -re * s[i] + im * c[i];
        } else {
            x[2 * i] = re * c[i] - im * s[i];
            x[2 * i + 1] = re * s[i] + im * c[i];
        }
    }
}

// ---------------------------------------------------------------------------
// HC (mHC): hc_pre / hc_post / hc_head, translated from oracle.py.
// fn is [rows][hc*n_embd] row-major; rows = 24 (pre/post/comb) or 4 (head).
// ---------------------------------------------------------------------------
void hc_sinkhorn20(float* comb /*[4][4]*/, double eps) {
    // softmax over dim -1 + eps
    for (int c = 0; c < 4; ++c) {
        float mx = comb[c * 4];
        for (int j = 1; j < 4; ++j) mx = std::max(mx, comb[c * 4 + j]);
        double sum = 0;
        for (int j = 0; j < 4; ++j) {
            comb[c * 4 + j] = std::exp(comb[c * 4 + j] - mx);
            sum += comb[c * 4 + j];
        }
        for (int j = 0; j < 4; ++j)
            comb[c * 4 + j] = static_cast<float>(comb[c * 4 + j] / sum + eps);
    }
    // col-norm, then 19 x (row-norm, col-norm) = 20 col-norms in total
    for (int iter = 0; iter < 20; ++iter) {
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int c = 0; c < 4; ++c) s += comb[c * 4 + j];
            for (int c = 0; c < 4; ++c)
                comb[c * 4 + j] = static_cast<float>(comb[c * 4 + j] / (s + eps));
        }
        if (iter == 19) break;
        for (int c = 0; c < 4; ++c) {
            double s = 0;
            for (int j = 0; j < 4; ++j) s += comb[c * 4 + j];
            for (int j = 0; j < 4; ++j)
                comb[c * 4 + j] = static_cast<float>(comb[c * 4 + j] / (s + eps));
        }
    }
}

// x: [4*n_embd] (fp32 view of h). y: [n_embd] out. post: [4] out. comb: [4*4] out.
void hc_pre_token(const float* x, const float* fn, const float* base, const float* scale,
                  int64_t n_embd, double rms_eps, double hc_eps, float* y, float* post,
                  float* comb) {
    const int64_t n4 = 4 * n_embd;
    double sq = 0;
    for (int64_t i = 0; i < n4; ++i) sq += static_cast<double>(x[i]) * x[i];
    const float rsqrt = static_cast<float>(1.0 / std::sqrt(sq / n4 + rms_eps));
    float mixes[24];
    for (int r = 0; r < 24; ++r) mixes[r] = dot_fp32(fn + r * n4, x, n4) * rsqrt;
    float pre[4];
    for (int c = 0; c < 4; ++c) {
        pre[c] = sigmoidf(mixes[c] * scale[0] + base[c]) + static_cast<float>(hc_eps);
        post[c] = 2.0f * sigmoidf(mixes[4 + c] * scale[1] + base[4 + c]);
    }
    for (int c = 0; c < 4; ++c)
        for (int j = 0; j < 4; ++j)
            comb[c * 4 + j] = mixes[8 + c * 4 + j] * scale[2] + base[8 + c * 4 + j];
    hc_sinkhorn20(comb, hc_eps);
    for (int64_t i = 0; i < n_embd; ++i) {
        float acc = 0;
        for (int c = 0; c < 4; ++c) acc += pre[c] * x[c * n_embd + i];
        y[i] = acc;
    }
}

void hc_head_token(const float* x, const float* fn, const float* base, const float* scale,
                   int64_t n_embd, double rms_eps, double hc_eps, float* y) {
    const int64_t n4 = 4 * n_embd;
    double sq = 0;
    for (int64_t i = 0; i < n4; ++i) sq += static_cast<double>(x[i]) * x[i];
    const float rsqrt = static_cast<float>(1.0 / std::sqrt(sq / n4 + rms_eps));
    for (int64_t i = 0; i < n_embd; ++i) y[i] = 0.f;
    for (int c = 0; c < 4; ++c) {
        const float p =
            sigmoidf(dot_fp32(fn + c * n4, x, n4) * rsqrt * scale[0] + base[c]) +
            static_cast<float>(hc_eps);
        for (int64_t i = 0; i < n_embd; ++i) y[i] += p * x[c * n_embd + i];
    }
}

// h: [4*n_embd] in/out; x: [n_embd] block output.
void hc_post_token(float* h, const float* x, const float* post, const float* comb,
                   int64_t n_embd) {
    std::vector<float> out((size_t)(4 * n_embd));
    for (int c = 0; c < 4; ++c)
        for (int64_t i = 0; i < n_embd; ++i) {
            float acc = post[c] * x[i];
            for (int j = 0; j < 4; ++j) acc += comb[c * 4 + j] * h[j * n_embd + i];
            out[(size_t)(c * n_embd + i)] = acc;
        }
    std::memcpy(h, out.data(), (size_t)(4 * n_embd) * sizeof(float));
}

// ---------------------------------------------------------------------------
// Per-layer weights
// ---------------------------------------------------------------------------
struct CompW {
    int64_t ratio = 0, d = 0, coff = 1;  // W = coff*d
    std::vector<float> wkv;              // [W][n_embd]
    std::vector<float> wgate;            // [W][n_embd]
    std::vector<float> ape;              // [ratio][W]
    std::vector<float> norm;             // [d]
};

struct LayerW {
    std::vector<float> attn_norm, q_norm, kv_norm, ffn_norm;
    std::vector<float> sinks;  // [n_head]
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;
    std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    std::vector<float> gate_w;     // [n_exp][n_embd]
    std::vector<float> gate_b;     // [n_exp] or empty (hash layers)
    std::vector<int32_t> tid2eid;  // [vocab][6] or empty (DSA layers)
    const uint8_t *wq_a = nullptr, *wq_b = nullptr, *wkv = nullptr, *wo_a = nullptr,
                  *wo_b = nullptr, *s_w1 = nullptr, *s_w2 = nullptr, *s_w3 = nullptr;
    CompW comp;  // attention compressor (ratio > 0)
    CompW idx_comp;
    std::vector<float> idx_proj;        // [idx_heads][n_embd]
    const uint8_t* idx_wq_b = nullptr;  // Q8_0 [idx_heads*idx_key_dim][q_lora_rank]
};

CompW load_comp(const GgufFile& gguf, int64_t layer, int64_t ratio, int64_t d,
                int64_t coff, const char* prefix) {
    CompW w;
    w.ratio = ratio;
    w.d = d;
    w.coff = coff;
    char name[128];
    std::snprintf(name, sizeof(name), "blk.%lld.%s_kv.weight", (long long)layer, prefix);
    w.wkv = load_f32_tensor(gguf, name);
    std::snprintf(name, sizeof(name), "blk.%lld.%s_gate.weight", (long long)layer, prefix);
    w.wgate = load_f32_tensor(gguf, name);
    std::snprintf(name, sizeof(name), "blk.%lld.%s_ape.weight", (long long)layer, prefix);
    w.ape = load_f32_tensor(gguf, name);
    std::snprintf(name, sizeof(name), "blk.%lld.%s_norm.weight", (long long)layer, prefix);
    w.norm = load_f32_tensor(gguf, name);
    return w;
}

LayerW load_layer(const GgufFile& gguf, const Dsv4Geometry& g, int64_t layer) {
    (void)g;
    LayerW w;
    char name[128];
    auto f32 = [&](const char* suffix) {
        std::snprintf(name, sizeof(name), "blk.%lld.%s", (long long)layer, suffix);
        return load_f32_tensor(gguf, name);
    };
    auto q8 = [&](const char* suffix) {
        std::snprintf(name, sizeof(name), "blk.%lld.%s", (long long)layer, suffix);
        return q8_ptr(gguf, name);
    };
    w.attn_norm = f32("attn_norm.weight");
    w.q_norm = f32("attn_q_a_norm.weight");
    w.kv_norm = f32("attn_kv_a_norm.weight");
    w.ffn_norm = f32("ffn_norm.weight");
    w.sinks = f32("attn_sinks.weight");
    w.hc_attn_fn = f32("hc_attn_fn.weight");
    w.hc_attn_base = f32("hc_attn_base.weight");
    w.hc_attn_scale = f32("hc_attn_scale.weight");
    w.hc_ffn_fn = f32("hc_ffn_fn.weight");
    w.hc_ffn_base = f32("hc_ffn_base.weight");
    w.hc_ffn_scale = f32("hc_ffn_scale.weight");
    w.gate_w = f32("ffn_gate_inp.weight");
    if (strata::dsv4::dsv4_is_hash_layer(layer)) {
        w.tid2eid = load_i32_tensor(gguf, "blk." + std::to_string(layer) +
                                              ".ffn_gate_tid2eid.weight");
    } else {
        w.gate_b = f32("exp_probs_b.bias");
    }
    w.wq_a = q8("attn_q_a.weight");
    w.wq_b = q8("attn_q_b.weight");
    w.wkv = q8("attn_kv.weight");
    w.wo_a = q8("attn_output_a.weight");
    w.wo_b = q8("attn_output_b.weight");
    w.s_w1 = q8("ffn_gate_shexp.weight");
    w.s_w2 = q8("ffn_down_shexp.weight");
    w.s_w3 = q8("ffn_up_shexp.weight");
    const int64_t ratio = strata::dsv4::dsv4_compress_ratio(layer);
    if (ratio > 0)
        w.comp = load_comp(gguf, layer, ratio, 512, ratio == 4 ? 2 : 1, "attn_compressor");
    if (strata::dsv4::dsv4_has_indexer(layer)) {
        w.idx_comp = load_comp(gguf, layer, 4, 128, 2, "indexer_compressor");
        w.idx_proj = f32("indexer.proj.weight");
        w.idx_wq_b = q8("indexer.attn_q_b.weight");
    }
    return w;
}

// ---------------------------------------------------------------------------
// Runtime state (KV cache + compressor states)
// ---------------------------------------------------------------------------
struct CompState {
    int64_t ratio = 0, d = 0, coff = 1;
    std::vector<float> kv_state;     // [coff*ratio][coff*d], zeros
    std::vector<float> score_state;  // [coff*ratio][coff*d], -inf
    void init(int64_t r, int64_t dd, int64_t cf) {
        ratio = r;
        d = dd;
        coff = cf;
        kv_state.assign((size_t)(cf * r * cf * dd), 0.0f);
        score_state.assign((size_t)(cf * r * cf * dd),
                           -std::numeric_limits<float>::infinity());
    }
};

struct LayerState {
    std::vector<float> kv_cache;   // [(win + max_seq/ratio) * head_dim]
    CompState comp;                // attention compressor
    CompState idx_comp;            // indexer compressor
    std::vector<float> idx_cache;  // [(max_seq/4) * idx_key_dim]
};

// ---------------------------------------------------------------------------
// Compressor (oracle compressor_forward). Returns #compressed tokens produced this
// call and writes them (after rmsnorm + rope at position j*ratio) into kv_out
// (prefill, may be null when none) and out_cache (slot j prefill / pos/ratio decode).
// ---------------------------------------------------------------------------
int64_t compressor_forward(CompState& st, const CompW& w, const float* x, int64_t s,
                           int64_t start_pos, const Rope& rope, double eps,
                           float* out_cache, float* kv_out) {
    const int64_t n_embd = (int64_t)w.wgate.size() / (w.coff * w.d);
    const int64_t ratio = w.ratio, d = w.d, W = w.coff * d;
    std::vector<float> kv((size_t)(s * W)), score((size_t)(s * W));
    gemm_f32(w.wkv.data(), W, n_embd, x, s, kv.data());
    gemm_f32(w.wgate.data(), W, n_embd, x, s, score.data());
    const bool overlap = (w.coff == 2);
    bool should = false;
    std::vector<float> kv_c;  // [ncomp][d]
    std::vector<int64_t> comp_pos, comp_slot;  // [G1] per emitted block: rope pos / out_cache slot
    int64_t ncomp = 0;

    if (start_pos == 0) {
        should = s >= ratio;
        const int64_t rem = s % ratio, cutoff = s - rem;
        const int64_t offset = overlap ? ratio : 0;
        if (overlap && cutoff >= ratio) {
            std::memcpy(st.kv_state.data(), kv.data() + (cutoff - ratio) * W,
                        (size_t)(ratio * W) * sizeof(float));
            for (int64_t i = 0; i < ratio; ++i)
                for (int64_t c = 0; c < W; ++c)
                    st.score_state[(size_t)(i * W + c)] =
                        score[(size_t)((cutoff - ratio + i) * W + c)] +
                        w.ape[(size_t)(i * W + c)];
        }
        if (rem > 0) {
            std::memcpy(st.kv_state.data() + offset * W, kv.data() + cutoff * W,
                        (size_t)(rem * W) * sizeof(float));
            for (int64_t i = 0; i < rem; ++i)
                for (int64_t c = 0; c < W; ++c)
                    st.score_state[(size_t)((offset + i) * W + c)] =
                        score[(size_t)((cutoff + i) * W + c)] + w.ape[(size_t)(i * W + c)];
        }
        if (!should) return 0;
        ncomp = cutoff / ratio;
        kv_c.assign((size_t)(ncomp * d), 0.0f);
        // Pool per feature dd: softmax over pooling rows of (score+ape), then dot kv.
        const int64_t rows = overlap ? 2 * ratio : ratio;
        for (int64_t j = 0; j < ncomp; ++j) {
            for (int64_t dd = 0; dd < d; ++dd) {
                // overlap: i < r -> token j*r+i-r dims [0:d]; i >= r -> token j*r+(i-r)
                // dims [d:2d]; token < 0 -> skipped (-inf). plain: i -> token j*r+i.
                double mx = -std::numeric_limits<double>::infinity();
                for (int64_t i = 0; i < rows; ++i) {
                    int64_t tok, off;
                    if (overlap) {
                        if (i < ratio) {
                            tok = j * ratio + i - ratio;
                            off = dd;
                        } else {
                            tok = j * ratio + (i - ratio);
                            off = d + dd;
                        }
                        if (tok < 0) continue;
                    } else {
                        tok = j * ratio + i;
                        off = dd;
                    }
                    const double v =
                        static_cast<double>(score[(size_t)(tok * W + off)]) +
                        w.ape[(size_t)((tok % ratio) * W + off)];
                    if (v > mx) mx = v;
                }
                double denom = 0;
                for (int64_t i = 0; i < rows; ++i) {
                    int64_t tok, off;
                    if (overlap) {
                        if (i < ratio) {
                            tok = j * ratio + i - ratio;
                            off = dd;
                        } else {
                            tok = j * ratio + (i - ratio);
                            off = d + dd;
                        }
                        if (tok < 0) continue;
                    } else {
                        tok = j * ratio + i;
                        off = dd;
                    }
                    denom += std::exp(static_cast<double>(score[(size_t)(tok * W + off)]) +
                                      w.ape[(size_t)((tok % ratio) * W + off)] - mx);
                }
                double acc = 0;
                for (int64_t i = 0; i < rows; ++i) {
                    int64_t tok, off;
                    if (overlap) {
                        if (i < ratio) {
                            tok = j * ratio + i - ratio;
                            off = dd;
                        } else {
                            tok = j * ratio + (i - ratio);
                            off = d + dd;
                        }
                        if (tok < 0) continue;
                    } else {
                        tok = j * ratio + i;
                        off = dd;
                    }
                    const double p =
                        std::exp(static_cast<double>(score[(size_t)(tok * W + off)]) +
                                 w.ape[(size_t)((tok % ratio) * W + off)] - mx) /
                        denom;
                    acc += p * kv[(size_t)(tok * W + off)];
                }
                kv_c[(size_t)(j * d + dd)] = static_cast<float>(acc);
            }
        }
    } else {
        // [G1] decode, s consecutive rows (positions start_pos .. start_pos+s-1).  Row j runs EXACTLY
        // the single-token body on its own row of kv/score and advances the state, so an R-row pass is
        // bit-identical to R sequential single-token calls.  A block boundary can fall inside the batch
        // (and a row can even complete more than one block), so each completed block records both the
        // rope position it belongs to and the out_cache slot it must be written to.
        kv_c.clear();
        comp_pos.clear();
        comp_slot.clear();
        for (int64_t j = 0; j < s; ++j) {
            const int64_t pos = start_pos + j;
            const float* krow = kv.data() + (size_t)(j * W);
            float* srow = score.data() + (size_t)(j * W);
            should = (pos + 1) % ratio == 0;
            const int64_t rem = pos % ratio;
            for (int64_t c = 0; c < W; ++c) srow[c] += w.ape[(size_t)(rem * W + c)];
            const size_t emit_at = kv_c.size();
            if (overlap) {
                std::memcpy(st.kv_state.data() + (ratio + rem) * W, krow,
                            (size_t)W * sizeof(float));
                std::memcpy(st.score_state.data() + (ratio + rem) * W, srow,
                            (size_t)W * sizeof(float));
                if (should) {
                    kv_c.resize(emit_at + (size_t)d);
                    float* out = kv_c.data() + emit_at;
                    for (int64_t dd = 0; dd < d; ++dd) {
                        // cat([state[:r, :d], state[r:, d:]])
                        double mx = -std::numeric_limits<double>::infinity();
                        for (int64_t i = 0; i < 2 * ratio; ++i) {
                            const double v = i < ratio ? st.score_state[(size_t)(i * W + dd)]
                                                       : st.score_state[(size_t)(i * W + d + dd)];
                            if (v > mx) mx = v;
                        }
                        double denom = 0, acc = 0;
                        for (int64_t i = 0; i < 2 * ratio; ++i) {
                            const double sv = i < ratio
                                                  ? st.score_state[(size_t)(i * W + dd)]
                                                  : st.score_state[(size_t)(i * W + d + dd)];
                            const double kvv = i < ratio ? st.kv_state[(size_t)(i * W + dd)]
                                                         : st.kv_state[(size_t)(i * W + d + dd)];
                            const double p = std::exp(sv - mx);
                            denom += p;
                            acc += p * kvv;
                        }
                        out[dd] = static_cast<float>(acc / denom);
                    }
                    std::memmove(st.kv_state.data(), st.kv_state.data() + ratio * W,
                                 (size_t)(ratio * W) * sizeof(float));
                    std::memmove(st.score_state.data(), st.score_state.data() + ratio * W,
                                 (size_t)(ratio * W) * sizeof(float));
                }
            } else {
                std::memcpy(st.kv_state.data() + rem * W, krow, (size_t)W * sizeof(float));
                std::memcpy(st.score_state.data() + rem * W, srow, (size_t)W * sizeof(float));
                if (should) {
                    kv_c.resize(emit_at + (size_t)d);
                    float* out = kv_c.data() + emit_at;
                    for (int64_t dd = 0; dd < d; ++dd) {
                        double mx = -std::numeric_limits<double>::infinity();
                        for (int64_t i = 0; i < ratio; ++i)
                            mx = std::max(mx, (double)st.score_state[(size_t)(i * W + dd)]);
                        double denom = 0, acc = 0;
                        for (int64_t i = 0; i < ratio; ++i) {
                            const double p = std::exp(st.score_state[(size_t)(i * W + dd)] - mx);
                            denom += p;
                            acc += p * st.kv_state[(size_t)(i * W + dd)];
                        }
                        out[dd] = static_cast<float>(acc / denom);
                    }
                }
            }
            if (should && kv_c.size() != emit_at) {
                comp_pos.push_back(pos + 1 - ratio);
                comp_slot.push_back(pos / ratio);
            }
        }
        ncomp = (int64_t)(kv_c.size() / (size_t)d);
        if (ncomp == 0) return 0;
    }

    // rmsnorm + rope + write out
    for (int64_t j = 0; j < ncomp; ++j) {
        float* row = kv_c.data() + j * d;
        double sq = 0;
        for (int64_t i = 0; i < d; ++i) sq += static_cast<double>(row[i]) * row[i];
        const float inv = static_cast<float>(1.0 / std::sqrt(sq / d + eps));
        for (int64_t i = 0; i < d; ++i) row[i] = w.norm[(size_t)i] * (row[i] * inv);
        const int64_t pos = start_pos == 0 ? j * ratio : comp_pos[(size_t)j];
        rope_apply_at(row, d, rope, pos, false);
        if (out_cache) {
            const int64_t slot = start_pos == 0 ? j : comp_slot[(size_t)j];
            std::memcpy(out_cache + slot * d, row, (size_t)d * sizeof(float));
        }
        if (kv_out) std::memcpy(kv_out + j * d, row, (size_t)d * sizeof(float));
    }
    return ncomp;
}

// ---------------------------------------------------------------------------
// Sparse attention (oracle sparse_attn_ref). q [s][nh][hd]; kv [n][hd];
// topk [s][K] (-1 = masked); sinks [nh]; out [s][nh][hd].
// ---------------------------------------------------------------------------
void sparse_attn(const float* q, const float* kv, const float* sinks, const int32_t* topk,
                 int64_t s, int64_t nh, int64_t hd, int64_t K, float scale, float* out) {
    const float NEG = -std::numeric_limits<float>::infinity();
#pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < nh; ++h) {
        std::vector<float> sc((size_t)K), ev((size_t)hd);
        for (int64_t j = 0; j < s; ++j) {
            const float* qh = q + (j * nh + h) * hd;
            float m = NEG;
            for (int64_t k = 0; k < K; ++k) {
                const int32_t idx = topk[j * K + k];
                sc[(size_t)k] =
                    idx < 0 ? NEG : dot_fp32(qh, kv + (int64_t)idx * hd, hd) * scale;
                if (sc[(size_t)k] > m) m = sc[(size_t)k];
            }
            float denom = std::exp(sinks[h] - m);
            std::fill(ev.begin(), ev.end(), 0.0f);
            for (int64_t k = 0; k < K; ++k) {
                const int32_t idx = topk[j * K + k];
                if (idx < 0) continue;
                const float e = std::exp(sc[(size_t)k] - m);
                denom += e;
                const float* v = kv + (int64_t)idx * hd;
                for (int64_t dd = 0; dd < hd; ++dd) ev[(size_t)dd] += e * v[dd];
            }
            float* o = out + (j * nh + h) * hd;
            for (int64_t dd = 0; dd < hd; ++dd) o[dd] = ev[(size_t)dd] / denom;
        }
    }
}

// ---------------------------------------------------------------------------
// Top-k index helpers (oracle get_window_topk_idxs / get_compress_topk_idxs)
// ---------------------------------------------------------------------------
int64_t window_topk(int64_t win, int64_t s, int64_t start_pos, std::vector<int32_t>& out) {
    if (start_pos > 0) {
        // [G1] multi-row decode: row j is position start_pos + j, so each row gets the ring
        // permutation of ITS position (or the masked short window before the ring is full).  The
        // s == 1 case is the single-row contract verbatim.
        out.resize((size_t)(s * win));
        for (int64_t j = 0; j < s; ++j) {
            const int64_t pos = start_pos + j;
            int32_t* row = out.data() + (size_t)(j * win);
            if (pos >= win - 1) {
                const int64_t sp = pos % win;
                for (int64_t i = 0; i < win; ++i)
                    row[i] = (int32_t)(i < win - sp - 1 ? sp + 1 + i : i - (win - sp - 1));
            } else {
                for (int64_t i = 0; i < win; ++i) row[i] = i <= pos ? (int32_t)i : -1;
            }
        }
        return win;
    }
    const int64_t k = std::min(s, win);
    out.resize((size_t)(s * k));
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < k; ++i) {
            const int64_t v = std::max<int64_t>(j - win + 1, 0) + i;
            out[(size_t)(j * k + i)] = (int32_t)(v > j ? -1 : v);
        }
    return k;
}

int64_t compress_topk(int64_t ratio, int64_t s, int64_t start_pos, int64_t offset,
                      std::vector<int32_t>& out) {
    if (start_pos > 0) {
        // [G1] multi-row decode: row j has n_j = (start_pos+j+1)/ratio compressed blocks available,
        // so the dense row stride is the LARGEST of them and the not-yet-written slots are -1.
        const int64_t kc = (start_pos + s) / ratio;
        out.assign((size_t)(s * kc), -1);
        for (int64_t j = 0; j < s; ++j) {
            const int64_t n = (start_pos + j + 1) / ratio;
            for (int64_t t = 0; t < n; ++t)
                out[(size_t)(j * kc + t)] = (int32_t)(t + offset);
        }
        return kc;
    }
    const int64_t nc = s / ratio;
    out.resize((size_t)(s * nc));
    for (int64_t j = 0; j < s; ++j)
        for (int64_t t = 0; t < nc; ++t)
            out[(size_t)(j * nc + t)] = (int32_t)(t >= (j + 1) / ratio ? -1 : t + offset);
    return nc;
}

// ---------------------------------------------------------------------------
// Indexer (oracle indexer_forward + topk_indices_ref). qr: [s][q_lora_rank] shared
// with attention. Fills topk [s][K] and returns K.
// ---------------------------------------------------------------------------
int64_t indexer_forward(const LayerW& w, LayerState& st, const Dsv4Geometry& g,
                        const float* x, const float* qr, int64_t s, int64_t start_pos,
                        int64_t offset, const Rope& rope, std::vector<int32_t>& topk) {
    const int64_t nh = g.idx_heads;   // 64
    const int64_t hd = g.idx_key_dim; // 128
    const int64_t ratio = 4;
    const int64_t end_pos = start_pos + s;
    // q = qr @ wq_b.T -> [s][nh][hd], rope last 64 dims
    std::vector<float> q((size_t)(s * nh * hd));
    gemm_q8(w.idx_wq_b, nh * hd, g.q_lora_rank, qr, s, q.data());
    for (int64_t j = 0; j < s; ++j)
        for (int64_t h = 0; h < nh; ++h)
            rope_apply_at(q.data() + (j * nh + h) * hd, hd, rope, start_pos + j, false);
    // compressor updates the indexer kv cache
    compressor_forward(st.idx_comp, w.idx_comp, x, s, start_pos, rope, g.rms_eps,
                       st.idx_cache.data(), nullptr);
    const int64_t nc = end_pos / ratio;  // compressed tokens available
    if (nc == 0) {
        topk.clear();
        return 0;
    }
    // weights = (x @ proj.T) * (hd^-0.5 * nh^-0.5)
    std::vector<float> wgt((size_t)(s * nh));
    gemm_f32(w.idx_proj.data(), nh, g.n_embd, x, s, wgt.data());
    const float wscale = static_cast<float>(1.0 / std::sqrt((double)hd) *
                                            (1.0 / std::sqrt((double)nh)));
    for (auto& v : wgt) v *= wscale;
    // index_score[j][t] = sum_h relu(q[j][h] . kv[t]) * wgt[j][h]
    const float* kvc = st.idx_cache.data();
    const float NEG = -std::numeric_limits<float>::infinity();
    const int64_t K = std::min<int64_t>(g.idx_top_k, nc);
    topk.assign((size_t)(s * K), -1);
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < s; ++j) {
        std::vector<float> score((size_t)nc);
        for (int64_t t = 0; t < nc; ++t) {
            float acc = 0;
            const float* kv = kvc + t * hd;
            for (int64_t h = 0; h < nh; ++h) {
                const float dprod = dot_fp32(q.data() + (j * nh + h) * hd, kv, hd);
                acc += std::max(dprod, 0.0f) * wgt[(size_t)(j * nh + h)];
            }
            score[(size_t)t] = acc;
        }
        // [G1] row j sees only the blocks its own position has produced: n_j = (start_pos+j+1)/ratio.
        // The formula also covers prefill (start_pos == 0) and s == 1 (n_j == nc, nothing masked).
        const int64_t n_j = (start_pos + j + 1) / ratio;
        for (int64_t t = n_j; t < nc; ++t) score[(size_t)t] = NEG;
        std::vector<int64_t> order((size_t)nc);
        for (int64_t i = 0; i < nc; ++i) order[(size_t)i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int64_t a, int64_t b) { return score[(size_t)a] > score[(size_t)b]; });
        for (int64_t k = 0; k < K; ++k) {
            int64_t idx = order[(size_t)k];
            if (idx >= n_j)
                idx = -1;
            else
                idx += offset;
            topk[(size_t)(j * K + k)] = (int32_t)idx;
        }
    }
    return K;
}

// ---------------------------------------------------------------------------
// Attention block (oracle attention_forward)
// ---------------------------------------------------------------------------
void attention_forward(int64_t layer, const LayerW& w, LayerState& st,
                       const Dsv4Geometry& g, const float* x, int64_t s, int64_t start_pos,
                       const Rope& rope, float* out) {
    const int64_t N = g.n_embd, QR = g.q_lora_rank, NH = g.n_head, HD = g.head_dim;
    const int64_t WIN = g.sliding_window;
    // qr = rmsnorm(x @ wq_a.T)
    std::vector<float> qr((size_t)(s * QR));
    gemm_q8(w.wq_a, QR, N, x, s, qr.data());
    rmsnorm_rows(qr.data(), s, QR, w.q_norm.data(), g.rms_eps);
    // q = qr @ wq_b.T -> [s][NH][HD]; per-head rms renorm; rope
    std::vector<float> q((size_t)(s * NH * HD));
    gemm_q8(w.wq_b, NH * HD, QR, qr.data(), s, q.data());
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t j = 0; j < s; ++j)
        for (int64_t h = 0; h < NH; ++h) {
            float* qh = q.data() + (j * NH + h) * HD;
            double sq = 0;
            for (int64_t i = 0; i < HD; ++i) sq += (double)qh[i] * qh[i];
            const float inv = (float)(1.0 / std::sqrt(sq / HD + g.rms_eps));
            for (int64_t i = 0; i < HD; ++i) qh[i] *= inv;
            rope_apply_at(qh, HD, rope, start_pos + j, false);
        }
    // kv = rope(rmsnorm(x @ wkv.T))
    std::vector<float> kv_new((size_t)(s * HD));
    gemm_q8(w.wkv, HD, N, x, s, kv_new.data());
    rmsnorm_rows(kv_new.data(), s, HD, w.kv_norm.data(), g.rms_eps);
    for (int64_t j = 0; j < s; ++j)
        rope_apply_at(kv_new.data() + j * HD, HD, rope, start_pos + j, false);

    const int64_t ratio = strata::dsv4::dsv4_compress_ratio(layer);
    const int64_t offset = start_pos == 0 ? s : WIN;
    // topk
    std::vector<int32_t> topk_win, topk_comp;
    const int64_t kw = window_topk(WIN, s, start_pos, topk_win);
    int64_t kc = 0;
    if (ratio == 4) {
        kc = indexer_forward(w, st, g, x, qr.data(), s, start_pos, offset, rope,
                             topk_comp);
    } else if (ratio > 0) {
        kc = compress_topk(ratio, s, start_pos, offset, topk_comp);
    }
    std::vector<int32_t> topk((size_t)(s * (kw + kc)), -1);
    for (int64_t j = 0; j < s; ++j) {
        std::memcpy(topk.data() + j * (kw + kc), topk_win.data() + j * kw,
                    (size_t)kw * sizeof(int32_t));
        if (kc > 0)
            std::memcpy(topk.data() + j * (kw + kc) + kw, topk_comp.data() + j * kc,
                        (size_t)kc * sizeof(int32_t));
    }

    std::vector<float> attn((size_t)(s * NH * HD));
    const float scale = 1.0f / std::sqrt((float)HD);
    std::vector<float> kv_all;
    const float* kv_ptr = nullptr;
    if (start_pos == 0) {
        // write window part of the cache
        if (s <= WIN) {
            std::memcpy(st.kv_cache.data(), kv_new.data(), (size_t)(s * HD) * sizeof(float));
        } else {
            // ring shift (oracle); not exercised by the 13-token golden prefill
            std::memmove(st.kv_cache.data(), st.kv_cache.data() + s * HD,
                         (size_t)((WIN - s) * HD) * sizeof(float));
            std::memcpy(st.kv_cache.data() + (WIN - s) * HD, kv_new.data(),
                        (size_t)(s * HD) * sizeof(float));
        }
        kv_ptr = kv_new.data();
        if (ratio > 0) {
            std::vector<float> kv_comp((size_t)((s / ratio) * HD));
            const int64_t ncomp =
                compressor_forward(st.comp, w.comp, x, s, 0, rope, g.rms_eps,
                                   st.kv_cache.data() + WIN * HD, kv_comp.data());
            if (ncomp > 0) {
                kv_all.assign(kv_new.begin(), kv_new.end());
                kv_all.insert(kv_all.end(), kv_comp.begin(),
                              kv_comp.begin() + ncomp * HD);
                kv_ptr = kv_all.data();
            }
        }
        sparse_attn(q.data(), kv_ptr, w.sinks.data(), topk.data(), s, NH, HD, kw + kc,
                    scale, attn.data());
    } else {
        // [G1] decode, s rows: row j's KV goes to its own ring slot and row j attends with its own
        // window/compress table, so row r only ever sees slots up to its own position (causal across
        // rows).  The compressor runs first because it appends the compressed slots those tables name.
        if (ratio > 0)
            compressor_forward(st.comp, w.comp, x, s, start_pos, rope, g.rms_eps,
                               st.kv_cache.data() + WIN * HD, nullptr);
        for (int64_t j = 0; j < s; ++j) {
            std::memcpy(st.kv_cache.data() + ((start_pos + j) % WIN) * HD,
                        kv_new.data() + j * HD, (size_t)HD * sizeof(float));
            sparse_attn(q.data() + j * NH * HD, st.kv_cache.data(), w.sinks.data(),
                        topk.data() + j * (kw + kc), 1, NH, HD, kw + kc, scale,
                        attn.data() + j * NH * HD);
        }
    }
    // inverse rope on the attention output
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t j = 0; j < s; ++j)
        for (int64_t h = 0; h < NH; ++h)
            rope_apply_at(attn.data() + (j * NH + h) * HD, HD, rope, start_pos + j, true);
    // grouped O projection: o.view(s, 8, 4096) einsum wo_a [8, 1024, 4096] -> [s][8192]
    std::vector<float> o8k((size_t)(s * 8192));
    gemm_q8_grouped(w.wo_a, 8192, 4096, 1024, attn.data(), s, o8k.data());
    gemm_q8(w.wo_b, N, 8192, o8k.data(), s, out);
}

// ---------------------------------------------------------------------------
// Experts: NVFP4 blob from experts.bin (per layer 256 blobs of 14,155,792 B).
// ---------------------------------------------------------------------------
constexpr uint64_t kExpertBlobBytes = 14155792;
constexpr uint64_t kExpertRoleBytes = 4718592;

struct ExpertW {
    std::vector<float> w1, w3, w2;  // [2048][4096], [2048][4096], [4096][2048]
};

bool load_expert(int fd, int64_t layer, int64_t eid, ExpertW& w) {
    const uint64_t off = ((uint64_t)layer * 256 + (uint64_t)eid) * kExpertBlobBytes;
    std::vector<uint8_t> blob(kExpertBlobBytes);
    uint64_t got = 0;
    while (got < kExpertBlobBytes) {
        const ssize_t r =
            pread(fd, blob.data() + got, kExpertBlobBytes - got, (off_t)(off + got));
        if (r <= 0) return false;
        got += (uint64_t)r;
    }
    float tail[4];
    std::memcpy(tail, blob.data() + 3 * kExpertRoleBytes, sizeof(tail));
    w.w1.resize(2048 * 4096);
    w.w3.resize(2048 * 4096);
    w.w2.resize(4096 * 2048);
    dequant_nvfp4(blob.data(), tail[0], 2048, 4096, w.w1.data());
    dequant_nvfp4(blob.data() + kExpertRoleBytes, tail[1], 2048, 4096, w.w3.data());
    dequant_nvfp4(blob.data() + 2 * kExpertRoleBytes, tail[2], 4096, 2048, w.w2.data());
    return true;
}

// out[m] += silu(clamp(x@w1.T, max 10)) * clamp(x@w3.T, +-10) [*wgt] @ w2.T
void expert_apply(const float* w1, const float* w3, const float* w2, int64_t inter,
                  int64_t n_embd, const float* x, const float* wgt, int64_t m, float* y) {
    std::vector<float> gate((size_t)(m * inter)), up((size_t)(m * inter));
    gemm_f32(w1, inter, n_embd, x, m, gate.data());
    gemm_f32(w3, inter, n_embd, x, m, up.data());
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < m; ++t)
        for (int64_t i = 0; i < inter; ++i) {
            const float gv = std::min(gate[(size_t)(t * inter + i)], 10.0f);
            const float uv =
                std::min(std::max(up[(size_t)(t * inter + i)], -10.0f), 10.0f);
            float hv = siluf(gv) * uv;
            if (wgt) hv *= wgt[t];
            gate[(size_t)(t * inter + i)] = hv;
        }
    gemm_f32(w2, n_embd, inter, gate.data(), m, y, true /*accumulate*/);
}

void expert_apply_q8(const uint8_t* w1, const uint8_t* w3, const uint8_t* w2,
                     int64_t inter, int64_t n_embd, const float* x, int64_t m, float* y) {
    std::vector<float> gate((size_t)(m * inter)), up((size_t)(m * inter));
    gemm_q8(w1, inter, n_embd, x, m, gate.data());
    gemm_q8(w3, inter, n_embd, x, m, up.data());
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t t = 0; t < m; ++t)
        for (int64_t i = 0; i < inter; ++i) {
            const float gv = std::min(gate[(size_t)(t * inter + i)], 10.0f);
            const float uv =
                std::min(std::max(up[(size_t)(t * inter + i)], -10.0f), 10.0f);
            gate[(size_t)(t * inter + i)] = siluf(gv) * uv;
        }
    gemm_q8(w2, n_embd, inter, gate.data(), m, y, true /*accumulate*/);
}

// ---------------------------------------------------------------------------
// MoE block (oracle moe_forward)
// ---------------------------------------------------------------------------
void moe_forward(int64_t layer, const LayerW& w, const Dsv4Geometry& g, int efd,
                 const float* x, int64_t s, const int64_t* token_ids, int64_t start_pos,
                 float* y) {
    const int64_t N = g.n_embd, NE = g.n_expert, NK = g.n_expert_used;
    const int64_t I = g.n_ff;
    std::fill(y, y + s * N, 0.0f);
    // scores = sqrt(softplus(x @ gate_w.T))
    std::vector<float> scores((size_t)(s * NE));
    gemm_f32(w.gate_w.data(), NE, N, x, s, scores.data());
    for (auto& v : scores) v = std::sqrt(softplusf(v));
    // selection
    std::vector<int64_t> idx((size_t)(s * NK));
    if (!w.tid2eid.empty()) {
        for (int64_t j = 0; j < s; ++j) {
            const int64_t tok = token_ids[start_pos + j];
            for (int64_t k = 0; k < NK; ++k) idx[(size_t)(j * NK + k)] = w.tid2eid[(size_t)(tok * NK + k)];
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < s; ++j) {
            std::vector<int64_t> order((size_t)NE);
            for (int64_t e = 0; e < NE; ++e) order[(size_t)e] = e;
            const float* sel = scores.data() + j * NE;
            const float* bias = w.gate_b.data();
            std::partial_sort(order.begin(), order.begin() + NK, order.end(),
                              [&](int64_t a, int64_t b) {
                                  const float sa = sel[a] + bias[a], sb = sel[b] + bias[b];
                                  return sa != sb ? sa > sb : a < b;
                              });
            for (int64_t k = 0; k < NK; ++k) idx[(size_t)(j * NK + k)] = order[(size_t)k];
        }
    }
    // weights from the original (pre-bias) scores, normalised * route_scale
    std::vector<float> wgt((size_t)(s * NK));
    for (int64_t j = 0; j < s; ++j) {
        double sum = 0;
        for (int64_t k = 0; k < NK; ++k) sum += scores[(size_t)(j * NE + idx[(size_t)(j * NK + k)])];
        for (int64_t k = 0; k < NK; ++k)
            wgt[(size_t)(j * NK + k)] = static_cast<float>(
                scores[(size_t)(j * NE + idx[(size_t)(j * NK + k)])] / sum * g.route_scale);
    }
    // group tokens per expert
    std::map<int64_t, std::vector<std::pair<int64_t, int64_t>>> by_expert;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t k = 0; k < NK; ++k)
            by_expert[idx[(size_t)(j * NK + k)]].emplace_back(j, k);
    for (auto& [eid, slots] : by_expert) {
        ExpertW ew;
        if (!load_expert(efd, layer, eid, ew)) {
            std::fprintf(stderr, "failed loading expert L%lld E%lld\n", (long long)layer,
                         (long long)eid);
            std::exit(2);
        }
        const int64_t m = (int64_t)slots.size();
        std::vector<float> xs((size_t)(m * N)), ws((size_t)m), ys((size_t)(m * N), 0.0f);
        for (int64_t t = 0; t < m; ++t) {
            std::memcpy(xs.data() + t * N, x + slots[(size_t)t].first * N,
                        (size_t)N * sizeof(float));
            ws[(size_t)t] = wgt[(size_t)(slots[(size_t)t].first * NK + slots[(size_t)t].second)];
        }
        expert_apply(ew.w1.data(), ew.w3.data(), ew.w2.data(), I, N, xs.data(), ws.data(),
                     m, ys.data());
        for (int64_t t = 0; t < m; ++t) {
            float* yr = y + slots[(size_t)t].first * N;
            const float* sr = ys.data() + t * N;
            for (int64_t i = 0; i < N; ++i) yr[i] += sr[i];
        }
    }
    // shared expert (unweighted)
    expert_apply_q8(w.s_w1, w.s_w3, w.s_w2, I, N, x, s, y);
}

// ---------------------------------------------------------------------------
// One full pass over the network.
// ---------------------------------------------------------------------------
struct Roots {
    const uint8_t* embd = nullptr;  // Q8_0 [vocab][n_embd]
    const uint8_t* head = nullptr;  // Q8_0 [vocab][n_embd]
    std::vector<float> output_norm;
    std::vector<float> hc_head_fn, hc_head_base, hc_head_scale;
};

struct PassResult {
    int64_t next_id = -1;
    std::vector<double> layer_ms;
    double total_ms = 0;
};

// all_ids is the FULL token sequence (hash routing looks up ids[start_pos+j]);
// s is this pass's length (13 prefill / 1 decode).
// replay_dir != "": before each layer, overwrite xb/hc_streams with golden inputs
// (<replay_dir>/<pass>_L%02d_in_main.bin / _in_hc.bin) — isolates layer logic from
// upstream accumulated noise (routing-flip vs real-bug discriminator).
PassResult run_pass(const GgufFile& gguf, const Dsv4Geometry& g, int efd,
                    const Roots& roots, std::vector<LayerState>& states,
                    const std::vector<int64_t>& all_ids, int64_t start_pos, int64_t s,
                    const Rope& rope_yarn, const Rope& rope_plain,
                    const std::string& dump_dir, int64_t max_layers,
                    const std::string& replay_dir = "") {
    const int64_t N = g.n_embd;
    const int64_t V = g.n_vocab;
    PassResult res;
    // embed -> 4 identical streams
    std::vector<float> h((size_t)(s * 4 * N));
    {
        std::vector<float> emb((size_t)(s * N));
        const int64_t row_bytes = (N / 32) * 34;
        for (int64_t j = 0; j < s; ++j)
            dequant_q8_0(roots.embd + all_ids[(size_t)(start_pos + j)] * row_bytes, N,
                         emb.data() + j * N);
        for (int64_t j = 0; j < s; ++j)
            for (int c = 0; c < 4; ++c)
                std::memcpy(h.data() + (j * 4 + c) * N, emb.data() + j * N,
                            (size_t)N * sizeof(float));
        gate_nan("embed", h.data(), s * 4 * N);
        dump_bin(dump_dir + "/embed.bin", emb.data(), s * N);
        print_stats("embed", emb.data(), s * N);
    }
    std::vector<float> y((size_t)(s * N)), xb((size_t)(s * N)), post((size_t)(s * 4)),
        comb((size_t)(s * 16));
    const double rms_eps = g.rms_eps, hc_eps = g.hc_eps;
    const int64_t layers = std::min<int64_t>(max_layers, g.n_layers);
    const std::string pass_name = dump_dir.substr(dump_dir.find_last_of('/') + 1);
    for (int64_t layer = 0; layer < layers; ++layer) {
        const auto t0 = std::chrono::steady_clock::now();
        const LayerW w = load_layer(gguf, g, layer);
        if (!replay_dir.empty()) {  // golden-input isolation (see run_pass comment)
            char fn[640];
            std::snprintf(fn, sizeof fn, "%s/%s_L%02lld_in_main.bin", replay_dir.c_str(),
                          pass_name.c_str(), (long long)layer);
            if (FILE* f = std::fopen(fn, "rb")) {
                if (std::fread(xb.data(), sizeof(float), (size_t)(s * N), f) !=
                    (size_t)(s * N)) {
                    std::fprintf(stderr, "replay: short read %s\n", fn);
                    std::exit(2);
                }
                std::fclose(f);
                std::snprintf(fn, sizeof fn, "%s/%s_L%02lld_in_hc.bin", replay_dir.c_str(),
                              pass_name.c_str(), (long long)layer);
                if (FILE* f2 = std::fopen(fn, "rb")) {
                    if (std::fread(h.data(), sizeof(float), (size_t)(s * 4 * N), f2) !=
                        (size_t)(s * 4 * N)) {
                        std::fprintf(stderr, "replay: short read %s\n", fn);
                        std::exit(2);
                    }
                    std::fclose(f2);
                }
            }
        }
        LayerState& st = states[(size_t)layer];
        const int64_t ratio = strata::dsv4::dsv4_compress_ratio(layer);
        const Rope& rope = ratio > 0 ? rope_yarn : rope_plain;
        char tag[64], path[512];
        for (int half = 0; half < 2; ++half) {
            const bool is_attn = half == 0;
            const float* fn = is_attn ? w.hc_attn_fn.data() : w.hc_ffn_fn.data();
            const float* hbase = is_attn ? w.hc_attn_base.data() : w.hc_ffn_base.data();
            const float* hscale = is_attn ? w.hc_attn_scale.data() : w.hc_ffn_scale.data();
            const float* norm = is_attn ? w.attn_norm.data() : w.ffn_norm.data();
            // hc_pre
            for (int64_t j = 0; j < s; ++j)
                hc_pre_token(h.data() + j * 4 * N, fn, hbase, hscale, N, rms_eps, hc_eps,
                             y.data() + j * N, post.data() + j * 4, comb.data() + j * 16);
            // norm
            std::memcpy(xb.data(), y.data(), (size_t)(s * N) * sizeof(float));
            rmsnorm_rows(xb.data(), s, N, norm, rms_eps);
            // block
            std::vector<float> bout((size_t)(s * N));
            if (is_attn) {
                attention_forward(layer, w, st, g, xb.data(), s, start_pos, rope,
                                  bout.data());
            } else {
                moe_forward(layer, w, g, efd, xb.data(), s, all_ids.data(), start_pos,
                            bout.data());
            }
            // hc_post
            for (int64_t j = 0; j < s; ++j)
                hc_post_token(h.data() + j * 4 * N, bout.data() + j * N, post.data() + j * 4,
                              comb.data() + j * 16, N);
            // dump main (mean over the 4 streams) + hc
            std::vector<float> main_out((size_t)(s * N));
            for (int64_t j = 0; j < s; ++j)
                for (int64_t i = 0; i < N; ++i) {
                    double acc = 0;
                    for (int c = 0; c < 4; ++c) acc += h[(size_t)((j * 4 + c) * N + i)];
                    main_out[(size_t)(j * N + i)] = (float)(acc / 4.0);
                }
            std::snprintf(tag, sizeof(tag), "L%02lld_post_%s_main", (long long)layer,
                          is_attn ? "attn" : "moe");
            gate_nan(tag, main_out.data(), s * N);
            std::snprintf(path, sizeof(path), "%s/%s.bin", dump_dir.c_str(), tag);
            dump_bin(path, main_out.data(), s * N);
            std::snprintf(path, sizeof(path), "%s/L%02lld_post_%s_hc.bin",
                          dump_dir.c_str(), (long long)layer, is_attn ? "attn" : "moe");
            dump_bin(path, h.data(), s * 4 * N);
            print_stats(tag, main_out.data(), s * N);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        res.layer_ms.push_back(ms);
        std::printf("  L%02lld done in %.1f ms\n", (long long)layer, ms);
    }
    if (layers == g.n_layers) {
        // hc_head + final norm + logits
        for (int64_t j = 0; j < s; ++j)
            hc_head_token(h.data() + j * 4 * N, roots.hc_head_fn.data(),
                          roots.hc_head_base.data(), roots.hc_head_scale.data(), N,
                          rms_eps, hc_eps, y.data() + j * N);
        gate_nan("hc_head_out", y.data(), s * N);
        dump_bin(dump_dir + "/hc_head_out.bin", y.data(), s * N);
        print_stats("hc_head_out", y.data(), s * N);
        std::memcpy(xb.data(), y.data(), (size_t)(s * N) * sizeof(float));
        rmsnorm_rows(xb.data(), s, N, roots.output_norm.data(), rms_eps);
        gate_nan("final_norm", xb.data(), s * N);
        dump_bin(dump_dir + "/final_norm.bin", xb.data(), s * N);
        print_stats("final_norm", xb.data(), s * N);
        std::vector<float> logits((size_t)(s * V));
        gemm_q8(roots.head, V, N, xb.data(), s, logits.data());
        gate_nan("logits", logits.data(), s * V);
        dump_bin(dump_dir + "/logits.bin", logits.data(), s * V);
        print_stats("logits", logits.data(), s * V);
        const float* last = logits.data() + (s - 1) * V;
        int64_t best = 0;
        for (int64_t i = 1; i < V; ++i)
            if (last[i] > last[best]) best = i;
        res.next_id = best;
        std::printf("argmax(next) = %lld\n", (long long)best);
        if (start_pos > 0 && s > 1) {
            // [G1] multi-row verify: every row's argmax, for the golden next-token chain check.
            for (int64_t j = 0; j < s; ++j) {
                const float* row = logits.data() + j * V;
                int64_t b = 0;
                for (int64_t i = 1; i < V; ++i)
                    if (row[i] > row[b]) b = i;
                std::printf("row %lld argmax = %lld\n", (long long)j, (long long)b);
            }
        }
    }
    return res;
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: dsv4ref <model.gguf> <pack-dir> <out-dir> [--layers N] "
                     "[--skip-decode] [--threads T] [--verify-rows R]\n");
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string pack_dir = argv[2];
    const std::string out_dir = argv[3];
    int64_t max_layers = 43;
    bool skip_decode = false;
    int64_t verify_rows = 0;
    std::string replay_dir;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--layers") && i + 1 < argc)
            max_layers = std::stoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--replay") && i + 1 < argc)
            replay_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--verify-rows") && i + 1 < argc)
            verify_rows = std::stoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--skip-decode"))
            skip_decode = true;
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(std::atoi(argv[++i]));
#else
            ++i;
#endif
        }
    }

    try {
        GgufFile gguf(gguf_path);
        Dsv4Geometry g;
        std::string err;
        if (!strata::dsv4::dsv4_geometry_from_gguf(gguf, g, err)) {
            std::fprintf(stderr, "ERROR: geometry/guard refused %s: %s\n",
                         gguf_path.c_str(), err.c_str());
            return 1;
        }
        if (!strata::dsv4::dsv4_check_gguf_layout(gguf, g, err)) {
            std::fprintf(stderr, "ERROR: layout vs GGUF directory: %s\n", err.c_str());
            return 1;
        }
        std::printf(
            "geometry: layers=%lld embd=%lld heads=%lld q_lora=%lld head_dim=%lld "
            "vocab=%lld rope_theta=%.1f yarn=(base=%.1f orig=%lld factor=%.1f "
            "beta=%.1f/%.1f) experts=%lld topk=%lld inter=%lld win=%lld\n",
            (long long)g.n_layers, (long long)g.n_embd, (long long)g.n_head,
            (long long)g.q_lora_rank, (long long)g.head_dim, (long long)g.n_vocab,
            g.rope_theta, g.compress_rope_theta, (long long)g.yarn_orig_ctx,
            g.yarn_factor, g.yarn_beta_fast, g.yarn_beta_slow, (long long)g.n_expert,
            (long long)g.n_expert_used, (long long)g.n_ff, (long long)g.sliding_window);

        Roots roots;
        roots.embd = q8_ptr(gguf, "token_embd.weight");
        roots.head = q8_ptr(gguf, "output.weight");
        roots.output_norm = load_f32_tensor(gguf, "output_norm.weight");
        roots.hc_head_fn = load_f32_tensor(gguf, "output_hc_fn.weight");
        roots.hc_head_base = load_f32_tensor(gguf, "output_hc_base.weight");
        roots.hc_head_scale = load_f32_tensor(gguf, "output_hc_scale.weight");

        const std::string experts_path = pack_dir + "/experts.bin";
        const int efd = ::open(experts_path.c_str(), O_RDONLY);
        if (efd < 0) {
            std::fprintf(stderr, "cannot open %s\n", experts_path.c_str());
            return 2;
        }

        const Rope rope_plain =
            rope_build(g.n_rot, kMaxSeqLen, 0, g.rope_theta, 0, 0, 0);
        const Rope rope_yarn =
            rope_build(g.n_rot, kMaxSeqLen, g.yarn_orig_ctx, g.compress_rope_theta,
                       g.yarn_factor, g.yarn_beta_fast, g.yarn_beta_slow);

        std::vector<LayerState> states((size_t)g.n_layers);
        for (int64_t layer = 0; layer < g.n_layers; ++layer) {
            const int64_t ratio = strata::dsv4::dsv4_compress_ratio(layer);
            const int64_t n_cache =
                g.sliding_window + (ratio > 0 ? kMaxSeqLen / ratio : 0);
            states[(size_t)layer].kv_cache.assign((size_t)(n_cache * g.head_dim), 0.0f);
            if (ratio > 0) states[(size_t)layer].comp.init(ratio, 512, ratio == 4 ? 2 : 1);
            if (strata::dsv4::dsv4_has_indexer(layer)) {
                states[(size_t)layer].idx_comp.init(4, 128, 2);
                states[(size_t)layer].idx_cache.assign((size_t)((kMaxSeqLen / 4) * 128),
                                                       0.0f);
            }
        }

        std::vector<int64_t> all_ids(kGoldenPrompt, kGoldenPrompt + kPromptTokens);
        const std::string mkdir_cmd = "mkdir -p '" + out_dir + "'";
        if (std::system(mkdir_cmd.c_str()) != 0) return 2;

        const auto run = [&](const std::string& name, int64_t start_pos, int64_t s) {
            const std::string dir = out_dir + "/" + name;
            const std::string mk = "mkdir -p '" + dir + "'";
            if (std::system(mk.c_str()) != 0) std::exit(2);
            std::printf("== pass %s (s=%lld, start_pos=%lld) ==\n", name.c_str(),
                        (long long)s, (long long)start_pos);
            const auto t0 = std::chrono::steady_clock::now();
            PassResult r = run_pass(gguf, g, efd, roots, states, all_ids, start_pos, s,
                                    rope_yarn, rope_plain, dir, max_layers, replay_dir);
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
            r.total_ms = ms;
            std::printf("== pass %s done in %.1f ms, next_id=%lld ==\n", name.c_str(), ms,
                        (long long)r.next_id);
            return r;
        };

        const PassResult pre = run("prefill", 0, kPromptTokens);
        int64_t cur = pre.next_id;
        if (max_layers == g.n_layers && !skip_decode) {
            for (int step = 1; step <= 3; ++step) {
                char pname[32];
                std::snprintf(pname, sizeof(pname), "decode%d", step);
                const int64_t start_pos = (int64_t)all_ids.size();
                all_ids.push_back(cur);
                const PassResult d = run(pname, start_pos, 1);
                cur = d.next_id;
                std::printf("decode%d: generated=%lld golden=%lld %s\n", step,
                            (long long)d.next_id, (long long)kGoldenGenerated[step - 1],
                            d.next_id == kGoldenGenerated[step - 1] ? "MATCH" : "MISMATCH");
            }
            // [G1] multi-row golden: feed the SAME token chain twice — once one-at-a-time (the judge),
            // then as a single r-row pass from the restored state, for every r = 2..verify_rows.  Row i
            // of the r-row pass must reproduce seq(i+1) bit for bit (same kernels, same order).
            if (verify_rows > 1) {
                const int64_t base = (int64_t)all_ids.size();
                const std::vector<LayerState> snap = states;
                std::vector<int64_t> row_ids;
                int64_t next = cur;
                for (int64_t r = 0; r < verify_rows; ++r) {
                    row_ids.push_back(next);
                    char pname[32];
                    std::snprintf(pname, sizeof(pname), "seq%lld", (long long)(r + 1));
                    const int64_t sp = (int64_t)all_ids.size();
                    all_ids.push_back(next);
                    const PassResult d = run(pname, sp, 1);
                    next = d.next_id;
                }
                std::printf("verify: chain =");
                for (int64_t v : row_ids) std::printf(" %lld", (long long)v);
                std::printf(" next=%lld\n", (long long)next);
                for (int64_t r = 2; r <= verify_rows; ++r) {
                    states = snap;
                    all_ids.resize((size_t)base);
                    for (int64_t i = 0; i < r; ++i) all_ids.push_back(row_ids[(size_t)i]);
                    char pname[32];
                    std::snprintf(pname, sizeof(pname), "mrow%lld", (long long)r);
                    const PassResult m = run(pname, base, r);
                    const int64_t want = r < verify_rows ? row_ids[(size_t)r] : next;
                    std::printf("multirow: rows=%lld last_argmax=%lld expected=%lld %s\n",
                                (long long)r, (long long)m.next_id, (long long)want,
                                m.next_id == want ? "MATCH" : "MISMATCH");
                }
            }
        }
        ::close(efd);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
