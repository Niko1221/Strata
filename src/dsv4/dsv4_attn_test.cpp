// src/dsv4/dsv4_attn_test.cpp - standalone golden-comparison test for the dsv4 S3 dense MLA
// attention block (src/dsv4/dsv4_attn.cu, hash layers L0/L1).
//
// Flow per layer (mirrors src/dsv4/dsv4ref.cpp run_pass, attention half):
//   1. load the golden layer-entry representation from the replay dir
//      (prefill_L%02d_in_main.bin [13][4096], prefill_L%02d_in_hc.bin [13][4][4096])
//   2. host: hc_pre_token -> y/post/comb; xb = rmsnorm(y, attn_norm)   (verbatim dsv4ref host code)
//   3. GPU: dsv4_attn_forward(xb) -> bout                              (the S3 kernels under test)
//   4. host: hc_post_token; main = mean of the 4 streams; dump
//      L%02d_post_attn_{main,hc}.bin for tools/dsv4/compare_layers.py vs the golden .npy tree.
//   5. decode self-check: the same 13 tokens fed one-by-one (start_pos = j, sliding-window cache)
//      must reproduce the prefill attention block output row for token j.
//
// The S0 engine loader (src/dsv4/dsv4_engine.cu) is not reused: it is not exported through
// include/strata/dsv4/engine.hpp and the file is owned by the parallel S2 effort, so the GGUF
// reads this test needs (attention weights only, no experts.bin) are done directly through the
// header-only GgufFile reader, exactly like src/dsv4/dsv4ref.cpp.
//
// usage: strata-dsv4-attn-test <model.gguf> <replay-dir> <out-dir> [--layers 0,1] [--no-decode-check]

#include "strata/dsv4/attn.hpp"
#include "strata/dsv4/layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::GgufFile;
using strata::TensorInfo;

namespace {

constexpr int64_t kNEmbd = 4096;
constexpr int64_t kQRank = 1024;
constexpr int64_t kNHead = 64;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kWin = 128;
constexpr int64_t kTokens = 13;
constexpr int64_t kMaxSeqLen = 4096;

#define CK(x)                                                                            \
    do {                                                                                 \
        cudaError_t err_ = (x);                                                          \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                                            \
            std::exit(3);                                                                \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// GGUF tensor access (verbatim dsv4ref helpers)
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
        std::memcpy(out.data(), tensor_payload(g, *t), n * 4);
    } else if (t->type == strata::dsv4::kGgufBF16) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(tensor_payload(g, *t));
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

std::vector<uint8_t> load_q8_tensor(const GgufFile& g, const std::string& name) {
    const TensorInfo* t = need_tensor(g, name);
    if (t->type != strata::dsv4::kGgufQ8_0) {
        std::fprintf(stderr, "tensor %s not Q8_0\n", name.c_str());
        std::exit(2);
    }
    const uint64_t bytes = strata::tensor_payload_bytes(*t);
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), tensor_payload(g, *t), bytes);
    return out;
}

// ---------------------------------------------------------------------------
// Host math (verbatim dsv4ref): dot / rmsnorm / rope build / hc_pre / hc_post
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

inline float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void rmsnorm_rows(float* x, int64_t rows, int64_t cols, const float* w, double eps) {
    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * cols;
        double sq = 0.0;
        for (int64_t i = 0; i < cols; ++i) sq += static_cast<double>(row[i]) * row[i];
        const float inv = static_cast<float>(1.0 / std::sqrt(sq / cols + eps));
        for (int64_t i = 0; i < cols; ++i) row[i] = w[i] * (row[i] * inv);
    }
}

struct Rope {
    int64_t half = 32;
    std::vector<float> cosv, sinv;
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
        const int64_t high = std::min<int64_t>(dim - 1, (int64_t)std::ceil(fcd(beta_slow)));
        for (int64_t i = 0; i < r.half; ++i) {
            double ramp = (static_cast<double>(i) - low) / static_cast<double>(high - low);
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

void hc_sinkhorn20(float* comb, double eps) {
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
        pre[c] = sigmoidf_(mixes[c] * scale[0] + base[c]) + static_cast<float>(hc_eps);
        post[c] = 2.0f * sigmoidf_(mixes[4 + c] * scale[1] + base[4 + c]);
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
// IO helpers
// ---------------------------------------------------------------------------
std::vector<float> read_bin(const std::string& path, int64_t n) {
    std::vector<float> out((size_t)n);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    if (std::fread(out.data(), sizeof(float), (size_t)n, f) != (size_t)n) {
        std::fprintf(stderr, "short read %s\n", path.c_str());
        std::exit(2);
    }
    std::fclose(f);
    return out;
}

void dump_bin(const std::string& path, const float* p, int64_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(2);
    }
    std::fwrite(p, sizeof(float), (size_t)n, f);
    std::fclose(f);
}

double cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

template <typename T>
T* dev_upload(const std::vector<T>& v) {
    T* p = nullptr;
    CK(cudaMalloc(&p, v.size() * sizeof(T)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

struct DevWeights {
    uint8_t *wq_a, *wq_b, *wkv, *wo_a, *wo_b;
    float *q_norm, *kv_norm, *sinks;
    void free() {
        cudaFree(wq_a); cudaFree(wq_b); cudaFree(wkv); cudaFree(wo_a); cudaFree(wo_b);
        cudaFree(q_norm); cudaFree(kv_norm); cudaFree(sinks);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: strata-dsv4-attn-test <model.gguf> <replay-dir> <out-dir> "
                     "[--layers 0,1] [--no-decode-check]\n");
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string replay_dir = argv[2];
    const std::string out_dir = argv[3];
    std::vector<int> layers = {0, 1};
    bool decode_check = true;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) {
            layers.clear();
            const std::string v = argv[++i];
            size_t pos = 0;
            while (pos <= v.size()) {
                const size_t comma = v.find(',', pos);
                layers.push_back(std::stoi(v.substr(pos, comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (!std::strcmp(argv[i], "--no-decode-check")) {
            decode_check = false;
        }
    }

    CK(cudaSetDevice(0));
    CK(cudaFree(0));  // establish the context before timing/locks matter

    GgufFile gguf(gguf_path);
    strata::dsv4::Dsv4Geometry g;
    std::string err;
    if (!strata::dsv4::dsv4_geometry_from_gguf(gguf, g, err)) {
        std::fprintf(stderr, "geometry guard refused %s: %s\n", gguf_path.c_str(), err.c_str());
        return 1;
    }

    // plain rope tables for the hash layers (theta 1e4, no YaRN), uploaded once
    const Rope rope = rope_build(g.n_rot, kMaxSeqLen, 0, g.rope_theta, 0, 0, 0);
    float* d_cos = dev_upload(rope.cosv);
    float* d_sin = dev_upload(rope.sinv);
    strata::dsv4::Dsv4RopeTables rope_dev;
    rope_dev.cos = d_cos;
    rope_dev.sin = d_sin;
    rope_dev.half = (int)rope.half;

    strata::dsv4::Dsv4AttnWorkspace ws;
    if (!strata::dsv4::dsv4_attn_workspace_init(ws, (int)kTokens, (int)kWin)) return 3;
    float* d_kv_cache = nullptr;
    CK(cudaMalloc(&d_kv_cache, (size_t)kWin * kHeadDim * 4));
    float* d_xb = nullptr;
    CK(cudaMalloc(&d_xb, (size_t)kTokens * kNEmbd * 4));
    float* d_bout = nullptr;
    CK(cudaMalloc(&d_bout, (size_t)kTokens * kNEmbd * 4));

    const std::string mkdir_cmd = "mkdir -p '" + out_dir + "/prefill'";
    if (std::system(mkdir_cmd.c_str()) != 0) return 2;

    std::vector<float> y((size_t)kNEmbd), post(4), comb(16);
    int rc = 0;
    for (const int layer : layers) {
        char nm[128];
        auto f32 = [&](const char* suffix) {
            std::snprintf(nm, sizeof(nm), "blk.%d.%s", layer, suffix);
            return load_f32_tensor(gguf, nm);
        };
        auto q8 = [&](const char* suffix) {
            std::snprintf(nm, sizeof(nm), "blk.%d.%s", layer, suffix);
            return load_q8_tensor(gguf, nm);
        };
        std::vector<float> attn_norm = f32("attn_norm.weight");
        std::vector<float> hc_fn = f32("hc_attn_fn.weight");
        std::vector<float> hc_base = f32("hc_attn_base.weight");
        std::vector<float> hc_scale = f32("hc_attn_scale.weight");
        DevWeights dw;
        dw.wq_a = dev_upload(q8("attn_q_a.weight"));
        dw.wq_b = dev_upload(q8("attn_q_b.weight"));
        dw.wkv = dev_upload(q8("attn_kv.weight"));
        dw.wo_a = dev_upload(q8("attn_output_a.weight"));
        dw.wo_b = dev_upload(q8("attn_output_b.weight"));
        dw.q_norm = dev_upload(f32("attn_q_a_norm.weight"));
        dw.kv_norm = dev_upload(f32("attn_kv_a_norm.weight"));
        dw.sinks = dev_upload(f32("attn_sinks.weight"));
        strata::dsv4::Dsv4AttnWeights w;
        w.wq_a = dw.wq_a; w.wq_b = dw.wq_b; w.wkv = dw.wkv;
        w.wo_a = dw.wo_a; w.wo_b = dw.wo_b;
        w.q_norm = dw.q_norm; w.kv_norm = dw.kv_norm; w.sinks = dw.sinks;

        // golden layer entry (replay representation)
        char fn[640];
        std::snprintf(fn, sizeof(fn), "%s/prefill_L%02d_in_main.bin", replay_dir.c_str(), layer);
        const std::vector<float> in_main = read_bin(fn, kTokens * kNEmbd);
        std::snprintf(fn, sizeof(fn), "%s/prefill_L%02d_in_hc.bin", replay_dir.c_str(), layer);
        const std::vector<float> in_hc = read_bin(fn, kTokens * 4 * kNEmbd);
        // sanity: the replay main must equal the mean of the 4 hc streams
        {
            std::vector<float> mean((size_t)kTokens * kNEmbd);
            for (int64_t j = 0; j < kTokens; ++j)
                for (int64_t i = 0; i < kNEmbd; ++i) {
                    double acc = 0;
                    for (int c = 0; c < 4; ++c) acc += in_hc[(size_t)((j * 4 + c) * kNEmbd + i)];
                    mean[(size_t)(j * kNEmbd + i)] = (float)(acc / 4.0);
                }
            const double cos = cosine(mean.data(), in_main.data(), kTokens * kNEmbd);
            std::printf("L%02d entry: cos(mean(hc), replay main) = %.8f\n", layer, cos);
            if (cos < 0.9999) {
                std::fprintf(stderr, "L%02d replay entry inconsistent\n", layer);
                rc = 1;
            }
        }

        // attention input: hc_pre -> rmsnorm(attn_norm)  (host, verbatim dsv4ref)
        std::vector<float> xb((size_t)kTokens * kNEmbd);
        std::vector<float> posts((size_t)kTokens * 4), combs((size_t)kTokens * 16);
        for (int64_t j = 0; j < kTokens; ++j) {
            hc_pre_token(in_hc.data() + j * 4 * kNEmbd, hc_fn.data(), hc_base.data(),
                         hc_scale.data(), kNEmbd, g.rms_eps, g.hc_eps, y.data(),
                         posts.data() + j * 4, combs.data() + j * 16);
            std::memcpy(xb.data() + j * kNEmbd, y.data(), (size_t)kNEmbd * 4);
        }
        rmsnorm_rows(xb.data(), kTokens, kNEmbd, attn_norm.data(), g.rms_eps);

        // GPU attention block
        CK(cudaMemcpy(d_xb, xb.data(), (size_t)kTokens * kNEmbd * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_kv_cache, 0, (size_t)kWin * kHeadDim * 4));
        strata::dsv4::dsv4_attn_forward(w, rope_dev, ws, d_xb, d_bout, d_kv_cache,
                                        (int)kTokens, 0, (int)kWin, nullptr);
        std::vector<float> bout((size_t)kTokens * kNEmbd);
        CK(cudaMemcpy(bout.data(), d_bout, (size_t)kTokens * kNEmbd * 4,
                      cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < kTokens * kNEmbd; ++i)
            if (std::isnan(bout[(size_t)i])) {
                std::fprintf(stderr, "L%02d NaN in attention output at %" PRId64 "\n", layer, i);
                rc = 1;
                break;
            }

        // hc_post + dumps (host)
        std::vector<float> h = in_hc;
        for (int64_t j = 0; j < kTokens; ++j)
            hc_post_token(h.data() + j * 4 * kNEmbd, bout.data() + j * kNEmbd,
                          posts.data() + j * 4, combs.data() + j * 16, kNEmbd);
        std::vector<float> main_out((size_t)kTokens * kNEmbd);
        for (int64_t j = 0; j < kTokens; ++j)
            for (int64_t i = 0; i < kNEmbd; ++i) {
                double acc = 0;
                for (int c = 0; c < 4; ++c) acc += h[(size_t)((j * 4 + c) * kNEmbd + i)];
                main_out[(size_t)(j * kNEmbd + i)] = (float)(acc / 4.0);
            }
        char path[640];
        std::snprintf(path, sizeof(path), "%s/prefill/L%02d_post_attn_main.bin",
                      out_dir.c_str(), layer);
        dump_bin(path, main_out.data(), kTokens * kNEmbd);
        std::snprintf(path, sizeof(path), "%s/prefill/L%02d_post_attn_hc.bin", out_dir.c_str(),
                      layer);
        dump_bin(path, h.data(), kTokens * 4 * kNEmbd);
        std::printf("L%02d prefill attention done, dumps written\n", layer);

        // decode self-check: same 13 tokens one at a time over the sliding-window cache
        if (decode_check) {
            CK(cudaMemset(d_kv_cache, 0, (size_t)kWin * kHeadDim * 4));
            double worst_cos = 1.0, worst_amax = 0.0;
            for (int64_t j = 0; j < kTokens; ++j) {
                hc_pre_token(in_hc.data() + j * 4 * kNEmbd, hc_fn.data(), hc_base.data(),
                             hc_scale.data(), kNEmbd, g.rms_eps, g.hc_eps, y.data(),
                             posts.data(), comb.data());
                std::vector<float> xr(y.begin(), y.end());  // one token's attention input
                rmsnorm_rows(xr.data(), 1, kNEmbd, attn_norm.data(), g.rms_eps);
                CK(cudaMemcpy(d_xb, xr.data(), (size_t)kNEmbd * 4, cudaMemcpyHostToDevice));
                strata::dsv4::dsv4_attn_forward(w, rope_dev, ws, d_xb, d_bout, d_kv_cache, 1,
                                                j, (int)kWin, nullptr);
                std::vector<float> br((size_t)kNEmbd);
                CK(cudaMemcpy(br.data(), d_bout, (size_t)kNEmbd * 4, cudaMemcpyDeviceToHost));
                const double cos = cosine(br.data(), bout.data() + j * kNEmbd, kNEmbd);
                double amax = 0;
                for (int64_t i = 0; i < kNEmbd; ++i)
                    amax = std::max(amax, (double)std::fabs(br[(size_t)i] -
                                                            bout[(size_t)(j * kNEmbd + i)]));
                worst_cos = std::min(worst_cos, cos);
                worst_amax = std::max(worst_amax, amax);
            }
            std::printf("L%02d decode-vs-prefill: worst cos %.8f, worst absmax_d %.3e\n", layer,
                        worst_cos, worst_amax);
            if (worst_cos < 0.99999) {
                std::fprintf(stderr, "L%02d decode path diverges from prefill\n", layer);
                rc = 1;
            }
        }
        dw.free();
    }

    cudaFree(d_kv_cache);
    cudaFree(d_xb);
    cudaFree(d_bout);
    cudaFree(d_cos);
    cudaFree(d_sin);
    strata::dsv4::dsv4_attn_workspace_free(ws);
    std::printf("strata-dsv4-attn-test: %s\n", rc == 0 ? "OK" : "PROBLEMS");
    return rc;
}
