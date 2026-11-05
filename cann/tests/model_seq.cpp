// cann/tests/model_seq.cpp - the whole model over a token sequence.
//
// model_forward.cpp proves one token matches llama.cpp. A real engine also has to carry state:
// the recurrent layers keep a short-conv history and a 128x128 matrix per value head, and the
// attention layers keep a KV cache. This program walks a prompt of T tokens through the 40 layers
// with those states live on the device, then prints the next-token distribution for the last
// position, which is what llama.cpp reports for the same prompt.
//
//   model_seq <model.gguf> <tokens comma separated>
#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <thread>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gdn.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"
#include "strata/cann/tensor_load.hpp"

using namespace strata::cann;

namespace {

constexpr int64_t kHidden = 2048;
constexpr int64_t kHeadDimSsm = 128;
constexpr int64_t kKHeads = 16;
constexpr int64_t kVHeads = 32;
constexpr int64_t kConvK = 4;
constexpr int64_t kKeyDim = kHeadDimSsm * kKHeads;
constexpr int64_t kValueDim = kHeadDimSsm * kVHeads;
constexpr int64_t kQkvDim = kKeyDim * 2 + kValueDim;
constexpr int64_t kHeadDimAttn = 256;
constexpr int64_t kHeads = 16;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kRot = 64;
constexpr float kFreqBase = 1e7f;
constexpr float kRmsEps = 1e-6f;
constexpr int64_t kQDim = kHeads * kHeadDimAttn * 2;
constexpr int64_t kKvDim = kKvHeads * kHeadDimAttn;
constexpr int64_t kExpertInter = 512;
constexpr int64_t kTopK = 8;

std::vector<float> round_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    std::vector<float> out(v.size());
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void upload(const std::vector<float> & v, Buffer & b) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), h.size());
    b.copy_from_host(h.data(), h.size() * 2);
}

std::vector<float> to_host(Stream & stream, const Buffer & b, int64_t n) {
    std::vector<uint16_t> h((size_t) n);
    b.copy_to_host(h.data(), h.size() * 2);
    stream.synchronize();
    std::vector<float> out((size_t) n);
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void gemv(Stream & stream, const std::vector<float> & w, int64_t m, int64_t k, const Tensor & x, Buffer & out) {
    Buffer bw(w.size() * 2);
    upload(w, bw);
    Tensor tw{ bw.data(), DType::F16, k, m };
    Tensor to{ out.data(), DType::F16, 1, m };
    matmul(stream, tw, x, to);
}

std::vector<float> need(const Gguf & g, const std::string & name) {
    const GgufTensor * t = g.find(name);
    if (!t) {
        throw std::runtime_error("model: missing tensor " + name);
    }
    return round_f16(load_tensor(g, *t));
}

const GgufTensor * need_t(const Gguf & g, const std::string & name) {
    const GgufTensor * t = g.find(name);
    if (!t) {
        throw std::runtime_error("model: missing tensor " + name);
    }
    return t;
}

std::vector<float> expert_slab(const Gguf & g, const GgufTensor & t, int expert, int64_t rows, int64_t cols) {
    const uint64_t n = (uint64_t) rows * cols;
    const int block_bytes = quant_type_size((int) t.type);
    const int block_elems = quant_block_elements((int) t.type);
    const uint64_t bytes = n / (uint64_t) block_elems * (uint64_t) block_bytes;
    const std::vector<uint8_t> raw = g.tensor_bytes(t, (uint64_t) expert * bytes, bytes);
    std::vector<float> out((size_t) n);
    if (!dequantize((int) t.type, raw.data(), out.data(), (int64_t) n)) {
        throw std::runtime_error("model: dequantize failed");
    }
    return out;
}

// Dequantization is the real host cost: a fresh expert slab is a few megabytes and the decoder is
// a scalar loop, so one token spent seconds in it. The formats are blockwise, so the work splits
// cleanly across threads by block range.
std::vector<float> dequantize_only(int type, const uint8_t * src, int64_t n) {
    std::vector<float> out((size_t) n);
    if (!dequantize(type, src, out.data(), n)) {
        throw std::runtime_error("model: dequantize failed");
    }
    return out;
}

// Device-resident weights. Every gemv used to dequantize on the host and upload again, which is
// why a token cost 33 s. Here each tensor is dequantized once, uploaded once, and kept in HBM.
// Expert slabs are cached separately with an LRU: 256 experts per layer will not fit.
class WeightCache {
public:
    explicit WeightCache(const Gguf & g) : g_(g) {}

    const Buffer & full(const std::string & name) {
        auto it = full_.find(name);
        if (it != full_.end()) {
            return *it->second;
        }
        const GgufTensor * t = g_.find(name);
        if (!t) {
            throw std::runtime_error("model: missing tensor " + name);
        }
        const int64_t n_elems = (int64_t) t->dims[0] * (int64_t) t->dims[1];
        const int be = quant_block_elements((int) t->type);
        const int bb = quant_type_size((int) t->type);
        std::vector<float> w;
        if (be > 0 && bb > 0 && n_elems % be == 0) {
            const uint64_t nbytes = (uint64_t) (n_elems / be) * (uint64_t) bb;
            const std::vector<uint8_t> bytes = g_.tensor_bytes(*t, 0, nbytes);
            w = dequantize_only((int) t->type, bytes.data(), n_elems);
        } else {
            w = load_tensor(g_, *t);
        }
        auto buf = std::make_unique<Buffer>(w.size() * 2);
        upload(w, *buf);
        Buffer * raw = buf.get();
        full_.emplace(name, std::move(buf));
        bytes_ += (int64_t) w.size() * 2;
        return *raw;
    }

    inline static double t_read = 0, t_deq = 0, t_up = 0;
    inline static long long n_read = 0;
    static void print_slab_stats() {
        if (n_read > 0) {
            printf("slabs: %lld new, disk %.1f s, dequant %.1f s, upload %.1f s\n", n_read, t_read, t_deq, t_up);
        }
    }

    const Buffer & slab(const std::string & tensor, int expert, int64_t rows, int64_t cols, int il) {
        const std::string key = tensor + "#" + std::to_string(expert);
        auto it = slab_.find(key);
        if (it != slab_.end()) {
            return *it->second;
        }
        std::vector<float> w = decode_only(tensor, expert, rows, cols);
        return store(tensor, expert, std::move(w));
    }

    // The eight selected experts share one input, so a layer needs up to 24 slabs at once. Decoding
    // them concurrently is what pays: a slab runs at ~150M values/s single-threaded, while spawning
    // a thread pool inside each slab cost more than the decode (22 s of a 37 s run).
    void prefetch(const std::string & tensor, const std::vector<int> & experts, int64_t rows, int64_t cols,
                  int il) {
        std::vector<int> missing;
        for (int e : experts) {
            if (slab_.find(tensor + "#" + std::to_string(e)) == slab_.end()) {
                missing.push_back(e);
            }
        }
        if (missing.size() <= 1) {
            return;
        }
        std::vector<std::vector<float>> decoded(missing.size());
        std::vector<std::thread> workers;
        workers.reserve(missing.size());
        for (size_t i = 0; i < missing.size(); ++i) {
            workers.emplace_back([&, i] { decoded[i] = decode_only(tensor, missing[i], rows, cols); });
        }
        for (auto & w : workers) {
            w.join();
        }
        for (size_t i = 0; i < missing.size(); ++i) {
            if (slab_.find(tensor + "#" + std::to_string(missing[i])) == slab_.end()) {
                store(tensor, missing[i], std::move(decoded[i]));
            }
        }
        (void) il;
    }

    int64_t bytes() const { return bytes_; }

private:
    std::vector<float> decode_only(const std::string & tensor, int expert, int64_t rows, int64_t cols) {
        const GgufTensor * t = g_.find(tensor);
        if (!t) {
            throw std::runtime_error("model: missing tensor " + tensor);
        }
        const uint64_t n = (uint64_t) rows * (uint64_t) cols;
        const int bb = quant_type_size((int) t->type);
        const int be = quant_block_elements((int) t->type);
        if (be <= 0 || bb <= 0 || n % (uint64_t) be != 0) {
            return expert_slab(g_, *t, expert, rows, cols);
        }
        const uint64_t slab_bytes = n / (uint64_t) be * (uint64_t) bb;
        const auto r0 = std::chrono::steady_clock::now();
        const std::vector<uint8_t> bytes = g_.tensor_bytes(*t, (uint64_t) expert * slab_bytes, slab_bytes);
        const auto r1 = std::chrono::steady_clock::now();
        std::vector<float> w = dequantize_only((int) t->type, bytes.data(), (int64_t) n);
        const auto r2 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(stats_mutex());
            t_read += std::chrono::duration<double>(r1 - r0).count();
            t_deq += std::chrono::duration<double>(r2 - r1).count();
            n_read += 1;
        }
        return w;
    }

    static std::mutex & stats_mutex() {
        static std::mutex m;
        return m;
    }

    const Buffer & store(const std::string & tensor, int expert, std::vector<float> w) {
        const std::string key = tensor + "#" + std::to_string(expert);
        const auto u0 = std::chrono::steady_clock::now();
        auto buf = std::make_unique<Buffer>(w.size() * 2);
        upload(w, *buf);
        t_up += std::chrono::duration<double>(std::chrono::steady_clock::now() - u0).count();
        Buffer * raw = buf.get();
        slab_.emplace(key, std::move(buf));
        slab_order_.push_back(key);
        bytes_ += (int64_t) w.size() * 2;
        while (bytes_ > kBudget && !slab_order_.empty()) {
            const std::string victim = slab_order_.front();
            slab_order_.pop_front();
            auto vit = slab_.find(victim);
            if (vit != slab_.end()) {
                bytes_ -= (int64_t) vit->second->size();
                slab_.erase(vit);
            }
        }
        return *raw;
    }

    // All 256 experts of all 40 layers are about 41 GB as fp16, so a budget above that lets a
    // long generation keep everything resident and stop paying the upload. The card has 64 GB.
    static constexpr int64_t kBudget = 40LL << 30;
    const Gguf & g_;
    std::unordered_map<std::string, std::unique_ptr<Buffer>> full_;
    std::unordered_map<std::string, std::unique_ptr<Buffer>> slab_;
    std::deque<std::string> slab_order_;
    int64_t bytes_ = 0;
};

void gemv_w(Stream & stream, WeightCache & cache, const std::string & name, int64_t m, int64_t k, const Tensor & x,
            Buffer & out) {
    const Buffer & bw = cache.full(name);
    Tensor tw{ bw.data(), DType::F16, k, m };
    Tensor to{ out.data(), DType::F16, 1, m };
    matmul(stream, tw, x, to);
}

// per-layer recurrent state, allocated once and updated in place
struct GdnState {
    std::vector<std::unique_ptr<Buffer>> conv; // kConvK-1 past qkv vectors, newest first
    std::unique_ptr<Buffer> ssm;               // [kVHeads][128][128], contiguous so the batched op can take it
};

struct AttnState {
    std::unique_ptr<Buffer> k; // T rows x kHeadDimAttn per kv head, laid out [tok][dim]
    std::unique_ptr<Buffer> v;
};

// ---- one GDN layer step ----
std::vector<float> gdn_step(Stream & stream, const Gguf & g, WeightCache & cache, int il, const std::vector<float> & x, const std::vector<std::vector<float>> & taps, GdnState & st) {
    const auto g_a0 = std::chrono::steady_clock::now();
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> Wnorm = need(g, p + "ssm_norm.weight");
    const std::vector<float> dt = load_tensor(g, *need_t(g, p + "ssm_dt.bias"));
    const std::vector<float> negA = load_tensor(g, *need_t(g, p + "ssm_a"));
    const auto g_a1 = std::chrono::steady_clock::now();

    const auto g_u0 = std::chrono::steady_clock::now();
    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };
    const auto g_u1 = std::chrono::steady_clock::now();

    const auto g_b0 = std::chrono::steady_clock::now();
    Buffer b_qkv((size_t) kQkvDim * 2), b_z((size_t) kValueDim * 2);
    gemv_w(stream, cache, p + "attn_qkv.weight", kQkvDim, kHidden, tx, b_qkv);
    gemv_w(stream, cache, p + "attn_gate.weight", kValueDim, kHidden, tx, b_z);
    const auto g_b1 = std::chrono::steady_clock::now();
    Tensor t_qkv{ b_qkv.data(), DType::F16, 1, kQkvDim };

    // short conv over the history: tap kConvK-1 goes with the current token, kConvK-2 with t-1
    Buffer b_acc((size_t) kQkvDim * 2), b_tmp((size_t) kQkvDim * 2), b_tap((size_t) kQkvDim * 2);
    upload(std::vector<float>((size_t) kQkvDim, 0.0f), b_acc);
    Buffer b_conv((size_t) kQkvDim * 2);
    for (int64_t d = 0; d < kConvK; ++d) {
        const int64_t tap = kConvK - 1 - d;
        upload(taps[(size_t) tap], b_tap);
        Tensor t_tap{ b_tap.data(), DType::F16, 1, kQkvDim };
        Tensor t_acc_in{ b_acc.data(), DType::F16, 1, kQkvDim };
        Tensor t_tmp{ b_tmp.data(), DType::F16, 1, kQkvDim };
        Tensor t_conv{ b_conv.data(), DType::F16, 1, kQkvDim };
        if (d == 0) {
            mul(stream, t_qkv, t_tap, t_conv);
        } else {
            Tensor hist{ st.conv[(size_t) (d - 1)]->data(), DType::F16, 1, kQkvDim };
            mul(stream, hist, t_tap, t_conv);
        }
        add(stream, t_acc_in, t_conv, t_tmp);
        muls(stream, t_tmp, 1.0f, t_acc_in);
    }
    Buffer b_silu((size_t) kQkvDim * 2);
    Tensor t_acc{ b_acc.data(), DType::F16, 1, kQkvDim };
    Tensor t_silu{ b_silu.data(), DType::F16, 1, kQkvDim };
    silu(stream, t_acc, t_silu);

    // rotate the history: newest first
    for (int64_t d = kConvK - 2; d > 0; --d) {
        Tensor dst{ st.conv[(size_t) d]->data(), DType::F16, 1, kQkvDim };
        Tensor src{ st.conv[(size_t) (d - 1)]->data(), DType::F16, 1, kQkvDim };
        muls(stream, src, 1.0f, dst);
    }
    {
        Tensor dst{ st.conv[0]->data(), DType::F16, 1, kQkvDim };
        muls(stream, t_qkv, 1.0f, dst);
    }

    Buffer b_ones((size_t) kHeadDimSsm * 2), b_wnorm((size_t) kHeadDimSsm * 2);
    upload(std::vector<float>((size_t) kHeadDimSsm, 1.0f), b_ones);
    upload(Wnorm, b_wnorm);
    Tensor t_ones{ b_ones.data(), DType::F16, kHeadDimSsm, 1 };
    Tensor t_wnorm{ b_wnorm.data(), DType::F16, kHeadDimSsm, 1 };

    const auto g_c0 = std::chrono::steady_clock::now();
    const auto g_t0 = std::chrono::steady_clock::now();
    // q and k are contiguous per-head blocks, so one norm over [heads, dim] replaces one per head:
    // four operator calls instead of 96, in a loop that runs 30 times per token.
    Buffer b_qn((size_t) kKeyDim * 2), b_kn((size_t) kKeyDim * 2);
    {
        Buffer t1((size_t) kKeyDim * 2), t2((size_t) kKeyDim * 2);
        Tensor qrows{ (char *) b_silu.data(), DType::F16, kHeadDimSsm, kKHeads };
        Tensor krows{ (char *) b_silu.data() + (size_t) kKeyDim * 2, DType::F16, kHeadDimSsm, kKHeads };
        Tensor nq{ t1.data(), DType::F16, kHeadDimSsm, kKHeads };
        Tensor nk{ t2.data(), DType::F16, kHeadDimSsm, kKHeads };
        Tensor qout{ b_qn.data(), DType::F16, kHeadDimSsm, kKHeads };
        Tensor kout{ b_kn.data(), DType::F16, kHeadDimSsm, kKHeads };
        rms_norm(stream, qrows, t_ones, kRmsEps / (double) kHeadDimSsm, nq);
        rms_norm(stream, krows, t_ones, kRmsEps / (double) kHeadDimSsm, nk);
        muls(stream, nq, 1.0f / std::sqrt((float) kHeadDimSsm), qout);
        muls(stream, nk, 1.0f / std::sqrt((float) kHeadDimSsm), kout);
    }

    const auto g_t1 = std::chrono::steady_clock::now();
    Buffer b_beta((size_t) kVHeads * 2), b_alpha((size_t) kVHeads * 2);
    gemv_w(stream, cache, p + "ssm_beta.weight", kVHeads, kHidden, tx, b_beta);
    gemv_w(stream, cache, p + "ssm_alpha.weight", kVHeads, kHidden, tx, b_alpha);
    const std::vector<float> beta_raw = to_host(stream, b_beta, kVHeads);
    const std::vector<float> alpha_raw = to_host(stream, b_alpha, kVHeads);

    const auto g_t2 = std::chrono::steady_clock::now();

    // The whole head batch in one pass: the recurrence is 8 operator calls for all 32 heads instead
    // of 8 per head (256), and the gated norm afterwards is 3 calls instead of 4 per head.
    Buffer b_q32((size_t) kValueDim * 2), b_k32((size_t) kValueDim * 2);
    {
        std::vector<uint16_t> q16((size_t) kKeyDim), k16((size_t) kKeyDim);
        b_qn.copy_to_host(q16.data(), q16.size() * 2);
        b_kn.copy_to_host(k16.data(), k16.size() * 2);
        stream.synchronize();
        std::vector<uint16_t> q32((size_t) kValueDim), k32((size_t) kValueDim);
        for (int64_t h = 0; h < kVHeads; ++h) {
            const int64_t kh = h % kKHeads;
            std::memcpy(&q32[(size_t) h * kHeadDimSsm], &q16[(size_t) kh * kHeadDimSsm], kHeadDimSsm * 2);
            std::memcpy(&k32[(size_t) h * kHeadDimSsm], &k16[(size_t) kh * kHeadDimSsm], kHeadDimSsm * 2);
        }
        b_q32.copy_from_host(q32.data(), q32.size() * 2);
        b_k32.copy_from_host(k32.data(), k32.size() * 2);
    }

    std::vector<float> decay_host((size_t) kVHeads), beta_host((size_t) kVHeads);
    for (int64_t h = 0; h < kVHeads; ++h) {
        const float bt = 1.0f / (1.0f + std::exp(-beta_raw[(size_t) h]));
        const float biased = alpha_raw[(size_t) h] + dt[(size_t) h];
        const float sp = std::log1p(std::exp(biased > 20.0f ? 20.0f : biased));
        beta_host[(size_t) h] = bt;
        decay_host[(size_t) h] = std::exp(sp * negA[(size_t) h]);
    }
    Buffer b_decay((size_t) kVHeads * 4), b_betas((size_t) kVHeads * 4);
    b_decay.copy_from_host(decay_host.data(), decay_host.size() * 4);
    b_betas.copy_from_host(beta_host.data(), beta_host.size() * 4);

    Buffer b_gdn((size_t) kValueDim * 2);
    {
        Tensor tq{ b_q32.data(), DType::F16, kValueDim, 1 };
        Tensor tk{ b_k32.data(), DType::F16, kValueDim, 1 };
        Tensor tv{ (char *) b_silu.data() + (size_t) 2 * kKeyDim * 2, DType::F16, kValueDim, 1 };
        Tensor td{ b_decay.data(), DType::F32, kVHeads, 1 };
        Tensor tb{ b_betas.data(), DType::F32, kVHeads, 1 };
        Tensor ts{ st.ssm->data(), DType::F32, kHeadDimSsm, kHeadDimSsm, kVHeads };
        Tensor ta{ b_gdn.data(), DType::F16, kValueDim, 1 };
        gdn_batched_f32(stream, tq, tk, tv, td, tb, ts, ta, kHeadDimSsm, kVHeads, 1);
    }

    Buffer b_normed((size_t) kValueDim * 2);
    {
        Buffer t1((size_t) kValueDim * 2), t2((size_t) kValueDim * 2);
        Tensor tin{ b_gdn.data(), DType::F16, kHeadDimSsm, kVHeads };
        Tensor tn{ t1.data(), DType::F16, kHeadDimSsm, kVHeads };
        Tensor tz{ b_z.data(), DType::F16, kHeadDimSsm, kVHeads };
        Tensor tzs{ t2.data(), DType::F16, kHeadDimSsm, kVHeads };
        Tensor tout{ b_normed.data(), DType::F16, kHeadDimSsm, kVHeads };
        rms_norm(stream, tin, t_wnorm, kRmsEps, tn);
        silu(stream, tz, tzs);
        mul(stream, tn, tzs, tout);
    }

    const auto g_t3 = std::chrono::steady_clock::now();
    Buffer b_y((size_t) kHidden * 2);
    Tensor t_normed{ b_normed.data(), DType::F16, 1, kValueDim };
    gemv_w(stream, cache, p + "ssm_out.weight", kHidden, kValueDim, t_normed, b_y);
    const std::vector<float> out = to_host(stream, b_y, kHidden);
    const auto g_t4 = std::chrono::steady_clock::now();
    if (getenv("SEQ_TRACE")) {
        auto d = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        printf("    gdn %2d: upload %.3f  need %.3f  proj %.3f  conv %.3f  head-norm %.3f  scalars %.3f  delta %.3f  out %.3f\n",
               il, d(g_u0, g_u1), d(g_a0, g_a1), d(g_b0, g_b1), d(g_b1, g_c0), d(g_t0, g_t1), d(g_t1, g_t2), d(g_t2, g_t3),
               d(g_t3, g_t4));
    }
    return out;
}

// ---- one attention layer step ----
std::vector<float> attn_step(Stream & stream, const Gguf & g, WeightCache & cache, int il, const std::vector<float> & x,
                             int64_t t, int64_t T, AttnState & st) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::vector<float> qn = need(g, p + "attn_q_norm.weight");
    const std::vector<float> knw = need(g, p + "attn_k_norm.weight");

    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };
    Buffer b_q((size_t) kQDim * 2), b_k((size_t) kKvDim * 2), b_v((size_t) kKvDim * 2);
    gemv_w(stream, cache, p + "attn_q.weight", kQDim, kHidden, tx, b_q);
    gemv_w(stream, cache, p + "attn_k.weight", kKvDim, kHidden, tx, b_k);
    gemv_w(stream, cache, p + "attn_v.weight", kKvDim, kHidden, tx, b_v);

    Buffer b_qn((size_t) kHeadDimAttn * 2), b_kn((size_t) kHeadDimAttn * 2);
    upload(qn, b_qn);
    upload(knw, b_kn);
    Tensor t_qnw{ b_qn.data(), DType::F16, kHeadDimAttn, 1 };
    Tensor t_knw{ b_kn.data(), DType::F16, kHeadDimAttn, 1 };

    const int64_t half = kRot / 2;
    std::vector<float> ct((size_t) half), stv((size_t) half);
    for (int64_t i = 0; i < half; ++i) {
        const double theta = (double) t * std::pow((double) kFreqBase, -2.0 * (double) i / (double) kRot);
        ct[(size_t) i] = (float) std::cos(theta);
        stv[(size_t) i] = (float) std::sin(theta);
    }
    Buffer b_cos((size_t) half * 2), b_sin((size_t) half * 2);
    upload(ct, b_cos);
    upload(stv, b_sin);
    Tensor t_cos{ b_cos.data(), DType::F16, half, 1 };
    Tensor t_sin{ b_sin.data(), DType::F16, half, 1 };

    // k norm + rope, then write into the cache row t
    for (int64_t kvh = 0; kvh < kKvHeads; ++kvh) {
        Buffer t1((size_t) kHeadDimAttn * 2), t2((size_t) kHeadDimAttn * 2);
        Buffer b_krope((size_t) kHeadDimAttn * 2);
        Tensor kraw{ (char *) b_k.data() + (size_t) kvh * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tss{ t2.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tkr{ b_krope.data(), DType::F16, kHeadDimAttn, 1 };
        rms_norm(stream, kraw, t_knw, kRmsEps, tn);
        rope(stream, tn, t_cos, t_sin, kRot, tkr);
        Tensor krow{ (char *) st.k->data() + (size_t) (kvh * T + t) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn,
                     1 };
        muls(stream, tkr, 1.0f, krow);
        Tensor vraw{ (char *) b_v.data() + (size_t) kvh * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor vrow{ (char *) st.v->data() + (size_t) (kvh * T + t) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn,
                     1 };
        muls(stream, vraw, 1.0f, vrow);
        (void) tss;
    }

    Buffer b_attn((size_t) kHeads * kHeadDimAttn * 2);
    const int64_t n = t + 1;
    for (int64_t h = 0; h < kHeads; ++h) {
        Buffer t1((size_t) kHeadDimAttn * 2), t2((size_t) kHeadDimAttn * 2);
        Buffer b_qrope((size_t) kHeadDimAttn * 2);
        Tensor qraw{ (char *) b_q.data() + (size_t) h * kHeadDimAttn * 2 * 2, DType::F16, kHeadDimAttn, 1 };
        Tensor tn{ t1.data(), DType::F16, kHeadDimAttn, 1 };
        Tensor tqr{ b_qrope.data(), DType::F16, kHeadDimAttn, 1 };
        rms_norm(stream, qraw, t_qnw, kRmsEps, tn);
        rope(stream, tn, t_cos, t_sin, kRot, tqr);

        const int64_t kvh = h / (kHeads / kKvHeads);
        Buffer b_scores((size_t) n * 2), b_w((size_t) n * 2), b_scaled((size_t) n * 2);
        Buffer b_mix((size_t) kHeadDimAttn * 2);
        Tensor kc{ (char *) st.k->data() + (size_t) (kvh * T) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, n };
        Tensor q_col{ b_qrope.data(), DType::F16, 1, kHeadDimAttn };
        Tensor scores{ b_scores.data(), DType::F16, 1, n };
        matmul(stream, kc, q_col, scores);

        Tensor sc_in{ b_scores.data(), DType::F16, n, 1 };
        Tensor sc_out{ b_scaled.data(), DType::F16, n, 1 };
        muls(stream, sc_in, 1.0f / std::sqrt((float) kHeadDimAttn), sc_out);
        Tensor srow{ b_scaled.data(), DType::F16, n, 1 };
        Tensor wrow{ b_w.data(), DType::F16, n, 1 };
        softmax(stream, srow, wrow);

        Tensor wmat{ b_w.data(), DType::F16, n, 1 };
        Tensor vc{ (char *) st.v->data() + (size_t) (kvh * T) * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, n };
        Tensor mix{ b_mix.data(), DType::F16, kHeadDimAttn, 1 };
        matmul(stream, wmat, vc, mix);

        Tensor graw{ (char *) b_q.data() + (size_t) (h * kHeadDimAttn * 2 + kHeadDimAttn) * 2, DType::F16,
                     kHeadDimAttn, 1 };
        Tensor tgate{ t2.data(), DType::F16, kHeadDimAttn, 1 };
        sigmoid(stream, graw, tgate);
        Tensor tout{ (char *) b_attn.data() + (size_t) h * kHeadDimAttn * 2, DType::F16, kHeadDimAttn, 1 };
        mul(stream, mix, tgate, tout);
    }

    Buffer b_y((size_t) kHidden * 2);
    Tensor t_attn{ b_attn.data(), DType::F16, 1, kHeads * kHeadDimAttn };
    gemv_w(stream, cache, p + "attn_output.weight", kHidden, kHeads * kHeadDimAttn, t_attn, b_y);
    return to_host(stream, b_y, kHidden);
}

std::vector<float> moe_ffn(Stream & stream, const Gguf & g, WeightCache & cache, int il, const std::vector<float> & x) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const GgufTensor * gate_inp = need_t(g, p + "ffn_gate_inp.weight");

    Buffer bx((size_t) kHidden * 2);
    upload(x, bx);
    Tensor tx{ bx.data(), DType::F16, 1, kHidden };

    const int64_t n_expert = (int64_t) gate_inp->dims[1];
    Buffer b_logits((size_t) n_expert * 2);
    gemv_w(stream, cache, p + "ffn_gate_inp.weight", n_expert, kHidden, tx, b_logits);
    const std::vector<float> logits = to_host(stream, b_logits, n_expert);

    double mx = logits[0];
    for (float v : logits) mx = std::max(mx, (double) v);
    std::vector<float> probs((size_t) n_expert);
    double sum = 0.0;
    for (int64_t e = 0; e < n_expert; ++e) {
        probs[(size_t) e] = (float) std::exp((double) logits[(size_t) e] - mx);
        sum += probs[(size_t) e];
    }
    for (auto & v : probs) v = (float) (v / sum);
    std::vector<int> order((size_t) n_expert);
    for (int64_t i = 0; i < n_expert; ++i) order[(size_t) i] = (int) i;
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                      [&](int a, int b) { return probs[(size_t) a] > probs[(size_t) b]; });
    std::vector<int> sel(order.begin(), order.begin() + kTopK);
    std::vector<float> w((size_t) kTopK);
    double wsum = 0.0;
    for (int64_t i = 0; i < kTopK; ++i) {
        w[(size_t) i] = probs[(size_t) sel[(size_t) i]];
        wsum += w[(size_t) i];
    }
    wsum = std::max(wsum, 6.103515625e-5);
    for (auto & v : w) v = (float) (v / wsum);

    Buffer acc_a((size_t) kHidden * 2), acc_b((size_t) kHidden * 2), tmp((size_t) kHidden * 2);
    Buffer b_gate((size_t) kExpertInter * 2), b_up((size_t) kExpertInter * 2);
    Buffer b_silu((size_t) kExpertInter * 2), b_act((size_t) kExpertInter * 2), b_y((size_t) kHidden * 2);
    upload(std::vector<float>((size_t) kHidden, 0.0f), acc_a);
    Buffer * acc = &acc_a;
    Buffer * other = &acc_b;

    cache.prefetch(p + "ffn_gate_exps.weight", sel, kExpertInter, kHidden, il);
    cache.prefetch(p + "ffn_up_exps.weight", sel, kExpertInter, kHidden, il);
    cache.prefetch(p + "ffn_down_exps.weight", sel, kHidden, kExpertInter, il);

    for (int64_t i = 0; i < kTopK; ++i) {
        const int e = sel[(size_t) i];
        {
            const Buffer & Wg = cache.slab(p + "ffn_gate_exps.weight", e, kExpertInter, kHidden, il);
            Tensor tw{ Wg.data(), DType::F16, kHidden, kExpertInter };
            Tensor to{ b_gate.data(), DType::F16, 1, kExpertInter };
            matmul(stream, tw, tx, to);
        }
        {
            const Buffer & Wu = cache.slab(p + "ffn_up_exps.weight", e, kExpertInter, kHidden, il);
            Tensor tw{ Wu.data(), DType::F16, kHidden, kExpertInter };
            Tensor to{ b_up.data(), DType::F16, 1, kExpertInter };
            matmul(stream, tw, tx, to);
        }
        Tensor g_t{ b_gate.data(), DType::F16, 1, kExpertInter };
        Tensor u_t{ b_up.data(), DType::F16, 1, kExpertInter };
        Tensor s_t{ b_silu.data(), DType::F16, 1, kExpertInter };
        Tensor a_t{ b_act.data(), DType::F16, 1, kExpertInter };
        silu(stream, g_t, s_t);
        mul(stream, s_t, u_t, a_t);
        {
            const Buffer & Wd = cache.slab(p + "ffn_down_exps.weight", e, kHidden, kExpertInter, il);
            Tensor tw{ Wd.data(), DType::F16, kExpertInter, kHidden };
            Tensor to{ b_y.data(), DType::F16, 1, kHidden };
            matmul(stream, tw, a_t, to);
        }
        Tensor y_t{ b_y.data(), DType::F16, 1, kHidden };
        Tensor t_tmp{ tmp.data(), DType::F16, 1, kHidden };
        muls(stream, y_t, w[(size_t) i], t_tmp);
        Tensor acc_t{ acc->data(), DType::F16, 1, kHidden };
        Tensor oth_t{ other->data(), DType::F16, 1, kHidden };
        add(stream, acc_t, t_tmp, oth_t);
        std::swap(acc, other);
    }
    {
        gemv_w(stream, cache, p + "ffn_gate_shexp.weight", kExpertInter, kHidden, tx, b_gate);
        gemv_w(stream, cache, p + "ffn_up_shexp.weight", kExpertInter, kHidden, tx, b_up);
        Tensor g_t{ b_gate.data(), DType::F16, 1, kExpertInter };
        Tensor u_t{ b_up.data(), DType::F16, 1, kExpertInter };
        Tensor s_t{ b_silu.data(), DType::F16, 1, kExpertInter };
        Tensor a_t{ b_act.data(), DType::F16, 1, kExpertInter };
        silu(stream, g_t, s_t);
        mul(stream, s_t, u_t, a_t);
        gemv_w(stream, cache, p + "ffn_down_shexp.weight", kHidden, kExpertInter, a_t, b_y);
        Buffer b_sg(2);
        gemv_w(stream, cache, p + "ffn_gate_inp_shexp.weight", 1, kHidden, tx, b_sg);
        const float gate_dot = to_host(stream, b_sg, 1)[0];
        const float sg = 1.0f / (1.0f + std::exp(-gate_dot));
        Tensor y_t{ b_y.data(), DType::F16, 1, kHidden };
        Tensor t_tmp{ tmp.data(), DType::F16, 1, kHidden };
        muls(stream, y_t, sg, t_tmp);
        Tensor acc_t{ acc->data(), DType::F16, 1, kHidden };
        Tensor oth_t{ other->data(), DType::F16, 1, kHidden };
        add(stream, acc_t, t_tmp, oth_t);
        std::swap(acc, other);
    }
    return to_host(stream, *acc, kHidden);
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        printf("usage: model_seq <model.gguf> <token ids, comma separated>\n");
        return 2;
    }
    const std::string path = argv[1];
    std::vector<int64_t> toks;
    {
        const std::string list = argv[2];
        size_t start = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const std::string item = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!item.empty()) toks.push_back(std::atoll(item.c_str()));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    if (toks.empty()) {
        printf("no tokens\n");
        return 2;
    }
    const int64_t T = (int64_t) toks.size();
    const int64_t n_predict = argc > 3 ? std::atoll(argv[3]) : 0;
    const float gen_temp = argc > 4 ? (float) std::atof(argv[4]) : 0.0f;
    const int64_t kCap = T + std::max<int64_t>(0, n_predict);

    try {
        const Gguf g(path);
        const int64_t interval = g.metadata_int(g.arch() + ".full_attention_interval");
        const GgufTensor * embd = g.find("token_embd.weight");
        const GgufTensor * out_w = g.find("output.weight");
        if (!embd || !out_w) {
            printf("embedding or output tensor missing\n");
            return 2;
        }

        Device device(0);
        Stream stream(device);
        WeightCache cache(g);

        // conv taps, newest last, per layer
        std::vector<std::vector<std::vector<float>>> layer_taps;
        std::vector<std::unique_ptr<GdnState>> gdn_states;
        std::vector<std::unique_ptr<AttnState>> attn_states;
        layer_taps.resize(40);
        gdn_states.resize(40);
        attn_states.resize(40);
        for (int64_t il = 0; il < 40; ++il) {
            const bool recr = ((il + 1) % interval) != 0;
            if (recr) {
                const std::string p = "blk." + std::to_string(il) + ".";
                const std::vector<float> Wconv = need(g, p + "ssm_conv1d.weight");
                std::vector<std::vector<float>> taps((size_t) kConvK, std::vector<float>((size_t) kQkvDim));
                for (int64_t k = 0; k < kConvK; ++k) {
                    for (int64_t c = 0; c < kQkvDim; ++c) {
                        taps[(size_t) k][(size_t) c] = Wconv[(size_t) c * kConvK + k];
                    }
                }
                layer_taps[(size_t) il] = std::move(taps);
                auto st = std::make_unique<GdnState>();
                for (int64_t d = 0; d < kConvK - 1; ++d) {
                    st->conv.push_back(std::make_unique<Buffer>((size_t) kQkvDim * 2));
                    upload(std::vector<float>((size_t) kQkvDim, 0.0f), *st->conv.back());
                }
                // float, not half: the recurrence accumulates here and llama.cpp computes it in float
                st->ssm = std::make_unique<Buffer>((size_t) kVHeads * kHeadDimSsm * kHeadDimSsm * 4);
                {
                    std::vector<float> zeros((size_t) kVHeads * kHeadDimSsm * kHeadDimSsm, 0.0f);
                    st->ssm->copy_from_host(zeros.data(), zeros.size() * 4);
                }
                gdn_states[(size_t) il] = std::move(st);
            } else {
                auto st = std::make_unique<AttnState>();
                st->k = std::make_unique<Buffer>((size_t) kKvHeads * kCap * kHeadDimAttn * 2);
                st->v = std::make_unique<Buffer>((size_t) kKvHeads * kCap * kHeadDimAttn * 2);
                const std::vector<float> zeros((size_t) kKvHeads * kCap * kHeadDimAttn, 0.0f);
                upload(zeros, *st->k);
                upload(zeros, *st->v);
                attn_states[(size_t) il] = std::move(st);
            }
        }

        // embedding rows for the prompt
        const int bb = quant_type_size((int) embd->type);
        const int be = quant_block_elements((int) embd->type);
        const uint64_t row_bytes = (uint64_t) kHidden / (uint64_t) be * (uint64_t) bb;
        std::vector<std::vector<float>> xs((size_t) T);
        for (int64_t t = 0; t < T; ++t) {
            const std::vector<uint8_t> row = g.tensor_bytes(*embd, (uint64_t) toks[(size_t) t] * row_bytes, row_bytes);
            std::vector<float> h((size_t) kHidden);
            dequantize((int) embd->type, row.data(), h.data(), kHidden);
            xs[(size_t) t] = round_f16(h);
        }

        double t_norm = 0, t_gdn = 0, t_attn = 0, t_moe = 0;
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        const auto t0 = std::chrono::steady_clock::now();
        // ---- one token through the whole stack ----
        auto step_once = [&](int64_t tok, int64_t pos, int64_t cap) -> std::vector<float> {
            const std::vector<uint8_t> row = g.tensor_bytes(*embd, (uint64_t) tok * row_bytes, row_bytes);
            std::vector<float> h((size_t) kHidden);
            dequantize((int) embd->type, row.data(), h.data(), kHidden);
            std::vector<float> xn = round_f16(h);
            for (int64_t il = 0; il < 40; ++il) {
                const std::string p = "blk." + std::to_string(il) + ".";
                const bool recr = ((il + 1) % interval) != 0;
                const std::vector<float> an = need(g, p + "attn_norm.weight");
                const std::vector<float> pan = need(g, p + "post_attention_norm.weight");

                const auto n0 = now();
                Buffer bx((size_t) kHidden * 2), bn((size_t) kHidden * 2), b_an((size_t) kHidden * 2);
                upload(xn, bx);
                upload(an, b_an);
                Tensor tx{ bx.data(), DType::F16, kHidden, 1 };
                Tensor tn{ bn.data(), DType::F16, kHidden, 1 };
                Tensor t_an{ b_an.data(), DType::F16, kHidden, 1 };
                rms_norm(stream, tx, t_an, kRmsEps, tn);
                const std::vector<float> hnorm = to_host(stream, bn, kHidden);
                t_norm += secs(n0, now());

                std::vector<float> layer_out;
                const auto a0 = now();
                if (recr) {
                    layer_out = gdn_step(stream, g, cache, il, hnorm, layer_taps[(size_t) il], *gdn_states[(size_t) il]);
                    t_gdn += secs(a0, now());
                } else {
                    layer_out = attn_step(stream, g, cache, il, hnorm, pos, cap, *attn_states[(size_t) il]);
                    t_attn += secs(a0, now());
                }
                for (int64_t i = 0; i < kHidden; ++i) xn[(size_t) i] += layer_out[(size_t) i];

                Buffer bx2((size_t) kHidden * 2), b_pan((size_t) kHidden * 2);
                upload(xn, bx2);
                upload(pan, b_pan);
                Tensor t_x2{ bx2.data(), DType::F16, kHidden, 1 };
                Tensor t_pan{ b_pan.data(), DType::F16, kHidden, 1 };
                rms_norm(stream, t_x2, t_pan, kRmsEps, tn);
                const std::vector<float> ffn_in = to_host(stream, bn, kHidden);
                const auto m0 = now();
                const std::vector<float> ffn_out = moe_ffn(stream, g, cache, il, ffn_in);
                t_moe += secs(m0, now());
                for (int64_t i = 0; i < kHidden; ++i) xn[(size_t) i] += ffn_out[(size_t) i];
            }
            return xn;
        };

        // ---- final norm and the top of the distribution ----
        const std::vector<float> on = need(g, "output_norm.weight");
        Buffer b_fn((size_t) kHidden * 2);
        Tensor t_fn{ b_fn.data(), DType::F16, kHidden, 1 };
        auto top_of = [&](const std::vector<float> & hidden, int64_t k) {
            Buffer b_f((size_t) kHidden * 2), b_on((size_t) kHidden * 2);
            upload(hidden, b_f);
            upload(on, b_on);
            Tensor t_f{ b_f.data(), DType::F16, kHidden, 1 };
            Tensor t_on{ b_on.data(), DType::F16, kHidden, 1 };
            rms_norm(stream, t_f, t_on, kRmsEps, t_fn);

            const int64_t vocab = (int64_t) out_w->dims[1];
            const int64_t chunk = 32768;
            std::vector<std::pair<float, int>> best;
            double lse_max = -INFINITY, lse_sum = 0.0;
            Tensor t_fn_col{ b_fn.data(), DType::F16, 1, kHidden };
            for (int64_t s = 0; s < vocab; s += chunk) {
                const int64_t n = std::min(chunk, vocab - s);
                const int bb2 = quant_type_size((int) out_w->type);
                const int be2 = quant_block_elements((int) out_w->type);
                const uint64_t row_b = (uint64_t) kHidden / (uint64_t) be2 * (uint64_t) bb2;
                const std::vector<uint8_t> raw = g.tensor_bytes(*out_w, (uint64_t) s * row_b, (uint64_t) n * row_b);
                std::vector<float> wf((size_t) n * kHidden);
                dequantize((int) out_w->type, raw.data(), wf.data(), n * kHidden);
                wf = round_f16(wf);
                Buffer b_l((size_t) n * 2);
                gemv(stream, wf, n, kHidden, t_fn_col, b_l);
                const std::vector<float> lg = to_host(stream, b_l, n);
                for (int64_t i = 0; i < n; ++i) {
                    const double v = (double) lg[(size_t) i];
                    if (v > lse_max) {
                        lse_sum = lse_sum * std::exp(lse_max - v) + 1.0;
                        lse_max = v;
                    } else {
                        lse_sum += std::exp(v - lse_max);
                    }
                    const std::pair<float, int> cand{ lg[(size_t) i], (int) (s + i) };
                    if ((int64_t) best.size() < k) {
                        best.push_back(cand);
                    } else if (cand.first > best.back().first) {
                        best.back() = cand;
                    } else {
                        continue;
                    }
                    std::sort(best.begin(), best.end(),
                              [](const auto & a, const auto & b) { return a.first > b.first; });
                }
            }
            const double log_z = lse_max + std::log(lse_sum);
            std::sort(best.begin(), best.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            printf("LOGP %.6f\n", log_z);
            return best;
        };

        // ---- prefill ----
        const auto t_prefill = std::chrono::steady_clock::now();
        std::vector<float> hidden;
        for (int64_t t = 0; t < T; ++t) {
            hidden = step_once(toks[(size_t) t], t, kCap);
            printf("token %lld/%lld done  %.1f s\n", (long long) (t + 1), (long long) T,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill).count());
        }
        std::vector<std::pair<float, int>> best = top_of(hidden, n_predict > 0 ? 32 : 5);
        printf("\nnext token from position %lld:\n", (long long) (T - 1));
        for (size_t i = 0; i < std::min<size_t>(5, best.size()); ++i) {
            printf("  token %d  logit %.4f\n", best[i].second, (double) best[i].first);
        }

        // ---- generation ----
        if (n_predict > 0) {
            std::mt19937 rng(1234u);
            int64_t pos = T;
            for (int64_t step = 0; step < n_predict; ++step) {
                int pick = best[0].second;
                if (gen_temp > 0.0f && best.size() > 1) {
                    std::vector<double> p(best.size());
                    double sum = 0.0;
                    for (size_t i = 0; i < best.size(); ++i) {
                        p[i] = std::exp((double) (best[i].first - best[0].first) / gen_temp);
                        sum += p[i];
                    }
                    std::uniform_real_distribution<double> draw(0.0, sum);
                    double r = draw(rng);
                    for (size_t i = 0; i < p.size(); ++i) {
                        r -= p[i];
                        if (r <= 0.0) {
                            pick = best[i].second;
                            break;
                        }
                    }
                }
                printf("GEN %d %.6f\n", pick, (double) best[0].first);
                if (pos >= kCap - 1) {
                    printf("GEN_STOP context full\n");
                    break;
                }
                hidden = step_once(pick, pos, kCap);
                ++pos;
                best = top_of(hidden, n_predict - step > 1 ? 32 : 5);
            }
        }
        WeightCache::print_slab_stats();
        shutdown();
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
