#include "strata/core/qwen35.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qwen35_attention.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {
namespace {

constexpr float RMS_EPS = 1e-6f;
constexpr int Q8_0_BYTES = 34;
constexpr int Q8K_BYTES = 292;
constexpr int64_t EOS_ID = 248044;

uint64_t q8_0_bytes(int64_t n) {
    return (uint64_t) (n / 32) * Q8_0_BYTES;
}
uint64_t align64(uint64_t x) {
    return (x + 63u) & ~uint64_t(63u);
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e == cudaSuccess) return true;
    err = std::string("Qwen3.5: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool read_index_names(const std::string & pack, std::vector<std::string> & names, std::string & err) {
    std::ifstream f(pack + "/index.txt");
    if (!f) {
        err = "cannot open " + pack + "/index.txt";
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string name;
        if (!(is >> name)) {
            err = "malformed index.txt line";
            return false;
        }
        names.push_back(name);
    }
    if (names.empty()) {
        err = "index.txt contains no tensors";
        return false;
    }
    return true;
}

std::set<std::string> skip_except(const std::vector<std::string> & names,
                                  const std::function<bool(const std::string &)> & keep) {
    std::set<std::string> skip;
    for (const auto & n : names)
        if (!keep(n)) skip.insert(n);
    return skip;
}

std::set<std::string> skip_layer(const std::vector<std::string> & names, int64_t layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    return skip_except(names, [&](const std::string & n) {
        return n.rfind(prefix, 0) == 0;
    });
}

bool make_sform(const WeightRef & r, strata::kernels::SForm & f, std::string & err, const char * name) {
    if (!r.quantized()) {
        err = std::string(name) + " is not a quantized tensor";
        return false;
    }
    f.code_bits = r.code_bits;
    f.code_bias = r.code_bias;
    f.group_elems = r.group_elems;
    f.codebook = r.codebook_iq4nl ? strata::kernels::Codebook::Iq4Nl
                                  : strata::kernels::Codebook::Affine;
    f.has_offset = r.has_offset;
    f.act_kind = r.act_kind;
    return true;
}

struct Planes {
    const uint8_t * codes = nullptr;
    const float * scales = nullptr;
    const float * offset = nullptr;
};

bool planes_of(const WeightRef & r, Planes & p, std::string & err, const char * name) {
    if (!r.quantized() || !r.data ||
        r.codes_bytes == 0 || r.scales_bytes == 0 ||
        r.codes_bytes + r.scales_bytes + r.offset_bytes != r.bytes) {
        err = std::string(name) + " has invalid canonical planes";
        return false;
    }
    const uint8_t * base = static_cast<const uint8_t *>(r.data);
    p.codes = base;
    p.scales = reinterpret_cast<const float *>(base + r.codes_bytes);
    p.offset = r.offset_bytes ? reinterpret_cast<const float *>(base + r.codes_bytes + r.scales_bytes) : nullptr;
    return true;
}

bool gemv(const WeightRef & w, const float * x, uint8_t * q8_0, uint8_t * q8k,
          float * y, int64_t n_in, int64_t n_out, void * stream, std::string & err,
          const char * name) {
    if (!w.data) {
        err = std::string(name) + " is not resident";
        return false;
    }
    if (!w.quantized()) {
        err = std::string(name) + " is not a quantized S-form in the Qwen3.5 runtime";
        return false;
    }
    strata::kernels::SForm f;
    Planes p;
    if (!make_sform(w, f, err, name) || !planes_of(w, p, err, name)) return false;

    if (w.wants_q8k()) {
        strata::kernels::quantize_q8_K(x, q8k, n_in, stream);
        strata::kernels::s_gemv_q8k_split(q8k, p.codes, p.scales, p.offset, y,
                                          n_in, n_out, f, stream);
    } else {
        strata::kernels::quantize_q8_0(x, q8_0, n_in, stream);
        strata::kernels::s_gemv_q8_0_split(q8_0, p.codes, p.scales, p.offset, y,
                                           n_in, n_out, f, stream);
    }
    return true;
}

bool f32_weight(const WeightTable & t, const std::string & name, int64_t n,
                const float *& out, std::string & err) {
    const WeightRef * w = t.find(name);
    if (!w || !w->data) {
        err = name + " is missing";
        return false;
    }
    if (w->kind != WeightKind::F32 || w->elements != n) {
        err = name + " must be F32 with " + std::to_string(n) + " elements";
        return false;
    }
    out = static_cast<const float *>(w->data);
    return true;
}

template<typename T>
void free_ptr(T *& p) {
    if (p) cudaFree(p);
    p = nullptr;
}

bool parse_i64_list(const std::string & s, std::vector<int64_t> & out, std::string & err) {
    size_t p = 0;
    while (p < s.size()) {
        char * end = nullptr;
        const long long v = std::strtoll(s.c_str() + p, &end, 10);
        if (end == s.c_str() + p) {
            err = "invalid token id list";
            return false;
        }
        out.push_back((int64_t)v);
        p = (size_t)(end - s.c_str());
        if (p == s.size()) break;
        if (s[p] != ',') {
            err = "token id list must be comma-separated";
            return false;
        }
        ++p;
    }
    return !out.empty();
}

struct RequestSampling {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 20;
    float min_p = 0.0f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    int penalty_last_n = 0;
    uint64_t seed = 0;
};

void parse_sampling_tokens(const std::vector<std::string> & toks, RequestSampling & s, size_t & id_index) {
    id_index = toks.size();
    for (size_t i = 0; i < toks.size(); ++i) {
        const size_t eq = toks[i].find('=');
        if (eq == std::string::npos) {
            id_index = i;
            return;
        }
        const std::string k = toks[i].substr(0, eq);
        const char * v = toks[i].c_str() + eq + 1;
        if (k == "temperature") s.temperature = std::strtof(v, nullptr);
        else if (k == "top_p") s.top_p = std::strtof(v, nullptr);
        else if (k == "top_k") s.top_k = std::atoi(v);
        else if (k == "min_p") s.min_p = std::strtof(v, nullptr);
        else if (k == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
        else if (k == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
        else if (k == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
        else if (k == "penalty_last_n") s.penalty_last_n = std::atoi(v);
        else if (k == "seed") s.seed = std::strtoull(v, nullptr, 10);
    }
}

int32_t sample_host(const std::vector<float> & logits, const std::vector<int32_t> & history,
                    const RequestSampling & p, uint64_t draw) {
    std::vector<float> work = logits;
    const int n = (int) work.size();

    if (p.penalty_last_n > 0) {
        const int first = std::max(0, (int)history.size() - p.penalty_last_n);
        std::unordered_map<int32_t, int> counts;
        for (int i = first; i < (int)history.size(); ++i) ++counts[history[(size_t)i]];
        for (const auto & kv : counts) {
            if (kv.first < 0 || kv.first >= n) continue;
            if (p.penalty_repeat != 1.0f) {
                if (work[kv.first] < 0) work[kv.first] *= p.penalty_repeat;
                else work[kv.first] /= p.penalty_repeat;
            }
            work[kv.first] -= p.penalty_freq * kv.second + p.penalty_present;
        }
    }

    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    const int k = p.top_k <= 0 ? n : std::min(n, std::max(1, p.top_k));
    if (k < n) {
        std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                          [&](int a, int b) { return work[a] > work[b]; });
        ids.resize(k);
    } else {
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return work[a] > work[b]; });
    }

    if (p.temperature <= 0.0f) return ids.front();

    const float temp = p.temperature;
    const float mx = work[ids.front()];
    std::vector<float> probs(ids.size());
    double sum = 0.0;
    for (size_t i = 0; i < ids.size(); ++i) {
        probs[i] = std::exp((work[ids[i]] - mx) / temp);
        sum += probs[i];
    }
    for (float & x : probs) x = (float)(x / sum);

    if (p.min_p > 0.0f) {
        const float cutoff = p.min_p * probs.front();
        size_t keep = 0;
        while (keep < probs.size() && probs[keep] >= cutoff) ++keep;
        keep = std::max<size_t>(1, keep);
        probs.resize(keep);
        ids.resize(keep);
    }
    if (p.top_p < 1.0f) {
        float acc = 0.0f;
        size_t keep = 0;
        for (; keep < probs.size(); ++keep) {
            acc += probs[keep];
            if (acc >= p.top_p) { ++keep; break; }
        }
        keep = std::max<size_t>(1, std::min(keep, probs.size()));
        probs.resize(keep);
        ids.resize(keep);
        float renorm = 0.0f;
        for (float x : probs) renorm += x;
        for (float & x : probs) x /= renorm;
    }

    std::mt19937_64 rng((p.seed ? p.seed : 0x9e3779b97f4a7c15ULL) + draw);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float r = uni(rng);
    float acc = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (r <= acc) return ids[i];
    }
    return ids.back();
}

} // namespace

Qwen35Runtime::~Qwen35Runtime() {
    if (stream_) cudaStreamSynchronize((cudaStream_t)stream_);
    delete transient_weights_;
    transient_weights_ = nullptr;
    for (auto & l : layers_) {
        delete l.weights;
        l.weights = nullptr;
        if (l.arena) cudaFree(l.arena);
        l.arena = nullptr;
    }
    free_ptr(transient_arena_);
    free_ptr(global_arena_);
    free_ptr(gdn_arena_);
    free_ptr(attn_cache_);
    free_ptr(attn_scores_);
    free_ptr(attn_q_);
    free_ptr(attn_k_);
    free_ptr(attn_v_);
    free_ptr(attn_gate_);
    free_ptr(attn_out_);
    free_ptr(attn_qfull_);
    free_ptr(ffn_gate_);
    free_ptr(ffn_up_);
    free_ptr(ffn_hidden_);
    free_ptr(ffn_up_q8_0_);
    free_ptr(attn_q8_0_);
    free_ptr(hidden_scratch_);
    free_ptr(residual_);
    free_ptr(logits_dev_);
    free_ptr(token_dev_);
    if (gdn_states_) cudaFree(gdn_states_);
    gdn_states_ = nullptr;
    kernels::qwen35_rope_free(rope_cos_, rope_sin_, rope_pos_);
    rope_cos_ = rope_sin_ = nullptr;
    rope_pos_ = nullptr;
    if (stream_) cudaStreamDestroy((cudaStream_t)stream_);
    stream_ = nullptr;
}

bool Qwen35Runtime::init(const std::string & pack_dir, int64_t max_context, int64_t resident_layers,
                         std::string & err) {
    pack_dir_ = pack_dir;
    max_context_ = max_context;
    if (max_context_ <= 0) {
        err = "max_context must be positive";
        return false;
    }

    g_.n_embd = qg_.n_embd;
    g_.n_layers = qg_.n_layers;
    g_.qsa_interval = qg_.full_attention_interval;
    g_.ssm_state_size = qg_.ssm_state_size;
    g_.ssm_k_heads = qg_.ssm_k_heads;
    g_.ssm_v_heads = qg_.ssm_v_heads;
    g_.ssm_d_conv = qg_.ssm_d_conv;
    g_.ssm_conv_channels = qg_.ssm_conv_channels;
    g_.ssm_value_dim = qg_.ssm_value_dim;
    g_.n_head = qg_.n_head;
    g_.n_head_kv = qg_.n_head_kv;
    g_.head_dim = qg_.head_dim;
    g_.n_ff = qg_.n_ff;
    g_.n_expert = 0;

    cudaStream_t st = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "create stream", err)) return false;
    stream_ = st;

    if (!read_index_names(pack_dir_, index_names_, err)) return false;

    const auto globals = skip_except(index_names_, [](const std::string & n) {
        return n == "token_embd.weight" || n == "output.weight" || n == "output_norm.weight";
    });
    if (!WeightTable::pool_bytes(pack_dir_, global_bytes_, err, &globals)) return false;
    if (!cuda_ok(cudaMalloc(&global_arena_, global_bytes_), "allocate global weights", err)) return false;
    if (!global_weights_.load(pack_dir_, global_arena_, global_bytes_, err, &globals)) return false;

    const uint64_t kv_layer = (uint64_t)max_context_ * qg_.n_head_kv * qg_.head_dim * 4u;
    const uint64_t kv_bytes = kv_layer * qg_.n_full_attention_layers();
    if (!cuda_ok(cudaMalloc(&attn_cache_, kv_bytes), "allocate full-attention KV cache", err)) return false;
    if (!cuda_ok(cudaMalloc(&attn_scores_,
                           (size_t)qg_.n_head * (size_t)max_context_ * sizeof(float)),
                 "allocate attention scores", err)) return false;

    const uint64_t gdn_scratch_bytes = gdn_buffers_bytes(g_);
    if (!cuda_ok(cudaMalloc(&gdn_arena_, gdn_scratch_bytes), "allocate GDN scratch", err)) return false;
    gdn_buffers_init(g_, gdn_arena_, gdn_);

    const uint64_t state_floats =
        (uint64_t)qg_.ssm_state_size * qg_.ssm_v_heads * qg_.ssm_state_size;
    const uint64_t conv_floats =
        (uint64_t)qg_.ssm_conv_channels * (qg_.ssm_d_conv - 1);
    gdn_state_bytes_ =
        (state_floats + conv_floats) * qg_.n_gdn_layers() * sizeof(float);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&gdn_states_), gdn_state_bytes_),
                 "allocate GDN state", err)) return false;

    const int64_t qdim = qg_.n_head * qg_.head_dim;
    const int64_t kdim = qg_.n_head_kv * qg_.head_dim;
    const int64_t qfull = qdim * 2;

    auto cmalloc = [&](void ** p, uint64_t bytes, const char * what) -> bool {
        return cuda_ok(cudaMalloc(p, bytes), what, err);
    };
    if (!cmalloc(reinterpret_cast<void **>(&attn_q_), qdim * 4, "allocate q")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&attn_k_), kdim * 4, "allocate k")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&attn_v_), kdim * 4, "allocate v")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&attn_gate_), qdim * 4, "allocate gate")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&attn_out_), std::max<int64_t>(qg_.n_embd, qdim) * 4,
                 "allocate attention/ffn output")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&attn_qfull_), qfull * 4, "allocate q/gate projection")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&ffn_gate_), qg_.n_ff * 4, "allocate ffn gate")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&ffn_up_), qg_.n_ff * 4, "allocate ffn up")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&ffn_hidden_), qg_.n_ff * 4, "allocate ffn hidden")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&hidden_scratch_), qg_.n_embd * 4, "allocate hidden")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&residual_), qg_.n_embd * 4, "allocate residual")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&logits_dev_), qg_.vocab_size * 4, "allocate logits")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&token_dev_), sizeof(int), "allocate token")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&ffn_up_q8_0_), q8_0_bytes(qg_.n_ff), "allocate ffn q8_0")) return false;
    if (!cmalloc(reinterpret_cast<void **>(&attn_q8_0_), q8_0_bytes(qdim), "allocate attention q8_0")) return false;

    kernels::qwen35_rope_init(max_context_, &rope_cos_, &rope_sin_, &rope_pos_);

    // Use the already parity-tested scalar GDN path. The dense model does not use
    // the existing Flash-Next native expert/QSA switches.
    layer_set_native_bf16(false);
    layer_set_fused_gdn(false);
    layer_set_qwen35_gdn(true);
    strata::kernels::native_gdn_set_enabled(false);

    if (!prepare_layers(resident_layers, err)) return false;

    cudaMemsetAsync(gdn_states_, 0, gdn_state_bytes_, st);
    cudaMemsetAsync(attn_cache_, 0, kv_bytes, st);
    if (!cuda_ok(cudaStreamSynchronize(st), "initialise recurrent/KV state", err)) return false;

    size_t resident = 0;
    for (const auto & l : layers_) resident += l.resident ? 1 : 0;
    std::fprintf(stderr,
                 "strata qwen35: pack=%s hidden=%lld layers=%lld context=%lld resident=%zu/%lld\n",
                 pack_dir_.c_str(), (long long)qg_.n_embd, (long long)qg_.n_layers,
                 (long long)max_context_, resident, (long long)qg_.n_layers);
    return true;
}

bool Qwen35Runtime::prepare_layers(int64_t resident_layers, std::string & err) {
    layers_.assign((size_t)qg_.n_layers, LayerSlot{});
    std::vector<uint64_t> bytes((size_t)qg_.n_layers);
    uint64_t max_layer = 0;
    for (int64_t l = 0; l < qg_.n_layers; ++l) {
        const auto skip = skip_layer(index_names_, l);
        if (!WeightTable::pool_bytes(pack_dir_, bytes[(size_t)l], err, &skip)) return false;
        if (bytes[(size_t)l] == 0) {
            err = "layer " + std::to_string(l) + " has no weights";
            return false;
        }
        max_layer = std::max(max_layer, bytes[(size_t)l]);
    }

    // Always keep one reusable upload arena for non-resident layers.
    transient_bytes_ = align64(max_layer);
    if (!cuda_ok(cudaMalloc(&transient_arena_, transient_bytes_), "allocate transient layer arena", err))
        return false;

    size_t target = 0;
    if (resident_layers < 0) {
        size_t free_b = 0, total_b = 0;
        if (!cuda_ok(cudaMemGetInfo(&free_b, &total_b), "query free VRAM", err)) return false;
        const uint64_t reserve = 512ull << 20;
        uint64_t budget = free_b > reserve ? (uint64_t)free_b - reserve : 0;
        uint64_t used = 0;
        for (int64_t l = 0; l < qg_.n_layers; ++l) {
            if (used + bytes[(size_t)l] > budget) break;
            used += bytes[(size_t)l];
            ++target;
        }
        std::fprintf(stderr,
                     "strata qwen35: auto residency: %.2f GiB available for layer weights -> %zu layers\n",
                     (double)budget / 1073741824.0, target);
    } else {
        target = (size_t)std::min<int64_t>(resident_layers, qg_.n_layers);
    }

    // Allocate resident layers from the first layer upward. If an explicit request is
    // larger than VRAM allows, keep as many as actually fit and leave the remainder
    // in the transient path.
    for (size_t l = 0; l < target; ++l) {
        void * arena = nullptr;
        if (cudaMalloc(&arena, bytes[l]) != cudaSuccess) {
            std::fprintf(stderr, "strata qwen35: resident layer %zu did not fit; remaining layers stream\n", l);
            break;
        }
        auto * wt = new WeightTable();
        const auto skip = skip_layer(index_names_, (int64_t)l);
        if (!wt->load(pack_dir_, arena, bytes[l], err, &skip)) {
            delete wt;
            cudaFree(arena);
            return false;
        }
        layers_[l].arena = arena;
        layers_[l].bytes = bytes[l];
        layers_[l].weights = wt;
        layers_[l].resident = true;
    }
    return true;
}

bool Qwen35Runtime::load_layer(int64_t layer, WeightTable *& table, void *& arena,
                               uint64_t & bytes, std::string & err) {
    if (layer < 0 || layer >= qg_.n_layers) {
        err = "layer index outside 0..63";
        return false;
    }
    LayerSlot & slot = layers_[(size_t)layer];
    if (slot.resident) {
        table = slot.weights;
        arena = slot.arena;
        bytes = slot.bytes;
        return true;
    }

    const auto skip = skip_layer(index_names_, layer);
    uint64_t need = 0;
    if (!WeightTable::pool_bytes(pack_dir_, need, err, &skip)) return false;
    if (need > transient_bytes_) {
        err = "transient layer arena is too small";
        return false;
    }
    delete transient_weights_;
    transient_weights_ = new WeightTable();
    if (!transient_weights_->load(pack_dir_, transient_arena_, transient_bytes_, err, &skip))
        return false;
    table = transient_weights_;
    arena = transient_arena_;
    bytes = need;
    return true;
}

bool Qwen35Runtime::eval_layer(int64_t layer, float * hidden, std::string & err) {
    WeightTable * wt = nullptr;
    void * arena = nullptr;
    uint64_t bytes = 0;
    if (!load_layer(layer, wt, arena, bytes, err)) return false;
    (void)arena;
    (void)bytes;

    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::string attn_norm_name = prefix + "attn_norm.weight";
    const std::string post_norm_name = prefix + "post_attention_norm.weight";
    const std::string ffn_gate_name = prefix + "ffn_gate.weight";
    const std::string ffn_up_name = prefix + "ffn_up.weight";
    const std::string ffn_down_name = prefix + "ffn_down.weight";

    const float * norm = nullptr;
    if (!f32_weight(*wt, attn_norm_name, qg_.n_embd, norm, err)) return false;

    cudaStream_t st = (cudaStream_t)stream_;
    if (!cuda_ok(cudaMemcpyAsync(residual_, hidden, qg_.n_embd * sizeof(float),
                                 cudaMemcpyDeviceToDevice, st), "save attention residual", err)) return false;
    strata::kernels::qwen35_rms_norm_weighted(hidden, norm, 1, qg_.n_embd, RMS_EPS, st);

    if (qg_.is_full_attention(layer)) {
        const WeightRef * wq = wt->find(prefix + "attn_q.weight");
        const WeightRef * wk = wt->find(prefix + "attn_k.weight");
        const WeightRef * wv = wt->find(prefix + "attn_v.weight");
        const WeightRef * wo = wt->find(prefix + "attn_output.weight");
        const float * qnorm = nullptr;
        const float * knorm = nullptr;
        if (!wq || !wk || !wv || !wo) {
            err = prefix + " is missing one or more full-attention projections";
            return false;
        }
        if (!f32_weight(*wt, prefix + "attn_q_norm.weight", qg_.head_dim, qnorm, err)) return false;
        if (!f32_weight(*wt, prefix + "attn_k_norm.weight", qg_.head_dim, knorm, err)) return false;

        if (!gemv(*wq, hidden, gdn_.x_q8_0, gdn_.x_q8k, attn_qfull_,
                  qg_.n_embd, qg_.n_head * qg_.head_dim * 2, st, err, (prefix + "attn_q.weight").c_str()))
            return false;
        if (!gemv(*wk, hidden, gdn_.x_q8_0, gdn_.x_q8k, attn_k_,
                  qg_.n_embd, qg_.n_head_kv * qg_.head_dim, st, err, (prefix + "attn_k.weight").c_str()))
            return false;
        if (!gemv(*wv, hidden, gdn_.x_q8_0, gdn_.x_q8k, attn_v_,
                  qg_.n_embd, qg_.n_head_kv * qg_.head_dim, st, err, (prefix + "attn_v.weight").c_str()))
            return false;

        // qfull is [head][Q 256][gate 256].
        if (!cuda_ok(cudaMemcpy2DAsync(attn_q_, qg_.head_dim * sizeof(float),
                                       attn_qfull_, 2 * qg_.head_dim * sizeof(float),
                                       qg_.head_dim * sizeof(float), qg_.n_head,
                                       cudaMemcpyDeviceToDevice, st),
                     "split attention Q", err)) return false;
        if (!cuda_ok(cudaMemcpy2DAsync(attn_gate_, qg_.head_dim * sizeof(float),
                                       attn_qfull_ + qg_.head_dim, 2 * qg_.head_dim * sizeof(float),
                                       qg_.head_dim * sizeof(float), qg_.n_head,
                                       cudaMemcpyDeviceToDevice, st),
                     "split attention gate", err)) return false;

        strata::kernels::qwen35_rms_norm_weighted(attn_q_, qnorm, qg_.n_head, qg_.head_dim, RMS_EPS, st);
        strata::kernels::qwen35_rms_norm_weighted(attn_k_, knorm, qg_.n_head_kv, qg_.head_dim, RMS_EPS, st);
        kernels::qwen35_rope_set_pos(rope_pos_, current_position_, st);
        kernels::qwen35_rope_apply(attn_q_, qg_.n_head, qg_.head_dim, rope_cos_, rope_sin_, rope_pos_, st);
        kernels::qwen35_rope_apply(attn_k_, qg_.n_head_kv, qg_.head_dim, rope_cos_, rope_sin_, rope_pos_, st);

        const int full_index = (int)(layer / qg_.full_attention_interval);
        const uint64_t kv_one_bytes =
            (uint64_t)max_context_ * qg_.n_head_kv * qg_.head_dim * 2u;
        const uint64_t kv_layer_bytes = kv_one_bytes * 2u;
        uint8_t * kv = static_cast<uint8_t *>(attn_cache_) +
                       (uint64_t)full_index * kv_layer_bytes;
        uint16_t * cache_k = reinterpret_cast<uint16_t *>(kv);
        uint16_t * cache_v = reinterpret_cast<uint16_t *>(kv + kv_one_bytes);

        kernels::qwen35_full_attention_step(attn_q_, attn_k_, attn_v_, cache_k, cache_v,
                                   current_position_, max_context_,
                                   qg_.n_head, qg_.n_head_kv, qg_.head_dim,
                                   attn_gate_, attn_scores_, attn_out_, st);

        if (!gemv(*wo, attn_out_, gdn_.x_q8_0, gdn_.x_q8k, hidden,
                  qg_.n_head * qg_.head_dim, qg_.n_embd, st, err,
                  (prefix + "attn_output.weight").c_str()))
            return false;
    } else {
        const int gi = (int)(layer - (layer + 1) / qg_.full_attention_interval);
        const uint64_t state_stride =
            ((uint64_t)qg_.ssm_state_size * qg_.ssm_v_heads * qg_.ssm_state_size +
             (uint64_t)qg_.ssm_conv_channels * (qg_.ssm_d_conv - 1)) * sizeof(float);
        gdn_.state = gdn_states_ + (uint64_t)gi * state_stride / sizeof(float);
        gdn_.conv_state = gdn_.state +
                          (uint64_t)qg_.ssm_state_size * qg_.ssm_v_heads * qg_.ssm_state_size;
        if (!gdn_layer(*wt, g_, layer, gdn_, hidden, attn_out_, st, err)) return false;
        if (!cuda_ok(cudaMemcpyAsync(hidden, attn_out_, qg_.n_embd * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st), "copy GDN output", err)) return false;
    }

    strata::kernels::add_inplace(hidden, residual_, qg_.n_embd, st);
    if (!f32_weight(*wt, post_norm_name, qg_.n_embd, norm, err)) return false;

    if (!cuda_ok(cudaMemcpyAsync(residual_, hidden, qg_.n_embd * sizeof(float),
                                 cudaMemcpyDeviceToDevice, st), "save FFN residual", err)) return false;
    strata::kernels::qwen35_rms_norm_weighted(hidden, norm, 1, qg_.n_embd, RMS_EPS, st);

    const WeightRef * wg = wt->find(ffn_gate_name);
    const WeightRef * wu = wt->find(ffn_up_name);
    const WeightRef * wd = wt->find(ffn_down_name);
    if (!wg || !wu || !wd) {
        err = prefix + " is missing one or more dense FFN projections";
        return false;
    }
    if (!gemv(*wg, hidden, gdn_.x_q8_0, gdn_.x_q8k, ffn_gate_,
              qg_.n_embd, qg_.n_ff, st, err, ffn_gate_name.c_str())) return false;
    if (!gemv(*wu, hidden, gdn_.x_q8_0, gdn_.x_q8k, ffn_up_,
              qg_.n_embd, qg_.n_ff, st, err, ffn_up_name.c_str())) return false;
    kernels::qwen35_silu_mul(ffn_gate_, ffn_up_, ffn_hidden_, qg_.n_ff, st);

    if (!gemv(*wd, ffn_hidden_, gdn_.x_q8_0, gdn_.x_q8k, attn_out_,
              qg_.n_ff, qg_.n_embd, st, err, ffn_down_name.c_str())) return false;
    if (!cuda_ok(cudaMemcpyAsync(hidden, attn_out_, qg_.n_embd * sizeof(float),
                                 cudaMemcpyDeviceToDevice, st), "copy FFN output", err)) return false;
    strata::kernels::add_inplace(hidden, residual_, qg_.n_embd, st);
    return true;
}

bool Qwen35Runtime::eval_token(int64_t token, int64_t pos, float * hidden, std::string & err) {
    if (pos < 0 || pos >= max_context_) {
        err = "position " + std::to_string(pos) + " is outside max-context";
        return false;
    }
    current_position_ = pos;
    if (!embed_row(global_weights_, g_, token, hidden, stream_, err)) return false;
    for (int64_t l = 0; l < qg_.n_layers; ++l)
        if (!eval_layer(l, hidden, err)) return false;
    return true;
}

bool Qwen35Runtime::output_logits(float * hidden, float * logits, std::string & err) {
    const float * norm = nullptr;
    if (!f32_weight(global_weights_, "output_norm.weight", qg_.n_embd, norm, err)) return false;
    cudaStream_t st = (cudaStream_t)stream_;
    strata::kernels::qwen35_rms_norm_weighted(hidden, norm, 1, qg_.n_embd, RMS_EPS, st);
    const WeightRef * w = global_weights_.find("output.weight");
    if (!w) {
        err = "output.weight is missing";
        return false;
    }
    return gemv(*w, hidden, gdn_.x_q8_0, gdn_.x_q8k, logits,
                qg_.n_embd, qg_.vocab_size, st, err, "output.weight");
}

bool Qwen35Runtime::generate(const std::vector<int64_t> & prompt, int max_new,
                             float temperature, float top_p, int top_k, float min_p,
                             uint64_t seed, std::vector<int32_t> & output,
                             std::string & finish, std::string & err) {
    if (prompt.empty()) {
        err = "prompt is empty";
        return false;
    }
    if ((int64_t)prompt.size() + max_new > max_context_) {
        err = "prompt + max_new exceeds max-context";
        return false;
    }

    RequestSampling samp;
    samp.temperature = temperature;
    samp.top_p = top_p;
    samp.top_k = top_k;
    samp.min_p = min_p;
    samp.seed = seed;

    history_.clear();
    history_.reserve(prompt.size() + (size_t)max_new);
    for (int64_t id : prompt) history_.push_back((int32_t)id);

    float * hidden = hidden_scratch_;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!eval_token(prompt[i], (int64_t)i, hidden, err)) return false;
    }
    if (!output_logits(hidden, logits_dev_, err)) return false;

    std::vector<float> logits((size_t)qg_.vocab_size);
    if (!cuda_ok(cudaMemcpy(logits.data(), logits_dev_, logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "read logits", err)) return false;

    output.clear();
    finish = "length";
    for (int step = 0; step < max_new; ++step) {
        const int32_t next = sample_host(logits, history_, samp, sample_counter_++);
        output.push_back(next);
        history_.push_back(next);
        if (next == EOS_ID) {
            finish = "stop";
            break;
        }
        const int64_t pos = (int64_t)prompt.size() + step;
        if (!eval_token(next, pos, hidden, err)) return false;
        if (!output_logits(hidden, logits_dev_, err)) return false;
        if (!cuda_ok(cudaMemcpy(logits.data(), logits_dev_, logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost), "read logits", err)) return false;
    }
    return true;
}

int qwen35_main(int argc, char ** argv) {
    bool serve = false;
    std::string pack;
    int64_t max_context = 32768;
    int64_t resident_layers = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata qwen35: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--family") {
            const std::string f = next("--family");
            if (f != "qwen35") {
                std::fprintf(stderr, "strata qwen35: this executable path only accepts --family qwen35\n");
                return 2;
            }
        } else if (a == "--serve") {
            serve = true;
        } else if (a == "--pack") {
            pack = next("--pack");
        } else if (a == "--max-context") {
            max_context = std::atoll(next("--max-context"));
        } else if (a == "--resident-layers") {
            const std::string v = next("--resident-layers");
            resident_layers = v == "auto" ? -1 : std::atoll(v.c_str());
        } else if (a == "--help" || a == "-h") {
            std::printf("Qwen3.8-27B native Strata runtime\n");
            std::printf("  --family qwen35 --serve --pack DIR --max-context N [--resident-layers auto|N]\n");
            return 0;
        } else if (a.rfind("--", 0) == 0) {
            // The common setup/server can pass harmless engine flags. Refuse the
            // ones that would change the model semantics instead of silently
            // pretending to support them.
            if (a == "--vision" || a == "--mmproj" || a == "--mtp" || a == "--spec") {
                std::fprintf(stderr, "strata qwen35: %s is not part of the dense Qwen3.5 text path yet\n", a.c_str());
                return 2;
            }
        }
    }

    if (!serve) {
        std::fprintf(stderr, "strata qwen35: --serve is required\n");
        return 2;
    }
    if (pack.empty()) {
        std::fprintf(stderr, "strata qwen35: --pack is required\n");
        return 2;
    }

    Qwen35Runtime rt;
    std::string err;
    if (!rt.init(pack, max_context, resident_layers, err)) {
        std::fprintf(stderr, "strata qwen35: init failed: %s\n", err.c_str());
        return 1;
    }

    std::printf("INFO engine=0.1.29-qwen35 model=qwen3.8-27b architecture=qwen3_5\n");
    std::printf("INFO attention=full_gqa layers=64 full_attention=16 gdn=48\n");
    std::printf("INFO kv=fp16 resident_layers=auto\n");
    std::printf("READY %lld stop\n", (long long)rt.max_context());
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        if (line == "STOP") continue; // synchronous baseline; the current request finishes normally
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [sampling keys] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }

        std::istringstream is(line.substr(4));
        std::string max_s;
        if (!(is >> max_s)) {
            std::printf("ERR bad max_new\n");
            std::fflush(stdout);
            continue;
        }
        const int max_new = std::atoi(max_s.c_str());
        std::vector<std::string> toks;
        std::string t;
        while (is >> t) toks.push_back(t);

        RequestSampling samp;
        size_t id_index = toks.size();
        parse_sampling_tokens(toks, samp, id_index);
        if (id_index >= toks.size()) {
            std::printf("ERR missing prompt token ids\n");
            std::fflush(stdout);
            continue;
        }
        std::vector<int64_t> prompt;
        std::string pe;
        if (!parse_i64_list(toks[id_index], prompt, pe)) {
            std::printf("ERR %s\n", pe.c_str());
            std::fflush(stdout);
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> out;
        std::string finish;
        if (!rt.generate(prompt, max_new, samp.temperature, samp.top_p, samp.top_k,
                         samp.min_p, samp.seed, out, finish, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        for (int32_t id : out) {
            std::printf("T %d\n", id);
            std::fflush(stdout);
        }
        std::printf("DONE %zu %zu %.1f %.1f %s 0 0 0 0 0\n",
                    out.size(), prompt.size(), ms, ms, finish.c_str());
        std::fflush(stdout);
    }
    return 0;
}

} // namespace strata::core
