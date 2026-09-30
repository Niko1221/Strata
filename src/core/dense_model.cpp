// src/core/dense_model.cpp - see include/strata/core/dense_model.hpp.
#include "strata/core/dense_model.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <regex>
#include <set>

namespace strata::core {
namespace {

using namespace strata::kernels;

constexpr uint32_t kF32 = 0, kF16 = 1, kBF16 = 30;

// The GGUF types a quantized weight may have: exactly what native_mmvq takes.
const char* kSupported = "Q4_0, Q5_0, Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS";

// defined below: dequantizes contiguous blocks of any type the host knows (F32/F16/BF16 and the quantized ones)
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out);

struct QMat {
    void* dev = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    QMat ffn_gate, ffn_up, ffn_down;
    // GDN mixer
    QMat qkv, z, ssm_out;
    float *conv_w = nullptr, *dt = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
    float *alpha_w = nullptr, *beta_w = nullptr;   // F32/F16/BF16 files
    QMat alpha_q, beta_q;                          // quantized files (Q8_0 in the Unsloth GGUFs)
    bool ab_quant = false;
    float *state = nullptr, *conv_state = nullptr;
    // attention mixer
    QMat q, k, v, o;
    float *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
};

std::string upper_type(uint32_t t) { return strata::ggml_type_name(t); }

// A tensor of one GGUF file, found by name across the shards.
struct Found {
    const strata::GgufFile* file = nullptr;
    const strata::TensorInfo* tensor = nullptr;
};

}  // namespace

struct DenseModel::Impl {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Found> index;
    std::vector<Layer> layers;
    std::vector<void*> allocations;                 // everything cudaMalloc'ed here, freed in the destructor
    std::vector<void*> host_allocs;                 // weights that did not fit in VRAM: mapped pinned host memory
    uint64_t budget = UINT64_MAX;                   // bytes of weights that may go to VRAM
    uint64_t dev_used = 0, host_bytes = 0;
    int host_tensors = 0;
    long long dbg_from = -1, dbg_to = -1;           // STRATA_DENSE_DEBUG=A:B traces the steps with A <= position < B
    cudaStream_t stream = nullptr;
    cudaEvent_t embed_done = nullptr;

    // token embedding: rows are dequantized on the host and staged through pinned memory
    Found embd;
    uint32_t embd_type = 0;
    size_t embd_row_bytes = 0;
    float* embd_stage = nullptr;                    // pinned, n_embd floats
    std::vector<float> row_tmp;

    float* output_norm = nullptr;
    QMat head;
    void* q8_1 = nullptr;                           // shared activation scratch of the native GEMVs

    // activations (device, f32)
    float *x = nullptr, *xn = nullptr, *mix = nullptr;
    float *qkv = nullptr, *conv_out = nullptr, *h = nullptr, *alpha = nullptr, *beta = nullptr, *gate = nullptr;
    float *o = nullptr, *z = nullptr, *y = nullptr;
    float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr, *attn32 = nullptr;
    float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
    float* logits = nullptr;
    float* attn_scratch = nullptr;
    int32_t* pos_dev = nullptr;

    // ---- helpers
    template <class T> bool alloc(T** out, uint64_t bytes, std::string& err, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " +
                  cudaGetErrorString(s);
            return false;
        }
        allocations.push_back(p);
        *out = (T*) p;
        return true;
    }

    const Found* find(const std::string& name) const {
        auto it = index.find(name);
        return it == index.end() ? nullptr : &it->second;
    }
    const Found* find_any(std::initializer_list<const char*> names, const std::string& prefix) const {
        for (const char* n : names)
            if (const Found* f = find(prefix + n)) return f;
        return nullptr;
    }

    // A float vector or matrix (F32, F16 or BF16 in the file) uploaded as F32.
    bool upload_floats(const Found* f, const std::string& name, uint64_t expect, float** out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        uint64_t n = 1;
        for (uint64_t d : f->tensor->shape) n *= d;
        if (expect && n != expect) {
            err = "dense model: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(expect);
            return false;
        }
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        std::vector<float> host(n);
        switch (f->tensor->type) {
        case kF32: strata::dequantize_f32(src, host.data(), (int) n); break;
        case kF16: strata::dequantize_f16(src, host.data(), (int) n); break;
        case kBF16: strata::dequantize_bf16(src, host.data(), (int) n); break;
        default:
            if (n > (uint64_t) INT_MAX || !dequant_row(f->tensor->type, src, (int) n, host.data())) {
                err = "dense model: " + name + " is " + upper_type(f->tensor->type) +
                      ", which cannot be read as floats (expected F32, F16, BF16 or a quantized type)";
                return false;
            }
        }
        if (!alloc(out, n * 4, err, name.c_str())) return false;
        const cudaError_t s = cudaMemcpy(*out, host.data(), n * 4, cudaMemcpyHostToDevice);
        if (s != cudaSuccess) { err = "dense model: upload " + name + ": " + cudaGetErrorString(s); return false; }
        return true;
    }

    // A quantized matrix, kept in its GGUF blocks.
    bool upload_mat(const Found* f, const std::string& name, int n_in, int n_out, QMat& out, uint64_t& total,
                    std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has shape [" + std::to_string(t.shape.empty() ? 0 : t.shape[0]) + ", " +
                  std::to_string(t.shape.size() > 1 ? t.shape[1] : 0) + "], expected [" + std::to_string(n_in) + ", " +
                  std::to_string(n_out) + "]";
            return false;
        }
        if (!native_mmvq_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take. "
                  "Supported: " + kSupported + "";
            return false;
        }
        int elems = 0, block_bytes = 0;
        if (!strata::block_geometry(t.type, elems, block_bytes) || (uint64_t) n_in % (uint64_t) elems) {
            err = "dense model: " + name + ": invalid block geometry"; return false;
        }
        const uint64_t bytes = native_mmvq_weight_bytes((int) t.type, n_in, n_out);
        const uint64_t in_file = (uint64_t) n_in * n_out / (uint64_t) elems * (uint64_t) block_bytes;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (bytes != in_file || t.offset > payload || bytes > payload - t.offset) {
            err = "dense model: " + name + ": payload size mismatch or truncated file"; return false;
        }
        void* dev = nullptr;
        cudaError_t s = cudaSuccess;
        if (dev_used + bytes <= budget) {
            s = cudaMalloc(&dev, bytes);
            if (s == cudaSuccess) s = cudaMemcpy(dev, f->file->tensor_data(t), bytes, cudaMemcpyHostToDevice);
            if (s != cudaSuccess) {                 // the card is fuller than cudaMemGetInfo said: spill this one
                if (dev) cudaFree(dev);
                dev = nullptr;
                cudaGetLastError();
            } else {
                allocations.push_back(dev);
                dev_used += bytes;
            }
        }
        if (!dev) {
            // Host-resident: pinned + mapped, so the GEMV kernel reads the blocks over PCIe on every token.
            void* host = nullptr;
            s = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
            if (s == cudaSuccess) s = cudaHostGetDevicePointer(&dev, host, 0);
            if (s != cudaSuccess) {
                if (host) cudaFreeHost(host);
                err = "dense model: " + name + " fits neither in VRAM nor in pinned host memory (" +
                      std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
                return false;
            }
            std::memcpy(host, f->file->tensor_data(t), bytes);
            host_allocs.push_back(host);
            host_bytes += bytes;
            ++host_tensors;
        }
        out = QMat{dev, (int) t.type, n_in, n_out};
        total += bytes;
        return true;
    }

    // y = W x for a native quantized W; x_q8_1 must already hold x quantized (quantize()).
    void gemv(const QMat& w, float* y, void* s) { native_mmvq(w.type, w.dev, q8_1, y, w.n_in, w.n_out, 1, s); }
    void quantize(const float* x, int n_in, void* s) { native_quantize_q8_1(x, q8_1, n_in, 1, s); }
};

namespace {

// Host-side dequantization of one embedding row (n values, whole blocks).
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out) {
    switch (type) {
    case kF32: strata::dequantize_f32(src, out, n); return true;
    case kF16: strata::dequantize_f16(src, out, n); return true;
    case kBF16: strata::dequantize_bf16(src, out, n); return true;
    default: break;
    }
    int elems = 0, bytes = 0;
    if (!strata::block_geometry(type, elems, bytes) || n % elems) return false;
    for (int b = 0; b < n / elems; ++b) {
        const uint8_t* blk = src + (size_t) b * bytes;
        float* o = out + (size_t) b * elems;
        switch (type) {
        case 2: strata::dequantize_q4_0(blk, o); break;
        case 6: strata::dequantize_q5_0(blk, o); break;
        case 8: strata::dequantize_q8_0(blk, o); break;
        case 11: strata::dequantize_q3_K(blk, o); break;
        case 12: strata::dequantize_q4_K(blk, o); break;
        case 13: strata::dequantize_q5_K(blk, o); break;
        case 14: strata::dequantize_q6_K(blk, o); break;
        case 20: strata::dequantize_iq4_nl(blk, o); break;
        case 23: strata::dequantize_iq4_xs(blk, o); break;
        case 42: strata::dequantize_q2_0(blk, o); break;
        default: return false;
        }
    }
    return true;
}

// llama.cpp names a split file <base>-00001-of-0000N.gguf
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re(R"((.*-)(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(m[1].str() + buf + "-of-" + m[3].str() + ".gguf");
    }
    return out;
}

// integer metadata that may be a scalar or (per layer) an array: the first element
bool meta_int(const strata::GgufFile& g, const std::string& key, int64_t& out) {
    const strata::MetaValue* v = g.get(key);
    if (!v) return false;
    if (v->type == strata::MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].u;
        return true;
    }
    out = (int64_t) v->u;
    return true;
}

// Debug aid (STRATA_DENSE_DEBUG): statistics of a device vector, printed to stderr.
void trace_vec(const char* what, int64_t pos, int layer, const char* kind, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<float> h((size_t) n);
    if (cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "dbg pos=%lld %s: copy failed\n", (long long) pos, what);
        return;
    }
    double mx = 0.0, sum2 = 0.0;
    int nan = 0, inf = 0;
    for (float v : h) {
        if (std::isnan(v)) { ++nan; continue; }
        if (std::isinf(v)) { ++inf; continue; }
        mx = std::max(mx, (double) std::fabs(v));
        sum2 += (double) v * v;
    }
    std::fprintf(stderr, "dbg pos=%lld layer=%2d %-4s %-7s max|v|=%-12.5g rms=%-12.5g nan=%d inf=%d\n", (long long) pos, layer,
                 kind, what, mx, std::sqrt(sum2 / std::max(1, n)), nan, inf);
}

void trace_logits(int64_t pos, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<float> h((size_t) n);
    if (cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    std::vector<int> idx;
    int nan = 0;
    for (int i = 0; i < n; ++i) { if (std::isnan(h[(size_t) i])) ++nan; else idx.push_back(i); }
    const size_t k = std::min<size_t>(5, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + (std::ptrdiff_t) k, idx.end(),
                      [&](int a, int b) { return h[(size_t) a] > h[(size_t) b]; });
    std::fprintf(stderr, "dbg pos=%lld logits nan=%d top:", (long long) pos, nan);
    for (size_t i = 0; i < k; ++i) std::fprintf(stderr, " %d(%.3f)", idx[i], h[(size_t) idx[i]]);
    std::fprintf(stderr, "\n");
}

}  // namespace

DenseModel::DenseModel() : impl_(new Impl) {}

DenseModel::~DenseModel() {
    if (!impl_) return;
    if (impl_->stream) cudaStreamSynchronize(impl_->stream);
    for (void* p : impl_->allocations) cudaFree(p);
    for (void* p : impl_->host_allocs) cudaFreeHost(p);
    if (impl_->embd_stage) cudaFreeHost(impl_->embd_stage);
    if (impl_->embed_done) cudaEventDestroy(impl_->embed_done);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

float* DenseModel::logits() const { return impl_->logits; }
void* DenseModel::stream() const { return impl_->stream; }

bool DenseModel::load(const std::string& path, int64_t max_context, std::string& err) {
    Impl& I = *impl_;
    try {
        // ---- 1. the files and the tensor index
        for (const std::string& p : shard_paths(path)) I.files.emplace_back(new strata::GgufFile(p));
        for (const auto& f : I.files)
            for (const auto& t : f->tensors()) I.index[t.name] = Found{f.get(), &t};
        const strata::GgufFile& g0 = *I.files.front();

        // ---- 2. the configuration, from the model's own metadata
        const strata::MetaValue* arch = g0.get("general.architecture");
        if (!arch) { err = "dense model: missing general.architecture"; return false; }
        cfg_.arch = arch->s;
        const std::string a = cfg_.arch + ".";
        if (cfg_.arch.rfind("qwen3", 0) != 0) {
            err = "dense model: architecture '" + cfg_.arch + "' is not a qwen35 dense model";
            return false;
        }
        if (g0.get(a + "expert_count") && g0.get(a + "expert_count")->u > 0) {
            err = "dense model: this GGUF is a MoE model; run it with `strata`, not `strata-dense`";
            return false;
        }
        int64_t v = 0;
        auto need = [&](const char* key, int& dst) {
            if (!meta_int(g0, a + key, v)) { err = std::string("dense model: missing ") + a + key; return false; }
            dst = (int) v;
            return true;
        };
        if (!need("block_count", cfg_.n_layer) || !need("embedding_length", cfg_.n_embd) ||
            !need("feed_forward_length", cfg_.n_ff) || !need("attention.head_count", cfg_.n_head) ||
            !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", cfg_.head_dim) ||
            !need("rope.dimension_count", cfg_.n_rot) || !need("ssm.conv_kernel", cfg_.ssm_d_conv) ||
            !need("ssm.state_size", cfg_.ssm_state) || !need("ssm.group_count", cfg_.ssm_k_heads) ||
            !need("ssm.time_step_rank", cfg_.ssm_v_heads))
            return false;
        if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
        if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
        if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
        if (meta_int(g0, a + "ssm.inner_size", v) && v != cfg_.value_dim()) {
            err = "dense model: ssm.inner_size " + std::to_string(v) + " != state_size * time_step_rank";
            return false;
        }
        if (cfg_.ssm_state != 128 || cfg_.ssm_d_conv != 4 || cfg_.ssm_v_heads % cfg_.ssm_k_heads != 0) {
            err = "dense model: the GDN kernels are written for state_size 128 and conv_kernel 4";
            return false;
        }
        if (cfg_.head_dim % 32 != 0 || cfg_.head_dim > 256 || cfg_.n_head % cfg_.n_head_kv != 0 ||
            cfg_.n_rot > cfg_.head_dim || cfg_.n_rot % 2 != 0) {
            err = "dense model: unsupported attention geometry";
            return false;
        }
        if (cfg_.n_embd % 256 != 0 || cfg_.n_ff % 256 != 0 || cfg_.value_dim() % 256 != 0 ||
            (cfg_.n_head * cfg_.head_dim) % 256 != 0) {
            err = "dense model: every GEMV input width must be a multiple of 256 (K-quant super-block)";
            return false;
        }
        const Found* tok = I.find("token_embd.weight");
        if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) cfg_.n_embd) {
            err = "dense model: token_embd.weight missing or not [n_embd, n_vocab]";
            return false;
        }
        cfg_.n_vocab = (int) tok->tensor->shape[1];
        max_context_ = max_context;
        if (max_context < 16) { err = "dense model: context must be at least 16"; return false; }

        cudaError_t cs = cudaStreamCreate(&I.stream);
        if (cs == cudaSuccess) cs = cudaEventCreateWithFlags(&I.embed_done, cudaEventDisableTiming);
        if (cs == cudaSuccess) cs = cudaHostAlloc((void**) &I.embd_stage, (size_t) cfg_.n_embd * 4, cudaHostAllocDefault);
        if (cs != cudaSuccess) { err = std::string("dense model: CUDA setup: ") + cudaGetErrorString(cs); return false; }

        if (const char* e = std::getenv("STRATA_DENSE_DEBUG")) {
            long long a0 = 0, b0 = 0;
            if (std::sscanf(e, "%lld:%lld", &a0, &b0) == 2) { I.dbg_from = a0; I.dbg_to = b0; }
        }

        // ---- 3. the embedding table stays in the (mapped) file; a row is dequantized per token
        I.embd = *tok;
        I.embd_type = tok->tensor->type;
        int elems = 0, bytes = 0;
        if (!strata::block_geometry(I.embd_type, elems, bytes) || cfg_.n_embd % elems) {
            err = "dense model: token_embd.weight has an unsupported type " + upper_type(I.embd_type);
            return false;
        }
        I.embd_row_bytes = (size_t) cfg_.n_embd / (size_t) elems * (size_t) bytes;
        I.row_tmp.resize((size_t) cfg_.n_embd);
        {   // a dry run on row 0 tells now whether the host dequantizer knows the type
            if (!dequant_row(I.embd_type, I.embd.file->tensor_data(*I.embd.tensor), cfg_.n_embd, I.row_tmp.data())) {
                err = "dense model: cannot dequantize token_embd.weight of type " + upper_type(I.embd_type);
                return false;
            }
        }

        // ---- 4. the layers
        uint64_t total = 0;
        {   // How much of the weights VRAM can take: what is free, minus the KV caches, minus a margin for the CUDA
            // context, the activations and the kernels.  The rest of the weights stays in pinned host memory.
            int n_attn = 0;
            for (int l = 0; l < cfg_.n_layer; ++l)
                if (I.find("blk." + std::to_string(l) + ".attn_q.weight")) ++n_attn;
            const uint64_t kv_need = (uint64_t) n_attn * 2 * (uint64_t) cfg_.n_head_kv * (uint64_t) max_context *
                                     (uint64_t) cfg_.head_dim * 2;
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t margin = 1024ull << 20;
            if (const char* e = std::getenv("STRATA_DENSE_GPU_MIB")) {       // explicit cap for the weights, in MiB
                I.budget = (uint64_t) std::atoll(e) << 20;
            } else if ((uint64_t) free_b > kv_need + margin) {
                I.budget = (uint64_t) free_b - kv_need - margin;
            } else {
                err = "dense model: the KV cache for a context of " + std::to_string(max_context) + " needs " +
                      std::to_string(kv_need >> 20) + " MiB, but only " + std::to_string(free_b >> 20) +
                      " MiB of VRAM are free (1 GiB is kept as margin). Use a shorter --context.";
                return false;
            }
            std::fprintf(stderr, "strata-dense: VRAM %zu MiB free of %zu; KV cache %llu MiB (context %lld); "
                         "%llu MiB of weights may go to the GPU, the rest stays in host memory\n",
                         free_b >> 20, total_b >> 20, (unsigned long long) (kv_need >> 20), (long long) max_context,
                         (unsigned long long) (I.budget >> 20));
        }
        const int C = cfg_.conv_channels(), V = cfg_.value_dim(), H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
        I.layers.assign((size_t) cfg_.n_layer, Layer{});
        for (int l = 0; l < cfg_.n_layer; ++l) {
            Layer& L = I.layers[(size_t) l];
            const std::string p = "blk." + std::to_string(l) + ".";
            const std::string at = "layer " + std::to_string(l) + ": ";
            const bool has_gdn = I.find(p + "attn_qkv.weight") != nullptr;
            const bool has_attn = I.find(p + "attn_q.weight") != nullptr;
            if (has_gdn == has_attn) { err = "dense model: " + at + "needs exactly one of attn_qkv.weight / attn_q.weight"; return false; }
            L.attn = has_attn;
            if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", cfg_.n_embd, &L.attn_norm, err)) return false;
            if (!I.upload_floats(I.find_any({"post_attention_norm.weight", "ffn_norm.weight"}, p), p + "post_attention_norm.weight",
                                 cfg_.n_embd, &L.post_norm, err)) return false;
            if (!I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", cfg_.n_embd, cfg_.n_ff, L.ffn_gate, total, err) ||
                !I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", cfg_.n_embd, cfg_.n_ff, L.ffn_up, total, err) ||
                !I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, cfg_.n_embd, L.ffn_down, total, err))
                return false;
            if (!L.attn) {
                if (!I.upload_mat(I.find(p + "attn_qkv.weight"), p + "attn_qkv.weight", cfg_.n_embd, C, L.qkv, total, err) ||
                    !I.upload_mat(I.find(p + "attn_gate.weight"), p + "attn_gate.weight", cfg_.n_embd, V, L.z, total, err) ||
                    !I.upload_mat(I.find(p + "ssm_out.weight"), p + "ssm_out.weight", V, cfg_.n_embd, L.ssm_out, total, err))
                    return false;
                if (!I.upload_floats(I.find(p + "ssm_conv1d.weight"), p + "ssm_conv1d.weight", (uint64_t) C * cfg_.ssm_d_conv, &L.conv_w, err) ||
                    !I.upload_floats(I.find_any({"ssm_dt.bias", "ssm_dt"}, p), p + "ssm_dt.bias", cfg_.ssm_v_heads, &L.dt, err) ||
                    !I.upload_floats(I.find_any({"ssm_a", "ssm_a.weight"}, p), p + "ssm_a", cfg_.ssm_v_heads, &L.ssm_a, err) ||
                    !I.upload_floats(I.find(p + "ssm_norm.weight"), p + "ssm_norm.weight", cfg_.ssm_state, &L.ssm_norm, err))
                    return false;
                {   // alpha / beta: a float matrix in some files, Q8_0 (or another block type) in the Unsloth ones
                    const Found* fa = I.find(p + "ssm_alpha.weight");
                    const Found* fb = I.find(p + "ssm_beta.weight");
                    if (!fa || !fb) { err = "dense model: " + at + "missing ssm_alpha.weight / ssm_beta.weight"; return false; }
                    auto is_float = [](uint32_t t) { return t == kF32 || t == kF16 || t == kBF16; };
                    if (is_float(fa->tensor->type) != is_float(fb->tensor->type)) {
                        err = "dense model: " + at + "ssm_alpha and ssm_beta have different kinds of type"; return false;
                    }
                    L.ab_quant = !is_float(fa->tensor->type);
                    if (L.ab_quant) {
                        if (!I.upload_mat(fa, p + "ssm_alpha.weight", cfg_.n_embd, cfg_.ssm_v_heads, L.alpha_q, total, err) ||
                            !I.upload_mat(fb, p + "ssm_beta.weight", cfg_.n_embd, cfg_.ssm_v_heads, L.beta_q, total, err))
                            return false;
                    } else if (!I.upload_floats(fa, p + "ssm_alpha.weight", (uint64_t) cfg_.n_embd * cfg_.ssm_v_heads, &L.alpha_w, err) ||
                               !I.upload_floats(fb, p + "ssm_beta.weight", (uint64_t) cfg_.n_embd * cfg_.ssm_v_heads, &L.beta_w, err))
                        return false;
                }
                const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
                const uint64_t cbytes = (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;
                if (!I.alloc(&L.state, sbytes, err, "GDN state") || !I.alloc(&L.conv_state, cbytes, err, "conv state")) return false;
            } else {
                if (!I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", cfg_.n_embd, H * 2 * D, L.q, total, err) ||
                    !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", cfg_.n_embd, HK * D, L.k, total, err) ||
                    !I.upload_mat(I.find(p + "attn_v.weight"), p + "attn_v.weight", cfg_.n_embd, HK * D, L.v, total, err) ||
                    !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, cfg_.n_embd, L.o, total, err))
                    return false;
                if (!I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
                    !I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err))
                    return false;
                const uint64_t kv = (uint64_t) HK * (uint64_t) max_context * (uint64_t) D * 2;   // half
                if (!I.alloc(&L.kc, kv, err, "K cache") || !I.alloc(&L.vc, kv, err, "V cache")) return false;
                kv_bytes_ += 2 * kv;
            }
        }

        // ---- 5. the output side
        if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", cfg_.n_embd, &I.output_norm, err)) return false;
        const Found* head = I.find("output.weight");
        if (!head) head = tok;                                   // tied embeddings
        if (!I.upload_mat(head, head == tok ? "token_embd.weight (tied head)" : "output.weight", cfg_.n_embd, cfg_.n_vocab,
                          I.head, total, err))
            return false;
        weight_bytes_ = total;
        host_bytes_ = I.host_bytes;
        if (I.host_tensors)
            std::fprintf(stderr, "strata-dense: %d tensors (%.2f GiB of %.2f) live in host memory and are read over PCIe "
                         "on every token - a shorter --context or a smaller quant puts more on the GPU\n",
                         I.host_tensors, I.host_bytes / 1073741824.0, total / 1073741824.0);

        // ---- 6. activations and scratch
        const int max_in = std::max({cfg_.n_embd, cfg_.n_ff, V, H * D});
        const uint64_t f = 4;
        bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(max_in), err, "q8_1 scratch") &&
                  I.alloc(&I.x, cfg_.n_embd * f, err, "x") && I.alloc(&I.xn, cfg_.n_embd * f, err, "xn") &&
                  I.alloc(&I.mix, cfg_.n_embd * f, err, "mix") && I.alloc(&I.qkv, C * f, err, "qkv") &&
                  I.alloc(&I.conv_out, C * f, err, "conv_out") && I.alloc(&I.h, C * f, err, "h") &&
                  I.alloc(&I.alpha, cfg_.ssm_v_heads * f, err, "alpha") && I.alloc(&I.beta, cfg_.ssm_v_heads * f, err, "beta") &&
                  I.alloc(&I.gate, cfg_.ssm_v_heads * f, err, "gate") && I.alloc(&I.o, V * f, err, "o") &&
                  I.alloc(&I.z, V * f, err, "z") && I.alloc(&I.y, V * f, err, "y") &&
                  I.alloc(&I.q_full, (uint64_t) H * 2 * D * f, err, "q_full") && I.alloc(&I.qcur, (uint64_t) H * D * f, err, "qcur") &&
                  I.alloc(&I.kcur, (uint64_t) HK * D * f, err, "kcur") && I.alloc(&I.vcur, (uint64_t) HK * D * f, err, "vcur") &&
                  I.alloc(&I.attn, (uint64_t) H * D * f, err, "attn") && I.alloc(&I.attn32, (uint64_t) H * D * f, err, "attn32") &&
                  I.alloc(&I.ffn_g, cfg_.n_ff * f, err, "ffn_g") && I.alloc(&I.ffn_u, cfg_.n_ff * f, err, "ffn_u") &&
                  I.alloc(&I.ffn_h, cfg_.n_ff * f, err, "ffn_h") && I.alloc(&I.logits, (uint64_t) cfg_.n_vocab * f, err, "logits") &&
                  I.alloc(&I.attn_scratch, dense_attn_scratch_bytes(H, D, (int) max_context), err, "attention scratch") &&
                  I.alloc(&I.pos_dev, (uint64_t) std::max(H, HK) * 4, err, "positions");
        if (!ok) return false;

        native_gdn_set_enabled(true);
        reset();
        std::string serr;
        if (!sync(serr)) { err = serr; return false; }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

void DenseModel::reset() {
    Impl& I = *impl_;
    const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cbytes = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers) {
        if (L.state) cudaMemsetAsync(L.state, 0, sbytes, I.stream);
        if (L.conv_state) cudaMemsetAsync(L.conv_state, 0, cbytes, I.stream);
    }
    pos_ = 0;
}

bool DenseModel::sync(std::string& err) {
    const cudaError_t s = cudaStreamSynchronize(impl_->stream);
    if (s != cudaSuccess) { err = std::string("dense model: ") + cudaGetErrorString(s); return false; }
    return true;
}

bool DenseModel::step(int32_t token, bool want_logits, std::string& err) {
    Impl& I = *impl_;
    if (token < 0 || token >= cfg_.n_vocab) { err = "dense model: token id outside the vocabulary"; return false; }
    if (pos_ >= max_context_) { err = "dense model: the context is full"; return false; }
    void* s = I.stream;
    const int C = cfg_.conv_channels(), V = cfg_.value_dim(), H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
    const int S = cfg_.ssm_state, KH = cfg_.ssm_k_heads, VH = cfg_.ssm_v_heads, E = cfg_.n_embd;
    const int qk = S * KH;
    const float eps = cfg_.rms_eps;
    const bool dbg = I.dbg_from >= 0 && pos_ >= I.dbg_from && pos_ < I.dbg_to;
    try {
        // ---- the embedding row: dequantized on the host into pinned memory, then one copy
        if (cudaEventSynchronize(I.embed_done) != cudaSuccess) { err = "dense model: embedding event"; return false; }
        const uint8_t* row = I.embd.file->tensor_data(*I.embd.tensor) + (size_t) token * I.embd_row_bytes;
        if (!dequant_row(I.embd_type, row, E, I.embd_stage)) { err = "dense model: embedding dequantization"; return false; }
        cudaMemcpyAsync(I.x, I.embd_stage, (size_t) E * 4, cudaMemcpyHostToDevice, I.stream);
        cudaEventRecord(I.embed_done, I.stream);
        dense_fill_i32(I.pos_dev, std::max(H, HK), (int32_t) pos_, s);
        if (dbg) trace_vec("embed", pos_, -1, "", I.x, E, s);

        for (int l = 0; l < cfg_.n_layer; ++l) {
            Layer& L = I.layers[(size_t) l];
            dense_rms_norm(I.x, L.attn_norm, I.xn, 1, E, eps, s);
            if (!L.attn) {
                // ---- gated delta net (the sequence follows layer.cpp's native path, stage by stage)
                I.quantize(I.xn, E, s);
                I.gemv(L.qkv, I.qkv, s);
                I.gemv(L.z, I.z, s);
                native_gdn_conv_silu(L.conv_state, I.qkv, L.conv_w, I.conv_out, I.h, C, cfg_.ssm_d_conv, s);
                native_gdn_l2_norm(I.h, KH, S, eps, s);
                native_gdn_l2_norm(I.h + qk, KH, S, eps, s);
                if (L.ab_quant) {           // xn is still quantized in the shared q8_1 scratch: nothing overwrote it
                    I.gemv(L.alpha_q, I.alpha, s);
                    I.gemv(L.beta_q, I.beta, s);
                } else {
                    dense_gemv_f32(L.alpha_w, I.xn, I.alpha, E, VH, s);
                    dense_gemv_f32(L.beta_w, I.xn, I.beta, E, VH, s);
                }
                native_gdn_beta_gate(I.beta, VH, s);
                native_gdn_gate(I.alpha, L.dt, L.ssm_a, I.gate, VH, s);
                GdnShapes gs{S, KH, VH};
                native_gdn_step(L.state, I.h, I.h + qk, I.h + 2 * qk, I.gate, I.beta, I.o, gs, s);
                native_gdn_out_norm(I.o, I.z, L.ssm_norm, I.y, VH, S, eps, s);
                I.quantize(I.y, V, s);
                I.gemv(L.ssm_out, I.mix, s);
            } else {
                // ---- gated attention over the KV cache
                I.quantize(I.xn, E, s);
                I.gemv(L.q, I.q_full, s);
                I.gemv(L.k, I.kcur, s);
                I.gemv(L.v, I.vcur, s);
                // q is the FIRST head_dim of every head's 2*head_dim block; the second half is the output gate
                cudaMemcpy2DAsync(I.qcur, (size_t) D * 4, I.q_full, (size_t) D * 2 * 4, (size_t) D * 4, (size_t) H,
                                  cudaMemcpyDeviceToDevice, I.stream);
                rms_norm_weighted(I.qcur, L.q_norm, H, D, eps, s);
                rms_norm_weighted(I.kcur, L.k_norm, HK, D, eps, s);
                native_rope_apply(I.qcur, I.qcur, H, D, cfg_.n_rot, cfg_.rope_base, I.pos_dev, s);
                native_rope_apply(I.kcur, I.kcur, HK, D, cfg_.n_rot, cfg_.rope_base, I.pos_dev, s);
                dense_kv_append(L.kc, L.vc, I.kcur, I.vcur, (int) pos_, HK, D, (int) max_context_, s);
                dense_attn_decode(I.qcur, L.kc, L.vc, I.attn, I.attn_scratch, H, HK, D, (int) pos_ + 1,
                                  (int) max_context_, 1.0f / std::sqrt((float) D), s);
                native_qsa_gate_apply(I.attn, I.q_full, I.attn32, H, D, s);
                I.quantize(I.attn32, H * D, s);
                I.gemv(L.o, I.mix, s);
            }
            add_inplace(I.x, I.mix, E, s);

            // ---- feed-forward
            dense_rms_norm(I.x, L.post_norm, I.xn, 1, E, eps, s);
            I.quantize(I.xn, E, s);
            I.gemv(L.ffn_gate, I.ffn_g, s);
            I.gemv(L.ffn_up, I.ffn_u, s);
            dense_swiglu(I.ffn_g, I.ffn_u, I.ffn_h, cfg_.n_ff, s);
            I.quantize(I.ffn_h, cfg_.n_ff, s);
            I.gemv(L.ffn_down, I.mix, s);
            add_inplace(I.x, I.mix, E, s);
            if (dbg) trace_vec("x", pos_, l, L.attn ? "attn" : "gdn", I.x, E, s);
        }
        ++pos_;
        if (want_logits) {
            dense_rms_norm(I.x, I.output_norm, I.xn, 1, E, eps, s);
            I.quantize(I.xn, E, s);
            I.gemv(I.head, I.logits, s);
            if (dbg) trace_logits(pos_ - 1, I.logits, cfg_.n_vocab, s);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

}  // namespace strata::core
