// CUDA image encoder for the hy4-flash mmproj GGUF.
#include "strata/hy4flash_gguf/vision.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include "vision_kernels.cuh"
#include <chrono>
#include <fstream>

namespace strata::hy4_flash {
namespace {
using namespace vision;

void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("hy4-flash vision CUDA: ") + cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("hy4-flash vision cuBLAS status " + std::to_string(status));
}

template <class T> struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t n = 0) {
        if (n)
            check(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~Buffer() {
        if (ptr)
            cudaFree(ptr);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const T* src, size_t n) { check(cudaMemcpy(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice)); }
};

struct Blas {
    cublasHandle_t handle;
    Blas() { check(cublasCreate(&handle)); }
    ~Blas() { cublasDestroy(handle); }
};

int blocks(int size) { return (size + threads - 1) / threads; }

// A probe validates each operand's shape, storage and file extent before starting CUDA.
struct Weights {
    GgufFile file;
    float epsilon;
    std::map<std::string, std::unique_ptr<Buffer<float>>> vectors;
    std::map<std::string, std::unique_ptr<Buffer<__half>>> matrices;

    explicit Weights(const std::string& path, bool load) : file(path) {
        auto meta = [&](const std::string& key) -> const MetaValue& {
            auto v = file.get(key);
            if (!v)
                throw std::runtime_error("mmproj missing metadata: " + key);
            return *v;
        };
        if (meta("general.architecture").s != "clip" || meta("clip.projector_type").s != "hy_v4_flash")
            throw std::runtime_error("hy4-flash requires a hy_v4_flash mmproj GGUF");
        for (auto [key, value] : std::map<std::string, int>{{"patch_size", 16},
                                                            {"image_size", 2048},
                                                            {"embedding_length", hidden},
                                                            {"block_count", layers},
                                                            {"attention.head_count", heads},
                                                            {"feed_forward_length", ffn},
                                                            {"projection_dim", output},
                                                            {"spatial_merge_size", 2},
                                                            {"image_min_pixels", 4096},
                                                            {"image_max_pixels", 4194304}}) {
            if (meta("clip.vision." + key).num() != value)
                throw std::runtime_error("unsupported hy4-flash mmproj " + key);
        }
        for (auto key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
            const auto& v = meta(key);
            if (v.count != 3 ||
                !std::all_of(v.items.begin(), v.items.end(), [](auto& i) { return i.num() == 0.5; }))
                throw std::runtime_error("unsupported hy4-flash mmproj image normalization");
        }
        epsilon = meta("clip.vision.attention.layer_norm_epsilon").num();
        if (!(epsilon > 0 && std::isfinite(epsilon)) || !meta("clip.use_gelu").u)
            throw std::runtime_error("unsupported hy4-flash mmproj normalization or activation");

        auto tensor = [&](const std::string& name, std::vector<uint64_t> dims, bool matrix = false) {
            auto t = file.find(name);
            if (!t || t->shape != dims || (t->type != 0 && !(matrix && (t->type == 1 || t->type == 30))))
                throw std::runtime_error("missing or incompatible mmproj tensor: " + name);
            int n = static_cast<int>(t->elements());
            size_t bytes = size_t(n) * (t->type == 0 ? 4 : 2);
            auto start = file.data_start(), size = file.file_size();
            if (start > size || t->offset > size - start || bytes > size - start - t->offset)
                throw std::runtime_error("truncated mmproj tensor: " + name);
            if (!load)
                return;
            auto src = file.tensor_data(*t);
            if (!matrix) {
                auto dst = std::make_unique<Buffer<float>>(n);
                dst->upload(reinterpret_cast<const float*>(src), n);
                vectors.emplace(name, std::move(dst));
            } else {
                auto dst = std::make_unique<Buffer<__half>>(n);
                if (t->type == 1)
                    dst->upload(reinterpret_cast<const __half*>(src), n);
                else {
                    Buffer<unsigned char> raw(bytes);
                    raw.upload(src, bytes);
                    if (t->type == 0)
                        as_half<<<blocks(n), threads>>>(reinterpret_cast<const float*>(raw.ptr), dst->ptr, n);
                    else
                        as_half<<<blocks(n), threads>>>(reinterpret_cast<const __nv_bfloat16*>(raw.ptr), dst->ptr,
                                                        n);
                    check(cudaGetLastError());
                }
                matrices.emplace(name, std::move(dst));
            }
        };
        tensor("v.patch_embd.weight", {16, 16, 3, hidden}, true);
        tensor("v.patch_embd.bias", {hidden});
        tensor("v.position_embd.weight", {hidden, edge * edge});
        for (int i = 0; i < layers; ++i) {
            auto prefix = "v.blk." + std::to_string(i) + ".";
            for (auto name : {"ln1", "ln2"}) {
                tensor(prefix + name + ".weight", {hidden});
                tensor(prefix + name + ".bias", {hidden});
            }
            for (auto name : {"attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down"}) {
                uint64_t in = !strcmp(name, "ffn_down") ? ffn : hidden;
                uint64_t out = !strcmp(name, "ffn_up") ? ffn : hidden;
                tensor(prefix + name + ".weight", {in, out}, true);
                tensor(prefix + name + ".bias", {out});
            }
        }
        tensor("mm.pre_norm.weight", {hidden});
        tensor("mm.0.weight", {2, 2, hidden, 4 * hidden}, true);
        tensor("mm.0.bias", {4 * hidden});
        tensor("mm.2.weight", {1, 1, 4 * hidden, output}, true);
        tensor("mm.2.bias", {output});
        tensor("v.image_newline", {output});
    }
    const float* vec(const std::string& name) const { return vectors.at(name)->ptr; }
};

struct Encoder {
    const Weights& w;
    Blas blas;

    void linear(const std::string& name, const __half* x, float* y, int rows, int in, int out,
                Activation activation = Activation::linear, bool residual = false) {
        const float one = 1, beta = residual ? 1.f : 0.f;
        check(cublasGemmEx(blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one,
                           w.matrices.at(name + ".weight")->ptr, CUDA_R_16F, in, x, CUDA_R_16F, in, &beta, y,
                           CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        epilogue<<<blocks(rows * out), threads>>>(y, w.vec(name + ".bias"), rows * out, out, activation);
    }

    void attention(const float* q, const float* k, const float* v, float* result, int n, float* scores) {
        constexpr int head_dim = hidden / heads;
        const float scale = 1.f / std::sqrt(float(head_dim)), zero = 0, one = 1;
        // Tile the query axis so even a 2048x2048 image needs O(patches) scratch.
        for (int offset = 0; offset < n; offset += 64) {
            int queries = std::min(64, n - offset);
            int stride = queries * n;
            check(cublasSgemmStridedBatched(blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, queries, head_dim, &scale, k,
                                            hidden, head_dim, q + offset * hidden, hidden, head_dim, &zero, scores,
                                            n, stride, heads));
            softmax<<<queries * heads, threads>>>(scores, n);
            check(cublasSgemmStridedBatched(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, queries, n, &one, v,
                                            hidden, head_dim, scores, n, stride, &zero, result + offset * hidden,
                                            hidden, head_dim, heads));
        }
    }

    void image(const std::vector<unsigned char>& rgb, int width, int height, float* result) {
        int gw = width / 16, gh = height / 16, n = gw * gh, rows = n / 4;
        Buffer<float> x(n * hidden);
        Buffer<__half> input(n * ffn);
        {
            Buffer<unsigned char> pixels(rgb.size());
            pixels.upload(rgb.data(), rgb.size());
            patchify<<<n, threads>>>(pixels.ptr, input.ptr, width);
            linear("v.patch_embd", input.ptr, x.ptr, n, 768, hidden);
            add_position<<<n, threads>>>(x.ptr, w.vec("v.position_embd.weight"), gh, gw);
        }
        {
            Buffer<float> q(n * hidden), k(n * hidden), v(n * hidden), work(n * ffn);
            Buffer<float> scores(size_t(heads) * std::min(n, 64) * n);
            for (int i = 0; i < layers; ++i) {
                auto p = "v.blk." + std::to_string(i) + ".";
                normalize<<<n, threads>>>(x.ptr, w.vec(p + "ln1.weight"), w.vec(p + "ln1.bias"), input.ptr,
                                          w.epsilon, gw);
                linear(p + "attn_q", input.ptr, q.ptr, n, hidden, hidden);
                linear(p + "attn_k", input.ptr, k.ptr, n, hidden, hidden);
                linear(p + "attn_v", input.ptr, v.ptr, n, hidden, hidden);
                attention(q.ptr, k.ptr, v.ptr, work.ptr, n, scores.ptr);
                as_half<<<blocks(n * hidden), threads>>>(work.ptr, input.ptr, n * hidden);
                linear(p + "attn_out", input.ptr, x.ptr, n, hidden, hidden, Activation::linear, true);
                normalize<<<n, threads>>>(x.ptr, w.vec(p + "ln2.weight"), w.vec(p + "ln2.bias"), input.ptr,
                                          w.epsilon, gw);
                linear(p + "ffn_up", input.ptr, work.ptr, n, hidden, ffn, Activation::gelu_tanh);
                as_half<<<blocks(n * ffn), threads>>>(work.ptr, input.ptr, n * ffn);
                linear(p + "ffn_down", input.ptr, x.ptr, n, ffn, hidden, Activation::linear, true);
            }
        }
        normalize<<<n, threads>>>(x.ptr, w.vec("mm.pre_norm.weight"), nullptr, input.ptr, w.epsilon, gw);
        Buffer<float> intermediate(rows * 4 * hidden), projected(rows * output);
        linear("mm.0", input.ptr, intermediate.ptr, rows, 4 * hidden, 4 * hidden, Activation::gelu_erf);
        as_half<<<blocks(n * hidden), threads>>>(intermediate.ptr, input.ptr, n * hidden);
        linear("mm.2", input.ptr, projected.ptr, rows, 4 * hidden, output);
        add_newlines<<<rows + gh / 2, threads>>>(projected.ptr, w.vec("v.image_newline"), result, gw / 2);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(nullptr));
    }
};

struct Picture {
    int start, width, height;
    std::vector<unsigned char> rgb;
    int rows() const { return height / 32 * (width / 32 + 1); }
};

template <class T> T read(std::istream& in) {
    T value;
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("truncated picture bundle");
    return value;
}

std::vector<Picture> pictures(const std::string& path, const std::vector<int32_t>& ids) {
    std::ifstream in(path, std::ios::binary);
    char magic[4];
    if (!in.read(magic, 4) || std::memcmp(magic, "DSI1", 4))
        throw std::runtime_error("invalid picture bundle");
    int count = read<int32_t>(in), end = 0, total = 0;
    if (count < 1 || size_t(count) > ids.size())
        throw std::runtime_error("invalid picture count");
    std::vector<Picture> result;
    for (int i = 0; i < count; ++i) {
        auto start = read<int64_t>(in);
        int width = read<int32_t>(in), height = read<int32_t>(in);
        if (start < end || start >= int64_t(ids.size()) || width < 32 || height < 32 || width % 32 ||
            height % 32 || int64_t(width) * height > 4194304)
            throw std::runtime_error("invalid picture dimensions or position");
        Picture p{int(start), width, height, {}};
        if (start + p.rows() >= int64_t(ids.size()))
            throw std::runtime_error("image span exceeds prompt");
        end = p.start + p.rows();
        if (!std::all_of(ids.begin() + start, ids.begin() + end, [](int id) { return id == 120120; }))
            throw std::runtime_error("invalid image placeholder span");
        total += p.rows();
        p.rgb.resize(size_t(width) * height * 3);
        if (!in.read(reinterpret_cast<char*>(p.rgb.data()), p.rgb.size()))
            throw std::runtime_error("truncated picture pixels");
        result.push_back(std::move(p));
    }
    if (std::count(ids.begin(), ids.end(), 120120) != total || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("picture bundle and prompt disagree");
    return result;
}

struct PicturePrompt final : VisionPrompt {
    Buffer<float> rows;
    Buffer<int> axes, indices;
    PicturePrompt(int n, int row_count) : rows(size_t(row_count) * output), axes(3 * size_t(n)), indices(n) {}
    void embed(float* state, int base, int count, void* stream) override {
        count = std::min(count, rope.length - base);
        if (count > 0)
            embed_rows<<<count, threads, 0, static_cast<cudaStream_t>(stream)>>>(state, rows.ptr,
                                                                                 indices.ptr + base);
    }
};

} // namespace

bool probe_vision(const std::string& path, std::string& error) {
    try {
        Weights weights(path, false);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

std::unique_ptr<VisionPrompt> encode_pictures(const std::string& shard, const std::string& path,
                                              const std::vector<int32_t>& ids, std::string& error) {
    try {
        auto images = pictures(path, ids);
        int total = 0, n = static_cast<int>(ids.size());
        for (const auto& image : images)
            total += image.rows();
        auto result = std::make_unique<PicturePrompt>(n, total);
        std::vector<int> axes(size_t(n) * 3), indices(n, -1);
        result->image_mask.assign(n, 0);
        auto begin = std::chrono::steady_clock::now();
        Weights weights(shard, true);
        Encoder encoder{weights, {}};
        int i = 0, pos = 0, row = 0;
        auto text_until = [&](int end) {
            for (; i < end; ++i, ++pos)
                axes[i] = axes[n + i] = axes[2 * n + i] = pos;
        };
        for (const auto& p : images) {
            text_until(p.start);
            encoder.image(p.rgb, p.width, p.height, result->rows.ptr + size_t(row) * output);
            int cols = p.width / 32 + 1;
            for (int j = 0; j < p.rows(); ++j, ++i, ++row) {
                axes[i] = pos;
                axes[n + i] = pos + j / cols;
                axes[2 * n + i] = pos + j % cols;
                indices[i] = row;
                result->image_mask[i] = 1;
            }
            pos += std::max(cols, p.height / 32);
        }
        text_until(n);
        result->axes.upload(axes.data(), axes.size());
        result->indices.upload(indices.data(), indices.size());
        result->rope = {result->axes.ptr, n, pos - n};
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::fprintf(stderr, "HY4_FLASH_VISION encoded pictures=%zu rows=%d encode_s=%.3f format=gguf\n", images.size(),
                     total, elapsed);
        return result;
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}
} // namespace strata::hy4_flash
