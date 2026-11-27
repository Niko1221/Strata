// Optional FP16 reference operators built against the matching llama.cpp CUDA backend.
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml.h"
#include <array>
#include <cuda_fp16.h>
#include <map>
#include <memory>
#include <stdexcept>

namespace {
struct Device {
    ggml_backend_t backend;
    // Supplies GGML buffer metadata for CUDA tensors whose payload is owned by the engine.
    ggml_backend_buffer_t storage;
    ggml_backend_cuda_context *ctx;
    cudaStream_t original;
    cudaStream_t current;
    cudaEvent_t done;
    void *decode_scratch = nullptr;
    size_t decode_capacity = 0;
    void *reuse_decode(size_t bytes, cudaStream_t st) {
        // Bound retained workspace when alternating short requests with long contexts.
        if (bytes > 16 * 1024 * 1024)
            return nullptr;
        if (decode_capacity < bytes) {
            void *next = nullptr;
            size_t capacity = (bytes + 1024 * 1024 - 1) & ~size_t(1024 * 1024 - 1);
            if (cudaMallocAsync(&next, capacity, st) != cudaSuccess) {
                cudaGetLastError();
                return nullptr;
            }
            if (decode_scratch)
                cudaFreeAsync(decode_scratch, st);
            decode_scratch = next;
            decode_capacity = capacity;
        }
        return decode_scratch;
    }
    int id;
    void on(cudaStream_t stream) {
        if (current != stream) {
            if (cudaStreamWaitEvent(stream, done, 0) != cudaSuccess)
                throw std::runtime_error("reference stream handoff failed");
            ctx->streams[id][0] = stream;
            if (auto handle = ctx->cublas_handles[id][0]) {
                CUBLAS_CHECK(cublasSetStream(handle, stream));
                CUBLAS_CHECK(
                    cublasSetWorkspace(handle, ctx->cublas_workspaces[id][0], ctx->cublas_workspace_sizes[id]));
            }
            current = stream;
        }
    }
    void finish() {
        if (cudaEventRecord(done, current) != cudaSuccess)
            throw std::runtime_error("reference event record failed");
    }
    Device(int device, cudaStream_t stream) : current(stream), id(device) {
        backend = ggml_backend_cuda_init(device);
        if (!backend)
            throw std::runtime_error("CUDA reference backend initialization failed");
        ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
        if (cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess)
            throw std::runtime_error("reference event creation failed");
        original = ctx->streams[device][0];
        ctx->streams[device][0] = stream;
        storage = ggml_backend_alloc_buffer(backend, 256);
        if (!storage)
            throw std::bad_alloc();
        ggml_backend_buffer_set_usage(storage, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ~Device() {
        cudaEventSynchronize(done);
        cudaEventDestroy(done);
        if (decode_scratch)
            cudaFree(decode_scratch);
        ctx->streams[id][0] = original;
        ggml_backend_buffer_free(storage);
        ggml_backend_free(backend);
    }
};
struct Op {
    Device &dev;
    ggml_context *c;
    ggml_cgraph *graph;
    ggml_tensor *x = nullptr, *w = nullptr, *u = nullptr, *down = nullptr, *y = nullptr, *gate = nullptr,
                *up = nullptr, *act = nullptr;

    explicit Op(Device &d) : dev(d) {
        c = ggml_init(
            {ggml_graph_overhead_custom(48, false) + 32 * ggml_tensor_overhead() + 8192, nullptr, true});
        graph = ggml_new_graph_custom(c, 48, false);
    }
    ~Op() { ggml_free(c); }
    ggml_tensor *tensor(int ty, int a, int b) { return ggml_new_tensor_2d(c, (ggml_type)ty, a, b); }
    void bind(ggml_tensor *t, const void *ptr) {
        t->buffer = dev.storage;
        t->data = const_cast<void *>(ptr);
    }
    void run() {
#ifdef USE_CUDA_GRAPH
        // The caller may rebind external buffers on every layer.
        dev.ctx->cuda_graph(ggml_graph_node(graph, 0))->disable_due_to_gpu_arch = true;
#endif
        if (ggml_backend_graph_compute_async(dev.backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CUDA reference graph failed");
    }
};
struct Plans {
    Device dev;
    std::map<std::array<int, 7>, std::unique_ptr<Op>> ops;
    ~Plans() { cudaEventSynchronize(dev.done); }
    Plans(int id, cudaStream_t st) : dev(id, st) {}
};
Plans &cache(void *s) {
    static thread_local std::map<int, std::unique_ptr<Plans>> devices;
    int id = -1;
    if (cudaGetDevice(&id) != cudaSuccess)
        throw std::runtime_error("CUDA device unavailable");
    auto &ptr = devices[id];
    if (!ptr)
        ptr = std::make_unique<Plans>(id, (cudaStream_t)s);
    ptr->dev.on((cudaStream_t)s);
    if (ptr->ops.size() > 64)
        ptr->ops.clear();
    return *ptr;
}
} // namespace
extern "C" void hy4_flash_reference_mm(const void *w, int ty, const float *x, float *y, int ni, int no, int nt,
                                       void *s) {
    auto &h = cache(s);
    auto &p = h.ops[{0, ty, ni, no, nt, 0, 0}];
    if (!p) {
        p = std::make_unique<Op>(h.dev);
        p->w = p->tensor(ty, ni, no);
        p->x = p->tensor(GGML_TYPE_F32, ni, nt);
        p->y = ggml_mul_mat(p->c, p->w, p->x);
        ggml_build_forward_expand(p->graph, p->y);
    }
    p->bind(p->w, w);
    p->bind(p->x, x);
    p->bind(p->y, y);
    p->run();
    p->dev.finish();
}
extern "C" void hy4_flash_reference_ffn(const float *x, const void *wg, int tg, const void *wu, int tu,
                                        const void *wd, int td, float *y, int ni, int nf, int nt, void *s,
                                        float *g, float *u) {
    auto &h = cache(s);
    auto &p = h.ops[{1, tg, tu, td, ni, nf, nt}];
    if (!p) {
        p = std::make_unique<Op>(h.dev);
        p->w = p->tensor(tg, ni, nf);
        p->u = p->tensor(tu, ni, nf);
        p->down = p->tensor(td, nf, ni);
        p->x = p->tensor(GGML_TYPE_F32, ni, nt);
        p->up = ggml_mul_mat(p->c, p->u, p->x);
        p->gate = ggml_mul_mat(p->c, p->w, p->x);
        p->act = ggml_swiglu_split(p->c, p->gate, p->up);
        p->y = ggml_mul_mat(p->c, p->down, p->act);
        ggml_build_forward_expand(p->graph, p->y);
    }
    p->bind(p->w, wg);
    p->bind(p->u, wu);
    p->bind(p->down, wd);
    p->bind(p->x, x);
    p->bind(p->gate, g);
    p->bind(p->up, u);
    p->bind(p->act, g);
    p->bind(p->y, y);
    p->run();
    p->dev.finish();
}
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void pack_query(const float *latent, const float *rope, float *packed, int width, int rotary,
                           int heads, int tokens) {
    const int row = blockIdx.x;
    const size_t token_head = (size_t)(row % tokens) * heads + row / tokens;
    for (int d = threadIdx.x; d < width + rotary; d += blockDim.x) {
        packed[(size_t)row * (width + rotary) + d] =
            d < width ? latent[(size_t)row * width + d] : rope[token_head * rotary + d - width];
    }
}

__global__ void pack_keys(const half *latent, const half *rope, half *packed, int width, int rotary, int rows,
                          const int *active) {
    const int row = blockIdx.x;
    const bool valid = row < rows && (!active || row < *active);
    for (int d = threadIdx.x; d < width + rotary; d += blockDim.x) {
        half value = __float2half(0.f);
        if (valid)
            value = d < width ? latent[(size_t)row * width + d] : rope[(size_t)row * rotary + d - width];
        packed[(size_t)row * (width + rotary) + d] = value;
    }
}

__device__ __forceinline__ void pack_key_vector(size_t i, const uint4 *latent, const uint4 *rope, uint4 *packed,
                                                int width, int rotary, int rows, const int *active, int padded) {
    const int chunks = width + rotary;
    if (i >= size_t(padded) * chunks)
        return;
    const int row = i / chunks, d = i % chunks;
    uint4 value = {};
    if (row < rows && (!active || row < *active))
        value = d < width ? latent[size_t(row) * width + d] : rope[size_t(row) * rotary + d - width];
    packed[i] = value;
}

__global__ void pack_keys_vec(const uint4 *latent, const uint4 *rope, uint4 *packed, int width, int rotary,
                              int rows, const int *active, int padded) {
    pack_key_vector(size_t(blockIdx.x) * blockDim.x + threadIdx.x, latent, rope, packed, width, rotary, rows,
                    active, padded);
}

__global__ void causal_mask(const int *query_pos, const int *key_pos, half *mask, int rows, int padded,
                            int window, const int *active) {
    const int pos = query_pos[blockIdx.x];
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        const bool valid =
            i < rows && (!active || i < *active) && key_pos[i] <= pos && (!window || key_pos[i] > pos - window);
        mask[(size_t)blockIdx.x * padded + i] = __float2half(valid ? 0.f : -INFINITY);
    }
}

__global__ void unpack_output(const float *src, float *dst, int width, int heads, int tokens) {
    const int row = blockIdx.x;
    const size_t token_head = (size_t)(row % tokens) * heads + row / tokens;
    for (int d = threadIdx.x; d < width; d += blockDim.x)
        dst[(size_t)row * width + d] = src[token_head * width + d];
}

struct Scratch {
    void *ptr;
    cudaStream_t stream;
    Scratch(size_t bytes, cudaStream_t s) : ptr(nullptr), stream(s) {
        if (bytes)
            check(cudaMallocAsync(&ptr, bytes, s));
    }
    ~Scratch() {
        if (ptr)
            cudaFreeAsync(ptr, stream);
    }
};
} // namespace

__device__ void selected_row(const int *sel, int nsel, int rows, int padded, half *mask, const int *qp,
                             const int *kp, const int *active, int t) {

    half *row = mask + size_t(t) * padded;
    for (int i = threadIdx.x; i < padded; i += blockDim.x)
        row[i] = __float2half(-INFINITY);
    __syncthreads();
    const int limit = active ? min(rows, *active) : rows;
    for (int i = threadIdx.x; i < nsel; i += blockDim.x) {
        int pos = sel[size_t(t) * nsel + i];
        if (pos < 0 || pos > qp[t])
            continue;
        int slot = pos;
        if (slot >= limit || kp[slot] != pos) {
            slot = 0;
            while (slot < limit && kp[slot] != pos)
                ++slot;
        }
        if (slot < limit)
            row[slot] = __float2half(0.f);
    }
}

__global__ void select_mask(const int *sel, int nsel, int rows, int padded, half *mask, const int *qp,
                            const int *kp, const int *active) {
    selected_row(sel, nsel, rows, padded, mask, qp, kp, active, blockIdx.x);
}

__global__ void pack_decode(const uint4 *latent, const uint4 *rotary_key, uint4 *keys, const float4 *query,
                            const float4 *rotary_query, float4 *packed_query, int width, int rotary, int rows,
                            int padded, int heads, int key_blocks, int query_blocks, const int *positions,
                            const int *query_pos, const int *active, int window, half *mask, const int *selected,
                            int nsel) {
    int block = blockIdx.x;
    if (block < key_blocks) {
        pack_key_vector(size_t(block) * blockDim.x + threadIdx.x, latent, rotary_key, keys, width / 8, rotary / 8,
                        rows, active, padded);
    } else if ((block -= key_blocks) < query_blocks) {
        int i = block * blockDim.x + threadIdx.x;
        const int chunks = (width + rotary) / 4;
        if (i >= heads * chunks)
            return;
        int h = i / chunks, d = i % chunks;
        packed_query[i] =
            d < width / 4 ? query[h * (width / 4) + d] : rotary_query[h * (rotary / 4) + d - width / 4];
    } else if (selected) {
        selected_row(selected, nsel, rows, padded, mask, query_pos, positions, active, 0);
    } else {
        int row = (block - query_blocks) * blockDim.x + threadIdx.x;
        if (row < padded) {
            int pos = *query_pos;
            bool valid = row < rows && (!active || row < *active) && positions[row] <= pos &&
                         (!window || positions[row] > pos - window);
            mask[row] = __float2half(valid ? 0.f : -INFINITY);
        }
    }
}

extern "C" void hy4_flash_reference_attention(const float *q_latent, const float *q_rope, const void *k_latent,
                                              const void *k_rope, const float *sinks, int width, int rotary,
                                              int heads, int tokens, int rows, const int *query_pos,
                                              const int *key_pos, const int *active, int window, float scale,
                                              float *out, void *stream, size_t out_bytes, const int *sel,
                                              int nsel) {
    auto s = static_cast<cudaStream_t>(stream);
    auto &backend = cache(s).dev;
    const int padded = (rows + 255) / 256 * 256;
    const size_t queries = (size_t)(width + rotary) * tokens * heads;
    const size_t outputs = (size_t)width * tokens * heads;
    const size_t keys = (size_t)(width + rotary) * padded;
    const size_t bytes = (queries + outputs) * sizeof(float) + (keys + (size_t)padded * tokens) * sizeof(half);
    // The packed Q is dead when unpack_output overwrites it with the smaller latent result.
    void *shared = tokens == 1 && bytes > out_bytes ? backend.reuse_decode(bytes, s) : nullptr;
    Scratch scratch(bytes > out_bytes && !shared ? bytes : 0, s);
    auto *storage = shared ? static_cast<float *>(shared) : scratch.ptr ? static_cast<float *>(scratch.ptr) : out;
    auto *q = storage + (tokens == 1 ? outputs : 0);
    auto *y = tokens == 1 ? out : storage + queries;
    auto *kv = reinterpret_cast<half *>(storage + outputs + queries);
    auto *mask = kv + keys;
    if (tokens == 1 && width % 8 == 0 && rotary % 8 == 0) {
        int kb = (size_t(padded) * (width + rotary) / 8 + 255) / 256;
        int qb = (heads * (width + rotary) / 4 + 255) / 256;
        int mb = sel ? 1 : (padded + 255) / 256;
        pack_decode<<<kb + qb + mb, 256, 0, s>>>(
            static_cast<const uint4 *>(k_latent), static_cast<const uint4 *>(k_rope),
            reinterpret_cast<uint4 *>(kv), reinterpret_cast<const float4 *>(q_latent),
            reinterpret_cast<const float4 *>(q_rope), reinterpret_cast<float4 *>(q), width, rotary, rows, padded,
            heads, kb, qb, key_pos, query_pos, active, window, mask, sel, nsel);
    } else {
        pack_query<<<heads * tokens, 256, 0, s>>>(q_latent, q_rope, q, width, rotary, heads, tokens);
        if ((width % 8) == 0 && (rotary % 8) == 0) {
            const size_t count = size_t(padded) * (width + rotary) / 8;
            pack_keys_vec<<<(count + 255) / 256, 256, 0, s>>>(
                static_cast<const uint4 *>(k_latent), static_cast<const uint4 *>(k_rope),
                reinterpret_cast<uint4 *>(kv), width / 8, rotary / 8, rows, active, padded);
        } else
            pack_keys<<<padded, 256, 0, s>>>(static_cast<const half *>(k_latent),
                                             static_cast<const half *>(k_rope), kv, width, rotary, rows, active);
        if (sel)
            select_mask<<<tokens, 256, 0, s>>>(sel, nsel, rows, padded, mask, query_pos, key_pos, active);
        else
            causal_mask<<<tokens, 256, 0, s>>>(query_pos, key_pos, mask, rows, padded, window, active);
    }
    alignas(64) char metadata[8192];
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({sizeof(metadata), metadata, true}),
                                                                ggml_free);
    auto *c = context.get();
    auto *Q = ggml_new_tensor_3d(c, GGML_TYPE_F32, width + rotary, tokens, heads);
    auto *K = ggml_new_tensor_3d(c, GGML_TYPE_F16, width + rotary, padded, 1);
    auto *V = ggml_view_3d(c, K, width, padded, 1, K->nb[1], K->nb[2], 0);
    auto *M = ggml_new_tensor_2d(c, GGML_TYPE_F16, padded, tokens);
    auto *Y = ggml_flash_attn_ext(c, Q, K, V, M, scale, 0.f, 0.f);
    if (sinks) {
        auto *S = ggml_new_tensor_1d(c, GGML_TYPE_F32, heads);
        S->data = const_cast<float *>(sinks);
        ggml_flash_attn_ext_add_sinks(Y, S);
    }
    ggml_prec_set_acc(Y, GGML_PREC_F32);
    ggml_flash_attn_ext_set_n_kv_max(Y, sel ? nsel : 0);
    Q->data = q;
    K->data = V->data = kv;
    M->data = mask;
    Y->data = y;
    ggml_cuda_flash_attn_ext(*backend.ctx, Y);
    backend.finish();
    if (tokens != 1)
        unpack_output<<<heads * tokens, 256, 0, s>>>(y, out, width, heads, tokens);
    check(cudaGetLastError());
}
