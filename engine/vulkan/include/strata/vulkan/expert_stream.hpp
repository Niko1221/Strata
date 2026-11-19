// expert_stream.hpp - Vulkan expert-streaming tier for Strata.
//
// This is the storage/cache side of the HF "expert streaming" idea: experts are
// NOT loaded into memory at startup. Only the experts the router asks for are
// read from the GGUF on the NVMe, into a bounded VRAM slot pool. A background
// I/O pool prefetches the next layer's experts while the current one computes.
//
// Scope: this module owns indexing, streaming, caching, the Q4_K/Q6_K expert
// matvec shaders and the SiLU*up activation shader. It is deliberately
// independent of the rest of Strata's engine
// (see README.md) so it can be built and tested before a full Vulkan backend
// exists. No CUDA, HIP, oneAPI or Level Zero is involved.
#pragma once

#include <vulkan/vulkan.h>

struct io_uring;  // liburing, used only when STRATA_HAVE_LIBURING is defined

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::vulkan {

enum class ExpertKind : uint32_t { Gate = 0, Up = 1, Down = 2 };

/// One of the three expert tensors of a layer (ffn_gate_exps / ffn_up_exps /
/// ffn_down_exps). `file_offset` is absolute in the GGUF file.
struct ExpertTensor {
    int layer = -1;
    ExpertKind kind = ExpertKind::Gate;
    std::string name;
    uint64_t file_offset = 0;   // absolute offset of expert 0
    uint64_t n_bytes = 0;       // bytes of ONE expert
    uint32_t n_rows = 0;        // output rows (ggml ne[1])
    uint32_t k = 0;             // input width (ggml ne[0])
    uint32_t n_experts = 0;     // ggml ne[2]
    int32_t ggml_type = 0;      // ggml type: 12=Q4_K, 14=Q6_K (selects the shader)
};

/// One layer's F32 router matrix (ffn_gate_inp.weight): [k, n_experts].
struct RouterTensor {
    int layer = -1;
    uint64_t file_offset = 0;   // absolute in the GGUF
    uint32_t k = 0;             // ne[0], e.g. 2048
    uint32_t n_experts = 0;     // ne[1], e.g. 256
    int32_t ggml_type = 0;      // 0 = F32 in the packs seen so far
};

/// Minimal GGUF v2/v3 reader: only the tensor directory is needed.
class ExpertIndex {
public:
    static bool load(const std::string& gguf_path, ExpertIndex& out, std::string& err);

    uint32_t n_layers() const { return n_layers_; }
    uint32_t n_experts() const { return n_experts_; }
    uint64_t data_base() const { return data_base_; }
    const std::string& path() const { return path_; }
    const std::vector<ExpertTensor>& tensors() const { return tensors_; }
    const ExpertTensor* find(int layer, ExpertKind kind) const;
    const std::vector<RouterTensor>& routers() const { return routers_; }
    /// The F32 router matrix of a layer, for routing-guided prefetch.
    const RouterTensor* find_router(int layer) const;

    /// Shared expert (ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp) of a layer.
    /// It has the same layout as one routed expert (n_experts == 1), so the
    /// streaming cache and matvec shaders handle it unchanged.
    const ExpertTensor* find_shared(int layer, ExpertKind kind) const;
    /// The F32 scalar gate of the shared expert (ffn_gate_inp_shexp.weight).
    /// Reuses RouterTensor's shape: k == n_embd, n_experts == 1.
    const RouterTensor* find_shared_gate(int layer) const;
    const std::vector<ExpertTensor>& shared_tensors() const { return shared_; }

    /// Absolute file offset / size of one expert inside a tensor.
    uint64_t expert_offset(const ExpertTensor& t, uint32_t expert) const {
        return t.file_offset + uint64_t(expert) * t.n_bytes;
    }

private:
    std::string path_;
    uint64_t data_base_ = 0;
    uint32_t n_layers_ = 0;
    uint32_t n_experts_ = 0;
    std::vector<ExpertTensor> tensors_;
    std::vector<RouterTensor> routers_;
    std::vector<ExpertTensor> shared_;         // gate/up/down, n_experts == 1
    std::vector<RouterTensor> shared_gates_;   // ffn_gate_inp_shexp, n_experts == 1
};

struct StreamConfig {
    uint32_t cache_slots = 512;        // bounded VRAM expert cache
    uint32_t io_threads = 8;           // parallel NVMe readers
    bool prefetch_next_layer = true;   // guess next layer's experts
    uint32_t prefetch_per_layer = 2;   // how many guesses per layer
};

struct StreamStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t prefetches = 0;
    uint64_t bytes_read = 0;
    uint64_t io_ms = 0;
};

/// Vulkan expert cache + async NVMe tier.
class ExpertStream {
public:
    struct Key {
        int layer = -1;
        ExpertKind kind = ExpertKind::Gate;
        uint32_t expert = 0;
        bool shared = false;   // true for the per-layer shared expert (expert == 0)
        bool operator==(const Key& o) const {
            return layer == o.layer && kind == o.kind && expert == o.expert && shared == o.shared;
        }
    };

    ~ExpertStream();
    ExpertStream(const ExpertStream&) = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    /// The `*_spv_path` arguments name compiled expert matvec shaders. matvec()
    /// picks the right one per tensor (a GGUF can mix quantizations across
    /// layers). An empty path leaves that pipeline unavailable; matvec() then
    /// reports a clear error for tensors of that type. The device must expose a
    /// compute queue; on the B390 any graphics/compute queue works.
    static std::unique_ptr<ExpertStream> create(const ExpertIndex& index,
                                                VkPhysicalDevice physical,
                                                VkDevice device,
                                                uint32_t queue_family,
                                                VkQueue queue,
                                                const std::string& q4_spv_path,
                                                const std::string& q6_spv_path,
                                                const std::string& silu_spv_path,
                                                const std::string& q8_spv_path,
                                                const std::string& iq3_xxs_spv_path,
                                                const std::string& iq2_xs_spv_path,
                                                const std::string& iq1_m_spv_path,
                                                const StreamConfig& cfg,
                                                std::string& err);

    /// Make (layer, kind, expert) resident. Returns its slot index.
    /// Blocks only on a cache miss (pread + copy into the slot pool).
    bool ensure(const Key& key, uint32_t& slot, std::string& err);

    /// Non-blocking hint: queue an NVMe read if the expert is not resident.
    void prefetch(const Key& key);

    /// Router-guided hint: prefetch all experts a top-k router selected for a
    /// layer (usually the next one). Each id is used for gate, up and down.
    void prefetch_route(int layer, const int32_t* expert_ids, uint32_t n);


    /// y = W_slot * x, computed by the Vulkan Q4_K matvec shader.
    /// `out_floats` must have at least n_rows elements.
    bool matvec(const Key& key, uint32_t slot,
                const float* x, uint32_t k,
                float* out_floats, uint32_t n_rows,
                std::string& err);

    /// Batched variant: record many independent matvecs into one command buffer
    /// and submit them with a single fence wait. The outputs are copied back to
    /// the caller's buffers by batch_end(). Use for the top-k experts of a layer,
    /// which are independent of one another. At most 32 matvecs per batch.
    bool batch_begin(std::string& err);
    bool batch_matvec(const Key& key, uint32_t slot,
                      const float* x, uint32_t k,
                      float* out_floats, uint32_t n_rows,
                      std::string& err);
    bool batch_end(std::string& err);

    /// out[i] = silu(gate[i]) * up[i] on the Vulkan compute queue. Replaces the
    /// CPU loop in the expert forward. All tensors must have `n` elements (n<=2048).
    bool silu_mul(const float* gate, const float* up, float* out,
                  uint32_t n, std::string& err);

    /// Shared expert of one layer, the structure pinned by SYCL's
    /// shared_expert_parity.cpp: gate/up/down FFN with silu(gate)*up, then a
    /// SCALAR sigmoid gate: out = sigmoid(dot(ffn_gate_inp_shexp, x)) * FFN(x).
    /// The three matvecs and the activation run on the Vulkan queue; the scalar
    /// gate is an F32 dot on the host (the GGUF stores that vector as F32).
    /// `x` must have `k` elements; `out` receives n_rows(down) elements.
    bool shared_forward(int layer, const float* x, uint32_t k,
                        float* out, uint32_t out_rows, std::string& err);

    StreamStats stats() const;
    uint64_t slot_stride() const { return slot_stride_; }

private:
    ExpertStream() = default;

    struct Slot {
        Key key;
        bool valid = false;
        bool loading = false;  // a read job owns this slot; never evict it
        uint64_t used = 0;     // LRU clock
    };
    struct ReadJob {
        Key key;
        uint32_t slot = 0;
        uint64_t bytes = 0;
        uint64_t offset = 0;
        bool speculative = false;
    };
    struct PendingCopy {
        float* dst = nullptr;
        uint32_t src_offset = 0;   // in floats, into out_
        uint32_t n = 0;
    };

    void io_loop();
    void prefetch_key(const Key& key);
    void finish_job(const ReadJob& job, bool ok,
                    std::chrono::steady_clock::time_point t0);
#ifdef STRATA_HAVE_LIBURING
    void io_loop_uring();
    io_uring* ring_ = nullptr;
#endif
    bool create_pipeline(const std::string& spv_path, VkPipeline& out, std::string& err);
    // Record one matvec into the active batch. x comes from a GPU buffer when
    // x_buf is set, otherwise x_host is copied into act_; out_off_floats is an
    // offset into out_, so intermediates can stay on the GPU between kernels.
    bool record_matvec(const Key& key, uint32_t slot,
                       const float* x_host, VkBuffer x_buf, uint32_t x_off_floats,
                       uint32_t out_off_floats, uint32_t k, uint32_t n_rows,
                       std::string& err);
    bool record_silu(uint32_t n, uint32_t gate_off, uint32_t up_off, uint32_t out_off,
                     std::string& err);
    void record_barrier();

    const ExpertIndex* index_ = nullptr;
    int fd_ = -1;                     // GGUF file, opened once for pread()
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    StreamConfig cfg_;

    VkBuffer slots_ = VK_NULL_HANDLE;
    VkDeviceMemory slots_mem_ = VK_NULL_HANDLE;
    uint8_t* slots_mapped_ = nullptr;     // host-visible on iGPUs (zero-copy)
    uint64_t slot_stride_ = 0;

    std::vector<Slot> slots_meta_;
    std::unordered_map<uint64_t, uint32_t> resident_;  // hash(Key) -> slot
    std::unordered_map<uint64_t, uint32_t> pending_;   // hash(Key) -> slot being read (dedup)
    uint64_t clock_ = 0;
    mutable std::mutex cache_mutex_;
    std::condition_variable cache_cv_;

    std::mutex job_mutex_;
    std::condition_variable job_cv_;
    std::deque<ReadJob> jobs_;
    std::vector<std::thread> io_threads_;
    bool stopping_ = false;

    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_silu_ = VK_NULL_HANDLE;  // second set from the same pool
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_q4_k_ = VK_NULL_HANDLE;  // gate/up (and some down) experts
    VkPipeline pipeline_q6_k_ = VK_NULL_HANDLE;  // Q6_K down experts
    VkPipeline pipeline_q8_0_ = VK_NULL_HANDLE;  // Q8_0 experts (type 8)
    VkPipeline pipeline_iq3_xxs_ = VK_NULL_HANDLE; // IQ3_XXS experts (type 18)
    VkPipeline pipeline_iq2_xs_ = VK_NULL_HANDLE;  // IQ2_XS experts (type 17)
    VkPipeline pipeline_iq1_m_ = VK_NULL_HANDLE;   // IQ1_M experts (type 29)
    VkPipeline pipeline_silu_ = VK_NULL_HANDLE;  // silu(gate)*up activation
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkFence copy_fence_ = VK_NULL_HANDLE;
    VkFence compute_fence_ = VK_NULL_HANDLE;

    VkBuffer act_ = VK_NULL_HANDLE;
    VkDeviceMemory act_mem_ = VK_NULL_HANDLE;
    float* act_mapped_ = nullptr;
    VkBuffer up_ = VK_NULL_HANDLE;              // second input of silu_mul
    VkDeviceMemory up_mem_ = VK_NULL_HANDLE;
    float* up_mapped_ = nullptr;
    VkBuffer out_ = VK_NULL_HANDLE;
    VkDeviceMemory out_mem_ = VK_NULL_HANDLE;
    float* out_mapped_ = nullptr;

    // Batched matvec state. One reusable command buffer and one descriptor set
    // per pending matvec; both are allocated once and rewritten per batch.
    VkCommandBuffer batch_cmd_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> batch_sets_;
    std::vector<PendingCopy> batch_copies_;
    uint32_t batch_cursor_ = 0;

    mutable std::mutex stats_mutex_;
    StreamStats stats_;
};

/// CPU references (ggml-compatible dequant + dot) used by the tests.
float q4_k_dot_reference(const uint8_t* block_bytes, const float* x, uint32_t k);
float q6_k_dot_reference(const uint8_t* block_bytes, const float* x, uint32_t k);
float q8_0_dot_reference(const uint8_t* block_bytes, const float* x, uint32_t k);
float iq3_xxs_dot_reference(const uint8_t* block_bytes, const float* x, uint32_t k);
float iq2_xs_dot_reference(const uint8_t* block_bytes, const float* x, uint32_t k);
float iq1_m_dot_reference(const uint8_t* block_bytes, const float* x, uint32_t k);

} // namespace strata::vulkan
