// model.hpp - native Vulkan forward for the qwen35moe family (Qwen3.6/3.8-35B-A3B).
//
// Owns the weights (streamed experts stay in ExpertStream, everything else is
// resident in one host-visible buffer), all scratch/state buffers, the generic
// compute pipelines and the one-token forward. Tokenization and sampling
// plumbing live in the protocol engine; this class returns raw logits.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::vulkan {

class ExpertStream;
class ExpertIndex;
class Router;

namespace native {

struct Geometry {
    int n_layers = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int n_experts = 0;
    int top_k = 0;
    int n_ff = 0;
    int n_ff_shexp = 0;
    int hk = 0;              // ssm.group_count  (K heads)
    int hv = 0;              // ssm.inner_size / ssm.state_size (V heads)
    int d_state = 0;
    int d_conv = 0;
    int conv_channels = 0;
    int n_rot = 0;
    int full_attn_interval = 4;
    float eps = 1e-5f;
    float rope_base = 1e7f;
    int n_vocab = 0;
    int eos = -1;
    bool is_attn(int l) const {
        return full_attn_interval > 0 && (l % full_attn_interval) == full_attn_interval - 1;
    }
};

struct VulkanCtx {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkQueue queue = VK_NULL_HANDLE;
};

struct TensorInfo {
    int type = -1;
    uint64_t ne[4] = {};
    int nd = 0;
    uint64_t offset = 0;   // aligned offset inside the weight buffer
    uint64_t n_bytes = 0;
};

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VkDeviceMemory m = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
    VkDeviceSize size = 0;
};

class NativeForward {
public:
    struct Deps {
        ExpertStream* stream = nullptr;
        Router* router = nullptr;
        const ExpertIndex* index = nullptr;
    };
    struct ShaderDirs {
        std::string native;   // directory with gemv_f32.comp.spv, rms.comp.spv, ...
        std::string expert;   // directory with expert_q4_k_gemv.comp.spv, silu, ...
    };

    ~NativeForward();
    NativeForward(const NativeForward&) = delete;
    NativeForward& operator=(const NativeForward&) = delete;

    static std::unique_ptr<NativeForward> create(const VulkanCtx& ctx,
                                                 const std::string& gguf_path,
                                                 const Geometry& geom,
                                                 const Deps& deps,
                                                 const ShaderDirs& dirs,
                                                 int ctx_len,
                                                 std::string& err);

    /// Run one token at position pos(); writes logits to `logits_out` (n_vocab floats).
    bool forward(int tok, float* logits_out, std::string& err);
    bool forward_dump(int tok, float* logits_out, FILE* dump, std::string& err);
    void set_stage_dump_dir(const std::string& d) { stage_dir_ = d; }
    /// Optional CSV of router top-k ids per layer, for the prefetch-trace gate.
    void set_trace(FILE* f) { trace_ = f; }
    void reset();
    int pos() const { return pos_; }
    const Geometry& geometry() const { return geom_; }
    uint64_t expert_misses() const { return expert_misses_; }

private:
    NativeForward() = default;

    bool make_buf(VkDeviceSize bytes, Buf& out, std::string& err);
    const TensorInfo* find_tensor(const std::string& name) const;
    bool add_pipeline(const std::string& name, const std::string& spv, std::string& err);
    bool dispatch(const std::string& pipe, uint32_t gx, const uint32_t* pc,
                  const VkBuffer b[6], const VkDeviceSize offs[6], std::string& err);
    // Record several independent kernels into one command buffer and submit them
    // with a single fence wait. The CPU only sees their results at batch_flush(),
    // so every host read of a GPU-written buffer must be preceded by a flush.
    bool batch_begin(std::string& err);
    bool batch_flush(std::string& err);
    void batch_abort();  // drop an unfinished batch after an error
    bool gemv_f32(const std::string& wname, const Buf& x, Buf& y, uint32_t n_out, uint32_t n_in,
                  std::string& err);
    bool gemv_q4(const std::string& wname, const Buf& x, Buf& y, uint32_t n_out, uint32_t n_in,
                 std::string& err);
    bool gemv_q6(const std::string& wname, const Buf& x, Buf& y, uint32_t n_out, uint32_t n_in,
                 std::string& err);
    bool gemv(const std::string& wname, const Buf& x, Buf& y, uint32_t n_out, uint32_t n_in,
              std::string& err);
    bool rms(const Buf& x, const std::string& wname, Buf& y, uint32_t n, std::string& err);
    bool embed(int tok, Buf& x, std::string& err);
    bool layer(int l, Buf& x, std::string& err);

    VulkanCtx ctx_;
    Geometry geom_;
    Deps deps_;
    int ctx_len_ = 4096;
    int pos_ = 0;
    uint64_t expert_misses_ = 0;
    std::string stage_dir_;
    FILE* trace_ = nullptr;

    // weights
    Buf weights_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    uint64_t weight_align_ = 256;

    // generic Vulkan plumbing
    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> batch_sets_;  // one per deferred dispatch
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;   // reused for every dispatch
    VkCommandBuffer batch_cmd_ = VK_NULL_HANDLE;  // one submit per group of kernels
    bool batching_ = false;
    uint32_t batch_cursor_ = 0;
    VkFence fence_ = VK_NULL_HANDLE;
    Buf dummy_;
    std::unordered_map<std::string, VkPipeline> pipes_;

    // scratch
    Buf x_, h_, y_, qkv_, z_, alpha_, beta_, conv_out_, gdn_out_;
    Buf qg_, kb_, vb_, q_, gate_, att_, router_;
    Buf eg_, eu_, eh_, ey_, sg_, su_, sh_, sy_, sgate_, wts_, logits_, ids_;
    std::vector<Buf> conv_state_, S_, kc_, vc_;
};

} // namespace native
} // namespace strata::vulkan
