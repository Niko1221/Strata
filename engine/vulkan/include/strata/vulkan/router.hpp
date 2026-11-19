// router.hpp - Vulkan MoE router kernels (native engine port).
//
// Two kernels, both checked against engine/sycl/src/kernels/router_top10_parity.cpp:
//   gemv: logits = W_router * hidden   (plain F32, 256 x 2048 on Qwen3.8)
//   topk: softmax -> stable argsort -> top-k -> clamp/renormalise
//
// The point of owning these is routing-guided prefetch: the native engine can
// ask the router which experts the next tokens need instead of guessing "same
// id, next layer" as the prototype does.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

namespace strata::vulkan {

class Router {
public:
    struct Limits {
        uint32_t max_experts = 256;   // GGUF qwen35moe.expert_count
        uint32_t max_k = 2048;        // router input width (embedding_length)
        uint32_t max_top_k = 64;      // GGUF qwen35moe.expert_used_count here is 8
        uint32_t max_tokens = 64;     // one token per shader invocation
    };

    static std::unique_ptr<Router> create(VkPhysicalDevice physical, VkDevice device,
                                          uint32_t queue_family, VkQueue queue,
                                          const std::string& gemv_spv,
                                          const std::string& topk_spv,
                                          const Limits& lim, std::string& err);
    ~Router();
    Router(const Router&) = delete;
    Router& operator=(const Router&) = delete;

    /// y[e] = sum_h W[e*width+h] * x[h]. W and x are host float arrays.
    bool gemv(const float* W, const float* x, float* logits,
              uint32_t n_experts, uint32_t width, std::string& err);

    /// gemv + topk in one submission. topk reads the logits gemv wrote to the
    /// GPU buffer, so a barrier between the two dispatches is enough and the
    /// second fence wait per layer disappears (40 waits per token).
    bool gemv_topk(const float* W, const float* x, float* logits,
                   uint32_t n_experts, uint32_t width, uint32_t top_k,
                   int32_t* ids, float* weights, std::string& err);

    /// ids/weights are n_tokens x top_k, row-major. Ties pick the smallest id.
    bool topk(const float* logits, uint32_t n_tokens, uint32_t n_experts,
              uint32_t top_k, int32_t* ids, float* weights, std::string& err);

private:
    Router() = default;
    bool make_pipeline(const std::string& spv, VkPipeline& out, std::string& err);
    void run(VkPipeline pipe, VkDescriptorSet set, const uint32_t pc[4],
             uint32_t groups_x);

    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    Limits lim_;

    VkBuffer w_ = VK_NULL_HANDLE, x_ = VK_NULL_HANDLE, logits_ = VK_NULL_HANDLE;
    VkBuffer ids_ = VK_NULL_HANDLE, weights_ = VK_NULL_HANDLE;
    VkDeviceMemory w_m_ = VK_NULL_HANDLE, x_m_ = VK_NULL_HANDLE, logits_m_ = VK_NULL_HANDLE;
    VkDeviceMemory ids_m_ = VK_NULL_HANDLE, weights_m_ = VK_NULL_HANDLE;
    float* w_map_ = nullptr, * x_map_ = nullptr, * logits_map_ = nullptr;
    int32_t* ids_map_ = nullptr;
    float* weights_map_ = nullptr;

    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_gemv_ = VK_NULL_HANDLE, set_topk_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline gemv_pipe_ = VK_NULL_HANDLE, topk_pipe_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace strata::vulkan
