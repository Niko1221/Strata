// moe_combine.hpp - Vulkan MoE combine kernel (native engine port).
//
// Checked against the `moe_combine` section of
// engine/sycl/src/kernels/shared_expert_parity.cpp. It pins three readings:
//   1. the routed sum is WEIGHTED by the router weights;
//   2. the shared expert is added UNWEIGHTED;
//   3. a null `shared` means there is no shared expert, not an undefined add.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

namespace strata::vulkan {

class MoECombine {
public:
    struct Limits {
        uint32_t max_experts = 64;   // top-k (8 in Qwen3.8, 10 in the parity test)
        uint32_t max_dim = 2048;     // embedding length
    };

    static std::unique_ptr<MoECombine> create(VkPhysicalDevice physical, VkDevice device,
                                              uint32_t queue_family, VkQueue queue,
                                              const std::string& spv,
                                              const Limits& lim, std::string& err);
    ~MoECombine();
    MoECombine(const MoECombine&) = delete;
    MoECombine& operator=(const MoECombine&) = delete;

    /// out[j] = sum_e weights[e] * parts[e*dim + j] + (shared ? shared[j] : 0).
    /// Pass shared == nullptr when the layer has no shared expert.
    bool combine(const float* parts, const float* weights, const float* shared,
                 float* out, uint32_t dim, uint32_t n_experts, std::string& err);

private:
    MoECombine() = default;

    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    Limits lim_;

    VkBuffer parts_ = VK_NULL_HANDLE, weights_ = VK_NULL_HANDLE;
    VkBuffer shared_ = VK_NULL_HANDLE, out_ = VK_NULL_HANDLE;
    VkDeviceMemory parts_m_ = VK_NULL_HANDLE, weights_m_ = VK_NULL_HANDLE;
    VkDeviceMemory shared_m_ = VK_NULL_HANDLE, out_m_ = VK_NULL_HANDLE;
    float* parts_map_ = nullptr, * weights_map_ = nullptr;
    float* shared_map_ = nullptr, * out_map_ = nullptr;

    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace strata::vulkan
