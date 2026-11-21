// bf16_gemv.hpp - Vulkan BF16 GEMV (native engine port).
//
// Checked against engine/sycl/src/kernels/bf16_gemv_parity.cpp. The kernel
// reads the same bf16 bits as the host reference, so the tolerance is set by
// summation order only. The activation contract is observable: a bf16 weight
// sees a BF16 activation, not an fp16 one.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::vulkan {

class Bf16Gemv {
public:
    struct Limits {
        uint32_t max_in = 4096;    // 2560 in the real projections
        uint32_t max_out = 4096;   // 2560 for ple_value
    };

    static std::unique_ptr<Bf16Gemv> create(VkPhysicalDevice physical, VkDevice device,
                                            uint32_t queue_family, VkQueue queue,
                                            const std::string& spv,
                                            const Limits& lim, std::string& err);
    ~Bf16Gemv();
    Bf16Gemv(const Bf16Gemv&) = delete;
    Bf16Gemv& operator=(const Bf16Gemv&) = delete;

    /// x holds n_in bf16 bit patterns; w holds n_out*n_in of them, row-major
    /// [out][in]. y receives n_out F32 results.
    bool gemv(const uint16_t* x, const uint16_t* w, float* y,
              uint32_t n_in, uint32_t n_out, std::string& err);

private:
    Bf16Gemv() = default;

    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    Limits lim_;

    VkBuffer x_ = VK_NULL_HANDLE, w_ = VK_NULL_HANDLE, y_ = VK_NULL_HANDLE;
    VkDeviceMemory x_m_ = VK_NULL_HANDLE, w_m_ = VK_NULL_HANDLE, y_m_ = VK_NULL_HANDLE;
    uint32_t* x_map_ = nullptr;
    uint32_t* w_map_ = nullptr;
    float* y_map_ = nullptr;
    // The shader takes uint32 words; expand the uint16 host arrays here.
    std::vector<uint32_t> x_words_, w_words_;

    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace strata::vulkan
