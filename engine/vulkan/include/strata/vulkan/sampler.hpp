// sampler.hpp - Vulkan greedy sampler kernels (native engine port).
//
// Checked against the verify-window fixtures in
// engine/sycl/src/kernels/verify_parity.cpp: argmax with ties to the lowest
// index, NaN never selected, and the top token's probability.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

namespace strata::vulkan {

class Sampler {
public:
    struct Limits {
        uint32_t max_vocab = 262144;   // 248320 in this GGUF
        uint32_t max_rows = 8;         // verify windows are 1..8 rows
    };

    static std::unique_ptr<Sampler> create(VkPhysicalDevice physical, VkDevice device,
                                           uint32_t queue_family, VkQueue queue,
                                           const std::string& argmax_spv,
                                           const std::string& top_prob_spv,
                                           const Limits& lim, std::string& err);
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    /// Strict `>`: ties pick the lowest index, NaNs are skipped, and a row with
    /// no value above -inf yields 0.
    bool argmax_rows(const float* logits, uint32_t rows, uint32_t n,
                     int32_t* ids, std::string& err);

    /// exp(best - best) / sum_i exp(logits[i] - best), skipping NaNs.
    bool row_top_prob(const float* logits, uint32_t rows, uint32_t n,
                      float* probs, std::string& err);

private:
    Sampler() = default;
    bool run(VkPipeline pipe, uint32_t rows, uint32_t n, std::string& err);

    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    Limits lim_;

    VkBuffer logits_ = VK_NULL_HANDLE, ids_ = VK_NULL_HANDLE, probs_ = VK_NULL_HANDLE;
    VkDeviceMemory logits_m_ = VK_NULL_HANDLE, ids_m_ = VK_NULL_HANDLE;
    VkDeviceMemory probs_m_ = VK_NULL_HANDLE;
    float* logits_map_ = nullptr;
    int32_t* ids_map_ = nullptr;
    float* probs_map_ = nullptr;

    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline argmax_pipe_ = VK_NULL_HANDLE, top_prob_pipe_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace strata::vulkan
