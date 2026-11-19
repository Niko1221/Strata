#include "strata/vulkan/router.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace strata::vulkan {
namespace {

bool find_mem(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props, uint32_t& out) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    // On the B390 the cached host-visible type (0xf) is ~2.4x faster for the
    // CPU-side streams and mapped logits than the plain coherent type (0x7).
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) &&
                (fl & (props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
                    == (props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                out = i; return true;
            }
        }
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            out = i; return true;
        }
    return false;
}

bool buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize bytes,
            VkBuffer& b, VkDeviceMemory& m, void** mapped) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = bytes;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &ci, nullptr, &b) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b, &req);
    uint32_t type = 0;
    if (!find_mem(phys, req.memoryTypeBits,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, type)) {
        vkDestroyBuffer(dev, b, nullptr); b = VK_NULL_HANDLE; return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = type;
    if (vkAllocateMemory(dev, &ai, nullptr, &m) != VK_SUCCESS) {
        vkDestroyBuffer(dev, b, nullptr); b = VK_NULL_HANDLE; return false;
    }
    if (vkBindBufferMemory(dev, b, m, 0) != VK_SUCCESS) {
        vkFreeMemory(dev, m, nullptr); vkDestroyBuffer(dev, b, nullptr);
        b = VK_NULL_HANDLE; m = VK_NULL_HANDLE; return false;
    }
    if (mapped && vkMapMemory(dev, m, 0, bytes, 0, mapped) != VK_SUCCESS) return false;
    return true;
}

std::vector<char> read_spv(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); return {}; }
    std::vector<char> data(static_cast<size_t>(n));
    bool ok = fread(data.data(), 1, size_t(n), f) == size_t(n);
    std::fclose(f);
    return ok ? data : std::vector<char>{};
}

} // namespace

Router::~Router() {
    if (device_ == VK_NULL_HANDLE) return;
    if (w_map_) vkUnmapMemory(device_, w_m_);
    if (x_map_) vkUnmapMemory(device_, x_m_);
    if (logits_map_) vkUnmapMemory(device_, logits_m_);
    if (ids_map_) vkUnmapMemory(device_, ids_m_);
    if (weights_map_) vkUnmapMemory(device_, weights_m_);
    if (gemv_pipe_) vkDestroyPipeline(device_, gemv_pipe_, nullptr);
    if (topk_pipe_) vkDestroyPipeline(device_, topk_pipe_, nullptr);
    if (pipe_layout_) vkDestroyPipelineLayout(device_, pipe_layout_, nullptr);
    if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (desc_layout_) vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
    if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    VkBuffer bs[] = {w_, x_, logits_, ids_, weights_};
    VkDeviceMemory ms[] = {w_m_, x_m_, logits_m_, ids_m_, weights_m_};
    for (int i = 0; i < 5; ++i) {
        if (bs[i]) vkDestroyBuffer(device_, bs[i], nullptr);
        if (ms[i]) vkFreeMemory(device_, ms[i], nullptr);
    }
}

bool Router::make_pipeline(const std::string& spv, VkPipeline& out, std::string& err) {
    auto code = read_spv(spv);
    if (code.empty()) { err = "cannot read shader: " + spv; return false; }
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = code.size();
    sci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &sci, nullptr, &mod) != VK_SUCCESS) {
        err = "vkCreateShaderModule failed: " + spv; return false;
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName = "main";
    ci.layout = pipe_layout_;
    bool ok = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &out) == VK_SUCCESS;
    if (!ok) err = "vkCreateComputePipelines failed: " + spv;
    vkDestroyShaderModule(device_, mod, nullptr);
    return ok;
}

std::unique_ptr<Router> Router::create(VkPhysicalDevice physical, VkDevice device,
                                       uint32_t queue_family, VkQueue queue,
                                       const std::string& gemv_spv,
                                       const std::string& topk_spv,
                                       const Limits& lim, std::string& err) {
    auto r = std::unique_ptr<Router>(new Router());
    r->physical_ = physical; r->device_ = device; r->queue_ = queue; r->lim_ = lim;

    if (!buffer(physical, device, VkDeviceSize(lim.max_experts) * lim.max_k * sizeof(float),
                r->w_, r->w_m_, reinterpret_cast<void**>(&r->w_map_)) ||
        !buffer(physical, device, VkDeviceSize(lim.max_k) * sizeof(float),
                r->x_, r->x_m_, reinterpret_cast<void**>(&r->x_map_)) ||
        !buffer(physical, device, VkDeviceSize(lim.max_tokens) * lim.max_experts * sizeof(float),
                r->logits_, r->logits_m_, reinterpret_cast<void**>(&r->logits_map_)) ||
        !buffer(physical, device, VkDeviceSize(lim.max_tokens) * lim.max_top_k * sizeof(int32_t),
                r->ids_, r->ids_m_, reinterpret_cast<void**>(&r->ids_map_)) ||
        !buffer(physical, device, VkDeviceSize(lim.max_tokens) * lim.max_top_k * sizeof(float),
                r->weights_, r->weights_m_, reinterpret_cast<void**>(&r->weights_map_))) {
        err = "cannot allocate router buffers"; return nullptr;
    }

    VkDescriptorSetLayoutBinding b[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3; dli.pBindings = b;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &r->desc_layout_) != VK_SUCCESS) {
        err = "vkCreateDescriptorSetLayout failed"; return nullptr;
    }
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 2; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(device, &pi, nullptr, &r->desc_pool_) != VK_SUCCESS) {
        err = "vkCreateDescriptorPool failed"; return nullptr;
    }
    VkDescriptorSetLayout layouts[2] = {r->desc_layout_, r->desc_layout_};
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = r->desc_pool_; dai.descriptorSetCount = 2; dai.pSetLayouts = layouts;
    VkDescriptorSet sets[2] = {};
    if (vkAllocateDescriptorSets(device, &dai, sets) != VK_SUCCESS) {
        err = "vkAllocateDescriptorSets failed"; return nullptr;
    }
    r->set_gemv_ = sets[0]; r->set_topk_ = sets[1];

    auto bind = [&](VkDescriptorSet set, uint32_t i, VkBuffer buf) {
        VkDescriptorBufferInfo bi{buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set; w.dstBinding = i; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    };
    bind(r->set_gemv_, 0, r->w_);
    bind(r->set_gemv_, 1, r->x_);
    bind(r->set_gemv_, 2, r->logits_);
    bind(r->set_topk_, 0, r->logits_);
    bind(r->set_topk_, 1, r->ids_);
    bind(r->set_topk_, 2, r->weights_);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 16;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &r->desc_layout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &r->pipe_layout_) != VK_SUCCESS) {
        err = "vkCreatePipelineLayout failed"; return nullptr;
    }
    if (!r->make_pipeline(gemv_spv, r->gemv_pipe_, err) ||
        !r->make_pipeline(topk_spv, r->topk_pipe_, err)) return nullptr;

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fi, nullptr, &r->fence_) != VK_SUCCESS) {
        err = "vkCreateFence failed"; return nullptr;
    }
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cp.queueFamilyIndex = queue_family;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &cp, nullptr, &r->cmd_pool_) != VK_SUCCESS) {
        err = "vkCreateCommandPool failed"; return nullptr;
    }
    return r;
}

void Router::run(VkPipeline pipe, VkDescriptorSet set, const uint32_t pc[4], uint32_t groups_x) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS) return;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cmd, groups_x, 1, 1);
    vkEndCommandBuffer(cmd);
    vkResetFences(device_, 1, &fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue_, 1, &si, fence_) == VK_SUCCESS)
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
}

bool Router::gemv(const float* W, const float* x, float* logits,
                  uint32_t n_experts, uint32_t width, std::string& err) {
    if (n_experts == 0 || n_experts > lim_.max_experts || width == 0 || width > lim_.max_k) {
        err = "router gemv: shape out of range"; return false;
    }
    std::memcpy(w_map_, W, size_t(n_experts) * width * sizeof(float));
    std::memcpy(x_map_, x, size_t(width) * sizeof(float));
    const uint32_t pc[4] = {n_experts, width, 0, 0};
    run(gemv_pipe_, set_gemv_, pc, n_experts);
    std::memcpy(logits, logits_map_, size_t(n_experts) * sizeof(float));
    return true;
}

bool Router::gemv_topk(const float* W, const float* x, float* logits,
                       uint32_t n_experts, uint32_t width, uint32_t top_k,
                       int32_t* ids, float* weights, std::string& err) {
    if (n_experts == 0 || n_experts > lim_.max_experts || width == 0 || width > lim_.max_k ||
        top_k == 0 || top_k > lim_.max_top_k || top_k > n_experts) {
        err = "router gemv_topk: shape out of range"; return false;
    }
    std::memcpy(w_map_, W, size_t(n_experts) * width * sizeof(float));
    std::memcpy(x_map_, x, size_t(width) * sizeof(float));

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS) {
        err = "router gemv_topk: command buffer failed"; return false;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    const uint32_t gpc[4] = {n_experts, width, 0, 0};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gemv_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set_gemv_, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, gpc);
    vkCmdDispatch(cmd, n_experts, 1, 1);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    const uint32_t tpc[4] = {n_experts, top_k, 1, 0};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, topk_pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set_topk_, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, tpc);
    vkCmdDispatch(cmd, 1, 1, 1);
    vkEndCommandBuffer(cmd);

    vkResetFences(device_, 1, &fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    bool ok = vkQueueSubmit(queue_, 1, &si, fence_) == VK_SUCCESS &&
              vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    if (!ok) { err = "router gemv_topk: submission failed"; return false; }

    std::memcpy(logits, logits_map_, size_t(n_experts) * sizeof(float));
    std::memcpy(ids, ids_map_, size_t(top_k) * sizeof(int32_t));
    std::memcpy(weights, weights_map_, size_t(top_k) * sizeof(float));
    return true;
}

bool Router::topk(const float* logits, uint32_t n_tokens, uint32_t n_experts,
                  uint32_t top_k, int32_t* ids, float* weights, std::string& err) {
    if (n_experts == 0 || n_experts > lim_.max_experts ||
        top_k == 0 || top_k > lim_.max_top_k || top_k > n_experts ||
        n_tokens == 0 || n_tokens > lim_.max_tokens) {
        err = "router topk: shape out of range"; return false;
    }
    std::memcpy(logits_map_, logits, size_t(n_tokens) * n_experts * sizeof(float));
    const uint32_t pc[4] = {n_experts, top_k, n_tokens, 0};
    run(topk_pipe_, set_topk_, pc, n_tokens);
    std::memcpy(ids, ids_map_, size_t(n_tokens) * top_k * sizeof(int32_t));
    std::memcpy(weights, weights_map_, size_t(n_tokens) * top_k * sizeof(float));
    return true;
}

} // namespace strata::vulkan
