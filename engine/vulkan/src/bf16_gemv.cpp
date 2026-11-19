// Vulkan BF16 GEMV. See bf16_gemv.hpp and the parity test in
// engine/sycl/src/kernels/bf16_gemv_parity.cpp.
#include "strata/vulkan/bf16_gemv.hpp"

#include <cstdio>
#include <cstring>

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
    if (vkBindBufferMemory(dev, b, m, 0) != VK_SUCCESS ||
        vkMapMemory(dev, m, 0, bytes, 0, mapped) != VK_SUCCESS) {
        vkFreeMemory(dev, m, nullptr); vkDestroyBuffer(dev, b, nullptr);
        b = VK_NULL_HANDLE; m = VK_NULL_HANDLE; return false;
    }
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
    bool ok = std::fread(data.data(), 1, size_t(n), f) == size_t(n);
    std::fclose(f);
    return ok ? data : std::vector<char>{};
}

} // namespace

Bf16Gemv::~Bf16Gemv() {
    if (device_ == VK_NULL_HANDLE) return;
    if (x_map_) vkUnmapMemory(device_, x_m_);
    if (w_map_) vkUnmapMemory(device_, w_m_);
    if (y_map_) vkUnmapMemory(device_, y_m_);
    if (pipe_) vkDestroyPipeline(device_, pipe_, nullptr);
    if (pipe_layout_) vkDestroyPipelineLayout(device_, pipe_layout_, nullptr);
    if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (desc_layout_) vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
    if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    VkBuffer bs[] = {x_, w_, y_};
    VkDeviceMemory ms[] = {x_m_, w_m_, y_m_};
    for (int i = 0; i < 3; ++i) {
        if (bs[i]) vkDestroyBuffer(device_, bs[i], nullptr);
        if (ms[i]) vkFreeMemory(device_, ms[i], nullptr);
    }
}

std::unique_ptr<Bf16Gemv> Bf16Gemv::create(VkPhysicalDevice physical, VkDevice device,
                                           uint32_t queue_family, VkQueue queue,
                                           const std::string& spv,
                                           const Limits& lim, std::string& err) {
    auto g = std::unique_ptr<Bf16Gemv>(new Bf16Gemv());
    g->physical_ = physical; g->device_ = device; g->queue_ = queue; g->lim_ = lim;

    const VkDeviceSize words_in = lim.max_in;
    const VkDeviceSize words_w = VkDeviceSize(lim.max_out) * lim.max_in;
    if (!buffer(physical, device, words_in * sizeof(uint32_t),
                g->x_, g->x_m_, reinterpret_cast<void**>(&g->x_map_)) ||
        !buffer(physical, device, words_w * sizeof(uint32_t),
                g->w_, g->w_m_, reinterpret_cast<void**>(&g->w_map_)) ||
        !buffer(physical, device, VkDeviceSize(lim.max_out) * sizeof(float),
                g->y_, g->y_m_, reinterpret_cast<void**>(&g->y_map_))) {
        err = "cannot allocate BF16 GEMV buffers"; return nullptr;
    }
    g->x_words_.resize(size_t(words_in));
    g->w_words_.resize(size_t(words_w));

    VkDescriptorSetLayoutBinding b[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3; dli.pBindings = b;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &g->desc_layout_) != VK_SUCCESS) {
        err = "vkCreateDescriptorSetLayout failed"; return nullptr;
    }
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(device, &pi, nullptr, &g->desc_pool_) != VK_SUCCESS) {
        err = "vkCreateDescriptorPool failed"; return nullptr;
    }
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g->desc_pool_; dai.descriptorSetCount = 1;
    dai.pSetLayouts = &g->desc_layout_;
    if (vkAllocateDescriptorSets(device, &dai, &g->set_) != VK_SUCCESS) {
        err = "vkAllocateDescriptorSets failed"; return nullptr;
    }
    VkDescriptorBufferInfo bi[3] = {
        {g->x_, 0, VK_WHOLE_SIZE}, {g->w_, 0, VK_WHOLE_SIZE}, {g->y_, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = g->set_; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device, 3, w, 0, nullptr);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 8;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &g->desc_layout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &g->pipe_layout_) != VK_SUCCESS) {
        err = "vkCreatePipelineLayout failed"; return nullptr;
    }

    auto code = read_spv(spv);
    if (code.empty()) { err = "cannot read shader: " + spv; return nullptr; }
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = code.size();
    sci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &sci, nullptr, &mod) != VK_SUCCESS) {
        err = "vkCreateShaderModule failed: " + spv; return nullptr;
    }
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod; cp.stage.pName = "main";
    cp.layout = g->pipe_layout_;
    bool ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &g->pipe_) == VK_SUCCESS;
    vkDestroyShaderModule(device, mod, nullptr);
    if (!ok) { err = "vkCreateComputePipelines failed: " + spv; return nullptr; }

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fi, nullptr, &g->fence_) != VK_SUCCESS) {
        err = "vkCreateFence failed"; return nullptr;
    }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = queue_family;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &cpi, nullptr, &g->cmd_pool_) != VK_SUCCESS) {
        err = "vkCreateCommandPool failed"; return nullptr;
    }
    return g;
}

bool Bf16Gemv::gemv(const uint16_t* x, const uint16_t* w, float* y,
                    uint32_t n_in, uint32_t n_out, std::string& err) {
    if (n_in == 0 || n_in > lim_.max_in || n_out == 0 || n_out > lim_.max_out) {
        err = "bf16 gemv: shape out of range"; return false;
    }
    for (uint32_t i = 0; i < n_in; ++i) x_words_[i] = x[i];
    const size_t nw = size_t(n_in) * n_out;
    for (size_t i = 0; i < nw; ++i) w_words_[i] = w[i];
    std::memcpy(x_map_, x_words_.data(), size_t(n_in) * sizeof(uint32_t));
    std::memcpy(w_map_, w_words_.data(), nw * sizeof(uint32_t));

    const uint32_t pc[2] = {n_in, n_out};
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS) {
        err = "vkAllocateCommandBuffers failed"; return false;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_,
                            0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(cmd, n_out, 1, 1);
    vkEndCommandBuffer(cmd);

    vkResetFences(device_, 1, &fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    bool ok = vkQueueSubmit(queue_, 1, &si, fence_) == VK_SUCCESS &&
              vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (ok) std::memcpy(y, y_map_, size_t(n_out) * sizeof(float));
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    if (!ok) { err = "Vulkan BF16 GEMV submit failed"; return false; }
    return true;
}

} // namespace strata::vulkan
