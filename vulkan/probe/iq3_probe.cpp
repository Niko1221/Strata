// Vulkan IQ3_XXS expert-row kernel: correctness against the engine's integer reference.
//
// Reads iq3_test.bin (written by iq3_ref.py) and runs iq3_gu.comp on each visible device,
// then compares with the reference the Python side computed by transcribing
// Split<18>::load/apply from src/kernels/cuda/iq_kernels.cu.  Because every step on both
// sides is int32 arithmetic with C truncation, the expected error is exactly zero - a
// nonzero max_err means a real transcription difference, not floating-point noise.
//
// Usage: vk_iq3 [device-index]   (-1 or absent: every device with a compute queue)

#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef STRATA_IQ3_SPV
#define STRATA_IQ3_SPV "iq3_gu.spv"
#endif

#define VKCHECK(x)                                                            \
    do {                                                                      \
        VkResult r_ = (x);                                                    \
        if (r_ != VK_SUCCESS) {                                               \
            printf("FAIL %s: VkResult %d\n", #x, (int)r_);                    \
            return 2;                                                         \
        }                                                                     \
    } while (0)

namespace {

uint32_t find_memory(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

bool make_buf(VkPhysicalDevice pd, VkDevice dev, VkDeviceSize size, const void* data, Buf& out) {
    out.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, nullptr, &out.buf) != VK_SUCCESS) return false;

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, out.buf, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    const uint32_t mt = find_memory(pd, mr.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == ~0u) return false;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(dev, &mai, nullptr, &out.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(dev, out.buf, out.mem, 0) != VK_SUCCESS) return false;

    if (data) {
        void* p = nullptr;
        if (vkMapMemory(dev, out.mem, 0, size, 0, &p) != VK_SUCCESS) return false;
        std::memcpy(p, data, (size_t)size);
        vkUnmapMemory(dev, out.mem);
    }
    return true;
}

void drop_buf(VkDevice dev, Buf& b) {
    if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
    b.buf = VK_NULL_HANDLE;
    b.mem = VK_NULL_HANDLE;
}

bool slurp(const char* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)n);
    const bool ok = fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int only = -1;
    if (argc > 1) only = atoi(argv[1]);

    std::vector<uint32_t> spv;
    {
        std::vector<uint8_t> raw;
        if (!slurp(STRATA_IQ3_SPV, raw)) {
            printf("missing %s (run build_probe.bat)\n", STRATA_IQ3_SPV);
            return 1;
        }
        spv.assign((const uint32_t*)raw.data(), (const uint32_t*)(raw.data() + raw.size()));
    }

    std::vector<uint8_t> test;
    if (!slurp("iq3_test.bin", test)) {
        printf("missing iq3_test.bin (run: python iq3_ref.py)\n");
        return 1;
    }
    uint32_t hdr[3];
    std::memcpy(hdr, test.data(), sizeof(hdr));
    const uint32_t n_embd = hdr[0], n_ff = hdr[1], n_blocks = hdr[2];
    const size_t row_bytes = (size_t)n_blocks * 98;
    const size_t xq_bytes = (size_t)(n_embd / 32) * 36;

    size_t off = sizeof(hdr);
    const uint8_t* blob = test.data() + off; off += (size_t)n_ff * row_bytes;
    const uint8_t* xq = test.data() + off;   off += xq_bytes;
    const float* ref = (const float*)(test.data() + off);

    std::vector<uint8_t> cbraw;
    if (!slurp("iq3_cb.bin", cbraw) || cbraw.size() != 1024) {
        printf("missing or bad iq3_cb.bin (run: python iq3_ref.py)\n");
        return 1;
    }

    printf("IQ3_XXS expert row: n_embd=%u n_ff=%u blocks=%u row_bytes=%zu\n",
           n_embd, n_ff, n_blocks, row_bytes);

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "strata-vulkan-iq3";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 0;
    VKCHECK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
    if (!nd) { printf("no Vulkan devices\n"); return 1; }
    std::vector<VkPhysicalDevice> devs(nd);
    VKCHECK(vkEnumeratePhysicalDevices(inst, &nd, devs.data()));

    for (uint32_t di = 0; di < nd; ++di) {
        if (only >= 0 && (int)di != only) continue;

        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[di], &p);

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, qs.data());
        int cq = -1;
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { cq = (int)i; break; }
        if (cq < 0) { printf("[%u] %s: no compute queue, skip\n", di, p.deviceName); continue; }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = (uint32_t)cq;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        VkDevice dev = VK_NULL_HANDLE;
        VkResult dr = vkCreateDevice(devs[di], &dci, nullptr, &dev);
        if (dr != VK_SUCCESS) { printf("[%u] %s: vkCreateDevice failed (%d)\n", di, p.deviceName, (int)dr); continue; }
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, (uint32_t)cq, 0, &queue);

        Buf bb{}, xb{}, cbb{}, ob{};
        if (!make_buf(devs[di], dev, row_bytes * n_ff, blob, bb) ||
            !make_buf(devs[di], dev, xq_bytes, xq, xb) ||
            !make_buf(devs[di], dev, 1024, cbraw.data(), cbb) ||
            !make_buf(devs[di], dev, sizeof(float) * n_ff, nullptr, ob)) {
            printf("[%u] %s: buffer allocation failed\n", di, p.deviceName);
            drop_buf(dev, bb); drop_buf(dev, xb); drop_buf(dev, cbb); drop_buf(dev, ob);
            vkDestroyDevice(dev, nullptr);
            continue;
        }

        VkDescriptorSetLayoutBinding binds[4]{};
        for (int i = 0; i < 4; ++i) {
            binds[i].binding = (uint32_t)i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 4;
        dslci.pBindings = binds;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(uint32_t);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size() * 4;
        smci.pCode = spv.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &mod));

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = mod;
        cpci.stage.pName = "main";
        cpci.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
        if (pr != VK_SUCCESS) {
            printf("[%u] %s: vkCreateComputePipelines failed (%d)\n", di, p.deviceName, (int)pr);
            drop_buf(dev, bb); drop_buf(dev, xb); drop_buf(dev, cbb); drop_buf(dev, ob);
            vkDestroyShaderModule(dev, mod, nullptr);
            vkDestroyPipelineLayout(dev, layout, nullptr);
            vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
            vkDestroyDevice(dev, nullptr);
            continue;
        }

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

        VkDescriptorBufferInfo bi[4] = {{bb.buf, 0, VK_WHOLE_SIZE}, {xb.buf, 0, VK_WHOLE_SIZE},
                                        {cbb.buf, 0, VK_WHOLE_SIZE}, {ob.buf, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[4]{};
        for (int i = 0; i < 4; ++i) {
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = dset;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev, 4, wr, 0, nullptr);

        VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci2.queueFamilyIndex = (uint32_t)cq;
        cpci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool cpool = VK_NULL_HANDLE;
        VKCHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cpool));

        const int ITERS = 20;
        std::vector<VkCommandBuffer> cmds(ITERS);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = (uint32_t)ITERS;
        VKCHECK(vkAllocateCommandBuffers(dev, &cai, cmds.data()));

        // One recorded buffer per iteration: a command buffer must not be re-recorded while
        // it is pending, and re-submitting one buffer without a fence is exactly that.
        for (int i = 0; i < ITERS; ++i) {
            VkCommandBufferBeginInfo bi2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cmds[i], &bi2));
            vkCmdBindPipeline(cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
            vkCmdPushConstants(cmds[i], layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n_blocks);
            vkCmdDispatch(cmds[i], n_ff, 1, 1);
            VKCHECK(vkEndCommandBuffer(cmds[i]));
        }

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmds[0];
        VKCHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VKCHECK(vkQueueWaitIdle(queue));

        std::vector<float> got(n_ff);
        {
            void* map = nullptr;
            VKCHECK(vkMapMemory(dev, ob.mem, 0, sizeof(float) * n_ff, 0, &map));
            std::memcpy(got.data(), map, sizeof(float) * n_ff);
            vkUnmapMemory(dev, ob.mem);
        }

        double max_err = 0.0;
        int bad = 0;
        for (uint32_t r = 0; r < n_ff; ++r) {
            const double e = std::fabs((double)got[r] - (double)ref[r]);
            if (e > 0.0) ++bad;
            if (e > max_err) max_err = e;
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            VkSubmitInfo s2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            s2.commandBufferCount = 1;
            s2.pCommandBuffers = &cmds[i];
            VKCHECK(vkQueueSubmit(queue, 1, &s2, VK_NULL_HANDLE));
        }
        VKCHECK(vkQueueWaitIdle(queue));
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        const double wbytes = (double)row_bytes * n_ff;

        printf("[%u] %-28s max_err=%.6f bad=%d/%u | %.3f ms/row-batch  %.1f GB/s\n",
               di, p.deviceName, max_err, bad, n_ff, ms, wbytes / (ms * 1e6));

        drop_buf(dev, bb); drop_buf(dev, xb); drop_buf(dev, cbb); drop_buf(dev, ob);
        vkDestroyCommandPool(dev, cpool, nullptr);
        vkDestroyDescriptorPool(dev, pool, nullptr);
        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyShaderModule(dev, mod, nullptr);
        vkDestroyPipelineLayout(dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyDevice(dev, nullptr);
    }

    vkDestroyInstance(inst, nullptr);
    return 0;
}
