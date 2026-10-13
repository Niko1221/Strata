// Strata Vulkan probe step 2: run one expert-GEMV dispatch on every GPU the
// instance sees, check it against a CPU reference, and time it.
// The compute shader is built from gemv.comp by glslc at build time.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <cmath>

#ifndef STRATA_GEMV_SPV
#define STRATA_GEMV_SPV "gemv.spv"
#endif

static const int ROWS = 4096;      // expert rows per dispatch
static const int K    = 1024;      // hidden dim (bytes)
static const int WG   = 32;        // workgroup size, specialized in the shader

#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    printf("FAIL %s: VkResult %d\n", #x, (int)r_); return 2; } } while (0)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int only = -1;                       // -1 = every device; otherwise this Vulkan index
    if (argc > 1) only = atoi(argv[1]);
    std::vector<uint32_t> spv;
    {
        FILE* f = fopen(STRATA_GEMV_SPV, "rb");
        if (!f) { printf("missing %s (build with build_probe.bat)\n", STRATA_GEMV_SPV); return 1; }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        spv.resize((size_t)(n + 3) / 4);
        if (fread(spv.data(), 1, (size_t)n, f) != (size_t)n) { fclose(f); printf("short read\n"); return 1; }
        fclose(f);
    }

    std::vector<int8_t> w((size_t)ROWS * K), x(K);
    std::vector<float> s(ROWS), ref(ROWS), got(ROWS);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d8(-127, 127);
    for (auto& v : w) v = (int8_t)d8(rng);
    for (auto& v : x) v = (int8_t)d8(rng);
    for (int r = 0; r < ROWS; ++r) {
        s[r] = 0.001f * (float)(r % 97 + 1);
        int acc = 0;
        for (int k = 0; k < K; ++k) acc += (int)w[(size_t)r * K + k] * (int)x[k];
        ref[r] = s[r] * (float)acc;
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "strata-vk-gemv";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    for (uint32_t di = 0; di < n; ++di) {
        if (only >= 0 && (int)di != only) continue;
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[di], &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            p.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) continue;

        uint32_t qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &qf, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qf);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &qf, qs.data());
        int cq = -1;
        for (uint32_t i = 0; i < qf; ++i)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { cq = (int)i; break; }
        if (cq < 0) { printf("[%u] %s: no compute queue, skip\n", di, p.deviceName); continue; }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = (uint32_t)cq;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        VkDevice dev = VK_NULL_HANDLE;
        if (vkCreateDevice(devs[di], &dci, nullptr, &dev) != VK_SUCCESS) {
            printf("[%u] %s: vkCreateDevice failed, skip\n", di, p.deviceName);
            continue;
        }
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, (uint32_t)cq, 0, &queue);

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(devs[di], &mp);
        auto find_mem = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
                if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
            return 0u;
        };

        // returns 0 on success, 2 on failure (so VKCHECK can return from it)
        auto make_buf = [&](VkBufferUsageFlags usage, const void* src, size_t bytes,
                            VkBuffer& buf, VkDeviceMemory& mem) -> int {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = bytes;
            bci.usage = usage;
            VKCHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(dev, buf, &req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = find_mem(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VKCHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
            VKCHECK(vkBindBufferMemory(dev, buf, mem, 0));
            if (!src) return 0;

            VkBuffer staging = VK_NULL_HANDLE;
            VkDeviceMemory smem = VK_NULL_HANDLE;
            VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            sci.size = bytes;
            sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VKCHECK(vkCreateBuffer(dev, &sci, nullptr, &staging));
            VkMemoryRequirements sreq{};
            vkGetBufferMemoryRequirements(dev, staging, &sreq);
            VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            sai.allocationSize = sreq.size;
            sai.memoryTypeIndex = find_mem(sreq.memoryTypeBits,
                (VkMemoryPropertyFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            VKCHECK(vkAllocateMemory(dev, &sai, nullptr, &smem));
            VKCHECK(vkBindBufferMemory(dev, staging, smem, 0));
            void* map = nullptr;
            VKCHECK(vkMapMemory(dev, smem, 0, bytes, 0, &map));
            memcpy(map, src, bytes);
            vkUnmapMemory(dev, smem);

            VkCommandPool tmp_pool = VK_NULL_HANDLE;
            VkCommandPoolCreateInfo tpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            tpci.queueFamilyIndex = (uint32_t)cq;
            VKCHECK(vkCreateCommandPool(dev, &tpci, nullptr, &tmp_pool));
            VkCommandBuffer tcb = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo tcai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            tcai.commandPool = tmp_pool;
            tcai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            tcai.commandBufferCount = 1;
            VKCHECK(vkAllocateCommandBuffers(dev, &tcai, &tcb));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VKCHECK(vkBeginCommandBuffer(tcb, &bi));
            VkBufferCopy cr{};
            cr.size = bytes;
            vkCmdCopyBuffer(tcb, staging, buf, 1, &cr);
            VKCHECK(vkEndCommandBuffer(tcb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &tcb;
            VKCHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(queue));
            vkDestroyCommandPool(dev, tmp_pool, nullptr);
            vkDestroyBuffer(dev, staging, nullptr);
            vkFreeMemory(dev, smem, nullptr);
            return 0;
        };

        VkBuffer wb = VK_NULL_HANDLE, xb = VK_NULL_HANDLE, sb = VK_NULL_HANDLE, ob = VK_NULL_HANDLE;
        VkDeviceMemory wm = VK_NULL_HANDLE, xm = VK_NULL_HANDLE, sm = VK_NULL_HANDLE, om = VK_NULL_HANDLE;
        const VkBufferUsageFlags sbuf = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const VkBufferUsageFlags obuf = sbuf | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (make_buf(sbuf, w.data(), w.size(), wb, wm)) return 2;
        if (make_buf(sbuf, x.data(), x.size(), xb, xm)) return 2;
        if (make_buf(sbuf, s.data(), s.size() * sizeof(float), sb, sm)) return 2;
        if (make_buf(obuf, nullptr, (size_t)ROWS * sizeof(float), ob, om)) return 2;

        // readback buffer: host-visible, the output is copied into it after the dispatch
        VkBuffer rbuf = VK_NULL_HANDLE;
        VkDeviceMemory rmem = VK_NULL_HANDLE;
        {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = (size_t)ROWS * sizeof(float);
            bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VKCHECK(vkCreateBuffer(dev, &bci, nullptr, &rbuf));
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(dev, rbuf, &req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = find_mem(req.memoryTypeBits,
                (VkMemoryPropertyFlags)(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            VKCHECK(vkAllocateMemory(dev, &ai, nullptr, &rmem));
            VKCHECK(vkBindBufferMemory(dev, rbuf, rmem, 0));
        }

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size() * 4;
        smci.pCode = spv.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &mod));

        // The set layout must exist before the pipeline layout that names it:
        // a layout without the shader's set 0 leaves the dispatch undefined
        // (reads of whatever the driver picks: wrong results, then device lost).
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

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = mod;
        cpci.stage.pName = "main";
        cpci.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
        if (pr != VK_SUCCESS) {
            printf("[%u] %s: vkCreateComputePipelines failed (%d)\n", di, p.deviceName, (int)pr);
            vkDestroyShaderModule(dev, mod, nullptr);
            vkDestroyDevice(dev, nullptr);
            continue;
        }

        VkDescriptorPoolSize psizes[1] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = psizes;
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

        VkDescriptorBufferInfo dbi[4] = {
            {wb, 0, VK_WHOLE_SIZE}, {xb, 0, VK_WHOLE_SIZE},
            {sb, 0, VK_WHOLE_SIZE}, {ob, 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[4]{};
        for (int i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dset;
            writes[i].dstBinding = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(dev, 4, writes, 0, nullptr);

        VkCommandPool cp = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci2.queueFamilyIndex = (uint32_t)cq;
        VKCHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cp));

        // One command buffer per timing iteration, each recorded once and submitted
        // together: re-recording one still-pending buffer is a spec violation that
        // faults the GPU (validation error, then DEVICE_LOST and a TDR).
        const int ITERS = 50;
        std::vector<VkCommandBuffer> cbs(ITERS, VK_NULL_HANDLE);
        VkCommandBufferAllocateInfo cai2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai2.commandPool = cp;
        cai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai2.commandBufferCount = (uint32_t)ITERS;
        VKCHECK(vkAllocateCommandBuffers(dev, &cai2, cbs.data()));

        auto record = [&](VkCommandBuffer cb) -> int {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &bi));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
            vkCmdDispatch(cb, (uint32_t)ROWS, 1, 1);
            // the readback copy needs a barrier: dispatch writes are not visible to transfer reads otherwise
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            VkBufferCopy rc{};
            rc.size = (size_t)ROWS * sizeof(float);
            vkCmdCopyBuffer(cb, ob, rbuf, 1, &rc);
            VKCHECK(vkEndCommandBuffer(cb));
            return 0;
        };

        // correctness: one dispatch, then read the result back
        if (record(cbs[0])) return 2;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = cbs.data();
        VKCHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VKCHECK(vkQueueWaitIdle(queue));

        {
            void* map = nullptr;
            VKCHECK(vkMapMemory(dev, rmem, 0, (size_t)ROWS * sizeof(float), 0, &map));
            memcpy(got.data(), map, (size_t)ROWS * sizeof(float));
            vkUnmapMemory(dev, rmem);
        }

        double max_err = 0.0;
        int bad = 0;
        for (int r = 0; r < ROWS; ++r) {
            double e = fabs((double)got[r] - (double)ref[r]);
            if (e > 1e-3) ++bad;
            if (e > max_err) max_err = e;
        }

        for (int i = 1; i < ITERS; ++i)
            if (record(cbs[i])) return 2;
        VkSubmitInfo si_many{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si_many.commandBufferCount = (uint32_t)ITERS;
        si_many.pCommandBuffers = cbs.data();
        auto t0 = std::chrono::steady_clock::now();
        VKCHECK(vkQueueSubmit(queue, 1, &si_many, VK_NULL_HANDLE));
        VKCHECK(vkQueueWaitIdle(queue));
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

        printf("[%u] %-28s max_err=%.4f bad=%d/%d | %.3f ms/dispatch  %.1f Gint-op/s\n",
               di, p.deviceName, max_err, bad, ROWS, ms,
               (double)ROWS * K * 2 / (ms * 1e6));

        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyPipelineLayout(dev, layout, nullptr);
        vkDestroyShaderModule(dev, mod, nullptr);
        vkDestroyDescriptorPool(dev, dpool, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyCommandPool(dev, cp, nullptr);
        vkDestroyBuffer(dev, wb, nullptr);
        vkDestroyBuffer(dev, xb, nullptr);
        vkDestroyBuffer(dev, sb, nullptr);
        vkDestroyBuffer(dev, ob, nullptr);
        vkDestroyBuffer(dev, rbuf, nullptr);
        vkFreeMemory(dev, wm, nullptr);
        vkFreeMemory(dev, xm, nullptr);
        vkFreeMemory(dev, sm, nullptr);
        vkFreeMemory(dev, om, nullptr);
        vkFreeMemory(dev, rmem, nullptr);
        vkDestroyDevice(dev, nullptr);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
