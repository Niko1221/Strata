// IQ4_NL down-projection benchmark: iq4nl.bin (iq4nl_ref.py) on every Vulkan device.
//
// Same shape as the engine's down projection: n_embd = 2560 rows of n_ff = 640 columns, 921600 bytes
// for the matrix.  Checks the first REF_ROWS rows against the Python reference and reports the
// effective weight bandwidth.
//
// usage: vk_iq4nl [deviceIndex] [--wg 64] [--r 4] [--repeat 8] [--iters 30] [--scalar]

#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define VKCHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d (%s)\n", (int)_r, __FILE__, __LINE__, #expr); exit(2); } } while (0)

#ifndef STRATA_GLSLC_HAS_DOT
#define STRATA_GLSLC_HAS_DOT 0
#endif

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    int only_dev = -1, repeat = 8, iters = 30;
    uint32_t wg = 32, rl = 4;
    bool force_scalar = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--scalar")) force_scalar = true;
        else if (!strcmp(argv[i], "--wg") && i + 1 < argc) wg = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--r") && i + 1 < argc) rl = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else only_dev = atoi(argv[i]);
    }

    std::ifstream in(STRATA_IQ4NL_BIN, std::ios::binary);
    if (!in) { fprintf(stderr, "iq4nl.bin not found (%s)\n", STRATA_IQ4NL_BIN); return 1; }
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    in.read((char*)hdr, sizeof(hdr));
    const uint32_t n_rows = hdr[0], n_cols = hdr[1], nb = hdr[2], ref_rows = hdr[3];
    const uint32_t n_calls = nb * 2;
    auto read_vec = [&](size_t bytes) {
        std::vector<uint8_t> v(bytes);
        in.read((char*)v.data(), (std::streamsize)bytes);
        return v;
    };
    std::vector<uint8_t> d_bytes = read_vec((size_t)n_rows * nb * 2);
    std::vector<uint8_t> qs_bytes = read_vec((size_t)n_rows * nb * 16);
    std::vector<uint8_t> x_ds = read_vec((size_t)nb * 2);
    std::vector<uint8_t> x_qs = read_vec((size_t)nb * 8 * 4);
    std::vector<uint8_t> tab_bytes = read_vec(4 * 4);
    std::vector<float> refs(ref_rows);
    in.read((char*)refs.data(), (std::streamsize)ref_rows * 4);

    const size_t weight_bytes = d_bytes.size() + qs_bytes.size();
    printf("iq4_nl down: rows=%u cols=%u nb=%u calls=%u  matrix=%.3f MB  repeat=%d\n",
           n_rows, n_cols, nb, n_calls, weight_bytes / 1e6, repeat);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "strata-vk-iq4nl";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t nphys = 0;
    vkEnumeratePhysicalDevices(inst, &nphys, nullptr);
    std::vector<VkPhysicalDevice> phys(nphys);
    vkEnumeratePhysicalDevices(inst, &nphys, phys.data());

    char spvfile[512];
    snprintf(spvfile, sizeof(spvfile), "%s/iq4nl_w%u_r%u.spv", STRATA_SPV_DIR, wg, rl);

    int worst_bad = 0;
    for (uint32_t di = 0; di < nphys; ++di) {
        if (only_dev >= 0 && (int)di != only_dev) continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys[di], &props);

        VkPhysicalDeviceShaderIntegerDotProductProperties dotp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        VkPhysicalDeviceProperties2 dp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        dp2.pNext = &dotp;
        vkGetPhysicalDeviceProperties2(phys[di], &dp2);
        VkPhysicalDeviceShaderIntegerDotProductFeatures dotf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &dotf;
        vkGetPhysicalDeviceFeatures2(phys[di], &f2);
        const bool use_dp4a = !force_scalar && dotf.shaderIntegerDotProduct == VK_TRUE && STRATA_GLSLC_HAS_DOT;
        if (use_dp4a && spvfile[0] && strstr(spvfile, "w32_r4")) snprintf(spvfile, sizeof(spvfile), "%s/iq4nl_w%u_r%u.spv", STRATA_SPV_DIR, wg, rl);

        uint32_t nqf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys[di], &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nqf);
        vkGetPhysicalDeviceQueueFamilyProperties(phys[di], &nqf, qfs.data());
        uint32_t qfam = UINT32_MAX;
        for (uint32_t q = 0; q < nqf; ++q)
            if ((qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[q].queueCount) { qfam = q; break; }
        if (qfam == UINT32_MAX) { printf("\n[%u] %s: no compute queue\n", di, props.deviceName); continue; }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot_enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        dot_enable.shaderIntegerDotProduct = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (use_dp4a) dci.pNext = &dot_enable;
        VkDevice dev = VK_NULL_HANDLE;
        VKCHECK(vkCreateDevice(phys[di], &dci, nullptr, &dev));
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, qfam, 0, &queue);

        const size_t mul = (size_t)repeat;
        const size_t off_d = 0;
        const size_t off_qs = off_d + d_bytes.size() * mul;
        const size_t off_xds = off_qs + qs_bytes.size() * mul;
        const size_t off_xqs = off_xds + x_ds.size();
        const size_t off_tab = off_xqs + x_qs.size();
        const size_t off_out = off_tab + 256 * 4 + 4 * 4;
        const size_t arena = off_out + (size_t)n_rows * mul * 4;

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = arena;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer abuf = VK_NULL_HANDLE;
        VKCHECK(vkCreateBuffer(dev, &bci, nullptr, &abuf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, abuf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_mem_type(phys[di], req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory amem = VK_NULL_HANDLE;
        VKCHECK(vkAllocateMemory(dev, &mai, nullptr, &amem));
        VKCHECK(vkBindBufferMemory(dev, abuf, amem, 0));

        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = arena;
        sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer stage = VK_NULL_HANDLE;
        VKCHECK(vkCreateBuffer(dev, &sci, nullptr, &stage));
        VkMemoryRequirements sreq{};
        vkGetBufferMemoryRequirements(dev, stage, &sreq);
        VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        smai.allocationSize = sreq.size;
        smai.memoryTypeIndex = find_mem_type(phys[di], sreq.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory smem = VK_NULL_HANDLE;
        VKCHECK(vkAllocateMemory(dev, &smai, nullptr, &smem));
        VKCHECK(vkBindBufferMemory(dev, stage, smem, 0));
        {
            void* p = nullptr;
            VKCHECK(vkMapMemory(dev, smem, 0, arena, 0, &p));
            uint8_t* base = (uint8_t*)p;
            for (int r = 0; r < repeat; ++r) {
                memcpy(base + off_d + (size_t)r * d_bytes.size(), d_bytes.data(), d_bytes.size());
                memcpy(base + off_qs + (size_t)r * qs_bytes.size(), qs_bytes.data(), qs_bytes.size());
            }
            memcpy(base + off_xds, x_ds.data(), x_ds.size());
            memcpy(base + off_xqs, x_qs.data(), x_qs.size());
            memcpy(base + off_tab + 256 * 4, tab_bytes.data(), tab_bytes.size());
            memset(base + off_out, 0, (size_t)n_rows * mul * 4);
            vkUnmapMemory(dev, smem);
        }

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = qfam;
        VkCommandPool pool = VK_NULL_HANDLE;
        VKCHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
        {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VKCHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
            VkBufferCopy cp{0, 0, arena};
            vkCmdCopyBuffer(cb, stage, abuf, 1, &cp);
            VKCHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VKCHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(dev, pool, 1, &cb);
        }

        const uint32_t n_rows_total = n_rows * (uint32_t)repeat;
        if (n_rows_total / rl > props.limits.maxComputeWorkGroupCount[0]) {
            printf("\n[%u] %s: grid too large\n", di, props.deviceName);
            continue;
        }

        VkDescriptorSetLayoutBinding binds[6]{};
        for (int b = 0; b < 6; ++b) {
            binds[b].binding = (uint32_t)b;
            binds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[b].descriptorCount = 1;
            binds[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 6;
        dslci.pBindings = binds;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = 8;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VKCHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));

        std::vector<uint32_t> spv;
        {
            std::ifstream f(spvfile, std::ios::binary | std::ios::ate);
            if (!f) { fprintf(stderr, "cannot open %s\n", spvfile); exit(1); }
            size_t n = (size_t)f.tellg();
            f.seekg(0);
            spv.resize(n / 4);
            f.read((char*)spv.data(), (std::streamsize)n);
        }
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size() * 4;
        smci.pCode = spv.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        VKCHECK(vkCreateShaderModule(dev, &smci, nullptr, &mod));
        VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci2.stage.module = mod;
        cpci2.stage.pName = "main";
        cpci2.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipe));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 2;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsa.descriptorPool = dpool;
        dsa.descriptorSetCount = 1;
        dsa.pSetLayouts = &dsl;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        VKCHECK(vkAllocateDescriptorSets(dev, &dsa, &dset));

        VkDescriptorBufferInfo bi[6];
        bi[0] = {abuf, off_d, VK_WHOLE_SIZE};
        bi[1] = {abuf, off_qs, VK_WHOLE_SIZE};
        bi[2] = {abuf, off_tab, VK_WHOLE_SIZE};
        bi[3] = {abuf, off_xds, VK_WHOLE_SIZE};
        bi[4] = {abuf, off_xqs, VK_WHOLE_SIZE};
        bi[5] = {abuf, off_out, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[6]{};
        for (int b = 0; b < 6; ++b) {
            wr[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[b].dstSet = dset;
            wr[b].dstBinding = (uint32_t)b;
            wr[b].descriptorCount = 1;
            wr[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[b].pBufferInfo = &bi[b];
        }
        vkUpdateDescriptorSets(dev, 6, wr, 0, nullptr);

        auto record = [&](VkCommandBuffer cb) {
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
            uint32_t pc[2] = {n_calls, n_rows_total};
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
            vkCmdDispatch(cb, n_rows_total / rl, 1, 1);
            VKCHECK(vkEndCommandBuffer(cb));
        };

        printf("\n[%u] %s  api=%u.%u  dot4x8 accelerated=%s\n", di, props.deviceName,
               VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
               dotp.integerDotProduct4x8BitPackedSignedAccelerated ? "yes" : "no");
        printf("    kernel: WG=%u R=%u  %s\n", wg, rl, use_dp4a ? "dotPacked4x8EXT" : "scalar expansion");

        {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VKCHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
            record(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VKCHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(dev, pool, 1, &cb);
        }
        std::vector<float> got(n_rows);
        {
            VkBufferCreateInfo rci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            rci.size = (size_t)n_rows * 4;
            rci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VkBuffer rbuf = VK_NULL_HANDLE;
            VKCHECK(vkCreateBuffer(dev, &rci, nullptr, &rbuf));
            VkMemoryRequirements rreq{};
            vkGetBufferMemoryRequirements(dev, rbuf, &rreq);
            VkMemoryAllocateInfo rmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            rmai.allocationSize = rreq.size;
            rmai.memoryTypeIndex = find_mem_type(phys[di], rreq.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkDeviceMemory rmem = VK_NULL_HANDLE;
            VKCHECK(vkAllocateMemory(dev, &rmai, nullptr, &rmem));
            VKCHECK(vkBindBufferMemory(dev, rbuf, rmem, 0));
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VKCHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
            VkBufferCopy cp{off_out, 0, (size_t)n_rows * 4};
            vkCmdCopyBuffer(cb, abuf, rbuf, 1, &cp);
            VKCHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VKCHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(queue));
            void* p = nullptr;
            VKCHECK(vkMapMemory(dev, rmem, 0, (size_t)n_rows * 4, 0, &p));
            memcpy(got.data(), p, (size_t)n_rows * 4);
            vkUnmapMemory(dev, rmem);
            vkDestroyBuffer(dev, rbuf, nullptr);
            vkFreeMemory(dev, rmem, nullptr);
            vkFreeCommandBuffers(dev, pool, 1, &cb);
        }
        double max_err = 0.0;
        int bad = 0, bad_at = -1;
        for (uint32_t r = 0; r < ref_rows; ++r) {
            double e = std::abs((double)got[r] - (double)refs[r]);
            if (e > 1e-3) { ++bad; if (bad_at < 0) bad_at = (int)r; }
            max_err = std::max(max_err, e);
        }
        printf("    check: max_err=%.6f  bad=%d/%u", max_err, bad, ref_rows);
        if (bad_at >= 0) printf("  (row %d: got %.6f want %.6f)", bad_at, got[bad_at], refs[bad_at]);
        printf("\n");
        worst_bad += bad;

        std::vector<VkCommandBuffer> cbs((size_t)iters);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = (uint32_t)iters;
        VKCHECK(vkAllocateCommandBuffers(dev, &cai, cbs.data()));
        for (int i = 0; i < iters; ++i) record(cbs[(size_t)i]);
        std::vector<VkFence> fences((size_t)iters, VK_NULL_HANDLE);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        for (int i = 0; i < iters; ++i) VKCHECK(vkCreateFence(dev, &fci, nullptr, &fences[(size_t)i]));
        {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cbs[0];
            VKCHECK(vkQueueSubmit(queue, 1, &si, fences[0]));
            VKCHECK(vkWaitForFences(dev, 1, &fences[0], VK_TRUE, UINT64_MAX));
            VKCHECK(vkResetFences(dev, 1, &fences[0]));
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cbs[(size_t)i];
            VKCHECK(vkQueueSubmit(queue, 1, &si, fences[(size_t)i]));
        }
        VKCHECK(vkWaitForFences(dev, (uint32_t)iters, fences.data(), VK_TRUE, UINT64_MAX));
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        printf("    time: %.3f ms/dispatch  (%.1f M rows/s)  %.1f GB/s of weights\n",
               ms, n_rows_total / ms / 1000.0, (double)weight_bytes * repeat / (ms * 1e6));

        for (auto f : fences) vkDestroyFence(dev, f, nullptr);
        vkFreeCommandBuffers(dev, pool, (uint32_t)iters, cbs.data());
        vkDestroyDescriptorPool(dev, dpool, nullptr);
        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyShaderModule(dev, mod, nullptr);
        vkDestroyPipelineLayout(dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr);
        vkDestroyBuffer(dev, stage, nullptr);
        vkFreeMemory(dev, smem, nullptr);
        vkDestroyBuffer(dev, abuf, nullptr);
        vkFreeMemory(dev, amem, nullptr);
        vkDestroyDevice(dev, nullptr);
    }
    vkDestroyInstance(inst, nullptr);
    printf("\n%s\n", worst_bad ? "FAILED verification" : "all devices verified");
    return worst_bad ? 1 : 0;
}
