// Fast-path IQ3_XXS expert-row benchmark.
//
// Loads iq_bench.bin (iq_bench_gen.py: one expert's gate+up, 1280 rows of n_embd = 2560, in the
// split d/aux/qs layout), runs the kernel on every Vulkan device it can find, checks the first
// REF_ROWS rows against the Python reference and reports the effective weight bandwidth.
//
// The number that matters is GB/s of weight bytes: per decoded token the engine walks ~318
// experts of ~2.18 MB each, so a helper device is only worth wiring in if it can move those bytes
// fast enough to stay off the critical path.
//
// usage: vk_iqbench [deviceIndex] [--scalar] [--wg 64] [--repeat N] [--iters N]

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

static uint32_t find_mem_type(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

struct Dev {
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mp{};
    bool dot_signed = false, dot_accel = false;
    std::string name;
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    int only_dev = -1;
    bool force_scalar = false, fixed = false;
    uint32_t rows_per_lane = 4;
    uint32_t wg = 32;
    int repeat = 8;
    int iters = 30;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--scalar")) force_scalar = true;
        else if (!strcmp(argv[i], "--fixed")) fixed = true;
        else if (!strcmp(argv[i], "--r") && i + 1 < argc) rows_per_lane = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wg") && i + 1 < argc) wg = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else only_dev = atoi(argv[i]);
    }

    // ---- the test data ----
    std::ifstream in(STRATA_IQBENCH_BIN, std::ios::binary);
    if (!in) { fprintf(stderr, "iq_bench.bin not found (%s) - run iq_bench_gen.py first\n", STRATA_IQBENCH_BIN); return 1; }
    uint32_t hdr[4] = {0, 0, 0, 0};
    in.read((char*)hdr, sizeof(hdr));
    const uint32_t n_rows = hdr[0], n_embd = hdr[1], nb = hdr[2], ref_rows = hdr[3];
    const uint32_t n_chunks = nb * 8;
    auto read_vec = [&](size_t bytes) {
        std::vector<uint8_t> v(bytes);
        in.read((char*)v.data(), (std::streamsize)bytes);
        return v;
    };
    std::vector<uint8_t> d_bytes = read_vec((size_t)n_rows * nb * 2);
    std::vector<uint8_t> aux_bytes = read_vec((size_t)n_rows * n_chunks * 4);
    std::vector<uint8_t> qs_bytes = read_vec((size_t)n_rows * n_chunks * 8);
    std::vector<uint8_t> x_ds = read_vec((size_t)n_chunks * 2);
    std::vector<uint8_t> x_qs = read_vec((size_t)n_chunks * 32);
    std::vector<uint8_t> cb_bytes = read_vec(256 * 4);
    std::vector<float> refs(ref_rows);
    in.read((char*)refs.data(), (std::streamsize)ref_rows * 4);

    const size_t weight_bytes = d_bytes.size() + aux_bytes.size() + qs_bytes.size();
    printf("data: rows=%u n_embd=%u nb=%u chunks=%u  weight=%.3f MB/row-set  repeat=%d\n",
           n_rows, n_embd, nb, n_chunks, weight_bytes / 1e6, repeat);

    // ---- SPIR-V ----
    auto load_spv = [](const char* path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
        size_t n = (size_t)f.tellg();
        f.seekg(0);
        std::vector<uint32_t> v(n / 4);
        f.read((char*)v.data(), (std::streamsize)n);
        return v;
    };
    auto spv_path = [&](const char* dot, const char* shape) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s/iq3_w%u_r%u_%s_%s.spv", STRATA_SPV_DIR, wg, rows_per_lane, dot, shape);
        return std::string(buf);
    };

    // ---- instance: 1.3 so shader integer dot product is core (both cards report >= 1.3) ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "strata-vk-iqbench";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nphys = 0;
    vkEnumeratePhysicalDevices(inst, &nphys, nullptr);
    printf("instance up, %u physical devices\n", nphys);
    std::vector<VkPhysicalDevice> phys(nphys);
    vkEnumeratePhysicalDevices(inst, &nphys, phys.data());

    int worst_bad = 0;
    for (uint32_t di = 0; di < nphys; ++di) {
        if (only_dev >= 0 && (int)di != only_dev) continue;

        Dev D;
        D.pd = phys[di];
        vkGetPhysicalDeviceProperties(D.pd, &D.props);
        D.name = D.props.deviceName;
        vkGetPhysicalDeviceMemoryProperties(D.pd, &D.mp);

        // the 4x8 packed formats are a *property*; only shaderIntegerDotProduct is a feature
        VkPhysicalDeviceShaderIntegerDotProductProperties dotp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        VkPhysicalDeviceProperties2 dp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        dp2.pNext = &dotp;
        vkGetPhysicalDeviceProperties2(D.pd, &dp2);
        VkPhysicalDeviceShaderIntegerDotProductFeatures dotf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &dotf;
        vkGetPhysicalDeviceFeatures2(D.pd, &f2);
        D.dot_signed = dotf.shaderIntegerDotProduct == VK_TRUE;
        D.dot_accel = dotp.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;

        uint32_t nqf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(D.pd, &nqf, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(nqf);
        vkGetPhysicalDeviceQueueFamilyProperties(D.pd, &nqf, qfs.data());
        D.queue_family = UINT32_MAX;
        for (uint32_t q = 0; q < nqf; ++q)
            if ((qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[q].queueCount > 0) { D.queue_family = q; break; }
        if (D.queue_family == UINT32_MAX) { printf("\n[%u] %s: no compute queue, skipped\n", di, D.name.c_str()); continue; }

        const bool use_dp4a = !force_scalar && D.dot_signed && STRATA_GLSLC_HAS_DOT;
        const std::vector<uint32_t> spv = load_spv(spv_path(use_dp4a ? "dp4a" : "scalar",
                                                            fixed ? "n80" : "dyn").c_str());

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = D.queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot_enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
        dot_enable.shaderIntegerDotProduct = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (use_dp4a) dci.pNext = &dot_enable;
        VKCHECK(vkCreateDevice(D.pd, &dci, nullptr, &D.dev));
        vkGetDeviceQueue(D.dev, D.queue_family, 0, &D.queue);

        // ---- buffers: one device-local arena, suballocated.  The three weight arrays each hold
        // `repeat` copies of the row set back to back, so row r of any copy addresses the same way.
        const size_t mul = (size_t)repeat;
        const size_t off_d = 0;
        const size_t off_aux = off_d + d_bytes.size() * mul;
        const size_t off_qs = off_aux + aux_bytes.size() * mul;
        const size_t off_xds = off_qs + qs_bytes.size() * mul;
        const size_t off_xqs = off_xds + x_ds.size();
        const size_t off_cb = off_xqs + x_qs.size();
        const size_t off_out = off_cb + 256 * 4;
        const size_t arena = off_out + (size_t)n_rows * mul * 4;

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = arena;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer arena_buf = VK_NULL_HANDLE;
        VKCHECK(vkCreateBuffer(D.dev, &bci, nullptr, &arena_buf));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(D.dev, arena_buf, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_mem_type(D.pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory arena_mem = VK_NULL_HANDLE;
        VKCHECK(vkAllocateMemory(D.dev, &mai, nullptr, &arena_mem));
        VKCHECK(vkBindBufferMemory(D.dev, arena_buf, arena_mem, 0));

        // staging
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = arena;
        sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer stage = VK_NULL_HANDLE;
        VKCHECK(vkCreateBuffer(D.dev, &sci, nullptr, &stage));
        VkMemoryRequirements sreq{};
        vkGetBufferMemoryRequirements(D.dev, stage, &sreq);
        VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        smai.allocationSize = sreq.size;
        smai.memoryTypeIndex = find_mem_type(D.pd, sreq.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory stage_mem = VK_NULL_HANDLE;
        VKCHECK(vkAllocateMemory(D.dev, &smai, nullptr, &stage_mem));
        VKCHECK(vkBindBufferMemory(D.dev, stage, stage_mem, 0));

        // fill the staging buffer: the row set repeated `repeat` times, then x, cb, out
        {
            void* p = nullptr;
            VKCHECK(vkMapMemory(D.dev, stage_mem, 0, arena, 0, &p));
            uint8_t* base = (uint8_t*)p;
            for (int r = 0; r < repeat; ++r) {
                memcpy(base + off_d + (size_t)r * d_bytes.size(), d_bytes.data(), d_bytes.size());
                memcpy(base + off_aux + (size_t)r * aux_bytes.size(), aux_bytes.data(), aux_bytes.size());
                memcpy(base + off_qs + (size_t)r * qs_bytes.size(), qs_bytes.data(), qs_bytes.size());
            }
            memcpy(base + off_xds, x_ds.data(), x_ds.size());
            memcpy(base + off_xqs, x_qs.data(), x_qs.size());
            memcpy(base + off_cb, cb_bytes.data(), cb_bytes.size());
            memset(base + off_out, 0, (size_t)n_rows * repeat * 4);
            vkUnmapMemory(D.dev, stage_mem);
        }

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = D.queue_family;
        VkCommandPool pool = VK_NULL_HANDLE;
        VKCHECK(vkCreateCommandPool(D.dev, &cpci, nullptr, &pool));

        const uint32_t n_rows_total = n_rows * (uint32_t)repeat;
        if (n_rows_total > D.props.limits.maxComputeWorkGroupCount[0]) {
            printf("\n[%u] %s: %u workgroups exceeds maxComputeWorkGroupCount[0]=%u, skipping\n",
                   di, D.name.c_str(), n_rows_total, D.props.limits.maxComputeWorkGroupCount[0]);
            continue;
        }

        // ---- descriptors ----
        VkDescriptorSetLayoutBinding binds[7]{};
        for (int b = 0; b < 7; ++b) {
            binds[b].binding = (uint32_t)b;
            binds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[b].descriptorCount = 1;
            binds[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = 7;
        dslci.pBindings = binds;
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorSetLayout(D.dev, &dslci, nullptr, &dsl));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = 12;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VKCHECK(vkCreatePipelineLayout(D.dev, &plci, nullptr, &layout));

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spv.size() * 4;
        smci.pCode = spv.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        VKCHECK(vkCreateShaderModule(D.dev, &smci, nullptr, &mod));

        VkComputePipelineCreateInfo cpci2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci2.stage.module = mod;
        cpci2.stage.pName = "main";
        cpci2.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        VKCHECK(vkCreateComputePipelines(D.dev, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipe));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 2;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        VKCHECK(vkCreateDescriptorPool(D.dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsa.descriptorPool = dpool;
        dsa.descriptorSetCount = 1;
        dsa.pSetLayouts = &dsl;
        VkDescriptorSet dset = VK_NULL_HANDLE;
        VKCHECK(vkAllocateDescriptorSets(D.dev, &dsa, &dset));

        VkDescriptorBufferInfo bi[7];
        bi[0] = {arena_buf, off_d, VK_WHOLE_SIZE};
        bi[1] = {arena_buf, off_aux, VK_WHOLE_SIZE};
        bi[2] = {arena_buf, off_qs, VK_WHOLE_SIZE};
        bi[3] = {arena_buf, off_cb, VK_WHOLE_SIZE};
        bi[4] = {arena_buf, off_xds, VK_WHOLE_SIZE};
        bi[5] = {arena_buf, off_xqs, VK_WHOLE_SIZE};
        bi[6] = {arena_buf, off_out, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[7]{};
        for (int b = 0; b < 7; ++b) {
            wr[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[b].dstSet = dset;
            wr[b].dstBinding = (uint32_t)b;
            wr[b].descriptorCount = 1;
            wr[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[b].pBufferInfo = &bi[b];
        }
        vkUpdateDescriptorSets(D.dev, 7, wr, 0, nullptr);

        // ---- upload ----
        {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VKCHECK(vkAllocateCommandBuffers(D.dev, &cai, &cb));
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
            VkBufferCopy cp{0, 0, arena};
            vkCmdCopyBuffer(cb, stage, arena_buf, 1, &cp);
            VKCHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VKCHECK(vkQueueSubmit(D.queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(D.queue));
            vkFreeCommandBuffers(D.dev, pool, 1, &cb);
        }
        printf("\n[%u] %s\n", di, D.name.c_str());
        printf("    api=%u.%u  vram=%.1f GiB  dot4x8 signed=%s accelerated=%s\n",
               VK_VERSION_MAJOR(D.props.apiVersion), VK_VERSION_MINOR(D.props.apiVersion),
               (double)D.mp.memoryHeaps[0].size / 1073741824.0,
               D.dot_signed ? "yes" : "no", D.dot_accel ? "yes" : "no");
        printf("    kernel: WG=%u  R=%u  dot=%s  chunk-bound=%s\n", wg, rows_per_lane,
               use_dp4a ? "dotPacked4x8EXT" : "scalar expansion", fixed ? "compile-time (80)" : "push constant");
        if (use_dp4a && !D.dot_accel)
            printf("    note: the driver does not call the 4x8 packed dot accelerated here\n");

        // ---- one dispatch, check against the reference ----
        auto record = [&](VkCommandBuffer cb) {
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
            uint32_t pc[3] = {n_chunks, nb, n_rows_total};
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
            vkCmdDispatch(cb, n_rows_total / rows_per_lane, 1, 1);
            VKCHECK(vkEndCommandBuffer(cb));
        };

        {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VKCHECK(vkAllocateCommandBuffers(D.dev, &cai, &cb));
            record(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VKCHECK(vkQueueSubmit(D.queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(D.queue));
            vkFreeCommandBuffers(D.dev, pool, 1, &cb);
        }

        // readback: map the arena (it is device-local, so map a host copy instead)
        std::vector<float> got(n_rows);
        {
            VkBufferCreateInfo rci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            rci.size = (size_t)n_rows * 4;
            rci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VkBuffer rbuf = VK_NULL_HANDLE;
            VKCHECK(vkCreateBuffer(D.dev, &rci, nullptr, &rbuf));
            VkMemoryRequirements rreq{};
            vkGetBufferMemoryRequirements(D.dev, rbuf, &rreq);
            VkMemoryAllocateInfo rmai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            rmai.allocationSize = rreq.size;
            rmai.memoryTypeIndex = find_mem_type(D.pd, rreq.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkDeviceMemory rmem = VK_NULL_HANDLE;
            VKCHECK(vkAllocateMemory(D.dev, &rmai, nullptr, &rmem));
            VKCHECK(vkBindBufferMemory(D.dev, rbuf, rmem, 0));

            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            VKCHECK(vkAllocateCommandBuffers(D.dev, &cai, &cb));
            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
            VkBufferCopy cp{off_out, 0, (size_t)n_rows * 4};
            vkCmdCopyBuffer(cb, arena_buf, rbuf, 1, &cp);
            VKCHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VKCHECK(vkQueueSubmit(D.queue, 1, &si, VK_NULL_HANDLE));
            VKCHECK(vkQueueWaitIdle(D.queue));
            void* p = nullptr;
            VKCHECK(vkMapMemory(D.dev, rmem, 0, (size_t)n_rows * 4, 0, &p));
            memcpy(got.data(), p, (size_t)n_rows * 4);
            vkUnmapMemory(D.dev, rmem);
            vkDestroyBuffer(D.dev, rbuf, nullptr);
            vkFreeMemory(D.dev, rmem, nullptr);
            vkFreeCommandBuffers(D.dev, pool, 1, &cb);
        }
        double max_err = 0.0;
        int bad = 0, bad_at = -1;
        for (uint32_t r = 0; r < ref_rows; ++r) {
            double e = std::abs((double)got[r] - (double)refs[r]);
            if (e > 1e-4) { ++bad; if (bad_at < 0) bad_at = (int)r; }
            max_err = std::max(max_err, e);
        }
        printf("    check: max_err=%.6f  bad=%d/%u", max_err, bad, ref_rows);
        if (bad_at >= 0) printf("  (first bad row %d: got %.6f want %.6f)", bad_at, got[bad_at], refs[bad_at]);
        printf("\n");
        if (bad) worst_bad += bad;

        // ---- timing ----
        std::vector<VkCommandBuffer> cbs((size_t)iters);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = (uint32_t)iters;
        VKCHECK(vkAllocateCommandBuffers(D.dev, &cai, cbs.data()));
        for (int i = 0; i < iters; ++i) record(cbs[(size_t)i]);
        std::vector<VkFence> fences((size_t)iters, VK_NULL_HANDLE);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        for (int i = 0; i < iters; ++i) VKCHECK(vkCreateFence(D.dev, &fci, nullptr, &fences[(size_t)i]));

        // warm up
        {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cbs[0];
            VKCHECK(vkQueueSubmit(D.queue, 1, &si, fences[0]));
            VKCHECK(vkWaitForFences(D.dev, 1, &fences[0], VK_TRUE, UINT64_MAX));
            VKCHECK(vkResetFences(D.dev, 1, &fences[0]));
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cbs[(size_t)i];
            VKCHECK(vkQueueSubmit(D.queue, 1, &si, fences[(size_t)i]));
        }
        VKCHECK(vkWaitForFences(D.dev, (uint32_t)iters, fences.data(), VK_TRUE, UINT64_MAX));
        auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        const double bytes = (double)weight_bytes * repeat;
        printf("    time: %.3f ms/dispatch  (%.1f M rows/s)  %.1f GB/s of weights\n",
               ms, n_rows_total / ms / 1000.0, bytes / (ms * 1e6));

        for (auto f : fences) vkDestroyFence(D.dev, f, nullptr);
        vkFreeCommandBuffers(D.dev, pool, (uint32_t)iters, cbs.data());
        vkDestroyDescriptorPool(D.dev, dpool, nullptr);
        vkDestroyPipeline(D.dev, pipe, nullptr);
        vkDestroyShaderModule(D.dev, mod, nullptr);
        vkDestroyPipelineLayout(D.dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(D.dev, dsl, nullptr);
        vkDestroyCommandPool(D.dev, pool, nullptr);
        vkDestroyBuffer(D.dev, stage, nullptr);
        vkFreeMemory(D.dev, stage_mem, nullptr);
        vkDestroyBuffer(D.dev, arena_buf, nullptr);
        vkFreeMemory(D.dev, arena_mem, nullptr);
        vkDestroyDevice(D.dev, nullptr);
    }

    vkDestroyInstance(inst, nullptr);
    printf("\n%s\n", worst_bad ? "FAILED verification" : "all devices verified");
    return worst_bad ? 1 : 0;
}
