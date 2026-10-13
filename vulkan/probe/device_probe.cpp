// Strata Vulkan backend - step 1 probe.
// Answers the questions the feasibility study starts with, on one machine:
//   * does one process see the NVIDIA card and the AMD card at the same time?
//   * what does each expose: VRAM, subgroup sizes, driver, queue families?
//   * can both run compute (the queue family check), including the MI50 with no display output?
// Prints plain lines; no dependencies beyond vulkan-1.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* type_name(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated-gpu";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual-gpu";
        default:                                     return "other";
    }
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "strata-vk-probe";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    const char* wanted[] = {"VK_KHR_get_physical_device_properties2"};
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = wanted;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ci, nullptr, &inst);
    if (r != VK_SUCCESS) { printf("vkCreateInstance failed: %d\n", r); return 1; }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    printf("devices visible to one process: %u\n", n);

    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);

        uint32_t qf = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qf, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qf);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qf, qs.data());
        bool has_compute = false;
        for (auto& q : qs)
            if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) has_compute = true;

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
        uint64_t vram = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                vram += mp.memoryHeaps[h].size;

        // Subgroup facts decide the kernel design (Strata's CUDA kernels assume 32-lane warps).
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(devs[i], &p2);

        printf("\n[%u] %s\n", i, p.deviceName);
        printf("    type=%s api=%u.%u driver=%llu\n", type_name(p.deviceType),
               VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
               (unsigned long long)p.driverVersion);
        printf("    vram=%.1f GiB compute_queue=%s\n", vram / 1073741824.0,
               has_compute ? "yes" : "NO");
        printf("    subgroup size_min=%u staged=%s ops=%s\n",
               sg.subgroupSize,
               (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "yes" : "no",
               (sg.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) ? "basic+" : "none");
        printf("    pipeline_cache_uuid=%02x%02x%02x%02x vendor=0x%04x device=0x%04x\n",
               p.pipelineCacheUUID[0], p.pipelineCacheUUID[1], p.pipelineCacheUUID[2], p.pipelineCacheUUID[3],
               p.vendorID, p.deviceID);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
