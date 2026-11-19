// Correctness test for the Vulkan greedy sampler kernels, using the fixtures
// from engine/sycl/src/kernels/verify_parity.cpp: ties at the maximum, NaN,
// -inf, rows with no value above -inf, and 1..8 rows on one set of buffers.
#include "strata/vulkan/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace strata::vulkan;

namespace {

bool select_device(VkInstance inst, VkPhysicalDevice& phys, uint32_t& qfam) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID != 0x8086) continue;
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { phys = d; qfam = i; return true; }
    }
    return false;
}

// Strict >, so ties keep the lowest index; NaN never wins; all -inf/NaN -> 0.
int32_t ref_argmax(const float* row, int n) {
    float best = -std::numeric_limits<float>::infinity();
    int32_t idx = 0;
    for (int i = 0; i < n; ++i)
        if (row[i] > best) { best = row[i]; idx = i; }
    return idx;
}

float ref_top_prob(const float* row, int n) {
    float best = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i)
        if (row[i] > best) best = row[i];
    if (std::isinf(best) && best < 0.0f) return 1.0f;
    double sum = 0;
    for (int i = 0; i < n; ++i)
        if (!std::isnan(row[i])) sum += std::exp(double(row[i] - best));
    return float(1.0 / sum);
}

} // namespace

int main(int argc, char** argv) {
    std::string argmax_spv = "engine/vulkan/shaders/argmax_rows.comp.spv";
    std::string prob_spv = "engine/vulkan/shaders/row_top_prob.comp.spv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--argmax-shader" && i + 1 < argc) argmax_spv = argv[++i];
        else if (a == "--top-prob-shader" && i + 1 < argc) prob_spv = argv[++i];
        else { std::fprintf(stderr, "usage: sampler_test [--argmax-shader F] [--top-prob-shader F]\n"); return 2; }
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    if (!select_device(inst, phys, qfam)) { std::fprintf(stderr, "no Intel Vulkan device\n"); return 1; }
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    if (vkCreateDevice(phys, &di, nullptr, &dev) != VK_SUCCESS) return 1;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    Sampler::Limits lim;
    lim.max_vocab = 262144; lim.max_rows = 8;
    std::string err;
    auto sampler = Sampler::create(phys, dev, qfam, queue, argmax_spv, prob_spv, lim, err);
    if (!sampler) { std::fprintf(stderr, "sampler: %s\n", err.c_str()); return 1; }

    std::mt19937 rng(20260925);
    std::normal_distribution<float> nd(0.0f, 4.0f);
    int bad = 0;
    for (int n : {248320, 40525, 4097, 33}) {
        for (int rows : {4, 8, 1, 3, 8, 2}) {
            std::vector<float> h(size_t(rows) * n);
            for (float& v : h) v = nd(rng);
            for (int r = 0; r < rows; ++r) {
                float* row = h.data() + size_t(r) * n;
                const int a = int(rng() % n), b = int(rng() % n);
                row[a] = row[b] = 40.0f;                 // a tie at the maximum
                row[rng() % n] = std::nanf("");          // never picked
                row[rng() % n] = -INFINITY;
                if (r == 2)                              // no value above -inf
                    for (int i = 0; i < n; ++i) row[i] = (i % 7) ? -INFINITY : std::nanf("");
            }
            std::vector<int32_t> ids(rows);
            std::vector<float> probs(rows);
            // Repeat launches on the same buffers: no per-call state may leak.
            for (int rep = 0; rep < 3; ++rep) {
                if (!sampler->argmax_rows(h.data(), rows, n, ids.data(), err) ||
                    !sampler->row_top_prob(h.data(), rows, n, probs.data(), err)) {
                    std::fprintf(stderr, "sampler: %s\n", err.c_str()); return 1;
                }
                for (int r = 0; r < rows; ++r) {
                    const int32_t want = ref_argmax(h.data() + size_t(r) * n, n);
                    const float wp = ref_top_prob(h.data() + size_t(r) * n, n);
                    const float errp = std::fabs(probs[r] - wp) /
                                       std::max<float>(1e-6f, std::fabs(wp));
                    if (ids[r] != want || errp > 1e-4f) {
                        if (bad < 6)
                            std::printf("  n=%d rows=%d r=%d id %d/%d prob %.9g/%.9g\n",
                                        n, rows, r, ids[r], want, probs[r], wp);
                        ++bad;
                    }
                }
            }
        }
    }
    std::printf("sampler: %d failures over %d shapes x 6 row counts x 3 launches\n", bad, 4);
    sampler.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    if (bad) return 1;
    std::printf("PASS\n");
    return 0;
}
