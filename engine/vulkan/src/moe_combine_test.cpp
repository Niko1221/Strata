// Correctness test for the Vulkan MoE combine kernel, using the same fixture
// and structure traps as engine/sycl/src/kernels/shared_expert_parity.cpp:
//   routed terms are weighted, the shared expert is not, and a null shared
//   vector really omits it.
#include "strata/vulkan/moe_combine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(double(a[i]) - double(b[i]));
        m += std::fabs(double(a[i]));
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

// weight_routed / weight_shared are the two rival readings the parity test
// requires to be distinguishable before it judges the kernel.
std::vector<float> reference(const std::vector<float>& parts,
                             const std::vector<float>& w,
                             const std::vector<float>& shared,
                             uint32_t dim, uint32_t k,
                             bool weight_routed, bool weight_shared) {
    std::vector<float> y(dim, 0.0f);
    for (uint32_t j = 0; j < dim; ++j) {
        double a = 0;
        for (uint32_t e = 0; e < k; ++e)
            a += (weight_routed ? double(w[e]) : 1.0) *
                 double(parts[size_t(e) * dim + j]);
        a += weight_shared ? double(w[0]) * double(shared[j]) : double(shared[j]);
        y[j] = float(a);
    }
    return y;
}

} // namespace

int main(int argc, char** argv) {
    std::string spv = "engine/vulkan/shaders/moe_combine.comp.spv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--shader" && i + 1 < argc) spv = argv[++i];
        else { std::fprintf(stderr, "usage: moe_combine_test [--shader FILE.spv]\n"); return 2; }
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

    MoECombine::Limits lim;
    lim.max_experts = 64; lim.max_dim = 2048;
    std::string err;
    auto combine = MoECombine::create(phys, dev, qfam, queue, spv, lim, err);
    if (!combine) { std::fprintf(stderr, "combine: %s\n", err.c_str()); return 1; }

    int bad = 0;
    const uint32_t cases[][2] = {{256, 10}, {2048, 8}};  // parity shape + full embedding
    for (const auto& cs : cases) {
        const uint32_t dim = cs[0], k = cs[1];
        std::mt19937 rng(31337 + dim);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> parts(size_t(k) * dim), w(k), shared(dim), got(dim);
        for (auto& v : parts) v = gauss(rng);
        double wsum = 0;
        for (auto& v : w) { v = std::fabs(gauss(rng)) + 0.05f; wsum += v; }
        for (auto& v : w) v = float(v / wsum);            // already renormalised
        for (auto& v : shared) v = gauss(rng) * 3.0f;

        const std::vector<float> want = reference(parts, w, shared, dim, k, true, false);
        // Trap checks first: a test that only compared with the correct answer
        // would pass against either rival reading.
        const double sep_unweighted = rel_l1(want, reference(parts, w, shared, dim, k, false, false));
        const double sep_weighted_shared = rel_l1(want, reference(parts, w, shared, dim, k, true, true));
        std::printf("dim=%u k=%u  unweighted routed %.2f%% apart, weighted shared %.2f%% apart\n",
                    dim, k, sep_unweighted * 100, sep_weighted_shared * 100);
        if (!(sep_unweighted > 0.05) || !(sep_weighted_shared > 0.05)) ++bad;

        if (!combine->combine(parts.data(), w.data(), shared.data(), got.data(), dim, k, err)) {
            std::fprintf(stderr, "combine: %s\n", err.c_str()); return 1;
        }
        const double rel = rel_l1(want, got);
        std::printf("  moe_combine vs reference: rel %.3e\n", rel);
        if (!(rel <= 1e-6)) ++bad;

        // A null shared pointer must mean "no shared expert".
        if (!combine->combine(parts.data(), w.data(), nullptr, got.data(), dim, k, err)) {
            std::fprintf(stderr, "combine (no shared): %s\n", err.c_str()); return 1;
        }
        const double no_shared = rel_l1(want, got);
        std::printf("  null shared omits it: %.2f%% apart\n", no_shared * 100);
        if (!(no_shared > 0.05)) ++bad;
    }

    std::printf("\nmoe_combine: %d failures\n", bad);
    combine.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    if (bad) return 1;
    std::printf("PASS\n");
    return 0;
}
