// Correctness test for the Vulkan MoE router kernels, checked against the same
// specification as engine/sycl/src/kernels/router_top10_parity.cpp.
#include "strata/vulkan/router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

// spec reference: softmax (double) -> stable descending argsort -> gather ->
// clamp at 2^-14 -> renormalise.
void cpu_topk(const float* logits, int ne, int k, int* ids, float* w) {
    const double clamp = 6.103515625e-05;
    std::vector<double> p(ne);
    double mx = logits[0];
    for (int e = 1; e < ne; ++e) mx = std::max(mx, double(logits[e]));
    double sum = 0;
    for (int e = 0; e < ne; ++e) { p[e] = std::exp(double(logits[e]) - mx); sum += p[e]; }
    for (int e = 0; e < ne; ++e) p[e] /= sum;
    std::vector<int> idx(ne);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[i]; w[i] = float(p[idx[i]]); s += p[idx[i]];
    }
    s = std::max(s, clamp);
    for (int i = 0; i < k; ++i) w[i] = float(double(w[i]) / s);
}

int check_case(Router& r, const char* name, const std::vector<float>& logits,
               int nt, int ne, int k, double tol) {
    std::vector<int32_t> ids(size_t(nt) * k);
    std::vector<float> w(size_t(nt) * k);
    std::string err;
    if (!r.topk(logits.data(), nt, ne, k, ids.data(), w.data(), err)) {
        std::printf("  %-24s ERROR %s\n", name, err.c_str()); return 1;
    }
    int bad_ids = 0, bad_w = 0;
    double worst = 0, worst_sum = 0;
    for (int t = 0; t < nt; ++t) {
        std::vector<int> rid(k);
        std::vector<float> rw(k);
        cpu_topk(&logits[size_t(t) * ne], ne, k, rid.data(), rw.data());
        double sum = 0;
        for (int i = 0; i < k; ++i) {
            if (ids[size_t(t) * k + i] != rid[i]) ++bad_ids;
            const double a = w[size_t(t) * k + i], b = rw[i];
            const double rel = std::fabs(a - b) / std::max(1e-30, std::fabs(b));
            worst = std::max(worst, rel);
            if (rel > tol) ++bad_w;
            sum += a;
        }
        worst_sum = std::max(worst_sum, std::fabs(sum - 1.0));
    }
    std::printf("  %-24s ids %s (%d)  worst rel %.2e (%d over tol)  |sum-1| %.1e\n",
                name, bad_ids ? "*** WRONG ***" : "exact", bad_ids, worst, bad_w, worst_sum);
    return bad_ids + bad_w;
}

} // namespace

int main(int argc, char** argv) {
    std::string gemv_spv = "engine/vulkan/shaders/router_gemv.comp.spv";
    std::string topk_spv = "engine/vulkan/shaders/router_topk.comp.spv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gemv-shader" && i + 1 < argc) gemv_spv = argv[++i];
        else if (a == "--topk-shader" && i + 1 < argc) topk_spv = argv[++i];
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

    Router::Limits lim;
    lim.max_experts = 256; lim.max_k = 2048; lim.max_top_k = 64; lim.max_tokens = 64;
    std::string err;
    auto router = Router::create(phys, dev, qfam, queue, gemv_spv, topk_spv, lim, err);
    if (!router) { std::fprintf(stderr, "router: %s\n", err.c_str()); return 1; }

    const int NE = 256, K = 8, NT = 64;
    std::mt19937 rng(2024);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int bad = 0;

    std::vector<float> a(size_t(NT) * NE);
    for (auto& v : a) v = gauss(rng);
    bad += check_case(*router, "random normal", a, NT, NE, K, 1e-4);

    std::vector<float> b(size_t(NT) * NE, 0.5f);
    bad += check_case(*router, "all equal (ties)", b, NT, NE, K, 1e-4);

    std::vector<float> c(size_t(NT) * NE);
    for (int t = 0; t < NT; ++t)
        for (int e = 0; e < NE; ++e) c[size_t(t) * NE + e] = (e < 12) ? 2.0f : gauss(rng);
    bad += check_case(*router, "12-way exact tie", c, NT, NE, K, 1e-4);

    std::vector<float> d(size_t(NT) * NE, 0.0f);
    for (int t = 0; t < NT; ++t) d[size_t(t) * NE + (t % NE)] = 20.0f;
    bad += check_case(*router, "dominant expert", d, NT, NE, K, 1e-4);

    std::vector<float> wide(size_t(NT) * NE);
    for (auto& v : wide) v = 6.0f * gauss(rng);
    bad += check_case(*router, "wide (peaked)", wide, NT, NE, K, 1e-4);

    // Router GEMV with the model's exact geometry: F32 [256 x 2048] * x[2048].
    const int WIDTH = 2048;
    std::vector<float> W(size_t(NE) * WIDTH), x(WIDTH), got(NE), ref(NE);
    for (auto& v : W) v = gauss(rng) * 0.05f;
    for (auto& v : x) v = gauss(rng);
    std::string gemv_err;
    if (!router->gemv(W.data(), x.data(), got.data(), NE, WIDTH, gemv_err)) {
        std::fprintf(stderr, "gemv: %s\n", gemv_err.c_str()); return 1;
    }
    double worst_g = 0;
    for (int e = 0; e < NE; ++e) {
        double s = 0;
        for (int h = 0; h < WIDTH; ++h) s += double(W[size_t(e) * WIDTH + h]) * x[h];
        ref[e] = float(s);
        worst_g = std::max(worst_g, std::fabs(got[e] - ref[e]) /
                                    std::max<double>(1e-6, std::fabs(ref[e])));
    }
    int gemv_bad = 0;
    for (int e = 0; e < NE; ++e)
        if (!(std::fabs(got[e] - ref[e]) <= 1e-4 + 1e-4 * std::fabs(ref[e]))) ++gemv_bad;
    std::printf("  %-24s worst rel %.2e  bad %d/%d\n", "gemv 256x2048", worst_g, gemv_bad, NE);
    bad += gemv_bad;

    // Route from those real-shaped logits: the two kernels chained.
    std::vector<int32_t> ids(size_t(NT) * K);
    std::vector<float> wt(size_t(NT) * K);
    std::vector<float> logits_nt(size_t(NT) * NE);
    for (int t = 0; t < NT; ++t)
        std::copy(got.begin(), got.end(), logits_nt.begin() + size_t(t) * NE);
    bad += check_case(*router, "chained gemv->topk", logits_nt, NT, NE, K, 1e-4);

    router.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    std::printf("\nrouter: %d failures\n", bad);
    if (bad) { std::fprintf(stderr, "FAILED\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
