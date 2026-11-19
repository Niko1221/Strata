// Correctness test for the Vulkan BF16 GEMV, checked against the same spec as
// engine/sycl/src/kernels/bf16_gemv_parity.cpp: accuracy over the same bf16
// bits, and an observable bf16-vs-fp16 activation contract.
#include "strata/vulkan/bf16_gemv.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Round-to-nearest-even f32 -> bf16 / bf16 -> f32, matching the parity file.
uint16_t bf16_from_f32(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return uint16_t((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16);
}
float f32_from_bf16(uint16_t h) {
    uint32_t i = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}

// Round-to-nearest-even f32 -> fp16 and back, for the rival activation.
uint16_t f16_from_f32(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    const uint32_t sign = (i >> 16) & 0x8000u;
    const uint32_t exp = (i >> 23) & 0xFFu;
    uint32_t mant = i & 0x7FFFFFu;
    if (exp == 0xFFu) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    int e = int(exp) - 127 + 15;
    if (e >= 0x1F) return uint16_t(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);
        mant |= 0x800000u;
        const uint32_t shift = uint32_t(14 - e);
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return uint16_t(sign | h);
    }
    uint32_t h = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        ++h;
        if (h == 0x400u) { ++e; h = 0; }
    }
    return uint16_t(sign | (uint32_t(e) << 10) | h);
}
float f32_from_f16(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
            mant &= 0x3FFu;
            bits = sign | (uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::fabs(double(a[i]) - double(b[i]));
        m += std::fabs(double(a[i]));
    }
    return d / (m > 1e-30 ? m : 1e-30);
}

// Host reference in double over the SAME bf16 bits (so only summation order differs).
void reference(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
               uint32_t n_in, uint32_t n_out, std::vector<float>& y) {
    y.assign(n_out, 0.0f);
    for (uint32_t o = 0; o < n_out; ++o) {
        double acc = 0;
        for (uint32_t i = 0; i < n_in; ++i)
            acc += double(f32_from_bf16(x[i])) * double(f32_from_bf16(w[size_t(o) * n_in + i]));
        y[o] = float(acc);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string spv = "engine/vulkan/shaders/bf16_gemv.comp.spv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--shader" && i + 1 < argc) spv = argv[++i];
        else { std::fprintf(stderr, "usage: bf16_gemv_test [--shader FILE.spv]\n"); return 2; }
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

    Bf16Gemv::Limits lim;
    lim.max_in = 4096; lim.max_out = 4096;
    std::string err;
    auto gemv = Bf16Gemv::create(phys, dev, qfam, queue, spv, lim, err);
    if (!gemv) { std::fprintf(stderr, "bf16 gemv: %s\n", err.c_str()); return 1; }

    struct Shape { uint32_t n_in, n_out; const char* what; };
    const Shape shapes[] = {
        {2560, 48,   "ssm_alpha.weight  [2560, 48]"},
        {2560, 128,  "indexer.k_proj   [2560, 128]"},
        {2560, 512,  "indexer.q_proj   [2560, 512]"},
        {2560, 2560, "ple_value        [2560, 2560]"},
    };
    int bad = 0;
    for (const Shape& s : shapes) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> fx(s.n_in), fw(size_t(s.n_in) * s.n_out);
        for (auto& v : fx) v = g(rng);
        for (auto& v : fw) v = g(rng) * 0.05f;
        std::vector<uint16_t> x(s.n_in), w(fw.size());
        for (size_t i = 0; i < fx.size(); ++i) x[i] = bf16_from_f32(fx[i]);
        for (size_t i = 0; i < fw.size(); ++i) w[i] = bf16_from_f32(fw[i]);

        std::vector<float> want;
        reference(x, w, s.n_in, s.n_out, want);
        std::vector<float> got(s.n_out);
        if (!gemv->gemv(x.data(), w.data(), got.data(), s.n_in, s.n_out, err)) {
            std::fprintf(stderr, "gemv: %s\n", err.c_str()); return 1;
        }
        const double rel = rel_l1(want, got);
        std::printf("  %-34s rel_l1 %.2e\n", s.what, rel);
        if (!(rel <= 1e-5)) ++bad;

        // The activation contract: a bf16 weight sees a BF16 activation. The
        // rival (values routed through fp16 first) must be observable.
        std::vector<uint16_t> x_rival(s.n_in);
        for (size_t i = 0; i < fx.size(); ++i)
            x_rival[i] = bf16_from_f32(f32_from_f16(f16_from_f32(fx[i])));
        if (!gemv->gemv(x_rival.data(), w.data(), got.data(), s.n_in, s.n_out, err)) {
            std::fprintf(stderr, "gemv (rival): %s\n", err.c_str()); return 1;
        }
        const double rival = rel_l1(want, got);
        std::printf("      %-30s %s (%.4f%% apart)\n", "bf16 vs fp16 activation",
                    rival > 1e-4 ? "yes" : "*** NO ***", rival * 100);
        if (!(rival > 1e-4)) ++bad;
    }

    std::printf("\nbf16_gemv: %d failures\n", bad);
    gemv.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    if (bad) return 1;
    std::printf("PASS\n");
    return 0;
}
