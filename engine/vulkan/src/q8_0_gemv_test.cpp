// Q8_0 expert matvec integration test — no model file needed.
//
// Builds a tiny synthetic GGUF holding one Q8_0 expert tensor, loads it through
// ExpertIndex, runs the Vulkan Q8_0 matvec via ExpertStream and compares every
// output row against the ggml-compatible CPU reference. This exercises the whole
// format path: block_bytes_for_type(8) == 272, GGUF indexing, pipeline selection
// and the shader's 34-byte-per-32-element dequant.
#include "strata/vulkan/expert_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

using namespace strata::vulkan;

namespace {

template <typename T> void put(std::vector<uint8_t>& b, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
void align32(std::vector<uint8_t>& b) {
    while (b.size() % 32u) b.push_back(0);
}

// IEEE float -> half bits (round-to-nearest-even is not needed for these test
// scales; truncation is fine as long as the CPU reference reads the same bits).
uint16_t float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = int((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0) return uint16_t(sign);
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint32_t(exp) << 10) | mant);
}

// One 256-element super-block = 8 * (2-byte half scale + 32 int8 quants) = 272 B.
constexpr uint32_t kK = 256;
constexpr uint32_t kRows = 64;
constexpr uint32_t kStride = 272;

void write_synthetic_gguf(const std::string& path, std::vector<float>& ref_scales,
                          std::vector<std::vector<int8_t>>& ref_quants) {
    std::vector<uint8_t> b;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    b.insert(b.end(), magic, magic + 4);
    put<uint32_t>(b, 2);   // version
    put<uint64_t>(b, 1);   // n_tensors
    put<uint64_t>(b, 0);   // n_kv

    put_str(b, "blk.0.ffn_gate_exps.weight");
    put<uint32_t>(b, 3);              // n_dims
    put<uint64_t>(b, kK);             // ne[0] = input width
    put<uint64_t>(b, kRows);          // ne[1] = output rows
    put<uint64_t>(b, 1);              // ne[2] = one expert
    put<int32_t>(b, 8);               // GGML_TYPE_Q8_0
    put<uint64_t>(b, 0);              // offset

    align32(b);

    ref_scales.resize(kRows * 8);
    ref_quants.assign(kRows * 8, std::vector<int8_t>(32));
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t blk = 0; blk < 8u; ++blk) {
            const float scale = 0.02f + 0.01f * float((r + blk) % 7u);
            const uint16_t h = float_to_half(scale);
            put<uint16_t>(b, h);
            ref_scales[r * 8 + blk] = scale;
            for (uint32_t q = 0; q < 32u; ++q) {
                const int8_t v = int8_t((int(r) * 37 + int(blk) * 11 + int(q) * 3) % 251 - 125);
                put<int8_t>(b, v);
                ref_quants[r * 8 + blk][q] = v;
            }
        }
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write synthetic GGUF\n"); std::exit(2); }
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
}

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

bool make_device(VkPhysicalDevice phys, uint32_t qfam, VkDevice& dev, VkQueue& queue) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    return vkCreateDevice(phys, &di, nullptr, &dev) == VK_SUCCESS &&
           (vkGetDeviceQueue(dev, qfam, 0, &queue), true);
}

} // namespace

int main(int argc, char** argv) {
    const char* spv = (argc > 1) ? argv[1] : "expert_q8_0_gemv.comp.spv";
    const std::string path = "/tmp/opencode/strata-q8-0-test.gguf";

    std::vector<float> ref_scales;
    std::vector<std::vector<int8_t>> ref_quants;
    write_synthetic_gguf(path, ref_scales, ref_quants);

    std::string err;
    ExpertIndex index;
    if (!ExpertIndex::load(path, index, err)) {
        std::fprintf(stderr, "index: %s\n", err.c_str()); return 1;
    }
    if (index.tensors().size() != 1) {
        std::fprintf(stderr, "FAILED: expected 1 expert tensor, got %zu\n", index.tensors().size());
        return 1;
    }
    const ExpertTensor& t = index.tensors()[0];
    if (t.ggml_type != 8 || t.k != kK || t.n_rows != kRows || t.n_bytes != kRows * kStride) {
        std::fprintf(stderr,
                     "FAILED: bad index: type=%d k=%u rows=%u bytes=%llu (want 8/%u/%u/%u)\n",
                     t.ggml_type, t.k, t.n_rows, (unsigned long long)t.n_bytes,
                     kK, kRows, kRows * kStride);
        return 1;
    }
    std::printf("GGUF index OK: type=%d k=%u rows=%u bytes=%llu (272 B / 256 elems)\n",
                t.ggml_type, t.k, t.n_rows, (unsigned long long)t.n_bytes);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n"); return 1;
    }
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    if (!select_device(inst, phys, qfam)) {
        std::fprintf(stderr, "no Intel Vulkan device with a compute queue\n"); return 1;
    }
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    if (!make_device(phys, qfam, dev, queue)) {
        std::fprintf(stderr, "vkCreateDevice failed\n"); return 1;
    }

    // Only the Q8_0 shader is supplied; empty paths leave the other pipelines
    // unavailable but must not break creation.
    StreamConfig cfg;
    cfg.cache_slots = 4; cfg.io_threads = 2; cfg.prefetch_next_layer = false;
    auto stream = ExpertStream::create(index, phys, dev, qfam, queue, "", "", "", spv,
                                       "", "", "", cfg, err);
    if (!stream) { std::fprintf(stderr, "stream: %s\n", err.c_str()); return 1; }

    // Deterministic activation vector.
    std::vector<float> x(kK);
    for (uint32_t i = 0; i < kK; ++i)
        x[i] = std::sin(float(i) * 0.017f) * 0.5f + 0.1f;

    // The tensor bytes on disk are the reference input (same layout the shader
    // reads). Rebuild them from the generated scales/quants for an independent
    // CPU reference via q8_0_dot_reference().
    std::vector<uint8_t> raw(t.n_bytes);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t blk = 0; blk < 8u; ++blk) {
            uint8_t* b = raw.data() + r * kStride + blk * 34u;
            const uint16_t h = float_to_half(ref_scales[r * 8 + blk]);
            b[0] = uint8_t(h & 0xffu); b[1] = uint8_t(h >> 8u);
            std::memcpy(b + 2, ref_quants[r * 8 + blk].data(), 32);
        }
    }

    ExpertStream::Key key{0, ExpertKind::Gate, 0};
    uint32_t slot = 0;
    if (!stream->ensure(key, slot, err)) { std::fprintf(stderr, "ensure: %s\n", err.c_str()); return 1; }
    std::vector<float> got(kRows), want(kRows);
    if (!stream->matvec(key, slot, x.data(), kK, got.data(), kRows, err)) {
        std::fprintf(stderr, "matvec: %s\n", err.c_str()); return 1;
    }
    for (uint32_t r = 0; r < kRows; ++r)
        want[r] = q8_0_dot_reference(raw.data() + size_t(r) * kStride, x.data(), kK);

    uint32_t bad = 0;
    double max_abs = 0.0, rel_l1 = 0.0, sum_a = 0.0, sum_d = 0.0;
    for (uint32_t r = 0; r < kRows; ++r) {
        const double d = std::fabs(double(got[r]) - double(want[r]));
        const double a = std::fabs(double(want[r]));
        max_abs = std::max(max_abs, d);
        sum_d += d; sum_a += a;
        if (d / std::max<double>(1e-6, a) > 2e-2 && d > 2e-3) ++bad;
        if (r < 3) std::printf("row %2u got=%+.6f ref=%+.6f\n", r, got[r], want[r]);
    }
    rel_l1 = sum_d / std::max<double>(1e-30, sum_a);
    std::printf("Q8_0 matvec: bad=%u/%u max_abs=%.3g rel_l1=%.3g\n",
                bad, kRows, max_abs, rel_l1);

    stream.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    std::remove(path.c_str());

    if (bad) { std::fprintf(stderr, "FAILED: %u rows differ from CPU reference\n", bad); return 1; }
    std::printf("PASS\n");
    return 0;
}
