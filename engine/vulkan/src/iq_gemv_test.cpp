// IQ dequant matvec integration test — no model file needed.
//
// For each of IQ3_XXS (18), IQ2_XS (17) and IQ1_M (29) this builds a tiny
// synthetic GGUF with one expert tensor, loads it through ExpertIndex, runs the
// Vulkan matvec through ExpertStream and compares every output row with the
// ggml-compatible CPU reference. The byte patterns are deterministic; both the
// shader and the reference read the same bytes, so the test pins the GGUF
// indexing (98 / 74 / 56 bytes per 256 elements), pipeline selection and every
// lookup in the dequant path.
#include "strata/vulkan/expert_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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

// 2048 columns = 8 super-blocks, the widest k the expert matvec path accepts.
constexpr uint32_t kK = 2048;
constexpr uint32_t kRows = 64;

struct Case {
    const char* name;
    int32_t type;
    uint32_t block_bytes;
    std::string spv;
    float (*reference)(const uint8_t*, const float*, uint32_t);
};

// The scale at the start of IQ1_M is reconstructed from the top bits of the
// first four uint16 words; a random pattern would give a NaN/Inf half and make
// the comparison meaningless. Force those four words to a known finite scale.
void fix_iq1_m_scale(std::vector<uint8_t>& raw, uint32_t row, uint32_t stride) {
    uint8_t* b = raw.data() + size_t(row) * stride;
    uint16_t sc[4];
    for (uint32_t i = 0; i < 4u; ++i)
        sc[i] = uint16_t(b[48u + 2u * i]) | (uint16_t(b[48u + 2u * i + 1u]) << 8);
    // Reconstruct scale.u16 = 0x2800 (a finite half, about 0.03125):
    //   sc[0]>>12 = 2, (sc[2]>>4)&0xf00 = 0x800, other two terms 0.
    // Preserve the low bits: they are the per-block dl1/dl2 scale nibbles.
    sc[0] = uint16_t((sc[0] & 0x0fffu) | 0x2000u);
    sc[1] = uint16_t(sc[1] & 0x0fffu);
    sc[2] = uint16_t((sc[2] & 0xf00fu) | 0x0800u);
    sc[3] = uint16_t(sc[3] & 0x0fffu);
    for (uint32_t i = 0; i < 4u; ++i) {
        b[48u + 2u * i] = uint8_t(sc[i]);
        b[48u + 2u * i + 1u] = uint8_t(sc[i] >> 8);
    }
}

std::vector<uint8_t> write_synthetic_gguf(const std::string& path, const Case& c) {
    std::vector<uint8_t> b;
    const char magic[4] = {'G', 'G', 'U', 'F'};
    b.insert(b.end(), magic, magic + 4);
    put<uint32_t>(b, 2);   // version
    put<uint64_t>(b, 1);   // n_tensors
    put<uint64_t>(b, 0);   // n_kv

    put_str(b, "blk.0.ffn_gate_exps.weight");
    put<uint32_t>(b, 3);              // n_dims
    put<uint64_t>(b, kK);             // ne[0] = input width (kK/256 super-blocks)
    put<uint64_t>(b, kRows);          // ne[1] = output rows
    put<uint64_t>(b, 1);              // ne[2] = one expert
    put<int32_t>(b, c.type);
    put<uint64_t>(b, 0);              // offset

    align32(b);

    // Deterministic bytes, one 256-element super-block after another. The
    // first two bytes of each IQ3/IQ2 super-block are the half scale; set them
    // to a finite value. IQ1_M has no half at the start (fixed below).
    const uint32_t superblocks = kK / 256u;
    const size_t row_bytes = size_t(superblocks) * c.block_bytes;
    const uint16_t h = float_to_half(0.05f);
    std::mt19937 rng(1234u + uint32_t(c.type));
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t sb = 0; sb < superblocks; ++sb) {
            for (uint32_t i = 0; i < c.block_bytes; ++i) {
                uint8_t v = uint8_t(byte_dist(rng));
                if (i < 2u && c.type != 29) v = uint8_t(h >> ((i == 0u) ? 0u : 8u));
                b.push_back(v);
            }
        }
    }
    std::vector<uint8_t> raw(b.end() - size_t(kRows) * row_bytes, b.end());
    if (c.type == 29) {
        for (uint32_t r = 0; r < kRows; ++r)
            for (uint32_t sb = 0; sb < superblocks; ++sb)
                fix_iq1_m_scale(raw, r * superblocks + sb, c.block_bytes);
        std::copy(raw.begin(), raw.end(), b.end() - size_t(kRows) * row_bytes);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write synthetic GGUF\n"); std::exit(2); }
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
    return raw;
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
    const std::string iq3 = argc > 1 ? argv[1] : "expert_iq3_xxs_gemv.comp.spv";
    const std::string iq2 = argc > 2 ? argv[2] : "expert_iq2_xs_gemv.comp.spv";
    const std::string iq1 = argc > 3 ? argv[3] : "expert_iq1_m_gemv.comp.spv";
    const Case cases[] = {
        {"IQ3_XXS", 18, 98, iq3, iq3_xxs_dot_reference},
        {"IQ2_XS",  17, 74, iq2, iq2_xs_dot_reference},
        {"IQ1_M",   29, 56, iq1, iq1_m_dot_reference},
    };

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return 1;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    if (!select_device(inst, phys, qfam)) {
        std::fprintf(stderr, "no Intel Vulkan device with a compute queue\n"); return 1;
    }
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    if (!make_device(phys, qfam, dev, queue)) return 1;

    std::vector<float> x(kK);
    for (uint32_t i = 0; i < kK; ++i)
        x[i] = std::sin(float(i) * 0.017f) * 0.5f + 0.1f;

    int failed = 0;
    for (const Case& c : cases) {
        const std::string path = std::string("/tmp/opencode/strata-iq-test-") + c.name + ".gguf";
        const std::vector<uint8_t> raw = write_synthetic_gguf(path, c);
        std::string err;
        ExpertIndex index;
        if (!ExpertIndex::load(path, index, err) || index.tensors().size() != 1) {
            std::fprintf(stderr, "%s: index: %s\n", c.name, err.c_str()); ++failed; continue;
        }
        const ExpertTensor& t = index.tensors()[0];
        if (t.ggml_type != c.type || t.k != kK || t.n_rows != kRows ||
            t.n_bytes != kRows * (kK / 256u) * c.block_bytes) {
            std::fprintf(stderr, "%s: bad index: type=%d k=%u rows=%u bytes=%llu\n",
                         c.name, t.ggml_type, t.k, t.n_rows, (unsigned long long)t.n_bytes);
            ++failed; continue;
        }

        StreamConfig cfg;
        cfg.cache_slots = 4; cfg.io_threads = 2; cfg.prefetch_next_layer = false;
        const std::string iq3 = c.type == 18 ? c.spv : "";
        const std::string iq2 = c.type == 17 ? c.spv : "";
        const std::string iq1 = c.type == 29 ? c.spv : "";
        auto stream = ExpertStream::create(index, phys, dev, qfam, queue, "", "", "", "",
                                           iq3, iq2, iq1, cfg, err);
        if (!stream) { std::fprintf(stderr, "%s: stream: %s\n", c.name, err.c_str()); ++failed; continue; }
        ExpertStream::Key key{0, ExpertKind::Gate, 0};
        uint32_t slot = 0;
        if (!stream->ensure(key, slot, err)) {
            std::fprintf(stderr, "%s: ensure: %s\n", c.name, err.c_str()); ++failed; continue;
        }
        std::vector<float> got(kRows), want(kRows);
        if (!stream->matvec(key, slot, x.data(), kK, got.data(), kRows, err)) {
            std::fprintf(stderr, "%s: matvec: %s\n", c.name, err.c_str()); ++failed; continue;
        }
        for (uint32_t r = 0; r < kRows; ++r)
            want[r] = c.reference(raw.data() + size_t(r) * (kK / 256u) * c.block_bytes, x.data(), kK);

        uint32_t bad = 0;
        double max_abs = 0.0, sum_d = 0.0, sum_a = 0.0;
        for (uint32_t r = 0; r < kRows; ++r) {
            const double d = std::fabs(double(got[r]) - double(want[r]));
            const double a = std::fabs(double(want[r]));
            max_abs = std::max(max_abs, d);
            sum_d += d; sum_a += a;
            if (d / std::max<double>(1e-6, a) > 2e-2 && d > 2e-3) ++bad;
            if (r < 2) std::printf("%s row %u got=%+.6f ref=%+.6f\n", c.name, r, got[r], want[r]);
        }
        const double rel = sum_d / std::max<double>(1e-30, sum_a);
        std::printf("%s: bytes=%u bad=%u/%u max_abs=%.3g rel_l1=%.3g\n",
                    c.name, c.block_bytes, bad, kRows, max_abs, rel);
        if (bad) ++failed;
        stream.reset();
        std::remove(path.c_str());
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    if (failed) { std::fprintf(stderr, "FAILED: %d IQ case(s) differ from the CPU reference\n", failed); return 1; }
    std::printf("PASS\n");
    return 0;
}
