#include "strata/vulkan/expert_stream.hpp"
#include "strata/vulkan/iq_tables.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unistd.h>

#include <sys/stat.h>

#ifdef STRATA_HAVE_LIBURING
#include <liburing.h>
#endif

namespace strata::vulkan {
namespace {

// Top-k experts of one layer are computed in one submission. 32 leaves room for
// a full 8-expert gate+up batch plus headroom without a larger descriptor pool.
constexpr uint32_t kBatchedMatvecs = 32;
constexpr uint32_t kMaxMatvecFloats = 2048;

template <typename T> bool rd(FILE* f, T& v) { return fread(&v, sizeof(T), 1, f) == 1; }

bool rd_str(FILE* f, std::string& s) {
    uint64_t n = 0;
    if (!rd(f, n) || n > (1u << 20)) return false;
    s.resize(size_t(n));
    return n == 0 || fread(s.data(), 1, size_t(n), f) == n;
}

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// Bytes per 256 elements. Every ggml K-quant uses a 256-element super-block;
// Q8_0 uses a 32-element block, so a 256-element slice spans 8 of them.
// Derivations are from ggml-common.h (QK_K = 256, QK8_0 = 32):
//   Q8_0    8 * (2 + 32)                     = 272
//   Q4_K    sizeof(block_q4_k)               = 144
//   Q5_K    sizeof(block_q5_k)               = 176
//   Q6_K    sizeof(block_q6_k)               = 210
//   IQ2_XS  2 + 256/8*sizeof(uint16_t) + 256/32 = 74
//   IQ3_XXS 2 + 3*(256/8)                    = 98
//   IQ1_M   256/8 + 256/16 + 256/32          = 56
uint64_t block_bytes_for_type(int32_t type) {
    switch (type) {
        case 8:  return 272;  // Q8_0   (8 x 34-byte, 32-element blocks)
        case 12: return 144;  // Q4_K
        case 13: return 176;  // Q5_K
        case 14: return 210;  // Q6_K
        case 17: return 74;   // IQ2_XS
        case 18: return 98;   // IQ3_XXS
        case 29: return 56;   // IQ1_M
        default: return 0;
    }
}

bool skip_value(FILE* f, int32_t type, uint64_t& alignment, const std::string& key,
                bool capture_alignment) {
    switch (type) {
        case 0: case 1: case 7: { int8_t v; return rd(f, v); }
        case 2: case 3: { int16_t v; return rd(f, v); }
        case 4: case 5: case 6: { int32_t v; return rd(f, v); }
        case 10: case 11: { int64_t v; return rd(f, v); }
        case 12: { double v; return rd(f, v); }
        case 8: { std::string v; return rd_str(f, v); }
        case 9: {
            int32_t et = 0; uint64_t n = 0;
            if (!rd(f, et) || !rd(f, n) || n > (1u << 24)) return false;
            for (uint64_t i = 0; i < n; ++i)
                if (!skip_value(f, et, alignment, key, false)) return false;
            return true;
        }
        default: return false;
    }
}

bool read_gguf_header(FILE* f, uint64_t& n_tensors, uint64_t& data_base, std::string& err) {
    char magic[4] = {};
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
        err = "not a GGUF file"; return false;
    }
    uint32_t version = 0;
    if (!rd(f, version) || version < 2 || version > 3) {
        err = "unsupported GGUF version"; return false;
    }
    if (!rd(f, n_tensors)) { err = "truncated GGUF header"; return false; }
    uint64_t n_kv = 0;
    if (!rd(f, n_kv)) { err = "truncated GGUF header"; return false; }

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        int32_t type = 0;
        if (!rd_str(f, key) || !rd(f, type)) { err = "truncated GGUF metadata"; return false; }
        // general.alignment is uint32; read it explicitly before skipping.
        if (key == "general.alignment" && type == 4) {
            uint32_t a = 0;
            if (!rd(f, a)) { err = "bad general.alignment"; return false; }
            alignment = a ? a : 32;
            continue;
        }
        if (!skip_value(f, type, alignment, key, false)) {
            err = "unsupported GGUF metadata type"; return false;
        }
    }
    (void)alignment;  // data_base is computed after the tensor directory
    return true;
}

uint64_t hash_key(const ExpertStream::Key& k) {
    // Layer and kind are small; fold in the shared flag as a distinct high bit.
    uint64_t h = uint64_t(uint32_t(k.layer)) << 40;
    h |= uint64_t(uint32_t(k.kind)) << 32;
    h |= k.expert;
    if (k.shared) h ^= 0x8000000000000000ull;
    return h;
}

bool find_memory_type(VkPhysicalDevice phys, uint32_t type_filter, VkMemoryPropertyFlags props,
                      uint32_t& out) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    // On the B390 the cached host-visible type (0xf) is ~2.4x faster for the
    // CPU-side streams and mapped logits than the plain coherent type (0x7).
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
            if ((type_filter & (1u << i)) &&
                (fl & (props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
                    == (props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                out = i; return true;
            }
        }
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            out = i;
            return true;
        }
    }
    return false;
}

bool create_buffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, buf, &req);
    uint32_t type = 0;
    if (!find_memory_type(phys, req.memoryTypeBits, props, type)) {
        vkDestroyBuffer(dev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(dev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(dev, buf, mem, 0) != VK_SUCCESS) {
        vkFreeMemory(dev, mem, nullptr);
        vkDestroyBuffer(dev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

std::vector<char> read_file(const std::string& path, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open shader: " + path; return {}; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { err = "empty shader: " + path; std::fclose(f); return {}; }
    std::vector<char> data(n);
    if (fread(data.data(), 1, size_t(n), f) != size_t(n)) {
        err = "short read: " + path; std::fclose(f); return {};
    }
    std::fclose(f);
    return data;
}

// ggml-compatible Q4_K dequantization (ggml-quants.c get_scale_min_k4).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)   | ((q[j]      >> 6) << 4);
    }
}

inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // IEEE subnormal half: value = mantissa * 2^-24.
        float f = float(mant) * (1.0f / 16777216.0f);
        return sign ? -f : f;
    }
    if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else           bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}
} // namespace

// ---------------------------------------------------------------------------
// ExpertIndex
// ---------------------------------------------------------------------------

bool ExpertIndex::load(const std::string& gguf_path, ExpertIndex& out, std::string& err) {
    FILE* f = std::fopen(gguf_path.c_str(), "rb");
    if (!f) { err = "cannot open GGUF: " + gguf_path; return false; }
    uint64_t n_tensors = 0, data_base = 0;
    if (!read_gguf_header(f, n_tensors, data_base, err)) { std::fclose(f); return false; }

    ExpertIndex idx;
    idx.path_ = gguf_path;

    struct Found { int layer; ExpertKind kind; std::string name; uint64_t off; uint64_t bytes;
                   uint32_t ne0, ne1, ne2; int32_t type; };
    struct FoundRouter { int layer; uint64_t off; uint32_t ne0, ne1; int32_t type; };
    struct FoundShared { int layer; ExpertKind kind; uint64_t off; uint64_t bytes;
                         uint32_t ne0, ne1; int32_t type; };
    struct FoundSharedGate { int layer; uint64_t off; uint32_t ne0; int32_t type; };
    std::vector<Found> found;
    std::vector<FoundRouter> found_routers;
    std::vector<FoundShared> found_shared;
    std::vector<FoundSharedGate> found_shared_gates;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        std::string name;
        uint32_t nd = 0;
        int32_t type = 0;
        uint64_t offset = 0;
        int64_t ne[4] = {1, 1, 1, 1};
        if (!rd_str(f, name) || !rd(f, nd) || nd > 4) {
            err = "truncated tensor directory"; std::fclose(f); return false;
        }
        for (uint32_t d = 0; d < nd; ++d)
            if (!rd(f, ne[d])) { err = "truncated tensor dims"; std::fclose(f); return false; }
        if (!rd(f, type) || !rd(f, offset)) {
            err = "truncated tensor info"; std::fclose(f); return false;
        }
        // Shared expert tensors have the same shapes and quantizations as one
        // routed expert, plus an F32 scalar gate. Index them separately so the
        // streaming cache can hold them with Key{..., shared=true}.
        if (name.find(".ffn_gate_inp_shexp.weight") != std::string::npos) {
            size_t b = name.find("blk.");
            if (b != std::string::npos && type == 0 && ne[0] > 0)
                found_shared_gates.push_back({std::atoi(name.c_str() + b + 4), offset,
                                              uint32_t(ne[0]), type});
            continue;
        }
        if (name.find(".ffn_gate_shexp.weight") != std::string::npos ||
            name.find(".ffn_up_shexp.weight") != std::string::npos ||
            name.find(".ffn_down_shexp.weight") != std::string::npos) {
            size_t b = name.find("blk.");
            uint64_t bpe = block_bytes_for_type(type);
            if (b != std::string::npos && bpe && ne[0] % 256 == 0 && ne[1] > 0) {
                ExpertKind kind = ExpertKind::Gate;
                if (name.find("ffn_up_shexp") != std::string::npos) kind = ExpertKind::Up;
                if (name.find("ffn_down_shexp") != std::string::npos) kind = ExpertKind::Down;
                found_shared.push_back({std::atoi(name.c_str() + b + 4), kind, offset,
                                        (uint64_t(ne[0]) / 256) * bpe * uint64_t(ne[1]),
                                        uint32_t(ne[0]), uint32_t(ne[1]), type});
            }
            continue;
        }
        // The F32 MoE router (ffn_gate_inp.weight) is the other half of
        // routing-guided prefetch.
        if (name.find(".ffn_gate_inp.weight") != std::string::npos) {
            size_t b = name.find("blk.");
            if (b != std::string::npos && type == 0 && ne[0] > 0 && ne[1] > 0)
                found_routers.push_back({std::atoi(name.c_str() + b + 4), offset,
                                         uint32_t(ne[0]), uint32_t(ne[1]), type});
            continue;
        }
        if (name.find(".ffn_gate_exps.weight") == std::string::npos &&
            name.find(".ffn_up_exps.weight") == std::string::npos &&
            name.find(".ffn_down_exps.weight") == std::string::npos) continue;

        uint64_t bpe = block_bytes_for_type(type);
        if (!bpe || ne[0] % 256 || ne[2] == 0) continue;

        ExpertKind kind = ExpertKind::Gate;
        if (name.find("ffn_up_exps") != std::string::npos) kind = ExpertKind::Up;
        if (name.find("ffn_down_exps") != std::string::npos) kind = ExpertKind::Down;

        // Parse "blk.<N>." from the tensor name.
        size_t b = name.find("blk.");
        if (b == std::string::npos) continue;
        int layer = std::atoi(name.c_str() + b + 4);

        uint64_t bytes = (uint64_t(ne[0]) / 256) * bpe * uint64_t(ne[1]); // one expert
        found.push_back({layer, kind, name,
                         offset, bytes,
                         uint32_t(ne[0]), uint32_t(ne[1]), uint32_t(ne[2]), type});
    }
    data_base = align_up(uint64_t(ftell(f)), 32);
    idx.data_base_ = data_base;
    for (const auto& r : found_routers)
        idx.routers_.push_back({r.layer, data_base + r.off, r.ne0, r.ne1, r.type});
    for (const auto& s : found_shared) {
        ExpertTensor t;
        t.layer = s.layer; t.kind = s.kind; t.file_offset = data_base + s.off;
        t.n_bytes = s.bytes; t.n_rows = s.ne1; t.k = s.ne0;
        t.n_experts = 1; t.ggml_type = s.type;
        idx.shared_.push_back(t);
    }
    for (const auto& g : found_shared_gates)
        idx.shared_gates_.push_back({g.layer, data_base + g.off, g.ne0, 1, g.type});
    std::fclose(f);

    if (found.empty()) {
        err = "no MoE expert tensors (ffn_*_exps.weight) found in the GGUF";
        return false;
    }
    std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) {
        if (a.layer != b.layer) return a.layer < b.layer;
        return uint32_t(a.kind) < uint32_t(b.kind);
    });
    for (const auto& x : found) {
        ExpertTensor t;
        t.layer = x.layer; t.kind = x.kind; t.name = x.name;
        t.file_offset = data_base + x.off; t.n_bytes = x.bytes;
        t.n_rows = x.ne1; t.k = x.ne0; t.n_experts = x.ne2;  // ggml: ne[0]=input, ne[1]=output
        t.ggml_type = x.type;
        idx.n_layers_ = std::max(idx.n_layers_, uint32_t(x.layer + 1));
        idx.n_experts_ = std::max(idx.n_experts_, x.ne2);
        idx.tensors_.push_back(t);
    }
    out = std::move(idx);
    return true;
}

const ExpertTensor* ExpertIndex::find(int layer, ExpertKind kind) const {
    for (const auto& t : tensors_)
        if (t.layer == layer && t.kind == kind) return &t;
    return nullptr;
}

const RouterTensor* ExpertIndex::find_router(int layer) const {
    for (const auto& r : routers_)
        if (r.layer == layer) return &r;
    return nullptr;
}

const ExpertTensor* ExpertIndex::find_shared(int layer, ExpertKind kind) const {
    for (const auto& t : shared_)
        if (t.layer == layer && t.kind == kind) return &t;
    return nullptr;
}

const RouterTensor* ExpertIndex::find_shared_gate(int layer) const {
    for (const auto& g : shared_gates_)
        if (g.layer == layer) return &g;
    return nullptr;
}

// ---------------------------------------------------------------------------
// ExpertStream
// ---------------------------------------------------------------------------

ExpertStream::~ExpertStream() {
    stopping_ = true;
    job_cv_.notify_all();
    for (auto& t : io_threads_) if (t.joinable()) t.join();
    if (device_ != VK_NULL_HANDLE) {
        if (act_mapped_) vkUnmapMemory(device_, act_mem_);
        if (up_mapped_) vkUnmapMemory(device_, up_mem_);
        if (out_mapped_) vkUnmapMemory(device_, out_mem_);
        if (slots_mapped_) vkUnmapMemory(device_, slots_mem_);
        if (pipeline_q4_k_) vkDestroyPipeline(device_, pipeline_q4_k_, nullptr);
        if (pipeline_q6_k_) vkDestroyPipeline(device_, pipeline_q6_k_, nullptr);
        if (pipeline_q8_0_) vkDestroyPipeline(device_, pipeline_q8_0_, nullptr);
        if (pipeline_iq3_xxs_) vkDestroyPipeline(device_, pipeline_iq3_xxs_, nullptr);
        if (pipeline_iq2_xs_) vkDestroyPipeline(device_, pipeline_iq2_xs_, nullptr);
        if (pipeline_iq1_m_) vkDestroyPipeline(device_, pipeline_iq1_m_, nullptr);
        if (pipeline_silu_) vkDestroyPipeline(device_, pipeline_silu_, nullptr);
        if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (desc_pool_) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
        if (desc_layout_) vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr);
        if (cmd_pool_) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        if (copy_fence_) vkDestroyFence(device_, copy_fence_, nullptr);
        if (compute_fence_) vkDestroyFence(device_, compute_fence_, nullptr);
        if (act_) { vkDestroyBuffer(device_, act_, nullptr); vkFreeMemory(device_, act_mem_, nullptr); }
        if (up_) { vkDestroyBuffer(device_, up_, nullptr); vkFreeMemory(device_, up_mem_, nullptr); }
        if (out_) { vkDestroyBuffer(device_, out_, nullptr); vkFreeMemory(device_, out_mem_, nullptr); }
        if (slots_) { vkDestroyBuffer(device_, slots_, nullptr); vkFreeMemory(device_, slots_mem_, nullptr); }
    }
    if (fd_ >= 0) ::close(fd_);
}

bool ExpertStream::create_pipeline(const std::string& spv_path, VkPipeline& out, std::string& err) {
    std::string e;
    auto code = read_file(spv_path, e);
    if (code.empty()) { err = e; return false; }

    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = code.size();
    sci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &sci, nullptr, &mod) != VK_SUCCESS) {
        err = "vkCreateShaderModule failed for " + spv_path; return false;
    }

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = pipeline_layout_;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &out) != VK_SUCCESS) {
        err = "vkCreateComputePipelines failed for " + spv_path;
        vkDestroyShaderModule(device_, mod, nullptr);
        return false;
    }
    vkDestroyShaderModule(device_, mod, nullptr);
    return true;
}

std::unique_ptr<ExpertStream> ExpertStream::create(const ExpertIndex& index,
                                                   VkPhysicalDevice physical,
                                                   VkDevice device,
                                                   uint32_t queue_family,
                                                   VkQueue queue,
                                                   const std::string& q4_spv_path,
                                                   const std::string& q6_spv_path,
                                                   const std::string& silu_spv_path,
                                                   const std::string& q8_spv_path,
                                                   const std::string& iq3_xxs_spv_path,
                                                   const std::string& iq2_xs_spv_path,
                                                   const std::string& iq1_m_spv_path,
                                                   const StreamConfig& cfg,
                                                   std::string& err) {
    auto s = std::unique_ptr<ExpertStream>(new ExpertStream());
    s->index_ = &index;
    s->physical_ = physical; s->device_ = device;
    s->queue_ = queue; s->queue_family_ = queue_family;
    s->cfg_ = cfg;

    s->fd_ = ::open(index.path().c_str(), O_RDONLY | O_CLOEXEC);
    if (s->fd_ < 0) { err = "cannot open GGUF for streaming"; return nullptr; }

    uint64_t max_bytes = 0;
    for (const auto& t : index.tensors()) max_bytes = std::max(max_bytes, t.n_bytes);
    for (const auto& t : index.shared_tensors()) max_bytes = std::max(max_bytes, t.n_bytes);
    if (!max_bytes) { err = "empty expert index"; return nullptr; }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    const VkDeviceSize align = props.limits.minStorageBufferOffsetAlignment
                                 ? props.limits.minStorageBufferOffsetAlignment : 256;
    s->slot_stride_ = align_up(max_bytes, uint64_t(align));

    const VkDeviceSize total = VkDeviceSize(s->slot_stride_) * cfg.cache_slots;
    // On an iGPU the same heap is DEVICE_LOCAL and HOST_VISIBLE: experts are
    // pread() directly into VRAM, which is the zero-copy path we want for B390.
    VkMemoryPropertyFlags mem_props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!create_buffer(physical, device, total,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       mem_props, s->slots_, s->slots_mem_)) {
        err = "cannot allocate expert slot pool"; return nullptr;
    }
    if (vkMapMemory(device, s->slots_mem_, 0, total, 0,
                    reinterpret_cast<void**>(&s->slots_mapped_)) != VK_SUCCESS) {
        err = "cannot map expert slot pool"; return nullptr;
    }

    s->slots_meta_.resize(cfg.cache_slots);
    s->clock_ = 0;

    const VkDeviceSize batch_bytes = VkDeviceSize(kBatchedMatvecs) * kMaxMatvecFloats * sizeof(float);
    if (!create_buffer(physical, device, batch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       s->act_, s->act_mem_) ||
        !create_buffer(physical, device, batch_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       s->out_, s->out_mem_) ||
        !create_buffer(physical, device, kMaxMatvecFloats * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       s->up_, s->up_mem_)) {
        err = "cannot allocate activation/output buffers"; return nullptr;
    }
    vkMapMemory(device, s->act_mem_, 0, batch_bytes, 0,
                reinterpret_cast<void**>(&s->act_mapped_));
    vkMapMemory(device, s->out_mem_, 0, batch_bytes, 0,
                reinterpret_cast<void**>(&s->out_mapped_));
    vkMapMemory(device, s->up_mem_, 0, kMaxMatvecFloats * sizeof(float), 0,
                reinterpret_cast<void**>(&s->up_mapped_));

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fi, nullptr, &s->copy_fence_) != VK_SUCCESS ||
        vkCreateFence(device, &fi, nullptr, &s->compute_fence_) != VK_SUCCESS) {
        err = "vkCreateFence failed"; return nullptr;
    }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = queue_family;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &cpi, nullptr, &s->cmd_pool_) != VK_SUCCESS) {
        err = "vkCreateCommandPool failed"; return nullptr;
    }

    // Both matvec shaders share one descriptor set layout / pipeline layout;
    // only the compute pipeline differs (Q4_K gate/up vs Q6_K down).
    VkDescriptorSetLayoutBinding b[3]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3; dli.pBindings = b;
    if (vkCreateDescriptorSetLayout(device, &dli, nullptr, &s->desc_layout_) != VK_SUCCESS) {
        err = "vkCreateDescriptorSetLayout failed"; return nullptr;
    }

    // Two sets from the same layout: one for matvec, one for silu_mul. The
    // dynamic binding of the second set is unused (offset 0).
    const uint32_t n_sets = 2 + kBatchedMatvecs;
    VkDescriptorPoolSize ps[2]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC; ps[0].descriptorCount = n_sets;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[1].descriptorCount = 2 * n_sets;
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = n_sets; pi.poolSizeCount = 2; pi.pPoolSizes = ps;
    if (vkCreateDescriptorPool(device, &pi, nullptr, &s->desc_pool_) != VK_SUCCESS) {
        err = "vkCreateDescriptorPool failed"; return nullptr;
    }
    std::vector<VkDescriptorSet> sets(n_sets);
    std::vector<VkDescriptorSetLayout> layouts(n_sets, s->desc_layout_);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = s->desc_pool_; dai.descriptorSetCount = n_sets; dai.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &dai, sets.data()) != VK_SUCCESS) {
        err = "vkAllocateDescriptorSets failed"; return nullptr;
    }
    s->desc_set_ = sets[0];
    s->desc_set_silu_ = sets[1];
    s->batch_sets_.assign(sets.begin() + 2, sets.end());

    // One reusable command buffer for a whole batch of expert matvecs.
    VkCommandBufferAllocateInfo bai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    bai.commandPool = s->cmd_pool_; bai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    bai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &bai, &s->batch_cmd_) != VK_SUCCESS) {
        err = "vkAllocateCommandBuffers failed (batch)"; return nullptr;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 8;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &s->desc_layout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &pli, nullptr, &s->pipeline_layout_) != VK_SUCCESS) {
        err = "vkCreatePipelineLayout failed"; return nullptr;
    }

    if (!q4_spv_path.empty() && !s->create_pipeline(q4_spv_path, s->pipeline_q4_k_, err))
        return nullptr;
    if (!q6_spv_path.empty() && !s->create_pipeline(q6_spv_path, s->pipeline_q6_k_, err))
        return nullptr;
    if (!q8_spv_path.empty() && !s->create_pipeline(q8_spv_path, s->pipeline_q8_0_, err))
        return nullptr;
    if (!iq3_xxs_spv_path.empty() && !s->create_pipeline(iq3_xxs_spv_path, s->pipeline_iq3_xxs_, err))
        return nullptr;
    if (!iq2_xs_spv_path.empty() && !s->create_pipeline(iq2_xs_spv_path, s->pipeline_iq2_xs_, err))
        return nullptr;
    if (!iq1_m_spv_path.empty() && !s->create_pipeline(iq1_m_spv_path, s->pipeline_iq1_m_, err))
        return nullptr;
    if (!silu_spv_path.empty() && !s->create_pipeline(silu_spv_path, s->pipeline_silu_, err))
        return nullptr;

    ExpertStream* raw = s.get();
#ifdef STRATA_HAVE_LIBURING
    if (getenv("STRATA_NO_URING") == nullptr) {
        s->io_threads_.emplace_back([raw] { raw->io_loop_uring(); });
        return s;
    }
#endif
    for (uint32_t i = 0; i < cfg.io_threads; ++i)
        s->io_threads_.emplace_back([raw] { raw->io_loop(); });
    return s;
}

void ExpertStream::finish_job(const ReadJob& job, bool ok,
                              std::chrono::steady_clock::time_point t0) {
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        const uint64_t h = hash_key(job.key);
        if (ok && slots_meta_[job.slot].key == job.key) {
            slots_meta_[job.slot].valid = true;
            slots_meta_[job.slot].loading = false;
            resident_[h] = job.slot;
            if (job.speculative) stats_.prefetches++;
            else stats_.misses++;
            stats_.bytes_read += job.bytes;
        } else if (slots_meta_[job.slot].key == job.key) {
            // failed read: release the slot so a waiting ensure() can retry
            slots_meta_[job.slot].valid = false;
            slots_meta_[job.slot].loading = false;
        }
        pending_.erase(h);
        stats_.io_ms += uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    }
    cache_cv_.notify_all();
}

void ExpertStream::io_loop() {
    while (true) {
        ReadJob job;
        {
            std::unique_lock<std::mutex> lk(job_mutex_);
            job_cv_.wait(lk, [&] { return stopping_ || !jobs_.empty(); });
            if (stopping_ && jobs_.empty()) return;
            job = jobs_.front(); jobs_.pop_front();
        }
        const auto t0 = std::chrono::steady_clock::now();
        uint8_t* dst = slots_mapped_ + uint64_t(job.slot) * slot_stride_;
        uint64_t done = 0;
        bool ok = true;
        while (done < job.bytes) {
            ssize_t n = ::pread(fd_, dst + done, size_t(job.bytes - done),
                                off_t(job.offset + done));
            if (n <= 0) { ok = false; break; }
            done += uint64_t(n);
        }
        finish_job(job, ok, t0);
    }
}

#ifdef STRATA_HAVE_LIBURING
// One thread drives all reads through io_uring. The old pool woke one thread
// per pread with a futex (~90k futex calls per short run on the B390); here a
// batch of reads is submitted at once and the thread sleeps in
// io_uring_submit_and_wait() until any one of them completes.
void ExpertStream::io_loop_uring() {
    io_uring ring;
    if (io_uring_queue_init(256, &ring, 0) < 0) { io_loop(); return; }
    ring_ = &ring;
    constexpr uint32_t kInflight = 256;
    struct SlotJob {
        ReadJob job;
        uint64_t done = 0;
        std::chrono::steady_clock::time_point t0;
    };
    std::vector<SlotJob> jobs(kInflight);
    std::vector<uint32_t> free_ids;
    for (uint32_t i = kInflight; i-- > 0;) free_ids.push_back(i);
    std::vector<bool> used(kInflight, false);

    auto reap = [&]() -> bool {
        io_uring_cqe* cqe = nullptr;
        if (io_uring_peek_cqe(&ring, &cqe) != 0 || !cqe) return false;
        const uint32_t id = uint32_t(cqe->user_data);
        const int res = cqe->res;
        io_uring_cqe_seen(&ring, cqe);
        SlotJob& sj = jobs[id];
        if (res <= 0) {
            finish_job(sj.job, false, sj.t0);
        } else {
            sj.done += uint64_t(res);
            if (sj.done < sj.job.bytes) {
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (sqe) {
                    uint8_t* dst = slots_mapped_ + uint64_t(sj.job.slot) * slot_stride_;
                    io_uring_prep_read(sqe, fd_, dst + sj.done,
                                       size_t(sj.job.bytes - sj.done),
                                       off_t(sj.job.offset + sj.done));
                    sqe->user_data = id;
                    io_uring_submit(&ring);
                    return true;
                }
                finish_job(sj.job, false, sj.t0);
            } else {
                finish_job(sj.job, true, sj.t0);
            }
        }
        used[id] = false;
        free_ids.push_back(id);
        return true;
    };

    while (true) {
        bool submitted = false;
        {
            std::lock_guard<std::mutex> lk(job_mutex_);
            while (!jobs_.empty() && !free_ids.empty()) {
                const uint32_t id = free_ids.back();
                SlotJob& sj = jobs[id];
                sj.job = jobs_.front(); jobs_.pop_front();
                sj.done = 0; sj.t0 = std::chrono::steady_clock::now();
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) { jobs_.push_front(sj.job); break; }
                uint8_t* dst = slots_mapped_ + uint64_t(sj.job.slot) * slot_stride_;
                io_uring_prep_read(sqe, fd_, dst, size_t(sj.job.bytes), off_t(sj.job.offset));
                sqe->user_data = id;
                used[id] = true;
                free_ids.pop_back();
                submitted = true;
            }
        }
        if (submitted) io_uring_submit(&ring);
        if (reap()) continue;
        bool any = false;
        for (bool b : used) if (b) { any = true; break; }
        if (any) { io_uring_submit_and_wait(&ring, 1); continue; }
        std::unique_lock<std::mutex> lk(job_mutex_);
        job_cv_.wait(lk, [&] { return stopping_ || !jobs_.empty(); });
        if (stopping_ && jobs_.empty()) break;
    }
    // Drain outstanding reads so no slot stays marked "loading" at teardown.
    bool any = true;
    while (any) {
        any = false;
        for (bool b : used) if (b) { any = true; break; }
        if (!any) break;
        io_uring_submit_and_wait(&ring, 1);
        while (reap()) {}
    }
    io_uring_queue_exit(&ring);
    ring_ = nullptr;
}
#endif

bool ExpertStream::ensure(const Key& key, uint32_t& slot, std::string& err) {
    const uint64_t h = hash_key(key);
    {
        std::unique_lock<std::mutex> lk(cache_mutex_);
        auto it = resident_.find(h);
        if (it != resident_.end()) {
            slot = it->second;
            slots_meta_[slot].used = ++clock_;
            stats_.hits++;
            return true;
        }
        if (pending_.count(h)) {
            // A prefetch or a concurrent ensure is already reading this expert.
            // Wait for that read rather than issuing a duplicate NVMe request.
            cache_cv_.wait(lk, [&] { return resident_.count(h) || !pending_.count(h); });
            if (resident_.count(h)) {
                slot = resident_[h];
                slots_meta_[slot].used = ++clock_;
                stats_.hits++;
                return true;
            }
            // the read failed: fall through and retry
        }
    }

    const ExpertTensor* t = key.shared ? index_->find_shared(key.layer, key.kind)
                                       : index_->find(key.layer, key.kind);
    if (!t || key.expert >= t->n_experts) {
        err = "expert out of range"; return false;
    }

    uint32_t victim = 0;
    {
        std::unique_lock<std::mutex> lk(cache_mutex_);
        // A prefetch may have claimed this key while we looked up the tensor.
        if (resident_.count(h)) {
            slot = resident_[h];
            slots_meta_[slot].used = ++clock_;
            stats_.hits++;
            return true;
        }
        if (pending_.count(h)) {
            cache_cv_.wait(lk, [&] { return resident_.count(h) || !pending_.count(h); });
            if (resident_.count(h)) {
                slot = resident_[h];
                slots_meta_[slot].used = ++clock_;
                stats_.hits++;
                return true;
            }
        }
        for (;;) {
            uint64_t oldest = std::numeric_limits<uint64_t>::max();
            bool found = false;
            victim = 0;
            for (uint32_t i = 0; i < slots_meta_.size(); ++i) {
                if (slots_meta_[i].loading) continue;
                found = true;
                if (!slots_meta_[i].valid) { victim = i; break; }
                if (slots_meta_[i].used < oldest) { oldest = slots_meta_[i].used; victim = i; }
            }
            if (found) break;
            cache_cv_.wait(lk);  // every slot is being filled: wait for one
        }
        if (slots_meta_[victim].valid) resident_.erase(hash_key(slots_meta_[victim].key));
        slots_meta_[victim] = Slot{key, false, true, ++clock_};
        pending_[h] = victim;
    }

    ReadJob job;
    job.key = key; job.slot = victim;
    job.bytes = t->n_bytes;
    job.offset = index_->expert_offset(*t, key.expert);
    {
        std::lock_guard<std::mutex> lk(job_mutex_);
        jobs_.push_back(job);
    }
    job_cv_.notify_one();

    {
        std::unique_lock<std::mutex> lk(cache_mutex_);
        cache_cv_.wait(lk, [&] { return resident_.count(h) || !pending_.count(h); });
        if (!resident_.count(h)) { err = "expert read failed"; return false; }
        slot = resident_[h];
    }
    return true;
}

void ExpertStream::prefetch(const Key& key) {
    if (!cfg_.prefetch_next_layer) return;
    prefetch_key(key);
}

void ExpertStream::prefetch_key(const Key& key) {
    const uint64_t h = hash_key(key);
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        if (resident_.count(h) || pending_.count(h)) return;
    }
    const ExpertTensor* t = key.shared ? index_->find_shared(key.layer, key.kind)
                                       : index_->find(key.layer, key.kind);
    if (!t || key.expert >= t->n_experts) return;

    uint32_t victim = 0;
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        bool found = false;
        for (uint32_t i = 0; i < slots_meta_.size(); ++i) {
            if (slots_meta_[i].loading) continue;
            found = true;
            if (!slots_meta_[i].valid) { victim = i; break; }
            if (slots_meta_[i].used < oldest) { oldest = slots_meta_[i].used; victim = i; }
        }
        if (!found) return;  // all slots in flight: skip this speculative read
        if (slots_meta_[victim].valid) resident_.erase(hash_key(slots_meta_[victim].key));
        slots_meta_[victim] = Slot{key, false, true, ++clock_};
        pending_[h] = victim;
    }
    ReadJob job;
    job.key = key; job.slot = victim; job.speculative = true;
    job.bytes = t->n_bytes;
    job.offset = index_->expert_offset(*t, key.expert);
    {
        std::lock_guard<std::mutex> lk(job_mutex_);
        jobs_.push_back(job);
    }
    job_cv_.notify_one();
}

void ExpertStream::prefetch_route(int layer, const int32_t* expert_ids, uint32_t n) {
    if (!cfg_.prefetch_next_layer) return;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t e = uint32_t(expert_ids[i]);
        if (expert_ids[i] < 0) continue;
        prefetch({layer, ExpertKind::Gate, e});
        prefetch({layer, ExpertKind::Up, e});
        prefetch({layer, ExpertKind::Down, e});
    }
}

bool ExpertStream::matvec(const Key& key, uint32_t slot,
                          const float* x, uint32_t k,
                          float* out, uint32_t n_rows, std::string& err) {
    const ExpertTensor* t = key.shared ? index_->find_shared(key.layer, key.kind)
                                       : index_->find(key.layer, key.kind);
    if (!t) { err = "unknown expert tensor"; return false; }
    if (k > 2048 || n_rows > 2048) { err = "matrix too large for prototype buffers"; return false; }

    // A GGUF can mix quantizations across layers: pick the shader by ggml type.
    VkPipeline pipeline = VK_NULL_HANDLE;
    switch (t->ggml_type) {
        case 8:  pipeline = pipeline_q8_0_; break;   // Q8_0
        case 12: pipeline = pipeline_q4_k_; break;   // Q4_K
        case 14: pipeline = pipeline_q6_k_; break;   // Q6_K
        case 17: pipeline = pipeline_iq2_xs_; break; // IQ2_XS
        case 18: pipeline = pipeline_iq3_xxs_; break;// IQ3_XXS
        case 29: pipeline = pipeline_iq1_m_; break;  // IQ1_M
        default: err = "unsupported expert quantization type"; return false;
    }
    if (pipeline == VK_NULL_HANDLE) {
        err = "no shader compiled for expert quantization type " + std::to_string(t->ggml_type);
        return false;
    }

    std::memcpy(act_mapped_, x, size_t(k) * sizeof(float));

    VkDescriptorBufferInfo bi[3]{};
    bi[0].buffer = slots_;        bi[0].offset = 0; bi[0].range = t->n_bytes;
    bi[1].buffer = act_;          bi[1].offset = 0; bi[1].range = VK_WHOLE_SIZE;
    bi[2].buffer = out_;          bi[2].offset = 0; bi[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = desc_set_;
        w[i].dstBinding = uint32_t(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS) {
        err = "vkAllocateCommandBuffers failed"; return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    const uint32_t dyn = uint32_t(uint64_t(slot) * slot_stride_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                            0, 1, &desc_set_, 1, &dyn);
    uint32_t pc[2] = {n_rows, k};
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(cmd, n_rows, 1, 1);
    vkEndCommandBuffer(cmd);

    vkResetFences(device_, 1, &compute_fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue_, 1, &si, compute_fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &compute_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        err = "Vulkan compute submit failed"; return false;
    }
    std::memcpy(out, out_mapped_, size_t(n_rows) * sizeof(float));
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    return true;
}

bool ExpertStream::batch_begin(std::string& err) {
    if (batch_cmd_ == VK_NULL_HANDLE) { err = "no batch command buffer"; return false; }
    if (vkResetCommandBuffer(batch_cmd_, 0) != VK_SUCCESS) { err = "reset batch cmd failed"; return false; }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(batch_cmd_, &begin) != VK_SUCCESS) { err = "begin batch cmd failed"; return false; }
    batch_copies_.clear();
    batch_cursor_ = 0;
    return true;
}

bool ExpertStream::record_matvec(const Key& key, uint32_t slot,
                                 const float* x_host, VkBuffer x_buf, uint32_t x_off_floats,
                                 uint32_t out_off_floats, uint32_t k, uint32_t n_rows,
                                 std::string& err) {
    if (batch_cmd_ == VK_NULL_HANDLE) { err = "no active batch"; return false; }
    if (batch_cursor_ >= kBatchedMatvecs) { err = "expert batch full"; return false; }
    if (k > kMaxMatvecFloats || n_rows > kMaxMatvecFloats) { err = "batch matvec shape too large"; return false; }
    const ExpertTensor* t = (key.shared ? index_->find_shared(key.layer, key.kind)
                                        : index_->find(key.layer, key.kind));
    if (!t) { err = "unknown expert tensor"; return false; }
    VkPipeline pipeline = VK_NULL_HANDLE;
    switch (t->ggml_type) {
        case 8:  pipeline = pipeline_q8_0_; break;
        case 12: pipeline = pipeline_q4_k_; break;
        case 14: pipeline = pipeline_q6_k_; break;
        case 17: pipeline = pipeline_iq2_xs_; break;
        case 18: pipeline = pipeline_iq3_xxs_; break;
        case 29: pipeline = pipeline_iq1_m_; break;
        default: err = "unsupported expert quantization type"; return false;
    }
    if (pipeline == VK_NULL_HANDLE) { err = "no shader for expert quantization"; return false; }

    if (x_host) std::memcpy(act_mapped_ + x_off_floats, x_host, size_t(k) * sizeof(float));
    const VkBuffer xb = x_buf ? x_buf : act_;
    VkDescriptorBufferInfo bi[3]{};
    bi[0].buffer = slots_; bi[0].offset = 0; bi[0].range = t->n_bytes;
    bi[1].buffer = xb;     bi[1].offset = uint64_t(x_off_floats) * sizeof(float);
    bi[1].range = uint64_t(k) * sizeof(float);
    bi[2].buffer = out_;   bi[2].offset = uint64_t(out_off_floats) * sizeof(float);
    bi[2].range = uint64_t(n_rows) * sizeof(float);
    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = batch_sets_[batch_cursor_]; w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);

    vkCmdBindPipeline(batch_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    const uint32_t dyn = uint32_t(uint64_t(slot) * slot_stride_);
    vkCmdBindDescriptorSets(batch_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                            0, 1, &batch_sets_[batch_cursor_], 1, &dyn);
    uint32_t pc[2] = {n_rows, k};
    vkCmdPushConstants(batch_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(batch_cmd_, n_rows, 1, 1);
    ++batch_cursor_;
    return true;
}

void ExpertStream::record_barrier() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(batch_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

bool ExpertStream::record_silu(uint32_t n, uint32_t gate_off, uint32_t up_off, uint32_t out_off,
                               std::string& err) {
    if (batch_cmd_ == VK_NULL_HANDLE || pipeline_silu_ == VK_NULL_HANDLE) {
        err = "no silu pipeline in batch"; return false;
    }
    if (batch_cursor_ >= kBatchedMatvecs) { err = "expert batch full"; return false; }
    if (n == 0 || n > kMaxMatvecFloats) { err = "batch silu shape too large"; return false; }
    const VkDeviceSize scale = sizeof(float);
    VkDescriptorBufferInfo bi[3]{};
    bi[0].buffer = out_; bi[0].offset = uint64_t(gate_off) * scale; bi[0].range = uint64_t(n) * scale;
    bi[1].buffer = out_; bi[1].offset = uint64_t(up_off)   * scale; bi[1].range = uint64_t(n) * scale;
    bi[2].buffer = out_; bi[2].offset = uint64_t(out_off)  * scale; bi[2].range = uint64_t(n) * scale;
    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = batch_sets_[batch_cursor_]; w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    vkCmdBindPipeline(batch_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_silu_);
    const uint32_t dyn = 0;  // binding 0 is the gate slice; no dynamic shift
    vkCmdBindDescriptorSets(batch_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                            0, 1, &batch_sets_[batch_cursor_], 1, &dyn);
    uint32_t pc[2] = {n, 0};
    vkCmdPushConstants(batch_cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(batch_cmd_, (n + 63) / 64, 1, 1);
    ++batch_cursor_;
    return true;
}

bool ExpertStream::batch_matvec(const Key& key, uint32_t slot,
                                const float* x, uint32_t k,
                                float* out_floats, uint32_t n_rows,
                                std::string& err) {
    if (batch_cmd_ == VK_NULL_HANDLE) { err = "no active batch"; return false; }
    const uint32_t ofs = batch_cursor_ * kMaxMatvecFloats;
    if (!record_matvec(key, slot, x, VK_NULL_HANDLE, ofs, ofs, k, n_rows, err)) return false;
    batch_copies_.push_back(PendingCopy{out_floats, ofs, n_rows});
    return true;
}

bool ExpertStream::batch_end(std::string& err) {
    if (batch_cmd_ == VK_NULL_HANDLE || batch_cursor_ == 0) { err = "empty batch"; return false; }
    vkEndCommandBuffer(batch_cmd_);
    vkResetFences(device_, 1, &compute_fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &batch_cmd_;
    if (vkQueueSubmit(queue_, 1, &si, compute_fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &compute_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        err = "batched expert matvec submit failed"; return false;
    }
    for (const auto& c : batch_copies_)
        std::memcpy(c.dst, out_mapped_ + c.src_offset, size_t(c.n) * sizeof(float));
    batch_copies_.clear();
    batch_cursor_ = 0;
    return true;
}

bool ExpertStream::silu_mul(const float* gate, const float* up, float* out,
                            uint32_t n, std::string& err) {
    if (pipeline_silu_ == VK_NULL_HANDLE) {
        err = "no SiLU*up shader compiled"; return false;
    }
    if (n == 0 || n > 2048) { err = "silu_mul: n out of range"; return false; }

    std::memcpy(act_mapped_, gate, size_t(n) * sizeof(float));
    std::memcpy(up_mapped_, up, size_t(n) * sizeof(float));

    VkDescriptorBufferInfo bi[3]{};
    bi[0].buffer = act_; bi[0].offset = 0; bi[0].range = VkDeviceSize(n) * sizeof(float);
    bi[1].buffer = up_;  bi[1].offset = 0; bi[1].range = VkDeviceSize(n) * sizeof(float);
    bi[2].buffer = out_; bi[2].offset = 0; bi[2].range = VkDeviceSize(n) * sizeof(float);
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = desc_set_silu_;
        w[i].dstBinding = uint32_t(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS) {
        err = "vkAllocateCommandBuffers failed (silu)"; return false;
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_silu_);
    const uint32_t dyn = 0;  // binding 0 is dynamic to match the shared layout
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                            0, 1, &desc_set_silu_, 1, &dyn);
    uint32_t pc[2] = {n, 0};
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(cmd, (n + 63) / 64, 1, 1);
    vkEndCommandBuffer(cmd);

    vkResetFences(device_, 1, &compute_fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue_, 1, &si, compute_fence_) != VK_SUCCESS ||
        vkWaitForFences(device_, 1, &compute_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        err = "Vulkan silu_mul submit failed"; return false;
    }
    std::memcpy(out, out_mapped_, size_t(n) * sizeof(float));
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    return true;
}

bool ExpertStream::shared_forward(int layer, const float* x, uint32_t k,
                                  float* out, uint32_t out_rows, std::string& err) {
    const ExpertTensor* g = index_->find_shared(layer, ExpertKind::Gate);
    const ExpertTensor* u = index_->find_shared(layer, ExpertKind::Up);
    const ExpertTensor* d = index_->find_shared(layer, ExpertKind::Down);
    const RouterTensor* gi = index_->find_shared_gate(layer);
    if (!g || !u || !d || !gi) { err = "no shared expert tensors for layer"; return false; }
    if (k != g->k || k != u->k || k != gi->k || d->k != g->n_rows ||
        u->n_rows != g->n_rows || out_rows != d->n_rows) {
        err = "shared expert shape mismatch"; return false;
    }
    if (k > 2048 || out_rows > 2048 || d->k > 2048) {
        err = "shared expert too large for prototype buffers"; return false;
    }

    const Key kg{layer, ExpertKind::Gate, 0, true};
    const Key ku{layer, ExpertKind::Up, 0, true};
    const Key kd{layer, ExpertKind::Down, 0, true};
    uint32_t sg = 0, su = 0, sd = 0;
    if (!ensure(kg, sg, err) || !ensure(ku, su, err) || !ensure(kd, sd, err)) return false;

    // Fuse gate, up, silu*up and down into ONE submission. The intermediates
    // stay in out_ (host-visible, same queue) and only the final down result is
    // copied back, so the three extra fence waits per layer disappear.
    constexpr uint32_t kGateOff = 0;
    constexpr uint32_t kUpOff = kMaxMatvecFloats;
    constexpr uint32_t kSiluOff = 2 * kMaxMatvecFloats;
    constexpr uint32_t kDownOff = 3 * kMaxMatvecFloats;
    if (!batch_begin(err)) return false;
    if (!record_matvec(kg, sg, x, VK_NULL_HANDLE, 0, kGateOff, k, g->n_rows, err) ||
        !record_matvec(ku, su, x, VK_NULL_HANDLE, 0, kUpOff, k, u->n_rows, err)) return false;
    record_barrier();
    if (!record_silu(d->k, kGateOff, kUpOff, kSiluOff, err)) return false;
    record_barrier();
    if (!record_matvec(kd, sd, nullptr, out_, kSiluOff, kDownOff,
                       d->k, out_rows, err)) return false;
    batch_copies_.push_back(PendingCopy{out, kDownOff, out_rows});
    if (!batch_end(err)) return false;

    // The scalar gate is a single F32 vector dot (ffn_gate_inp_shexp.weight).
    // It stays on the host for now: one 2048-element dot is not the bottleneck,
    // and the GGUF weight is F32, so no quantization path is involved.
    std::vector<float> gw(k);
    const uint64_t bytes = uint64_t(k) * sizeof(float);
    uint64_t done = 0;
    while (done < bytes) {
        ssize_t n = ::pread(fd_, gw.data() + done, bytes - done,
                            off_t(gi->file_offset + done));
        if (n <= 0) { err = "short shared gate read"; return false; }
        done += uint64_t(n);
    }
    double dot = 0;
    for (uint32_t i = 0; i < k; ++i) dot += double(gw[i]) * double(x[i]);
    const float sgate = float(1.0 / (1.0 + std::exp(-dot)));
    for (uint32_t i = 0; i < out_rows; ++i) out[i] *= sgate;
    return true;
}

StreamStats ExpertStream::stats() const {
    std::lock_guard<std::mutex> lk(cache_mutex_);
    return stats_;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

float q4_k_dot_reference(const uint8_t* row, const float* x, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t base = 0; base < k; base += 256) {
        const uint8_t* b = row + (base / 256) * 144;
        const uint8_t* q = b + 16;
        const float d = half_to_float(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
        const float mn = half_to_float(uint16_t(b[2]) | (uint16_t(b[3]) << 8));
        int is = 0;
        for (uint32_t j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is, b + 4, sc, m);     const float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, b + 4, sc, m); const float d2 = d * sc, m2 = mn * m;
            for (uint32_t i = 0; i < 32; ++i) sum += (d1 * (q[i] & 15) - m1) * x[base + j + i];
            for (uint32_t i = 0; i < 32; ++i) sum += (d2 * (q[i] >> 4) - m2) * x[base + j + 32 + i];
            q += 32; is += 2;
        }
    }
    return sum;
}

// ggml-compatible Q6_K dequant + dot. Mirrors dequantize_row_q6_K() from
// ggml/src/ggml-quants.c: block_q6_K = ql[128], qh[64], scales[16] (int8), d
// (half); the two 128-element halves of a 256-superblock are dequantized in
// four 32-element slices each.
float q6_k_dot_reference(const uint8_t* row, const float* x, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t base = 0; base < k; base += 256) {
        const uint8_t* b  = row + (base / 256u) * 210u;
        const uint8_t* ql = b;             // lower 4 bits, 2 per byte
        const uint8_t* qh = b + 128u;      // upper 2 bits, 4 per byte
        const int8_t*  sc = reinterpret_cast<const int8_t*>(b + 192u);
        const float d = half_to_float(uint16_t(b[208]) | (uint16_t(b[209]) << 8));

        for (uint32_t hb = 0; hb < 2u; ++hb) {
            const uint8_t* qlh = ql + hb * 64u;
            const uint8_t* qhh = qh + hb * 32u;
            const int8_t*  sch = sc + hb * 8u;
            for (uint32_t l = 0; l < 32u; ++l) {
                const int32_t q1 = int32_t((qlh[l]      & 15u) | ((uint32_t(qhh[l] & 3u))       << 4)) - 32;
                const int32_t q2 = int32_t((qlh[l + 32] & 15u) | ((uint32_t((qhh[l] >> 2) & 3u)) << 4)) - 32;
                const int32_t q3 = int32_t((qlh[l]      >> 4)  | ((uint32_t((qhh[l] >> 4) & 3u)) << 4)) - 32;
                const int32_t q4 = int32_t((qlh[l + 32] >> 4)  | ((uint32_t((qhh[l] >> 6) & 3u)) << 4)) - 32;
                const uint32_t is = l / 16u;
                const uint32_t o = base + hb * 128u + l;
                sum += d * float(sch[is + 0]) * float(q1) * x[o];
                sum += d * float(sch[is + 2]) * float(q2) * x[o + 32];
                sum += d * float(sch[is + 4]) * float(q3) * x[o + 64];
                sum += d * float(sch[is + 6]) * float(q4) * x[o + 96];
            }
        }
    }
    return sum;
}

// ggml-compatible Q8_0 dequant + dot. ggml block_q8_0 = ggml_half d (2 bytes)
// + int8 qs[32] (32 bytes) = 34 bytes per 32 elements; eight blocks make up a
// 256-element super-block, so the row stride is (k / 256) * 272 bytes.
float q8_0_dot_reference(const uint8_t* row, const float* x, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t base = 0; base < k; base += 256) {
        for (uint32_t blk = 0; blk < 8u; ++blk) {
            const uint8_t* b = row + (base / 256u) * 272u + blk * 34u;
            const float d = half_to_float(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
            const int8_t* q = reinterpret_cast<const int8_t*>(b + 2u);
            for (uint32_t i = 0; i < 32u; ++i)
                sum += d * float(q[i]) * x[base + blk * 32u + i];
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// IQ dequant + F32 dot references. These mirror ggml's dequantize_row_iq*_xxs
// byte-for-byte (including the generated lookup tables in iq_tables.hpp), so
// the GLSL shaders have an independent oracle that needs no ggml link.
// ---------------------------------------------------------------------------

// IQ3_XXS: 2 B scale d + 64 B qs + 32 B scales_and_signs = 98 B / 256 elems.
float iq3_xxs_dot_reference(const uint8_t* row, const float* x, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t base = 0; base < k; base += 256u) {
        const uint8_t* b = row + (base / 256u) * 98u;
        const float d = half_to_float(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            uint32_t aux = 0;
            std::memcpy(&aux, b + 66u + 4u * ib32, sizeof(aux));  // d(2) + qs[64]
            const float db = d * (0.5f + float(aux >> 28u)) * 0.5f;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint8_t signs = iq_tables::KSIGNS_IQ2XS[(aux >> (7u * l)) & 127u];
                const uint8_t* qs = b + 2u + 8u * ib32 + 2u * l;
                for (uint32_t j = 0; j < 4u; ++j) {
                    const float g1 = float(iq_tables::IQ3XXS_GRID[uint32_t(qs[0]) * 4u + j]);
                    const float g2 = float(iq_tables::IQ3XXS_GRID[uint32_t(qs[1]) * 4u + j]);
                    const uint32_t o = base + ib32 * 32u + l * 8u + j;
                    sum += db * g1 * ((signs & (1u << j)) ? -1.f : 1.f) * x[o];
                    sum += db * g2 * ((signs & (1u << (j + 4u))) ? -1.f : 1.f) * x[o + 4u];
                }
            }
        }
    }
    return sum;
}

// IQ2_XS: 2 B scale d + 64 B qs (uint16) + 8 B scales = 74 B / 256 elems.
float iq2_xs_dot_reference(const uint8_t* row, const float* x, uint32_t k) {
    float sum = 0.0f;
    for (uint32_t base = 0; base < k; base += 256u) {
        const uint8_t* b = row + (base / 256u) * 74u;
        const float d = half_to_float(uint16_t(b[0]) | (uint16_t(b[1]) << 8));
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            const uint8_t scale = b[66u + ib32];
            const float db[2] = {d * (0.5f + float(scale & 15u)) * 0.25f,
                                 d * (0.5f + float(scale >> 4u)) * 0.25f};
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint16_t q = uint16_t(b[2u + 2u * (4u * ib32 + l)])
                                 | (uint16_t(b[3u + 2u * (4u * ib32 + l)]) << 8);
                const uint8_t signs = iq_tables::KSIGNS_IQ2XS[q >> 9u];
                const uint8_t* grid = iq_tables::IQ2XS_GRID + uint32_t(q & 511u) * 8u;
                for (uint32_t j = 0; j < 8u; ++j)
                    sum += db[l / 2u] * float(grid[j]) * ((signs & (1u << j)) ? -1.f : 1.f)
                           * x[base + ib32 * 32u + l * 8u + j];
            }
        }
    }
    return sum;
}

// IQ1_M: 16 B scales (8 uint16) + 32 B qs + 8 B qh = 56 B / 256 elems.
float iq1_m_dot_reference(const uint8_t* row, const float* x, uint32_t k) {
    float sum = 0.0f;
    constexpr float kDelta = 0.125f;                 // IQ1S_DELTA
    for (uint32_t base = 0; base < k; base += 256u) {
        const uint8_t* b = row + (base / 256u) * 56u;
        // block_iq1_m layout (ggml-common.h): qs[32] at 0, qh[16] at 32,
        // scales[8] (four uint16) at 48.
        uint16_t sc[4];
        for (uint32_t i = 0; i < 4u; ++i)
            sc[i] = uint16_t(b[48u + 2u * i]) | (uint16_t(b[49u + 2u * i]) << 8);
        const uint16_t scale_bits = uint16_t((sc[0] >> 12)
            | ((sc[1] >> 8) & 0x00f0u) | ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u));
        const float d = half_to_float(scale_bits);
        const uint8_t* qs = b;
        const uint8_t* qh = b + 32u;
        for (uint32_t ib = 0; ib < 8u; ++ib) {
            const uint32_t sc_index = ib / 2u;
            const uint32_t shift = 6u * (ib % 2u);
            const float dl[2] = {d * (2.f * float((sc[sc_index] >> shift) & 7u) + 1.f),
                                 d * (2.f * float((sc[sc_index] >> (shift + 3u)) & 7u) + 1.f)};
            const uint32_t idx[4] = {
                uint32_t(qs[0]) | ((uint32_t(qh[0]) << 8) & 0x700u),
                uint32_t(qs[1]) | ((uint32_t(qh[0]) << 4) & 0x700u),
                uint32_t(qs[2]) | ((uint32_t(qh[1]) << 8) & 0x700u),
                uint32_t(qs[3]) | ((uint32_t(qh[1]) << 4) & 0x700u),
            };
            const float delta[4] = {(qh[0] & 0x08u) ? -kDelta : kDelta,
                                    (qh[0] & 0x80u) ? -kDelta : kDelta,
                                    (qh[1] & 0x08u) ? -kDelta : kDelta,
                                    (qh[1] & 0x80u) ? -kDelta : kDelta};
            for (uint32_t l = 0; l < 4u; ++l) {
                const int8_t* grid = reinterpret_cast<const int8_t*>(iq_tables::IQ1S_GRID) + idx[l] * 8u;
                for (uint32_t j = 0; j < 8u; ++j)
                    sum += (l < 2u ? dl[0] : dl[1]) * (float(grid[j]) + delta[l])
                           * x[base + ib * 32u + l * 8u + j];
            }
            qs += 4;
            qh += 2;
        }
    }
    return sum;
}

} // namespace strata::vulkan
