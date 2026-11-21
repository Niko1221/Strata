// model.cpp - native Vulkan forward for qwen35moe (Qwen3.6/3.8-35B-A3B).
// The forward mirrors engine/src/qwen35/q36ref.cu exactly (fp32 activations,
// GGUF weights read in their native quantization). MoE experts are streamed
// through ExpertStream; everything else is resident in one host-visible buffer.
#include "model.hpp"
#include "strata/vulkan/expert_stream.hpp"
#include "strata/vulkan/router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unistd.h>

namespace strata::vulkan {
namespace native {
namespace {

template <typename T> bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }
bool rd_str(FILE* f, std::string& s) {
    uint64_t n = 0;
    if (!rd(f, n) || n > (1u << 22)) return false;
    s.resize(size_t(n));
    return n == 0 || std::fread(s.data(), 1, size_t(n), f) == n;
}
uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// ggml block sizes (bytes per 256 elements), from ggml-common.h.
uint64_t block_bytes(int32_t t) {
    switch (t) {
        case 0:  return 4;    // F32: 1 element/block, 4 bytes
        case 8:  return 272;  // Q8_0
        case 12: return 144;  // Q4_K
        case 14: return 210;  // Q6_K
        default: return 0;
    }
}
uint32_t block_elems(int32_t t) { return t == 0 ? 1u : (t == 8 ? 32u : 256u); }

bool skip_val(FILE* f, int32_t t) {
    switch (t) {
        case 0: case 1: case 7: { int8_t v; return rd(f, v); }
        case 2: case 3: { int16_t v; return rd(f, v); }
        case 4: case 5: case 6: { int32_t v; return rd(f, v); }
        case 10: case 11: { int64_t v; return rd(f, v); }
        case 12: { double v; return rd(f, v); }
        case 8: { std::string v; return rd_str(f, v); }
        case 9: {
            int32_t et = 0; uint64_t n = 0;
            if (!rd(f, et) || !rd(f, n)) return false;
            for (uint64_t i = 0; i < n; ++i) if (!skip_val(f, et)) return false;
            return true;
        }
        default: return false;
    }
}

struct KV {
    int type = -1;
    int64_t i = 0;
    float f = 0;
    std::string s;
    uint64_t count = 0;
};

bool read_kv(FILE* f, int32_t t, KV& kv) {
    kv.type = t;
    switch (t) {
        case 4: { uint32_t v; if (!rd(f, v)) return false; kv.i = v; return true; }
        case 5: { uint32_t v; if (!rd(f, v)) return false; kv.i = v; return true; }
        case 6: { float v; if (!rd(f, v)) return false; kv.f = v; kv.i = int64_t(v); return true; }
        case 8: return rd_str(f, kv.s);
        case 9: {
            int32_t et = 0; uint64_t n = 0;
            if (!rd(f, et) || !rd(f, n)) return false;
            kv.count = n;
            for (uint64_t i = 0; i < n; ++i) if (!skip_val(f, et)) return false;
            return true;
        }
        default: return skip_val(f, t);
    }
}

std::vector<char> read_spv(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); return {}; }
    std::vector<char> d(static_cast<size_t>(n));
    bool ok = std::fread(d.data(), 1, size_t(n), f) == size_t(n);
    std::fclose(f);
    return ok ? d : std::vector<char>{};
}

bool find_mem(VkPhysicalDevice p, uint32_t bits, VkMemoryPropertyFlags props, uint32_t& out) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(p, &mp);
    // On the B390 the cached host-visible type (0xf) is ~2.4x faster for the
    // CPU-side streams and mapped logits than the plain coherent type (0x7).
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags fl = mp.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) &&
                (fl & (props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
                    == (props | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                out = i; return true;
            }
        }
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) { out = i; return true; }
    return false;
}

float softplus(float v) { return v > 20.f ? v : std::log1p(std::exp(v)); }
float sigmoid(float v) { return 1.f / (1.f + std::exp(-v)); }

} // namespace

// ---------------------------------------------------------------------------
// construction / teardown
// ---------------------------------------------------------------------------

bool NativeForward::make_buf(VkDeviceSize bytes, Buf& b, std::string& err) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = bytes; ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(ctx_.dev, &ci, nullptr, &b.b) != VK_SUCCESS) { err = "vkCreateBuffer failed"; return false; }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_.dev, b.b, &req);
    uint32_t type = 0;
    if (!find_mem(ctx_.phys, req.memoryTypeBits,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, type)) {
        err = "no host-visible memory"; return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = type;
    if (vkAllocateMemory(ctx_.dev, &ai, nullptr, &b.m) != VK_SUCCESS) { err = "vkAllocateMemory failed"; return false; }
    if (vkBindBufferMemory(ctx_.dev, b.b, b.m, 0) != VK_SUCCESS ||
        vkMapMemory(ctx_.dev, b.m, 0, bytes, 0, reinterpret_cast<void**>(&b.map)) != VK_SUCCESS) {
        err = "bind/map failed"; return false;
    }
    b.size = bytes;
    std::memset(b.map, 0, size_t(bytes));
    return true;
}

bool NativeForward::add_pipeline(const std::string& name, const std::string& spv, std::string& err) {
    auto code = read_spv(spv);
    if (code.empty()) { err = "cannot read shader: " + spv; return false; }
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = code.size(); sci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx_.dev, &sci, nullptr, &mod) != VK_SUCCESS) {
        err = "vkCreateShaderModule: " + spv; return false;
    }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod; cpi.stage.pName = "main";
    cpi.layout = pipe_layout_;
    VkPipeline pipe = VK_NULL_HANDLE;
    bool ok = vkCreateComputePipelines(ctx_.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) == VK_SUCCESS;
    vkDestroyShaderModule(ctx_.dev, mod, nullptr);
    if (!ok) { err = "vkCreateComputePipelines: " + spv; return false; }
    pipes_[name] = pipe;
    return true;
}

bool NativeForward::dispatch(const std::string& pipe, uint32_t gx, const uint32_t pc[6],
                             const VkBuffer bufs[6], const VkDeviceSize offs[6], std::string& err) {
    auto it = pipes_.find(pipe);
    if (it == pipes_.end()) { err = "unknown pipeline: " + pipe; return false; }
    if (batching_ && batch_cursor_ >= batch_sets_.size()) {
        err = "batch dispatch overflow"; return false;
    }
    // One descriptor set per kernel: descriptors are written when a command is
    // recorded but read when it executes, so deferred dispatches cannot share
    // one set. The sets are reused across batches, never allocated per launch.
    const VkDescriptorSet set = batching_ ? batch_sets_[batch_cursor_] : desc_set_;
    VkCommandBuffer cmd = batching_ ? batch_cmd_ : cmd_;
    VkDescriptorBufferInfo bi[6]{};
    VkWriteDescriptorSet w[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        bi[i].buffer = bufs[i] ? bufs[i] : dummy_.b;
        bi[i].offset = 0;
        bi[i].range = bufs[i] ? VK_WHOLE_SIZE : 4;
        // ANV appears to mishandle VK_WHOLE_SIZE with a nonzero dynamic offset
        // for large buffers; use an explicit range that stays inside the buffer.
        if (bufs[i] == weights_.b) bi[i].range = weights_.size - offs[i];
        else if (bufs[i] == x_.b) bi[i].range = x_.size - offs[i];
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(ctx_.dev, 6, w, 0, nullptr);
    if (!batching_) {
        if (vkResetCommandBuffer(cmd_, 0) != VK_SUCCESS) { err = "reset cmd failed"; return false; }
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &begin);
    }
    uint32_t dyn[6] = {};
    for (uint32_t i = 0; i < 6; ++i) dyn[i] = uint32_t(offs[i]);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, it->second);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, 1, &set, 6, dyn);
    vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, pc);
    vkCmdDispatch(cmd, gx, 1, 1);
    if (batching_) {
        // Consecutive compute dispatches execute in order, but without a barrier
        // a later kernel may read stale cache lines written by the previous one.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        ++batch_cursor_;
        return true;
    }
    vkEndCommandBuffer(cmd);
    vkResetFences(ctx_.dev, 1, &fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    bool ok = vkQueueSubmit(ctx_.queue, 1, &si, fence_) == VK_SUCCESS &&
              vkWaitForFences(ctx_.dev, 1, &fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (!ok) { err = "Vulkan dispatch failed"; return false; }
    return true;
}

void NativeForward::batch_abort() {
    if (!batching_) return;
    vkResetCommandBuffer(batch_cmd_, 0);
    batching_ = false;
    batch_cursor_ = 0;
}

bool NativeForward::batch_begin(std::string& err) {
    if (batching_) { err = "nested batch"; return false; }
    if (vkResetCommandBuffer(batch_cmd_, 0) != VK_SUCCESS) { err = "reset batch cmd failed"; return false; }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(batch_cmd_, &begin) != VK_SUCCESS) { err = "begin batch cmd failed"; return false; }
    batch_cursor_ = 0;
    batching_ = true;
    return true;
}

bool NativeForward::batch_flush(std::string& err) {
    if (!batching_) return true;
    if (batch_cursor_ == 0) { batching_ = false; return true; }
    vkEndCommandBuffer(batch_cmd_);
    vkResetFences(ctx_.dev, 1, &fence_);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &batch_cmd_;
    bool ok = vkQueueSubmit(ctx_.queue, 1, &si, fence_) == VK_SUCCESS &&
              vkWaitForFences(ctx_.dev, 1, &fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    batching_ = false;
    if (!ok) { err = "Vulkan batch submit failed"; return false; }
    return true;
}

NativeForward::~NativeForward() {
    if (ctx_.dev == VK_NULL_HANDLE) return;
    for (auto& p : pipes_) if (p.second) vkDestroyPipeline(ctx_.dev, p.second, nullptr);
    auto free_buf = [&](Buf& b) {
        if (b.map) vkUnmapMemory(ctx_.dev, b.m);
        if (b.b) vkDestroyBuffer(ctx_.dev, b.b, nullptr);
        if (b.m) vkFreeMemory(ctx_.dev, b.m, nullptr);
        b = Buf{};
    };
    free_buf(weights_); free_buf(dummy_);
    free_buf(x_); free_buf(h_); free_buf(y_); free_buf(qkv_); free_buf(z_);
    free_buf(alpha_); free_buf(beta_); free_buf(conv_out_); free_buf(gdn_out_);
    free_buf(qg_); free_buf(kb_); free_buf(vb_); free_buf(q_); free_buf(gate_);
    free_buf(att_); free_buf(router_); free_buf(eg_); free_buf(eu_); free_buf(eh_);
    free_buf(ey_); free_buf(sg_); free_buf(su_); free_buf(sh_); free_buf(sy_);
    free_buf(sgate_); free_buf(wts_); free_buf(logits_); free_buf(ids_);
    for (auto& b : conv_state_) free_buf(b);
    for (auto& b : S_) free_buf(b);
    for (auto& b : kc_) free_buf(b);
    for (auto& b : vc_) free_buf(b);
    if (fence_) vkDestroyFence(ctx_.dev, fence_, nullptr);
    if (desc_pool_) vkDestroyDescriptorPool(ctx_.dev, desc_pool_, nullptr);
    if (desc_layout_) vkDestroyDescriptorSetLayout(ctx_.dev, desc_layout_, nullptr);
    if (pipe_layout_) vkDestroyPipelineLayout(ctx_.dev, pipe_layout_, nullptr);
    if (cmd_pool_) vkDestroyCommandPool(ctx_.dev, cmd_pool_, nullptr);
}

std::unique_ptr<NativeForward> NativeForward::create(const VulkanCtx& ctx,
                                                     const std::string& gguf_path,
                                                     const Geometry& geom,
                                                     const Deps& deps,
                                                     const ShaderDirs& dirs,
                                                     int ctx_len,
                                                     std::string& err) {
    auto f = std::unique_ptr<NativeForward>(new NativeForward());
    f->ctx_ = ctx; f->geom_ = geom; f->deps_ = deps; f->ctx_len_ = ctx_len;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.phys, &props);
    f->weight_align_ = std::max<uint64_t>(256, props.limits.minStorageBufferOffsetAlignment);

    // ---- parse GGUF header, metadata and tensor directory -------------------
    FILE* g = std::fopen(gguf_path.c_str(), "rb");
    if (!g) { err = "cannot open GGUF"; return nullptr; }
    char magic[4] = {};
    if (std::fread(magic, 1, 4, g) != 4 || std::memcmp(magic, "GGUF", 4)) { err = "not a GGUF"; std::fclose(g); return nullptr; }
    uint32_t ver = 0;
    if (!rd(g, ver) || ver < 2 || ver > 3) { err = "bad GGUF version"; std::fclose(g); return nullptr; }
    uint64_t n_tensors = 0, n_kv = 0;
    if (!rd(g, n_tensors) || !rd(g, n_kv)) { err = "bad GGUF header"; std::fclose(g); return nullptr; }

    std::unordered_map<std::string, KV> kv;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key; int32_t t = 0;
        if (!rd_str(g, key) || !rd(g, t)) { err = "bad GGUF metadata"; std::fclose(g); return nullptr; }
        KV v;
        if (!read_kv(g, t, v)) { err = "bad GGUF metadata value: " + key; std::fclose(g); return nullptr; }
        kv[key] = v;
    }

    struct DirEntry { std::string name; int type; uint64_t ne[4]; int nd; uint64_t file_off; uint64_t buf_off; uint64_t bytes; };
    std::vector<DirEntry> dir;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        DirEntry d{};
        if (!rd_str(g, d.name) || !rd(g, d.nd) || d.nd > 4) { err = "bad tensor directory"; std::fclose(g); return nullptr; }
        for (int k = 0; k < d.nd; ++k) if (!rd(g, d.ne[k])) { err = "bad tensor dims"; std::fclose(g); return nullptr; }
        int32_t t = 0;
        if (!rd(g, t) || !rd(g, d.file_off)) { err = "bad tensor info"; std::fclose(g); return nullptr; }
        d.type = t;
        uint64_t bb = block_bytes(t), be = block_elems(t);
        if (d.ne[0] % be == 0) {
            d.bytes = (uint64_t(d.ne[0]) / be) * bb;
            for (int k = 1; k < d.nd; ++k) d.bytes *= d.ne[k];
        }
        dir.push_back(d);
    }
    const uint64_t data_base = align_up(uint64_t(std::ftell(g)), 32);
    std::fclose(g);

    // ---- geometry from metadata (fall back to the known model constants) ----
    Geometry gm = geom;
    auto get_i = [&](const char* k, int def) {
        auto it = kv.find(k); return it != kv.end() && it->second.type >= 4 && it->second.type <= 6
            ? int(it->second.i) : def;
    };
    auto get_f = [&](const char* k, float def) {
        auto it = kv.find(k); return it != kv.end() && it->second.type == 6 ? it->second.f : def;
    };
    gm.n_layers = get_i("qwen35moe.block_count", gm.n_layers) - get_i("qwen35moe.nextn_predict_layers", 1);
    gm.n_embd = get_i("qwen35moe.embedding_length", gm.n_embd);
    gm.n_head = get_i("qwen35moe.attention.head_count", gm.n_head);
    gm.n_head_kv = get_i("qwen35moe.attention.head_count_kv", gm.n_head_kv);
    gm.head_dim = get_i("qwen35moe.attention.key_length", gm.head_dim);
    gm.n_experts = get_i("qwen35moe.expert_count", gm.n_experts);
    gm.top_k = get_i("qwen35moe.expert_used_count", gm.top_k);
    gm.n_ff = get_i("qwen35moe.expert_feed_forward_length", gm.n_ff);
    gm.n_ff_shexp = get_i("qwen35moe.expert_shared_feed_forward_length", gm.n_ff_shexp);
    gm.d_conv = get_i("qwen35moe.ssm.conv_kernel", gm.d_conv);
    gm.d_state = get_i("qwen35moe.ssm.state_size", gm.d_state);
    gm.hk = get_i("qwen35moe.ssm.group_count", gm.hk);
    gm.hv = get_i("qwen35moe.ssm.inner_size", gm.hv) / std::max(1, gm.d_state);
    gm.conv_channels = get_i("qwen35moe.ssm.inner_size", gm.conv_channels) * 2;
    gm.n_rot = get_i("qwen35moe.rope.dimension_count", gm.n_rot);
    gm.full_attn_interval = get_i("qwen35moe.full_attention_interval", gm.full_attn_interval);
    gm.eps = get_f("qwen35moe.attention.layer_norm_rms_epsilon", gm.eps);
    gm.rope_base = get_f("qwen35moe.rope.freq_base", gm.rope_base);
    {
        auto it = kv.find("tokenizer.ggml.eos_token_id");
        if (it != kv.end()) gm.eos = int(it->second.i);
    }
    // n_vocab from output.weight's output dimension.
    for (const auto& d : dir) if (d.name == "output.weight" && d.nd >= 2) gm.n_vocab = int(d.ne[1]);
    f->geom_ = gm;
    std::fprintf(stderr, "native: %d layers, %d experts (top %d), embd %d, vocab %d, ctx %d\n",
                 gm.n_layers, gm.n_experts, gm.top_k, gm.n_embd, gm.n_vocab, ctx_len);

    // ---- allocate + fill the resident weight buffer --------------------------
    // Expert tensors are intentionally excluded: ExpertStream streams them.
    uint64_t total = 0;
    for (auto& d : dir) {
        if (d.name.find("_exps.weight") != std::string::npos) continue;
        if (!d.bytes) { err = "unsupported weight type in " + d.name; return nullptr; }
        d.buf_off = align_up(total, f->weight_align_);
        total = d.buf_off + d.bytes;
        TensorInfo ti{};
        ti.type = d.type; ti.nd = d.nd; ti.n_bytes = d.bytes; ti.offset = d.buf_off;
        for (int k = 0; k < d.nd; ++k) ti.ne[k] = d.ne[k];
        f->tensors_[d.name] = ti;
    }
    std::fprintf(stderr, "native: resident weights %.2f GiB (%zu tensors)\n",
                 double(total) / (1 << 30), f->tensors_.size());
    if (!f->make_buf(total, f->weights_, err)) return nullptr;
    g = std::fopen(gguf_path.c_str(), "rb");
    for (const auto& d : dir) {
        if (d.name.find("_exps.weight") != std::string::npos) continue;
        uint64_t done = 0;
        while (done < d.bytes) {
            ssize_t n = ::pread(::fileno(g), f->weights_.map + d.buf_off + done,
                                d.bytes - done, off_t(data_base + d.file_off + done));
            if (n <= 0) { err = "short GGUF read: " + d.name; std::fclose(g); return nullptr; }
            done += uint64_t(n);
        }
    }
    std::fclose(g);

    // ---- Vulkan descriptors / pipelines -------------------------------------
    VkDescriptorSetLayoutBinding b[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6; dli.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx.dev, &dli, nullptr, &f->desc_layout_) != VK_SUCCESS) {
        err = "descriptor layout failed"; return nullptr;
    }
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 6};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 100000; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(ctx.dev, &pi, nullptr, &f->desc_pool_) != VK_SUCCESS) {
        err = "descriptor pool failed"; return nullptr;
    }
    // One set for the immediate path plus one for each kernel a batch may hold.
    constexpr uint32_t kBatchKernels = 32;
    std::vector<VkDescriptorSetLayout> set_layouts(1 + kBatchKernels, f->desc_layout_);
    std::vector<VkDescriptorSet> sets(set_layouts.size());
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = f->desc_pool_; dai.descriptorSetCount = uint32_t(sets.size());
    dai.pSetLayouts = set_layouts.data();
    if (vkAllocateDescriptorSets(ctx.dev, &dai, sets.data()) != VK_SUCCESS) {
        err = "descriptor set failed"; return nullptr;
    }
    f->desc_set_ = sets[0];
    f->batch_sets_.assign(sets.begin() + 1, sets.end());
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.offset = 0; pcr.size = 28;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &f->desc_layout_;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx.dev, &pli, nullptr, &f->pipe_layout_) != VK_SUCCESS) {
        err = "pipeline layout failed"; return nullptr;
    }
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(ctx.dev, &fi, nullptr, &f->fence_) != VK_SUCCESS) { err = "fence failed"; return nullptr; }
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = ctx.qfam; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx.dev, &cpi, nullptr, &f->cmd_pool_) != VK_SUCCESS) {
        err = "command pool failed"; return nullptr;
    }
    VkCommandBufferAllocateInfo cmd_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_ai.commandPool = f->cmd_pool_; cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 2;
    VkCommandBuffer cmds[2]{};
    if (vkAllocateCommandBuffers(ctx.dev, &cmd_ai, cmds) != VK_SUCCESS) {
        err = "command buffer failed"; return nullptr;
    }
    f->cmd_ = cmds[0];
    f->batch_cmd_ = cmds[1];
    if (!f->make_buf(256, f->dummy_, err)) return nullptr;

    auto D = [&](const std::string& dirp, const char* name) { return dirs.native + "/" + name + ".comp.spv"; };
    struct PipeDef { const char* name; const char* file; bool native_dir; };
    const PipeDef defs[] = {
        {"gemv_f32", "gemv_f32", true},
        {"dequant_row_q4_k", "dequant_row_q4_k", true},
        {"rms", "rms", true},
        {"add", "add", true},
        {"gdn_conv", "gdn_conv", true},
        {"gdn_step", "gdn_step", true},
        {"gdn_norm", "gdn_norm", true},
        {"attn_q_prep", "attn_q_prep", true},
        {"attn_k_prep", "attn_k_prep", true},
        {"attn_v_prep", "attn_v_prep", true},
        {"attn", "attn", true},
        {"moe_add", "moe_add", true},
        {"gemv_q4", "expert_q4_k_gemv", false},
        {"gemv_q4_wide", "gemv_q4_wide", true},
        {"gemv_q6_wide", "gemv_q6_wide", true},
        {"gemv_q6", "expert_q6_k_gemv", false},
        {"swiglu", "expert_silu_mul", false},
    };
    for (const auto& d : defs) {
        std::string path = (d.native_dir ? dirs.native : dirs.expert) + "/" + d.file + ".comp.spv";
        if (!f->add_pipeline(d.name, path, err)) return nullptr;
    }

    // ---- scratch and per-layer state ----------------------------------------
    auto nb = [&](size_t n) { return VkDeviceSize(n * sizeof(float)); };
    const int N = gm.n_embd, K = gm.top_k, FF = gm.n_ff;
    if (!f->make_buf(nb(N), f->x_, err) || !f->make_buf(nb(N), f->h_, err) ||
        !f->make_buf(nb(N), f->y_, err) ||
        !f->make_buf(nb(gm.conv_channels), f->qkv_, err) ||
        !f->make_buf(nb(gm.hv * gm.d_state), f->z_, err) ||
        !f->make_buf(nb(gm.hv), f->alpha_, err) || !f->make_buf(nb(gm.hv), f->beta_, err) ||
        !f->make_buf(nb(gm.conv_channels), f->conv_out_, err) ||
        !f->make_buf(nb(gm.hv * gm.d_state), f->gdn_out_, err) ||
        !f->make_buf(nb(2 * gm.n_head * gm.head_dim), f->qg_, err) ||
        !f->make_buf(nb(gm.n_head_kv * gm.head_dim), f->kb_, err) ||
        !f->make_buf(nb(gm.n_head_kv * gm.head_dim), f->vb_, err) ||
        !f->make_buf(nb(gm.n_head * gm.head_dim), f->q_, err) ||
        !f->make_buf(nb(gm.n_head * gm.head_dim), f->gate_, err) ||
        !f->make_buf(nb(gm.n_head * gm.head_dim), f->att_, err) ||
        !f->make_buf(nb(gm.n_experts), f->router_, err) ||
        !f->make_buf(nb(size_t(K) * FF), f->eg_, err) ||
        !f->make_buf(nb(size_t(K) * FF), f->eu_, err) ||
        !f->make_buf(nb(size_t(K) * FF), f->eh_, err) ||
        !f->make_buf(nb(size_t(K) * N), f->ey_, err) ||
        !f->make_buf(nb(gm.n_ff_shexp), f->sg_, err) ||
        !f->make_buf(nb(gm.n_ff_shexp), f->su_, err) ||
        !f->make_buf(nb(gm.n_ff_shexp), f->sh_, err) ||
        !f->make_buf(nb(N), f->sy_, err) ||
        !f->make_buf(nb(1), f->sgate_, err) ||
        !f->make_buf(nb(K), f->wts_, err) ||
        !f->make_buf(nb(gm.n_vocab), f->logits_, err) ||
        !f->make_buf(nb(K), f->ids_, err)) {
        return nullptr;
    }
    for (int l = 0; l < gm.n_layers; ++l) {
        Buf c, s, kc, vc;
        if (gm.is_attn(l)) {
            size_t kv = size_t(gm.n_head_kv) * ctx_len * gm.head_dim;
            if (!f->make_buf(nb(kv), kc, err) || !f->make_buf(nb(kv), vc, err)) return nullptr;
        } else {
            size_t csz = size_t(gm.d_conv - 1) * gm.conv_channels;
            size_t ssz = size_t(gm.hv) * gm.d_state * gm.d_state;
            if (!f->make_buf(nb(csz), c, err) || !f->make_buf(nb(ssz), s, err)) return nullptr;
        }
        f->conv_state_.push_back(c); f->S_.push_back(s);
        f->kc_.push_back(kc); f->vc_.push_back(vc);
    }
    return f;
}

void NativeForward::reset() {
    pos_ = 0;
    auto zero = [&](Buf& b) { if (b.map) std::memset(b.map, 0, size_t(b.size)); };
    for (auto& b : conv_state_) zero(b);
    for (auto& b : S_) zero(b);
    for (auto& b : kc_) zero(b);
    for (auto& b : vc_) zero(b);
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

const TensorInfo* NativeForward::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool NativeForward::gemv_f32(const std::string& wname, const Buf& x, Buf& y,
                             uint32_t n_out, uint32_t n_in, std::string& err) {
    const TensorInfo* t = find_tensor(wname);
    if (!t || t->type != 0) { err = "no F32 weight " + wname; return false; }
    const VkBuffer bufs[6] = {weights_.b, x.b, y.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkDeviceSize offs[6] = {t->offset, 0, 0, 0, 0, 0};
    const uint32_t pc[7] = {n_out, n_in, 0, 0, 0, 0, 0};
    return dispatch("gemv_f32", n_out, pc, bufs, offs, err);
}

bool NativeForward::gemv_q4(const std::string& wname, const Buf& x, Buf& y,
                            uint32_t n_out, uint32_t n_in, std::string& err) {
    const TensorInfo* t = find_tensor(wname);
    if (!t || t->type != 12) { err = "no Q4_K weight " + wname; return false; }
    const VkBuffer bufs[6] = {weights_.b, x.b, y.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkDeviceSize offs[6] = {t->offset, 0, 0, 0, 0, 0};
    const uint32_t pc[7] = {n_out, n_in, 0, 0, 0, 0, 0};
    return dispatch(n_in > 2048 ? "gemv_q4_wide" : "gemv_q4", n_out, pc, bufs, offs, err);
}

bool NativeForward::gemv_q6(const std::string& wname, const Buf& x, Buf& y,
                            uint32_t n_out, uint32_t n_in, std::string& err) {
    const TensorInfo* t = find_tensor(wname);
    if (!t || t->type != 14) { err = "no Q6_K weight " + wname; return false; }
    const VkBuffer bufs[6] = {weights_.b, x.b, y.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkDeviceSize offs[6] = {t->offset, 0, 0, 0, 0, 0};
    const uint32_t pc[7] = {n_out, n_in, 0, 0, 0, 0, 0};
    return dispatch(n_in > 2048 ? "gemv_q6_wide" : "gemv_q6", n_out, pc, bufs, offs, err);
}

bool NativeForward::gemv(const std::string& wname, const Buf& x, Buf& y,
                         uint32_t n_out, uint32_t n_in, std::string& err) {
    const TensorInfo* t = find_tensor(wname);
    if (!t) { err = "no weight " + wname; return false; }
    if (t->type == 12) return gemv_q4(wname, x, y, n_out, n_in, err);
    if (t->type == 14) return gemv_q6(wname, x, y, n_out, n_in, err);
    if (t->type == 0)  return gemv_f32(wname, x, y, n_out, n_in, err);
    err = "unsupported weight type in " + wname;
    return false;
}

bool NativeForward::rms(const Buf& x, const std::string& wname, Buf& y, uint32_t n, std::string& err) {
    const TensorInfo* t = find_tensor(wname);
    if (!t || t->type != 0) { err = "no F32 norm weight " + wname; return false; }
    const VkBuffer bufs[6] = {x.b, weights_.b, y.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkDeviceSize offs[6] = {0, t->offset, 0, 0, 0, 0};
    uint32_t e;
    std::memcpy(&e, &geom_.eps, 4);
    const uint32_t pc[7] = {n, e, 0, 0, 0, 0, 0};
    return dispatch("rms", 1, pc, bufs, offs, err);
}

bool NativeForward::embed(int tok, Buf& x, std::string& err) {
    const TensorInfo* t = find_tensor("token_embd.weight");
    if (!t || t->type != 12) { err = "no Q4_K embedding"; return false; }
    const VkBuffer bufs[6] = {weights_.b, VK_NULL_HANDLE, x.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    const VkDeviceSize offs[6] = {t->offset, 0, 0, 0, 0, 0};
    const uint32_t pc[7] = {uint32_t(tok), uint32_t(geom_.n_embd), 0, 0, 0, 0, 0};
    return dispatch("dequant_row_q4_k", (geom_.n_embd + 127u) / 128u, pc, bufs, offs, err);
}

// ---------------------------------------------------------------------------
// one token
// ---------------------------------------------------------------------------

bool NativeForward::layer(int l, Buf& x, std::string& err) {
    const Geometry& g = geom_;
    const int N = g.n_embd, H = g.n_head, HKV = g.n_head_kv, D = g.head_dim;
    const int HK = g.hk, HV = g.hv, DS = g.d_state;
    auto L = [&](const char* s) { return "blk." + std::to_string(l) + "." + s; };
    auto dbg = [&](const char* name, const Buf& b, size_t n) {
        if (l != 0 || pos_ != 0 || stage_dir_.empty()) return;
        // The dump reads GPU-written memory on the host, so drain any batch.
        if (!batch_flush(err)) return;
        std::string p = stage_dir_ + "/" + name + ".bin";
        if (FILE* f = std::fopen(p.c_str(), "wb")) {
            std::fwrite(b.map, sizeof(float), n, f); std::fclose(f);
        }
    };

    // Batch the mixer kernels; a single fence wait replaces one per kernel.
    if (!batch_begin(err)) return false;
    if (!rms(x, L("attn_norm.weight"), h_, N, err)) return false;
    dbg("h_norm1", h_, N);
    if (!g.is_attn(l)) {
        if (!gemv(L("attn_qkv.weight"), h_, qkv_, g.conv_channels, N, err) ||
            !gemv_q4(L("attn_gate.weight"), h_, z_, HV * DS, N, err) ||
            !gemv_q4(L("ssm_beta.weight"), h_, beta_, HV, N, err) ||
            !gemv_q4(L("ssm_alpha.weight"), h_, alpha_, HV, N, err)) return false;
        // conv + silu
        const TensorInfo* taps = find_tensor(L("ssm_conv1d.weight"));
        if (!taps) { err = "no conv taps"; return false; }
        {
            const VkBuffer bufs[6] = {qkv_.b, conv_state_[l].b, weights_.b, conv_out_.b,
                                      VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, 0, taps->offset, 0, 0, 0};
            const uint32_t pc[7] = {uint32_t(g.conv_channels), uint32_t(g.d_conv), 0, 0, 0, 0, 0};
            if (!dispatch("gdn_conv", (g.conv_channels + 255u) / 256u, pc, bufs, offs, err)) return false;
        }
        if (!batch_flush(err)) return false;  // the host math below reads alpha_/beta_
        // host: g[h] = exp(softplus(alpha+dt)*ssm_a), b[h] = sigmoid(beta)
        const float* dt = reinterpret_cast<const float*>(weights_.map + find_tensor(L("ssm_dt.bias"))->offset);
        const float* sa = reinterpret_cast<const float*>(weights_.map + find_tensor(L("ssm_a"))->offset);
        float* af = reinterpret_cast<float*>(alpha_.map);
        float* bf = reinterpret_cast<float*>(beta_.map);
        for (int hh = 0; hh < HV; ++hh) {
            float a = af[hh] + dt[hh];
            af[hh] = std::exp(softplus(a) * sa[hh]);
            bf[hh] = sigmoid(bf[hh]);
        }
        // recurrent step
        if (!batch_begin(err)) return false;
        {
            const VkBuffer bufs[6] = {conv_out_.b, alpha_.b, beta_.b, S_[l].b, gdn_out_.b, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, 0, 0, 0, 0, 0};
            uint32_t e; std::memcpy(&e, &g.eps, 4);
            const uint32_t pc[7] = {uint32_t(HK), uint32_t(HV), uint32_t(DS), e, 0, 0, 0};
            if (!dispatch("gdn_step", HV, pc, bufs, offs, err)) return false;
        }
        dbg("gdn_out", gdn_out_, HV * DS);  // raw recurrent output
        // gated norm
        {
            const TensorInfo* nw = find_tensor(L("ssm_norm.weight"));
            const VkBuffer bufs[6] = {gdn_out_.b, weights_.b, z_.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, nw->offset, 0, 0, 0, 0};
            uint32_t e; std::memcpy(&e, &g.eps, 4);
            const uint32_t pc[7] = {uint32_t(DS), e, 0, 0, 0, 0, 0};
            if (!dispatch("gdn_norm", HV, pc, bufs, offs, err)) return false;
            dbg("gdn_normed", gdn_out_, HV * DS);
            dbg("z", z_, HV * DS);
        }
        dbg("qkv", qkv_, g.conv_channels);
        dbg("conv_out", conv_out_, g.conv_channels);
        dbg("alpha", alpha_, HV);
        dbg("beta", beta_, HV);
        if (!batch_flush(err)) return false;
        if (!batch_begin(err)) return false;
        if (!gemv_q4(L("ssm_out.weight"), gdn_out_, y_, N, HV * DS, err)) return false;
    } else {
        if (!gemv_q4(L("attn_q.weight"), h_, qg_, 2 * H * D, N, err) ||
            !gemv_q4(L("attn_k.weight"), h_, kb_, HKV * D, N, err) ||
            !gemv(L("attn_v.weight"), h_, vb_, HKV * D, N, err)) return false;
        uint32_t rb = 0, pbd = 0;
        std::memcpy(&rb, &g.rope_base, 4);
        float scale = 1.f / std::sqrt(float(D));
        std::memcpy(&pbd, &scale, 4);
        {
            const TensorInfo* nw = find_tensor(L("attn_q_norm.weight"));
            const VkBuffer bufs[6] = {qg_.b, weights_.b, q_.b, gate_.b, VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, nw->offset, 0, 0, 0, 0};
            uint32_t e; std::memcpy(&e, &g.eps, 4);
            const uint32_t pc[7] = {uint32_t(D), uint32_t(g.n_rot), uint32_t(pos_), rb, e, 0, 0};
            if (!dispatch("attn_q_prep", H, pc, bufs, offs, err)) return false;
        }
        {
            const TensorInfo* nw = find_tensor(L("attn_k_norm.weight"));
            const VkBuffer bufs[6] = {kb_.b, weights_.b, kc_[l].b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, nw->offset, 0, 0, 0, 0};
            uint32_t e; std::memcpy(&e, &g.eps, 4);
            const uint32_t pc[7] = {uint32_t(D), uint32_t(g.n_rot), uint32_t(pos_), rb, e, uint32_t(ctx_len_), 0};
            if (!dispatch("attn_k_prep", HKV, pc, bufs, offs, err)) return false;
        }
        {
            const VkBuffer bufs[6] = {vb_.b, VK_NULL_HANDLE, vc_[l].b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, 0, 0, 0, 0, 0};
            const uint32_t pc[7] = {uint32_t(D), uint32_t(pos_), uint32_t(ctx_len_), 0, 0, 0, 0};
            if (!dispatch("attn_v_prep", HKV, pc, bufs, offs, err)) return false;
        }
        {
            const VkBuffer bufs[6] = {q_.b, kc_[l].b, vc_[l].b, gate_.b, att_.b, VK_NULL_HANDLE};
            const VkDeviceSize offs[6] = {0, 0, 0, 0, 0, 0};
            const uint32_t pc[7] = {uint32_t(H), uint32_t(HKV), uint32_t(D), uint32_t(pos_ + 1),
                                    uint32_t(ctx_len_), pbd, 0};
            if (!dispatch("attn", H, pc, bufs, offs, err)) return false;
        }
        dbg("qg", qg_, 2 * H * D);
        dbg("att", att_, H * D);
        if (!batch_flush(err)) return false;
        if (!batch_begin(err)) return false;
        if (!gemv_q4(L("attn_output.weight"), att_, y_, N, H * D, err)) return false;
    }
    dbg("y_mixer", y_, N);
    // residual
    {
        const VkBuffer bufs[6] = {x.b, y_.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
        const VkDeviceSize offs[6] = {0, 0, 0, 0, 0, 0};
        const uint32_t pc[7] = {uint32_t(N), 0, 0, 0, 0, 0, 0};
        if (!dispatch("add", (N + 255u) / 256u, pc, bufs, offs, err)) return false;
    }
    dbg("x_after_add", x, N);
    if (!rms(x, L("post_attention_norm.weight"), h_, N, err)) return false;
    if (!batch_flush(err)) return false;  // the router copies h_ on the host
    dbg("h_norm2", h_, N);

    // ---- MoE: router (real hidden state), prefetch, experts, shared, combine --
    const TensorInfo* rt = find_tensor(L("ffn_gate_inp.weight"));
    if (!rt || !deps_.router) { err = "no router"; return false; }
    const float* W = reinterpret_cast<const float*>(weights_.map + rt->offset);
    std::vector<float> logits(size_t(g.n_experts));
    std::vector<int32_t> ids(size_t(g.top_k));
    std::vector<float> wts(size_t(g.top_k));
    if (!deps_.router->gemv_topk(W, reinterpret_cast<const float*>(h_.map), logits.data(),
                                 g.n_experts, N, g.top_k, ids.data(), wts.data(), err)) return false;
    std::memcpy(wts_.map, wts.data(), size_t(g.top_k) * sizeof(float));
    if (trace_) {
        std::fprintf(trace_, "%d", l);
        for (int i = 0; i < g.top_k; ++i) std::fprintf(trace_, " %d", ids[i]);
        std::fputc('\n', trace_);
    }
    if (l + 1 < g.n_layers) deps_.stream->prefetch_route(l + 1, ids.data(), g.top_k);

    const uint32_t FF = uint32_t(g.n_ff);
    if (!deps_.stream->batch_begin(err)) return false;
    for (int s = 0; s < g.top_k; ++s) {
        const auto key = [&](ExpertKind k) {
            return ExpertStream::Key{l, k, uint32_t(ids[s])};
        };
        uint32_t sg = 0, su = 0, sd = 0;
        if (!deps_.stream->ensure(key(ExpertKind::Gate), sg, err) ||
            !deps_.stream->ensure(key(ExpertKind::Up), su, err) ||
            !deps_.stream->ensure(key(ExpertKind::Down), sd, err)) return false;
        float* eg = reinterpret_cast<float*>(eg_.map) + size_t(s) * FF;
        float* eu = reinterpret_cast<float*>(eu_.map) + size_t(s) * FF;
        if (!deps_.stream->batch_matvec(key(ExpertKind::Gate), sg, reinterpret_cast<const float*>(h_.map), N, eg, FF, err) ||
            !deps_.stream->batch_matvec(key(ExpertKind::Up), su, reinterpret_cast<const float*>(h_.map), N, eu, FF, err)) return false;
        (void)sd;
    }
    if (!deps_.stream->batch_end(err)) return false;
    if (!batch_begin(err)) return false;
    {
        const VkBuffer bufs[6] = {eg_.b, eu_.b, eh_.b, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
        const VkDeviceSize offs[6] = {0, 0, 0, 0, 0, 0};
        const uint32_t pc[7] = {uint32_t(g.top_k * FF), 0, 0, 0, 0, 0, 0};
        if (!dispatch("swiglu", (g.top_k * FF + 63u) / 64u, pc, bufs, offs, err)) return false;
    }
    if (!batch_flush(err)) return false;  // the down batch copies eh_ on the host
    if (!deps_.stream->batch_begin(err)) return false;
    for (int s = 0; s < g.top_k; ++s) {
        const auto key = [&](ExpertKind k) { return ExpertStream::Key{l, k, uint32_t(ids[s])}; };
        uint32_t sd = 0;
        if (!deps_.stream->ensure(key(ExpertKind::Down), sd, err)) return false;
        if (!deps_.stream->batch_matvec(key(ExpertKind::Down), sd,
                                        reinterpret_cast<float*>(eh_.map) + size_t(s) * FF, FF,
                                        reinterpret_cast<float*>(ey_.map) + size_t(s) * N, N, err)) return false;
    }
    if (!deps_.stream->batch_end(err)) return false;
    if (!deps_.stream->shared_forward(l, reinterpret_cast<const float*>(h_.map), N, reinterpret_cast<float*>(sy_.map), N, err)) return false;
    if (!batch_begin(err)) return false;
    {
        const VkBuffer bufs[6] = {x.b, ey_.b, wts_.b, sy_.b, VK_NULL_HANDLE, VK_NULL_HANDLE};
        const VkDeviceSize offs[6] = {0, 0, 0, 0, 0, 0};
        const uint32_t pc[7] = {uint32_t(N), uint32_t(g.top_k), 0, 0, 0, 0, 0};
        if (!dispatch("moe_add", (N + 255u) / 256u, pc, bufs, offs, err)) return false;
    }
    if (!batch_flush(err)) return false;
    return true;
}

bool NativeForward::forward_dump(int tok, float* logits_out, FILE* dump, std::string& err) {
    const int N = geom_.n_embd;
    batch_abort();  // a failed previous request must not leave a batch recording
    if (!embed(tok, x_, err)) return false;
    if (dump) std::fwrite(x_.map, sizeof(float), size_t(N), dump);  // record -1: embed
    for (int l = 0; l < geom_.n_layers; ++l) {
        if (!layer(l, x_, err)) return false;
        if (dump) std::fwrite(x_.map, sizeof(float), size_t(N), dump);
    }
    if (!batch_begin(err)) return false;
    if (!rms(x_, "output_norm.weight", h_, N, err)) return false;
    if (!gemv_q6("output.weight", h_, logits_, geom_.n_vocab, N, err)) return false;
    if (!batch_flush(err)) return false;
    std::memcpy(logits_out, logits_.map, size_t(geom_.n_vocab) * sizeof(float));
    ++pos_;
    return true;
}

bool NativeForward::forward(int tok, float* logits_out, std::string& err) {
    return forward_dump(tok, logits_out, nullptr, err);
}

} // namespace native
} // namespace strata::vulkan
