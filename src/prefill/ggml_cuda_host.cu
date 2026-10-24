// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "strata/core/emulate.hpp"
#include "common.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

#if defined(GGML_USE_HIP)
namespace {

unsigned parse_hex(const std::string& value) {
    if (value.empty()) return 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 16);
    return end == value.c_str() || *end != '\0' ? 0 : static_cast<unsigned>(parsed);
}

// Match GGML's AMD cc encoding: OFFSET_AMD + gfx major * 0x100 + minor/stepping.
// For example, gfx1100 maps to GGML_CUDA_CC_RDNA3 (0x01001100).
int ggml_cuda_parse_amd_id(const char* device_name) {
    if (device_name == nullptr) return GGML_CUDA_CC_OFFSET_AMD;

    std::string arch(device_name);
    if (arch.compare(0, 3, "gfx") == 0) arch.erase(0, 3);
    const size_t suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    const size_t generic = arch.rfind("-generic");
    if (generic != std::string::npos && generic + 8 == arch.size()) arch.resize(generic);

    unsigned major = 0;
    unsigned minor = 0;
    const size_t separator = arch.find('-');
    if (separator != std::string::npos) {
        major = parse_hex(arch.substr(0, separator));
        minor = parse_hex(arch.substr(separator + 1)) * 0x10;
    } else if (arch.size() >= 3) {
        minor = parse_hex(arch.substr(arch.size() - 2));
        major = parse_hex(arch.substr(0, arch.size() - 2));
    }
    return GGML_CUDA_CC_OFFSET_AMD + static_cast<int>(major * 0x100 + minor);
}

}  // namespace
#endif

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto & d = in.devices[id];
#if defined(GGML_USE_HIP)
            d.cc = ggml_cuda_parse_amd_id(prop.gcnArchName);
            if ((d.cc & 0xff00) == 0) {
                // Fall back to HIP's device version fields if its architecture name is unavailable.
                d.cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100 + prop.minor * 0x10;
            }
            d.smpbo = prop.sharedMemPerBlock;
            d.integrated = false;
            d.supports_cooperative_launch = false;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.warp_size = prop.warpSize;
#else
            // #542: the fields MMQ's choices read come from cudaDeviceGetAttribute, which is ABI-stable: a
            // cudaDeviceProp filled by an older libcudart than the headers (a CUDA 13 build that links a CUDA 12
            // libcudart.so) is shifted - sharedMemPerBlockOptin read 1 there, and every i-quant lost MMQ (#420's
            // fits()).  In a matched build both give the same values.
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            d.cc = 100 * strata::cc_major_of(attr(cudaDevAttrComputeCapabilityMajor)) +
                   10 * strata::cc_minor_of(attr(cudaDevAttrComputeCapabilityMinor));   // STRATA_EMULATE_CC
            d.smpbo = strata::smem_optin_of(attr(cudaDevAttrMaxSharedMemoryPerBlockOptin));
            d.integrated = attr(cudaDevAttrIntegrated) != 0;
            d.supports_cooperative_launch = attr(cudaDevAttrCooperativeLaunch) != 0;
            d.nsm = attr(cudaDevAttrMultiProcessorCount);
            d.smpb = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlock);
            d.warp_size = attr(cudaDevAttrWarpSize);
#endif
            d.vmm = false;
            d.total_vram = prop.totalGlobalMem;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// The prompt path's MMQ scratch, where the pool has no business taking VRAM: the part of the borrowed expert-cache
// region (a request's loan, Prefill::init/relayout) that this chunk's layout did not use.  It is the prompt's memory
// for as long as the prompt holds the loan, so the fixup tiles cost the card nothing - and on a card whose free VRAM
// is a handful of MB at prompt time the cudaMalloc below is what used to abort the engine mid-prompt.  Set per
// layout (mmq::pool_slab, from Prefill::init and Prefill::relayout on the device that will run the chunk), so the
// buffers cached from the previous one are dropped when it moves.
struct Slab {
    uint8_t * base = nullptr;
    size_t bytes = 0, used = 0;
};
Slab g_slab[GGML_CUDA_MAX_DEVICES];
std::mutex g_slab_mu;

// MMQ's stream-k fixup asks for nsm*J*I floats, and its J search (8..128 by 8) picks a different J per launch, so a
// cache keyed on the exact size holds one buffer per J: 32 buffers, 31.5 MiB on a 579-token chunk (and the same at
// 16k, since J does not grow with the chunk - the Maya checkout measured it on its rig).  Rounding a request up to
// a size class (a power of two, 128 KiB floor - 128 KiB..2 MiB covers the whole J range on a 30-SM card) bounds the
// pool to a few MiB whatever J wins.  Handing back more than was asked is what every caching allocator does; the
// kernels only touch the first nsm*J*I floats.
size_t size_class(size_t n) {
    size_t c = 128u << 10;
    while (c < n) c <<= 1;
    return c;
}

// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; bool slab; };
    std::vector<Buf> bufs;
    std::mutex mu;
    int dev;
    void * slab_seen = nullptr;   // the slab base the cached buffers came from; a move makes them stale
    explicit CachingPool(int device) : dev(device) {}
    size_t held() const {   // the bytes this pool holds on the card (never given back)
        size_t n = 0;
        for (const auto & b : bufs) n += b.size;
        return n;
    }
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        const size_t cls = size_class(size);
        {
            std::lock_guard<std::mutex> sl(g_slab_mu);
            Slab & s = g_slab[dev];
            if (s.base != slab_seen) {
                slab_seen = s.base;
                // the layout moved under the cached buffers: nothing is in flight between chunks, so drop them
                bufs.erase(std::remove_if(bufs.begin(), bufs.end(),
                                          [](const Buf & b) { return b.slab; }), bufs.end());
            }
            for (auto & b : bufs)
                if (!b.used && b.size >= cls) { b.used = true; *actual_size = b.size; return b.p; }
            if (s.base != nullptr && s.bytes >= s.used && s.bytes - s.used >= cls) {
                void * p = s.base + s.used;
                s.used += cls;
                *actual_size = cls;
                bufs.push_back({p, cls, true, true});
                return p;
            }
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        void * p = nullptr;
        const cudaError_t e = cudaMalloc(&p, cls);
        if (e != cudaSuccess) {
            // this is the whole reason a prompt used to abort mid-read with "out of memory" and nothing else
            std::fprintf(stderr,
                         "ggml-cuda (strata mmq): CUDA%d wanted %.1f MiB of scratch (the prompt path's pool held %.1f "
                         "MiB in %zu buffers) and only %.1f MiB of %.1f MiB was free: %s\n",
                         dev, (double) cls / 1048576.0, (double) held() / 1048576.0, bufs.size(),
                         (double) free_b / 1048576.0, (double) total_b / 1048576.0, cudaGetErrorString(e));
            CUDA_CHECK(e);
        }
        if (cls >= ((size_t) 4 << 20))
            std::fprintf(stderr, "strata prefill mmq: CUDA%d the scratch pool takes a new %.1f MiB buffer (%.1f MiB free of "
                                 "%.1f MiB, %.1f MiB held)\n",
                         dev, (double) cls / 1048576.0, (double) free_b / 1048576.0, (double) total_b / 1048576.0,
                         (double) held() / 1048576.0);
        bufs.push_back({p, cls, true, false});
        *actual_size = cls;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs)
            if (!b.slab) cudaFree(b.p);   // slab buffers are the borrowed region's tail, not ours to free
    }
};
}  // namespace

// Called by the prompt path once per layout, before that layout's MMQ launches: `base`/`bytes` are the tail of the
// borrowed expert-cache region this chunk's carve left over (prefill.cpp's carve, which knows both).  The bump
// restarts there; a tail somewhere else invalidates the buffers cached from the old one.  A null `base` (the loan is
// gone, or the layout used all of it) drops them and the pool allocates on the device again.
extern "C" void strata_mmq_pool_slab(int dev, void * base, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_slab_mu);
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return;
    g_slab[dev] = {(uint8_t *) base, base != nullptr ? bytes : 0, 0};
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
